import sys
import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import QAction, QBrush, QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
    QInputDialog,
)

from coco_io import _read_image_size
from image_loader import load_image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class BBoxRectItem(QGraphicsRectItem):
    def __init__(self, ann_id: int, category_id: int, category_name: str, rect: QRectF):
        super().__init__(rect)
        self.ann_id = ann_id
        self.category_id = category_id
        self.category_name = category_name
        self.setFlags(
            QGraphicsRectItem.ItemIsSelectable
            | QGraphicsRectItem.ItemIsMovable
            | QGraphicsRectItem.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)
        self.update_style()

    def update_style(self):
        color = QColor(0, 180, 255) if not self.isSelected() else QColor(255, 80, 80)
        pen = QPen(color, 2)
        cosmetic_pen = True
        pen.setCosmetic(cosmetic_pen)
        self.setPen(pen)
        self.setBrush(QBrush(QColor(0, 180, 255, 35)))

    def itemChange(self, change, value):
        if change == QGraphicsRectItem.ItemSelectedChange:
            self.update_style()
        return super().itemChange(change, value)


class AnnotatorView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setMouseTracking(True)
        self._drawing = False
        self._start_scene_pos: Optional[QPointF] = None
        self._temp_rect: Optional[QGraphicsRectItem] = None
        self._panning = False
        self._pan_start = None

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            factor = 1.15
        else:
            factor = 1 / 1.15
        self.scale(factor, factor)

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton or (event.button() == Qt.LeftButton and event.modifiers() & Qt.SpaceModifier):
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            return

        if event.button() == Qt.LeftButton and self.main_window and self.main_window.image_loaded:
            scene_pos = self.mapToScene(event.pos())
            if self.main_window.is_inside_image(scene_pos):
                clicked_item = self.itemAt(event.pos())
                if isinstance(clicked_item, BBoxRectItem):
                    super().mousePressEvent(event)
                    self.main_window.refresh_annotation_list()
                    return
                self._drawing = True
                self._start_scene_pos = scene_pos
                self._temp_rect = QGraphicsRectItem(QRectF(scene_pos, scene_pos))
                pen = QPen(QColor(255, 200, 0), 2)
                pen.setCosmetic(True)
                self._temp_rect.setPen(pen)
                self._temp_rect.setBrush(QBrush(QColor(255, 200, 0, 35)))
                self.scene().addItem(self._temp_rect)
                return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning and self._pan_start is not None:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            return

        if self._drawing and self._temp_rect and self._start_scene_pos:
            current_pos = self.mapToScene(event.pos())
            current_pos = self.main_window.clamp_to_image(current_pos)
            rect = QRectF(self._start_scene_pos, current_pos).normalized()
            self._temp_rect.setRect(rect)
            if self.main_window:
                self.main_window.statusBar().showMessage(
                    f"Drawing bbox: x={rect.x():.0f}, y={rect.y():.0f}, w={rect.width():.0f}, h={rect.height():.0f}"
                )
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._panning:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
            return

        if event.button() == Qt.LeftButton and self._drawing:
            self._drawing = False
            if self._temp_rect:
                rect = self._temp_rect.rect().normalized()
                self.scene().removeItem(self._temp_rect)
                self._temp_rect = None
                if rect.width() >= 3 and rect.height() >= 3:
                    self.main_window.add_annotation(rect)
            self._start_scene_pos = None
            return

        super().mouseReleaseEvent(event)


class YOLOXCOCOAnnotator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOX COCO Annotator - PySide6")
        self.resize(1400, 900)

        self.image_dir: Optional[Path] = None
        self.image_paths: List[Path] = []
        self.current_index: int = -1
        self.image_loaded = False
        self.current_image_width = 0
        self.current_image_height = 0

        self.categories: List[Dict] = [
            {"id": 1, "name": "defect"},
        ]
        self.annotations_by_image: Dict[str, List[Dict]] = {}
        self.next_ann_id = 1

        self.scene = QGraphicsScene(self)
        self.view = AnnotatorView(self)
        self.view.setScene(self.scene)
        self.pixmap_item: Optional[QGraphicsPixmapItem] = None

        self.category_combo = QComboBox()
        self.reload_category_combo()

        self.image_list = QListWidget()
        self.image_list.currentRowChanged.connect(self.load_image_by_index)

        self.annotation_list = QListWidget()
        self.annotation_list.itemClicked.connect(self.select_annotation_from_list)

        self.info_label = QLabel("尚未載入資料夾")
        self.info_label.setWordWrap(True)

        self._build_ui()
        self._build_actions()
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Open image folder to start. 左鍵畫框，滾輪縮放，Space+左鍵或中鍵拖曳平移。")

    def _build_ui(self):
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(QLabel("Images"))
        left_layout.addWidget(self.image_list, 2)
        left_layout.addWidget(QLabel("Category"))
        left_layout.addWidget(self.category_combo)

        btn_add_cat = QPushButton("新增類別")
        btn_add_cat.clicked.connect(self.add_category_dialog)
        left_layout.addWidget(btn_add_cat)

        btn_del_ann = QPushButton("刪除選取框 Del")
        btn_del_ann.clicked.connect(self.delete_selected_annotation)
        left_layout.addWidget(btn_del_ann)

        left_layout.addWidget(QLabel("Annotations"))
        left_layout.addWidget(self.annotation_list, 2)
        left_layout.addWidget(self.info_label)

        splitter = QSplitter()
        splitter.addWidget(left_panel)
        splitter.addWidget(self.view)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([320, 1080])
        self.setCentralWidget(splitter)

    def _build_actions(self):
        toolbar = QToolBar("Main")
        self.addToolBar(toolbar)

        open_action = QAction("開啟圖片資料夾", self)
        open_action.triggered.connect(self.open_image_folder)
        toolbar.addAction(open_action)

        import_action = QAction("匯入 COCO JSON", self)
        import_action.triggered.connect(self.import_coco_json)
        toolbar.addAction(import_action)

        save_action = QAction("輸出 COCO JSON", self)
        save_action.triggered.connect(self.export_coco_json_dialog)
        toolbar.addAction(save_action)

        prev_action = QAction("上一張 A", self)
        prev_action.triggered.connect(self.prev_image)
        toolbar.addAction(prev_action)

        next_action = QAction("下一張 D", self)
        next_action.triggered.connect(self.next_image)
        toolbar.addAction(next_action)

        fit_action = QAction("適合視窗 F", self)
        fit_action.triggered.connect(self.fit_image)
        toolbar.addAction(fit_action)

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Delete:
            self.delete_selected_annotation()
        elif key == Qt.Key_A:
            self.prev_image()
        elif key == Qt.Key_D:
            self.next_image()
        elif key == Qt.Key_F:
            self.fit_image()
        elif key == Qt.Key_S and event.modifiers() & Qt.ControlModifier:
            self.export_coco_json_dialog()
        else:
            super().keyPressEvent(event)

    def reload_category_combo(self):
        self.category_combo.clear()
        for c in self.categories:
            self.category_combo.addItem(f"{c['id']}: {c['name']}", c["id"])

    def add_category_dialog(self):
        name, ok = QInputDialog.getText(self, "新增類別", "Category name:")
        name = name.strip()
        if not ok or not name:
            return
        if any(c["name"] == name for c in self.categories):
            QMessageBox.warning(self, "類別已存在", f"Category '{name}' already exists.")
            return
        next_id = max([c["id"] for c in self.categories], default=0) + 1
        self.categories.append({"id": next_id, "name": name})
        self.reload_category_combo()
        self.category_combo.setCurrentIndex(self.category_combo.count() - 1)

    def open_image_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "選擇圖片資料夾")
        if not folder:
            return
        self.image_dir = Path(folder)
        self.image_paths = sorted([p for p in self.image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])
        self.image_list.clear()
        self.annotations_by_image.clear()
        self.current_index = -1
        self.next_ann_id = 1

        for p in self.image_paths:
            self.image_list.addItem(p.name)
            self.annotations_by_image[p.name] = []

        if not self.image_paths:
            QMessageBox.warning(self, "沒有圖片", "這個資料夾沒有 jpg/png/bmp/tif 圖片。")
            return

        self.image_list.setCurrentRow(0)
        self.update_info_label()

    def load_image_by_index(self, index: int):
        if index < 0 or index >= len(self.image_paths):
            return
        self.save_current_scene_items_to_memory()
        self.current_index = index
        image_path = self.image_paths[index]

        try:
            pixmap, width, height = load_image(image_path)
        except Exception as exc:
            QMessageBox.warning(self, "圖片讀取失敗", f"{image_path.name}\n{exc}")
            return

        self.current_image_width = width
        self.current_image_height = height
        self.scene.clear()
        self.pixmap_item = self.scene.addPixmap(pixmap)
        self.pixmap_item.setZValue(-10)
        self.scene.setSceneRect(QRectF(0, 0, self.current_image_width, self.current_image_height))
        self.image_loaded = True

        self.draw_annotations_for_current_image()
        self.fit_image()
        self.refresh_annotation_list()
        self.update_info_label()
        self.statusBar().showMessage(f"Loaded {image_path.name} ({self.current_image_width}x{self.current_image_height})")

    def current_image_name(self) -> Optional[str]:
        if 0 <= self.current_index < len(self.image_paths):
            return self.image_paths[self.current_index].name
        return None

    def is_inside_image(self, point: QPointF) -> bool:
        return 0 <= point.x() <= self.current_image_width and 0 <= point.y() <= self.current_image_height

    def clamp_to_image(self, point: QPointF) -> QPointF:
        x = min(max(point.x(), 0), self.current_image_width)
        y = min(max(point.y(), 0), self.current_image_height)
        return QPointF(x, y)

    def add_annotation(self, rect: QRectF):
        image_name = self.current_image_name()
        if not image_name:
            return
        category_id = int(self.category_combo.currentData())
        category_name = next(c["name"] for c in self.categories if c["id"] == category_id)
        ann = {
            "id": self.next_ann_id,
            "image_name": image_name,
            "category_id": category_id,
            "category_name": category_name,
            "bbox": [float(rect.x()), float(rect.y()), float(rect.width()), float(rect.height())],
            "area": float(rect.width() * rect.height()),
            "iscrowd": 0,
        }
        self.next_ann_id += 1
        self.annotations_by_image.setdefault(image_name, []).append(ann)
        item = BBoxRectItem(ann["id"], category_id, category_name, rect)
        self.scene.addItem(item)
        self.refresh_annotation_list()
        self.update_info_label()

    def draw_annotations_for_current_image(self):
        image_name = self.current_image_name()
        if not image_name:
            return
        for ann in self.annotations_by_image.get(image_name, []):
            x, y, w, h = ann["bbox"]
            cat_name = ann.get("category_name") or self.category_name_by_id(ann["category_id"])
            item = BBoxRectItem(ann["id"], ann["category_id"], cat_name, QRectF(x, y, w, h))
            self.scene.addItem(item)

    def save_current_scene_items_to_memory(self):
        image_name = self.current_image_name()
        if not image_name or not self.image_loaded:
            return
        saved = []
        for item in self.scene.items():
            if isinstance(item, BBoxRectItem):
                rect = item.mapRectToScene(item.rect()).normalized()
                rect = self.clamp_rect(rect)
                saved.append({
                    "id": item.ann_id,
                    "image_name": image_name,
                    "category_id": item.category_id,
                    "category_name": item.category_name,
                    "bbox": [float(rect.x()), float(rect.y()), float(rect.width()), float(rect.height())],
                    "area": float(rect.width() * rect.height()),
                    "iscrowd": 0,
                })
        self.annotations_by_image[image_name] = saved

    def clamp_rect(self, rect: QRectF) -> QRectF:
        x1 = min(max(rect.left(), 0), self.current_image_width)
        y1 = min(max(rect.top(), 0), self.current_image_height)
        x2 = min(max(rect.right(), 0), self.current_image_width)
        y2 = min(max(rect.bottom(), 0), self.current_image_height)
        return QRectF(QPointF(x1, y1), QPointF(x2, y2)).normalized()

    def refresh_annotation_list(self):
        self.save_current_scene_items_to_memory()
        self.annotation_list.clear()
        image_name = self.current_image_name()
        if not image_name:
            return
        anns = self.annotations_by_image.get(image_name, [])
        for ann in anns:
            x, y, w, h = ann["bbox"]
            text = f"#{ann['id']} [{ann['category_name']}] x={x:.0f}, y={y:.0f}, w={w:.0f}, h={h:.0f}"
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, ann["id"])
            self.annotation_list.addItem(item)

    def select_annotation_from_list(self, list_item: QListWidgetItem):
        ann_id = list_item.data(Qt.UserRole)
        for item in self.scene.items():
            if isinstance(item, BBoxRectItem):
                item.setSelected(item.ann_id == ann_id)
                item.update_style()

    def delete_selected_annotation(self):
        selected = [item for item in self.scene.selectedItems() if isinstance(item, BBoxRectItem)]
        if not selected:
            row = self.annotation_list.currentRow()
            if row >= 0:
                ann_id = self.annotation_list.item(row).data(Qt.UserRole)
                selected = [item for item in self.scene.items() if isinstance(item, BBoxRectItem) and item.ann_id == ann_id]
        for item in selected:
            self.scene.removeItem(item)
        self.save_current_scene_items_to_memory()
        self.refresh_annotation_list()
        self.update_info_label()

    def fit_image(self):
        if self.pixmap_item:
            self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def prev_image(self):
        if self.current_index > 0:
            self.image_list.setCurrentRow(self.current_index - 1)

    def next_image(self):
        if self.current_index < len(self.image_paths) - 1:
            self.image_list.setCurrentRow(self.current_index + 1)

    def category_name_by_id(self, category_id: int) -> str:
        for c in self.categories:
            if c["id"] == category_id:
                return c["name"]
        return "unknown"

    def export_coco_json_dialog(self):
        if not self.image_paths:
            QMessageBox.warning(self, "尚未載入圖片", "請先開啟圖片資料夾。")
            return
        default_path = str((self.image_dir or Path.cwd()) / "annotations_coco.json")
        path, _ = QFileDialog.getSaveFileName(self, "輸出 COCO JSON", default_path, "JSON Files (*.json)")
        if not path:
            return
        self.export_coco_json(Path(path))

    def export_coco_json(self, output_path: Path):
        self.save_current_scene_items_to_memory()
        images = []
        annotations = []
        image_id_by_name = {}

        for idx, img_path in enumerate(self.image_paths, start=1):
            width, height = _read_image_size(img_path)
            image_id_by_name[img_path.name] = idx
            images.append({
                "id": idx,
                "file_name": img_path.name,
                "width": width,
                "height": height,
            })

        ann_id = 1
        for image_name, anns in self.annotations_by_image.items():
            if image_name not in image_id_by_name:
                continue
            for ann in anns:
                x, y, w, h = ann["bbox"]
                if w <= 0 or h <= 0:
                    continue
                annotations.append({
                    "id": ann_id,
                    "image_id": image_id_by_name[image_name],
                    "category_id": int(ann["category_id"]),
                    "bbox": [round(float(x), 2), round(float(y), 2), round(float(w), 2), round(float(h), 2)],
                    "area": round(float(w * h), 2),
                    "iscrowd": 0,
                    "segmentation": [],
                })
                ann_id += 1

        coco = {
            "info": {"description": "YOLOX COCO annotations generated by PySide6 annotator"},
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": self.categories,
        }
        output_path.write_text(json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8")
        QMessageBox.information(self, "輸出完成", f"已輸出：\n{output_path}\n\nImages: {len(images)}\nAnnotations: {len(annotations)}")
        self.statusBar().showMessage(f"Saved COCO JSON: {output_path}")

    def import_coco_json(self):
        if not self.image_paths:
            QMessageBox.warning(self, "尚未載入圖片", "請先開啟圖片資料夾，再匯入 COCO JSON。")
            return
        path, _ = QFileDialog.getOpenFileName(self, "匯入 COCO JSON", str(self.image_dir or Path.cwd()), "JSON Files (*.json)")
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as exc:
            QMessageBox.critical(self, "匯入失敗", str(exc))
            return

        if "categories" in data:
            self.categories = [{"id": int(c["id"]), "name": str(c["name"])} for c in data["categories"]]
            self.reload_category_combo()

        image_name_by_id = {int(img["id"]): img["file_name"] for img in data.get("images", [])}
        valid_names = {p.name for p in self.image_paths}
        self.annotations_by_image = {p.name: [] for p in self.image_paths}
        max_ann_id = 0

        for ann in data.get("annotations", []):
            image_name = image_name_by_id.get(int(ann["image_id"]))
            if image_name not in valid_names:
                continue
            cat_id = int(ann["category_id"])
            cat_name = self.category_name_by_id(cat_id)
            x, y, w, h = ann["bbox"]
            ann_id = int(ann.get("id", max_ann_id + 1))
            max_ann_id = max(max_ann_id, ann_id)
            self.annotations_by_image[image_name].append({
                "id": ann_id,
                "image_name": image_name,
                "category_id": cat_id,
                "category_name": cat_name,
                "bbox": [float(x), float(y), float(w), float(h)],
                "area": float(w * h),
                "iscrowd": int(ann.get("iscrowd", 0)),
            })
        self.next_ann_id = max_ann_id + 1
        self.load_image_by_index(self.current_index if self.current_index >= 0 else 0)
        QMessageBox.information(self, "匯入完成", f"已匯入：{path}")

    def update_info_label(self):
        total_imgs = len(self.image_paths)
        total_anns = sum(len(v) for v in self.annotations_by_image.values())
        current_name = self.current_image_name() or "-"
        current_anns = len(self.annotations_by_image.get(current_name, [])) if current_name != "-" else 0
        self.info_label.setText(
            f"Current: {current_name}\n"
            f"Image: {self.current_index + 1 if self.current_index >= 0 else 0}/{total_imgs}\n"
            f"Current boxes: {current_anns}\n"
            f"Total boxes: {total_anns}\n\n"
            f"快捷鍵：A上一張 / D下一張 / Del刪框 / F適合視窗 / Ctrl+S輸出"
        )


def main():
    app = QApplication(sys.argv)
    window = YOLOXCOCOAnnotator()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
