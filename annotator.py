import copy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import QAction, QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from config import AppConfig
from coco_io import export_coco, import_coco
from graphics import AnnotatorView, BBoxRectItem

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
_UNDO_LIMIT = 50


class YOLOXCOCOAnnotator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOX COCO Annotator")
        self.resize(1400, 900)

        # --- persistent config ---
        self.config = AppConfig.load()
        self.categories: List[Dict] = self.config.categories

        # --- runtime state ---
        self.image_dir: Optional[Path] = None
        self.image_paths: List[Path] = []
        self.current_index: int = -1
        self.image_loaded = False
        self.current_image_width = 0
        self.current_image_height = 0
        self.image_size_cache: Dict[str, Tuple[int, int]] = {}
        self.annotations_by_image: Dict[str, List[Dict]] = {}
        self.next_ann_id = 1
        self._undo_stack: List[Tuple[str, List]] = []

        # --- Qt scene / view ---
        self.scene = QGraphicsScene(self)
        self.view = AnnotatorView(self)
        self.view.setScene(self.scene)
        self.pixmap_item: Optional[QGraphicsPixmapItem] = None

        # --- widgets ---
        self.category_combo = QComboBox()
        self.reload_category_combo()
        self.image_list = QListWidget()
        self.image_list.currentRowChanged.connect(self.load_image_by_index)
        self.annotation_list = QListWidget()
        self.annotation_list.itemClicked.connect(self.select_annotation_from_list)
        self.info_label = QLabel("尚未載入資料夾")
        self.info_label.setWordWrap(True)

        self._build_ui()
        self._build_toolbar()
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage(
            "開啟資料夾以開始。左鍵畫框，滾輪縮放，Space/中鍵拖曳，Ctrl+Z 復原。"
        )

        # restore last session
        if self.config.last_folder and Path(self.config.last_folder).exists():
            self._load_folder(Path(self.config.last_folder), self.config.last_index)

    # ------------------------------------------------------------------ UI --

    def _build_ui(self):
        left = QWidget()
        layout = QVBoxLayout(left)
        layout.addWidget(QLabel("Images"))
        layout.addWidget(self.image_list, 2)
        layout.addWidget(QLabel("Category"))
        layout.addWidget(self.category_combo)

        btn_add_cat = QPushButton("新增類別")
        btn_add_cat.clicked.connect(self.add_category_dialog)
        layout.addWidget(btn_add_cat)

        btn_del_ann = QPushButton("刪除選取框 Del")
        btn_del_ann.clicked.connect(self.delete_selected_annotation)
        layout.addWidget(btn_del_ann)

        layout.addWidget(QLabel("Annotations"))
        layout.addWidget(self.annotation_list, 2)
        layout.addWidget(self.info_label)

        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(self.view)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([320, 1080])
        self.setCentralWidget(splitter)

    def _build_toolbar(self):
        toolbar = QToolBar("Main")
        self.addToolBar(toolbar)
        for label, handler in [
            ("開啟圖片資料夾", self.open_image_folder),
            ("匯入 COCO JSON",  self.import_coco_json),
            ("輸出 COCO JSON",  self.export_coco_json_dialog),
            ("上一張 A",         self.prev_image),
            ("下一張 D",         self.next_image),
            ("適合視窗 F",       self.fit_image),
        ]:
            action = QAction(label, self)
            action.triggered.connect(handler)
            toolbar.addAction(action)

    # --------------------------------------------------------- key events --

    def keyPressEvent(self, event):
        k, mod = event.key(), event.modifiers()
        if k == Qt.Key_Delete:
            self.delete_selected_annotation()
        elif k == Qt.Key_A:
            self.prev_image()
        elif k == Qt.Key_D:
            self.next_image()
        elif k == Qt.Key_F:
            self.fit_image()
        elif k == Qt.Key_Z and mod & Qt.ControlModifier:
            self.undo()
        elif k == Qt.Key_S and mod & Qt.ControlModifier:
            self.export_coco_json_dialog()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self.save_current_scene_items_to_memory()
        self._persist_config()
        super().closeEvent(event)

    # ------------------------------------------------------------ config --

    def _persist_config(self):
        self.config.categories = self.categories
        self.config.last_folder = str(self.image_dir) if self.image_dir else None
        self.config.last_index = max(self.current_index, 0)
        self.config.save()

    # --------------------------------------------------------- categories --

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
            QMessageBox.warning(self, "類別已存在", f"'{name}' 已存在。")
            return
        next_id = max((c["id"] for c in self.categories), default=0) + 1
        self.categories.append({"id": next_id, "name": name})
        self.reload_category_combo()
        self.category_combo.setCurrentIndex(self.category_combo.count() - 1)
        self._persist_config()

    def category_name_by_id(self, category_id: int) -> str:
        for c in self.categories:
            if c["id"] == category_id:
                return c["name"]
        return "unknown"

    # ------------------------------------------------ folder / image load --

    def open_image_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "選擇圖片資料夾")
        if folder:
            self._load_folder(Path(folder), 0)

    def _load_folder(self, folder: Path, initial_index: int = 0):
        self.image_dir = folder
        self.image_paths = sorted(
            p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS
        )
        self.annotations_by_image = {p.name: [] for p in self.image_paths}
        self.image_size_cache.clear()
        self.current_index = -1
        self.next_ann_id = 1
        self._undo_stack.clear()

        self.image_list.clear()
        for p in self.image_paths:
            self.image_list.addItem(p.name)

        if not self.image_paths:
            QMessageBox.warning(self, "沒有圖片", "該資料夾內沒有支援的圖片格式。")
            return

        self.image_list.setCurrentRow(min(initial_index, len(self.image_paths) - 1))
        self.update_info_label()

    def load_image_by_index(self, index: int):
        if index < 0 or index >= len(self.image_paths):
            return
        self.save_current_scene_items_to_memory()
        self.current_index = index
        img_path = self.image_paths[index]

        image = QImage(str(img_path))
        if image.isNull():
            QMessageBox.warning(self, "圖片讀取失敗", str(img_path))
            return

        self.current_image_width = image.width()
        self.current_image_height = image.height()
        self.image_size_cache[img_path.name] = (self.current_image_width, self.current_image_height)

        self.scene.clear()
        self.pixmap_item = self.scene.addPixmap(QPixmap.fromImage(image))
        self.pixmap_item.setZValue(-10)
        self.scene.setSceneRect(QRectF(0, 0, self.current_image_width, self.current_image_height))
        self.image_loaded = True

        self.draw_annotations_for_current_image()
        self.fit_image()
        self.refresh_annotation_list()
        self.update_info_label()
        self.statusBar().showMessage(
            f"Loaded {img_path.name} ({self.current_image_width}×{self.current_image_height})"
        )

    def current_image_name(self) -> Optional[str]:
        if 0 <= self.current_index < len(self.image_paths):
            return self.image_paths[self.current_index].name
        return None

    def is_inside_image(self, point: QPointF) -> bool:
        return 0 <= point.x() <= self.current_image_width and 0 <= point.y() <= self.current_image_height

    def clamp_to_image(self, point: QPointF) -> QPointF:
        return QPointF(
            min(max(point.x(), 0), self.current_image_width),
            min(max(point.y(), 0), self.current_image_height),
        )

    # ---------------------------------------------------- undo / history --

    def _push_undo(self):
        name = self.current_image_name()
        if not name:
            return
        self._undo_stack.append((name, copy.deepcopy(self.annotations_by_image.get(name, []))))
        if len(self._undo_stack) > _UNDO_LIMIT:
            self._undo_stack.pop(0)

    def undo(self):
        if not self._undo_stack:
            self.statusBar().showMessage("沒有可復原的操作。")
            return
        name, snapshot = self._undo_stack[-1]
        if name != self.current_image_name():
            self.statusBar().showMessage("Undo 只支援當前圖片的操作。")
            return
        self._undo_stack.pop()
        self.annotations_by_image[name] = snapshot
        self._redraw_current_annotations()
        self.refresh_annotation_list()
        self.update_info_label()
        self._update_image_list_item(self.current_index)
        self.statusBar().showMessage("已復原。")

    # --------------------------------------------------- annotation CRUD --

    def add_annotation(self, rect: QRectF):
        image_name = self.current_image_name()
        if not image_name:
            return
        self._push_undo()
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
        self.scene.addItem(BBoxRectItem(ann["id"], category_id, category_name, rect))
        self.refresh_annotation_list()
        self.update_info_label()
        self._update_image_list_item(self.current_index)

    def delete_selected_annotation(self):
        self._push_undo()
        selected = [i for i in self.scene.selectedItems() if isinstance(i, BBoxRectItem)]
        if not selected:
            row = self.annotation_list.currentRow()
            if row >= 0:
                ann_id = self.annotation_list.item(row).data(Qt.UserRole)
                selected = [
                    i for i in self.scene.items()
                    if isinstance(i, BBoxRectItem) and i.ann_id == ann_id
                ]
        for item in selected:
            self.scene.removeItem(item)
        self.save_current_scene_items_to_memory()
        self.refresh_annotation_list()
        self.update_info_label()
        self._update_image_list_item(self.current_index)

    def draw_annotations_for_current_image(self):
        image_name = self.current_image_name()
        if not image_name:
            return
        for ann in self.annotations_by_image.get(image_name, []):
            x, y, w, h = ann["bbox"]
            cat_name = ann.get("category_name") or self.category_name_by_id(ann["category_id"])
            self.scene.addItem(BBoxRectItem(ann["id"], ann["category_id"], cat_name, QRectF(x, y, w, h)))

    def _redraw_current_annotations(self):
        for item in [i for i in self.scene.items() if isinstance(i, BBoxRectItem)]:
            self.scene.removeItem(item)
        self.draw_annotations_for_current_image()

    def save_current_scene_items_to_memory(self):
        image_name = self.current_image_name()
        if not image_name or not self.image_loaded:
            return
        saved = []
        for item in self.scene.items():
            if isinstance(item, BBoxRectItem):
                rect = self._clamp_rect(item.mapRectToScene(item.rect()).normalized())
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

    def _clamp_rect(self, rect: QRectF) -> QRectF:
        x1 = min(max(rect.left(),  0), self.current_image_width)
        y1 = min(max(rect.top(),   0), self.current_image_height)
        x2 = min(max(rect.right(), 0), self.current_image_width)
        y2 = min(max(rect.bottom(),0), self.current_image_height)
        return QRectF(QPointF(x1, y1), QPointF(x2, y2)).normalized()

    # ------------------------------------------------- annotation list UI --

    def refresh_annotation_list(self):
        self.save_current_scene_items_to_memory()
        self.annotation_list.clear()
        image_name = self.current_image_name()
        if not image_name:
            return
        for ann in self.annotations_by_image.get(image_name, []):
            x, y, w, h = ann["bbox"]
            text = f"#{ann['id']} [{ann['category_name']}] x={x:.0f} y={y:.0f} w={w:.0f} h={h:.0f}"
            list_item = QListWidgetItem(text)
            list_item.setData(Qt.UserRole, ann["id"])
            self.annotation_list.addItem(list_item)

    def select_annotation_from_list(self, list_item: QListWidgetItem):
        ann_id = list_item.data(Qt.UserRole)
        for item in self.scene.items():
            if isinstance(item, BBoxRectItem):
                item.setSelected(item.ann_id == ann_id)

    # --------------------------------------------- image list with counts --

    def _update_image_list_item(self, index: int):
        if index < 0 or index >= len(self.image_paths):
            return
        name = self.image_paths[index].name
        count = len(self.annotations_by_image.get(name, []))
        self.image_list.item(index).setText(f"{name}  ({count})" if count else name)

    def _refresh_all_image_list_counts(self):
        for i in range(len(self.image_paths)):
            self._update_image_list_item(i)

    # ------------------------------------------------------- view helpers --

    def fit_image(self):
        if self.pixmap_item:
            self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def prev_image(self):
        if self.current_index > 0:
            self.image_list.setCurrentRow(self.current_index - 1)

    def next_image(self):
        if self.current_index < len(self.image_paths) - 1:
            self.image_list.setCurrentRow(self.current_index + 1)

    # ----------------------------------------------------------- COCO I/O --

    def export_coco_json_dialog(self):
        if not self.image_paths:
            QMessageBox.warning(self, "尚未載入圖片", "請先開啟圖片資料夾。")
            return
        default = str((self.image_dir or Path.cwd()) / "annotations_coco.json")
        path, _ = QFileDialog.getSaveFileName(self, "輸出 COCO JSON", default, "JSON Files (*.json)")
        if not path:
            return
        self.save_current_scene_items_to_memory()
        n_imgs, n_anns = export_coco(
            self.image_paths, self.annotations_by_image,
            self.categories, self.image_size_cache, Path(path),
        )
        QMessageBox.information(
            self, "輸出完成",
            f"已輸出：\n{path}\n\nImages: {n_imgs}\nAnnotations: {n_anns}",
        )
        self.statusBar().showMessage(f"Saved → {path}")

    def import_coco_json(self):
        if not self.image_paths:
            QMessageBox.warning(self, "尚未載入圖片", "請先開啟圖片資料夾，再匯入 COCO JSON。")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "匯入 COCO JSON", str(self.image_dir or Path.cwd()), "JSON Files (*.json)"
        )
        if not path:
            return
        try:
            cats, anns_by_img, max_id = import_coco(
                Path(path),
                {p.name for p in self.image_paths},
                self.category_name_by_id,
            )
        except Exception as exc:
            QMessageBox.critical(self, "匯入失敗", str(exc))
            return

        if cats is not None:
            self.categories = cats
            self.reload_category_combo()
        self.annotations_by_image = anns_by_img
        self.next_ann_id = max_id + 1
        self._refresh_all_image_list_counts()
        self.load_image_by_index(max(self.current_index, 0))
        QMessageBox.information(self, "匯入完成", f"已匯入：{path}")

    # ------------------------------------------------------ info label --

    def update_info_label(self):
        total_imgs = len(self.image_paths)
        total_anns = sum(len(v) for v in self.annotations_by_image.values())
        cur_name = self.current_image_name() or "-"
        cur_anns = len(self.annotations_by_image.get(cur_name, [])) if cur_name != "-" else 0
        self.info_label.setText(
            f"Current: {cur_name}\n"
            f"Image: {self.current_index + 1 if self.current_index >= 0 else 0}/{total_imgs}\n"
            f"Current boxes: {cur_anns}\n"
            f"Total boxes: {total_anns}\n\n"
            f"A 上一張 / D 下一張\n"
            f"Del 刪框 / F 適合視窗\n"
            f"Ctrl+Z 復原 / Ctrl+S 輸出"
        )
