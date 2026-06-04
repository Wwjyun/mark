from typing import Optional

from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QGraphicsRectItem,
    QGraphicsView,
    QStyleOptionGraphicsItem,
)

# One distinct colour per category (cycles if > 8 categories)
_PALETTE = [
    QColor(0, 180, 255),    # blue
    QColor(0, 220, 80),     # green
    QColor(255, 160, 0),    # orange
    QColor(220, 0, 220),    # magenta
    QColor(255, 60, 60),    # red
    QColor(0, 220, 220),    # cyan
    QColor(180, 100, 255),  # purple
    QColor(255, 220, 0),    # yellow
]


def category_color(category_id: int) -> QColor:
    return _PALETTE[(category_id - 1) % len(_PALETTE)]


class BBoxRectItem(QGraphicsRectItem):
    def __init__(self, ann_id: int, category_id: int, category_name: str, rect: QRectF):
        super().__init__(rect)
        self.ann_id = ann_id
        self.category_id = category_id
        self.category_name = category_name
        self._color = category_color(category_id)
        self.setFlags(
            QGraphicsRectItem.ItemIsSelectable
            | QGraphicsRectItem.ItemIsMovable
            | QGraphicsRectItem.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)
        self._apply_style(selected=False)

    def _apply_style(self, selected: bool):
        color = QColor(255, 80, 80) if selected else self._color
        pen = QPen(color, 2)
        pen.setCosmetic(True)
        self.setPen(pen)
        fill = QColor(color)
        fill.setAlpha(35)
        self.setBrush(QBrush(fill))

    # Fix: use ItemSelectedHasChanged so isSelected() is already updated
    def itemChange(self, change, value):
        if change == QGraphicsRectItem.ItemSelectedHasChanged:
            self._apply_style(selected=bool(value))
        return super().itemChange(change, value)

    def paint(self, painter: QPainter, option, widget=None):
        super().paint(painter, option, widget)
        rect = self.rect()
        lod = QStyleOptionGraphicsItem.levelOfDetailFromTransform(painter.worldTransform())
        # Keep label text at roughly 12 px on screen regardless of zoom
        px = max(8, min(16, int(12 / lod)))
        font = QFont()
        font.setPixelSize(px)
        painter.setFont(font)
        text_w = len(self.category_name) * px * 0.65 + 4
        painter.fillRect(QRectF(rect.left(), rect.top(), text_w, px + 3), QColor(0, 0, 0, 160))
        painter.setPen(QPen(Qt.white))
        painter.drawText(QPointF(rect.left() + 2, rect.top() + px), self.category_name)


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
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def mousePressEvent(self, event):
        mw = self.main_window
        if event.button() == Qt.MiddleButton or (
            event.button() == Qt.LeftButton and event.modifiers() & Qt.SpaceModifier
        ):
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            return

        if event.button() == Qt.LeftButton and mw and mw.image_loaded:
            scene_pos = self.mapToScene(event.pos())
            if mw.is_inside_image(scene_pos):
                clicked_item = self.itemAt(event.pos())
                if isinstance(clicked_item, BBoxRectItem):
                    super().mousePressEvent(event)
                    mw.refresh_annotation_list()
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
            current_pos = self.main_window.clamp_to_image(self.mapToScene(event.pos()))
            rect = QRectF(self._start_scene_pos, current_pos).normalized()
            self._temp_rect.setRect(rect)
            self.main_window.statusBar().showMessage(
                f"Drawing bbox: x={rect.x():.0f}, y={rect.y():.0f}, "
                f"w={rect.width():.0f}, h={rect.height():.0f}"
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
