"""
Interactive annotation canvas built on QGraphicsView.

Supports:
  - Zoom: Ctrl+scroll
  - Pan: middle-mouse drag
  - Draw new box: left-drag on empty canvas space
  - Move box: left-drag on box body
  - Resize box: left-drag on any of the 8 corner/edge handles
  - Select box: left-click
  - Delete box: Delete key (selected box)
"""

from __future__ import annotations

from qtpy.QtCore import Qt, QRectF, QPointF, Signal
from qtpy.QtGui import QColor, QPen, QBrush, QPixmap, QCursor, QPainter
from qtpy.QtWidgets import (
    QGraphicsView,
    QGraphicsScene,
    QGraphicsRectItem,
    QGraphicsEllipseItem,
    QGraphicsPixmapItem,
    QApplication,
)

from app.core.sam3_handler import BBox

# Visual constants
HANDLE_SIZE     = 8.0
HANDLE_COLOR    = QColor(255, 220, 0)
BOX_COLOR_SAM   = QColor(0, 180, 255, 180)
BOX_COLOR_MANUAL = QColor(0, 255, 120, 180)
BOX_SEL_COLOR   = QColor(255, 200, 0, 220)
BOX_THICKNESS   = 2

_HANDLES = [
    "tl", "tm", "tr",
    "ml",        "mr",
    "bl", "bm", "br",
]


class HandleItem(QGraphicsEllipseItem):
    def __init__(self, pos_key: str, parent: "AnnotationRect"):
        super().__init__(-HANDLE_SIZE / 2, -HANDLE_SIZE / 2, HANDLE_SIZE, HANDLE_SIZE, parent)
        self.pos_key = pos_key
        self.setBrush(QBrush(HANDLE_COLOR))
        self.setPen(QPen(Qt.black, 1))
        self.setZValue(10)
        self.setVisible(False)
        self.setCursor(self._cursor_for(pos_key))

    @staticmethod
    def _cursor_for(key: str) -> QCursor:
        mapping = {
            "tl": Qt.SizeFDiagCursor, "br": Qt.SizeFDiagCursor,
            "tr": Qt.SizeBDiagCursor, "bl": Qt.SizeBDiagCursor,
            "tm": Qt.SizeVerCursor,   "bm": Qt.SizeVerCursor,
            "ml": Qt.SizeHorCursor,   "mr": Qt.SizeHorCursor,
        }
        return QCursor(mapping.get(key, Qt.SizeAllCursor))


class AnnotationRect(QGraphicsRectItem):
    """A single annotation rectangle with resize handles."""

    def __init__(self, bbox: BBox, scene_rect: QRectF):
        super().__init__(scene_rect)
        self.bbox = bbox

        color = BOX_COLOR_SAM if bbox.source == "sam" else BOX_COLOR_MANUAL
        self.setPen(QPen(color, BOX_THICKNESS))
        self.setBrush(QBrush(Qt.transparent))
        self.setFlag(QGraphicsRectItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsRectItem.ItemIsMovable, False)  # movement handled manually
        self.setFlag(QGraphicsRectItem.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)

        self._handles: dict[str, HandleItem] = {}
        for key in _HANDLES:
            h = HandleItem(key, self)
            self._handles[key] = h

        self._sync_handles()

    # ------------------------------------------------------------------

    def _sync_handles(self) -> None:
        r = self.rect()
        cx, cy = r.center().x(), r.center().y()
        positions = {
            "tl": (r.left(),  r.top()),
            "tm": (cx,        r.top()),
            "tr": (r.right(), r.top()),
            "ml": (r.left(),  cy),
            "mr": (r.right(), cy),
            "bl": (r.left(),  r.bottom()),
            "bm": (cx,        r.bottom()),
            "br": (r.right(), r.bottom()),
        }
        for key, (x, y) in positions.items():
            self._handles[key].setPos(x, y)

    def show_handles(self, visible: bool) -> None:
        for h in self._handles.values():
            h.setVisible(visible)

    def set_selected_style(self, selected: bool) -> None:
        if selected:
            self.setPen(QPen(BOX_SEL_COLOR, BOX_THICKNESS + 1))
            self.show_handles(True)
        else:
            color = BOX_COLOR_SAM if self.bbox.source == "sam" else BOX_COLOR_MANUAL
            self.setPen(QPen(color, BOX_THICKNESS))
            self.show_handles(False)

    def apply_rect(self, r: QRectF) -> None:
        """Update the displayed rect and sync the underlying BBox."""
        self.setRect(r.normalized())
        r = self.rect()
        self.bbox.x1 = r.left()
        self.bbox.y1 = r.top()
        self.bbox.x2 = r.right()
        self.bbox.y2 = r.bottom()
        self._sync_handles()

    def handle_at(self, scene_pos: QPointF) -> str | None:
        for key, h in self._handles.items():
            if h.isVisible() and h.sceneBoundingRect().contains(scene_pos):
                return key
        return None


# ---------------------------------------------------------------------------
# Canvas
# ---------------------------------------------------------------------------

class AnnotationCanvas(QGraphicsView):
    box_added           = Signal(object)   # BBox
    box_deleted         = Signal(object)   # BBox
    box_edited          = Signal(object)   # BBox
    exemplars_changed   = Signal()         # a positive/negative exemplar was added/cleared

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self.setRenderHint(QPainter.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setBackgroundBrush(QBrush(QColor(20, 20, 20)))

        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._annotation_rects: list[AnnotationRect] = []
        self._selected_rect: AnnotationRect | None = None

        # Drawing state
        self._draw_start: QPointF | None = None
        self._rubber_band: QGraphicsRectItem | None = None

        # Resize / move state
        self._drag_handle: str | None = None
        self._drag_rect: AnnotationRect | None = None
        self._drag_start_scene: QPointF | None = None
        self._drag_orig_rect: QRectF | None = None

        # SAM 3 prompt mode (positive / negative exemplar boxes)
        self._sam_mode: bool = False
        self._current_class_id: int = 0
        self._exemplar_kind: str | None = None  # "pos" | "neg" while drawing
        self._pos_exemplars: list[QRectF] = []
        self._neg_exemplars: list[QRectF] = []
        self._exemplar_items: list[QGraphicsRectItem] = []

        # Pan state
        self._pan_last: QPointF | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_frame(self, png_path: str, boxes: list[BBox]) -> None:
        self._scene.clear()
        self._annotation_rects = []
        self._selected_rect = None
        self._rubber_band = None
        self._draw_start = None
        self._drag_handle = None
        self._drag_rect = None
        self._drag_start_scene = None
        self._drag_orig_rect = None
        # exemplars belong to a single frame — reset on frame change
        self._exemplar_items = []
        self._pos_exemplars = []
        self._neg_exemplars = []
        self._exemplar_kind = None
        self.exemplars_changed.emit()

        pixmap = QPixmap(png_path)
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))

        for bbox in boxes:
            self._add_rect_item(bbox)

        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def clear_canvas(self) -> None:
        self._scene.clear()
        self._annotation_rects = []
        self._selected_rect = None
        self._rubber_band = None
        self._draw_start = None
        self._drag_handle = None
        self._drag_rect = None
        self._drag_start_scene = None
        self._drag_orig_rect = None

    def set_class_id(self, class_id: int) -> None:
        self._current_class_id = class_id

    def set_sam_mode(self, enabled: bool) -> None:
        """Toggle SAM 3 prompt mode.

        In prompt mode the user draws exemplar boxes instead of annotations:
          left-drag  → POSITIVE exemplar (green)  — 'find more like this'
          right-drag → NEGATIVE exemplar (red)    — 'exclude things like this'
        """
        self._sam_mode = enabled
        self.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)
        if enabled:
            self._select(None)
        else:
            self.clear_exemplars()

    def get_exemplars(self):
        """Return (positive, negative) exemplar boxes as (x1, y1, x2, y2) tuples."""
        pos = [(r.left(), r.top(), r.right(), r.bottom()) for r in self._pos_exemplars]
        neg = [(r.left(), r.top(), r.right(), r.bottom()) for r in self._neg_exemplars]
        return pos, neg

    def clear_exemplars(self) -> None:
        for item in self._exemplar_items:
            if item.scene() is self._scene:
                self._scene.removeItem(item)
        self._exemplar_items = []
        self._pos_exemplars = []
        self._neg_exemplars = []
        self.exemplars_changed.emit()

    def _add_exemplar(self, rect: QRectF, kind: str) -> None:
        color = QColor(0, 230, 0) if kind == "pos" else QColor(255, 40, 40)
        pen = QPen(color, 2, Qt.DashLine)
        fill = QColor(color.red(), color.green(), color.blue(), 40)
        item = self._scene.addRect(rect, pen, QBrush(fill))
        item.setZValue(5)
        self._exemplar_items.append(item)
        (self._pos_exemplars if kind == "pos" else self._neg_exemplars).append(rect)
        self.exemplars_changed.emit()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add_rect_item(self, bbox: BBox) -> AnnotationRect:
        r = QRectF(bbox.x1, bbox.y1, bbox.x2 - bbox.x1, bbox.y2 - bbox.y1)
        item = AnnotationRect(bbox, r)
        self._scene.addItem(item)
        self._annotation_rects.append(item)
        return item

    def _select(self, item: AnnotationRect | None) -> None:
        if self._selected_rect and self._selected_rect is not item:
            self._selected_rect.set_selected_style(False)
        self._selected_rect = item
        if item:
            item.set_selected_style(True)

    def _rect_at(self, scene_pos: QPointF) -> AnnotationRect | None:
        for item in reversed(self._annotation_rects):
            if item.rect().contains(scene_pos):
                return item
        return None

    # ------------------------------------------------------------------
    # Mouse events
    # ------------------------------------------------------------------

    def wheelEvent(self, event) -> None:
        if event.modifiers() & Qt.ControlModifier:
            factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            self.scale(factor, factor)
        else:
            super().wheelEvent(event)

    def mousePressEvent(self, event) -> None:
        scene_pos = self.mapToScene(event.pos())

        if event.button() == Qt.MiddleButton:
            self._pan_last = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            return

        # SAM 3 prompt mode: left-drag = positive exemplar, right-drag = negative
        if self._sam_mode and event.button() in (Qt.LeftButton, Qt.RightButton):
            if self._pixmap_item is not None and \
                    self._pixmap_item.boundingRect().contains(scene_pos):
                self._exemplar_kind = "pos" if event.button() == Qt.LeftButton else "neg"
                self._draw_start = scene_pos
                color = QColor(0, 230, 0) if self._exemplar_kind == "pos" else QColor(255, 40, 40)
                pen = QPen(color, 2, Qt.DashLine)
                self._rubber_band = self._scene.addRect(
                    QRectF(scene_pos, scene_pos), pen, QBrush(Qt.transparent)
                )
            return

        if event.button() == Qt.LeftButton:
            # Check if clicking a resize handle
            if self._selected_rect:
                handle_key = self._selected_rect.handle_at(scene_pos)
                if handle_key:
                    self._drag_handle = handle_key
                    self._drag_rect = self._selected_rect
                    self._drag_start_scene = scene_pos
                    self._drag_orig_rect = QRectF(self._selected_rect.rect())
                    return

            # Check if clicking inside an existing rect body
            hit = self._rect_at(scene_pos)
            if hit:
                self._select(hit)
                # Prepare for body move
                self._drag_handle = "move"
                self._drag_rect = hit
                self._drag_start_scene = scene_pos
                self._drag_orig_rect = QRectF(hit.rect())
                return

            # Start drawing a new box
            self._select(None)
            self._draw_start = scene_pos
            pen = QPen(QColor(255, 255, 255, 180), 1, Qt.DashLine)
            self._rubber_band = self._scene.addRect(
                QRectF(scene_pos, scene_pos), pen, QBrush(Qt.transparent)
            )

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        scene_pos = self.mapToScene(event.pos())

        if self._pan_last is not None:
            delta = event.pos() - self._pan_last
            self._pan_last = event.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
            return

        if self._rubber_band and self._draw_start:
            self._rubber_band.setRect(
                QRectF(self._draw_start, scene_pos).normalized()
            )
            return

        if self._drag_rect and self._drag_start_scene and self._drag_orig_rect:
            dx = scene_pos.x() - self._drag_start_scene.x()
            dy = scene_pos.y() - self._drag_start_scene.y()
            orig = self._drag_orig_rect

            if self._drag_handle == "move":
                new_rect = orig.translated(dx, dy)
            else:
                new_rect = self._apply_handle_drag(
                    orig, self._drag_handle, dx, dy
                )
            self._drag_rect.apply_rect(new_rect)
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MiddleButton:
            self._pan_last = None
            self.setCursor(Qt.CrossCursor if self._sam_mode else Qt.ArrowCursor)
            return

        # SAM 3 prompt mode: finalize the exemplar box
        if self._exemplar_kind is not None and \
                event.button() in (Qt.LeftButton, Qt.RightButton):
            scene_pos = self.mapToScene(event.pos())
            kind = self._exemplar_kind
            self._exemplar_kind = None
            if self._rubber_band is not None:
                self._scene.removeItem(self._rubber_band)
                self._rubber_band = None
            start = self._draw_start
            self._draw_start = None
            if start is not None:
                rect = QRectF(start, scene_pos).normalized()
                if rect.width() > 5 and rect.height() > 5:
                    self._add_exemplar(rect, kind)
            return

        if event.button() == Qt.LeftButton:
            scene_pos = self.mapToScene(event.pos())

            if self._rubber_band and self._draw_start:
                final_rect = QRectF(self._draw_start, scene_pos).normalized()
                self._scene.removeItem(self._rubber_band)
                self._rubber_band = None
                self._draw_start = None

                if final_rect.width() > 5 and final_rect.height() > 5:
                    class_id = getattr(self, "_current_class_id", 0)
                    bbox = BBox(
                        x1=final_rect.left(), y1=final_rect.top(),
                        x2=final_rect.right(), y2=final_rect.bottom(),
                        class_id=class_id, source="manual",
                    )
                    item = self._add_rect_item(bbox)
                    self._select(item)
                    self.box_added.emit(bbox)
                return

            if self._drag_rect:
                handle = self._drag_handle
                rect_item = self._drag_rect
                self._drag_handle = None
                self._drag_rect = None
                self._drag_start_scene = None
                self._drag_orig_rect = None
                if handle is not None:
                    self.box_edited.emit(rect_item.bbox)
                return

        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Delete and self._selected_rect:
            bbox = self._selected_rect.bbox
            self._scene.removeItem(self._selected_rect)
            self._annotation_rects.remove(self._selected_rect)
            self._selected_rect = None
            self.box_deleted.emit(bbox)
            return
        super().keyPressEvent(event)

    # ------------------------------------------------------------------
    # Handle drag math
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_handle_drag(
        orig: QRectF, handle: str, dx: float, dy: float
    ) -> QRectF:
        l, t, r, b = orig.left(), orig.top(), orig.right(), orig.bottom()
        if "l" in handle:
            l += dx
        if "r" in handle:
            r += dx
        if "t" in handle:
            t += dy
        if "b" in handle:
            b += dy
        return QRectF(QPointF(l, t), QPointF(r, b)).normalized()
