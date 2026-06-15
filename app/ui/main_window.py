"""
Main application window.

Layout
──────
  ┌─────────────────────────────────────────────────────────────┐
  │  Header — app title  ·  [Media] [Annotate] [Train] tabs     │
  ├────────────┬────────────────────────────────────────────────┤
  │  Context   │  Thumbnail strip  │  Annotation canvas         │
  │  panel     │                   │                            │
  │  (changes  │                   │                            │
  │  per tab)  │                   │                            │
  ├────────────┴────────────────────────────────────────────────┤
  │  Status bar — progress bar · status text                    │
  └─────────────────────────────────────────────────────────────┘

The three tabs drive a QStackedWidget panel on the left:
  0 · Media    — import + stride
  1 · Annotate — class selector / SAM / prompt / propagate
  2 · Train    — export + train
"""

from __future__ import annotations

import os
import tempfile

# Project root — two directories above this file (app/ui/ → app/ → project root)
_APP_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)

from qtpy.QtCore import Qt, QSize, Slot
from qtpy.QtGui import QColor, QPalette, QIcon, QPixmap, QImage, QFont
from qtpy.QtWidgets import (
    QMainWindow,
    QWidget,
    QSplitter,
    QListWidget,
    QListWidgetItem,
    QComboBox,
    QProgressBar,
    QStatusBar,
    QLabel,
    QAction,
    QToolButton,
    QScrollArea,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QFileDialog,
    QInputDialog,
    QMessageBox,
    QSpinBox,
    QLineEdit,
    QApplication,
    QSizePolicy,
    QStackedWidget,
    QPushButton,
    QButtonGroup,
)

from app.core.sam3_handler import (
    AnnotationStore, BBox, FrameAnnotation,
    SAM3TextWorker, SAM3LoadWorker, SAM3PromptWorker, SAM3TrackWorker, SAM_MODELS,
)
from app.core.media_loader import MediaLoaderWorker
from app.core.yolo_trainer import (
    YOLOTrainWorker, MODEL_REGISTRY,
    DETECTION_MODELS, SEGMENTATION_MODELS,
    _is_sam2_key,
)
from app.core.sam2_trainer import SAM2TrainWorker, SAM2_MODELS
from app.core.onnx_exporter import ONNXExportWorker
from app.ui.canvas import AnnotationCanvas
from app.utils.yolo_exporter import export_dataset


# ── Colour palette ──────────────────────────────────────────────────────────

_BG       = "#0d0d16"    # window background (deepest)
_HDR      = "#0a0a12"    # header bar
_PANEL    = "#10101a"    # left panel
_SURFACE  = "#181826"    # card surface / subtle elevation
_BORDER   = "#22223a"    # divider lines
_TEXT     = "#e2e8f0"    # primary text
_MUTED    = "#64748b"    # secondary / label text
_ACCENT   = "#3b82f6"    # blue — primary actions
_GREEN    = "#10b981"    # green — train / success
_PURPLE   = "#7c3aed"    # purple — prompt mode active

_THUMBNAIL_SIZE = 100

_STATUS_COLORS = {
    "pending":  QColor(245, 158, 11),
    "verified": QColor(16, 185, 129),
    "exported": QColor(100, 116, 139),
}


# ── Palette helper ───────────────────────────────────────────────────────────

def _apply_dark_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    pal = QPalette()
    pairs = {
        QPalette.Window:          QColor(13, 13, 22),
        QPalette.WindowText:      QColor(226, 232, 240),
        QPalette.Base:            QColor(10, 10, 18),
        QPalette.AlternateBase:   QColor(22, 22, 34),
        QPalette.ToolTipBase:     QColor(255, 255, 220),
        QPalette.ToolTipText:     QColor(0, 0, 0),
        QPalette.Text:            QColor(226, 232, 240),
        QPalette.Button:          QColor(24, 24, 38),
        QPalette.ButtonText:      QColor(226, 232, 240),
        QPalette.BrightText:      QColor(255, 100, 100),
        QPalette.Link:            QColor(59, 130, 246),
        QPalette.Highlight:       QColor(59, 130, 246),
        QPalette.HighlightedText: QColor(255, 255, 255),
        QPalette.Disabled + QPalette.Text:       QColor(70, 70, 90),
        QPalette.Disabled + QPalette.ButtonText: QColor(70, 70, 90),
    }
    for role, color in pairs.items():
        pal.setColor(role, color)
    app.setPalette(pal)


# ── Small UI factories ───────────────────────────────────────────────────────

def _make_thumbnail(png_path: str) -> QPixmap:
    img = QImage(png_path)
    if img.isNull():
        img = QImage(_THUMBNAIL_SIZE, _THUMBNAIL_SIZE, QImage.Format_RGB32)
        img.fill(QColor(24, 24, 38))
    return QPixmap.fromImage(img).scaled(
        _THUMBNAIL_SIZE, _THUMBNAIL_SIZE,
        Qt.KeepAspectRatio, Qt.SmoothTransformation,
    )


def _hr() -> QFrame:
    """Thin horizontal rule used as a section divider."""
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Plain)
    line.setStyleSheet(f"background: {_BORDER}; border: none;")
    line.setFixedHeight(1)
    return line


def _section_lbl(text: str) -> QLabel:
    lbl = QLabel(text.upper())
    lbl.setStyleSheet(
        f"color: {_MUTED}; font-size: 9px; font-weight: bold; "
        "letter-spacing: 1.5px; padding: 12px 0 4px 0;"
    )
    return lbl


def _tool_btn(action: QAction, css: str) -> QToolButton:
    btn = QToolButton()
    btn.setDefaultAction(action)
    btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
    btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    btn.setMinimumHeight(34)
    btn.setStyleSheet(css)
    return btn


def _row_layout(label: str, widget: QWidget) -> QHBoxLayout:
    row = QHBoxLayout()
    lbl = QLabel(label)
    lbl.setStyleSheet(f"color: {_MUTED}; font-size: 11px;")
    lbl.setMinimumWidth(58)
    row.addWidget(lbl)
    row.addWidget(widget, 1)
    return row


# ── Button CSS factories ─────────────────────────────────────────────────────

def _css_primary(color: str = _ACCENT, hover: str = "#2563eb") -> str:
    return (
        f"QToolButton {{ background: {color}; color: #ffffff; border: none; "
        f"border-radius: 7px; padding: 7px 10px; font-size: 12px; font-weight: 600; }}"
        f"QToolButton:hover:enabled {{ background: {hover}; }}"
        f"QToolButton:disabled {{ background: {_SURFACE}; color: {_MUTED}; }}"
    )


def _css_neutral() -> str:
    return (
        f"QToolButton {{ background: {_SURFACE}; color: {_TEXT}; border: 1px solid {_BORDER}; "
        f"border-radius: 7px; padding: 7px 10px; font-size: 12px; }}"
        f"QToolButton:hover:enabled {{ background: #202032; border-color: #3a3a55; }}"
        f"QToolButton:disabled {{ color: {_MUTED}; border-color: {_BORDER}; background: {_SURFACE}; }}"
    )


def _css_toggle() -> str:
    """Checkable button — lights up purple when active."""
    return (
        f"QToolButton {{ background: {_SURFACE}; color: {_TEXT}; border: 1px solid {_BORDER}; "
        f"border-radius: 7px; padding: 7px 10px; font-size: 12px; }}"
        f"QToolButton:hover:enabled {{ background: #202032; }}"
        f"QToolButton:checked {{ background: {_PURPLE}; color: #ffffff; border-color: {_PURPLE}; }}"
        f"QToolButton:disabled {{ color: {_MUTED}; }}"
    )


def _combo_css() -> str:
    return (
        f"QComboBox {{ background: {_SURFACE}; color: {_TEXT}; border: 1px solid {_BORDER}; "
        f"border-radius: 6px; padding: 4px 8px; font-size: 11px; }}"
        f"QComboBox::drop-down {{ border: none; width: 20px; }}"
        f"QComboBox QAbstractItemView {{ background: {_SURFACE}; color: {_TEXT}; "
        f"selection-background-color: {_ACCENT}; }}"
    )


def _spin_css() -> str:
    return (
        f"QSpinBox {{ background: {_SURFACE}; color: {_TEXT}; border: 1px solid {_BORDER}; "
        f"border-radius: 6px; padding: 4px 6px; font-size: 11px; }}"
        f"QSpinBox::up-button, QSpinBox::down-button {{ border: none; background: transparent; width: 16px; }}"
    )


def _line_edit_css() -> str:
    return (
        f"QLineEdit {{ background: {_SURFACE}; color: {_TEXT}; border: 1px solid {_BORDER}; "
        f"border-radius: 6px; padding: 4px 8px; font-size: 11px; }}"
        f"QLineEdit:focus {{ border-color: {_ACCENT}; }}"
    )


# ── Main window ──────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    _TAB_MEDIA    = 0
    _TAB_ANNOTATE = 1
    _TAB_TRAIN    = 2

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ModelTrainer")
        self.resize(1280, 820)

        app = QApplication.instance()
        if app:
            _apply_dark_theme(app)

        # ── App state ─────────────────────────────────────────────
        self.store: AnnotationStore = {}
        self.class_names: list[str] = ["object"]
        self.current_frame_index: int | None = None
        self._temp_dir: str = tempfile.mkdtemp(prefix="modeltrainer_")
        self._frame_count: int = 0
        # Maps source_video path → import stride used (for tracking warnings)
        self._video_import_stride: dict[str, int] = {}
        # Set when importing a YOLO dataset — labels dir to apply after import
        self._pending_labels_dir: str | None = None

        # Active workers
        self._media_worker: MediaLoaderWorker | None = None
        self._train_worker: YOLOTrainWorker | None = None
        self._sam2_train_worker: SAM2TrainWorker | None = None
        self._seg_converter_worker = None
        self._onnx_worker: ONNXExportWorker | None = None

        # SAM 3 state
        self._sam_load_worker: SAM3LoadWorker | None = None
        self._sam_text_worker: SAM3TextWorker | None = None
        self._sam_prompt_worker: SAM3PromptWorker | None = None
        self._sam_track_worker: SAM3TrackWorker | None = None
        self._sam_model_obj = None
        self._sam_processor_obj = None
        self._sam_pending_action = None
        self._sam_tracker_model = None
        self._sam_tracker_processor = None
        self._track_start_index: int | None = None

        # ── Build UI ──────────────────────────────────────────────
        self._create_controls()
        self._build_header()
        self._build_panel()
        self._build_work_area()
        self._build_statusbar()
        self._assemble()
        self._switch_tab(self._TAB_MEDIA)

    # ──────────────────────────────────────────────────────────────
    # Control / action creation
    # ──────────────────────────────────────────────────────────────

    def _create_controls(self) -> None:
        """Create every QAction, QComboBox, QSpinBox, etc. (no layout)."""

        # ── Media ─────────────────────────────────────────────────
        self._act_import = QAction("↺  Replace All Media", self)
        self._act_import.setToolTip(
            "Clear everything and import a fresh folder of images or videos"
        )
        self._act_import.triggered.connect(self._on_import)

        self._act_add_more = QAction("＋  Add More Media", self)
        self._act_add_more.setToolTip(
            "Append images/videos to the current session without clearing existing frames"
        )
        self._act_add_more.triggered.connect(self._on_add_more)

        self._act_import_dataset = QAction("📦  Import YOLO Dataset", self)
        self._act_import_dataset.setToolTip(
            "Import a folder containing images/, labels/, and data.yaml.\n"
            "Automatically loads class names and existing bounding box annotations."
        )
        self._act_import_dataset.triggered.connect(self._on_import_dataset)

        self._import_stride_spin = QSpinBox()
        self._import_stride_spin.setRange(1, 120)
        self._import_stride_spin.setValue(1)
        self._import_stride_spin.setSpecialValueText("every frame")
        self._import_stride_spin.setToolTip(
            "Video import stride: keep every Nth raw frame.\n"
            "1 = keep every frame; 5 = keep 1 in 5 (≈ 80% fewer frames)."
        )
        self._import_stride_spin.setStyleSheet(_spin_css())

        self._stride_spin = QSpinBox()
        self._stride_spin.setRange(1, 120)
        self._stride_spin.setValue(1)
        self._stride_spin.setToolTip("Run SAM on every Nth imported frame (1 = all frames)")
        self._stride_spin.setStyleSheet(_spin_css())

        self._sam_start_spin = QSpinBox()
        self._sam_start_spin.setRange(0, 999_999)
        self._sam_start_spin.setValue(0)
        self._sam_start_spin.setSpecialValueText("from start")
        self._sam_start_spin.setToolTip(
            "First frame index to process.\n"
            "0 = process all frames; 50 = skip frames 0–49."
        )
        self._sam_start_spin.setStyleSheet(_spin_css())

        self._propagate_start_spin = QSpinBox()
        self._propagate_start_spin.setRange(0, 999_999)
        self._propagate_start_spin.setValue(0)
        self._propagate_start_spin.setSpecialValueText("from start")
        self._propagate_start_spin.setToolTip(
            "First frame index to propagate to.\n"
            "0 = propagate to all frames; 50 = skip frames 0–49."
        )
        self._propagate_start_spin.setStyleSheet(_spin_css())

        self._frame_count_lbl = QLabel("No media loaded")
        self._frame_count_lbl.setStyleSheet(
            f"color: {_MUTED}; font-size: 11px; padding: 4px 0;"
        )

        # ── Auto-label ────────────────────────────────────────────
        self._act_sam = QAction("▶  Run SAM 3", self)
        self._act_sam.setToolTip(
            "Auto-label every frame: SAM 3 finds all objects matching\n"
            "the Concept text (or the current class name if blank)."
        )
        self._act_sam.setEnabled(False)
        self._act_sam.triggered.connect(self._on_run_sam)

        self._concept_edit = QLineEdit()
        self._concept_edit.setPlaceholderText("e.g. car  (blank = class name)")
        self._concept_edit.setToolTip(
            "Text concept SAM 3 looks for.\n"
            "Leave blank to use the selected class name."
        )
        self._concept_edit.setStyleSheet(_line_edit_css())

        self._sam_model_combo = QComboBox()
        self._sam_model_combo.addItems(list(SAM_MODELS.keys()))
        self._sam_model_combo.setCurrentText("SAM 3.1 (facebook/sam3.1)")
        self._sam_model_combo.setStyleSheet(_combo_css())

        # ── Interactive prompt ────────────────────────────────────
        self._act_sam_prompt = QAction("⊙  SAM 3 Prompt", self)
        self._act_sam_prompt.setCheckable(True)
        self._act_sam_prompt.setEnabled(False)
        self._act_sam_prompt.setToolTip(
            "Toggle interactive prompt mode:\n"
            "  left-drag  → POSITIVE example (green)\n"
            "  right-drag → NEGATIVE example (red)\n"
            "Then press Run Prompt."
        )
        self._act_sam_prompt.toggled.connect(self._on_sam_prompt_toggled)

        self._act_run_prompt = QAction("▶  Run Prompt", self)
        self._act_run_prompt.setEnabled(False)
        self._act_run_prompt.setToolTip("Segment the current frame from your examples")
        self._act_run_prompt.triggered.connect(self._on_run_sam_prompt)

        self._act_clear_prompt = QAction("✕  Clear Examples", self)
        self._act_clear_prompt.setEnabled(False)
        self._act_clear_prompt.setToolTip("Remove all positive / negative example boxes")
        self._act_clear_prompt.triggered.connect(self._on_clear_prompts)

        # ── Propagate ─────────────────────────────────────────────
        self._act_propagate = QAction("↗  Propagate Labels", self)
        self._act_propagate.setEnabled(False)
        self._act_propagate.setToolTip(
            "Use THIS frame's labels to auto-label all frames.\n"
            "Each class becomes a SAM 3 concept searched everywhere.\n"
            "Best for image sets / mixed scenes."
        )
        self._act_propagate.triggered.connect(self._on_propagate_labels)

        self._act_track = QAction("▶▶  Track Video", self)
        self._act_track.setEnabled(False)
        self._act_track.setToolTip(
            "Follow the exact objects boxed on THIS frame\n"
            "through the video using SAM 3 memory tracking.\n"
            "Best for ordered video of the same moving objects."
        )
        self._act_track.triggered.connect(self._on_track_video)

        self._track_range_spin = QSpinBox()
        self._track_range_spin.setRange(0, 100_000)
        self._track_range_spin.setValue(0)
        self._track_range_spin.setSpecialValueText("all")
        self._track_range_spin.setToolTip(
            "Max frames to track forward from the seed frame.\n0 = whole video."
        )
        self._track_range_spin.setStyleSheet(_spin_css())

        # ── Classes ───────────────────────────────────────────────
        self._act_classes = QAction("Edit Classes…", self)
        self._act_classes.setToolTip("Define annotation class names")
        self._act_classes.triggered.connect(self._on_edit_classes)

        self._class_combo = QComboBox()
        self._class_combo.addItems(self.class_names)
        self._class_combo.setStyleSheet(_combo_css())
        self._class_combo.currentIndexChanged.connect(self._on_class_changed)

        # ── Export & train ────────────────────────────────────────
        self._act_export = QAction("↓  Export YOLO", self)
        self._act_export.setToolTip("Export annotations to YOLO .txt format")
        self._act_export.setEnabled(False)
        self._act_export.triggered.connect(self._on_export)

        self._task_combo = QComboBox()
        self._task_combo.addItems(["Detection", "Segmentation"])
        self._task_combo.setStyleSheet(_combo_css())
        self._task_combo.currentIndexChanged.connect(self._on_task_changed)

        self._model_combo = QComboBox()
        self._model_combo.addItems(DETECTION_MODELS)
        self._model_combo.setStyleSheet(_combo_css())

        self._epoch_spin = QSpinBox()
        self._epoch_spin.setRange(1, 1000)
        self._epoch_spin.setValue(50)
        self._epoch_spin.setStyleSheet(_spin_css())

        self._act_train = QAction("▶  Start Training", self)
        self._act_train.setToolTip("Train the selected YOLO model on exported annotations")
        self._act_train.setEnabled(True)   # always available — picks data.yaml folder at runtime
        self._act_train.triggered.connect(self._on_train)

        self._act_convert = QAction("⬡  Convert to Segmentation", self)
        self._act_convert.setToolTip(
            "Upgrade a YOLO detection dataset (bounding boxes) to instance segmentation\n"
            "polygon masks using SAM 3 — no re-labeling required."
        )
        self._act_convert.triggered.connect(self._on_convert_to_seg)

        # ── ONNX export ───────────────────────────────────────────
        self._onnx_prec_combo = QComboBox()
        self._onnx_prec_combo.addItems(["FP32", "FP16"])
        self._onnx_prec_combo.setStyleSheet(_combo_css())
        self._onnx_prec_combo.setToolTip(
            "FP32: full precision, runs on any device.\n"
            "FP16: half precision — smaller file, faster GPU inference (requires CUDA)."
        )

        self._onnx_shape_combo = QComboBox()
        self._onnx_shape_combo.addItems(["Dynamic", "Static"])
        self._onnx_shape_combo.setStyleSheet(_combo_css())
        self._onnx_shape_combo.setToolTip(
            "Dynamic: variable batch size / input resolution (more flexible).\n"
            "Static: fixed input shape — faster on some runtimes, required by a few."
        )

        self._act_export_onnx = QAction("⬇  Export to ONNX", self)
        self._act_export_onnx.setToolTip(
            "Convert a trained .pt checkpoint to ONNX for cross-platform deployment"
        )
        self._act_export_onnx.triggered.connect(self._on_export_onnx)

    # ──────────────────────────────────────────────────────────────
    # Header bar
    # ──────────────────────────────────────────────────────────────

    def _build_header(self) -> None:
        hdr = QWidget()
        hdr.setFixedHeight(52)
        hdr.setStyleSheet(
            f"background: {_HDR}; border-bottom: 1px solid {_BORDER};"
        )

        layout = QHBoxLayout(hdr)
        layout.setContentsMargins(20, 0, 20, 0)
        layout.setSpacing(0)

        # App title
        title = QLabel("ModelTrainer")
        title.setStyleSheet(
            "color: #ffffff; font-size: 15px; font-weight: 700; "
            "letter-spacing: 0.5px; background: transparent; border: none;"
        )
        layout.addWidget(title)
        layout.addSpacing(28)

        # Workflow step tabs (checkable, exclusive)
        tab_css = (
            "QPushButton {"
            f"  background: transparent; color: {_MUTED};"
            "  border: none; border-bottom: 3px solid transparent;"
            "  padding: 0 20px; font-size: 13px; font-weight: 500;"
            "  min-height: 52px;"
            "}"
            "QPushButton:hover {"
            f"  color: {_TEXT}; background: rgba(255,255,255,0.03);"
            "}"
            "QPushButton:checked {"
            f"  color: {_ACCENT}; border-bottom: 3px solid {_ACCENT};"
            "  font-weight: 700;"
            "}"
        )

        self._tab_group = QButtonGroup(self)
        self._tab_group.setExclusive(True)

        for label, idx in [
            ("📁  Media",    self._TAB_MEDIA),
            ("🏷  Annotate", self._TAB_ANNOTATE),
            ("🚀  Train",    self._TAB_TRAIN),
        ]:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setStyleSheet(tab_css)
            btn.setFocusPolicy(Qt.NoFocus)
            btn.clicked.connect(lambda _, i=idx: self._switch_tab(i))
            self._tab_group.addButton(btn, idx)
            layout.addWidget(btn)

        layout.addStretch(1)

        # Subtle workflow hint (very dim, decorative)
        hint = QLabel("Import  →  Label  →  Train")
        hint.setStyleSheet(
            "color: #1e1e30; font-size: 11px; background: transparent; border: none;"
        )
        layout.addWidget(hint)

        self._header = hdr

    # ──────────────────────────────────────────────────────────────
    # Side panel (stacked — one page per tab)
    # ──────────────────────────────────────────────────────────────

    def _build_panel(self) -> None:
        self._panel_stack = QStackedWidget()
        self._panel_stack.setFixedWidth(270)
        self._panel_stack.setStyleSheet(
            f"QStackedWidget {{ background: {_PANEL}; "
            f"border-right: 1px solid {_BORDER}; }}"
        )
        self._panel_stack.addWidget(self._build_media_page())
        self._panel_stack.addWidget(self._build_annotate_page())
        self._panel_stack.addWidget(self._build_train_page())

    # ── Panel pages ───────────────────────────────────────────────

    def _build_media_page(self) -> QWidget:
        page = QWidget()
        page.setStyleSheet(f"background: {_PANEL};")
        v = QVBoxLayout(page)
        v.setContentsMargins(16, 20, 16, 16)
        v.setSpacing(6)

        # ── Primary import ────────────────────────────────────────
        v.addWidget(_section_lbl("Import"))

        btn_replace = _tool_btn(self._act_import, _css_primary())
        btn_replace.setMinimumHeight(40)
        v.addWidget(btn_replace)

        v.addSpacing(2)
        btn_add = _tool_btn(self._act_add_more, _css_neutral())
        v.addWidget(btn_add)

        v.addSpacing(2)
        btn_ds = _tool_btn(self._act_import_dataset, _css_neutral())
        v.addWidget(btn_ds)

        # Dataset import hint
        hint_ds = QLabel(
            "Import Dataset loads an existing\n"
            "YOLO folder (images/ + labels/ +\n"
            "data.yaml) with annotations."
        )
        hint_ds.setStyleSheet(f"color: {_MUTED}; font-size: 10px; padding: 4px 0 0 0;")
        hint_ds.setWordWrap(True)
        v.addWidget(hint_ds)

        v.addSpacing(4)
        v.addWidget(self._frame_count_lbl)
        v.addWidget(_hr())

        # ── Video frame sampling ──────────────────────────────────
        v.addWidget(_section_lbl("Video Frame Sampling"))
        v.addLayout(_row_layout("Stride:", self._import_stride_spin))
        hint = QLabel(
            "Keep 1 in every N video frames.\n"
            "Higher = fewer frames imported.\n"
            "Images are always kept as-is."
        )
        hint.setStyleSheet(f"color: {_MUTED}; font-size: 10px; padding: 4px 0 0 0;")
        hint.setWordWrap(True)
        v.addWidget(hint)

        v.addStretch(1)
        return page

    def _build_annotate_page(self) -> QWidget:
        """Scrollable annotate panel with class / auto-label / prompt / propagate."""
        page = QWidget()
        page.setStyleSheet(f"background: {_PANEL};")

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet(
            f"QScrollArea {{ border: none; background: {_PANEL}; }}"
            f"QScrollBar:vertical {{ background: {_PANEL}; width: 6px; border: none; }}"
            f"QScrollBar::handle:vertical {{ background: {_BORDER}; border-radius: 3px; }}"
            f"QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}"
        )

        inner = QWidget()
        inner.setStyleSheet(f"background: {_PANEL};")
        v = QVBoxLayout(inner)
        v.setContentsMargins(16, 16, 16, 16)
        v.setSpacing(5)

        # ── Active class (always at top) ──────────────────────────
        v.addWidget(_section_lbl("Active Class"))
        v.addLayout(_row_layout("Class:", self._class_combo))
        v.addWidget(_tool_btn(self._act_classes, _css_neutral()))
        v.addWidget(_hr())

        # ── Auto-label ────────────────────────────────────────────
        v.addWidget(_section_lbl("Auto-Label"))
        v.addWidget(QLabel("Concept (what to find):") )
        v.itemAt(v.count() - 1).widget().setStyleSheet(
            f"color: {_MUTED}; font-size: 10px;"
        )
        v.addWidget(self._concept_edit)
        v.addSpacing(4)
        v.addWidget(_tool_btn(self._act_sam, _css_primary()))
        v.addSpacing(2)
        v.addLayout(_row_layout("Model:", self._sam_model_combo))
        v.addLayout(_row_layout("Stride:", self._stride_spin))
        v.addLayout(_row_layout("Start:", self._sam_start_spin))
        hint_s = QLabel(
            "Stride: every Nth frame.\n"
            "Start: skip frames below this index."
        )
        hint_s.setStyleSheet(f"color: {_MUTED}; font-size: 10px;")
        hint_s.setWordWrap(True)
        v.addWidget(hint_s)
        v.addWidget(_hr())

        # ── Interactive prompt ────────────────────────────────────
        v.addWidget(_section_lbl("Interactive Prompt"))
        hint_p = QLabel("Toggle on, then draw example\nboxes on the canvas.")
        hint_p.setStyleSheet(f"color: {_MUTED}; font-size: 10px;")
        hint_p.setWordWrap(True)
        v.addWidget(hint_p)
        v.addSpacing(4)
        v.addWidget(_tool_btn(self._act_sam_prompt, _css_toggle()))
        v.addWidget(_tool_btn(self._act_run_prompt, _css_primary()))
        v.addWidget(_tool_btn(self._act_clear_prompt, _css_neutral()))
        v.addWidget(_hr())

        # ── Propagate ─────────────────────────────────────────────
        v.addWidget(_section_lbl("Propagate from This Frame"))
        hint_q = QLabel("Label this frame first, then\nspread those labels everywhere.")
        hint_q.setStyleSheet(f"color: {_MUTED}; font-size: 10px;")
        hint_q.setWordWrap(True)
        v.addWidget(hint_q)
        v.addSpacing(4)
        v.addWidget(_tool_btn(self._act_propagate, _css_primary()))
        v.addWidget(_tool_btn(self._act_track, _css_primary()))
        v.addSpacing(4)
        v.addLayout(_row_layout("Start:", self._propagate_start_spin))
        v.addLayout(_row_layout("Range:", self._track_range_spin))

        v.addStretch(1)
        scroll.setWidget(inner)

        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(scroll)
        return page

    def _build_train_page(self) -> QWidget:
        page = QWidget()
        page.setStyleSheet(f"background: {_PANEL};")
        v = QVBoxLayout(page)
        v.setContentsMargins(16, 20, 16, 16)
        v.setSpacing(6)

        v.addWidget(_section_lbl("Export"))
        v.addWidget(_tool_btn(self._act_export, _css_primary(_GREEN, hover="#059669")))
        hint_e = QLabel("Exports YOLO .txt labels +\ndata.yaml to a folder you choose.")
        hint_e.setStyleSheet(f"color: {_MUTED}; font-size: 10px; padding: 4px 0 0 0;")
        hint_e.setWordWrap(True)
        v.addWidget(hint_e)
        v.addWidget(_hr())

        v.addWidget(_section_lbl("Convert to Segmentation"))
        v.addWidget(_tool_btn(self._act_convert, _css_primary("#7c3aed", hover="#6d28d9")))
        hint_c = QLabel(
            "Converts a detection dataset\n(bboxes) → polygon masks\nusing SAM 3."
        )
        hint_c.setStyleSheet(f"color: {_MUTED}; font-size: 10px; padding: 4px 0 0 0;")
        hint_c.setWordWrap(True)
        v.addWidget(hint_c)
        v.addWidget(_hr())

        v.addWidget(_section_lbl("Training"))
        v.addLayout(_row_layout("Task:", self._task_combo))
        v.addSpacing(4)
        v.addLayout(_row_layout("Model:", self._model_combo))
        v.addSpacing(4)
        v.addLayout(_row_layout("Epochs:", self._epoch_spin))
        v.addSpacing(8)
        v.addWidget(_tool_btn(self._act_train, _css_primary(_GREEN, hover="#059669")))
        hint_t = QLabel(
            "YOLO / FastSAM: point to your\nexported dataset folder.\n"
            "SAM 2: trains directly from\nyour annotations — no export."
        )
        hint_t.setStyleSheet(f"color: {_MUTED}; font-size: 10px; padding: 4px 0 0 0;")
        hint_t.setWordWrap(True)
        v.addWidget(hint_t)
        v.addWidget(_hr())

        v.addWidget(_section_lbl("Export to ONNX"))
        v.addLayout(_row_layout("Precision:", self._onnx_prec_combo))
        v.addSpacing(4)
        v.addLayout(_row_layout("Shape:", self._onnx_shape_combo))
        v.addSpacing(8)
        v.addWidget(_tool_btn(self._act_export_onnx, _css_primary(_ACCENT, hover="#2563eb")))
        hint_o = QLabel(
            "Converts a trained .pt checkpoint\nto ONNX. FP16 requires a CUDA GPU."
        )
        hint_o.setStyleSheet(f"color: {_MUTED}; font-size: 10px; padding: 4px 0 0 0;")
        hint_o.setWordWrap(True)
        v.addWidget(hint_o)

        v.addStretch(1)
        return page

    # ──────────────────────────────────────────────────────────────
    # Work area (thumbnail strip + canvas)
    # ──────────────────────────────────────────────────────────────

    def _build_work_area(self) -> None:
        self._thumb_list = QListWidget()
        self._thumb_list.setIconSize(QSize(_THUMBNAIL_SIZE, _THUMBNAIL_SIZE))
        self._thumb_list.setFixedWidth(_THUMBNAIL_SIZE + 48)
        self._thumb_list.setSpacing(4)
        self._thumb_list.setStyleSheet(
            f"QListWidget {{ background: {_BG}; border: none; "
            f"border-right: 1px solid {_BORDER}; }}"
            f"QListWidget::item {{ padding: 4px; border-radius: 4px; }}"
            f"QListWidget::item:selected {{ background: {_SURFACE}; }}"
        )
        self._thumb_list.currentRowChanged.connect(self._on_frame_selected)

        self._canvas = AnnotationCanvas()
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._canvas.setStyleSheet(f"background: {_BG};")
        self._canvas.box_added.connect(self._on_box_added)
        self._canvas.box_deleted.connect(self._on_box_deleted)
        self._canvas.box_edited.connect(self._on_box_edited)
        self._canvas.exemplars_changed.connect(self._on_exemplars_changed)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._thumb_list)
        splitter.addWidget(self._canvas)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStyleSheet(
            f"QSplitter::handle {{ background: {_BORDER}; width: 1px; }}"
        )
        self._work_area = splitter

    # ──────────────────────────────────────────────────────────────
    # Status bar
    # ──────────────────────────────────────────────────────────────

    def _build_statusbar(self) -> None:
        sb = QStatusBar(self)
        sb.setStyleSheet(
            f"QStatusBar {{ background: {_HDR}; border-top: 1px solid {_BORDER}; }}"
            f"QStatusBar QLabel {{ color: {_TEXT}; font-size: 11px; padding: 0 4px; }}"
        )
        self.setStatusBar(sb)

        self._status_label = QLabel("Ready — import a folder to begin")
        self._status_label.setMinimumWidth(300)
        sb.addWidget(self._status_label)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setFixedWidth(260)
        self._progress.setFixedHeight(6)
        self._progress.setTextVisible(False)
        self._progress.setStyleSheet(
            f"QProgressBar {{ background: {_SURFACE}; border: none; border-radius: 3px; }}"
            f"QProgressBar::chunk {{ background: {_ACCENT}; border-radius: 3px; }}"
        )
        sb.addPermanentWidget(self._progress)

    # ──────────────────────────────────────────────────────────────
    # Assembly
    # ──────────────────────────────────────────────────────────────

    def _assemble(self) -> None:
        body = QWidget()
        body.setStyleSheet(f"background: {_BG};")
        bl = QHBoxLayout(body)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(0)
        bl.addWidget(self._panel_stack)
        bl.addWidget(self._work_area, 1)

        root = QWidget()
        rl = QVBoxLayout(root)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)
        rl.addWidget(self._header)
        rl.addWidget(body, 1)

        self.setCentralWidget(root)

    # ──────────────────────────────────────────────────────────────
    # Tab switching
    # ──────────────────────────────────────────────────────────────

    def _switch_tab(self, index: int) -> None:
        self._panel_stack.setCurrentIndex(index)
        btn = self._tab_group.button(index)
        if btn:
            btn.setChecked(True)

        hints = {
            self._TAB_MEDIA:    "Step 1 — Import a folder of images or video files",
            self._TAB_ANNOTATE: "Step 2 — Label frames, then propagate to the rest",
            self._TAB_TRAIN:    "Step 3 — Export labels and train your YOLO model",
        }
        self._status_label.setText(hints.get(index, "Ready"))

    # ──────────────────────────────────────────────────────────────
    # Import
    # ──────────────────────────────────────────────────────────────

    def _abort_media_worker(self) -> None:
        if self._media_worker and self._media_worker.isRunning():
            self._media_worker.abort()
            self._media_worker.wait()

    @Slot()
    def _on_import(self) -> None:
        """Fresh import — clears all existing frames and annotations."""
        path = QFileDialog.getExistingDirectory(
            self, "Select Media Directory", _APP_ROOT
        )
        if not path:
            return

        self._abort_media_worker()

        # Clear existing session
        self.store.clear()
        self._thumb_list.clear()
        self.current_frame_index = None
        self._canvas.load_frame("", [])
        self._frame_count = 0
        self._video_import_stride.clear()
        self._pending_labels_dir = None

        self._start_media_import([path], frame_offset=0)

    @Slot()
    def _on_add_more(self) -> None:
        """Append more images/videos without clearing the existing session."""
        path = QFileDialog.getExistingDirectory(
            self, "Select Media Directory to Append", _APP_ROOT
        )
        if not path:
            return

        self._abort_media_worker()
        offset = max(self.store.keys(), default=-1) + 1
        self._pending_labels_dir = None
        self._start_media_import([path], frame_offset=offset)

    @Slot()
    def _on_import_dataset(self) -> None:
        """Import a YOLO-format dataset folder (images/ + labels/ + data.yaml)."""
        path = QFileDialog.getExistingDirectory(
            self, "Select YOLO Dataset Folder", _APP_ROOT
        )
        if not path:
            return

        self._abort_media_worker()

        # ── Load class names from data.yaml ───────────────────────
        # Search the picked folder and its parent for data.yaml
        yaml_path = None
        for candidate in (path, os.path.dirname(path)):
            p = os.path.join(candidate, "data.yaml")
            if os.path.isfile(p):
                yaml_path = p
                break

        if yaml_path:
            try:
                import yaml  # PyYAML
                with open(yaml_path, encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                names = data.get("names", [])

                # YOLO data.yaml uses either a list or an int-keyed dict
                if isinstance(names, dict):
                    # {0: 'car', 1: 'bus', ...} — sort by key
                    names = [names[k] for k in sorted(names.keys())]

                if isinstance(names, list) and names:
                    self.class_names = [str(n) for n in names]
                    self._class_combo.blockSignals(True)
                    self._class_combo.clear()
                    self._class_combo.addItems(self.class_names)
                    self._class_combo.blockSignals(False)
                    self._canvas.set_class_id(0)
                    self._status_label.setText(
                        f"Loaded {len(self.class_names)} class(es) from data.yaml: "
                        + ", ".join(self.class_names[:6])
                        + ("…" if len(self.class_names) > 6 else "")
                    )
                else:
                    self._status_label.setText(
                        "data.yaml found but no class names detected — check 'names:' field"
                    )
            except Exception as exc:
                self._status_label.setText(f"Warning: could not read data.yaml — {exc}")
        else:
            self._status_label.setText(
                "No data.yaml found — class names unchanged. "
                "Place data.yaml in the dataset folder or its parent."
            )

        # ── Resolve images dir ────────────────────────────────────
        images_dir = os.path.join(path, "images")
        if not os.path.isdir(images_dir):
            # Fall back: look for train/ sub-folder inside images/
            for candidate in ("train", "val", ""):
                d = os.path.join(path, "images", candidate) if candidate else path
                if os.path.isdir(d):
                    images_dir = d
                    break

        # ── Resolve labels dir ────────────────────────────────────
        labels_dir = os.path.join(path, "labels")
        if not os.path.isdir(labels_dir):
            labels_dir = None  # no labels — import images only

        self._pending_labels_dir = labels_dir

        # Clear and import fresh (dataset replaces current session)
        self.store.clear()
        self._thumb_list.clear()
        self.current_frame_index = None
        self._canvas.load_frame("", [])
        self._frame_count = 0
        self._video_import_stride.clear()

        self._start_media_import([images_dir], frame_offset=0)

    def _start_media_import(self, paths: list[str], frame_offset: int) -> None:
        """Shared helper — creates and starts a MediaLoaderWorker."""
        stride = self._import_stride_spin.value()
        self._pending_import_stride = stride

        self._status_label.setText("Importing media…")
        self._progress.setValue(0)
        self._act_sam.setEnabled(False)
        self._frame_count_lbl.setText("Importing…")

        self._media_worker = MediaLoaderWorker(
            paths, self._temp_dir,
            import_stride=stride,
            frame_offset=frame_offset,
        )
        self._media_worker.frame_ready.connect(self._on_frame_ready)
        self._media_worker.progress.connect(self._on_media_progress)
        self._media_worker.finished.connect(self._on_media_finished)
        self._media_worker.error.connect(self._on_worker_error)
        self._media_worker.start()

    def _current_concept(self) -> str:
        text = self._concept_edit.text().strip()
        if text:
            return text
        idx = self._class_combo.currentIndex()
        if 0 <= idx < len(self.class_names):
            return self.class_names[idx]
        return self.class_names[0] if self.class_names else "object"

    def _ensure_sam_loaded(self, then) -> None:
        if self._sam_model_obj is not None:
            then()
            return

        self._sam_pending_action = then
        if self._sam_load_worker and self._sam_load_worker.isRunning():
            return

        self._status_label.setText("Loading SAM 3 model…")
        QApplication.processEvents()
        try:
            import torch  # noqa: F401
            from transformers import Sam3Model, Sam3Processor  # noqa: F401
        except Exception as exc:
            self._sam_pending_action = None
            QMessageBox.critical(
                self, "PyTorch / Transformers Error",
                f"{exc}\n\n"
                "SAM 3 requires transformers v5+:\n"
                "  pip install -U transformers\n\n"
                "If this is a CUDA DLL error, install the CPU-only torch build:\n"
                "  pip install torch torchvision --index-url "
                "https://download.pytorch.org/whl/cpu"
            )
            return

        model_key = self._sam_model_combo.currentText()
        self._sam_load_worker = SAM3LoadWorker(model_key)
        self._sam_load_worker.loaded.connect(self._on_sam_model_loaded)
        self._sam_load_worker.error.connect(self._on_sam_load_error)
        self._sam_load_worker.start()

    @Slot(object, object)
    def _on_sam_model_loaded(self, model, processor) -> None:
        self._sam_model_obj = model
        self._sam_processor_obj = processor
        self._status_label.setText("SAM 3 model ready")
        action = self._sam_pending_action
        self._sam_pending_action = None
        if action:
            action()

    @Slot(str)
    def _on_sam_load_error(self, msg: str) -> None:
        self._sam_pending_action = None
        self._on_worker_error(msg)

    # ──────────────────────────────────────────────────────────────
    # Auto-label (SAM 3 text)
    # ──────────────────────────────────────────────────────────────

    @Slot()
    def _on_run_sam(self) -> None:
        if self._sam_text_worker and self._sam_text_worker.isRunning():
            return

        concept = self._current_concept()
        stride = self._stride_spin.value()
        start = self._sam_start_spin.value()
        frames = [
            (idx, ann.image_path)
            for idx, ann in sorted(self.store.items())
            if idx >= start and idx % stride == 0
        ]
        if not frames:
            QMessageBox.information(self, "SAM 3", "No frames to process.")
            return

        self._ensure_sam_loaded(lambda: self._start_sam_text(frames, concept))

    def _start_sam_text(self, frames: list, concept: str) -> None:
        class_id = self._class_combo.currentIndex()
        self._status_label.setText(
            f"SAM 3: searching '{concept}' across {len(frames)} frames…"
        )
        self._progress.setRange(0, len(frames))
        self._progress.setValue(0)
        self._act_sam.setEnabled(False)

        self._sam_text_worker = SAM3TextWorker(
            self._sam_model_obj, self._sam_processor_obj,
            frames, [(concept, class_id)],
        )
        self._sam_text_worker.boxes_ready.connect(self._on_boxes_ready)
        self._sam_text_worker.progress.connect(self._on_sam_progress)
        self._sam_text_worker.finished.connect(self._on_sam_finished)
        self._sam_text_worker.error.connect(self._on_worker_error)
        self._sam_text_worker.start()

    # ──────────────────────────────────────────────────────────────
    # Classes
    # ──────────────────────────────────────────────────────────────

    @Slot()
    def _on_edit_classes(self) -> None:
        current = ", ".join(self.class_names)
        text, ok = QInputDialog.getText(
            self,
            "Edit Class Names",
            "Enter class names separated by commas:",
            text=current,
        )
        if ok and text.strip():
            self.class_names = [n.strip() for n in text.split(",") if n.strip()]
            self._class_combo.blockSignals(True)
            self._class_combo.clear()
            self._class_combo.addItems(self.class_names)
            self._class_combo.blockSignals(False)
            self._canvas.set_class_id(0)

    @Slot(int)
    def _on_class_changed(self, index: int) -> None:
        self._canvas.set_class_id(index)

    # ──────────────────────────────────────────────────────────────
    # Export & train
    # ──────────────────────────────────────────────────────────────

    @Slot()
    def _on_export(self) -> None:
        out_dir = QFileDialog.getExistingDirectory(
            self, "Select Export Directory", _APP_ROOT
        )
        if not out_dir:
            return
        is_seg = self._task_combo.currentIndex() != 0  # index 0 = Detection
        count = export_dataset(self.store, self.class_names, out_dir, is_seg=is_seg)
        task_label = "segmentation" if is_seg else "detection"
        self._status_label.setText(f"Exported {count} frames ({task_label}) to {out_dir}")
        self._update_thumbnail_colors()
        self._act_train.setEnabled(count > 0)

    @Slot(int)
    def _on_task_changed(self, index: int) -> None:
        self._model_combo.clear()
        if index == 0:
            self._model_combo.addItems(DETECTION_MODELS)
        else:
            # YOLO-based seg models first, then SAM 2 variants
            self._model_combo.addItems(SEGMENTATION_MODELS)
            self._model_combo.addItems(list(SAM2_MODELS.keys()))

    @Slot()
    def _on_train(self) -> None:
        model_key = self._model_combo.currentText()

        # ── SAM 2 fine-tuning (trains directly from annotation store) ──
        if _is_sam2_key(model_key):
            self._on_train_sam2(model_key)
            return

        # ── YOLO / FastSAM training (requires exported data.yaml) ──
        out_dir = QFileDialog.getExistingDirectory(
            self, "Select Dataset Directory (containing data.yaml)", _APP_ROOT
        )
        if not out_dir:
            return

        yaml_path = os.path.join(out_dir, "data.yaml")
        if not os.path.isfile(yaml_path):
            QMessageBox.warning(
                self, "Training",
                f"No data.yaml found in:\n{out_dir}\n\nExport annotations first."
            )
            return

        if self._train_worker and self._train_worker.isRunning():
            return

        try:
            import torch  # noqa: F401
        except Exception as exc:
            QMessageBox.critical(self, "PyTorch Error", str(exc))
            return

        from app.core.yolo_trainer import _is_seg_key
        is_seg_model = _is_seg_key(model_key)

        # Dataset compatibility check
        try:
            import yaml as _yaml
            with open(yaml_path, encoding="utf-8") as _f:
                _yd = _yaml.safe_load(_f)
            dataset_task = (_yd or {}).get("task", "detect")
        except Exception:
            dataset_task = "detect"

        is_seg_dataset = dataset_task == "segment"

        if is_seg_model and not is_seg_dataset:
            reply = QMessageBox.warning(
                self, "Dataset Mismatch",
                f"'{model_key}' is a segmentation model, but the selected dataset "
                f"does not have 'task: segment' in its data.yaml.\n\n"
                f"Segmentation training requires polygon-mask labels. "
                f"Use 'Convert to Segmentation' to upgrade a detection dataset first.\n\n"
                f"Proceed anyway?",
                QMessageBox.Ok | QMessageBox.Cancel,
            )
            if reply == QMessageBox.Cancel:
                return

        epochs = self._epoch_spin.value()
        metric_label = "mAP50-mask" if is_seg_model else "mAP50"

        self._status_label.setText(f"Training {model_key} for {epochs} epochs…")
        self._progress.setRange(0, epochs)
        self._progress.setValue(0)
        self._act_train.setEnabled(False)
        self._train_metric_label = metric_label

        self._train_worker = YOLOTrainWorker(
            model_key=model_key,
            data_yaml=yaml_path,
            epochs=epochs,
        )
        self._train_worker.epoch_done.connect(self._on_epoch_done)
        self._train_worker.finished.connect(self._on_train_finished)
        self._train_worker.error.connect(self._on_worker_error)
        self._train_worker.start()

    def _on_train_sam2(self, model_key: str) -> None:
        """Start SAM 2 fine-tuning from an exported dataset folder."""
        if self._sam2_train_worker and self._sam2_train_worker.isRunning():
            return

        try:
            import torch  # noqa: F401
        except Exception as exc:
            QMessageBox.critical(self, "PyTorch Error", str(exc))
            return

        # Pick dataset directory (same flow as YOLO training)
        dataset_dir = QFileDialog.getExistingDirectory(
            self, "Select Dataset Directory (contains images/ and labels/)", _APP_ROOT
        )
        if not dataset_dir:
            return

        # Pick output directory for the saved checkpoint
        output_dir = QFileDialog.getExistingDirectory(
            self, "Select Output Directory for SAM 2 Checkpoint", _APP_ROOT
        )
        if not output_dir:
            return

        epochs = self._epoch_spin.value()

        self._status_label.setText(
            f"Fine-tuning {model_key} for {epochs} epochs…"
        )
        self._progress.setRange(0, epochs)
        self._progress.setValue(0)
        self._act_train.setEnabled(False)
        self._train_metric_label = "loss"

        self._sam2_train_worker = SAM2TrainWorker(
            model_key=model_key,
            dataset_dir=dataset_dir,
            epochs=epochs,
            output_dir=output_dir,
        )
        self._sam2_train_worker.epoch_done.connect(self._on_epoch_done)
        self._sam2_train_worker.finished.connect(self._on_sam2_train_finished)
        self._sam2_train_worker.error.connect(self._on_sam2_train_error)
        self._sam2_train_worker.start()

    # ──────────────────────────────────────────────────────────────
    # Segmentation converter
    # ──────────────────────────────────────────────────────────────

    @Slot()
    def _on_convert_to_seg(self) -> None:
        source_dir = QFileDialog.getExistingDirectory(
            self, "Select Detection Dataset (contains data.yaml)",
            _APP_ROOT
        )
        if not source_dir:
            return

        # Output is a sibling folder named <source_folder>seg
        source_path = os.path.normpath(source_dir)
        parent_dir = os.path.dirname(source_path)
        folder_name = os.path.basename(source_path)
        output_dir = os.path.join(parent_dir, folder_name + "seg")

        if self._seg_converter_worker and self._seg_converter_worker.isRunning():
            return

        from app.core.yolo_seg_converter import YoloSegConverterWorker

        self._status_label.setText("Starting segmentation conversion…")
        self._progress.setRange(0, 0)

        # Reuse already-loaded SAM 3 model if available — avoids a second load
        preloaded_model     = self._sam_model_obj
        preloaded_processor = self._sam_processor_obj

        self._seg_converter_worker = YoloSegConverterWorker(
            source_root=source_dir,
            output_root=output_dir,
            model=preloaded_model,
            processor=preloaded_processor,
        )
        self._seg_converter_worker.progress.connect(self._on_seg_convert_progress)
        self._seg_converter_worker.status_update.connect(self._status_label.setText)
        self._seg_converter_worker.finished.connect(self._on_seg_convert_finished)
        self._seg_converter_worker.error.connect(self._on_worker_error)
        self._seg_converter_worker.start()

    @Slot(int, int)
    def _on_seg_convert_progress(self, current: int, total: int) -> None:
        if total > 0:
            self._progress.setRange(0, total)
            self._progress.setValue(current)
            pct = int(current / total * 100)
            self._status_label.setText(
                f"Converting: {pct}%  ({current}/{total} images)"
            )

    @Slot(dict)
    def _on_seg_convert_finished(self, result: dict) -> None:
        self._progress.setRange(0, 100)
        self._progress.setValue(100)
        summary = (
            f"Conversion complete — {result['converted_items']} SAM 3 masks, "
            f"{result['fallback_items']} bbox fallback, "
            f"{result['failed_items']} failed  "
            f"({result['time_taken_sec']:.1f}s)"
        )
        self._status_label.setText(summary)
        QMessageBox.information(
            self, "Conversion Complete",
            f"{summary}\n\nOutput: {result['output_dir']}"
        )

    # ──────────────────────────────────────────────────────────────
    # Worker slots — media loader
    # ──────────────────────────────────────────────────────────────

    @Slot(int, str, str)
    def _on_frame_ready(self, frame_index: int, png_path: str, source_path: str) -> None:
        from app.core.media_loader import VIDEO_EXTS
        ext = os.path.splitext(source_path)[1].lower()
        is_video = ext in VIDEO_EXTS

        # source_video only set for actual video files — image source stored separately
        self.store[frame_index] = FrameAnnotation(
            frame_index=frame_index,
            image_path=png_path,
            source_video=source_path if is_video else "",
        )
        # For dataset label matching we need the original image path per frame
        if not is_video and source_path:
            self.store[frame_index].__dict__["_orig_image_path"] = source_path

        # Record the import stride used for this source video (for tracking warnings)
        if is_video and source_path not in self._video_import_stride:
            self._video_import_stride[source_path] = getattr(
                self, "_pending_import_stride", 1
            )
        item = QListWidgetItem()
        item.setIcon(QIcon(_make_thumbnail(png_path)))
        item.setText(str(frame_index))
        item.setData(Qt.UserRole, frame_index)
        self._thumb_list.addItem(item)

        self._frame_count += 1
        self._frame_count_lbl.setText(f"{self._frame_count} frame(s) loaded")

        if self._thumb_list.count() == 1:
            self._thumb_list.setCurrentRow(0)

    @Slot(int, int)
    def _on_media_progress(self, current: int, total: int) -> None:
        if total > 0:
            self._progress.setRange(0, total)
            self._progress.setValue(current)

    @Slot()
    def _on_media_finished(self) -> None:
        count = len(self.store)
        self._progress.setValue(self._progress.maximum())
        self._frame_count_lbl.setText(f"{count} frame(s) loaded ✓")
        self._act_sam.setEnabled(count > 0)
        self._act_sam_prompt.setEnabled(count > 0)
        self._act_export.setEnabled(count > 0)

        # If importing a YOLO dataset, apply existing label annotations now
        if self._pending_labels_dir and os.path.isdir(self._pending_labels_dir):
            labeled = self._apply_dataset_labels(self._pending_labels_dir)
            self._pending_labels_dir = None
            # Reload canvas so boxes appear without requiring a click
            if self.current_frame_index is not None:
                self._reload_canvas(self.current_frame_index)
            self._status_label.setText(
                f"Imported {count} frame(s), loaded labels for {labeled} ✓"
            )
        else:
            self._pending_labels_dir = None
            self._status_label.setText(
                f"Imported {count} frame(s) — switch to Annotate to label them"
            )

        # Auto-advance to the Annotate tab
        self._switch_tab(self._TAB_ANNOTATE)

    def _apply_dataset_labels(self, labels_dir: str) -> int:
        """
        Parse YOLO .txt label files and populate BBoxes in the store.
        Returns the number of frames that had labels applied.
        """
        import cv2 as _cv2
        labeled = 0
        for frame_index, ann in self.store.items():
            # Recover original image path stored in _on_frame_ready
            orig_path = ann.__dict__.get("_orig_image_path", "")
            if not orig_path:
                continue
            stem = os.path.splitext(os.path.basename(orig_path))[0]
            label_path = os.path.join(labels_dir, stem + ".txt")
            if not os.path.isfile(label_path):
                continue

            img = _cv2.imread(ann.image_path)
            if img is None:
                continue
            h, w = img.shape[:2]

            boxes: list[BBox] = []
            with open(label_path, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    try:
                        cid = int(parts[0])
                        floats = [float(p) for p in parts[1:]]
                    except ValueError:
                        continue

                    # Segmentation format: class_id + even number of xy pairs (≥3 pts)
                    if len(floats) >= 6 and len(floats) % 2 == 0:
                        xs = floats[0::2]
                        ys = floats[1::2]
                        x1 = min(xs) * w
                        y1 = min(ys) * h
                        x2 = max(xs) * w
                        y2 = max(ys) * h
                        boxes.append(BBox(x1=x1, y1=y1, x2=x2, y2=y2,
                                          class_id=cid, source="dataset",
                                          polygon=floats))
                    elif len(floats) == 4:
                        # Detection format: cx cy w h
                        cx, cy, bw, bh = floats
                        x1 = (cx - bw / 2) * w
                        y1 = (cy - bh / 2) * h
                        x2 = (cx + bw / 2) * w
                        y2 = (cy + bh / 2) * h
                        boxes.append(BBox(x1=x1, y1=y1, x2=x2, y2=y2,
                                          class_id=cid, source="dataset"))
            if boxes:
                ann.boxes = boxes
                ann.status = "verified"
                labeled += 1

        self._update_thumbnail_colors()
        return labeled

    # ──────────────────────────────────────────────────────────────
    # Worker slots — SAM 3 text / bulk
    # ──────────────────────────────────────────────────────────────

    @Slot(int, list)
    def _on_boxes_ready(self, frame_index: int, boxes: list) -> None:
        ann = self.store.get(frame_index)
        if ann is None:
            return
        ann.boxes = boxes
        ann.status = "pending"
        self._update_thumbnail_color(frame_index)

        # Update the canvas only if this frame is already selected — don't
        # auto-jump the selection or scroll the thumbnail list while SAM runs.
        if frame_index == self.current_frame_index:
            self._reload_canvas(frame_index)
        self._refresh_propagation_actions()

    @Slot(int, int)
    def _on_sam_progress(self, current: int, total: int) -> None:
        if total > 0:
            self._progress.setRange(0, total)
            self._progress.setValue(current)
            pct = int(current / total * 100)
            self._status_label.setText(f"SAM 3: {pct}%  ({current}/{total} frames)")

    @Slot()
    def _on_sam_finished(self) -> None:
        self._status_label.setText("SAM 3 labeling complete ✓")
        self._progress.setValue(self._progress.maximum())
        self._act_sam.setEnabled(True)
        self._act_export.setEnabled(True)
        self._refresh_propagation_actions()  # re-enables Propagate + Track based on current frame

    # ──────────────────────────────────────────────────────────────
    # SAM 3 interactive prompt
    # ──────────────────────────────────────────────────────────────

    @Slot(bool)
    def _on_sam_prompt_toggled(self, checked: bool) -> None:
        self._canvas.set_sam_mode(checked)
        self._act_run_prompt.setEnabled(checked)
        self._act_clear_prompt.setEnabled(checked)
        if checked:
            self._status_label.setText(
                "Prompt ON — left-drag = positive  ·  right-drag = negative  ·  then Run Prompt"
            )
        else:
            self._status_label.setText("Ready")

    @Slot()
    def _on_exemplars_changed(self) -> None:
        if not self._act_sam_prompt.isChecked():
            return
        pos, neg = self._canvas.get_exemplars()
        self._status_label.setText(
            f"Prompt: {len(pos)} positive, {len(neg)} negative — press Run Prompt"
        )

    @Slot()
    def _on_clear_prompts(self) -> None:
        self._canvas.clear_exemplars()

    @Slot()
    def _on_run_sam_prompt(self) -> None:
        if self._sam_prompt_worker and self._sam_prompt_worker.isRunning():
            return
        if self.current_frame_index is None:
            return
        ann = self.store.get(self.current_frame_index)
        if ann is None or not os.path.isfile(ann.image_path):
            return

        pos, neg = self._canvas.get_exemplars()
        text = self._concept_edit.text().strip() or None
        if not pos and not text:
            QMessageBox.information(
                self, "SAM 3 Prompt",
                "Draw at least one positive example box (left-drag),\n"
                "or type a Concept in the Auto-Label section.\n\n"
                "Green box = positive  |  Red box = negative"
            )
            return

        self._ensure_sam_loaded(
            lambda: self._start_sam_prompt(
                self.current_frame_index, ann.image_path, text, pos, neg
            )
        )

    def _start_sam_prompt(self, frame_index: int, image_path: str,
                          text, pos, neg) -> None:
        from PIL import Image as PilImage
        pil_img = PilImage.open(image_path).convert("RGB")
        class_id = self._class_combo.currentIndex()
        self._status_label.setText("SAM 3: segmenting from examples…")

        self._sam_prompt_worker = SAM3PromptWorker(
            self._sam_model_obj, self._sam_processor_obj,
            frame_index, pil_img, text, pos, neg, class_id=class_id,
        )
        self._sam_prompt_worker.boxes_ready.connect(self._on_sam_prompt_boxes_ready)
        self._sam_prompt_worker.error.connect(self._on_worker_error)
        self._sam_prompt_worker.start()

    @Slot(int, list)
    def _on_sam_prompt_boxes_ready(self, frame_index: int, boxes: list) -> None:
        ann = self.store.get(frame_index)
        if ann is None:
            return
        if not boxes:
            self._status_label.setText("SAM 3 Prompt: no matching objects found")
            return
        ann.boxes.extend(boxes)
        ann.status = "verified"
        self._update_thumbnail_color(frame_index)
        if frame_index == self.current_frame_index:
            self._reload_canvas(frame_index)
        self._refresh_propagation_actions()
        self._status_label.setText(
            f"SAM 3 Prompt: added {len(boxes)} object(s) — refine or toggle Prompt off"
        )

    # ──────────────────────────────────────────────────────────────
    # Propagate from first frame
    # ──────────────────────────────────────────────────────────────

    def _refresh_propagation_actions(self) -> None:
        ann = (self.store.get(self.current_frame_index)
               if self.current_frame_index is not None else None)
        has_labels = bool(ann and ann.boxes)
        self._act_propagate.setEnabled(has_labels)
        self._act_track.setEnabled(has_labels)

    def _seed_boxes(self) -> list:
        ann = (self.store.get(self.current_frame_index)
               if self.current_frame_index is not None else None)
        if not ann:
            return []
        return [((b.x1, b.y1, b.x2, b.y2), b.class_id) for b in ann.boxes]

    @Slot()
    def _on_propagate_labels(self) -> None:
        if self._sam_text_worker and self._sam_text_worker.isRunning():
            return
        if self.current_frame_index is None:
            return
        ann = self.store.get(self.current_frame_index)
        if not ann or not ann.boxes:
            QMessageBox.information(self, "Propagate Labels",
                                    "Label the current frame first.")
            return

        class_ids = sorted({b.class_id for b in ann.boxes})
        concepts: list[tuple[str, int]] = []
        for cid in class_ids:
            if 0 <= cid < len(self.class_names) and self.class_names[cid].strip():
                concepts.append((self.class_names[cid].strip(), cid))
        if not concepts:
            QMessageBox.information(
                self, "Propagate Labels",
                "Your labels need class names.\n"
                "Use 'Edit Classes…' to name them, then assign boxes a class."
            )
            return

        prop_start = self._propagate_start_spin.value()
        frames = [
            (idx, a.image_path)
            for idx, a in sorted(self.store.items())
            if idx != self.current_frame_index and idx >= prop_start
        ]
        if not frames:
            QMessageBox.information(self, "Propagate Labels",
                                    "No frames to propagate to with the current Start setting.")
            return
        self._ensure_sam_loaded(lambda: self._start_propagate(frames, concepts))

    def _start_propagate(self, frames: list, concepts: list) -> None:
        names = ", ".join(c for c, _ in concepts)
        self._status_label.setText(
            f"Propagating [{names}] to {len(frames)} frames…"
        )
        self._progress.setRange(0, len(frames))
        self._progress.setValue(0)
        self._act_sam.setEnabled(False)
        self._act_propagate.setEnabled(False)

        self._sam_text_worker = SAM3TextWorker(
            self._sam_model_obj, self._sam_processor_obj, frames, concepts,
        )
        self._sam_text_worker.boxes_ready.connect(self._on_boxes_ready)
        self._sam_text_worker.progress.connect(self._on_sam_progress)
        self._sam_text_worker.finished.connect(self._on_sam_finished)
        self._sam_text_worker.error.connect(self._on_worker_error)
        self._sam_text_worker.start()

    @Slot()
    def _on_track_video(self) -> None:
        if self._sam_track_worker and self._sam_track_worker.isRunning():
            return
        seeds = self._seed_boxes()
        if not seeds:
            QMessageBox.information(self, "Track Video",
                                    "Label the current frame first.")
            return

        # Only track within the same source video — crossing file boundaries
        # would feed completely different scenes to the memory tracker.
        seed_ann = self.store.get(self.current_frame_index)
        seed_source = seed_ann.source_video if seed_ann else ""
        frames = [
            (idx, a.image_path)
            for idx, a in sorted(self.store.items())
            if a.source_video == seed_source
        ]
        self._track_start_index = self.current_frame_index

        # Warn if this video was imported with stride > 1 — the tracker needs
        # temporally consecutive frames; large gaps will cause objects to jump
        # too far between frames for the memory tracker to follow them.
        used_stride = self._video_import_stride.get(seed_source, 1)
        if used_stride > 1:
            msg = (
                f"This video was imported with stride {used_stride} "
                f"(1 in every {used_stride} frames kept).\n\n"
                f"The SAM 3 memory tracker expects consecutive frames — "
                f"with stride {used_stride}, objects may jump too far between frames "
                f"for tracking to work reliably.\n\n"
                f"For best results:\n"
                f"  • Re-import this video with stride = 1\n"
                f"  • Or use Propagate Labels instead (works per-frame independently)\n\n"
                f"Continue tracking anyway?"
            )
            reply = QMessageBox.warning(
                self, "Frame Gaps May Break Tracking", msg,
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        self._status_label.setText("Loading SAM 3 video tracker…")
        QApplication.processEvents()
        try:
            import torch  # noqa: F401
            from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor  # noqa: F401
        except Exception as exc:
            QMessageBox.critical(
                self, "Transformers Error",
                f"{exc}\n\nSAM 3 video tracking requires transformers v5+."
            )
            return

        self._progress.setRange(0, max(1, len(frames)))
        self._progress.setValue(0)
        self._act_track.setEnabled(False)

        n_videos = len({a.source_video for a in self.store.values() if a.source_video})
        if n_videos > 1:
            self._status_label.setText(
                f"Tracking within '{os.path.basename(seed_source)}' "
                f"({len(frames)} frames)…"
            )
        else:
            self._status_label.setText(f"Tracking {len(frames)} frames…")

        max_frames = self._track_range_spin.value() or None
        self._sam_track_worker = SAM3TrackWorker(
            frames, seeds, self._track_start_index,
            max_frames=max_frames,
            model=self._sam_tracker_model,
            processor=self._sam_tracker_processor,
        )
        self._sam_track_worker.model_ready.connect(self._on_track_model_ready)
        self._sam_track_worker.boxes_ready.connect(self._on_track_boxes_ready)
        self._sam_track_worker.progress.connect(self._on_sam_progress)
        self._sam_track_worker.finished.connect(self._on_track_finished)
        self._sam_track_worker.error.connect(self._on_worker_error)
        self._sam_track_worker.start()

    @Slot(object, object)
    def _on_track_model_ready(self, model, processor) -> None:
        self._sam_tracker_model = model
        self._sam_tracker_processor = processor

    @Slot(int, list)
    def _on_track_boxes_ready(self, frame_index: int, boxes: list) -> None:
        if frame_index == getattr(self, "_track_start_index", None):
            return
        ann = self.store.get(frame_index)
        if ann is None:
            return
        ann.boxes = boxes
        ann.status = "pending"
        self._update_thumbnail_color(frame_index)
        if frame_index == self.current_frame_index:
            self._reload_canvas(frame_index)

    @Slot()
    def _on_track_finished(self) -> None:
        self._status_label.setText("SAM 3 tracking complete ✓")
        self._progress.setValue(self._progress.maximum())
        self._act_track.setEnabled(True)
        self._act_export.setEnabled(True)

    # ──────────────────────────────────────────────────────────────
    # Worker slots — training
    # ──────────────────────────────────────────────────────────────

    @Slot(int, int, float)
    def _on_epoch_done(self, epoch: int, total: int, map50: float) -> None:
        self._progress.setValue(epoch)
        metric = getattr(self, "_train_metric_label", "mAP50")
        self._status_label.setText(f"Epoch {epoch}/{total}  —  {metric}: {map50:.4f}")

    @Slot(str)
    def _on_train_finished(self, best_weights: str) -> None:
        self._status_label.setText(
            f"Training complete ✓  Best: {best_weights or 'runs/train/'}"
        )
        self._act_train.setEnabled(True)
        QMessageBox.information(
            self, "Training Complete",
            f"Training finished.\n\nBest weights:\n"
            f"{best_weights or 'runs/train/exp/weights/best.pt'}"
        )

    @Slot(str)
    def _on_sam2_train_finished(self, save_dir: str) -> None:
        self._status_label.setText(f"SAM 2 fine-tuning complete ✓  Saved: {save_dir}")
        self._progress.setValue(self._progress.maximum())
        self._act_train.setEnabled(True)
        QMessageBox.information(
            self, "SAM 2 Fine-tuning Complete",
            f"Fine-tuning finished.\n\n"
            f"Model checkpoint saved to:\n{save_dir}\n\n"
            f"Load it with:\n"
            f"  Sam2Model.from_pretrained('{save_dir}')\n"
            f"  Sam2Processor.from_pretrained('{save_dir}')"
        )

    @Slot(str)
    def _on_sam2_train_error(self, msg: str) -> None:
        self._status_label.setText(f"SAM 2 training error: {msg[:80]}")
        self._act_train.setEnabled(True)
        QMessageBox.warning(self, "SAM 2 Training Error", msg)

    # ──────────────────────────────────────────────────────────────
    # ONNX export
    # ──────────────────────────────────────────────────────────────

    @Slot()
    def _on_export_onnx(self) -> None:
        """Convert a trained .pt checkpoint to ONNX at the selected precision."""
        if self._onnx_worker and self._onnx_worker.isRunning():
            return

        pt_path, _ = QFileDialog.getOpenFileName(
            self, "Select .pt checkpoint to export", _APP_ROOT,
            "PyTorch weights (*.pt)"
        )
        if not pt_path:
            return

        try:
            import torch
        except Exception as exc:
            QMessageBox.critical(self, "PyTorch Error", str(exc))
            return

        half = self._onnx_prec_combo.currentText().upper().startswith("FP16")
        if half and not torch.cuda.is_available():
            QMessageBox.warning(
                self, "ONNX Export",
                "FP16 export requires a CUDA GPU, which isn't available.\n\n"
                "Select FP32 precision instead."
            )
            return

        dynamic = self._onnx_shape_combo.currentText() == "Dynamic"

        precision = "FP16" if half else "FP32"
        shape = "dynamic" if dynamic else "static"
        self._status_label.setText(
            f"Exporting {os.path.basename(pt_path)} → ONNX ({precision}, {shape})…"
        )
        self._progress.setRange(0, 0)  # indeterminate / busy
        self._act_export_onnx.setEnabled(False)

        self._onnx_worker = ONNXExportWorker(
            pt_path=pt_path, half=half, dynamic=dynamic
        )
        self._onnx_worker.finished.connect(self._on_onnx_finished)
        self._onnx_worker.error.connect(self._on_onnx_error)
        self._onnx_worker.start()

    @Slot(str)
    def _on_onnx_finished(self, onnx_path: str) -> None:
        self._progress.setRange(0, 1)
        self._progress.setValue(1)
        self._act_export_onnx.setEnabled(True)
        self._status_label.setText(f"ONNX export complete ✓  {onnx_path}")
        QMessageBox.information(
            self, "ONNX Export Complete",
            f"Model exported to:\n{onnx_path}"
        )

    @Slot(str)
    def _on_onnx_error(self, msg: str) -> None:
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        self._act_export_onnx.setEnabled(True)
        self._status_label.setText(f"ONNX export error: {msg[:80]}")
        QMessageBox.warning(self, "ONNX Export Error", msg)

    # ──────────────────────────────────────────────────────────────
    # Shared error handler
    # ──────────────────────────────────────────────────────────────

    @Slot(str)
    def _on_worker_error(self, msg: str) -> None:
        # Re-enable training so the user can retry after fixing the cause, and
        # clear any in-progress bar left behind by the failed worker. (_act_train
        # is only ever disabled by the training flows, so this is a no-op when a
        # non-training worker errors.)
        self._act_train.setEnabled(True)
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        self._status_label.setText(f"Error: {msg[:80]}")
        QMessageBox.warning(self, "Error", msg)

    # ──────────────────────────────────────────────────────────────
    # Canvas slots
    # ──────────────────────────────────────────────────────────────

    @Slot(object)
    def _on_box_added(self, bbox: BBox) -> None:
        if self.current_frame_index is not None:
            ann = self.store.get(self.current_frame_index)
            if ann:
                ann.boxes.append(bbox)
                if ann.status == "pending":
                    ann.status = "verified"
                    self._update_thumbnail_color(self.current_frame_index)
                self._refresh_propagation_actions()

    @Slot(object)
    def _on_box_deleted(self, bbox: BBox) -> None:
        if self.current_frame_index is not None:
            ann = self.store.get(self.current_frame_index)
            if ann:
                try:
                    ann.boxes.remove(bbox)
                except ValueError:
                    pass
                self._refresh_propagation_actions()

    @Slot(object)
    def _on_box_edited(self, bbox: BBox) -> None:
        if self.current_frame_index is not None:
            ann = self.store.get(self.current_frame_index)
            if ann and ann.status == "pending":
                ann.status = "verified"
                self._update_thumbnail_color(self.current_frame_index)

    # ──────────────────────────────────────────────────────────────
    # Frame navigation
    # ──────────────────────────────────────────────────────────────

    @Slot(int)
    def _on_frame_selected(self, row: int) -> None:
        if row < 0:
            return
        item = self._thumb_list.item(row)
        if item is None:
            return
        frame_index = item.data(Qt.UserRole)
        if frame_index == self.current_frame_index:
            return

        if self.current_frame_index is not None:
            ann = self.store.get(self.current_frame_index)
            if ann and ann.boxes and ann.status == "pending":
                ann.status = "verified"
                self._update_thumbnail_color(self.current_frame_index)

        self.current_frame_index = frame_index
        self._reload_canvas(frame_index)
        self._refresh_propagation_actions()

    def _reload_canvas(self, frame_index: int) -> None:
        ann = self.store.get(frame_index)
        if ann and os.path.isfile(ann.image_path):
            self._canvas.load_frame(ann.image_path, list(ann.boxes))

    # ──────────────────────────────────────────────────────────────
    # Thumbnail colour helpers
    # ──────────────────────────────────────────────────────────────

    def _update_thumbnail_color(self, frame_index: int) -> None:
        ann = self.store.get(frame_index)
        if ann is None:
            return
        color = _STATUS_COLORS.get(ann.status, QColor(180, 180, 180))
        for row in range(self._thumb_list.count()):
            item = self._thumb_list.item(row)
            if item and item.data(Qt.UserRole) == frame_index:
                item.setForeground(color)
                break

    def _update_thumbnail_colors(self) -> None:
        for row in range(self._thumb_list.count()):
            item = self._thumb_list.item(row)
            if item:
                self._update_thumbnail_color(item.data(Qt.UserRole))

    # ──────────────────────────────────────────────────────────────
    # Cleanup
    # ──────────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        for worker in (
            self._media_worker, self._train_worker,
            self._sam_load_worker, self._sam_text_worker,
            self._sam_prompt_worker, self._sam_track_worker,
            self._seg_converter_worker,
        ):
            if worker and worker.isRunning():
                if hasattr(worker, "abort"):
                    worker.abort()
                worker.quit()
                worker.wait(2000)
        super().closeEvent(event)
