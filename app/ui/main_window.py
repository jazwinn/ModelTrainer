"""
Main application window.

Owns the AnnotationStore and wires all worker threads to UI slots.
Layout:
  Vertical control panel (left) — grouped sections: Media / Auto-Label /
    Interactive Prompt / Propagate / Classes / Export & Train
  QSplitter  — thumbnail QListWidget + AnnotationCanvas
  Status bar — QProgressBar + status label
"""

from __future__ import annotations

import os
import tempfile

from qtpy.QtCore import Qt, QSize, Slot
from qtpy.QtGui import QColor, QPalette, QIcon, QPixmap, QImage
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
    QGroupBox,
    QScrollArea,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QFileDialog,
    QInputDialog,
    QMessageBox,
    QSpinBox,
    QLineEdit,
    QApplication,
    QSizePolicy,
)

from app.core.sam3_handler import (
    AnnotationStore, BBox, FrameAnnotation,
    SAM3TextWorker, SAM3LoadWorker, SAM3PromptWorker, SAM3TrackWorker, SAM_MODELS,
)
from app.core.media_loader import MediaLoaderWorker
from app.core.yolo_trainer import YOLOTrainWorker, MODEL_REGISTRY
from app.ui.canvas import AnnotationCanvas
from app.utils.yolo_exporter import export_dataset


def _apply_dark_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    pal = QPalette()
    c = {
        QPalette.Window:          QColor(30, 30, 30),
        QPalette.WindowText:      QColor(220, 220, 220),
        QPalette.Base:            QColor(20, 20, 20),
        QPalette.AlternateBase:   QColor(35, 35, 35),
        QPalette.ToolTipBase:     QColor(255, 255, 220),
        QPalette.ToolTipText:     QColor(0, 0, 0),
        QPalette.Text:            QColor(220, 220, 220),
        QPalette.Button:          QColor(45, 45, 45),
        QPalette.ButtonText:      QColor(220, 220, 220),
        QPalette.BrightText:      QColor(255, 100, 100),
        QPalette.Link:            QColor(42, 130, 218),
        QPalette.Highlight:       QColor(42, 130, 218),
        QPalette.HighlightedText: QColor(255, 255, 255),
        QPalette.Disabled + QPalette.Text:       QColor(100, 100, 100),
        QPalette.Disabled + QPalette.ButtonText: QColor(100, 100, 100),
    }
    for role, color in c.items():
        pal.setColor(role, color)
    app.setPalette(pal)


_THUMBNAIL_SIZE = 80


def _make_thumbnail(png_path: str) -> QPixmap:
    img = QImage(png_path)
    if img.isNull():
        img = QImage(_THUMBNAIL_SIZE, _THUMBNAIL_SIZE, QImage.Format_RGB32)
        img.fill(QColor(50, 50, 50))
    pix = QPixmap.fromImage(img).scaled(
        _THUMBNAIL_SIZE, _THUMBNAIL_SIZE,
        Qt.KeepAspectRatio,
        Qt.SmoothTransformation,
    )
    return pix


_STATUS_COLORS = {
    "pending":  QColor(200, 180, 0),
    "verified": QColor(0, 200, 80),
    "exported": QColor(120, 120, 120),
}


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ModelTrainer")
        self.resize(1280, 800)

        # Apply dark theme to the application
        app = QApplication.instance()
        if app:
            _apply_dark_theme(app)

        # ── State ─────────────────────────────────────────────────────
        self.store: AnnotationStore = {}
        self.class_names: list[str] = ["object"]
        self.current_frame_index: int | None = None
        self._temp_dir: str = tempfile.mkdtemp(prefix="modeltrainer_")

        # Active workers
        self._media_worker: MediaLoaderWorker | None = None
        self._train_worker: YOLOTrainWorker | None = None

        # SAM 3 state
        self._sam_load_worker: SAM3LoadWorker | None = None
        self._sam_text_worker: SAM3TextWorker | None = None
        self._sam_prompt_worker: SAM3PromptWorker | None = None
        self._sam_track_worker: SAM3TrackWorker | None = None
        self._sam_model_obj = None          # cached Sam3Model (concept/prompt)
        self._sam_processor_obj = None
        self._sam_pending_action = None     # callable to run once the model loads
        self._sam_tracker_model = None      # cached Sam3TrackerVideoModel
        self._sam_tracker_processor = None
        self._track_start_index: int | None = None

        # ── Widgets ───────────────────────────────────────────────────
        self._build_actions()
        self._build_central()
        self._build_statusbar()

    # ------------------------------------------------------------------
    # Layout builders
    # ------------------------------------------------------------------

    # -- small panel helpers -------------------------------------------

    def _panel_button(self, action: QAction) -> QToolButton:
        """A full-width button bound to a QAction (mirrors its state/signals)."""
        btn = QToolButton()
        btn.setDefaultAction(action)
        btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
        btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        btn.setMinimumHeight(28)
        return btn

    @staticmethod
    def _row(label_text: str, widget: QWidget) -> QHBoxLayout:
        row = QHBoxLayout()
        lbl = QLabel(label_text)
        lbl.setMinimumWidth(70)
        row.addWidget(lbl)
        row.addWidget(widget, 1)
        return row

    def _build_actions(self) -> None:
        """Create all actions/widgets and lay them out in a vertical panel."""
        panel = QWidget()
        v = QVBoxLayout(panel)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(10)

        # ── 1 · Media ────────────────────────────────────────────────
        self._act_import = QAction("Import Media", self)
        self._act_import.setToolTip("Import a directory of images or videos")
        self._act_import.triggered.connect(self._on_import)

        g_media = QGroupBox("1 · Media")
        lm = QVBoxLayout(g_media)
        lm.addWidget(self._panel_button(self._act_import))
        v.addWidget(g_media)

        # ── 2 · Auto-Label (SAM 3) ───────────────────────────────────
        self._act_sam = QAction("Run SAM 3", self)
        self._act_sam.setToolTip(
            "Auto-label every frame: SAM 3 segments all objects matching the\n"
            "Concept text (or the selected class name if Concept is blank)."
        )
        self._act_sam.setEnabled(False)
        self._act_sam.triggered.connect(self._on_run_sam)

        self._concept_edit = QLineEdit()
        self._concept_edit.setPlaceholderText("e.g. car (blank = class)")
        self._concept_edit.setToolTip(
            "Text concept SAM 3 looks for. Leave blank to use the selected class name."
        )
        self._sam_model_combo = QComboBox()
        self._sam_model_combo.addItems(list(SAM_MODELS.keys()))
        self._sam_model_combo.setCurrentText("SAM 3.1 (facebook/sam3.1)")
        self._sam_model_combo.setToolTip(
            "SAM 3 checkpoint. 3.1 falls back to facebook/sam3 if the 3.1 weights\n"
            "aren't available through transformers."
        )
        self._stride_spin = QSpinBox()
        self._stride_spin.setRange(1, 120)
        self._stride_spin.setValue(1)
        self._stride_spin.setToolTip("Process every Nth frame (1 = all frames, 5 = every 5th)")

        self._conf_spin = QSpinBox()
        self._conf_spin.setRange(1, 100)
        self._conf_spin.setValue(50)
        self._conf_spin.setSuffix("%")
        self._conf_spin.setToolTip("Confidence threshold (lower = more boxes but more false positives)")

        g_auto = QGroupBox("2 · Auto-Label (SAM 3)")
        la = QVBoxLayout(g_auto)
        la.addLayout(self._row("Concept:", self._concept_edit))
        la.addLayout(self._row("Confidence:", self._conf_spin))
        la.addWidget(self._panel_button(self._act_sam))
        la.addLayout(self._row("Model:", self._sam_model_combo))
        la.addLayout(self._row("Stride:", self._stride_spin))
        v.addWidget(g_auto)

        # ── 3 · Interactive Prompt ───────────────────────────────────
        self._act_sam_prompt = QAction("SAM 3 Prompt", self)
        self._act_sam_prompt.setCheckable(True)
        self._act_sam_prompt.setEnabled(False)
        self._act_sam_prompt.setToolTip(
            "Interactive positive/negative prompting on the current frame:\n"
            "  left-drag  = POSITIVE example box (green) — find more like this\n"
            "  right-drag = NEGATIVE example box (red)  — exclude things like this\n"
            "Then press 'Run Prompt'."
        )
        self._act_sam_prompt.toggled.connect(self._on_sam_prompt_toggled)

        self._act_run_prompt = QAction("Run Prompt", self)
        self._act_run_prompt.setEnabled(False)
        self._act_run_prompt.setToolTip("Segment the current frame using the drawn positive/negative boxes")
        self._act_run_prompt.triggered.connect(self._on_run_sam_prompt)

        self._act_clear_prompt = QAction("Clear Prompts", self)
        self._act_clear_prompt.setEnabled(False)
        self._act_clear_prompt.setToolTip("Remove all positive/negative example boxes")
        self._act_clear_prompt.triggered.connect(self._on_clear_prompts)

        g_prompt = QGroupBox("3 · Interactive Prompt")
        lp = QVBoxLayout(g_prompt)
        lp.addWidget(self._panel_button(self._act_sam_prompt))
        lp.addWidget(self._panel_button(self._act_run_prompt))
        lp.addWidget(self._panel_button(self._act_clear_prompt))
        v.addWidget(g_prompt)

        # ── 4 · Propagate from this frame ────────────────────────────
        self._act_propagate = QAction("Propagate Labels", self)
        self._act_propagate.setEnabled(False)
        self._act_propagate.setToolTip(
            "Use the labels on THIS frame to auto-label all frames.\n"
            "Each class you've boxed here becomes a SAM 3 concept searched on\n"
            "every frame. Best for unrelated images / mixed scenes."
        )
        self._act_propagate.triggered.connect(self._on_propagate_labels)

        self._act_track = QAction("Track Video", self)
        self._act_track.setEnabled(False)
        self._act_track.setToolTip(
            "Track the exact objects boxed on THIS frame through the video\n"
            "(SAM 3 memory tracker). Best for ordered video of the same moving\n"
            "objects. Start from the first frame."
        )
        self._act_track.triggered.connect(self._on_track_video)

        self._track_range_spin = QSpinBox()
        self._track_range_spin.setRange(0, 100000)
        self._track_range_spin.setValue(0)
        self._track_range_spin.setSpecialValueText("all")
        self._track_range_spin.setToolTip(
            "How many frames forward to track from the seed frame.\n"
            "0 = whole video. Set a limit to bound memory/time on long videos."
        )

        g_prop = QGroupBox("4 · Propagate from this frame")
        lpr = QVBoxLayout(g_prop)
        lpr.addWidget(self._panel_button(self._act_propagate))
        lpr.addWidget(self._panel_button(self._act_track))
        lpr.addLayout(self._row("Track range:", self._track_range_spin))
        v.addWidget(g_prop)

        # ── 5 · Classes ──────────────────────────────────────────────
        self._act_classes = QAction("Edit Classes", self)
        self._act_classes.setToolTip("Define annotation class names")
        self._act_classes.triggered.connect(self._on_edit_classes)

        self._class_combo = QComboBox()
        self._class_combo.addItems(self.class_names)
        self._class_combo.currentIndexChanged.connect(self._on_class_changed)

        g_cls = QGroupBox("5 · Classes")
        lc = QVBoxLayout(g_cls)
        lc.addWidget(self._panel_button(self._act_classes))
        lc.addLayout(self._row("Class:", self._class_combo))
        v.addWidget(g_cls)

        # ── 6 · Export & Train ───────────────────────────────────────
        self._act_export = QAction("Export YOLO", self)
        self._act_export.setToolTip("Export annotations to YOLO .txt format")
        self._act_export.setEnabled(False)
        self._act_export.triggered.connect(self._on_export)

        self._model_combo = QComboBox()
        self._model_combo.addItems(list(MODEL_REGISTRY.keys()))

        self._epoch_spin = QSpinBox()
        self._epoch_spin.setRange(1, 1000)
        self._epoch_spin.setValue(50)

        self._act_train = QAction("Train", self)
        self._act_train.setToolTip("Train the selected YOLO model on exported annotations")
        self._act_train.setEnabled(False)
        self._act_train.triggered.connect(self._on_train)

        g_train = QGroupBox("6 · Export & Train")
        lt = QVBoxLayout(g_train)
        lt.addWidget(self._panel_button(self._act_export))
        lt.addLayout(self._row("Model:", self._model_combo))
        lt.addLayout(self._row("Epochs:", self._epoch_spin))
        lt.addWidget(self._panel_button(self._act_train))
        v.addWidget(g_train)

        v.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        scroll.setMinimumWidth(250)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._control_panel = scroll

    def _build_central(self) -> None:
        # Thumbnail list
        self._thumb_list = QListWidget()
        self._thumb_list.setIconSize(QSize(_THUMBNAIL_SIZE, _THUMBNAIL_SIZE))
        self._thumb_list.setMinimumWidth(_THUMBNAIL_SIZE + 40)
        self._thumb_list.setSpacing(4)
        self._thumb_list.currentRowChanged.connect(self._on_frame_selected)

        # Canvas
        self._canvas = AnnotationCanvas()
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._canvas.box_added.connect(self._on_box_added)
        self._canvas.box_deleted.connect(self._on_box_deleted)
        self._canvas.box_edited.connect(self._on_box_edited)
        self._canvas.exemplars_changed.connect(self._on_exemplars_changed)

        main_split = QSplitter(Qt.Horizontal)
        main_split.addWidget(self._control_panel)
        main_split.addWidget(self._thumb_list)
        main_split.addWidget(self._canvas)
        
        main_split.setStretchFactor(0, 0)
        main_split.setStretchFactor(1, 0)
        main_split.setStretchFactor(2, 1)

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(main_split)
        self.setCentralWidget(central)

    def _build_statusbar(self) -> None:
        sb = QStatusBar(self)
        self.setStatusBar(sb)

        self._status_label = QLabel("Ready")
        self._status_label.setMinimumWidth(220)
        sb.addWidget(self._status_label)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setFixedWidth(300)
        self._progress.setTextVisible(True)
        sb.addWidget(self._progress)

    # ------------------------------------------------------------------
    # Toolbar actions
    # ------------------------------------------------------------------

    @Slot()
    def _on_import(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Select Media Directory", os.path.expanduser("~")
        )
        if not path:
            return

        if self._media_worker and self._media_worker.isRunning():
            self._media_worker.abort()
            self._media_worker.wait()

        self._status_label.setText("Importing media…")
        self._progress.setValue(0)
        self._act_sam.setEnabled(False)

        self._media_worker = MediaLoaderWorker([path], self._temp_dir)
        self._media_worker.frame_ready.connect(self._on_frame_ready)
        self._media_worker.progress.connect(self._on_media_progress)
        self._media_worker.finished.connect(self._on_media_finished)
        self._media_worker.error.connect(self._on_worker_error)
        self._media_worker.start()

    def _current_concept(self) -> str:
        """Concept text for SAM 3 — the Concept field, or the selected class name."""
        text = self._concept_edit.text().strip()
        if text:
            return text
        idx = self._class_combo.currentIndex()
        if 0 <= idx < len(self.class_names):
            return self.class_names[idx]
        return self.class_names[0] if self.class_names else "object"

    def _ensure_sam_loaded(self, then) -> None:
        """Run `then` once the SAM 3 model is loaded (loading it if needed)."""
        if self._sam_model_obj is not None:
            then()
            return

        self._sam_pending_action = then
        if self._sam_load_worker and self._sam_load_worker.isRunning():
            return

        # Pre-load torch + SAM 3 classes in the main thread (Windows DLL requirement).
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
                "SAM 3 requires a recent transformers (v5+):\n"
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
        self._status_label.setText("SAM 3 model loaded")
        action = self._sam_pending_action
        self._sam_pending_action = None
        if action:
            action()

    @Slot(str)
    def _on_sam_load_error(self, msg: str) -> None:
        self._sam_pending_action = None
        self._on_worker_error(msg)

    @Slot()
    def _on_run_sam(self) -> None:
        if self._sam_text_worker and self._sam_text_worker.isRunning():
            return

        concept = self._current_concept()
        stride = self._stride_spin.value()
        frames = [
            (idx, ann.image_path)
            for idx, ann in sorted(self.store.items())
            if idx % stride == 0
        ]
        if not frames:
            QMessageBox.information(self, "SAM 3", "No frames to process.")
            return

        self._ensure_sam_loaded(lambda: self._start_sam_text(frames, concept))

    def _start_sam_text(self, frames: list, concept: str) -> None:
        class_id = self._class_combo.currentIndex()
        self._status_label.setText(
            f"SAM 3: segmenting '{concept}' on {len(frames)} frames…"
        )
        self._progress.setRange(0, len(frames))
        self._progress.setValue(0)
        self._act_sam.setEnabled(False)

        threshold = self._conf_spin.value() / 100.0
        self._sam_text_worker = SAM3TextWorker(
            self._sam_model_obj, self._sam_processor_obj,
            frames, [(concept, class_id)], threshold=threshold
        )
        self._sam_text_worker.boxes_ready.connect(self._on_boxes_ready)
        self._sam_text_worker.progress.connect(self._on_sam_progress)
        self._sam_text_worker.finished.connect(self._on_sam_finished)
        self._sam_text_worker.error.connect(self._on_worker_error)
        self._sam_text_worker.start()

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

    @Slot()
    def _on_export(self) -> None:
        out_dir = QFileDialog.getExistingDirectory(
            self, "Select Export Directory", os.path.expanduser("~")
        )
        if not out_dir:
            return

        count = export_dataset(self.store, self.class_names, out_dir)
        self._status_label.setText(f"Exported {count} frames to {out_dir}")
        self._update_thumbnail_colors()
        self._act_train.setEnabled(count > 0)

    @Slot()
    def _on_train(self) -> None:
        out_dir = QFileDialog.getExistingDirectory(
            self, "Select Dataset Directory (containing data.yaml)", os.path.expanduser("~")
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

        # Pre-load PyTorch in the main thread (Windows DLL requirement).
        try:
            import torch  # noqa: F401
        except Exception as exc:
            QMessageBox.critical(self, "PyTorch Error", str(exc))
            return

        model_key = self._model_combo.currentText()
        epochs = self._epoch_spin.value()

        self._status_label.setText(f"Training {model_key} for {epochs} epochs…")
        self._progress.setRange(0, epochs)
        self._progress.setValue(0)
        self._act_train.setEnabled(False)

        self._train_worker = YOLOTrainWorker(
            model_key=model_key,
            data_yaml=yaml_path,
            epochs=epochs,
        )
        self._train_worker.epoch_done.connect(self._on_epoch_done)
        self._train_worker.finished.connect(self._on_train_finished)
        self._train_worker.error.connect(self._on_worker_error)
        self._train_worker.start()

    # ------------------------------------------------------------------
    # Worker slots — media loader
    # ------------------------------------------------------------------

    @Slot(int, str)
    def _on_frame_ready(self, frame_index: int, png_path: str) -> None:
        annotation = FrameAnnotation(
            frame_index=frame_index,
            image_path=png_path,
        )
        self.store[frame_index] = annotation

        item = QListWidgetItem()
        item.setIcon(QIcon(_make_thumbnail(png_path)))
        item.setText(f"{frame_index}")
        item.setData(Qt.UserRole, frame_index)
        self._thumb_list.addItem(item)

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
        self._status_label.setText(f"Imported {count} frames")
        self._progress.setValue(self._progress.maximum())
        self._act_sam.setEnabled(count > 0)
        self._act_sam_prompt.setEnabled(count > 0)
        self._act_export.setEnabled(count > 0)

    # ------------------------------------------------------------------
    # Worker slots — SAM
    # ------------------------------------------------------------------

    @Slot(int, list)
    def _on_boxes_ready(self, frame_index: int, boxes: list) -> None:
        ann = self.store.get(frame_index)
        if ann is None:
            return
        ann.boxes = boxes
        ann.status = "pending"
        self._update_thumbnail_color(frame_index)

        # Always jump to the frame SAM just finished so the user can
        # watch boxes appear live.  Block the list signal so we don't
        # trigger the manual-navigation "mark as verified" side-effect.
        self.current_frame_index = frame_index
        self._reload_canvas(frame_index)
        for row in range(self._thumb_list.count()):
            item = self._thumb_list.item(row)
            if item and item.data(Qt.UserRole) == frame_index:
                self._thumb_list.blockSignals(True)
                self._thumb_list.setCurrentRow(row)
                self._thumb_list.scrollToItem(item)
                self._thumb_list.blockSignals(False)
                break
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
        self._status_label.setText("SAM 3 labeling complete")
        self._progress.setValue(self._progress.maximum())
        self._act_sam.setEnabled(True)
        self._act_export.setEnabled(True)

    # ------------------------------------------------------------------
    # SAM 3 positive/negative prompt handlers
    # ------------------------------------------------------------------

    @Slot(bool)
    def _on_sam_prompt_toggled(self, checked: bool) -> None:
        self._canvas.set_sam_mode(checked)
        self._act_run_prompt.setEnabled(checked)
        self._act_clear_prompt.setEnabled(checked)
        if checked:
            self._status_label.setText(
                "SAM 3 Prompt ON — left-drag = positive (green), "
                "right-drag = negative (red), then 'Run Prompt'"
            )
        else:
            self._status_label.setText("Ready")

    @Slot()
    def _on_exemplars_changed(self) -> None:
        if not self._act_sam_prompt.isChecked():
            return
        pos, neg = self._canvas.get_exemplars()
        self._status_label.setText(
            f"SAM 3 Prompt: {len(pos)} positive, {len(neg)} negative example(s) — 'Run Prompt' to segment"
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
                "Add at least one positive example box, or type a Concept first.\n\n"
                "Left-drag = positive example (green)\n"
                "Right-drag = negative example (red)"
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

        threshold = self._conf_spin.value() / 100.0
        self._sam_prompt_worker = SAM3PromptWorker(
            self._sam_model_obj, self._sam_processor_obj,
            frame_index, pil_img, text, pos, neg, class_id=class_id, threshold=threshold
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
            self._reload_canvas(frame_index)  # also clears exemplar overlays
        self._refresh_propagation_actions()
        self._status_label.setText(
            f"SAM 3 Prompt: added {len(boxes)} object(s) — refine or toggle SAM 3 Prompt off"
        )

    # ------------------------------------------------------------------
    # Propagate first-frame labels to the whole dataset
    # ------------------------------------------------------------------

    def _refresh_propagation_actions(self) -> None:
        ann = self.store.get(self.current_frame_index) if self.current_frame_index is not None else None
        has_labels = bool(ann and ann.boxes)
        self._act_propagate.setEnabled(has_labels)
        self._act_track.setEnabled(has_labels)

    def _seed_boxes(self) -> list:
        """Current frame's boxes as ((x1,y1,x2,y2), class_id) seeds for tracking."""
        ann = self.store.get(self.current_frame_index) if self.current_frame_index is not None else None
        if not ann:
            return []
        return [((b.x1, b.y1, b.x2, b.y2), b.class_id) for b in ann.boxes]

    # ---- concept propagation (text, works on images or video) ----

    @Slot()
    def _on_propagate_labels(self) -> None:
        if self._sam_text_worker and self._sam_text_worker.isRunning():
            return
        if self.current_frame_index is None:
            return
        ann = self.store.get(self.current_frame_index)
        if not ann or not ann.boxes:
            QMessageBox.information(self, "Propagate Labels", "Label the current frame first.")
            return

        class_ids = sorted({b.class_id for b in ann.boxes})
        concepts: list[tuple[str, int]] = []
        for cid in class_ids:
            if 0 <= cid < len(self.class_names) and self.class_names[cid].strip():
                concepts.append((self.class_names[cid].strip(), cid))
        if not concepts:
            QMessageBox.information(
                self, "Propagate Labels",
                "Your labels need class names. Use 'Edit Classes' to name them, "
                "then assign each box a class."
            )
            return

        # Every frame except the seed frame (keep the user's own labels there).
        frames = [
            (idx, a.image_path)
            for idx, a in sorted(self.store.items())
            if idx != self.current_frame_index
        ]
        if not frames:
            return
        self._ensure_sam_loaded(lambda: self._start_propagate(frames, concepts))

    def _start_propagate(self, frames: list, concepts: list) -> None:
        names = ", ".join(c for c, _ in concepts)
        self._status_label.setText(
            f"SAM 3: propagating [{names}] to {len(frames)} frames…"
        )
        self._progress.setRange(0, len(frames))
        self._progress.setValue(0)
        self._act_sam.setEnabled(False)
        self._act_propagate.setEnabled(False)

        threshold = self._conf_spin.value() / 100.0
        self._sam_text_worker = SAM3TextWorker(
            self._sam_model_obj, self._sam_processor_obj, frames, concepts, threshold=threshold
        )
        self._sam_text_worker.boxes_ready.connect(self._on_boxes_ready)
        self._sam_text_worker.progress.connect(self._on_sam_progress)
        self._sam_text_worker.finished.connect(self._on_sam_finished)
        self._sam_text_worker.error.connect(self._on_worker_error)
        self._sam_text_worker.start()

    # ---- video tracking (visual memory, ordered video) ----

    @Slot()
    def _on_track_video(self) -> None:
        if self._sam_track_worker and self._sam_track_worker.isRunning():
            return
            
        seed_frames = []
        for idx, ann in self.store.items():
            if ann.boxes and ann.status in ("verified", "pending"):
                boxes = [((b.x1, b.y1, b.x2, b.y2), b.class_id) for b in ann.boxes]
                seed_frames.append((idx, boxes))

        if not seed_frames:
            # Fallback to current frame if no verified/pending labels exist
            seeds = self._seed_boxes()
            if seeds:
                seed_frames.append((self.current_frame_index, seeds))
            else:
                QMessageBox.information(self, "Track Video", "Label at least one frame first.")
                return

        frames = [(idx, a.image_path) for idx, a in sorted(self.store.items())]
        self._track_start_index = min(fidx for fidx, _ in seed_frames)

        # Pre-import tracker classes in the main thread (Windows DLL requirement).
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

        max_frames = self._track_range_spin.value() or None  # 0 → None (whole video)
        self._sam_track_worker = SAM3TrackWorker(
            frames, seed_frames, self._track_start_index,
            max_frames=max_frames,
            model=self._sam_tracker_model, processor=self._sam_tracker_processor,
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
        # Keep the user's own labels on the seed frame.
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
        self._status_label.setText("SAM 3 tracking complete")
        self._progress.setValue(self._progress.maximum())
        self._act_track.setEnabled(True)
        self._act_export.setEnabled(True)

    # ------------------------------------------------------------------
    # Worker slots — training
    # ------------------------------------------------------------------

    @Slot(int, int, float)
    def _on_epoch_done(self, epoch: int, total: int, map50: float) -> None:
        self._progress.setValue(epoch)
        self._status_label.setText(
            f"Epoch {epoch}/{total}  —  mAP50: {map50:.4f}"
        )

    @Slot(str)
    def _on_train_finished(self, best_weights: str) -> None:
        self._status_label.setText(
            f"Training complete. Best weights: {best_weights or 'see runs/train/'}"
        )
        self._act_train.setEnabled(True)
        QMessageBox.information(
            self, "Training Complete",
            f"Training finished.\n\nBest weights:\n{best_weights or 'runs/train/exp/weights/best.pt'}"
        )

    # ------------------------------------------------------------------
    # Worker slots — shared error handler
    # ------------------------------------------------------------------

    @Slot(str)
    def _on_worker_error(self, msg: str) -> None:
        self._status_label.setText(f"Error: {msg[:80]}")
        QMessageBox.warning(self, "Worker Error", msg)

    # ------------------------------------------------------------------
    # Canvas slots
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Frame navigation
    # ------------------------------------------------------------------

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

        # Mark current frame as verified if it has boxes and was pending
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

    # ------------------------------------------------------------------
    # Thumbnail color helpers
    # ------------------------------------------------------------------

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
                frame_index = item.data(Qt.UserRole)
                self._update_thumbnail_color(frame_index)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        for worker in (
            self._media_worker, self._train_worker,
            self._sam_load_worker, self._sam_text_worker, self._sam_prompt_worker,
            self._sam_track_worker,
        ):
            if worker and worker.isRunning():
                if hasattr(worker, "abort"):
                    worker.abort()
                worker.quit()
                worker.wait(2000)
        super().closeEvent(event)
