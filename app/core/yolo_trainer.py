"""
YOLO training wrapper.

MODEL_REGISTRY maps UI display names to Ultralytics weight filenames.
DETECTION_MODELS / SEGMENTATION_MODELS provide pre-filtered lists for the UI.
"""

from __future__ import annotations

from qtpy.QtCore import QThread, Signal

MODEL_REGISTRY: dict[str, str] = {
    # ── Detection ─────────────────────────────────────────────────────────
    "YOLOv8n":       "yolov8n.pt",
    "YOLOv8s":       "yolov8s.pt",
    "YOLOv8m":       "yolov8m.pt",
    "YOLOv8l":       "yolov8l.pt",
    "YOLOv8x":       "yolov8x.pt",
    "YOLO11n":       "yolo11n.pt",
    "YOLO11s":       "yolo11s.pt",
    "YOLO11m":       "yolo11m.pt",
    "YOLO11l":       "yolo11l.pt",
    "YOLO12n":       "yolo12n.pt",
    "YOLO12s":       "yolo12s.pt",
    "YOLO12m":       "yolo12m.pt",
    "YOLO12l":       "yolo12l.pt",
    "YOLO26n":       "yolo26n.pt",
    "YOLO26s":       "yolo26s.pt",
    "YOLO26m":       "yolo26m.pt",
    "YOLO26l":       "yolo26l.pt",
    # ── Segmentation ──────────────────────────────────────────────────────
    # v8 / v11 ship pretrained -seg.pt checkpoints (direct download).
    "YOLOv8n-seg":   "yolov8n-seg.pt",
    "YOLOv8s-seg":   "yolov8s-seg.pt",
    "YOLOv8m-seg":   "yolov8m-seg.pt",
    "YOLOv8l-seg":   "yolov8l-seg.pt",
    "YOLOv8x-seg":   "yolov8x-seg.pt",
    "YOLO11n-seg":   "yolo11n-seg.pt",
    "YOLO11s-seg":   "yolo11s-seg.pt",
    "YOLO11m-seg":   "yolo11m-seg.pt",
    "YOLO11l-seg":   "yolo11l-seg.pt",
    # v12 / v26 support the segment task but have no published -seg.pt weights;
    # these are built from the architecture YAML and transfer-load the detection
    # backbone (see SEG_TRANSFER_BASE / YOLOTrainWorker._build_model).
    "YOLO12n-seg":   "yolo12n-seg.yaml",
    "YOLO12s-seg":   "yolo12s-seg.yaml",
    "YOLO12m-seg":   "yolo12m-seg.yaml",
    "YOLO12l-seg":   "yolo12l-seg.yaml",
    "YOLO26n-seg":   "yolo26n-seg.yaml",
    "YOLO26s-seg":   "yolo26s-seg.yaml",
    "YOLO26m-seg":   "yolo26m-seg.yaml",
    "YOLO26l-seg":   "yolo26l-seg.yaml",
    # FastSAM — pretrained segmentation checkpoints (always seg-only).
    "FastSAM-s":     "FastSAM-s.pt",
    "FastSAM-x":     "FastSAM-x.pt",
}

# For seg models built from a .yaml (no pretrained seg weights), transfer the
# matching detection checkpoint's backbone for a warm start instead of training
# from scratch.  Maps the seg display name → detection .pt to .load().
SEG_TRANSFER_BASE: dict[str, str] = {
    "YOLO12n-seg": "yolo12n.pt",
    "YOLO12s-seg": "yolo12s.pt",
    "YOLO12m-seg": "yolo12m.pt",
    "YOLO12l-seg": "yolo12l.pt",
    "YOLO26n-seg": "yolo26n.pt",
    "YOLO26s-seg": "yolo26s.pt",
    "YOLO26m-seg": "yolo26m.pt",
    "YOLO26l-seg": "yolo26l.pt",
}

# Keys that are segmentation models (ends with -seg OR is a FastSAM variant)
def _is_seg_key(key: str) -> bool:
    return key.endswith("-seg") or key.startswith("FastSAM")

DETECTION_MODELS:    list[str] = [k for k in MODEL_REGISTRY if not _is_seg_key(k)]
SEGMENTATION_MODELS: list[str] = [k for k in MODEL_REGISTRY if _is_seg_key(k)]

# SAM 2 keys are handled by sam2_trainer; re-exported here for convenience.
def _is_sam2_key(key: str) -> bool:
    return key.startswith("SAM 2")


class YOLOTrainWorker(QThread):
    """
    Runs YOLO training in a background thread.

    ultralytics YOLO.train() is blocking; the on_train_epoch_end callback
    fires on this thread, so Signal.emit() is safe (Qt queues cross-thread
    signals automatically).
    """

    epoch_done = Signal(int, int, float)  # (current_epoch, total_epochs, map50)
    finished   = Signal(str)              # path to best weights file
    error      = Signal(str)

    def __init__(
        self,
        model_key: str,
        data_yaml: str,
        epochs: int = 50,
        imgsz: int = 640,
        project: str = "runs/train",
        name: str = "exp",
        parent=None,
    ):
        super().__init__(parent)
        if model_key not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model key: {model_key!r}")
        self.model_key     = model_key
        self.data_yaml     = data_yaml
        self.epochs        = epochs
        self.imgsz         = imgsz
        self.project       = project
        self.name          = name
        self._total_epochs = epochs
        self._is_seg       = _is_seg_key(model_key)

    def _build_model(self, YOLO):
        """Construct the YOLO model for self.model_key.

        Most models load directly from a pretrained .pt.  Seg models that have
        no published -seg.pt (v12 / v26) are mapped to a .yaml architecture and
        transfer-load the matching detection backbone for a warm start.
        """
        weights = MODEL_REGISTRY[self.model_key]
        model = YOLO(weights)

        if weights.endswith(".yaml"):
            base = SEG_TRANSFER_BASE.get(self.model_key)
            if base:
                try:
                    model = model.load(base)  # transfer detection backbone
                except Exception:
                    # No compatible pretrained weights available — train the
                    # architecture from scratch rather than failing outright.
                    pass
        return model

    def run(self) -> None:
        try:
            from ultralytics import YOLO

            model = self._build_model(YOLO)

            is_seg = self._is_seg

            def _on_epoch_end(trainer) -> None:
                epoch = trainer.epoch + 1
                m = trainer.metrics
                if is_seg:
                    # For seg models prefer mask mAP50; fall back to box mAP50
                    map50 = float(
                        m.get("metrics/mAP50(M)", 0.0)
                        or m.get("metrics/mAP50(B)", 0.0)
                        or m.get("mAP50", 0.0)
                    )
                else:
                    map50 = float(
                        m.get("metrics/mAP50(B)", 0.0)
                        or m.get("mAP50", 0.0)
                    )
                self.epoch_done.emit(epoch, self._total_epochs, map50)

            model.add_callback("on_train_epoch_end", _on_epoch_end)

            results = model.train(
                data=self.data_yaml,
                epochs=self.epochs,
                imgsz=self.imgsz,
                project=self.project,
                name=self.name,
                exist_ok=True,
                verbose=False,
                batch=-1,    # auto-batch to maximise VRAM
                cache=True,
                workers=8,
            )

            best = str(getattr(results, "best", "") or "")
            self.finished.emit(best)

        except Exception as exc:
            self.error.emit(f"Training failed: {exc}")
