"""
YOLO training wrapper.

MODEL_REGISTRY maps UI display names to Ultralytics weight filenames.
Adding a new model requires only one dict entry — no other code changes.
"""

from __future__ import annotations

from qtpy.QtCore import QThread, Signal

MODEL_REGISTRY: dict[str, str] = {
    # YOLOv8
    "YOLOv8n":  "yolov8n.pt",
    "YOLOv8s":  "yolov8s.pt",
    "YOLOv8m":  "yolov8m.pt",
    "YOLOv8l":  "yolov8l.pt",
    "YOLOv8x":  "yolov8x.pt",
    # YOLO11
    "YOLO11n":  "yolo11n.pt",
    "YOLO11s":  "yolo11s.pt",
    "YOLO11m":  "yolo11m.pt",
    "YOLO11l":  "yolo11l.pt",
    # YOLO26
    "YOLO26n":  "yolo26n.pt",
    "YOLO26s":  "yolo26s.pt",
    "YOLO26m":  "yolo26m.pt",
    "YOLO26l":  "yolo26l.pt",
}


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
        self.model_key  = model_key
        self.data_yaml  = data_yaml
        self.epochs     = epochs
        self.imgsz      = imgsz
        self.project    = project
        self.name       = name
        self._total_epochs = epochs

    def run(self) -> None:
        try:
            from ultralytics import YOLO

            weights = MODEL_REGISTRY[self.model_key]
            model = YOLO(weights)

            def _on_epoch_end(trainer) -> None:
                epoch = trainer.epoch + 1
                map50 = float(
                    trainer.metrics.get("metrics/mAP50(B)", 0.0)
                    or trainer.metrics.get("mAP50", 0.0)
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
                batch=-1,     # Auto-batch to maximize VRAM usage
                cache=True,   # Cache dataset in RAM to prevent disk bottlenecks
                workers=8,    # Increase dataloader workers
            )

            best = str(getattr(results, "best", "") or "")
            self.finished.emit(best)

        except Exception as exc:
            self.error.emit(f"Training failed: {exc}")
