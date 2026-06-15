"""
ONNX export wrapper.

Converts a trained Ultralytics ``.pt`` checkpoint (detection or segmentation)
to ONNX in a background thread so the UI stays responsive.

Precision is selectable: FP32 (default, runs anywhere) or FP16 (half — smaller
file / faster GPU inference, requires a CUDA device for export).
"""

from __future__ import annotations

import os
import shutil

from qtpy.QtCore import QThread, Signal

from app.core.yolo_trainer import (
    RUNS_DIR, _PROJECT_ROOT, configure_ultralytics_dirs,
)


def _inside_root(path: str) -> bool:
    """True if ``path`` lives under the project root."""
    try:
        return os.path.commonpath(
            [os.path.abspath(path), _PROJECT_ROOT]
        ) == _PROJECT_ROOT
    except ValueError:  # different drives on Windows
        return False


class ONNXExportWorker(QThread):
    """Runs ``YOLO(...).export(format="onnx")`` off the GUI thread."""

    finished = Signal(str)  # path to the written .onnx file
    error    = Signal(str)

    def __init__(
        self,
        pt_path: str,
        half: bool = False,
        imgsz: int = 640,
        dynamic: bool = True,
        parent=None,
    ):
        super().__init__(parent)
        self.pt_path = pt_path
        self.half    = half
        self.imgsz   = imgsz
        self.dynamic = dynamic

    def run(self) -> None:
        try:
            import torch
            from ultralytics import YOLO

            configure_ultralytics_dirs()  # keep any downloads inside the root

            if not os.path.isfile(self.pt_path):
                raise FileNotFoundError(f"Checkpoint not found: {self.pt_path}")

            half = self.half
            # FP16 export needs a CUDA device — torch can't trace half ops on CPU
            # for most layers, and an FP16 graph is only useful for GPU inference.
            if half and not torch.cuda.is_available():
                raise RuntimeError(
                    "FP16 export requires a CUDA GPU. Select FP32 precision, "
                    "or run on a machine with CUDA available."
                )
            device = 0 if torch.cuda.is_available() else "cpu"

            model = YOLO(self.pt_path)
            out = model.export(
                format="onnx",
                half=half,
                imgsz=self.imgsz,
                device=device,
                dynamic=self.dynamic,
                simplify=False,  # avoids pulling onnxslim/onnxruntime at runtime
            )

            out_path = str(out)
            if not out_path or not os.path.isfile(out_path):
                # Older return shapes: fall back to the conventional sibling path
                out_path = os.path.splitext(self.pt_path)[0] + ".onnx"

            # Ultralytics writes the .onnx next to the input .pt. If that lands
            # outside the project root (e.g. a checkpoint picked from elsewhere),
            # move it into <root>/runs/export so nothing is generated outside.
            if not _inside_root(out_path):
                export_dir = os.path.join(RUNS_DIR, "export")
                os.makedirs(export_dir, exist_ok=True)
                dest = os.path.join(export_dir, os.path.basename(out_path))
                shutil.move(out_path, dest)
                out_path = dest

            self.finished.emit(out_path)

        except Exception as exc:
            self.error.emit(f"ONNX export failed: {exc}")
