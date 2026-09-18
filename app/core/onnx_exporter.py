"""
ONNX export wrapper.

Converts a trained Ultralytics ``.pt`` checkpoint (detection, segmentation or
pose) to ONNX.  Precision is selectable: FP32 (default, runs anywhere) or FP16
(half — smaller file / faster GPU inference, requires a CUDA device to export).
"""

from __future__ import annotations

import os
import shutil
from typing import Callable

from app.core.yolo_trainer import RUNS_DIR, _PROJECT_ROOT, configure_ultralytics_dirs


def _inside_root(path: str) -> bool:
    """True if ``path`` lives under the project root."""
    try:
        return os.path.commonpath([os.path.abspath(path), _PROJECT_ROOT]) == _PROJECT_ROOT
    except ValueError:  # different drives on Windows
        return False


def export_onnx(
    pt_path: str,
    *,
    half: bool = False,
    imgsz: int = 640,
    dynamic: bool = True,
    on_log: Callable[[str], None] | None = None,
) -> dict:
    """Export *pt_path* to ONNX and return {"path": <written .onnx>}."""
    import torch
    from ultralytics import YOLO

    configure_ultralytics_dirs()  # keep any downloads inside the root

    if not os.path.isfile(pt_path):
        raise FileNotFoundError(f"Checkpoint not found: {pt_path}")

    # FP16 export needs a CUDA device — torch cannot trace half ops on CPU for
    # most layers, and an FP16 graph is only useful for GPU inference anyway.
    if half and not torch.cuda.is_available():
        raise RuntimeError(
            "FP16 export requires a CUDA GPU. Select FP32 precision, "
            "or run on a machine with CUDA available."
        )
    device = 0 if torch.cuda.is_available() else "cpu"

    if on_log:
        on_log(f"Loading {os.path.basename(pt_path)}…")

    model = YOLO(pt_path)
    out = model.export(
        format="onnx",
        half=half,
        imgsz=imgsz,
        device=device,
        dynamic=dynamic,
        simplify=False,  # avoids pulling onnxslim/onnxruntime at runtime
    )

    out_path = str(out)
    if not out_path or not os.path.isfile(out_path):
        # Older return shapes: fall back to the conventional sibling path
        out_path = os.path.splitext(pt_path)[0] + ".onnx"

    # Ultralytics writes the .onnx next to the input .pt.  If that lands outside
    # the project root (e.g. a checkpoint picked from elsewhere), move it into
    # <root>/runs/export so nothing is generated outside the project.
    if not _inside_root(out_path):
        export_dir = os.path.join(RUNS_DIR, "export")
        os.makedirs(export_dir, exist_ok=True)
        dest = os.path.join(export_dir, os.path.basename(out_path))
        shutil.move(out_path, dest)
        out_path = dest

    return {"path": out_path}
