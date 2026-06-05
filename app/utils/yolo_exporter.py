"""
Exports verified annotations to the standard Ultralytics YOLO dataset format:

  output_dir/
    images/  *.png
    labels/  *.txt   (one line per box: class_id cx cy w h, normalized 0–1)
    data.yaml
"""

from __future__ import annotations

import os
import shutil

import cv2
import yaml

from app.core.sam3_handler import AnnotationStore, BBox, FrameAnnotation


def _bbox_to_yolo(bbox: BBox, img_w: int, img_h: int) -> str:
    cx = ((bbox.x1 + bbox.x2) / 2.0) / img_w
    cy = ((bbox.y1 + bbox.y2) / 2.0) / img_h
    w  = (bbox.x2 - bbox.x1) / img_w
    h  = (bbox.y2 - bbox.y1) / img_h
    # clamp to [0, 1] to handle edge cases from manual drawing
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w  = max(0.0, min(1.0, w))
    h  = max(0.0, min(1.0, h))
    return f"{bbox.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def export_dataset(
    store: AnnotationStore,
    class_names: list[str],
    output_dir: str,
    skip_unverified: bool = False,
) -> int:
    """
    Write YOLO dataset to output_dir.  Returns the number of frames exported.
    Frames with status "exported" are re-exported (idempotent).
    If skip_unverified is True, frames still in "pending" status are skipped.
    """
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    exported_count = 0

    for frame_index, annotation in sorted(store.items()):
        if skip_unverified and annotation.status == "pending":
            continue
        if not annotation.boxes:
            continue
        if not os.path.isfile(annotation.image_path):
            continue

        img = cv2.imread(annotation.image_path)
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        stem = f"frame_{frame_index:06d}"
        dst_img = os.path.join(images_dir, f"{stem}.png")
        dst_lbl = os.path.join(labels_dir, f"{stem}.txt")

        if annotation.image_path != dst_img:
            shutil.copy2(annotation.image_path, dst_img)

        lines = [_bbox_to_yolo(b, img_w, img_h) for b in annotation.boxes]
        with open(dst_lbl, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")

        annotation.status = "exported"
        exported_count += 1

    # Write data.yaml
    yaml_path = os.path.join(output_dir, "data.yaml")
    names_dict = {i: name for i, name in enumerate(class_names)} if class_names else {0: "object"}
    data_yaml = {
        "path": os.path.abspath(output_dir),
        "train": "images",
        "val":   "images",
        "nc":    len(names_dict),
        "names": names_dict,
    }
    with open(yaml_path, "w", encoding="utf-8") as fh:
        yaml.dump(data_yaml, fh, default_flow_style=False, sort_keys=False)

    return exported_count
