"""
Media import: images are copied directly; videos are decoded frame-by-frame
via OpenCV and saved as PNGs in the session directory.

Pure Python + OpenCV — progress and per-frame results are delivered through
callbacks so any front-end (web server, CLI, tests) can drive it.
"""

from __future__ import annotations

import math
import os
import shutil
from typing import Callable

import cv2

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".m4v", ".webm"}

# (frame_index, png_path, source_path)
FrameCallback = Callable[[int, str, str], None]
# (current, total)
ProgressCallback = Callable[[int, int], None]
AbortCallback = Callable[[], bool]


def classify(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    return "unknown"


def is_video(path: str) -> bool:
    return classify(path) == "video"


def collect_media(source_dir: str) -> list[str]:
    paths: list[str] = []
    for root, _, files in os.walk(source_dir):
        for name in sorted(files):
            full = os.path.join(root, name)
            if classify(full) in ("image", "video"):
                paths.append(full)
    return paths


def _video_frame_count(path: str) -> int:
    cap = cv2.VideoCapture(path)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(count, 0)


def _expand(source_paths: list[str]) -> list[str]:
    media: list[str] = []
    for path in source_paths:
        if os.path.isdir(path):
            media.extend(collect_media(path))
        elif classify(path) in ("image", "video"):
            media.append(path)
    return media


def count_frames(source_paths: list[str], import_stride: int = 1) -> int:
    """Estimate how many frames an import would produce (for the UI preview)."""
    stride = max(1, import_stride)
    total = 0
    for path in _expand(source_paths):
        if classify(path) == "image":
            total += 1
        else:
            total += math.ceil(_video_frame_count(path) / stride)
    return total


def import_media(
    source_paths: list[str],
    output_dir: str,
    *,
    import_stride: int = 1,
    frame_offset: int = 0,
    on_frame: FrameCallback,
    on_progress: ProgressCallback | None = None,
    should_abort: AbortCallback | None = None,
    on_error: Callable[[str], None] | None = None,
) -> int:
    """
    Decode every image and video frame found under *source_paths* into
    ``output_dir/frames`` as PNGs, calling *on_frame* for each one.

    ``import_stride`` skips video frames (5 keeps frames 0, 5, 10, …); images
    are never skipped.  ``frame_offset`` shifts the first emitted index so an
    append doesn't overwrite an existing session.

    Returns the number of frames written.  Aborts cleanly (returning what was
    written so far) as soon as *should_abort* returns True.
    """
    stride = max(1, import_stride)
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    media = _expand(source_paths)
    if not media:
        raise FileNotFoundError("No supported images or videos found in the selected path.")

    total = count_frames(source_paths, stride)
    frame_index = max(0, frame_offset)
    written = 0

    def aborted() -> bool:
        return bool(should_abort and should_abort())

    for path in media:
        if aborted():
            break
        try:
            if classify(path) == "image":
                frame_index, n = _import_image(path, frame_index, frames_dir, on_frame)
            else:
                frame_index, n = _import_video(
                    path, frame_index, frames_dir, stride, on_frame, aborted
                )
            written += n
        except Exception as exc:  # one bad file must not kill the whole import
            if on_error:
                on_error(f"Could not read {os.path.basename(path)}: {exc}")
        if on_progress:
            on_progress(written, total)

    return written


def _import_image(
    path: str, frame_index: int, frames_dir: str, on_frame: FrameCallback
) -> tuple[int, int]:
    dst = os.path.join(frames_dir, f"frame_{frame_index:06d}.png")
    if path.lower().endswith(".png"):
        shutil.copy2(path, dst)
    else:
        img = cv2.imread(path)
        if img is None:
            raise ValueError("unsupported or corrupt image")
        cv2.imwrite(dst, img)
    # The original path is passed through so callers can match label files by stem.
    on_frame(frame_index, dst, path)
    return frame_index + 1, 1


def _import_video(
    path: str,
    frame_index: int,
    frames_dir: str,
    stride: int,
    on_frame: FrameCallback,
    aborted: Callable[[], bool],
) -> tuple[int, int]:
    cap = cv2.VideoCapture(path)
    raw_idx = 0
    written = 0
    try:
        while True:
            if aborted():
                break
            ret, frame = cap.read()
            if not ret:
                break
            if raw_idx % stride == 0:
                dst = os.path.join(frames_dir, f"frame_{frame_index:06d}.png")
                cv2.imwrite(dst, frame)
                on_frame(frame_index, dst, path)
                frame_index += 1
                written += 1
            raw_idx += 1
    finally:
        cap.release()
    return frame_index, written
