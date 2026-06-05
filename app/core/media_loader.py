"""
Async media loading: images are copied directly; videos are decoded
frame-by-frame via OpenCV and saved as PNGs in a temp directory.
"""

from __future__ import annotations

import os
import shutil

import cv2
from qtpy.QtCore import QThread, Signal

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".m4v", ".webm"}


def _classify(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    return "unknown"


def _collect_media(source_dir: str) -> list[str]:
    paths = []
    for root, _, files in os.walk(source_dir):
        for f in sorted(files):
            full = os.path.join(root, f)
            if _classify(full) in ("image", "video"):
                paths.append(full)
    return paths


def _video_frame_count(path: str) -> int:
    cap = cv2.VideoCapture(path)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(count, 0)


class MediaLoaderWorker(QThread):
    """
    Scans source_paths (files or a single directory), decodes every image and
    video frame, saves PNGs to output_dir/frames/, and emits frame_ready for
    each one so the UI can populate the thumbnail list progressively.
    """

    frame_ready = Signal(int, str)   # (frame_index, png_path)
    progress    = Signal(int, int)   # (current, total)
    finished    = Signal()
    error       = Signal(str)

    def __init__(
        self,
        source_paths: list[str],
        output_dir: str,
        parent=None,
    ):
        super().__init__(parent)
        self.source_paths = source_paths
        self.output_dir = output_dir
        self._abort = False

    def abort(self) -> None:
        self._abort = True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _total_frames(self, media: list[str]) -> int:
        total = 0
        for p in media:
            if _classify(p) == "image":
                total += 1
            else:
                total += _video_frame_count(p)
        return total

    def _emit_image(self, path: str, frame_index: int, frames_dir: str) -> int:
        dst = os.path.join(frames_dir, f"frame_{frame_index:06d}.png")
        if path.lower().endswith(".png"):
            shutil.copy2(path, dst)
        else:
            import cv2 as _cv2
            img = _cv2.imread(path)
            if img is not None:
                _cv2.imwrite(dst, img)
        self.frame_ready.emit(frame_index, dst)
        return frame_index + 1

    def _emit_video(self, path: str, frame_index: int, frames_dir: str) -> int:
        cap = cv2.VideoCapture(path)
        while True:
            if self._abort:
                break
            ret, frame = cap.read()
            if not ret:
                break
            dst = os.path.join(frames_dir, f"frame_{frame_index:06d}.png")
            cv2.imwrite(dst, frame)
            self.frame_ready.emit(frame_index, dst)
            frame_index += 1
        cap.release()
        return frame_index

    # ------------------------------------------------------------------
    # QThread entry
    # ------------------------------------------------------------------

    def run(self) -> None:
        frames_dir = os.path.join(self.output_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        # Expand single directory argument
        media: list[str] = []
        for p in self.source_paths:
            if os.path.isdir(p):
                media.extend(_collect_media(p))
            elif _classify(p) in ("image", "video"):
                media.append(p)

        if not media:
            self.error.emit("No supported media files found in the selected path.")
            self.finished.emit()
            return

        total = self._total_frames(media)
        frame_index = 0

        for path in media:
            if self._abort:
                break
            try:
                kind = _classify(path)
                if kind == "image":
                    frame_index = self._emit_image(path, frame_index, frames_dir)
                elif kind == "video":
                    frame_index = self._emit_video(path, frame_index, frames_dir)
            except Exception as exc:
                self.error.emit(f"Error loading {os.path.basename(path)}: {exc}")

            self.progress.emit(frame_index, total)

        self.finished.emit()
