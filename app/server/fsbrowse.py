"""
Server-side filesystem browsing.

The UI runs in a browser but the data lives on the machine running the server,
so the old native folder pickers are replaced by a small directory browser
backed by these helpers.
"""

from __future__ import annotations

import os
import string
from pathlib import Path

from app.core.media_loader import IMAGE_EXTS, VIDEO_EXTS
from app.core.yolo_trainer import _PROJECT_ROOT

_MEDIA_EXTS = IMAGE_EXTS | VIDEO_EXTS


def drives() -> list[str]:
    """Drive roots on Windows, "/" elsewhere."""
    if os.name != "nt":
        return ["/"]
    found = []
    for letter in string.ascii_uppercase:
        root = f"{letter}:\\"
        if os.path.exists(root):
            found.append(root)
    return found


def home_places() -> list[dict]:
    """Handy starting points shown in the picker sidebar."""
    home = Path.home()
    places = [
        {"label": "Project", "path": _PROJECT_ROOT},
        {"label": "Home", "path": str(home)},
        {"label": "Desktop", "path": str(home / "Desktop")},
        {"label": "Documents", "path": str(home / "Documents")},
        {"label": "Downloads", "path": str(home / "Downloads")},
    ]
    return [p for p in places if os.path.isdir(p["path"])]


def _describe(path: Path) -> dict:
    """Summarise a directory so the picker can show what is inside it."""
    info = {"images": 0, "videos": 0, "hasDataYaml": False, "hasImagesDir": False,
            "hasLabelsDir": False, "weights": 0}
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                name = entry.name.lower()
                if entry.is_dir():
                    if name == "images":
                        info["hasImagesDir"] = True
                    elif name == "labels":
                        info["hasLabelsDir"] = True
                    continue
                ext = os.path.splitext(name)[1]
                if name == "data.yaml":
                    info["hasDataYaml"] = True
                elif ext in IMAGE_EXTS:
                    info["images"] += 1
                elif ext in VIDEO_EXTS:
                    info["videos"] += 1
                elif ext == ".pt":
                    info["weights"] += 1
    except (PermissionError, OSError):
        pass
    return info


def listdir(path: str | None, *, want: str = "dir") -> dict:
    """
    List directories (and, when want == "file", matching files) under *path*.

    ``want`` is "dir" (folder picker), "media" (folders + images/videos) or
    a file extension such as ".pt".
    """
    target = Path(path).expanduser() if path else Path(_PROJECT_ROOT)
    if not target.is_dir():
        target = Path(_PROJECT_ROOT)
    target = target.resolve()

    dirs: list[dict] = []
    files: list[dict] = []
    error = ""
    try:
        with os.scandir(target) as entries:
            for entry in entries:
                if entry.name.startswith("."):
                    continue
                try:
                    if entry.is_dir():
                        dirs.append({"name": entry.name, "path": entry.path})
                    elif want != "dir":
                        ext = os.path.splitext(entry.name)[1].lower()
                        keep = (ext in _MEDIA_EXTS) if want == "media" else (ext == want)
                        if keep:
                            files.append({
                                "name": entry.name,
                                "path": entry.path,
                                "size": entry.stat().st_size,
                            })
                except OSError:
                    continue
    except PermissionError:
        error = "Permission denied for this folder."
    except OSError as exc:
        error = str(exc)

    dirs.sort(key=lambda d: d["name"].lower())
    files.sort(key=lambda f: f["name"].lower())

    parent = str(target.parent) if target.parent != target else None
    return {
        "path": str(target),
        "parent": parent,
        "dirs": dirs,
        "files": files,
        "info": _describe(target),
        "drives": drives(),
        "places": home_places(),
        "error": error,
    }
