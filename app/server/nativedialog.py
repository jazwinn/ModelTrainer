"""
Native Windows file and folder pickers.

The interface runs in a browser, and a browser cannot hand back a filesystem
path — but the server is on the same machine, so it can open the real Explorer
dialog and return what was chosen.

This uses the modern IFileOpenDialog (the Explorer window with the address bar,
Quick access and search), falling back to Tk's dialogs elsewhere.  Every entry
point returns None when the user cancels and raises DialogUnavailable when no
desktop dialog can be shown at all — callers then fall back to the in-app
folder browser.
"""

from __future__ import annotations

import os
import sys
import threading
from ctypes import (
    POINTER, Structure, WINFUNCTYPE, byref, c_byte, c_long, c_ulong,
    c_ushort, c_void_p, c_wchar_p, cast,
)

IS_WINDOWS = sys.platform == "win32"

# Only one dialog at a time — two modal windows fighting for focus helps nobody.
_lock = threading.Lock()

# Handle of the hidden window owning the dialog currently on screen, so the app
# can take the dialog back down if the person gives up on it in the browser.
_owner_hwnd = None


class DialogUnavailable(RuntimeError):
    """No desktop dialog can be shown (headless server, non-Windows, COM failure)."""


class DialogBusy(RuntimeError):
    """A dialog is already open — the one on screen has to be dealt with first."""


def in_use() -> bool:
    return _lock.locked()


def available() -> bool:
    """True when this process can put a dialog on someone's screen."""
    if IS_WINDOWS:
        return True
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        try:
            import tkinter  # noqa: F401
            return True
        except Exception:
            return False
    return False


# ---------------------------------------------------------------------------
# Windows COM plumbing
# ---------------------------------------------------------------------------

if IS_WINDOWS:
    from ctypes import windll

    class _GUID(Structure):
        _fields_ = [("Data1", c_ulong), ("Data2", c_ushort),
                    ("Data3", c_ushort), ("Data4", c_byte * 8)]

        def __init__(self, text: str):
            super().__init__()
            windll.ole32.CLSIDFromString(text, byref(self))

    class _FilterSpec(Structure):
        _fields_ = [("pszName", c_wchar_p), ("pszSpec", c_wchar_p)]

    CLSID_FileOpenDialog = "{DC1C5A9C-E88A-4DDE-A5A1-60F82A20AEF7}"
    IID_IFileOpenDialog  = "{D57C7288-D4AD-4768-BE02-9D969532D960}"
    IID_IShellItem       = "{43826D1E-E718-42EE-BC55-A1E261C37BFE}"

    CLSCTX_INPROC_SERVER = 1
    COINIT_APARTMENTTHREADED = 0x2

    FOS_PICKFOLDERS      = 0x00000020
    FOS_FORCEFILESYSTEM  = 0x00000040
    FOS_PATHMUSTEXIST    = 0x00000800
    FOS_FILEMUSTEXIST    = 0x00001000

    SIGDN_FILESYSPATH = 0x80058000
    ERROR_CANCELLED = 0x800704C7

    # IUnknown 0-2 · IModalWindow 3 · IFileDialog 4-26 · IFileOpenDialog 27-28
    _RELEASE      = 2
    _SHOW         = 3
    _SET_FILETYPES = 4
    _SET_OPTIONS  = 9
    _GET_OPTIONS  = 10
    _SET_FOLDER   = 12
    _SET_TITLE    = 17
    _SET_OK_LABEL = 18
    _GET_RESULT   = 20
    # IShellItem
    _GET_DISPLAY_NAME = 5

    def _method(iface: c_void_p, index: int, *argtypes):
        """Bind vtable slot *index* of a COM interface pointer."""
        vtable = cast(iface, POINTER(c_void_p))[0]
        slots = cast(vtable, POINTER(c_void_p))
        proto = WINFUNCTYPE(c_long, c_void_p, *argtypes)
        return proto(slots[index])

    def _release(iface: c_void_p) -> None:
        if iface:
            _method(iface, _RELEASE)(iface)

    windll.user32.CreateWindowExW.restype = c_void_p
    windll.user32.CreateWindowExW.argtypes = [
        c_ulong, c_wchar_p, c_wchar_p, c_ulong,
        c_long, c_long, c_long, c_long,
        c_void_p, c_void_p, c_void_p, c_void_p,
    ]
    windll.user32.DestroyWindow.argtypes = [c_void_p]
    windll.ole32.CoTaskMemFree.argtypes = [c_void_p]

    def _hidden_owner():
        """A zero-size top-most window to own the dialog.

        Without an owner the dialog can open behind the browser, which looks
        like the app has frozen; a top-most owner keeps it visible even though
        the browser still holds focus.
        """
        try:
            hwnd = windll.user32.CreateWindowExW(
                0x00000008 | 0x00000080,   # WS_EX_TOPMOST | WS_EX_TOOLWINDOW
                "STATIC", None, 0x80000000,  # WS_POPUP
                0, 0, 0, 0, None, None, None, None,
            )
            return hwnd or None
        except Exception:
            return None

    def _shell_item_path(item: c_void_p) -> str | None:
        buffer = c_wchar_p()
        hr = _method(item, _GET_DISPLAY_NAME, c_long, POINTER(c_wchar_p))(
            item, SIGDN_FILESYSPATH, byref(buffer)
        )
        if hr < 0 or not buffer.value:
            return None
        path = buffer.value
        windll.ole32.CoTaskMemFree(buffer)
        return path

    def _run_dialog(
        *, title: str, start: str | None, folders: bool,
        filters: list[tuple[str, str]] | None, ok_label: str,
    ) -> str | None:
        hr = windll.ole32.CoInitializeEx(None, COINIT_APARTMENTTHREADED)
        # S_FALSE (1) means this thread was already initialised — still ours to
        # uninitialise, anything negative means we never got an apartment.
        if hr < 0:
            raise DialogUnavailable("COM could not be initialised")

        dialog = c_void_p()
        owner = None
        try:
            hr = windll.ole32.CoCreateInstance(
                byref(_GUID(CLSID_FileOpenDialog)), None, CLSCTX_INPROC_SERVER,
                byref(_GUID(IID_IFileOpenDialog)), byref(dialog),
            )
            if hr < 0 or not dialog:
                raise DialogUnavailable("the Explorer dialog is not available")

            options = c_ulong()
            _method(dialog, _GET_OPTIONS, POINTER(c_ulong))(dialog, byref(options))
            flags = options.value | FOS_FORCEFILESYSTEM | FOS_PATHMUSTEXIST
            flags |= FOS_PICKFOLDERS if folders else FOS_FILEMUSTEXIST
            _method(dialog, _SET_OPTIONS, c_ulong)(dialog, flags)

            _method(dialog, _SET_TITLE, c_wchar_p)(dialog, title)
            if ok_label:
                _method(dialog, _SET_OK_LABEL, c_wchar_p)(dialog, ok_label)

            if filters:
                specs = (_FilterSpec * len(filters))(
                    *[_FilterSpec(name, spec) for name, spec in filters]
                )
                _method(dialog, _SET_FILETYPES, c_ulong, POINTER(_FilterSpec))(
                    dialog, len(filters), specs
                )

            if start and os.path.isdir(start):
                folder = c_void_p()
                hr = windll.shell32.SHCreateItemFromParsingName(
                    c_wchar_p(os.path.abspath(start)), None,
                    byref(_GUID(IID_IShellItem)), byref(folder),
                )
                if hr >= 0 and folder:
                    _method(dialog, _SET_FOLDER, c_void_p)(dialog, folder)
                    _release(folder)

            owner = _hidden_owner()
            global _owner_hwnd
            _owner_hwnd = owner
            hr = _method(dialog, _SHOW, c_void_p)(dialog, owner)
            if (hr & 0xFFFFFFFF) == ERROR_CANCELLED:
                return None
            if hr < 0:
                raise DialogUnavailable(f"the dialog closed with error 0x{hr & 0xFFFFFFFF:08X}")

            item = c_void_p()
            hr = _method(dialog, _GET_RESULT, POINTER(c_void_p))(dialog, byref(item))
            if hr < 0 or not item:
                return None
            try:
                return _shell_item_path(item)
            finally:
                _release(item)
        finally:
            _owner_hwnd = None
            _release(dialog)
            if owner:
                windll.user32.DestroyWindow(owner)
            windll.ole32.CoUninitialize()


# ---------------------------------------------------------------------------
# Tk fallback (non-Windows desktops)
# ---------------------------------------------------------------------------

def cancel_open() -> bool:
    """Close the dialog currently on screen, as if Cancel had been pressed.

    Finds it by ownership rather than by title, so it can only ever close the
    window this process put up.
    """
    if not IS_WINDOWS or not _owner_hwnd:
        return False

    from ctypes import WINFUNCTYPE, c_bool

    found: list[int] = []
    owner = _owner_hwnd

    @WINFUNCTYPE(c_bool, c_void_p, c_void_p)
    def _visit(hwnd, _param):
        windll.user32.GetWindow.restype = c_void_p
        windll.user32.GetWindow.argtypes = [c_void_p, c_ulong]
        if windll.user32.GetWindow(hwnd, 4) == owner:   # GW_OWNER
            found.append(hwnd)
            return False
        return True

    windll.user32.EnumWindows.argtypes = [type(_visit), c_void_p]
    windll.user32.EnumWindows(_visit, None)
    if not found:
        return False

    windll.user32.PostMessageW.argtypes = [c_void_p, c_ulong, c_void_p, c_void_p]
    windll.user32.PostMessageW(found[0], 0x0010, None, None)   # WM_CLOSE
    return True


def _run_tk(*, title: str, start: str | None, folders: bool,
            filters: list[tuple[str, str]] | None) -> str | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        raise DialogUnavailable(f"no desktop toolkit available ({exc})") from exc

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        if folders:
            chosen = filedialog.askdirectory(title=title, initialdir=start or None)
        else:
            chosen = filedialog.askopenfilename(
                title=title, initialdir=start or None,
                filetypes=[(name, spec) for name, spec in (filters or [])] or None,
            )
    finally:
        root.destroy()
    return chosen or None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def pick(
    *,
    folders: bool = True,
    title: str = "Select a folder",
    start: str | None = None,
    filters: list[tuple[str, str]] | None = None,
    ok_label: str = "",
) -> str | None:
    """Show a native picker and return the chosen path, or None if cancelled.

    Blocks the calling thread until the dialog closes, so call it from a worker
    thread rather than an event loop.
    """
    if not available():
        raise DialogUnavailable("this machine has no desktop to show a dialog on")

    if not _lock.acquire(blocking=False):
        raise DialogBusy("a file dialog is already open")
    try:
        if IS_WINDOWS:
            try:
                return _run_dialog(title=title, start=start, folders=folders,
                                   filters=filters, ok_label=ok_label)
            except DialogUnavailable:
                raise
            except Exception as exc:
                raise DialogUnavailable(f"the Windows dialog failed ({exc})") from exc
        return _run_tk(title=title, start=start, folders=folders, filters=filters)
    finally:
        _lock.release()
