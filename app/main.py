"""Application entry point — starts the local web server and opens the UI."""

from __future__ import annotations

import argparse
import signal
import socket
import threading
import time
import webbrowser


def _print_torch_info() -> None:
    try:
        import torch
    except Exception as exc:  # torch is optional until you actually run a model
        print(f"PyTorch not available: {exc}")
        return

    print(f"PyTorch version: {torch.__version__}")
    print(f"Is CUDA available? {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current CUDA device ID: {torch.cuda.current_device()}")
        print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("No CUDA-compatible GPU detected by PyTorch.")


def _port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def _any_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


def _modeltrainer_at(host: str, port: int) -> bool:
    """True when the thing already holding this port is a ModelTrainer server."""
    import json
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(f"http://{host}:{port}/api/state", timeout=1.5) as response:
            data = json.load(response)
        return isinstance(data, dict) and "frames" in data and "classes" in data
    except (urllib.error.URLError, OSError, ValueError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="ModelTrainer — annotate and train in your browser")
    parser.add_argument("--host", default="127.0.0.1", help="interface to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8321, help="port to serve on (default: 8321)")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    args = parser.parse_args()

    _print_torch_info()

    import uvicorn

    from app.server.api import app

    def address(at: int) -> str:
        shown = "localhost" if args.host in ("127.0.0.1", "0.0.0.0") else args.host
        return f"http://{shown}:{at}"

    port = args.port
    if not _port_is_free(args.host, port):
        # Quietly starting a second server on another port would give it its own
        # view of the same session folder, and the two would overwrite each other.
        if _modeltrainer_at(args.host, port):
            running = address(port)
            print(f"\nModelTrainer is already running at {running} — opening that one.")
            print("Close it first if you meant to start a fresh server.\n")
            if not args.no_browser:
                webbrowser.open(running)
            return
        port = _any_free_port(args.host)
        print(f"\nPort {args.port} is taken by something else — using {port} instead.")

    url = address(port)
    print(f"\nModelTrainer is running at {url}\nPress Ctrl+C to stop.\n")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    _serve(uvicorn, app, args.host, port)


def _serve(uvicorn, app, host: str, port: int) -> None:
    """Run the server, and stop it cleanly on Ctrl+C.

    `uvicorn.run` would do this in one line, but it re-raises the signal it
    caught once the server is already shutting down.  That arrives as a
    KeyboardInterrupt inside the event loop's own teardown, and the tasks it
    cancels on the way are then reported as application errors — so an ordinary
    Ctrl+C printed two pages of traceback.

    uvicorn only installs signal handling when it is on the main thread, so
    running it elsewhere leaves the signal to us and stopping becomes a request
    like any other.
    """
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, name="uvicorn", daemon=True)
    thread.start()

    asked_to_stop = threading.Event()

    def on_signal(signum, frame) -> None:
        if asked_to_stop.is_set():
            print("Forcing it.")
            server.force_exit = True
            return
        print("\nStopping...")
        asked_to_stop.set()
        server.should_exit = True

    # Ctrl+Break as well as Ctrl+C, because uvicorn used to handle both and
    # Python's own SIGBREAK would otherwise kill the process where it stands.
    for name in ("SIGINT", "SIGBREAK", "SIGTERM"):
        sig = getattr(signal, name, None)
        if sig is not None:
            signal.signal(sig, on_signal)

    # Waiting with `thread.join` would hang: on Windows a lock wait is not
    # interruptible, so the handler above would never get to run.  A sleep is.
    while thread.is_alive():
        time.sleep(0.2)
    print("Stopped.")


if __name__ == "__main__":
    main()
