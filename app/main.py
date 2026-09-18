"""Application entry point — starts the local web server and opens the UI."""

from __future__ import annotations

import argparse
import socket
import threading
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


def _free_port(host: str, preferred: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, preferred))
            return preferred
        except OSError:
            pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="ModelTrainer — annotate and train in your browser")
    parser.add_argument("--host", default="127.0.0.1", help="interface to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8321, help="port to serve on (default: 8321)")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    args = parser.parse_args()

    _print_torch_info()

    import uvicorn

    from app.server.api import app

    port = _free_port(args.host, args.port)
    url = f"http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{port}"

    print(f"\nModelTrainer is running at {url}\nPress Ctrl+C to stop.\n")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    uvicorn.run(app, host=args.host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
