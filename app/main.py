"""Application entry point."""

import sys

import torch
from qtpy.QtWidgets import QApplication

from app.ui.main_window import MainWindow


def _print_torch_info() -> None:
    print(f"PyTorch version: {torch.__version__}")
    print(f"Is CUDA available? {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current CUDA device ID: {torch.cuda.current_device()}")
        print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("No CUDA-compatible GPU detected by PyTorch.")


def main() -> None:
    _print_torch_info()
    app = QApplication(sys.argv)
    app.setApplicationName("ModelTrainer")
    app.setOrganizationName("ModelTrainer")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
