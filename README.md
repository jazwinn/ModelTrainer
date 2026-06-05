# Model Trainer

A powerful, native desktop application for dataset annotation and computer vision model training. Model Trainer integrates Meta's state-of-the-art **SAM 3** (Segment Anything Model 3) for zero-shot auto-labeling and visual tracking, along with **Ultralytics YOLO** for seamless, localized model training—all within a single, hardware-accelerated GUI.

## Features

* **Zero-Shot Auto-Labeling:** Leverage SAM 3's text-to-image capabilities to automatically find and bound objects across your entire dataset simply by typing a description (e.g., "car", "solar panel").
* **SAM 3 Video Tracking:** Label a single frame and use SAM 3's temporal memory to visually track and propagate the bounding box across continuous video frames.
* **Interactive Prompting:** Use point-and-click positive (green) and negative (red) visual hints to interactively guide SAM 3 to segment exact, complex objects.
* **Manual Annotation:** Fast and intuitive manual bounding box drawing with full resize and drag support.
* **YOLO Dataset Export:** Automatically format and export your labeled frames into standard YOLO format with a generated `data.yaml`.
* **Embedded YOLO Training:** Train YOLOv8, YOLOv10, or YOLO11 models directly inside the application. Features automatic VRAM batch scaling (`batch=-1`), dataset caching, and real-time training progress UI.
* **GPU Acceleration:** Fully utilizes CUDA for blazing fast SAM 3 inference and YOLO training.

## Requirements

* Python 3.10+
* A CUDA-capable NVIDIA GPU is highly recommended for reasonable SAM 3 and YOLO performance.
* Dependencies listed in `requirements.txt`.

## Installation

1. Clone this repository.
2. Create a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: Ensure you install the CUDA version of PyTorch if you are using an NVIDIA GPU for hardware acceleration).*

## Usage

Start the application by running the main entry point:
```bash
python -m app.main
```

### Typical Workflow
1. **Import Media:** Import a folder of images or videos. Videos are automatically extracted into individual frames.
2. **Define Classes:** Use "Edit Classes" to add the categories you want to detect (e.g., "drone", "car").
3. **Annotate:**
   - Use **SAM 3 Prompt** to interactively bound a few examples.
   - Use **Propagate Labels** (for disjoint images) or **Track Video** (for continuous video) to let the AI auto-annotate the rest of your frames.
4. **Export & Train:** Click **Export YOLO Dataset**, choose your YOLO model architecture (e.g., YOLO11s), set your epochs, and click **Train**.

## Architecture

* **UI Layer:** PyQt/PySide6
* **Inference Layer:** Hugging Face `transformers` (SAM 3, SAM 3 Video Tracker)
* **Training Layer:** `ultralytics` YOLO API

## License

MIT License
