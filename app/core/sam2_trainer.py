"""
SAM 2 / SAM 2.1 fine-tuning worker.

Fine-tuning strategy
--------------------
* The image encoder (Hiera backbone) is **frozen** — it is extremely well
  pre-trained and far too large to fit in VRAM alongside gradients.
* The mask decoder (and optionally the prompt encoder) are **trained**.
* Each annotated bounding box becomes one training sample:
    prompt  = the bounding box corners
    target  = polygon rasterised to a binary mask, or a filled rectangle
              when no polygon is available.
* Loss: focal + dice (the same combination used in SAM 2's original
  training, weighted equally at 1:1).

The worker emits the same signal shape as YOLOTrainWorker so the existing
progress-bar / status-label wiring in main_window.py works unchanged; the
third value carries average loss per epoch instead of mAP50.
"""

from __future__ import annotations

import os
import random

import numpy as np
from PIL import Image
from qtpy.QtCore import QThread, Signal

from app.core.sam3_handler import AnnotationStore

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

SAM2_MODELS: dict[str, str] = {
    # SAM 2.1 — improved checkpoints (recommended)
    "SAM 2.1 Tiny":      "facebook/sam2.1-hiera-tiny",
    "SAM 2.1 Small":     "facebook/sam2.1-hiera-small",
    "SAM 2.1 Base+":     "facebook/sam2.1-hiera-base-plus",
    "SAM 2.1 Large":     "facebook/sam2.1-hiera-large",
    # SAM 2 — original release
    "SAM 2 Tiny":        "facebook/sam2-hiera-tiny",
    "SAM 2 Small":       "facebook/sam2-hiera-small",
    "SAM 2 Base+":       "facebook/sam2-hiera-base-plus",
    "SAM 2 Large":       "facebook/sam2-hiera-large",
}


def _is_sam2_key(key: str) -> bool:
    return key.startswith("SAM 2")


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def _focal_dice_loss(pred_logits, gt_mask, focal_alpha: float = 0.25,
                     focal_gamma: float = 2.0):
    """Focal loss + dice loss, equally weighted.

    *pred_logits* — raw (un-sigmoid'd) mask tensor, shape (N, H, W)
    *gt_mask*     — binary float tensor, same spatial shape
    """
    import torch
    import torch.nn.functional as F

    pred = pred_logits.float()
    gt   = gt_mask.float()

    # ── Focal loss ────────────────────────────────────────────────
    bce    = F.binary_cross_entropy_with_logits(pred, gt, reduction="none")
    prob   = torch.sigmoid(pred)
    p_t    = prob * gt + (1 - prob) * (1 - gt)
    alpha  = focal_alpha * gt + (1 - focal_alpha) * (1 - gt)
    focal  = (alpha * (1 - p_t) ** focal_gamma * bce).mean()

    # ── Dice loss ─────────────────────────────────────────────────
    prob   = torch.sigmoid(pred)
    inter  = (2 * prob * gt).sum(dim=(-1, -2))
    union  = (prob + gt).sum(dim=(-1, -2)) + 1e-6
    dice   = (1 - inter / union).mean()

    return focal + dice


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _label_to_mask_and_box(
    parts: list[str], img_w: int, img_h: int
) -> tuple[list[float], np.ndarray] | None:
    """Parse one YOLO label line and return (box_xyxy, mask).

    Handles both formats exported by this app:
      • Detection:    class cx cy w h          (5 values)
      • Segmentation: class x1 y1 x2 y2 …     (≥7 values, polygon points)

    Returns None if the line is malformed or produces a degenerate mask.
    """
    import cv2

    if len(parts) < 5:
        return None
    try:
        values = [float(p) for p in parts[1:]]  # skip class_id
    except ValueError:
        return None

    mask = np.zeros((img_h, img_w), dtype=np.float32)

    if len(values) == 4:
        # Detection format: cx cy w h  (normalised)
        cx, cy, w, h = values
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        xi1, yi1 = max(0, int(x1)), max(0, int(y1))
        xi2, yi2 = min(img_w, int(x2)), min(img_h, int(y2))
        mask[yi1:yi2, xi1:xi2] = 1.0
        box = [x1, y1, x2, y2]
    else:
        # Segmentation format: x1 y1 x2 y2 … (normalised polygon)
        pts = np.array(values, dtype=np.float32).reshape(-1, 2)
        pts_px = pts.copy()
        pts_px[:, 0] *= img_w
        pts_px[:, 1] *= img_h
        cv2.fillPoly(mask, [pts_px.astype(np.int32)], 1.0)
        # Derive axis-aligned box from polygon extents
        x1, y1 = float(pts_px[:, 0].min()), float(pts_px[:, 1].min())
        x2, y2 = float(pts_px[:, 0].max()), float(pts_px[:, 1].max())
        box = [x1, y1, x2, y2]

    if mask.sum() < 4:
        return None
    return box, mask


def _collect_samples_from_dir(
    dataset_dir: str,
) -> list[tuple[Image.Image, list[float], np.ndarray]]:
    """Load (PIL image, box_xyxy, mask) samples from an exported YOLO dataset folder.

    Expects the standard layout written by export_dataset():
        <dataset_dir>/images/*.png
        <dataset_dir>/labels/*.txt
    Handles both detection and segmentation label formats.
    """
    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")

    if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
        return []

    samples: list[tuple[Image.Image, list[float], np.ndarray]] = []

    for img_name in sorted(os.listdir(images_dir)):
        stem, ext = os.path.splitext(img_name)
        if ext.lower() not in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
            continue
        img_path = os.path.join(images_dir, img_name)
        lbl_path = os.path.join(labels_dir, stem + ".txt")
        if not os.path.isfile(lbl_path):
            continue
        try:
            pil = Image.open(img_path).convert("RGB")
        except Exception:
            continue
        W, H = pil.size

        with open(lbl_path, encoding="utf-8") as fh:
            for line in fh:
                parts = line.strip().split()
                if not parts:
                    continue
                result = _label_to_mask_and_box(parts, W, H)
                if result is None:
                    continue
                box, mask = result
                samples.append((pil, box, mask))

    return samples


def _collect_samples_from_store(
    store: AnnotationStore,
) -> list[tuple[Image.Image, list[float], np.ndarray]]:
    """Return samples from the in-memory annotation store (legacy path)."""
    import cv2

    samples: list[tuple[Image.Image, list[float], np.ndarray]] = []
    for ann in store.values():
        if not ann.boxes or not os.path.isfile(ann.image_path):
            continue
        try:
            pil = Image.open(ann.image_path).convert("RGB")
        except Exception:
            continue
        W, H = pil.size
        for bbox in ann.boxes:
            box  = [float(bbox.x1), float(bbox.y1),
                    float(bbox.x2), float(bbox.y2)]
            mask = np.zeros((H, W), dtype=np.float32)
            if bbox.polygon:
                pts = np.array(bbox.polygon, dtype=np.float32).reshape(-1, 2)
                pts[:, 0] *= W
                pts[:, 1] *= H
                cv2.fillPoly(mask, [pts.astype(np.int32)], 1.0)
            else:
                x1, y1 = max(0, int(bbox.x1)), max(0, int(bbox.y1))
                x2, y2 = min(W, int(bbox.x2)), min(H, int(bbox.y2))
                mask[y1:y2, x1:x2] = 1.0
            if mask.sum() < 4:
                continue
            samples.append((pil, box, mask))
    return samples


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

class SAM2TrainWorker(QThread):
    """Fine-tune a SAM 2 model on an exported YOLO dataset folder.

    Pass *dataset_dir* to read images + label files from disk (the standard
    path when training from an exported dataset).  Pass *store* as a fallback
    to train directly from the in-memory annotation store.

    Signals
    -------
    epoch_done(epoch, total_epochs, avg_loss)
        Emitted after each epoch.  avg_loss replaces the mAP50 value used
        by YOLO workers so the same progress-bar wiring can be reused.
    finished(save_dir)
        Emitted on success with the directory where the model was saved.
    error(message)
        Emitted on any fatal error.
    """

    epoch_done = Signal(int, int, float)   # epoch, total_epochs, avg_loss
    finished   = Signal(str)               # checkpoint save directory
    error      = Signal(str)

    def __init__(
        self,
        model_key: str,
        dataset_dir: str | None = None,
        store: AnnotationStore | None = None,
        epochs: int = 10,
        lr: float = 1e-5,
        output_dir: str = "runs/sam2_finetune",
        parent=None,
    ):
        super().__init__(parent)
        if model_key not in SAM2_MODELS:
            raise ValueError(f"Unknown SAM 2 model key: {model_key!r}")
        if dataset_dir is None and store is None:
            raise ValueError("Provide either dataset_dir or store.")
        self.model_id    = SAM2_MODELS[model_key]
        self.dataset_dir = dataset_dir
        self.store       = store
        self.epochs      = epochs
        self.lr          = lr
        self.output_dir  = output_dir
        self._abort      = False

    def abort(self) -> None:
        self._abort = True

    # ------------------------------------------------------------------
    def run(self) -> None:
        try:
            self._train()
        except Exception as exc:
            import traceback
            self.error.emit(f"SAM 2 training failed: {exc}\n{traceback.format_exc()}")

    # ------------------------------------------------------------------
    def _train(self) -> None:
        import torch
        import torch.nn.functional as F
        from torch.optim import AdamW
        from transformers import AutoModel, AutoProcessor

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # ── Load model & processor ────────────────────────────────
        # All published facebook/sam2* checkpoints have model_type="sam2_video",
        # not "sam2".  AutoModel reads the config and picks the right class
        # automatically, avoiding the "loading sam2_video into Sam2Model" crash.
        try:
            processor = AutoProcessor.from_pretrained(
                self.model_id, local_files_only=True
            )
            model = AutoModel.from_pretrained(
                self.model_id, local_files_only=True
            )
        except Exception:
            # Not cached — download
            processor = AutoProcessor.from_pretrained(self.model_id)
            model     = AutoModel.from_pretrained(self.model_id)

        model = model.to(device).train()

        # ── Freeze image encoder ──────────────────────────────────
        # The Hiera backbone is called 'vision_encoder' in most HF builds;
        # fall back to 'image_encoder' for older checkpoints.
        enc = getattr(model, "vision_encoder",
                      getattr(model, "image_encoder", None))
        if enc is not None:
            for p in enc.parameters():
                p.requires_grad = False
        # Also freeze memory modules (not used for single-image fine-tuning)
        for attr in ("memory_attention", "memory_encoder"):
            mod = getattr(model, attr, None)
            if mod is not None:
                for p in mod.parameters():
                    p.requires_grad = False

        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            self.error.emit(
                "SAM 2: no trainable parameters found after freezing the "
                "image encoder. Check the model architecture."
            )
            return

        optimizer = AdamW(trainable, lr=self.lr, weight_decay=1e-4)

        # ── Build dataset ─────────────────────────────────────────
        if self.dataset_dir:
            samples = _collect_samples_from_dir(self.dataset_dir)
            if not samples:
                self.error.emit(
                    f"SAM 2 training: no labelled images found in:\n"
                    f"  {self.dataset_dir}\n\n"
                    f"Make sure the folder contains images/ and labels/ "
                    f"sub-directories (export your annotations first)."
                )
                return
        else:
            samples = _collect_samples_from_store(self.store)
            if not samples:
                self.error.emit(
                    "SAM 2 training: no valid annotated frames found in the "
                    "annotation store. Annotate some frames first."
                )
                return

        os.makedirs(self.output_dir, exist_ok=True)

        # ── Training loop ─────────────────────────────────────────
        for epoch in range(1, self.epochs + 1):
            if self._abort:
                break

            random.shuffle(samples)
            epoch_loss = 0.0

            for pil_img, box, gt_mask_np in samples:
                if self._abort:
                    break

                try:
                    # Process image + box prompt.
                    # Sam2Processor expects input_boxes as
                    # [[[x1, y1, x2, y2]]] — batch × images × boxes × coords
                    inputs = processor(
                        images=pil_img,
                        input_boxes=[[[box]]],
                        return_tensors="pt",
                    )
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                    # Ground-truth mask: (1, H, W)
                    gt = torch.from_numpy(gt_mask_np).unsqueeze(0).to(device)

                    # Forward pass (single mask output for clean gradient)
                    outputs = model(**inputs, multimask_output=False)

                    # pred_masks shape varies: (B, num_obj, num_masks, H, W)
                    # or (B, num_masks, H, W) — normalise to (N, H, W)
                    pred = outputs.pred_masks
                    while pred.dim() > 3:
                        pred = pred.squeeze(0)   # remove batch + object dims
                    # pred is now (num_masks, H, W); take the first (only) mask
                    pred = pred[0:1]             # (1, H, W)

                    # Resize GT to match predicted spatial dimensions
                    pred_h, pred_w = pred.shape[-2], pred.shape[-1]
                    gt_r = F.interpolate(
                        gt.unsqueeze(0),
                        size=(pred_h, pred_w),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)  # (1, H, W)

                    loss = _focal_dice_loss(pred, gt_r)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                except Exception as sample_exc:
                    # Skip bad samples; don't abort the whole epoch
                    self.error.emit(
                        f"SAM 2: skipping sample — {sample_exc}"
                    )

                finally:
                    # Release GPU memory between samples to avoid OOM
                    try:
                        del inputs, outputs, pred, gt, gt_r, loss
                    except Exception:
                        pass
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            avg_loss = epoch_loss / max(len(samples), 1)
            self.epoch_done.emit(epoch, self.epochs, avg_loss)

        # ── Save checkpoint ───────────────────────────────────────
        save_dir = os.path.join(self.output_dir, "finetuned_sam2")
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        processor.save_pretrained(save_dir)
        self.finished.emit(save_dir)
