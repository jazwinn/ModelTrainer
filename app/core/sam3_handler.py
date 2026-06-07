"""
SAM 3 / SAM 3.1 inference handler — Promptable Concept Segmentation (PCS).

Unlike SAM 2 (point-grid mask generation), SAM 3 segments by *concept*:
  - a TEXT phrase  (e.g. "car", "yellow school bus")  → segments ALL matches
  - POSITIVE box exemplars (label 1) → "find more things like this"
  - NEGATIVE box exemplars (label 0) → "but exclude things like this"
in a single forward pass, returning instance boxes for every matching object.

transformers integration loads through `facebook/sam3` (Sam3Model / Sam3Processor).
The `facebook/sam3.1` repo ships improved checkpoints but no transformers config,
so we load via `facebook/sam3` and fall back gracefully.

Prompt label conventions (SAM 3):
  1  = positive (include)
  0  = negative (exclude)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from qtpy.QtCore import QThread, Signal


# ---------------------------------------------------------------------------
# Shared data model
# ---------------------------------------------------------------------------

@dataclass
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int = 0
    source: str = "sam"  # "sam" | "manual" | "dataset"
    polygon: list[float] | None = None  # flat [x1,y1,...] normalized 0–1 for seg labels

    def to_dict(self) -> dict:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2,
                "class_id": self.class_id, "source": self.source,
                "polygon": self.polygon}

    @staticmethod
    def from_dict(d: dict) -> "BBox":
        return BBox(x1=d["x1"], y1=d["y1"], x2=d["x2"], y2=d["y2"],
                    class_id=d.get("class_id", 0), source=d.get("source", "sam"),
                    polygon=d.get("polygon"))


@dataclass
class FrameAnnotation:
    frame_index: int
    image_path: str
    boxes: list[BBox] = field(default_factory=list)
    status: str = "pending"   # "pending" | "verified" | "exported"
    source_video: str = ""    # absolute path of the source video (empty for still images)


AnnotationStore = dict[int, FrameAnnotation]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

SAM_MODELS = {
    "SAM 3.1 (facebook/sam3.1)": "facebook/sam3.1",
    "SAM 3 (facebook/sam3)":     "facebook/sam3",
}

# Transformers integration is published under facebook/sam3; sam3.1 is a
# checkpoint-only repo.  If a selected id can't be loaded we fall back to this.
_FALLBACK_MODEL_ID = "facebook/sam3"


def _load_sam3(model_id: str, local_only: bool):
    """Load a SAM 3 model + processor, falling back to facebook/sam3."""
    from transformers import Sam3Model, Sam3Processor
    import torch

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    def _try(mid: str):
        proc = Sam3Processor.from_pretrained(mid, local_files_only=local_only)
        mdl = Sam3Model.from_pretrained(
            mid, local_files_only=local_only, torch_dtype=dtype
        )
        return mdl, proc

    try:
        return _try(model_id)
    except Exception:
        if model_id != _FALLBACK_MODEL_ID:
            return _try(_FALLBACK_MODEL_ID)
        raise


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def _model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        import torch
        return torch.device("cpu")


def _mask_to_bbox(mask) -> tuple[int, int, int, int] | None:
    """Tight bounding box (x1, y1, x2, y2) of a boolean mask, or None if empty."""
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2 or not m.any():
        return None
    rows = np.any(m, axis=1)
    cols = np.any(m, axis=0)
    ys = np.where(rows)[0]
    xs = np.where(cols)[0]
    return int(xs[0]), int(ys[0]), int(xs[-1]), int(ys[-1])


def _run_sam3(
    model,
    processor,
    pil_img: Image.Image,
    *,
    text: str | None = None,
    pos_boxes: list[tuple[float, float, float, float]] | None = None,
    neg_boxes: list[tuple[float, float, float, float]] | None = None,
    class_id: int = 0,
    threshold: float = 0.5,
) -> list[BBox]:
    """
    Run SAM 3 concept segmentation on a single image.

    Returns one BBox per matching instance.  Requires at least a text prompt
    or one positive exemplar box (otherwise SAM 3 has no concept to find).
    """
    import torch

    device = _model_device(model)
    W, H = pil_img.size

    proc_kwargs: dict = dict(images=pil_img, return_tensors="pt")
    if text:
        proc_kwargs["text"] = text

    boxes_xyxy: list[list[float]] = []
    labels: list[int] = []
    for b in (pos_boxes or []):
        boxes_xyxy.append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])
        labels.append(1)
    for b in (neg_boxes or []):
        boxes_xyxy.append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])
        labels.append(0)
    if boxes_xyxy:
        proc_kwargs["input_boxes"] = [boxes_xyxy]
        proc_kwargs["input_boxes_labels"] = [labels]

    if not text and not any(lbl == 1 for lbl in labels):
        return []  # nothing to segment

    inputs = processor(**proc_kwargs)
    # Move to device; only convert float tensors to model's dtype (keep integers as-is)
    model_dtype = next(model.parameters()).dtype
    inputs = {
        k: (v.to(device, dtype=model_dtype) if hasattr(v, "to") and v.dtype.is_floating_point else
            v.to(device) if hasattr(v, "to") else v)
        for k, v in inputs.items()
    }

    with torch.inference_mode():
        outputs = model(**inputs)

    target_sizes = inputs.get("original_sizes")
    if hasattr(target_sizes, "tolist"):
        target_sizes = target_sizes.tolist()
    if not target_sizes:
        target_sizes = [[H, W]]

    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=0.5,
        target_sizes=target_sizes,
    )[0]

    out: list[BBox] = []
    boxes = results.get("boxes")
    if boxes is None:
        return out
    for box in boxes:
        coords = box.tolist() if hasattr(box, "tolist") else list(box)
        x1, y1, x2, y2 = (float(coords[0]), float(coords[1]),
                          float(coords[2]), float(coords[3]))
        # clamp to image bounds
        x1 = max(0.0, min(W, x1)); x2 = max(0.0, min(W, x2))
        y1 = max(0.0, min(H, y1)); y2 = max(0.0, min(H, y2))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        out.append(BBox(x1=x1, y1=y1, x2=x2, y2=y2, class_id=class_id, source="sam"))
    return out


def _run_sam3_batch_text(
    model,
    processor,
    pil_img: Image.Image,
    texts: list[str],
    class_ids: list[int],
    threshold: float = 0.5,
) -> list[BBox]:
    """Batch process multiple text concepts for a single image."""
    import torch

    device = _model_device(model)
    W, H = pil_img.size

    # SAM 3 processor accepts a list of lists of strings for batched text
    inputs = processor(images=pil_img, text=[texts], return_tensors="pt")
    # Move to device; only convert float tensors to model's dtype (keep integers as-is)
    model_dtype = next(model.parameters()).dtype
    inputs = {
        k: (v.to(device, dtype=model_dtype) if hasattr(v, "to") and v.dtype.is_floating_point else
            v.to(device) if hasattr(v, "to") else v)
        for k, v in inputs.items()
    }

    with torch.inference_mode():
        outputs = model(**inputs)

    target_sizes = [[H, W]]
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=0.5,
        target_sizes=target_sizes,
    )[0]

    out: list[BBox] = []
    boxes = results.get("boxes")
    labels = results.get("labels") # corresponds to indices of the text array
    if boxes is None or labels is None:
        return out

    for i, box in enumerate(boxes):
        label_idx = int(labels[i])
        if label_idx < 0 or label_idx >= len(class_ids):
            continue
        cid = class_ids[label_idx]
        
        coords = box.tolist() if hasattr(box, "tolist") else list(box)
        x1, y1, x2, y2 = (float(coords[0]), float(coords[1]),
                          float(coords[2]), float(coords[3]))
        x1 = max(0.0, min(W, x1)); x2 = max(0.0, min(W, x2))
        y1 = max(0.0, min(H, y1)); y2 = max(0.0, min(H, y2))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        out.append(BBox(x1=x1, y1=y1, x2=x2, y2=y2, class_id=cid, source="sam"))
    return out



# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

class SAM3LoadWorker(QThread):
    """Loads a SAM 3 model once; the (model, processor) are reused by both
    the text bulk worker and the positive/negative prompt worker."""

    loaded = Signal(object, object)  # model, processor
    error  = Signal(str)

    def __init__(self, model_key: str, parent=None):
        super().__init__(parent)
        self.model_key = model_key

    def run(self) -> None:
        import torch

        device   = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = SAM_MODELS.get(self.model_key, _FALLBACK_MODEL_ID)
        try:
            model, processor = _load_sam3(model_id, local_only=True)
        except Exception:
            try:
                model, processor = _load_sam3(model_id, local_only=False)
            except Exception as exc:
                self.error.emit(f"SAM 3 load failed: {exc}")
                return
        model = model.to(device).eval()
        self.loaded.emit(model, processor)


class SAM3TextWorker(QThread):
    """Bulk auto-label by concept across many frames.

    `concepts` is a list of (text, class_id) pairs.  Every frame is segmented
    once per concept and the results merged — this is how 'label the first
    frame, auto-label the rest' works for loose images: the class names you
    assigned become the concepts SAM 3 looks for everywhere.
    """

    boxes_ready = Signal(int, list)  # frame_index, list[BBox]
    progress    = Signal(int, int)   # done, total
    finished    = Signal()
    error       = Signal(str)

    def __init__(self, model, processor, frame_paths: list[tuple[int, str]],
                 concepts: list[tuple[str, int]], threshold: float = 0.5, parent=None):
        super().__init__(parent)
        self.model       = model
        self.processor   = processor
        self.frame_paths = frame_paths
        self.concepts    = concepts
        self.threshold   = threshold
        self._abort      = False

    def abort(self) -> None:
        self._abort = True

    def run(self) -> None:
        total = len(self.frame_paths)
        for done, (frame_index, png_path) in enumerate(self.frame_paths, start=1):
            if self._abort:
                break
            try:
                pil_img = Image.open(png_path).convert("RGB")
                boxes: list[BBox] = []
                for text, class_id in self.concepts:
                    if not text:
                        continue
                    boxes.extend(_run_sam3(
                        self.model, self.processor, pil_img,
                        text=text, class_id=class_id, threshold=self.threshold,
                    ))
                self.boxes_ready.emit(frame_index, boxes)
            except Exception as exc:
                self.error.emit(f"SAM 3 error on frame {frame_index}: {exc}")
            self.progress.emit(done, total)
        self.finished.emit()


class SAM3PromptWorker(QThread):
    """Interactive positive/negative prompt on a single frame.

    Positive exemplar boxes → 'find more like this'.
    Negative exemplar boxes → 'exclude things like this'.
    Optionally combined with a text concept.
    """

    boxes_ready = Signal(int, list)  # frame_index, list[BBox]
    error       = Signal(str)

    def __init__(self, model, processor, frame_index: int, pil_img: Image.Image,
                 text: str | None, pos_boxes, neg_boxes,
                 class_id: int = 0, threshold: float = 0.5, parent=None):
        super().__init__(parent)
        self.model       = model
        self.processor   = processor
        self.frame_index = frame_index
        self.pil_img     = pil_img
        self.text        = text
        self.pos_boxes   = pos_boxes
        self.neg_boxes   = neg_boxes
        self.class_id    = class_id
        self.threshold   = threshold

    def run(self) -> None:
        try:
            boxes = _run_sam3(
                self.model, self.processor, self.pil_img,
                text=self.text, pos_boxes=self.pos_boxes, neg_boxes=self.neg_boxes,
                class_id=self.class_id, threshold=self.threshold,
            )
            self.boxes_ready.emit(self.frame_index, boxes)
        except Exception as exc:
            self.error.emit(f"SAM 3 prompt error: {exc}")


# The video tracker is a distinct model (SAM2-style memory tracker) loaded from
# facebook/sam3 — it propagates the exact objects you box on one frame through
# the rest of an ordered video by visual memory.
_TRACKER_MODEL_ID = "facebook/sam3"


def _load_sam3_tracker(local_only: bool):
    from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor
    processor = Sam3TrackerVideoProcessor.from_pretrained(
        _TRACKER_MODEL_ID, local_files_only=local_only
    )
    model = Sam3TrackerVideoModel.from_pretrained(
        _TRACKER_MODEL_ID, local_files_only=local_only
    )
    return model, processor


class SAM3TrackWorker(QThread):
    """Propagate the boxes drawn on one frame through an ordered video.

    Each seed box becomes a tracked object (with its class_id); SAM 3's memory
    tracker follows it forward through the remaining frames, emitting a box per
    frame per still-visible object.
    """

    boxes_ready = Signal(int, list)    # frame_index, list[BBox]
    progress    = Signal(int, int)     # done, total
    finished    = Signal()
    error       = Signal(str)
    model_ready = Signal(object, object)  # cache tracker (model, processor)

    def __init__(self, frame_paths: list[tuple[int, str]],
                 seed_frames: list[tuple[int, list[tuple[tuple[float, float, float, float], int]]]],
                 start_frame_index: int,
                 max_frames: int | None = None,
                 model=None, processor=None, parent=None):
        super().__init__(parent)
        self.frame_paths       = frame_paths       # sorted [(frame_index, path)]
        self.seed_frames       = seed_frames       # [(frame_index, [((x1,y1,x2,y2), class_id)])]
        self.start_frame_index = start_frame_index
        self.max_frames        = max_frames        # track at most this many frames forward
        self.model             = model
        self.processor         = processor
        self._abort            = False

    def abort(self) -> None:
        self._abort = True

    def run(self) -> None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, processor = self.model, self.processor

        if model is None or processor is None:
            try:
                model, processor = _load_sam3_tracker(local_only=True)
            except Exception:
                try:
                    model, processor = _load_sam3_tracker(local_only=False)
                except Exception as exc:
                    self.error.emit(f"SAM 3 tracker load failed: {exc}")
                    return
            model = model.to(device).eval()
            self.model_ready.emit(model, processor)

        ordered = self.frame_paths
        if not ordered or not self.seed_frames:
            self.finished.emit()
            return

        try:
            frames = [Image.open(p).convert("RGB") for _, p in ordered]
        except Exception as exc:
            self.error.emit(f"Failed to load video frames: {exc}")
            return

        pos_of = {fi: pos for pos, (fi, _) in enumerate(ordered)}
        start_pos = pos_of.get(self.start_frame_index, 0)

        try:
            session = processor.init_video_session(video=frames, inference_device=device)
        except Exception as exc:
            self.error.emit(f"Tracker session init failed: {exc}")
            return

        # Seed objects dynamically as we iterate through frames.
        objid_to_class: dict[int, int] = {}
        class_to_objids: dict[int, list[int]] = {}
        next_obj_id = 1
        
        sorted_seed_frames = sorted(self.seed_frames, key=lambda x: x[0])
        pos_to_boxes = {}
        for fidx, boxes in sorted_seed_frames:
            frame_pos = pos_of.get(fidx)
            if frame_pos is not None:
                pos_to_boxes[frame_pos] = boxes

        remaining = len(ordered) - start_pos
        total = min(remaining, self.max_frames) if self.max_frames else remaining
        end_pos = start_pos + total
        done = 0

        try:
            import torch
            with torch.inference_mode():
                for fpos in range(start_pos, end_pos):
                    if self._abort:
                        break
                        
                    # Inject inputs dynamically exactly when they appear
                    if fpos in pos_to_boxes:
                        class_to_boxes = {}
                        for box, class_id in pos_to_boxes[fpos]:
                            class_to_boxes.setdefault(class_id, []).append(box)
                            
                        frame_obj_ids = []
                        frame_input_boxes = []
                        for cid, bbox_list in class_to_boxes.items():
                            existing_obj_ids = class_to_objids.get(cid, [])
                            while len(existing_obj_ids) < len(bbox_list):
                                new_id = next_obj_id
                                next_obj_id += 1
                                existing_obj_ids.append(new_id)
                                objid_to_class[new_id] = cid
                            class_to_objids[cid] = existing_obj_ids
                            
                            for obj_id, box in zip(existing_obj_ids, bbox_list):
                                frame_obj_ids.append(obj_id)
                                frame_input_boxes.append([float(box[0]), float(box[1]), float(box[2]), float(box[3])])
                                
                        if frame_obj_ids:
                            try:
                                processor.add_inputs_to_inference_session(
                                    inference_session=session,
                                    frame_idx=fpos,
                                    obj_ids=frame_obj_ids,
                                    input_boxes=[frame_input_boxes],
                                )
                            except Exception as exc:
                                self.error.emit(f"Failed to seed objects at frame {ordered[fpos][0]}: {exc}")
                                return
                                
                    if not class_to_objids:
                        continue

                    out = model(session, frame_idx=fpos)
                    
                    frame_index = ordered[fpos][0]
                    res = processor.post_process_masks(
                        [out.pred_masks],
                        original_sizes=[[session.video_height, session.video_width]],
                        binarize=True,
                    )[0]
                    scores = getattr(out, "object_score_logits", None)
                    obj_ids = list(getattr(session, "obj_ids", []))

                    boxes: list[BBox] = []
                    for i in range(res.shape[0]):
                        if scores is not None:
                            try:
                                if float(scores[i]) <= 0:
                                    continue  # object not present in this frame
                            except Exception:
                                pass
                        mask = res[i, 0].cpu().numpy().astype(bool)
                        bb = _mask_to_bbox(mask)
                        if bb is None:
                            continue
                        oid = obj_ids[i] if i < len(obj_ids) else None
                        class_id = objid_to_class.get(oid, 0)
                        x1, y1, x2, y2 = bb
                        boxes.append(BBox(x1=x1, y1=y1, x2=x2, y2=y2,
                                          class_id=class_id, source="sam"))

                    self.boxes_ready.emit(frame_index, boxes)
                    done += 1
                    self.progress.emit(done, total)
        except Exception as exc:
            self.error.emit(f"Tracking failed: {exc}")

        self.finished.emit()
