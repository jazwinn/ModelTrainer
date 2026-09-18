"""
Box geometry: merging several boxes into one, and folding away the duplicate
detections that auto-labelling tends to leave behind.

Both the "merge what I selected" and "merge overlapping boxes" features go
through here, so a merged box means exactly the same thing either way.
"""

from __future__ import annotations

from collections import Counter

from app.core.sam3_handler import BBox


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def area(box: BBox) -> float:
    return max(0.0, box.x2 - box.x1) * max(0.0, box.y2 - box.y1)


def intersection(a: BBox, b: BBox) -> float:
    w = min(a.x2, b.x2) - max(a.x1, b.x1)
    h = min(a.y2, b.y2) - max(a.y1, b.y1)
    return w * h if w > 0 and h > 0 else 0.0


def containment(a: BBox, b: BBox) -> float:
    """How much of the smaller box sits inside the larger one, 0–1.

    This is the question people actually ask of a duplicate detection ("is box
    A basically inside box B?"), and unlike IoU it stays high when one box is
    much larger than the other — which is exactly the duplicate case.
    """
    smaller = min(area(a), area(b))
    if smaller <= 0:
        return 0.0
    return intersection(a, b) / smaller


def iou(a: BBox, b: BBox) -> float:
    union = area(a) + area(b) - intersection(a, b)
    return intersection(a, b) / union if union > 0 else 0.0


def convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Andrew's monotone chain hull, used to fuse several polygons into one."""
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------

def merge_boxes(boxes: list[BBox], width: int, height: int) -> BBox:
    """Fuse *boxes* into one covering all of them.

    * The rectangle is the union of every box.
    * The class is the one most of them agree on, falling back to the first.
    * A polygon appears only if at least one box had one; it is the convex hull
      of every contributing outline, so the mask still wraps the whole object.
    * Keypoints appear only if at least one box had them, and become the four
      corners of the merged rectangle.
    * The result counts as hand-made, because a person asked for it.
    """
    if not boxes:
        raise ValueError("Nothing to merge.")
    if len(boxes) == 1:
        return boxes[0]

    x1 = min(b.x1 for b in boxes)
    y1 = min(b.y1 for b in boxes)
    x2 = max(b.x2 for b in boxes)
    y2 = max(b.y2 for b in boxes)

    counts = Counter(b.class_id for b in boxes)
    top = counts.most_common(1)[0][1]
    class_id = next(b.class_id for b in boxes if counts[b.class_id] == top)

    polygon = None
    if any(b.polygon for b in boxes):
        points: list[tuple[float, float]] = []
        for box in boxes:
            if box.polygon:
                flat = box.polygon
                points.extend((flat[i], flat[i + 1]) for i in range(0, len(flat) - 1, 2))
            elif width and height:
                points.extend([
                    (box.x1 / width, box.y1 / height), (box.x2 / width, box.y1 / height),
                    (box.x2 / width, box.y2 / height), (box.x1 / width, box.y2 / height),
                ])
        hull = convex_hull(points)
        if len(hull) >= 3:
            polygon = [coord for point in hull for coord in point]

    keypoints = None
    if any(b.keypoints for b in boxes) and width and height:
        keypoints = [
            (x1 / width, y1 / height), (x2 / width, y1 / height),
            (x2 / width, y2 / height), (x1 / width, y2 / height),
        ]

    scores = [b.score for b in boxes if b.score is not None]
    return BBox(
        x1=x1, y1=y1, x2=x2, y2=y2,
        class_id=class_id, source="manual",
        polygon=polygon, keypoints=keypoints,
        score=max(scores) if scores else None,
    )


def merge_selection(boxes: list[BBox], indices: list[int], width: int, height: int) -> list[BBox]:
    """Replace the boxes at *indices* with a single merged one.

    The merged box takes the place of the first one selected, so it does not
    jump to the end of the list.
    """
    picked = sorted({i for i in indices if 0 <= i < len(boxes)})
    if len(picked) < 2:
        raise ValueError("Select at least two boxes to merge.")

    merged = merge_boxes([boxes[i] for i in picked], width, height)
    out: list[BBox] = []
    for i, box in enumerate(boxes):
        if i == picked[0]:
            out.append(merged)
        elif i not in set(picked):
            out.append(box)
    return out


def merge_overlaps(
    boxes: list[BBox],
    *,
    threshold: float = 0.8,
    same_class_only: bool = True,
    width: int = 0,
    height: int = 0,
) -> tuple[list[BBox], int]:
    """
    Fold boxes that sit on top of each other into one.

    Two boxes join when *threshold* of the smaller one lies inside the other.
    Joining is transitive — A-B and B-C means A, B and C all become one box —
    so a pile of duplicate detections collapses in a single pass.

    Returns the new box list and how many boxes disappeared.
    """
    count = len(boxes)
    if count < 2:
        return list(boxes), 0

    parent = list(range(count))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    for i in range(count):
        for j in range(i + 1, count):
            if same_class_only and boxes[i].class_id != boxes[j].class_id:
                continue
            if containment(boxes[i], boxes[j]) >= threshold:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(count):
        groups.setdefault(find(i), []).append(i)

    out: list[BBox] = []
    for root in sorted(groups):
        members = groups[root]
        if len(members) == 1:
            out.append(boxes[members[0]])
        else:
            out.append(merge_boxes([boxes[i] for i in members], width, height))

    return out, count - len(out)
