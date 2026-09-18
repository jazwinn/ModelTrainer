// Annotation editor: image display, box drawing/editing, pose keypoints,
// segmentation polygons, and the green/red example boxes used by SAM 3.

export const CLASS_COLORS = [
  '#7c9cff', '#f0a03c', '#3fb27f', '#e2564d', '#c78bff',
  '#39c0c7', '#e5d15b', '#ef7fae', '#8fce5a', '#b08968',
];

export function classColor(id) {
  return CLASS_COLORS[((id % CLASS_COLORS.length) + CLASS_COLORS.length) % CLASS_COLORS.length];
}

const HANDLES = ['tl', 'tm', 'tr', 'ml', 'mr', 'bl', 'bm', 'br'];
const HANDLE_PX = 5;       // half-size of a resize handle, in screen pixels
const KPT_PX = 6.5;        // keypoint radius, in screen pixels

// Line weights, in screen pixels — they do not grow with zoom, so these are
// what an outline actually looks like. Raise them together to go bolder.
const LINE = {
  box: 2.6,
  boxSelected: 3.6,
  dash: [9, 5],          // machine suggestion, not yet looked at
  polygon: 2.2,
  keypoint: 2,
  selection: 2,          // white halo inside a selected box
  example: 2.8,          // green / red SAM example boxes
  exampleDash: [7, 5],
  marquee: 1.5,
};
const KPT_NAMES = ['Top-left', 'Top-right', 'Bottom-right', 'Bottom-left'];

const cursorFor = {
  tl: 'nwse-resize', br: 'nwse-resize',
  tr: 'nesw-resize', bl: 'nesw-resize',
  tm: 'ns-resize', bm: 'ns-resize',
  ml: 'ew-resize', mr: 'ew-resize',
};

export class Editor {
  constructor(canvas, host) {
    this.canvas = canvas;
    this.host = host;
    this.ctx = canvas.getContext('2d');

    this.image = null;
    this.boxes = [];
    this.classNames = ['object'];
    this.classId = 0;
    this.tool = 'select';

    this.positives = [];
    this.negatives = [];

    this.view = { scale: 1, ox: 0, oy: 0 };
    // Selection is ordered: the first box picked leads (its class wins a merge,
    // and resize handles only appear when it is the only one selected).
    this.selection = [];
    this.hover = null;
    this.spaceHeld = false;
    this.drag = null;
    this.undoStack = [];

    this.onChange = () => {};
    this.onSelect = () => {};
    this.onExemplars = () => {};

    this._bind();
    this._observe();
  }

  // ── Public API ───────────────────────────────────────────────

  get selected() {
    return this.selection[0] || null;
  }

  async load(src, boxes) {
    this.boxes = boxes || [];
    this.selection = [];
    this.undoStack = [];
    this.clearExamples(true);
    if (!src) { this.image = null; this.render(); return; }

    const img = new Image();
    img.decoding = 'async';
    await new Promise((resolve) => {
      img.onload = resolve;
      img.onerror = resolve;
      img.src = src;
    });
    this.image = img.naturalWidth ? img : null;
    this.fit();
  }

  setBoxes(boxes) {
    this.boxes = boxes || [];
    this.selection = [];
    this.onSelect(null);
    this.render();
  }

  setTool(tool) {
    this.tool = tool;
    if (tool !== 'select') this.select(null);
    this.hover = null;
    this._cursor();
    this.render();
  }

  setClassId(id) {
    this.classId = id;
    if (this.selection.length) {
      this.pushUndo();
      for (const box of this.selection) box.class_id = id;
      this.commit();
    }
    this.render();
  }

  setClassNames(names) {
    this.classNames = names && names.length ? names : ['object'];
    this.render();
  }

  isSelected(box) {
    return this.selection.includes(box);
  }

  /** Select one box. With `additive`, toggle it in and out of the selection. */
  select(box, { additive = false } = {}) {
    if (!box) {
      this.selection = [];
    } else if (!additive) {
      this.selection = [box];
    } else if (this.isSelected(box)) {
      this.selection = this.selection.filter((b) => b !== box);
    } else {
      this.selection = [...this.selection, box];
    }
    this.onSelect(this.selection);
    this.render();
  }

  selectMany(boxes, { additive = false } = {}) {
    const base = additive ? this.selection : [];
    this.selection = [...base, ...boxes.filter((b) => !base.includes(b))];
    this.onSelect(this.selection);
    this.render();
  }

  selectAll() {
    this.selectMany(this.boxes);
  }

  /** Indices of the selected boxes, in the order they sit in the frame. */
  selectionIndices() {
    return this.selection
      .map((box) => this.boxes.indexOf(box))
      .filter((i) => i >= 0)
      .sort((a, b) => a - b);
  }

  deleteSelected() {
    if (!this.selection.length) return;
    this.pushUndo();
    const doomed = new Set(this.selection);
    this.boxes = this.boxes.filter((b) => !doomed.has(b));
    this.select(null);
    this.commit();
  }

  clearAll() {
    if (!this.boxes.length) return;
    this.pushUndo();
    this.boxes = [];
    this.select(null);
    this.commit();
  }

  clearExamples(silent) {
    this.positives = [];
    this.negatives = [];
    if (!silent) { this.onExemplars(this.positives, this.negatives); this.render(); }
  }

  getExamples() {
    return {
      positive: this.positives.map((r) => [r.x1, r.y1, r.x2, r.y2]),
      negative: this.negatives.map((r) => [r.x1, r.y1, r.x2, r.y2]),
    };
  }

  pushUndo() {
    this.undoStack.push(JSON.stringify(this.boxes));
    if (this.undoStack.length > 60) this.undoStack.shift();
  }

  undo() {
    const snapshot = this.undoStack.pop();
    if (snapshot === undefined) return false;
    this.boxes = JSON.parse(snapshot);
    this.select(null);
    this.commit();
    return true;
  }

  commit() {
    this.onChange(this.boxes);
    this.render();
  }

  // ── View ─────────────────────────────────────────────────────

  fit() {
    if (!this.image) { this.render(); return; }
    const { width, height } = this._size();
    const pad = 24;
    const scale = Math.min(
      (width - pad) / this.image.naturalWidth,
      (height - pad) / this.image.naturalHeight,
    );
    this.view.scale = Math.max(scale, 0.02);
    this.view.ox = (width - this.image.naturalWidth * this.view.scale) / 2;
    this.view.oy = (height - this.image.naturalHeight * this.view.scale) / 2;
    this.render();
  }

  zoomBy(factor, cx, cy) {
    const { width, height } = this._size();
    const px = cx ?? width / 2;
    const py = cy ?? height / 2;
    const before = this._toImage(px, py);
    this.view.scale = Math.min(40, Math.max(0.02, this.view.scale * factor));
    this.view.ox = px - before.x * this.view.scale;
    this.view.oy = py - before.y * this.view.scale;
    this.render();
  }

  _size() {
    return { width: this.host.clientWidth, height: this.host.clientHeight };
  }

  _toImage(sx, sy) {
    return { x: (sx - this.view.ox) / this.view.scale, y: (sy - this.view.oy) / this.view.scale };
  }

  _toScreen(ix, iy) {
    return { x: ix * this.view.scale + this.view.ox, y: iy * this.view.scale + this.view.oy };
  }

  // ── Rendering ────────────────────────────────────────────────

  render() {
    const dpr = window.devicePixelRatio || 1;
    const { width, height } = this._size();
    if (!width || !height) return;

    if (this.canvas.width !== Math.round(width * dpr) || this.canvas.height !== Math.round(height * dpr)) {
      this.canvas.width = Math.round(width * dpr);
      this.canvas.height = Math.round(height * dpr);
    }
    const ctx = this.ctx;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    if (!this.image) return;

    const { scale, ox, oy } = this.view;
    ctx.imageSmoothingEnabled = scale < 3;
    ctx.drawImage(this.image, ox, oy,
      this.image.naturalWidth * scale, this.image.naturalHeight * scale);

    for (const box of this.boxes) this._drawBox(ctx, box);
    for (const rect of this.positives) this._drawExample(ctx, rect, '#3fb27f');
    for (const rect of this.negatives) this._drawExample(ctx, rect, '#e2564d');
    if (this.drag?.kind === 'new' && this.drag.rect) {
      const color = this.tool === 'pos' ? '#3fb27f' : this.tool === 'neg' ? '#e2564d' : classColor(this.classId);
      this._drawExample(ctx, this.drag.rect, color);
    }
    if (this.drag?.kind === 'marquee' && this.drag.rect) {
      const a = this._toScreen(this.drag.rect.x1, this.drag.rect.y1);
      const b = this._toScreen(this.drag.rect.x2, this.drag.rect.y2);
      ctx.save();
      ctx.setLineDash([4, 3]);
      ctx.strokeStyle = '#dfe3ec';
      ctx.lineWidth = LINE.marquee;
      ctx.fillStyle = 'rgba(223,227,236,.10)';
      ctx.fillRect(a.x, a.y, b.x - a.x, b.y - a.y);
      ctx.strokeRect(a.x, a.y, b.x - a.x, b.y - a.y);
      ctx.restore();
    }
  }

  _drawBox(ctx, box) {
    const { scale } = this.view;
    const a = this._toScreen(box.x1, box.y1);
    const b = this._toScreen(box.x2, box.y2);
    const w = b.x - a.x;
    const h = b.y - a.y;
    const color = classColor(box.class_id);
    const isSelected = this.isSelected(box);
    const machine = box.source === 'sam';

    // Polygon (segmentation) first, so the box outline sits on top.
    if (box.polygon && box.polygon.length >= 6) {
      ctx.save();
      ctx.beginPath();
      for (let i = 0; i < box.polygon.length; i += 2) {
        const p = this._toScreen(box.polygon[i] * this.image.naturalWidth,
                                 box.polygon[i + 1] * this.image.naturalHeight);
        if (i === 0) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y);
      }
      ctx.closePath();
      ctx.fillStyle = hexToRgba(color, 0.18);
      ctx.fill();
      ctx.strokeStyle = hexToRgba(color, 0.9);
      ctx.lineWidth = LINE.polygon;
      ctx.stroke();
      ctx.restore();
    }

    ctx.save();
    ctx.lineWidth = isSelected ? LINE.boxSelected : LINE.box;
    ctx.strokeStyle = color;
    // Dashed = suggested by SAM and not yet confirmed; solid = human-made.
    ctx.setLineDash(machine && !isSelected ? LINE.dash : []);
    ctx.strokeRect(a.x, a.y, w, h);
    ctx.setLineDash([]);

    if (isSelected) {
      ctx.fillStyle = hexToRgba(color, 0.14);
      ctx.fillRect(a.x, a.y, w, h);
      // A pale marching-ants outline inside the class colour: selection has to
      // stay obvious on a bright image and when several boxes are picked.
      ctx.save();
      ctx.setLineDash([5, 3]);
      ctx.lineWidth = LINE.selection;
      ctx.strokeStyle = 'rgba(255,255,255,.95)';
      ctx.strokeRect(a.x + 3.5, a.y + 3.5, w - 7, h - 7);
      ctx.restore();
    }

    // Label chip
    const name = this.classNames[box.class_id] ?? `class ${box.class_id}`;
    const text = box.score != null ? `${name}  ${Math.round(box.score * 100)}%` : name;
    ctx.font = '500 11px "IBM Plex Sans", sans-serif';
    const tw = ctx.measureText(text).width;
    const ty = a.y > 16 ? a.y - 15 : a.y + 1;
    ctx.fillStyle = hexToRgba(color, 0.92);
    ctx.fillRect(a.x, ty, tw + 10, 14);
    ctx.fillStyle = '#10131a';
    ctx.fillText(text, a.x + 5, ty + 10.5);

    // Handles only make sense on a single box — dragging one corner of five
    // boxes has no obvious meaning.
    if (isSelected && this.selection.length === 1) {
      ctx.fillStyle = '#ffffff';
      ctx.strokeStyle = '#10131a';
      ctx.lineWidth = 1.5;
      for (const key of HANDLES) {
        const p = this._handlePoint(box, key);
        const s = this._toScreen(p.x, p.y);
        ctx.fillRect(s.x - HANDLE_PX, s.y - HANDLE_PX, HANDLE_PX * 2, HANDLE_PX * 2);
        ctx.strokeRect(s.x - HANDLE_PX, s.y - HANDLE_PX, HANDLE_PX * 2, HANDLE_PX * 2);
      }
    }
    ctx.restore();

    if (box.keypoints) this._drawKeypoints(ctx, box, color);
  }

  _drawKeypoints(ctx, box, color) {
    const corners = [
      [box.x1, box.y1], [box.x2, box.y1], [box.x2, box.y2], [box.x1, box.y2],
    ];
    box.keypoints.forEach((kpt, i) => {
      const ghost = kpt === null;
      const pos = ghost
        ? this._toScreen(corners[i]?.[0] ?? box.x1, corners[i]?.[1] ?? box.y1)
        : this._toScreen(kpt[0] * this.image.naturalWidth, kpt[1] * this.image.naturalHeight);

      ctx.save();
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, KPT_PX, 0, Math.PI * 2);
      if (ghost) {
        ctx.setLineDash([3, 3]);
        ctx.strokeStyle = hexToRgba(color, 0.7);
        ctx.lineWidth = LINE.keypoint;
        ctx.stroke();
      } else {
        ctx.fillStyle = color;
        ctx.fill();
        ctx.strokeStyle = '#0f1218';
        ctx.lineWidth = LINE.keypoint;
        ctx.stroke();
        ctx.fillStyle = '#0f1218';
        ctx.font = '600 8px "IBM Plex Sans", sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(String(i), pos.x, pos.y + 0.5);
      }
      ctx.restore();

      if (this.hover?.type === 'kpt' && this.hover.box === box && this.hover.index === i) {
        const label = `${KPT_NAMES[i] ?? `Point ${i}`}${ghost ? ' · removed' : ''}`;
        ctx.save();
        ctx.font = '500 11px "IBM Plex Sans", sans-serif';
        const tw = ctx.measureText(label).width;
        ctx.fillStyle = 'rgba(14,16,21,.92)';
        ctx.fillRect(pos.x + 9, pos.y - 9, tw + 10, 17);
        ctx.fillStyle = '#dfe3ec';
        ctx.fillText(label, pos.x + 14, pos.y + 3);
        ctx.restore();
      }
    });
  }

  _drawExample(ctx, rect, color) {
    const a = this._toScreen(rect.x1, rect.y1);
    const b = this._toScreen(rect.x2, rect.y2);
    ctx.save();
    ctx.setLineDash(LINE.exampleDash);
    ctx.lineWidth = LINE.example;
    ctx.strokeStyle = color;
    ctx.fillStyle = hexToRgba(color, 0.14);
    ctx.fillRect(a.x, a.y, b.x - a.x, b.y - a.y);
    ctx.strokeRect(a.x, a.y, b.x - a.x, b.y - a.y);
    ctx.restore();
  }

  // ── Hit testing ──────────────────────────────────────────────

  _handlePoint(box, key) {
    const mx = (box.x1 + box.x2) / 2;
    const my = (box.y1 + box.y2) / 2;
    return {
      tl: { x: box.x1, y: box.y1 }, tm: { x: mx, y: box.y1 }, tr: { x: box.x2, y: box.y1 },
      ml: { x: box.x1, y: my }, mr: { x: box.x2, y: my },
      bl: { x: box.x1, y: box.y2 }, bm: { x: mx, y: box.y2 }, br: { x: box.x2, y: box.y2 },
    }[key];
  }

  _pick(pt) {
    const tol = HANDLE_PX / this.view.scale + 1;

    // Keypoints win — they sit on top and are small.
    const kptTol = (KPT_PX + 2) / this.view.scale;
    for (const box of this.boxes) {
      if (!box.keypoints) continue;
      const corners = [
        [box.x1, box.y1], [box.x2, box.y1], [box.x2, box.y2], [box.x1, box.y2],
      ];
      for (let i = 0; i < box.keypoints.length; i += 1) {
        const kpt = box.keypoints[i];
        const p = kpt === null
          ? { x: corners[i]?.[0] ?? box.x1, y: corners[i]?.[1] ?? box.y1 }
          : { x: kpt[0] * this.image.naturalWidth, y: kpt[1] * this.image.naturalHeight };
        if (Math.hypot(p.x - pt.x, p.y - pt.y) <= kptTol) {
          return { type: 'kpt', box, index: i };
        }
      }
    }

    if (this.selection.length === 1) {
      for (const key of HANDLES) {
        const p = this._handlePoint(this.selected, key);
        if (Math.abs(p.x - pt.x) <= tol && Math.abs(p.y - pt.y) <= tol) {
          return { type: 'handle', box: this.selected, key };
        }
      }
    }

    for (let i = this.boxes.length - 1; i >= 0; i -= 1) {
      const box = this.boxes[i];
      if (pt.x >= box.x1 && pt.x <= box.x2 && pt.y >= box.y1 && pt.y <= box.y2) {
        return { type: 'box', box };
      }
    }
    return null;
  }

  // ── Interaction ──────────────────────────────────────────────

  _bind() {
    const c = this.canvas;
    c.addEventListener('pointerdown', (e) => this._down(e));
    c.addEventListener('pointermove', (e) => this._move(e));
    c.addEventListener('pointerup', (e) => this._up(e));
    c.addEventListener('pointercancel', () => { this.drag = null; });
    c.addEventListener('contextmenu', (e) => e.preventDefault());
    c.addEventListener('wheel', (e) => {
      e.preventDefault();
      const rect = c.getBoundingClientRect();
      this.zoomBy(e.deltaY < 0 ? 1.12 : 1 / 1.12, e.clientX - rect.left, e.clientY - rect.top);
    }, { passive: false });
  }

  /** Space held = pan with the left button, the usual canvas convention. */
  setSpaceHeld(held) {
    this.spaceHeld = held;
    this._cursor(this.hover);
  }

  _observe() {
    const ro = new ResizeObserver(() => this.render());
    ro.observe(this.host);
  }

  _pointer(e) {
    const rect = this.canvas.getBoundingClientRect();
    return this._toImage(e.clientX - rect.left, e.clientY - rect.top);
  }

  _down(e) {
    if (!this.image) return;
    // Capture keeps a drag alive past the canvas edge; losing it is not fatal.
    try { this.canvas.setPointerCapture(e.pointerId); } catch { /* ignore */ }
    const pt = this._pointer(e);

    // Pan with the middle button, or Space/Alt held — Shift is taken, it adds
    // to the selection.
    if (e.button === 1 || this.spaceHeld || e.altKey) {
      this.drag = { kind: 'pan', sx: e.clientX, sy: e.clientY, ox: this.view.ox, oy: this.view.oy };
      return;
    }

    const hit = this._pick(pt);

    // Right-click on a keypoint removes it (or restores a removed one).
    if (e.button === 2) {
      if (hit?.type === 'kpt') {
        this.pushUndo();
        const kpts = [...hit.box.keypoints];
        if (kpts[hit.index] === null) {
          const corners = [
            [hit.box.x1, hit.box.y1], [hit.box.x2, hit.box.y1],
            [hit.box.x2, hit.box.y2], [hit.box.x1, hit.box.y2],
          ];
          const c = corners[hit.index] ?? [hit.box.x1, hit.box.y1];
          kpts[hit.index] = [c[0] / this.image.naturalWidth, c[1] / this.image.naturalHeight];
        } else {
          kpts[hit.index] = null;
        }
        hit.box.keypoints = kpts;
        this.commit();
      } else if (hit?.type === 'box') {
        this.select(hit.box);
        this.deleteSelected();
      }
      return;
    }

    if (e.button !== 0) return;

    if (this.tool === 'pos' || this.tool === 'neg') {
      this.drag = { kind: 'new', start: pt, rect: null, example: this.tool };
      return;
    }

    if (this.tool === 'draw') {
      this.drag = { kind: 'new', start: pt, rect: null };
      return;
    }

    // select tool
    const additive = e.shiftKey || e.ctrlKey || e.metaKey;

    if (hit?.type === 'kpt') {
      this.pushUndo();
      this.drag = { kind: 'kpt', box: hit.box, index: hit.index, before: shape(hit.box) };
      this.select(hit.box);
      return;
    }
    if (hit?.type === 'handle') {
      this.pushUndo();
      this.drag = { kind: 'resize', box: hit.box, key: hit.key, before: shape(hit.box) };
      return;
    }
    if (hit?.type === 'box') {
      if (additive) {
        this.select(hit.box, { additive: true });
        return;   // a modifier click adjusts the selection, it does not drag it
      }
      // Clicking a box that is already selected moves the whole group; clicking
      // a new one narrows the selection to it first.
      if (!this.isSelected(hit.box)) this.select(hit.box);
      this.pushUndo();
      this.drag = {
        kind: 'move', start: pt,
        before: this.selection.map(shape),
        items: this.selection.map((box) => ({
          box,
          origin: { x1: box.x1, y1: box.y1, x2: box.x2, y2: box.y2 },
          kpts: box.keypoints ? JSON.parse(JSON.stringify(box.keypoints)) : null,
        })),
      };
      return;
    }

    // Empty space: drag a marquee across the boxes you want.
    if (!additive) this.select(null);
    this.drag = { kind: 'marquee', start: pt, rect: null, additive };
  }

  _move(e) {
    if (!this.image) return;
    const pt = this._pointer(e);

    if (!this.drag) {
      const hit = this._pick(pt);
      const changed = JSON.stringify(this.hover ? [this.hover.type, this.hover.index] : null)
        !== JSON.stringify(hit ? [hit.type, hit.index] : null);
      this.hover = hit;
      this._cursor(hit);
      if (changed) this.render();
      return;
    }

    switch (this.drag.kind) {
      case 'pan': {
        this.view.ox = this.drag.ox + (e.clientX - this.drag.sx);
        this.view.oy = this.drag.oy + (e.clientY - this.drag.sy);
        this.render();
        break;
      }
      case 'new':
      case 'marquee': {
        this.drag.rect = normalize(this.drag.start, pt);
        this.render();
        break;
      }
      case 'move': {
        const dx = pt.x - this.drag.start.x;
        const dy = pt.y - this.drag.start.y;
        for (const item of this.drag.items) {
          const { box, origin } = item;
          box.x1 = origin.x1 + dx; box.x2 = origin.x2 + dx;
          box.y1 = origin.y1 + dy; box.y2 = origin.y2 + dy;
          if (item.kpts) {
            box.keypoints = item.kpts.map((k) => (k === null ? null : [
              k[0] + dx / this.image.naturalWidth,
              k[1] + dy / this.image.naturalHeight,
            ]));
          }
        }
        this.render();
        break;
      }
      case 'resize': {
        const box = this.drag.box;
        const key = this.drag.key;
        if (key.includes('l')) box.x1 = pt.x;
        if (key.includes('r')) box.x2 = pt.x;
        if (key.startsWith('t')) box.y1 = pt.y;
        if (key.startsWith('b')) box.y2 = pt.y;
        this.render();
        break;
      }
      case 'kpt': {
        const box = this.drag.box;
        const kpts = [...box.keypoints];
        kpts[this.drag.index] = [
          clamp(pt.x / this.image.naturalWidth, 0, 1),
          clamp(pt.y / this.image.naturalHeight, 0, 1),
        ];
        box.keypoints = kpts;
        this.render();
        break;
      }
      default: break;
    }
  }

  _up(e) {
    if (!this.drag) return;
    const drag = this.drag;
    this.drag = null;
    try { this.canvas.releasePointerCapture(e.pointerId); } catch { /* ignore */ }

    if (drag.kind === 'new') {
      const rect = drag.rect;
      if (!rect || (rect.x2 - rect.x1) < 3 || (rect.y2 - rect.y1) < 3) { this.render(); return; }
      const bounded = this._clampRect(rect);
      if (drag.example === 'pos') {
        this.positives.push(bounded);
        this.onExemplars(this.positives, this.negatives);
      } else if (drag.example === 'neg') {
        this.negatives.push(bounded);
        this.onExemplars(this.positives, this.negatives);
      } else {
        this.pushUndo();
        const box = {
          x1: bounded.x1, y1: bounded.y1, x2: bounded.x2, y2: bounded.y2,
          class_id: this.classId, source: 'manual', polygon: null, keypoints: null, score: null,
        };
        this.boxes.push(box);
        this.select(box);
        this.commit();
        return;
      }
      this.render();
      return;
    }

    if (drag.kind === 'marquee') {
      const rect = drag.rect;
      if (!rect || (rect.x2 - rect.x1) < 2 || (rect.y2 - rect.y1) < 2) { this.render(); return; }
      // Anything the marquee touches counts — asking people to fully enclose a
      // box makes selecting overlapping detections needlessly fiddly.
      const caught = this.boxes.filter((b) => (
        b.x1 < rect.x2 && b.x2 > rect.x1 && b.y1 < rect.y2 && b.y2 > rect.y1
      ));
      this.selectMany(caught, { additive: drag.additive });
      return;
    }

    if (drag.kind === 'move' || drag.kind === 'resize' || drag.kind === 'kpt') {
      const touched = drag.kind === 'move' ? drag.items.map((i) => i.box) : [drag.box];
      for (const box of touched) {
        if (box.x1 > box.x2) [box.x1, box.x2] = [box.x2, box.x1];
        if (box.y1 > box.y2) [box.y1, box.y2] = [box.y2, box.y1];
        Object.assign(box, this._clampRect(box));
      }

      // A click that selects without dragging must not count as an edit: it
      // would mark a SAM suggestion as human-adjusted and save a no-op.
      const before = [].concat(drag.before);
      const unchanged = touched.every((box, i) => shape(box) === before[i]);
      if (unchanged) {
        this.undoStack.pop();
        this.render();
        return;
      }

      for (const box of touched) {
        if (box.source === 'sam') box.source = 'manual';  // a human adjusted it
      }
      this.commit();
    }
  }

  _clampRect(rect) {
    const w = this.image.naturalWidth;
    const h = this.image.naturalHeight;
    return {
      x1: clamp(Math.min(rect.x1, rect.x2), 0, w),
      y1: clamp(Math.min(rect.y1, rect.y2), 0, h),
      x2: clamp(Math.max(rect.x1, rect.x2), 0, w),
      y2: clamp(Math.max(rect.y1, rect.y2), 0, h),
    };
  }

  _cursor(hit) {
    let cursor = 'default';
    if (this.spaceHeld) cursor = 'grab';
    else if (this.tool === 'draw') cursor = 'crosshair';
    else if (this.tool === 'pos' || this.tool === 'neg') cursor = 'crosshair';
    else if (hit?.type === 'handle') cursor = cursorFor[hit.key] || 'default';
    else if (hit?.type === 'kpt') cursor = 'grab';
    else if (hit?.type === 'box') cursor = 'move';
    this.canvas.style.cursor = cursor;
  }
}

/** A box's geometry as a comparable string — used to spot a drag that did nothing. */
function shape(box) {
  return JSON.stringify([
    Math.round(box.x1 * 100), Math.round(box.y1 * 100),
    Math.round(box.x2 * 100), Math.round(box.y2 * 100),
    box.keypoints || null,
  ]);
}

function normalize(a, b) {
  return {
    x1: Math.min(a.x, b.x), y1: Math.min(a.y, b.y),
    x2: Math.max(a.x, b.x), y2: Math.max(a.y, b.y),
  };
}

function clamp(v, lo, hi) { return Math.min(hi, Math.max(lo, v)); }

function hexToRgba(hex, alpha) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
}
