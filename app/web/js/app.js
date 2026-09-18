// Bootstrap: wires the filmstrip, the editor, the step panels, the job bar
// and the keyboard shortcuts together.

import { api, connect } from './api.js';
import { Editor } from './canvas.js';
import { labelPanel, mediaPanel, trainPanel } from './panels.js';
import { confirmDialog, el, nativePicker, plural, toast } from './ui.js';

const UI_DEFAULTS = {
  importStride: 1,
  policy: 'replace',
  method: 'describe',
  concept: '',
  promptText: '',
  classId: 0,
  start: 0,
  stride: 1,
  trackRange: 0,
  epochs: 50,
  imgsz: 640,
  cache: 'off',
  workers: 4,
  model: '',
  datasetDir: '',
  precision: 'FP32',
  shape: 'dynamic',
  skipUnreviewed: false,
  filter: 'all',
  nativeDialogs: true,
};

const app = {
  state: { classes: ['object'], task: 'detect', samModel: '', samModels: [], threshold: 0.5, frames: [], stats: {}, device: {} },
  models: null,
  ui: loadUi(),
  step: 'media',
  currentIndex: null,
  frameDetail: null,
  examples: { positive: [], negative: [] },
  jobs: new Map(),
  jobLogs: new Map(),
  editor: null,
  suppressReload: 0,
};

function loadUi() {
  try {
    return { ...UI_DEFAULTS, ...JSON.parse(localStorage.getItem('modeltrainer.ui') || '{}') };
  } catch { return { ...UI_DEFAULTS }; }
}

function saveUi() {
  try { localStorage.setItem('modeltrainer.ui', JSON.stringify(app.ui)); } catch { /* private mode */ }
}

// ── Elements ───────────────────────────────────────────────────

const dom = {
  steps: document.getElementById('steps'),
  panel: document.getElementById('panel'),
  strip: document.getElementById('strip'),
  stripEmpty: document.getElementById('filmstripEmpty'),
  stripCount: document.getElementById('filmstripCount'),
  filter: document.getElementById('frameFilter'),
  canvas: document.getElementById('canvas'),
  host: document.getElementById('canvasHost'),
  stageEmpty: document.getElementById('stageEmpty'),
  stageHud: document.getElementById('stageHud'),
  toolbar: document.getElementById('stageToolbar'),
  tools: document.getElementById('toolButtons'),
  activeClass: document.getElementById('activeClass'),
  frameLabel: document.getElementById('frameLabel'),
  boxSummary: document.getElementById('boxSummary'),
  saveState: document.getElementById('saveState'),
  jobbar: document.getElementById('jobbar'),
  device: document.getElementById('deviceInfo'),
  frameCount: document.getElementById('frameCount'),
};

// ── App methods used by the panels ─────────────────────────────

app.currentFrame = () => app.state.frames.find((f) => f.index === app.currentIndex) || null;

app.goStep = (step) => {
  app.step = step;
  for (const btn of dom.steps.querySelectorAll('.step')) {
    btn.setAttribute('aria-current', btn.dataset.step === step ? 'true' : 'false');
  }
  dom.toolbar.style.display = step === 'label' ? '' : 'none';
  app.renderPanel();
};

app.renderPanel = () => {
  const builder = { media: mediaPanel, label: labelPanel, train: trainPanel }[app.step];
  dom.panel.innerHTML = '';
  dom.panel.appendChild(builder(app));
  saveUi();
};

app.onMethodChange = () => {
  // Selecting the example method puts the right tool in the user's hand.
  if (app.ui.method === 'example') setTool('pos');
  else if (app.editor?.tool === 'pos' || app.editor?.tool === 'neg') setTool('select');
  app.renderPanel();
  renderHud();
};

// Picking a class in the toolbar also recolours whatever box is selected.
app.setClass = (id) => {
  app.ui.classId = id;
  dom.activeClass.value = String(id);
  app.editor.setClassId(id);
};

// Picking a class inside an auto-label method only says where results should
// land — it must not relabel the box the user happens to have selected.
app.setSearchClass = (id) => {
  app.ui.classId = id;
  dom.activeClass.value = String(id);
  app.editor.classId = id;
};

app.clearExamples = () => {
  app.editor.clearExamples();
  app.examples = { positive: [], negative: [] };
  app.renderPanel();
  renderHud();
};

// ── Filmstrip ──────────────────────────────────────────────────

function visibleFrames() {
  const filter = app.ui.filter;
  return app.state.frames.filter((f) => (
    filter === 'labelled' ? f.boxes > 0 : filter === 'empty' ? f.boxes === 0 : true
  ));
}

// A timer rather than requestAnimationFrame: rAF is paused while the tab is in
// the background, and a long import in another tab must still land in the strip.
let stripTimer = null;
function scheduleStrip() {
  if (stripTimer) return;
  stripTimer = setTimeout(() => { stripTimer = null; renderStrip(); }, 60);
}

function renderStrip() {
  const frames = visibleFrames();
  dom.stripCount.textContent = frames.length
    ? `${frames.length} of ${app.state.frames.length} shown`
    : '';
  dom.stripEmpty.hidden = app.state.frames.length > 0;
  dom.frameCount.textContent = app.state.frames.length
    ? `${app.state.stats.labelled || 0}/${app.state.frames.length} labelled`
    : 'no frames';

  const existing = new Map();
  for (const node of dom.strip.children) existing.set(Number(node.dataset.index), node);

  const fragment = document.createDocumentFragment();
  for (const frame of frames) {
    let node = existing.get(frame.index);
    if (node) {
      existing.delete(frame.index);
      updateThumb(node, frame);
    } else {
      node = makeThumb(frame);
    }
    fragment.appendChild(node);
  }
  for (const node of existing.values()) node.remove();
  dom.strip.appendChild(fragment);
}

function makeThumb(frame) {
  const img = el('img', { loading: 'lazy', alt: `Frame ${frame.index}`, src: `/api/frames/${frame.index}/thumb` });
  const node = el('div', {
    class: 'thumb', 'data-index': String(frame.index), role: 'option', tabindex: '0',
    onclick: () => selectFrame(frame.index),
    onkeydown: (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); selectFrame(frame.index); } },
  }, [img, el('span', { class: 'thumb-idx', text: String(frame.index) }), el('span', { class: 'thumb-n' })]);
  updateThumb(node, frame);
  return node;
}

function updateThumb(node, frame) {
  node.setAttribute('aria-selected', frame.index === app.currentIndex ? 'true' : 'false');
  const badge = node.querySelector('.thumb-n');
  badge.textContent = frame.boxes ? String(frame.boxes) : '';
  badge.className = `thumb-n ${frame.status}`;
}

// ── Frame selection ────────────────────────────────────────────

async function selectFrame(index, { scroll = false } = {}) {
  if (index === null || index === undefined) return;
  await flushSave();
  app.currentIndex = index;

  let detail;
  try {
    detail = await api.frame(index);
  } catch {
    return;
  }
  app.frameDetail = detail;
  dom.stageEmpty.hidden = true;
  await app.editor.load(`/api/frames/${index}/image`, detail.boxes);
  app.examples = { positive: [], negative: [] };

  dom.frameLabel.textContent = `Frame ${index}`;
  updateBoxSummary();
  updateReviewButton();
  for (const node of dom.strip.children) {
    node.setAttribute('aria-selected', Number(node.dataset.index) === index ? 'true' : 'false');
  }
  if (scroll) {
    const node = dom.strip.querySelector(`[data-index="${index}"]`);
    node?.scrollIntoView({ block: 'nearest' });
  }
  renderHud();
  app.renderPanel();
}

function updateReviewButton() {
  const btn = document.getElementById('reviewBtn');
  const status = app.frameDetail?.status;
  const reviewed = status === 'verified' || status === 'exported';
  btn.textContent = reviewed ? 'Reviewed ✓' : 'Mark reviewed';
  btn.disabled = app.currentIndex === null;
  btn.title = reviewed
    ? 'You have checked this frame. Click to put it back on the pile.'
    : 'Mark this frame as checked by you — export can then skip the rest.';
}

function updateBoxSummary() {
  const boxes = app.editor?.boxes || [];
  const detail = app.frameDetail;
  const parts = [];
  parts.push(`${boxes.length} object${boxes.length === 1 ? '' : 's'}`);
  if (detail?.source) parts.push(detail.source.split(/[\\/]/).pop());
  if (detail?.width) parts.push(`${detail.width}×${detail.height}`);
  dom.boxSummary.textContent = parts.join('  ·  ');
}

function step(delta) {
  const frames = visibleFrames();
  if (!frames.length) return;
  const at = frames.findIndex((f) => f.index === app.currentIndex);
  const next = frames[Math.min(frames.length - 1, Math.max(0, (at < 0 ? 0 : at) + delta))];
  if (next) selectFrame(next.index, { scroll: true });
}

// ── Saving ─────────────────────────────────────────────────────

let saveTimer = null;
let pendingSave = null;

function queueSave(boxes) {
  pendingSave = { index: app.currentIndex, boxes: JSON.parse(JSON.stringify(boxes)) };
  dom.saveState.textContent = 'Saving…';
  clearTimeout(saveTimer);
  saveTimer = setTimeout(flushSave, 350);
}

async function flushSave() {
  clearTimeout(saveTimer);
  const job = pendingSave;
  pendingSave = null;
  if (!job) return;
  app.suppressReload = Date.now();
  try {
    const res = await api.putBoxes(job.index, job.boxes, 'verified');
    if (res.stats) app.state.stats = res.stats;
    dom.saveState.textContent = 'Saved';
    setTimeout(() => { if (dom.saveState.textContent === 'Saved') dom.saveState.textContent = ''; }, 1400);
  } catch (err) {
    dom.saveState.textContent = '';
    toast(err.message, 'error', 'Could not save labels');
  }
}

// ── Tools & HUD ────────────────────────────────────────────────

function setTool(tool) {
  app.editor.setTool(tool);
  for (const btn of dom.tools.querySelectorAll('.tool')) {
    btn.setAttribute('aria-pressed', btn.dataset.tool === tool ? 'true' : 'false');
  }
  renderHud();
}

function renderHud() {
  const notes = [];
  const tool = app.editor?.tool;
  if (tool === 'pos') {
    notes.push({ kind: 'pos', html: '<b>Example mode.</b> Drag a box around one object you want SAM 3 to find.' });
  } else if (tool === 'neg') {
    notes.push({ kind: 'neg', html: '<b>Exclude mode.</b> Drag around anything that should not be matched.' });
  } else if (tool === 'draw') {
    notes.push({ kind: 'info', html: '<b>Draw mode.</b> Drag to add a box. Press V to go back to selecting.' });
  }
  const { positive, negative } = app.examples;
  if (positive.length || negative.length) {
    notes.push({
      kind: 'info',
      html: `<b>${positive.length}</b> example${positive.length === 1 ? '' : 's'} · <b>${negative.length}</b> exclusion${negative.length === 1 ? '' : 's'} — run “Point at an example” in the panel.`,
    });
  }
  dom.stageHud.innerHTML = '';
  for (const note of notes) {
    dom.stageHud.appendChild(el('div', { class: `hud-note ${note.kind}`, html: note.html }));
  }
}

// ── Job bar ────────────────────────────────────────────────────

function renderJobs() {
  const jobs = [...app.jobs.values()].filter((j) => j.status === 'running' || j.status === 'pending');
  dom.jobbar.hidden = jobs.length === 0;
  dom.jobbar.innerHTML = '';

  for (const job of jobs) {
    const pct = job.total > 0 ? Math.min(100, (job.current / job.total) * 100) : 0;
    const fill = el('div', { class: `job-fill ${job.total > 0 ? '' : 'indet'}`, style: `width:${job.total > 0 ? pct : 35}%` });

    const stop = el('button', {
      class: 'ghost job-stop',
      text: job.cancelRequested ? 'Stopping…' : 'Stop',
      disabled: !job.cancellable || job.cancelRequested ? 'disabled' : null,
      onclick: () => api.cancelJob(job.id).catch((err) => toast(err.message, 'error')),
    });

    const node = el('div', { class: `job ${job.kind}` }, [
      el('div', { class: 'job-top' }, [
        el('span', { class: 'job-label', text: job.label }),
        el('span', { class: 'job-msg', text: job.total > 0 ? `${job.message || ''}` : (job.message || 'Working…') }),
        stop,
      ]),
      el('div', { class: 'job-track' }, [fill]),
    ]);

    const lines = app.jobLogs.get(job.id);
    if (lines && lines.length && (job.kind === 'train' || job.kind === 'convert')) {
      const log = el('div', { class: 'joblog', text: lines.slice(-8).join('\n') });
      node.appendChild(log);
    }
    dom.jobbar.appendChild(node);
  }
}

function onJobFinished(job) {
  if (job.status === 'error') {
    toast(job.error || 'Something went wrong.', 'error', job.label);
    return;
  }
  if (job.status === 'cancelled') {
    toast('Stopped. Everything finished so far is kept.', 'warn', job.label);
    return;
  }
  const r = job.result || {};
  if (job.kind === 'import') {
    toast(`${plural(r.frames ?? 0, 'frame')} ready${r.labelled ? `, ${r.labelled} with labels from the dataset` : ''}.`, 'ok', 'Import complete');
    refreshFrames();
  } else if (job.kind === 'autolabel') {
    toast(r.objects
      ? `${plural(r.objects, 'object')} found across ${plural(r.framesWithObjects, 'frame')}.`
      : 'No matches. Try a lower confidence, or describe the object differently.',
      r.objects ? 'ok' : 'warn', 'Auto-label complete');
  } else if (job.kind === 'track') {
    toast(`${plural(r.objects ?? 0, 'object')} tracked through ${plural(r.frames ?? 0, 'frame')}.`, 'ok', 'Tracking complete');
  } else if (job.kind === 'prompt') {
    toast(r.objects
      ? `${plural(r.objects, 'match', 'matches')} added to this frame.`
      : 'No matches — try another example box, or lower the confidence.',
      r.objects ? 'ok' : 'warn', 'Search complete');
  } else if (job.kind === 'train') {
    const where = r.best || r.save_dir || '';
    toast(where ? `Weights saved to ${where}` : 'Training finished.', 'ok', 'Training complete');
  } else if (job.kind === 'convert') {
    toast(`${r.converted_items ?? 0} converted, ${r.fallback_items ?? 0} fell back, ${r.failed_items ?? 0} failed → ${r.output_dir || ''}`, 'ok', 'Conversion complete');
  } else if (job.kind === 'onnx') {
    toast(`Saved to ${r.path}`, 'ok', 'ONNX export complete');
  }
}

// ── Events ─────────────────────────────────────────────────────

function handleEvent(msg) {
  switch (msg.type) {
    case 'hello':
      applyState(msg.state);
      break;
    case 'job': {
      const job = msg.job;
      app.jobs.set(job.id, job);
      if (msg.event === 'finished') {
        onJobFinished(job);
        setTimeout(() => { app.jobs.delete(job.id); app.jobLogs.delete(job.id); renderJobs(); }, 600);
        if (['autolabel', 'track', 'prompt'].includes(job.kind)) {
          reloadCurrentFrame();
          refreshStats();
        }
        if (job.kind === 'import') app.renderPanel();
      }
      renderJobs();
      break;
    }
    case 'log': {
      const lines = app.jobLogs.get(msg.jobId) || [];
      lines.push(msg.line);
      app.jobLogs.set(msg.jobId, lines.slice(-60));
      renderJobs();
      break;
    }
    case 'frames': {
      if (msg.event === 'added') {
        for (const frame of msg.frames) upsertFrame(frame);
        scheduleStrip();
        if (app.currentIndex === null && msg.frames.length) selectFrame(msg.frames[0].index);
      } else if (msg.event === 'cleared') {
        app.state.frames = [];
        app.currentIndex = null;
        app.frameDetail = null;
        app.editor.load(null, []);
        dom.stageEmpty.hidden = false;
        dom.frameLabel.textContent = '—';
        renderStrip();
        app.renderPanel();
      } else if (msg.event === 'removed') {
        app.state.frames = app.state.frames.filter((f) => !msg.indices.includes(f.index));
        renderStrip();
      } else if (msg.event === 'reload') {
        refreshFrames();
      }
      break;
    }
    case 'frame': {
      upsertFrame(msg.frame);
      scheduleStrip();
      if (msg.frame.index === app.currentIndex && Date.now() - app.suppressReload > 1200) {
        reloadCurrentFrame();
      }
      break;
    }
    case 'classes':
      app.state.classes = msg.classes;
      app.editor.setClassNames(msg.classes);
      renderClassSelect();
      app.renderPanel();
      break;
    case 'task':
      app.state.task = msg.task;
      app.renderPanel();
      break;
    case 'toast':
      toast(msg.message, msg.level === 'warn' ? 'warn' : msg.level === 'error' ? 'error' : 'info');
      break;
    default:
      break;
  }
}

function upsertFrame(meta) {
  const at = app.state.frames.findIndex((f) => f.index === meta.index);
  if (at >= 0) app.state.frames[at] = meta;
  else {
    app.state.frames.push(meta);
    app.state.frames.sort((a, b) => a.index - b.index);
  }
}

async function reloadCurrentFrame() {
  if (app.currentIndex === null) return;
  try {
    const detail = await api.frame(app.currentIndex);
    app.frameDetail = detail;
    app.editor.setBoxes(detail.boxes);
    updateBoxSummary();
    app.renderPanel();
  } catch { /* frame went away */ }
}

async function refreshFrames() {
  const data = await api.frames();
  app.state.frames = data.frames;
  app.state.stats = data.stats;
  renderStrip();
  if (app.currentIndex === null && data.frames.length) selectFrame(data.frames[0].index, { scroll: true });
  else if (app.currentIndex !== null) reloadCurrentFrame();
  app.renderPanel();
}

async function refreshStats() {
  try {
    const data = await api.frames();
    app.state.frames = data.frames;
    app.state.stats = data.stats;
    renderStrip();
    app.renderPanel();
  } catch { /* ignore */ }
}

function applyState(state) {
  Object.assign(app.state, state);
  nativePicker.available = !!state.nativeDialogs;
  nativePicker.enabled = app.ui.nativeDialogs !== false;
  app.editor.setClassNames(state.classes);
  renderClassSelect();
  renderStrip();
  renderDevice();
  for (const job of state.jobs || []) app.jobs.set(job.id, job);
  renderJobs();
  if (app.currentIndex === null && state.frames.length) {
    selectFrame(state.frames[0].index);
    app.goStep('label');
  } else {
    app.renderPanel();
  }
}

function renderClassSelect() {
  dom.activeClass.innerHTML = '';
  app.state.classes.forEach((name, i) => {
    dom.activeClass.appendChild(el('option', { value: String(i), text: `${i} · ${name}` }));
  });
  const id = Math.min(app.ui.classId, app.state.classes.length - 1);
  app.ui.classId = Math.max(0, id);
  dom.activeClass.value = String(app.ui.classId);
  app.editor.classId = app.ui.classId;
}

function renderDevice() {
  const d = app.state.device || {};
  if (d.cuda) {
    dom.device.textContent = `${d.gpu} · ${d.vram} GB`;
    dom.device.className = 'chip gpu';
    dom.device.title = `Running on GPU · PyTorch ${d.torch}`;
  } else {
    dom.device.textContent = 'CPU only';
    dom.device.className = 'chip cpu';
    dom.device.title = d.error || 'No CUDA GPU detected — SAM 3 and training will be slow.';
  }
}

// ── Wiring ─────────────────────────────────────────────────────

function bind() {
  dom.steps.addEventListener('click', (e) => {
    const btn = e.target.closest('.step');
    if (btn) app.goStep(btn.dataset.step);
  });

  dom.tools.addEventListener('click', (e) => {
    const btn = e.target.closest('.tool');
    if (btn) setTool(btn.dataset.tool);
  });

  dom.activeClass.addEventListener('change', () => app.setClass(+dom.activeClass.value));
  dom.filter.value = app.ui.filter;
  dom.filter.addEventListener('change', () => { app.ui.filter = dom.filter.value; saveUi(); renderStrip(); });

  document.getElementById('prevFrame').onclick = () => step(-1);
  document.getElementById('nextFrame').onclick = () => step(1);
  document.getElementById('zoomIn').onclick = () => app.editor.zoomBy(1.25);
  document.getElementById('zoomOut').onclick = () => app.editor.zoomBy(1 / 1.25);
  document.getElementById('zoomFit').onclick = () => app.editor.fit();
  document.getElementById('undoBtn').onclick = () => { if (!app.editor.undo()) toast('Nothing left to undo on this frame.', 'warn'); };
  document.getElementById('clearFrameBtn').onclick = () => app.editor.clearAll();
  document.getElementById('reviewBtn').onclick = async () => {
    if (app.currentIndex === null) return;
    const reviewed = app.frameDetail?.status === 'verified' || app.frameDetail?.status === 'exported';
    const res = await api.putStatus(app.currentIndex, reviewed ? 'pending' : 'verified');
    app.frameDetail.status = res.frame.status;
    app.state.stats = res.stats;
    updateReviewButton();
    scheduleStrip();
  };
  document.getElementById('dropFrameBtn').onclick = async () => {
    if (app.currentIndex === null) return;
    const index = app.currentIndex;
    const ok = await confirmDialog({
      title: `Drop frame ${index}?`,
      message: 'The frame and its labels leave this session. The original file on disk is untouched.',
      confirmLabel: 'Drop it',
      danger: true,
    });
    if (!ok) return;
    const frames = visibleFrames();
    const at = frames.findIndex((f) => f.index === index);
    await api.deleteFrame(index);
    app.state.frames = app.state.frames.filter((f) => f.index !== index);
    const next = frames[at + 1] || frames[at - 1];
    app.currentIndex = null;
    renderStrip();
    if (next) selectFrame(next.index, { scroll: true });
    else {
      app.frameDetail = null;
      app.editor.load(null, []);
      dom.stageEmpty.hidden = false;
      dom.frameLabel.textContent = '—';
      app.renderPanel();
    }
  };

  document.addEventListener('keydown', (e) => {
    const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName);
    if (typing) return;

    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
      e.preventDefault();
      app.editor.undo();
      return;
    }
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 's') {
      e.preventDefault();
      flushSave().then(() => api.saveSession()).then(() => toast('Session saved.', 'ok'));
      return;
    }
    switch (e.key) {
      case 'ArrowLeft': case 'ArrowUp': e.preventDefault(); step(-1); break;
      case 'ArrowRight': case 'ArrowDown': e.preventDefault(); step(1); break;
      case 'Delete': case 'Backspace': app.editor.deleteSelected(); break;
      case 'v': case 'V': setTool('select'); break;
      case 'b': case 'B': setTool('draw'); break;
      case 'e': case 'E': setTool('pos'); break;
      case 'x': case 'X': setTool('neg'); break;
      case 'f': case 'F': app.editor.fit(); break;
      default:
        if (/^[0-9]$/.test(e.key)) {
          const id = e.key === '0' ? 9 : Number(e.key) - 1;
          if (id < app.state.classes.length) app.setClass(id);
        }
    }
  });

  window.addEventListener('beforeunload', () => { flushSave(); saveUi(); });
}

// ── Start ──────────────────────────────────────────────────────

async function main() {
  app.editor = new Editor(dom.canvas, dom.host);
  app.editor.onChange = (boxes) => { queueSave(boxes); updateBoxSummary(); scheduleStrip(); };
  app.editor.onSelect = (box) => {
    if (box) {
      dom.activeClass.value = String(box.class_id);
      app.ui.classId = box.class_id;
      app.editor.classId = box.class_id;
    }
  };
  app.editor.onExemplars = (positive, negative) => {
    app.examples = {
      positive: positive.map((r) => [r.x1, r.y1, r.x2, r.y2]),
      negative: negative.map((r) => [r.x1, r.y1, r.x2, r.y2]),
    };
    app.renderPanel();
    renderHud();
  };

  bind();
  setTool('select');
  app.goStep('media');

  try {
    app.models = await api.models();
    const state = await api.state();
    applyState(state);
  } catch (err) {
    toast(err.message, 'error', 'Could not reach the server');
  }

  connect(handleEvent, (online) => {
    if (!online) dom.saveState.textContent = 'Reconnecting…';
    else if (dom.saveState.textContent === 'Reconnecting…') dom.saveState.textContent = '';
  });
}

main();
