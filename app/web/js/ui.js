// Shared UI pieces: toasts, modals, and the folder/file picker that replaces
// the native dialogs (the data lives on the machine running the server).

import { api } from './api.js';

const toastHost = document.getElementById('toasts');
const backdrop = document.getElementById('modalBackdrop');
const modal = document.getElementById('modal');

export function toast(message, level = 'info', title = '') {
  const el = document.createElement('div');
  el.className = `toast ${level}`;
  if (title) {
    const b = document.createElement('b');
    b.textContent = title;
    el.appendChild(b);
  }
  el.appendChild(document.createTextNode(message));
  toastHost.appendChild(el);
  const life = level === 'error' ? 9000 : 4800;
  setTimeout(() => el.remove(), life);
  return el;
}

export function plural(n, one, many) {
  return `${n} ${n === 1 ? one : (many || `${one}s`)}`;
}

export function el(tag, props = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (key === 'class') node.className = value;
    else if (key === 'text') node.textContent = value;
    else if (key === 'html') node.innerHTML = value;
    else if (key.startsWith('on')) node.addEventListener(key.slice(2).toLowerCase(), value);
    else if (value !== null && value !== undefined && value !== false) node.setAttribute(key, value);
  }
  for (const child of [].concat(children)) {
    if (child) node.appendChild(typeof child === 'string' ? document.createTextNode(child) : child);
  }
  return node;
}

// ── Modal ──────────────────────────────────────────────────────

let closeCurrent = null;

export function openModal({ title, subtitle, body, actions = [], onClose }) {
  closeModal();
  modal.innerHTML = '';

  const head = el('div', { class: 'modal-head' }, [
    el('div', {}, [
      el('h2', { text: title }),
      subtitle ? el('p', { text: subtitle }) : null,
    ]),
  ]);
  const bodyEl = el('div', { class: 'modal-body' });
  if (body) bodyEl.appendChild(body);

  const foot = el('div', { class: 'modal-foot' });
  for (const action of actions) {
    foot.appendChild(el('button', {
      class: `btn ${action.kind || ''}`,
      text: action.label,
      onclick: () => action.onClick?.(),
    }));
  }

  modal.append(head, bodyEl, foot);
  backdrop.hidden = false;

  const onKey = (e) => { if (e.key === 'Escape') closeModal(); };
  document.addEventListener('keydown', onKey);
  const onBackdrop = (e) => { if (e.target === backdrop) closeModal(); };
  backdrop.addEventListener('mousedown', onBackdrop);

  closeCurrent = () => {
    document.removeEventListener('keydown', onKey);
    backdrop.removeEventListener('mousedown', onBackdrop);
    backdrop.hidden = true;
    modal.innerHTML = '';
    closeCurrent = null;
    onClose?.();
  };
  return { close: closeModal, body: bodyEl, foot };
}

export function closeModal() {
  if (closeCurrent) closeCurrent();
}

// A dialog settles exactly once: whichever happens first — a button, Escape,
// or a click on the backdrop — decides the answer.
function once(resolve) {
  let settled = false;
  return (value) => {
    if (settled) return;
    settled = true;
    resolve(value);
  };
}

export function confirmDialog({ title, message, confirmLabel = 'Continue', danger = false }) {
  return new Promise((resolveRaw) => {
    const resolve = once(resolveRaw);
    const body = el('div', { class: 'block' }, [
      el('div', { class: danger ? 'warnbox' : 'block-note', text: message }),
    ]);
    openModal({
      title,
      body,
      actions: [
        { label: 'Cancel', onClick: () => { resolve(false); closeModal(); } },
        { label: confirmLabel, kind: danger ? 'danger' : 'primary', onClick: () => { resolve(true); closeModal(); } },
      ],
      onClose: () => resolve(false),
    });
  });
}

// ── Folder / file picker ───────────────────────────────────────

const lastPath = {};

// Set from the session state: `available` says the server can put a dialog on a
// screen this person is sitting at, `enabled` is their preference.
export const nativePicker = { available: false, enabled: true };

export async function pickPath(options = {}) {
  if (nativePicker.available && nativePicker.enabled) {
    const chosen = await pickNative(options);
    if (chosen !== FALL_BACK) return chosen;
  }
  return pickInApp(options);
}

const FALL_BACK = Symbol('fall back to the built-in browser');

// Opens the operating system's own dialog, which runs on the server's desktop.
// Resolves to a path, null when cancelled, or FALL_BACK when it cannot be shown.
function pickNative({ title = 'Choose a folder', want = 'dir', start = null, confirmLabel = 'Choose' }) {
  return new Promise((resolveRaw) => {
    const resolve = once(resolveRaw);
    let switched = false;

    openModal({
      title: 'Waiting for the Windows dialog',
      subtitle: 'It opens in its own window — check behind this one if you cannot see it.',
      body: el('div', { class: 'block' }, [
        el('p', { class: 'block-note', text: title }),
      ]),
      actions: [{
        label: 'Use the built-in browser instead',
        onClick: () => {
          switched = true;
          api.nativeCancel().catch(() => {});   // take the stray dialog back down
          resolve(FALL_BACK);
          closeModal();
        },
      }],
    });

    api.nativePick({
      want,
      title,
      okLabel: confirmLabel,
      start: start || lastPath[want] || null,
    }).then((res) => {
      if (switched) return;   // they already moved on; ignore whatever came back
      closeModal();
      if (res.path) {
        lastPath[want] = want === 'dir' || want === 'media' ? res.path : null;
        resolve(res.path);
      } else if (res.cancelled) {
        resolve(null);
      } else {
        if (res.unavailable) {
          nativePicker.available = false;
          toast(res.unavailable, 'warn', 'Windows dialog unavailable');
        } else if (res.busy) {
          toast('A file dialog is already open. Finish with that one first.', 'warn');
        }
        resolve(FALL_BACK);
      }
    }).catch((err) => {
      if (switched) return;
      closeModal();
      toast(err.message, 'warn', 'Windows dialog failed');
      resolve(FALL_BACK);
    });
  });
}

function pickInApp({
  title = 'Choose a folder',
  subtitle = '',
  want = 'dir',
  start = null,
  confirmLabel = 'Choose',
  describe = null,
} = {}) {
  return new Promise((resolveRaw) => {
    const resolve = once(resolveRaw);
    let current = start || lastPath[want] || null;
    let chosenFile = null;

    const places = el('div', { class: 'fp-places' });
    const list = el('div', { class: 'fp-list' });
    const pathInput = el('input', { type: 'text', class: 'mini', spellcheck: 'false' });
    const summary = el('div', { class: 'fp-summary' });
    const upBtn = el('button', { class: 'ghost', text: '↑ Up', onclick: () => go(parentPath) });

    let parentPath = null;

    const main = el('div', { class: 'fp-main' }, [
      el('div', { class: 'fp-path' }, [
        upBtn,
        pathInput,
        el('button', { class: 'ghost', text: 'Go', onclick: () => go(pathInput.value.trim()) }),
      ]),
      list,
      summary,
    ]);

    const body = el('div', { class: 'fp' }, [places, main]);

    const confirm = el('button', {
      class: 'btn primary',
      text: confirmLabel,
      onclick: () => {
        const value = want === 'dir' || want === 'media' ? (pathInput.value.trim() || current) : chosenFile;
        if (!value) { toast('Pick something first.', 'warn'); return; }
        lastPath[want] = want === 'dir' || want === 'media' ? value : current;
        resolve(value);
        closeModal();
      },
    });

    const { foot } = openModal({
      title,
      subtitle,
      body,
      actions: [{ label: 'Cancel', onClick: () => { resolve(null); closeModal(); } }],
      onClose: () => resolve(null),
    });
    foot.appendChild(confirm);

    async function go(path) {
      let data;
      try {
        data = await api.browse(path, want);
      } catch (err) {
        toast(err.message, 'error', 'Could not open folder');
        return;
      }
      current = data.path;
      parentPath = data.parent;
      chosenFile = null;
      pathInput.value = data.path;
      upBtn.disabled = !data.parent;

      places.innerHTML = '';
      for (const place of data.places) {
        places.appendChild(el('button', {
          class: 'fp-place', text: place.label, onclick: () => go(place.path),
        }));
      }
      for (const drive of data.drives) {
        places.appendChild(el('button', {
          class: 'fp-place', text: drive, onclick: () => go(drive),
        }));
      }

      list.innerHTML = '';
      if (data.error) list.appendChild(el('div', { class: 'fp-item', text: data.error }));
      for (const dir of data.dirs) {
        list.appendChild(el('div', {
          class: 'fp-item', ondblclick: () => go(dir.path), onclick: () => go(dir.path),
        }, [el('span', { class: 'fp-ico', text: '▸' }), el('span', { text: dir.name })]));
      }
      for (const file of data.files) {
        const row = el('div', { class: 'fp-item' }, [
          el('span', { class: 'fp-ico', text: '·' }),
          el('span', { text: file.name }),
          el('span', { class: 'fp-tag', text: `${(file.size / 1e6).toFixed(1)} MB` }),
        ]);
        row.onclick = () => {
          list.querySelectorAll('.fp-item').forEach((n) => n.removeAttribute('aria-selected'));
          row.setAttribute('aria-selected', 'true');
          chosenFile = file.path;
        };
        row.ondblclick = () => { chosenFile = file.path; confirm.click(); };
        list.appendChild(row);
      }

      const info = data.info || {};
      const bits = [];
      if (info.images) bits.push(`${info.images} image${info.images === 1 ? '' : 's'}`);
      if (info.videos) bits.push(`${info.videos} video${info.videos === 1 ? '' : 's'}`);
      if (info.weights) bits.push(`${info.weights} checkpoint${info.weights === 1 ? '' : 's'}`);
      if (info.hasImagesDir && info.hasLabelsDir) bits.push('images/ + labels/');
      if (info.hasDataYaml) bits.push('data.yaml');
      summary.textContent = bits.length ? `This folder holds ${bits.join(' · ')}` : 'This folder has nothing to import.';
      if (describe) {
        const extra = await describe(data.path);
        if (extra) summary.textContent = extra;
      }
    }

    go(current);
  });
}
