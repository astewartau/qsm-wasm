/**
 * Deep-learning model weight manager (browser side).
 *
 * The WASM build can't download its own weights (qsm-core's `download` feature is native-
 * only), so the model registry — exposed by `get_model_registry_wasm()` in the base wasm —
 * tells us each model's weight file URLs / sizes / hashes, and this module fetches them,
 * caches them in IndexedDB (so a model is downloaded once), and hands the raw bytes to the
 * DL wasm functions. It also lazy-loads the larger `onnx` wasm bundle on first DL use.
 *
 * Runs in the Web Worker (has `fetch` + `indexedDB`).
 */

const DB_NAME = 'qsmbly-model-weights';
const STORE = 'weights';

function openDb() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(STORE);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function idbGet(key) {
  const db = await openDb();
  try {
    return await new Promise((resolve, reject) => {
      const req = db.transaction(STORE, 'readonly').objectStore(STORE).get(key);
      req.onsuccess = () => resolve(req.result || null);
      req.onerror = () => reject(req.error);
    });
  } finally { db.close(); }
}

async function idbPut(key, value) {
  const db = await openDb();
  try {
    await new Promise((resolve, reject) => {
      const tx = db.transaction(STORE, 'readwrite');
      tx.objectStore(STORE).put(value, key);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  } finally { db.close(); }
}

// Cache key = file name + sha256, so re-hosted/updated weights bust the cache automatically.
function cacheKey(file) {
  return `${file.name}:${file.sha256 || 'nohash'}`;
}

/** Parse the JSON from `get_model_registry_wasm()` into `{ id: modelSpec }`. */
export function parseRegistry(json) {
  const byId = {};
  try {
    for (const m of JSON.parse(json)) byId[m.id] = m;
  } catch (e) {
    console.error('parseRegistry failed:', e);
  }
  return byId;
}

/** Total download size (bytes) for a model, ignoring already-cached files. */
export async function uncachedBytes(model) {
  let total = 0;
  for (const f of model.files) {
    if (!(await idbGet(cacheKey(f)))) total += Number(f.bytes) || 0;
  }
  return total;
}

async function fetchFileWithProgress(url, file, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetching ${file.name}: HTTP ${res.status}`);
  const total = Number(file.bytes) || Number(res.headers.get('Content-Length')) || 0;
  if (!res.body || !res.body.getReader) {
    // No streaming — fall back to a single buffer (no fine-grained progress).
    const buf = new Uint8Array(await res.arrayBuffer());
    if (onProgress) onProgress(buf.length, total || buf.length);
    return buf;
  }
  const reader = res.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    if (onProgress) onProgress(received, total || received);
  }
  const out = new Uint8Array(received);
  let off = 0;
  for (const c of chunks) { out.set(c, off); off += c.length; }
  return out;
}

/**
 * Ensure all of a model's weight files are available and return them as an array of
 * Uint8Array (in registry file order — the DL wasm fns rely on that order, e.g. NeXtQSM's
 * two U-Nets). Cached files are used without re-downloading.
 *
 * `onProgress(fileIndex, fileName, done, total, cached)` fires as each file downloads.
 *
 * `weightBaseUrl` (optional): when set, each file is fetched from `${weightBaseUrl}/${name}`
 * instead of its registry URL. OSF (the registry host) does NOT send CORS headers, so browser
 * fetches from it fail — point this at a CORS-enabled mirror (e.g. a Hugging Face repo) in
 * production, or the app's own `models/` dir when serving weights locally.
 */
export async function fetchModelWeights(model, onProgress, weightBaseUrl = '') {
  const out = [];
  for (let i = 0; i < model.files.length; i++) {
    const file = model.files[i];
    const key = cacheKey(file);
    const cached = await idbGet(key);
    if (cached) {
      out.push(cached instanceof Uint8Array ? cached : new Uint8Array(cached));
      if (onProgress) onProgress(i, file.name, Number(file.bytes) || 0, Number(file.bytes) || 0, true);
      continue;
    }
    const url = weightBaseUrl ? `${weightBaseUrl.replace(/\/$/, '')}/${file.name}` : file.url;
    const bytes = await fetchFileWithProgress(url, file, (done, total) => {
      if (onProgress) onProgress(i, file.name, done, total, false);
    });
    try { await idbPut(key, bytes); } catch (e) {
      // Storage quota exceeded or private mode — proceed without caching.
      console.warn(`Could not cache ${file.name} (${e}); using in-memory bytes.`);
    }
    out.push(bytes);
  }
  return out;
}

// Lazy-load the onnx wasm bundle (bigger; only needed for deep-learning inference).
let _dlModulePromise = null;

/** Load + init the DL wasm bundle once; resolves to its module namespace. */
export function loadDlWasm(baseUrl, version) {
  if (!_dlModulePromise) {
    _dlModulePromise = (async () => {
      const jsUrl = `${baseUrl}/wasm/qsm_wasm_dl.js?v=${version}`;
      const wasmUrl = `${baseUrl}/wasm/qsm_wasm_dl_bg.wasm?v=${version}`;
      const mod = await import(jsUrl);
      await mod.default(wasmUrl);
      return mod;
    })().catch((e) => { _dlModulePromise = null; throw e; });
  }
  return _dlModulePromise;
}
