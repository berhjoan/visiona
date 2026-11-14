// Worker de la prueba técnica de D-FINE-S. Es un archivo estático (no pasa por el compilador) y
// trae el motor y el modelo solo cuando alguien abre la prueba: el resto de VISIONA no carga nada de esto.
// Motor: ONNX Runtime Web (MIT). Modelo: D-FINE-S entrenado solo con COCO (pesos Apache-2.0).
// No se usa la variante N: su ONNX publicado (onnx-community/dfine_n_coco-ONNX) devuelve el mismo
// objeto muchas veces y con poca confianza, también con Transformers.js (medido el 9-11-2026).
import { LAB_MODEL, pickOutputs, postprocess, preprocess } from './dfine-core.js';

const ORT_BASE = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/';
const MODEL_URL = LAB_MODEL.url;
const SIZE = LAB_MODEL.input;

let ort = null;
let session = null;
let inputName = 'pixel_values';
let pixels = null;

const post = (message) => self.postMessage(message);
const describe = (err) => (err && err.message) || String(err);

/** ¿Hay WebGPU aquí dentro? La página puede tenerlo y el Worker no: por eso se mide en los dos. */
async function probeGpu() {
  const gpu = self.navigator.gpu;
  if (!gpu) return { ok: false, detail: 'El Worker no expone WebGPU.' };
  try {
    const adapter = await gpu.requestAdapter();
    if (!adapter) return { ok: false, detail: 'requestAdapter() no devolvió adaptador.' };
    const info = adapter.info || {};
    return { ok: true, detail: [info.vendor, info.architecture].filter(Boolean).join(' ') || 'adaptador sin nombre', f16: adapter.features.has('shader-f16') };
  } catch (err) {
    return { ok: false, detail: describe(err) };
  }
}

async function download(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`El modelo respondió ${res.status}.`);
  const total = Number(res.headers.get('content-length')) || 0;
  if (!res.body) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader();
  const chunks = [];
  let loaded = 0;
  let last = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    const now = performance.now();
    if (now - last > 250) {
      last = now;
      post({ type: 'progress', stage: 'modelo', loaded, total });
    }
  }
  const bytes = new Uint8Array(loaded);
  let at = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, at);
    at += chunk.length;
  }
  return bytes;
}

/** Tres pasadas en vacío: la primera compila los sombreadores y no debe contar como inferencia real. */
async function warmUp() {
  const blank = new ort.Tensor('float32', new Float32Array(3 * SIZE * SIZE), [1, 3, SIZE, SIZE]);
  let firstMs = 0;
  let lastMs = 0;
  for (let i = 0; i < 3; i++) {
    const t0 = performance.now();
    await session.run({ [inputName]: blank });
    lastMs = performance.now() - t0;
    if (i === 0) firstMs = lastMs;
  }
  return { firstMs, lastMs };
}

async function init({ forceWasm }) {
  const gpu = await probeGpu();
  post({ type: 'probe', gpu });
  post({ type: 'progress', stage: 'motor' });
  ort = await import(`${ORT_BASE}ort.webgpu.bundle.min.mjs`);
  ort.env.wasm.wasmPaths = ORT_BASE;
  ort.env.wasm.numThreads = 1; // GitHub Pages no aísla el origen, y Safari no admite workers anidados
  const attempts = [];
  for (const backend of !forceWasm && gpu.ok ? ['webgpu', 'wasm'] : ['wasm']) {
    try {
      const t0 = performance.now();
      const bytes = await download(MODEL_URL);
      const t1 = performance.now();
      post({ type: 'progress', stage: 'sesion', backend });
      session = await ort.InferenceSession.create(bytes, { executionProviders: [backend], graphOptimizationLevel: 'all' });
      const t2 = performance.now();
      inputName = session.inputNames[0];
      pixels = new Float32Array(3 * SIZE * SIZE);
      const warm = await warmUp();
      post({ type: 'ready', backend, file: 'dfine_s_coco model.onnx', bytes: bytes.length, downloadMs: t1 - t0, createMs: t2 - t1, firstMs: warm.firstMs, warmMs: warm.lastMs, attempts });
      return;
    } catch (err) {
      attempts.push({ backend, error: describe(err) });
      session = null;
    }
  }
  post({ type: 'failed', attempts });
}

async function frame({ id, width, height, buffer }) {
  const t0 = performance.now();
  preprocess(new Uint8ClampedArray(buffer), width, height, pixels);
  const input = new ort.Tensor('float32', pixels, [1, 3, height, width]);
  const t1 = performance.now();
  const outputs = await session.run({ [inputName]: input });
  const t2 = performance.now();
  const { logits, boxes } = pickOutputs(outputs);
  const detections = postprocess(logits.data, boxes.data, { labels: LAB_MODEL.labels });
  post({ type: 'result', id, inferMs: t2 - t1, totalMs: performance.now() - t0, detections });
}

self.onmessage = (event) => {
  const run = event.data.type === 'init' ? init(event.data) : event.data.type === 'frame' ? frame(event.data) : null;
  run?.catch((err) => post({ type: 'error', message: describe(err) }));
};
