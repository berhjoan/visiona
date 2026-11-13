// Lógica pura de la prueba de D-FINE (sin red ni navegador): la usan el Worker y las pruebas.

/** Las 80 clases, en el orden de salida del modelo y con los nombres que usa VISIONA (labels.ts). */
export const COCO_80 = Object.freeze([
  'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
  'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
  'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
  'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
  'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
  'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
  'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
  'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
]);

/**
 * El paquete del modelo de la prueba: de dónde se baja (una versión fija), a qué tamaño trabaja y
 * sus clases en el orden de salida. Un detector entrenado por VISIONA entra cambiando solo esto.
 */
export const LAB_MODEL = Object.freeze({
  id: 'dfine_s_coco',
  url: 'https://huggingface.co/onnx-community/dfine_s_coco-ONNX/resolve/a3cf03147a9b86c78475139115c8ac142577352d/onnx/model.onnx',
  input: 640,
  labels: COCO_80,
});

const clamp01 = (v) => Math.min(1, Math.max(0, v));

/**
 * RGBA intercalado → canales separados (CHW) en 0..1, como pide el modelo: sin normalizar
 * (`do_normalize: false` en su preprocessor_config.json). Reutiliza `out` para no crear basura.
 */
export function preprocess(rgba, width, height, out = new Float32Array(3 * width * height)) {
  const n = width * height;
  for (let i = 0, j = 0; i < n; i++, j += 4) {
    out[i] = rgba[j] / 255;
    out[n + i] = rgba[j + 1] / 255;
    out[2 * n + i] = rgba[j + 2] / 255;
  }
  return out;
}

/**
 * Salida del modelo → detecciones. Cada consulta da un logit por clase y una caja (centro x,
 * centro y, ancho, alto, normalizados). Un DETR no necesita NMS: cada objeto sale una vez.
 * `labels` son las clases del modelo en su orden de salida (por defecto, las 80 de COCO).
 */
export function postprocess(logits, boxes, { labels = COCO_80, classes = labels.length, threshold = 0.5, max = 20 } = {}) {
  const queries = boxes.length / 4;
  const found = [];
  for (let q = 0; q < queries; q++) {
    let best = -Infinity;
    let cls = -1;
    for (let c = 0; c < classes; c++) {
      const v = logits[q * classes + c];
      if (v > best) {
        best = v;
        cls = c;
      }
    }
    const score = 1 / (1 + Math.exp(-best)); // sigmoide: el modelo usa focal loss
    if (score < threshold) continue;
    const [cx, cy, w, h] = [boxes[q * 4], boxes[q * 4 + 1], boxes[q * 4 + 2], boxes[q * 4 + 3]];
    const x0 = clamp01(cx - w / 2);
    const y0 = clamp01(cy - h / 2);
    const x1 = clamp01(cx + w / 2);
    const y1 = clamp01(cy + h / 2);
    if (!(x1 > x0 && y1 > y0)) continue;
    found.push({ label: labels[cls], score, box: { x: x0, y: y0, w: x1 - x0, h: y1 - y0 } });
  }
  found.sort((a, b) => b.score - a.score);
  return found.slice(0, max);
}

/** Localiza las dos salidas: por nombre y, si el modelo las llama distinto, por su forma. */
export function pickOutputs(outputs) {
  const all = Object.values(outputs);
  const boxes = outputs.pred_boxes ?? all.find((t) => t.dims[t.dims.length - 1] === 4);
  const logits = outputs.logits ?? all.find((t) => t !== boxes && t.dims.length === 3);
  if (!boxes || !logits) throw new Error('El modelo no devolvió cajas y clases.');
  return { logits, boxes };
}
