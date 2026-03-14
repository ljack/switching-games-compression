// app.js — SWG3 parser, zlib decompression, UI orchestration

import { GPUDecoder } from './gpu.js';

// ─── Float16 → Float32 conversion ────────────────────────────────────

function f16ToF32(bits) {
  const sign = (bits >> 15) & 1;
  const exp = (bits >> 10) & 0x1F;
  const frac = bits & 0x3FF;

  if (exp === 0) {
    if (frac === 0) return sign ? -0 : 0;
    return (sign ? -1 : 1) * (frac / 1024) * (2 ** -14);
  }
  if (exp === 31) {
    return frac === 0 ? (sign ? -Infinity : Infinity) : NaN;
  }
  return (sign ? -1 : 1) * (1 + frac / 1024) * (2 ** (exp - 15));
}

function readFloat16Array(dv, offset, count) {
  const out = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    out[i] = f16ToF32(dv.getUint16(offset + i * 2, true));
  }
  return out;
}

// ─── SWG3 Parser ─────────────────────────────────────────────────────

async function decompressZlib(compressed) {
  // DecompressionStream('deflate') handles zlib-wrapped deflate (RFC 1950)
  const ds = new DecompressionStream('deflate');
  const writer = ds.writable.getWriter();
  writer.write(compressed);
  writer.close();

  const reader = ds.readable.getReader();
  const chunks = [];
  let totalLen = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    totalLen += value.byteLength;
  }

  const result = new Uint8Array(totalLen);
  let offset = 0;
  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return result;
}

function decodeIndicesBitmap(data, totalPixels) {
  const indices = [];
  for (let byteIdx = 0; byteIdx < data.length; byteIdx++) {
    const byte = data[byteIdx];
    for (let bit = 7; bit >= 0; bit--) {
      const pixelIdx = byteIdx * 8 + (7 - bit);
      if (pixelIdx >= totalPixels) break;
      if ((byte >> bit) & 1) {
        indices.push(pixelIdx);
      }
    }
  }
  return new Uint32Array(indices);
}

function decodeIndicesDelta(mode, data, nnz) {
  if (nnz === 0) return new Uint32Array(0);

  const dv = new DataView(data.buffer, data.byteOffset, data.byteLength);
  if (nnz === 1) return new Uint32Array([dv.getUint32(0, true)]);

  const first = dv.getUint32(0, true);
  // slice() copies to a new aligned buffer (subarray would share the
  // potentially-unaligned underlying ArrayBuffer)
  const deltaBytes = data.slice(4);

  let deltas;
  if (mode === 1) {
    deltas = new Uint8Array(deltaBytes.buffer);
  } else if (mode === 2) {
    deltas = new Uint16Array(deltaBytes.buffer);
  } else {
    deltas = new Uint32Array(deltaBytes.buffer);
  }

  const indices = new Uint32Array(nnz);
  indices[0] = first;
  for (let i = 1; i < nnz; i++) {
    indices[i] = indices[i - 1] + deltas[i - 1];
  }
  return indices;
}

function deltaDecode16(deltas) {
  // Cumulative sum of int16 deltas
  const out = new Int16Array(deltas.length);
  if (deltas.length === 0) return out;
  let acc = 0;
  for (let i = 0; i < deltas.length; i++) {
    acc += deltas[i];
    // Wrap to int16 range
    acc = (acc << 16) >> 16;
    out[i] = acc;
  }
  return out;
}

async function parseSWG3(arrayBuffer) {
  const headerDV = new DataView(arrayBuffer, 0, 24);

  // Magic check
  const magic = String.fromCharCode(
    headerDV.getUint8(0), headerDV.getUint8(1),
    headerDV.getUint8(2), headerDV.getUint8(3)
  );
  if (magic !== 'SWG3') throw new Error(`Invalid file: expected SWG3, got ${magic}`);

  const n = headerDV.getUint32(4, true);        // height
  const k = headerDV.getUint32(8, true);        // width
  const channels = headerDV.getUint8(12);
  const layers = headerDV.getUint16(13, true);
  const targetPsnr = headerDV.getFloat32(15, true);
  const maxIter = headerDV.getUint8(19);
  const compressedSize = headerDV.getUint32(20, true);

  // Decompress payload
  const compressedData = new Uint8Array(arrayBuffer, 24, compressedSize);
  const raw = await decompressZlib(compressedData);
  const totalPixels = n * k;

  // Parse per-channel data
  const channelData = [];
  let offset = 0;
  const dv = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);

  for (let ch = 0; ch < channels; ch++) {
    const actualLayers = dv.getUint8(offset); offset += 1;
    const ratio = dv.getFloat32(offset, true); offset += 4;
    const nnz = dv.getUint32(offset, true); offset += 4;
    const indexMode = dv.getUint8(offset); offset += 1;
    const indexDataLen = dv.getUint32(offset, true); offset += 4;
    const indexData = raw.subarray(offset, offset + indexDataLen); offset += indexDataLen;

    // Decode indices
    let indices;
    if (indexMode === 0) {
      indices = decodeIndicesBitmap(indexData, totalPixels);
    } else {
      indices = decodeIndicesDelta(indexMode, indexData, nnz);
    }

    // Decode values
    const scale = dv.getFloat32(offset, true); offset += 4;
    // slice() to get an aligned copy for Int16Array
    const deltaSlice = raw.slice(offset, offset + nnz * 2);
    const deltaValues = new Int16Array(deltaSlice.buffer);
    offset += nnz * 2;

    const quantized = deltaDecode16(deltaValues);
    const values = new Float32Array(nnz);
    for (let i = 0; i < nnz; i++) {
      values[i] = (quantized[i] / 32767) * scale;
    }

    // Read diagonal matrices (float16 → float32)
    const leftDiags = readFloat16Array(dv, offset, actualLayers * n);
    offset += actualLayers * n * 2;
    const rightDiags = readFloat16Array(dv, offset, actualLayers * k);
    offset += actualLayers * k * 2;

    channelData.push({
      layers: actualLayers,
      ratio,
      nnz,
      indices,
      values,
      leftDiags,
      rightDiags,
    });
  }

  return { n, k, channels, layers, targetPsnr, maxIter, channelData };
}

// ─── CPU Fallback ────────────────────────────────────────────────────

function cpuIDCT1D(input, N) {
  const output = new Float32Array(N);
  const alpha0 = Math.sqrt(1 / N);
  const alpha = Math.sqrt(2 / N);
  const pi = Math.PI;
  for (let i = 0; i < N; i++) {
    let sum = alpha0 * input[0];
    const theta = pi * (2 * i + 1) / (2 * N);
    for (let f = 1; f < N; f++) {
      sum += alpha * input[f] * Math.cos(f * theta);
    }
    output[i] = sum;
  }
  return output;
}

function cpuDecodeChannel(ch, n, k) {
  // Scatter into DCT grid
  const grid = new Float32Array(n * k);
  for (let i = 0; i < ch.indices.length; i++) {
    grid[ch.indices[i]] = ch.values[i];
  }

  // Column IDCT
  const afterCols = new Float32Array(n * k);
  for (let j = 0; j < k; j++) {
    const col = new Float32Array(n);
    for (let i = 0; i < n; i++) col[i] = grid[i * k + j];
    const out = cpuIDCT1D(col, n);
    for (let i = 0; i < n; i++) afterCols[i * k + j] = out[i];
  }

  // Row IDCT → C
  const C = new Float32Array(n * k);
  for (let i = 0; i < n; i++) {
    const row = afterCols.subarray(i * k, (i + 1) * k);
    const out = cpuIDCT1D(row, k);
    C.set(out, i * k);
  }

  // L^T @ R
  const LTR = new Float32Array(n * k);
  for (let l = 0; l < ch.layers; l++) {
    for (let i = 0; i < n; i++) {
      const lVal = ch.leftDiags[l * n + i];
      for (let j = 0; j < k; j++) {
        LTR[i * k + j] += lVal * ch.rightDiags[l * k + j];
      }
    }
  }

  // Combine
  const result = new Uint8Array(n * k);
  for (let i = 0; i < n * k; i++) {
    result[i] = Math.max(0, Math.min(255, Math.round(C[i] * LTR[i])));
  }
  return result;
}

// ─── UI ──────────────────────────────────────────────────────────────

const $ = id => document.getElementById(id);
const canvas = $('canvas');
const ctx = canvas.getContext('2d');

let gpuDecoder = null;
let useGPU = false;

// Comparison state
let decodedImageData = null;  // stored after SWG3 decode
let originalImage = null;     // HTMLImageElement of original
let compareMode = '2up';      // '2up' | 'swipe' | 'onion'
let imageWidth = 0;
let imageHeight = 0;

function showError(msg) {
  const box = $('error-box');
  box.textContent = msg;
  box.style.display = 'block';
}

function clearError() {
  $('error-box').style.display = 'none';
}

function setProgress(text, fraction) {
  const area = $('progress-area');
  area.style.display = 'block';
  $('progress-text').textContent = text;
  $('progress-fill').style.width = (fraction * 100) + '%';
}

function hideProgress() {
  $('progress-area').style.display = 'none';
}

function setInfo(parts) {
  $('info-panel').innerHTML = parts.map(p => `<span>${p}</span>`).join('');
}

async function initGPU() {
  const badge = $('gpu-badge');
  const banner = $('warning-banner');

  if (!navigator.gpu) {
    badge.textContent = 'CPU only';
    badge.className = 'badge badge-cpu';
    banner.textContent = 'WebGPU not available. Using CPU fallback (slow for large images).';
    banner.style.display = 'block';
    return;
  }

  try {
    gpuDecoder = new GPUDecoder();
    await gpuDecoder.init();
    useGPU = true;
    badge.textContent = 'GPU: ' + gpuDecoder.getAdapterDescription();
    badge.className = 'badge badge-gpu';
  } catch (e) {
    console.error('GPU init failed:', e);
    badge.textContent = 'GPU init failed';
    badge.className = 'badge badge-cpu';
    banner.textContent = `WebGPU initialization failed: ${e.message}. Using CPU fallback.`;
    banner.style.display = 'block';
  }
}

async function decodeAndRender(arrayBuffer, fileName) {
  clearError();
  setProgress('Parsing SWG3 file...', 0);

  let swg;
  try {
    swg = await parseSWG3(arrayBuffer);
  } catch (e) {
    hideProgress();
    showError(`Parse error: ${e.message}`);
    return;
  }

  const { n, k, channels, channelData } = swg;
  const total = n * k;

  setProgress(`Decoded header: ${k}x${n}, ${channels}ch`, 0.05);

  // Decode each channel
  const channelResults = [];
  const channelNames = ['R', 'G', 'B'];
  const t0 = performance.now();

  for (let ch = 0; ch < channels; ch++) {
    const baseFrac = 0.05 + (ch / channels) * 0.9;
    const name = channelNames[ch] || `Ch${ch}`;

    if (useGPU) {
      setProgress(`GPU: decoding ${name}...`, baseFrac);
      try {
        const result = await gpuDecoder.decodeChannel(
          channelData[ch], n, k,
          stage => setProgress(`GPU ${name}: ${stage}`, baseFrac + 0.25 / channels)
        );
        channelResults.push(result);
      } catch (e) {
        console.error(`GPU decode failed for channel ${ch}:`, e);
        showError(`GPU error on channel ${name}: ${e.message}. Falling back to CPU.`);
        useGPU = false;
        // Fall through to CPU
        setProgress(`CPU: decoding ${name}...`, baseFrac);
        channelResults.push(cpuDecodeChannel(channelData[ch], n, k));
      }
    } else {
      setProgress(`CPU: decoding ${name}...`, baseFrac);
      channelResults.push(cpuDecodeChannel(channelData[ch], n, k));
    }
  }

  const elapsed = performance.now() - t0;

  // Combine channels into RGBA ImageData
  setProgress('Rendering...', 0.95);
  canvas.width = k;
  canvas.height = n;

  const imageData = ctx.createImageData(k, n);
  const pixels = imageData.data;

  for (let i = 0; i < total; i++) {
    const off = i * 4;
    pixels[off]     = channelResults[0] ? channelResults[0][i] : 0;
    pixels[off + 1] = channelResults[1] ? channelResults[1][i] : 0;
    pixels[off + 2] = channelResults[2] ? channelResults[2][i] : 0;
    pixels[off + 3] = 255;
  }

  ctx.putImageData(imageData, 0, 0);
  decodedImageData = imageData;
  imageWidth = k;
  imageHeight = n;
  hideProgress();

  // Show compare bar
  $('compare-bar').classList.add('active');
  if (originalImage) renderComparison();

  // Display info
  const compSize = arrayBuffer.byteLength;
  const rawSize = k * n * 3;
  const ratio = (compSize / rawSize * 100).toFixed(1);
  setInfo([
    `${k} x ${n}`,
    `${channels} channels`,
    `${(compSize / 1024).toFixed(0)} KB compressed`,
    `${ratio}% of raw`,
    `${elapsed.toFixed(0)} ms decode`,
    useGPU ? 'WebGPU' : 'CPU',
    fileName || '',
  ]);

  $('drop-zone').classList.add('hidden');
}

async function loadDemo() {
  clearError();
  setProgress('Loading demo.swg...', 0);
  $('btn-demo').disabled = true;
  $('btn-file').disabled = true;

  try {
    const resp = await fetch('demo.swg');
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const buf = await resp.arrayBuffer();
    await decodeAndRender(buf, 'demo.swg');
  } catch (e) {
    hideProgress();
    showError(`Failed to load demo: ${e.message}`);
  } finally {
    $('btn-demo').disabled = false;
    $('btn-file').disabled = false;
  }
}

async function loadFile(file) {
  clearError();
  $('btn-demo').disabled = true;
  $('btn-file').disabled = true;

  try {
    const buf = await file.arrayBuffer();
    await decodeAndRender(buf, file.name);
  } catch (e) {
    hideProgress();
    showError(`Failed to load file: ${e.message}`);
  } finally {
    $('btn-demo').disabled = false;
    $('btn-file').disabled = false;
  }
}

// ─── Image Comparison ────────────────────────────────────────────────

function loadOriginalImage(file) {
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => {
    originalImage = img;
    $('compare-hint').textContent = `Original: ${img.naturalWidth}x${img.naturalHeight}`;
    if (decodedImageData) renderComparison();
  };
  img.onerror = () => showError('Failed to load original image');
  img.src = url;
}

function makeCanvas(imageData) {
  const c = document.createElement('canvas');
  c.width = imageData.width;
  c.height = imageData.height;
  c.getContext('2d').putImageData(imageData, 0, 0);
  return c;
}

function makeCanvasFromImage(img, w, h) {
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  c.getContext('2d').drawImage(img, 0, 0, w, h);
  return c;
}

function renderComparison() {
  if (!decodedImageData || !originalImage) return;

  const container = $('compare-container');
  container.innerHTML = '';
  container.classList.add('active');
  // Hide the single canvas view
  $('canvas-container').style.display = 'none';

  const w = imageWidth;
  const h = imageHeight;

  if (compareMode === '2up') {
    renderTwoUp(container, w, h);
  } else if (compareMode === 'swipe') {
    renderSwipe(container, w, h);
  } else if (compareMode === 'onion') {
    renderOnion(container, w, h);
  }
}

function renderTwoUp(container, w, h) {
  const wrap = document.createElement('div');
  wrap.className = 'two-up';

  const leftPanel = document.createElement('div');
  leftPanel.className = 'panel';
  const leftLabel = document.createElement('div');
  leftLabel.className = 'label';
  leftLabel.textContent = 'Original';
  leftPanel.appendChild(leftLabel);
  leftPanel.appendChild(makeCanvasFromImage(originalImage, w, h));

  const rightPanel = document.createElement('div');
  rightPanel.className = 'panel';
  const rightLabel = document.createElement('div');
  rightLabel.className = 'label';
  rightLabel.textContent = 'SWG3 Decoded';
  rightPanel.appendChild(rightLabel);
  rightPanel.appendChild(makeCanvas(decodedImageData));

  wrap.appendChild(leftPanel);
  wrap.appendChild(rightPanel);
  container.appendChild(wrap);
}

function renderSwipe(container, w, h) {
  const wrap = document.createElement('div');
  wrap.className = 'swipe-wrap';

  // Bottom layer: decoded
  const decodedCanvas = makeCanvas(decodedImageData);
  wrap.appendChild(decodedCanvas);

  // Clip overlay: original
  const overlay = document.createElement('div');
  overlay.className = 'swipe-overlay';
  const origCanvas = makeCanvasFromImage(originalImage, w, h);
  overlay.appendChild(origCanvas);
  wrap.appendChild(overlay);

  // Divider line
  const line = document.createElement('div');
  line.className = 'swipe-line';
  wrap.appendChild(line);

  // Labels
  const lblLeft = document.createElement('div');
  lblLeft.className = 'swipe-label swipe-label-left';
  lblLeft.textContent = 'Original';
  wrap.appendChild(lblLeft);

  const lblRight = document.createElement('div');
  lblRight.className = 'swipe-label swipe-label-right';
  lblRight.textContent = 'SWG3';
  wrap.appendChild(lblRight);

  container.appendChild(wrap);

  let fraction = 0.5;

  function updateSwipe(f) {
    fraction = Math.max(0, Math.min(1, f));
    const displayW = decodedCanvas.getBoundingClientRect().width;
    const px = fraction * displayW;
    overlay.style.width = px + 'px';
    line.style.left = px + 'px';
  }

  updateSwipe(0.5);

  // Observe resize to keep overlay canvas sized correctly
  const ro = new ResizeObserver(() => {
    const rect = decodedCanvas.getBoundingClientRect();
    origCanvas.style.width = rect.width + 'px';
    origCanvas.style.height = rect.height + 'px';
    updateSwipe(fraction);
  });
  ro.observe(decodedCanvas);

  let dragging = false;
  wrap.addEventListener('mousedown', e => { dragging = true; handleSwipeMove(e); });
  wrap.addEventListener('mousemove', e => { if (dragging) handleSwipeMove(e); });
  window.addEventListener('mouseup', () => { dragging = false; });
  wrap.addEventListener('touchstart', e => { handleSwipeTouch(e); }, { passive: false });
  wrap.addEventListener('touchmove', e => { handleSwipeTouch(e); }, { passive: false });

  function handleSwipeMove(e) {
    const rect = decodedCanvas.getBoundingClientRect();
    updateSwipe((e.clientX - rect.left) / rect.width);
  }

  function handleSwipeTouch(e) {
    e.preventDefault();
    const touch = e.touches[0];
    const rect = decodedCanvas.getBoundingClientRect();
    updateSwipe((touch.clientX - rect.left) / rect.width);
  }
}

function renderOnion(container, w, h) {
  const wrap = document.createElement('div');
  wrap.className = 'onion-wrap';

  // Bottom: decoded
  wrap.appendChild(makeCanvas(decodedImageData));

  // Top: original with opacity
  const topDiv = document.createElement('div');
  topDiv.className = 'onion-top';
  topDiv.id = 'onion-top-layer';
  topDiv.appendChild(makeCanvasFromImage(originalImage, w, h));
  wrap.appendChild(topDiv);

  container.appendChild(wrap);

  const opacity = $('opacity-slider').value / 100;
  topDiv.style.opacity = opacity;
}

function exitComparison() {
  $('compare-container').classList.remove('active');
  $('compare-container').innerHTML = '';
  $('canvas-container').style.display = 'flex';
}

// ─── Event Handlers ──────────────────────────────────────────────────

$('btn-demo').addEventListener('click', loadDemo);

$('btn-file').addEventListener('click', () => $('file-input').click());
$('file-input').addEventListener('change', e => {
  if (e.target.files[0]) loadFile(e.target.files[0]);
});

// Drag and drop
const dropZone = $('drop-zone');
const canvasContainer = $('canvas-container');

for (const target of [dropZone, canvasContainer]) {
  target.addEventListener('dragover', e => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
  });
  target.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');
  });
  target.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (!file) return;
    if (file.name.endsWith('.swg')) {
      loadFile(file);
    } else if (file.type.startsWith('image/') && decodedImageData) {
      loadOriginalImage(file);
    } else if (file.type.startsWith('image/')) {
      showError('Load a .swg file first, then drop the original image to compare');
    } else {
      showError('Please drop a .swg file or an image file');
    }
  });
}

// Compare: load original image
$('btn-original').addEventListener('click', () => $('original-input').click());
$('original-input').addEventListener('change', e => {
  if (e.target.files[0]) loadOriginalImage(e.target.files[0]);
});

// Compare: mode tabs
for (const tab of document.querySelectorAll('.mode-tab')) {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.mode-tab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    compareMode = tab.dataset.mode;
    // Show opacity slider only for onion skin
    $('opacity-control').classList.toggle('visible', compareMode === 'onion');
    if (originalImage && decodedImageData) renderComparison();
  });
}

// Compare: opacity slider
$('opacity-slider').addEventListener('input', e => {
  const val = e.target.value;
  $('opacity-value').textContent = val + '%';
  const top = document.getElementById('onion-top-layer');
  if (top) top.style.opacity = val / 100;
});

// Allow dropping original images on the compare container
$('compare-container').addEventListener('dragover', e => e.preventDefault());
$('compare-container').addEventListener('drop', e => {
  e.preventDefault();
  const file = e.dataTransfer.files[0];
  if (!file) return;
  if (file.name.endsWith('.swg')) {
    loadFile(file);
  } else if (file.type.startsWith('image/') && decodedImageData) {
    loadOriginalImage(file);
  }
});

// ─── Init ────────────────────────────────────────────────────────────

initGPU();
