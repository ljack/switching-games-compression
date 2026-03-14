// app.js — SWG3 parser, zlib decompression, UI orchestration, compression

import { GPUDecoder } from './gpu.js';
import { GPUCompressor } from './gpu-compress.js';

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

function f32ToF16(val) {
  const buf = new ArrayBuffer(4);
  new Float32Array(buf)[0] = val;
  const bits = new Uint32Array(buf)[0];
  const sign = (bits >> 31) & 1;
  let exp = ((bits >> 23) & 0xFF) - 127 + 15;
  let frac = (bits >> 13) & 0x3FF;

  if (exp <= 0) {
    if (exp < -10) return sign << 15;
    frac = (frac | 0x400) >> (1 - exp);
    return (sign << 15) | frac;
  }
  if (exp >= 31) {
    return (sign << 15) | 0x7C00;
  }
  return (sign << 15) | (exp << 10) | frac;
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

async function compressZlib(data) {
  const cs = new CompressionStream('deflate');
  const writer = cs.writable.getWriter();
  writer.write(data);
  writer.close();

  const reader = cs.readable.getReader();
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
  const out = new Int16Array(deltas.length);
  if (deltas.length === 0) return out;
  let acc = 0;
  for (let i = 0; i < deltas.length; i++) {
    acc += deltas[i];
    acc = (acc << 16) >> 16;
    out[i] = acc;
  }
  return out;
}

async function parseSWG3(arrayBuffer) {
  const headerDV = new DataView(arrayBuffer, 0, 24);

  const magic = String.fromCharCode(
    headerDV.getUint8(0), headerDV.getUint8(1),
    headerDV.getUint8(2), headerDV.getUint8(3)
  );
  if (magic !== 'SWG3') throw new Error(`Invalid file: expected SWG3, got ${magic}`);

  const n = headerDV.getUint32(4, true);
  const k = headerDV.getUint32(8, true);
  const channels = headerDV.getUint8(12);
  const layers = headerDV.getUint16(13, true);
  const targetPsnr = headerDV.getFloat32(15, true);
  const maxIter = headerDV.getUint8(19);
  const compressedSize = headerDV.getUint32(20, true);

  const compressedData = new Uint8Array(arrayBuffer, 24, compressedSize);
  const raw = await decompressZlib(compressedData);
  const totalPixels = n * k;

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

    let indices;
    if (indexMode === 0) {
      indices = decodeIndicesBitmap(indexData, totalPixels);
    } else {
      indices = decodeIndicesDelta(indexMode, indexData, nnz);
    }

    const scale = dv.getFloat32(offset, true); offset += 4;
    const deltaSlice = raw.slice(offset, offset + nnz * 2);
    const deltaValues = new Int16Array(deltaSlice.buffer);
    offset += nnz * 2;

    const quantized = deltaDecode16(deltaValues);
    const values = new Float32Array(nnz);
    for (let i = 0; i < nnz; i++) {
      values[i] = (quantized[i] / 32767) * scale;
    }

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

// ─── SWG3 Encoder ─────────────────────────────────────────────────────

function deltaEncode16(values) {
  const out = new Int16Array(values.length);
  if (values.length === 0) return out;
  out[0] = values[0];
  for (let i = 1; i < values.length; i++) {
    out[i] = (values[i] - values[i - 1]) & 0xFFFF;
  }
  return out;
}

function encodeIndices(indices, totalPixels) {
  if (indices.length === 0) return { mode: 1, data: new Uint8Array(0) };

  // Sort indices
  const sorted = new Uint32Array(indices);
  sorted.sort();

  // Compute deltas
  const deltas = new Uint32Array(sorted.length - 1);
  let maxDelta = 0;
  for (let i = 1; i < sorted.length; i++) {
    deltas[i - 1] = sorted[i] - sorted[i - 1];
    if (deltas[i - 1] > maxDelta) maxDelta = deltas[i - 1];
  }

  // Try bitmap encoding
  const bitmapSize = Math.ceil(totalPixels / 8);
  // Compute delta encoding sizes
  let deltaMode, deltaSize;
  if (maxDelta <= 255) {
    deltaMode = 1;
    deltaSize = 4 + deltas.length; // first index (u32) + u8 deltas
  } else if (maxDelta <= 65535) {
    deltaMode = 2;
    deltaSize = 4 + deltas.length * 2;
  } else {
    deltaMode = 3;
    deltaSize = 4 + deltas.length * 4;
  }

  // Choose smaller encoding
  if (bitmapSize < deltaSize && sorted.length > 100) {
    // Bitmap mode
    const bitmap = new Uint8Array(bitmapSize);
    for (const idx of sorted) {
      const byteIdx = Math.floor(idx / 8);
      const bitIdx = 7 - (idx % 8);
      bitmap[byteIdx] |= (1 << bitIdx);
    }
    return { mode: 0, data: bitmap, sortedIndices: sorted };
  }

  // Delta mode
  const data = new Uint8Array(deltaSize);
  const dv = new DataView(data.buffer);
  dv.setUint32(0, sorted[0], true);
  if (deltaMode === 1) {
    for (let i = 0; i < deltas.length; i++) data[4 + i] = deltas[i];
  } else if (deltaMode === 2) {
    for (let i = 0; i < deltas.length; i++) dv.setUint16(4 + i * 2, deltas[i], true);
  } else {
    for (let i = 0; i < deltas.length; i++) dv.setUint32(4 + i * 4, deltas[i], true);
  }

  return { mode: deltaMode, data, sortedIndices: sorted };
}

async function encodeSWG3(n, k, channels, layers, targetPsnr, maxIter, channelResults) {
  // Build payload
  const parts = [];
  const totalPixels = n * k;

  for (const ch of channelResults) {
    // actual_layers (u8)
    parts.push(new Uint8Array([ch.layers]));

    // ratio (f32)
    const ratioBuf = new ArrayBuffer(4);
    new Float32Array(ratioBuf)[0] = ch.ratio;
    parts.push(new Uint8Array(ratioBuf));

    // nnz (u32)
    const nnzBuf = new ArrayBuffer(4);
    new Uint32Array(nnzBuf)[0] = ch.nnz;
    parts.push(new Uint8Array(nnzBuf));

    // Encode indices
    const { mode, data: indexData, sortedIndices } = encodeIndices(ch.indices, totalPixels);
    parts.push(new Uint8Array([mode])); // index_mode (u8)
    const idlBuf = new ArrayBuffer(4);
    new Uint32Array(idlBuf)[0] = indexData.length;
    parts.push(new Uint8Array(idlBuf)); // index_data_len (u32)
    parts.push(indexData);

    // Quantize and delta-encode values
    // Reorder values to match sorted index order
    const indexMap = new Map();
    for (let i = 0; i < ch.indices.length; i++) {
      indexMap.set(ch.indices[i], ch.values[i]);
    }
    const sortedValues = new Float32Array(ch.nnz);
    for (let i = 0; i < sortedIndices.length; i++) {
      sortedValues[i] = indexMap.get(sortedIndices[i]) || 0;
    }

    // Scale and quantize
    let maxAbs = 0;
    for (let i = 0; i < sortedValues.length; i++) {
      const a = Math.abs(sortedValues[i]);
      if (a > maxAbs) maxAbs = a;
    }
    const scale = maxAbs || 1;
    const scaleBuf = new ArrayBuffer(4);
    new Float32Array(scaleBuf)[0] = scale;
    parts.push(new Uint8Array(scaleBuf)); // scale (f32)

    const quantized = new Int16Array(ch.nnz);
    for (let i = 0; i < ch.nnz; i++) {
      quantized[i] = Math.round((sortedValues[i] / scale) * 32767);
    }
    const deltaValues = deltaEncode16(quantized);
    parts.push(new Uint8Array(deltaValues.buffer)); // delta values (i16[nnz])

    // Left diags as float16
    const leftF16 = new Uint16Array(ch.layers * n);
    for (let i = 0; i < ch.leftDiags.length; i++) {
      leftF16[i] = f32ToF16(ch.leftDiags[i]);
    }
    parts.push(new Uint8Array(leftF16.buffer));

    // Right diags as float16
    const rightF16 = new Uint16Array(ch.layers * k);
    for (let i = 0; i < ch.rightDiags.length; i++) {
      rightF16[i] = f32ToF16(ch.rightDiags[i]);
    }
    parts.push(new Uint8Array(rightF16.buffer));
  }

  // Concatenate payload
  let payloadSize = 0;
  for (const p of parts) payloadSize += p.byteLength;
  const payload = new Uint8Array(payloadSize);
  let offset = 0;
  for (const p of parts) {
    payload.set(p, offset);
    offset += p.byteLength;
  }

  // Compress payload
  const compressed = await compressZlib(payload);

  // Build final file: 24-byte header + compressed data
  const fileSize = 24 + compressed.length;
  const file = new Uint8Array(fileSize);
  const headerDV = new DataView(file.buffer);

  // Magic
  file[0] = 0x53; file[1] = 0x57; file[2] = 0x47; file[3] = 0x33; // 'SWG3'
  headerDV.setUint32(4, n, true);
  headerDV.setUint32(8, k, true);
  file[12] = channels;
  headerDV.setUint16(13, layers, true);
  headerDV.setFloat32(15, targetPsnr, true);
  file[19] = maxIter;
  headerDV.setUint32(20, compressed.length, true);

  file.set(compressed, 24);
  return file;
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
  const grid = new Float32Array(n * k);
  for (let i = 0; i < ch.indices.length; i++) {
    grid[ch.indices[i]] = ch.values[i];
  }

  const afterCols = new Float32Array(n * k);
  for (let j = 0; j < k; j++) {
    const col = new Float32Array(n);
    for (let i = 0; i < n; i++) col[i] = grid[i * k + j];
    const out = cpuIDCT1D(col, n);
    for (let i = 0; i < n; i++) afterCols[i * k + j] = out[i];
  }

  const C = new Float32Array(n * k);
  for (let i = 0; i < n; i++) {
    const row = afterCols.subarray(i * k, (i + 1) * k);
    const out = cpuIDCT1D(row, k);
    C.set(out, i * k);
  }

  const LTR = new Float32Array(n * k);
  for (let l = 0; l < ch.layers; l++) {
    for (let i = 0; i < n; i++) {
      const lVal = ch.leftDiags[l * n + i];
      for (let j = 0; j < k; j++) {
        LTR[i * k + j] += lVal * ch.rightDiags[l * k + j];
      }
    }
  }

  const result = new Uint8Array(n * k);
  for (let i = 0; i < n * k; i++) {
    result[i] = Math.max(0, Math.min(255, Math.round(C[i] * LTR[i])));
  }
  return result;
}

// Web Worker-based CPU decode
async function cpuDecodeChannelWorkers(ch, n, k) {
  const numWorkers = Math.min(navigator.hardwareConcurrency || 4, 16);

  // Scatter into DCT grid
  const grid = new Float32Array(n * k);
  for (let i = 0; i < ch.indices.length; i++) {
    grid[ch.indices[i]] = ch.values[i];
  }

  // Column IDCT with workers
  const afterCols = new Float32Array(n * k);
  await runWorkersOnGrid(grid, n, k, 'idct-columns', numWorkers, (startIdx, endIdx, results) => {
    for (let j = startIdx; j < endIdx; j++) {
      const off = (j - startIdx) * n;
      for (let i = 0; i < n; i++) {
        afterCols[i * k + j] = results[off + i];
      }
    }
  });

  // Row IDCT with workers
  const C = new Float32Array(n * k);
  await runWorkersOnGrid(afterCols, n, k, 'idct-rows', numWorkers, (startIdx, endIdx, results) => {
    for (let i = startIdx; i < endIdx; i++) {
      const off = (i - startIdx) * k;
      for (let j = 0; j < k; j++) {
        C[i * k + j] = results[off + j];
      }
    }
  });

  // Matmul + combine on main thread
  const LTR = new Float32Array(n * k);
  for (let l = 0; l < ch.layers; l++) {
    for (let i = 0; i < n; i++) {
      const lVal = ch.leftDiags[l * n + i];
      for (let j = 0; j < k; j++) {
        LTR[i * k + j] += lVal * ch.rightDiags[l * k + j];
      }
    }
  }

  const result = new Uint8Array(n * k);
  for (let i = 0; i < n * k; i++) {
    result[i] = Math.max(0, Math.min(255, Math.round(C[i] * LTR[i])));
  }
  return result;
}

function runWorkersOnGrid(grid, n, k, type, numWorkers, onResult) {
  return new Promise((resolve, reject) => {
    const total = type === 'idct-columns' ? k : n;
    const chunkSize = Math.ceil(total / numWorkers);
    const workers = [];
    let completed = 0;

    for (let w = 0; w < numWorkers; w++) {
      const startIdx = w * chunkSize;
      const endIdx = Math.min(startIdx + chunkSize, total);
      if (startIdx >= total) break;

      const worker = new Worker('worker.js');
      workers.push(worker);

      worker.onmessage = (e) => {
        onResult(e.data.startIdx, e.data.endIdx, e.data.results);
        worker.terminate();
        completed++;
        if (completed === workers.length) resolve();
      };

      worker.onerror = (e) => {
        workers.forEach(w => w.terminate());
        reject(new Error(`Worker error: ${e.message}`));
      };

      // Copy grid data for each worker (structured clone)
      worker.postMessage({ type, grid, n, k, startIdx, endIdx });
    }
  });
}

// ─── Half-resolution fallback for OOM ─────────────────────────────────

function halveChannelData(ch, n, k) {
  const halfN = Math.floor(n / 2);
  const halfK = Math.floor(k / 2);
  const halfTotal = halfN * halfK;

  // Filter indices to top-left quadrant and re-index
  const newIndices = [];
  const newValues = [];
  for (let i = 0; i < ch.indices.length; i++) {
    const idx = ch.indices[i];
    const row = Math.floor(idx / k);
    const col = idx % k;
    if (row < halfN && col < halfK) {
      newIndices.push(row * halfK + col);
      newValues.push(ch.values[i]);
    }
  }

  // Truncate diagonals
  const newLeftDiags = new Float32Array(ch.layers * halfN);
  for (let l = 0; l < ch.layers; l++) {
    for (let i = 0; i < halfN; i++) {
      newLeftDiags[l * halfN + i] = ch.leftDiags[l * n + i];
    }
  }
  const newRightDiags = new Float32Array(ch.layers * halfK);
  for (let l = 0; l < ch.layers; l++) {
    for (let j = 0; j < halfK; j++) {
      newRightDiags[l * halfK + j] = ch.rightDiags[l * k + j];
    }
  }

  return {
    layers: ch.layers,
    ratio: ch.ratio,
    nnz: newIndices.length,
    indices: new Uint32Array(newIndices),
    values: new Float32Array(newValues),
    leftDiags: newLeftDiags,
    rightDiags: newRightDiags,
  };
}

// ─── UI ──────────────────────────────────────────────────────────────

const $ = id => document.getElementById(id);
const canvas = $('canvas');
const ctx = canvas.getContext('2d');

let gpuDecoder = null;
let gpuCompressor = null;
let useGPU = false;

// Comparison state
let decodedImageData = null;
let originalImage = null;
let compareMode = '2up';
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

function showWarning(msg) {
  const banner = $('warning-banner');
  banner.textContent = msg;
  banner.style.display = 'block';
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

    // Initialize compressor too
    try {
      gpuCompressor = new GPUCompressor(gpuDecoder.device);
    } catch (e) {
      console.warn('GPU compressor init failed:', e);
    }
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
  let decN = n, decK = k;
  let decChannelData = channelData;
  let halfRes = false;

  setProgress(`Decoded header: ${k}x${n}, ${channels}ch`, 0.05);

  const channelResults = [];
  const channelNames = ['R', 'G', 'B'];
  const t0 = performance.now();

  let gpuFailed = false;

  for (let ch = 0; ch < channels; ch++) {
    const baseFrac = 0.05 + (ch / channels) * 0.9;
    const name = channelNames[ch] || `Ch${ch}`;

    if (useGPU && !gpuFailed) {
      setProgress(`GPU: decoding ${name}...`, baseFrac);
      try {
        const result = await gpuDecoder.decodeChannel(
          decChannelData[ch], decN, decK,
          stage => setProgress(`GPU ${name}: ${stage}`, baseFrac + 0.25 / channels)
        );
        channelResults.push(result);
      } catch (e) {
        if (e.oom && !halfRes) {
          // Try half resolution
          halfRes = true;
          decN = Math.floor(n / 2);
          decK = Math.floor(k / 2);
          decChannelData = channelData.map(cd => halveChannelData(cd, n, k));
          showWarning(`GPU memory insufficient for ${k}x${n}. Decoding at half resolution (${decK}x${decN}).`);
          channelResults.length = 0;
          ch = -1; // Restart loop
          continue;
        }
        console.error(`GPU decode failed for channel ${ch}:`, e);
        showError(`GPU error on channel ${name}: ${e.message}. Falling back to CPU.`);
        gpuFailed = true;
        // Fall through to CPU
      }
    }

    if (!useGPU || gpuFailed) {
      setProgress(`CPU: decoding ${name}...`, baseFrac);
      try {
        const result = await cpuDecodeChannelWorkers(decChannelData[ch], decN, decK);
        channelResults.push(result);
      } catch (e) {
        // Workers failed, fall back to synchronous
        channelResults.push(cpuDecodeChannel(decChannelData[ch], decN, decK));
      }
    }
  }

  const elapsed = performance.now() - t0;

  // Combine channels into RGBA ImageData
  setProgress('Rendering...', 0.95);
  const total = decN * decK;
  canvas.width = decK;
  canvas.height = decN;

  const imageData = ctx.createImageData(decK, decN);
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
  imageWidth = decK;
  imageHeight = decN;
  hideProgress();

  // Show compare bar
  $('compare-bar').classList.add('active');
  if (originalImage) renderComparison();

  // Display info
  const compSize = arrayBuffer.byteLength;
  const rawSize = decK * decN * 3;
  const ratio = (compSize / rawSize * 100).toFixed(1);
  const usedFFT = useGPU && !gpuFailed && gpuDecoder.complexA != null;
  const method = useGPU && !gpuFailed ? (usedFFT ? 'WebGPU (FFT)' : 'WebGPU') : 'CPU Workers';
  setInfo([
    `${decK} x ${decN}${halfRes ? ' (half res)' : ''}`,
    `${channels} channels`,
    `${(compSize / 1024).toFixed(0)} KB compressed`,
    `${ratio}% of raw`,
    `${elapsed.toFixed(0)} ms decode`,
    method,
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

// ─── Compression ──────────────────────────────────────────────────────

async function compressImage(sourceImageData, options) {
  const { targetPSNR = 35, layers = 6, maxIter = 7, autoTune = false } = options;
  const n = sourceImageData.height;
  const k = sourceImageData.width;
  const pixels = sourceImageData.data;

  if (!gpuCompressor) {
    showError('GPU compressor not available. Compression requires WebGPU.');
    return null;
  }

  const channelResults = [];
  const channelNames = ['R', 'G', 'B'];

  for (let ch = 0; ch < 3; ch++) {
    const name = channelNames[ch];
    setProgress(`Compressing ${name}...`, ch / 3);

    // Load channel to GPU
    const channelBuf = gpuCompressor.loadChannelToGPU(pixels, n, k, ch);

    // Source channel for residual computation
    const sourceChannel = new Float32Array(n * k);
    for (let i = 0; i < n * k; i++) {
      sourceChannel[i] = pixels[i * 4 + ch];
    }
    const sourceBuf = gpuCompressor._upload(sourceChannel);

    let bestRatio = 0.30;
    if (autoTune) {
      // Coarse search then binary refinement (matching Python auto-tune)
      const coarseRatios = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50];
      let coarseResults = [];
      for (let ci = 0; ci < coarseRatios.length; ci++) {
        const r = coarseRatios[ci];
        setProgress(`Auto-tune ${name}: coarse ratio=${r.toFixed(3)}...`, ch / 3 + ci / (coarseRatios.length * 4));

        const result = await gpuCompressor.compressChannel(
          channelBuf, sourceBuf, n, k, layers, targetPSNR, 3, r,
          (msg) => setProgress(`Auto-tune ${name}: ${msg}`, ch / 3 + ci / (coarseRatios.length * 4))
        );
        coarseResults.push({ ratio: r, psnr: result.psnr, result });
        if (result.psnr >= targetPSNR) break;
      }

      // Find straddling pair
      let lo, hi;
      const last = coarseResults[coarseResults.length - 1];
      if (last.psnr < targetPSNR) {
        // Even max ratio didn't reach target — use it
        bestRatio = last.ratio;
      } else if (coarseResults[0].psnr >= targetPSNR) {
        bestRatio = coarseResults[0].ratio;
      } else {
        // Find straddling pair and binary search
        for (let i = 0; i < coarseResults.length - 1; i++) {
          if (coarseResults[i].psnr < targetPSNR && coarseResults[i + 1].psnr >= targetPSNR) {
            lo = coarseResults[i].ratio;
            hi = coarseResults[i + 1].ratio;
            bestRatio = hi;
            break;
          }
        }
        if (lo !== undefined) {
          for (let step = 0; step < 6; step++) {
            const mid = (lo + hi) / 2;
            setProgress(`Auto-tune ${name}: refine ratio=${mid.toFixed(4)}...`, ch / 3 + 0.25 + step / 24);

            const result = await gpuCompressor.compressChannel(
              channelBuf, sourceBuf, n, k, layers, targetPSNR, 3, mid,
              (msg) => setProgress(`Auto-tune ${name}: ${msg}`, ch / 3 + 0.25 + step / 24)
            );
            if (result.psnr >= targetPSNR) {
              hi = mid;
              bestRatio = mid;
            } else {
              lo = mid;
            }
          }
        }
      }
      // Final compression with best ratio and full iterations
      const result = await gpuCompressor.compressChannel(
        channelBuf, sourceBuf, n, k, layers, targetPSNR, maxIter, bestRatio,
        (msg) => setProgress(`Compress ${name}: ${msg}`, (ch + 0.5) / 3)
      );
      channelResults.push(result);
    } else {
      // Fixed ratio compression (0.30 matches Python default)
      const result = await gpuCompressor.compressChannel(
        channelBuf, sourceBuf, n, k, layers, targetPSNR, maxIter, 0.30,
        (msg) => setProgress(`Compress ${name}: ${msg}`, (ch + 0.5) / 3)
      );
      channelResults.push(result);
    }

    channelBuf.destroy();
    sourceBuf.destroy();
  }

  // Encode SWG3 file
  setProgress('Encoding SWG3...', 0.95);
  const swgData = await encodeSWG3(n, k, 3, layers, targetPSNR, maxIter, channelResults);

  hideProgress();
  return { data: swgData, channelResults, n, k };
}

// ─── Compress Tab UI ──────────────────────────────────────────────────

function initCompressTab() {
  const compressTab = $('tab-compress');
  const viewerTab = $('tab-viewer');
  if (!compressTab || !viewerTab) return; // Not present in HTML

  viewerTab.addEventListener('click', () => {
    viewerTab.classList.add('active');
    compressTab.classList.remove('active');
    $('viewer-panel').style.display = 'block';
    $('compress-panel').style.display = 'none';
  });

  compressTab.addEventListener('click', () => {
    compressTab.classList.add('active');
    viewerTab.classList.remove('active');
    $('viewer-panel').style.display = 'none';
    $('compress-panel').style.display = 'block';
  });

  // Source image drop/select
  const sourceZone = $('source-drop-zone');
  const sourceInput = $('source-input');

  if (sourceZone) {
    sourceZone.addEventListener('dragover', e => {
      e.preventDefault();
      sourceZone.classList.add('drag-over');
    });
    sourceZone.addEventListener('dragleave', () => sourceZone.classList.remove('drag-over'));
    sourceZone.addEventListener('drop', e => {
      e.preventDefault();
      sourceZone.classList.remove('drag-over');
      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith('image/')) loadSourceImage(file);
    });
  }

  if (sourceInput) {
    $('btn-source')?.addEventListener('click', () => sourceInput.click());
    sourceInput.addEventListener('change', e => {
      if (e.target.files[0]) loadSourceImage(e.target.files[0]);
    });
  }

  // Camera
  $('btn-camera')?.addEventListener('click', openCamera);
  $('btn-shutter')?.addEventListener('click', capturePhoto);
  $('btn-camera-cancel')?.addEventListener('click', closeCamera);
  $('camera-select')?.addEventListener('change', switchCamera);

  // Compress button
  $('btn-compress')?.addEventListener('click', startCompression);
}

let sourceImageData = null;
let cameraStream = null;

async function openCamera() {
  const video = $('camera-preview');
  const area = $('camera-area');
  const sel = $('camera-select');
  if (!video || !area) return;

  try {
    // Start with environment camera (rear) on mobile, any on desktop
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment', width: { ideal: 4096 }, height: { ideal: 3072 } }
    });
    cameraStream = stream;
    video.srcObject = stream;
    area.style.display = 'block';
    $('source-drop-zone')?.classList.add('hidden');

    // Populate camera selector
    const devices = await navigator.mediaDevices.enumerateDevices();
    const cameras = devices.filter(d => d.kind === 'videoinput');
    sel.innerHTML = '';
    cameras.forEach((cam, i) => {
      const opt = document.createElement('option');
      opt.value = cam.deviceId;
      opt.textContent = cam.label || `Camera ${i + 1}`;
      sel.appendChild(opt);
    });
    // Select current camera
    const activeTrack = stream.getVideoTracks()[0];
    const activeSetting = activeTrack.getSettings();
    if (activeSetting.deviceId) sel.value = activeSetting.deviceId;
  } catch (e) {
    showError(`Camera access denied: ${e.message}`);
  }
}

async function switchCamera() {
  const sel = $('camera-select');
  const video = $('camera-preview');
  if (!sel || !video) return;

  // Stop current stream
  if (cameraStream) {
    cameraStream.getTracks().forEach(t => t.stop());
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { deviceId: { exact: sel.value }, width: { ideal: 4096 }, height: { ideal: 3072 } }
    });
    cameraStream = stream;
    video.srcObject = stream;
  } catch (e) {
    showError(`Camera switch failed: ${e.message}`);
  }
}

function capturePhoto() {
  const video = $('camera-preview');
  if (!video || !cameraStream) return;

  const c = document.createElement('canvas');
  c.width = video.videoWidth;
  c.height = video.videoHeight;
  const ctx2 = c.getContext('2d');
  ctx2.drawImage(video, 0, 0);
  setSourceFromCanvas(c);
  closeCamera();
}

function closeCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach(t => t.stop());
    cameraStream = null;
  }
  const video = $('camera-preview');
  if (video) video.srcObject = null;
  const area = $('camera-area');
  if (area) area.style.display = 'none';
}

// Find nearest value of form 2^a * 3^b that is <= N
function nearestFFTSize(N) {
  // Generate all 2^a * 3^b values up to N
  const vals = [];
  for (let a = 0; (1 << a) <= N; a++) {
    let v = 1 << a;
    while (v <= N) {
      vals.push(v);
      v *= 3;
    }
  }
  vals.sort((a, b) => a - b);
  // Return largest value <= N (at least 2)
  for (let i = vals.length - 1; i >= 0; i--) {
    if (vals[i] <= N) return vals[i];
  }
  return 2;
}

function setSourceFromCanvas(c) {
  // Resize to FFT-compatible dimensions (2^a * 3^b)
  const origW = c.width, origH = c.height;
  const fitW = nearestFFTSize(origW);
  const fitH = nearestFFTSize(origH);

  let srcCanvas = c;
  if (fitW !== origW || fitH !== origH) {
    srcCanvas = document.createElement('canvas');
    srcCanvas.width = fitW;
    srcCanvas.height = fitH;
    srcCanvas.getContext('2d').drawImage(c, 0, 0, origW, origH, 0, 0, fitW, fitH);
  }

  const ctx2 = srcCanvas.getContext('2d');
  sourceImageData = ctx2.getImageData(0, 0, srcCanvas.width, srcCanvas.height);

  const preview = $('source-preview');
  if (preview) {
    preview.width = srcCanvas.width;
    preview.height = srcCanvas.height;
    preview.getContext('2d').putImageData(sourceImageData, 0, 0);
    preview.style.display = 'block';
  }

  let info = `${srcCanvas.width} x ${srcCanvas.height}`;
  if (fitW !== origW || fitH !== origH) {
    info += ` (resized from ${origW} x ${origH})`;
  }
  $('source-info').textContent = info;
  $('btn-compress').disabled = false;
  $('source-drop-zone')?.classList.add('hidden');
}

function loadSourceImage(file) {
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => {
    const c = document.createElement('canvas');
    c.width = img.naturalWidth;
    c.height = img.naturalHeight;
    c.getContext('2d').drawImage(img, 0, 0);
    setSourceFromCanvas(c);
    URL.revokeObjectURL(url);
  };
  img.src = url;
}

async function startCompression() {
  if (!sourceImageData) return;

  $('btn-compress').disabled = true;
  clearError();

  const targetPSNR = parseFloat($('psnr-slider')?.value || '35');
  const layers = parseInt($('layers-input')?.value || '6');
  const maxIter = parseInt($('iter-input')?.value || '7');
  const autoTune = $('autotune-check')?.checked || false;

  try {
    const result = await compressImage(sourceImageData, { targetPSNR, layers, maxIter, autoTune });
    if (!result) return;

    // Show results
    const rawSize = result.n * result.k * 3;
    const compRatio = (result.data.length / rawSize * 100).toFixed(1);
    const avgPSNR = result.channelResults.reduce((s, c) => s + c.psnr, 0) / 3;

    $('compress-results').style.display = 'block';
    $('compress-results').innerHTML = `
      <div>File size: ${(result.data.length / 1024).toFixed(1)} KB (${compRatio}% of raw)</div>
      <div>Average PSNR: ${avgPSNR.toFixed(2)} dB</div>
      <div>Per-channel: ${result.channelResults.map((c, i) => `${['R','G','B'][i]}=${c.psnr.toFixed(1)}`).join(', ')}</div>
    `;

    // Download button
    const blob = new Blob([result.data], { type: 'application/octet-stream' });
    const downloadLink = $('download-swg');
    if (downloadLink) {
      downloadLink.href = URL.createObjectURL(blob);
      downloadLink.download = 'compressed.swg';
      downloadLink.style.display = 'inline-block';
    }

    // Decode and show preview
    await decodeAndRender(result.data.buffer, 'compressed.swg');
  } catch (e) {
    showError(`Compression failed: ${e.message}`);
  } finally {
    $('btn-compress').disabled = false;
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

  const decodedCanvas = makeCanvas(decodedImageData);
  wrap.appendChild(decodedCanvas);

  const overlay = document.createElement('div');
  overlay.className = 'swipe-overlay';
  const origCanvas = makeCanvasFromImage(originalImage, w, h);
  overlay.appendChild(origCanvas);
  wrap.appendChild(overlay);

  const line = document.createElement('div');
  line.className = 'swipe-line';
  wrap.appendChild(line);

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

  wrap.appendChild(makeCanvas(decodedImageData));

  const topDiv = document.createElement('div');
  topDiv.className = 'onion-top';
  topDiv.id = 'onion-top-layer';
  topDiv.appendChild(makeCanvasFromImage(originalImage, w, h));
  wrap.appendChild(topDiv);

  container.appendChild(wrap);

  const opacity = $('opacity-slider').value / 100;
  topDiv.style.opacity = opacity;
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
initCompressTab();
