// gpu.js — WebGPU init, buffer management, shader pipelines, dispatch
// Supports both shared-memory IDCT (fallback) and FFT-based IDCT (fast path).

import { FFTEngine } from './gpu-fft.js';

const WG = 256;
const SMEM_FLOATS = 4096; // 16KB shared memory (WebGPU minimum guarantee)

function dims(total) {
  const g = Math.ceil(total / WG);
  if (g <= 65535) return [g, 1, 1];
  return [65535, Math.ceil(g / 65535), 1];
}

// ─── Shaders ──────────────────────────────────────────────────────────

const CLEAR = /* wgsl */`
@group(0) @binding(0) var<storage, read_write> buf: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
  let idx = gid.y * (nwg.x * ${WG}u) + gid.x;
  if (idx < arrayLength(&buf)) { buf[idx] = 0.0; }
}`;

const SCATTER = /* wgsl */`
struct P { nnz: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> idx_buf: array<u32>;
@group(0) @binding(2) var<storage, read> val_buf: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = gid.y * (nwg.x * ${WG}u) + gid.x;
  if (i < p.nnz) { out[idx_buf[i]] = val_buf[i]; }
}`;

const TRANSPOSE = /* wgsl */`
struct P { rows_in: u32, cols_in: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
  let idx = gid.y * (nwg.x * ${WG}u) + gid.x;
  let total = p.rows_in * p.cols_in;
  if (idx >= total) { return; }
  let row = idx / p.cols_in;
  let col = idx % p.cols_in;
  output[col * p.rows_in + row] = input[idx];
}`;

const IDCT_SHARED = /* wgsl */`
struct P { num_rows: u32, row_len: u32, freq_start: u32, freq_end: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

var<workgroup> sh: array<f32, ${SMEM_FLOATS}>;

@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  let row = wg.x;
  if (row >= p.num_rows) { return; }

  let chunk_len = p.freq_end - p.freq_start;
  let base_in = row * p.row_len + p.freq_start;

  for (var i = lid.x; i < chunk_len; i += ${WG}u) {
    sh[i] = input[base_in + i];
  }
  workgroupBarrier();

  let N = p.row_len;
  let alpha0 = sqrt(1.0 / f32(N));
  let alpha  = sqrt(2.0 / f32(N));
  let pi_val = 3.14159265358979;
  let base_out = row * p.row_len;

  for (var out_idx = lid.x; out_idx < p.row_len; out_idx += ${WG}u) {
    let theta = pi_val * f32(2u * out_idx + 1u) / f32(2u * N);
    let cos_theta = cos(theta);

    var cos_prev = cos(f32(p.freq_start) * theta - theta);
    var cos_curr = cos(f32(p.freq_start) * theta);

    let w0 = select(alpha, alpha0, p.freq_start == 0u);
    var sum = w0 * sh[0] * cos_curr;

    for (var fi = 1u; fi < chunk_len; fi++) {
      let cos_next = 2.0 * cos_theta * cos_curr - cos_prev;
      sum += alpha * sh[fi] * cos_next;
      cos_prev = cos_curr;
      cos_curr = cos_next;
    }

    output[base_out + out_idx] += sum;
  }
}`;

const MATMUL = /* wgsl */`
struct P { n: u32, k: u32, layers: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> L: array<f32>;
@group(0) @binding(2) var<storage, read> R: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
  let idx = gid.y * (nwg.x * ${WG}u) + gid.x;
  if (idx >= p.n * p.k) { return; }
  let i = idx / p.k;
  let j = idx % p.k;
  var s = 0.0;
  for (var l = 0u; l < p.layers; l++) {
    s += L[l * p.n + i] * R[l * p.k + j];
  }
  out[idx] = s;
}`;

const COMBINE = /* wgsl */`
@group(0) @binding(0) var<storage, read> C: array<f32>;
@group(0) @binding(1) var<storage, read_write> LTR: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
  let idx = gid.y * (nwg.x * ${WG}u) + gid.x;
  if (idx < arrayLength(&C)) {
    LTR[idx] = clamp(C[idx] * LTR[idx], 0.0, 255.0);
  }
}`;

const PACK = /* wgsl */`
struct P { packed_len: u32, total: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
  let idx = gid.y * (nwg.x * ${WG}u) + gid.x;
  if (idx >= p.packed_len) { return; }
  let b = idx * 4u;
  let b0 = select(0u, u32(input[b]),      b      < p.total);
  let b1 = select(0u, u32(input[b + 1u]), b + 1u < p.total);
  let b2 = select(0u, u32(input[b + 2u]), b + 2u < p.total);
  let b3 = select(0u, u32(input[b + 3u]), b + 3u < p.total);
  output[idx] = b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u);
}`;

// ─── GPUDecoder ───────────────────────────────────────────────────────

export class GPUDecoder {
  constructor() {
    this.device = null;
    this.adapterInfo = null;
    this.P = {};
    this.fft = null;
    this.bufA = null; this.bufB = null;
    this.bufPacked = null; this.staging = null;
    this.complexA = null; this.complexB = null;
    this.n = 0; this.k = 0;
    this.useFFT = true; // Use FFT-based IDCT by default
  }

  async init() {
    if (!navigator.gpu) throw new Error('WebGPU not supported');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('No WebGPU adapter found');
    this.adapterInfo = adapter.info || null;

    const max = 268435456;
    this.device = await adapter.requestDevice({
      requiredLimits: { maxBufferSize: max, maxStorageBufferBindingSize: max }
    });
    this.device.lost.then(i => console.error('GPU lost:', i.message));

    const mk = code => {
      const m = this.device.createShaderModule({ code });
      return this.device.createComputePipeline({ layout: 'auto', compute: { module: m, entryPoint: 'main' } });
    };
    this.P = {
      clear: mk(CLEAR), scatter: mk(SCATTER), transpose: mk(TRANSPOSE),
      idct: mk(IDCT_SHARED), matmul: mk(MATMUL), combine: mk(COMBINE), pack: mk(PACK),
    };

    // Initialize FFT engine
    try {
      this.fft = new FFTEngine(this.device);
    } catch (e) {
      console.warn('FFT init failed, falling back to shared-mem IDCT:', e);
      this.useFFT = false;
    }
  }

  _upload(data, usage = GPUBufferUsage.STORAGE) {
    const b = this.device.createBuffer({ size: data.byteLength, usage: usage | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(b, 0, data);
    return b;
  }

  _uniform(u32arr) {
    const pad = new ArrayBuffer(Math.ceil(u32arr.byteLength / 16) * 16);
    new Uint8Array(pad).set(new Uint8Array(u32arr.buffer, u32arr.byteOffset, u32arr.byteLength));
    return this._upload(new Uint8Array(pad), GPUBufferUsage.UNIFORM);
  }

  _bg(pipeline, entries) {
    return this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: entries.map((buf, i) => ({ binding: i, resource: { buffer: buf } }))
    });
  }

  // Check if N is a valid FFT size (only 2^a * 3^b)
  _isFFTCompatible(N) {
    let n = N;
    while (n % 2 === 0) n /= 2;
    while (n % 3 === 0) n /= 3;
    return n === 1;
  }

  ensureBuffers(n, k) {
    if (this.n === n && this.k === k) return;
    this.bufA?.destroy(); this.bufB?.destroy();
    this.bufPacked?.destroy(); this.staging?.destroy();
    this.complexA?.destroy(); this.complexB?.destroy();

    const total = n * k;
    const pLen = Math.ceil(total / 4) * 4;
    const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

    try {
      this.bufA = this.device.createBuffer({ size: total * 4, usage: S });
      this.bufB = this.device.createBuffer({ size: total * 4, usage: S });
      this.bufPacked = this.device.createBuffer({ size: pLen, usage: S });
      this.staging = this.device.createBuffer({ size: pLen, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

      // Allocate complex buffers for FFT if using FFT path and size fits
      const complexSize = total * 8;
      const maxBuf = this.device.limits.maxBufferSize;
      if (this.useFFT && this._isFFTCompatible(n) && this._isFFTCompatible(k) && complexSize <= maxBuf) {
        this.complexA = this.device.createBuffer({ size: complexSize, usage: S });
        this.complexB = this.device.createBuffer({ size: complexSize, usage: S });
      } else {
        this.complexA = null;
        this.complexB = null;
      }
    } catch (e) {
      // OOM — destroy any partially allocated buffers
      this.bufA?.destroy(); this.bufB?.destroy();
      this.bufPacked?.destroy(); this.staging?.destroy();
      this.complexA?.destroy(); this.complexB?.destroy();
      this.bufA = this.bufB = this.bufPacked = this.staging = null;
      this.complexA = this.complexB = null;
      this.n = 0; this.k = 0;
      const err = new Error(`GPU_OOM: Failed to allocate buffers for ${k}x${n} image`);
      err.oom = true;
      throw err;
    }

    this.n = n; this.k = k;
  }

  async decodeChannel(ch, n, k, onProgress) {
    this.ensureBuffers(n, k);
    const total = n * k;
    const packedLen = Math.ceil(total / 4);
    const d = dims(total);
    const dev = this.device;
    const tmp = [];

    // Upload data
    const bIdx = this._upload(ch.indices);
    const bVal = this._upload(ch.values);
    const bL = this._upload(ch.leftDiags);
    const bR = this._upload(ch.rightDiags);
    tmp.push(bIdx, bVal, bL, bR);

    const uScatter = this._uniform(new Uint32Array([ch.indices.length]));
    const uMatmul = this._uniform(new Uint32Array([n, k, ch.layers]));
    const uPack = this._uniform(new Uint32Array([packedLen, total]));
    tmp.push(uScatter, uMatmul, uPack);

    const canUseFFT = this.useFFT && this.complexA && this.complexB &&
                      this._isFFTCompatible(n) && this._isFFTCompatible(k);

    const enc = dev.createCommandEncoder();
    const run = (pass, pipe, bg, d_) => {
      pass.setPipeline(pipe);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(...d_);
    };

    // ── 1. Clear + Scatter → bufA ──
    onProgress?.('scatter');
    {
      const p = enc.beginComputePass();
      run(p, this.P.clear, this._bg(this.P.clear, [this.bufA]), d);
      run(p, this.P.scatter, this._bg(this.P.scatter, [uScatter, bIdx, bVal, this.bufA]), dims(ch.indices.length));
      p.end();
    }

    if (canUseFFT) {
      // ── FFT-based 2D IDCT ──
      onProgress?.('fft_idct');
      const scratchBuf = this.bufB; // reuse bufB as scratch
      // We need a temporary output buffer since bufB is used as scratch
      // Actually, encode2DIDCT uses: input→output, scratch, complexA, complexB
      // bufA has scattered DCT coefficients
      // We want result in bufA (C matrix)
      // Use bufB as scratch, but we need another buffer for output...
      // Let's create a temp buffer for the FFT output
      const fftOut = dev.createBuffer({ size: total * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      tmp.push(fftOut);

      this.fft.encode2DIDCT(enc, n, k, this.bufA, fftOut, this.bufB, this.complexA, this.complexB, tmp);

      // Copy fftOut → bufA
      enc.copyBufferToBuffer(fftOut, 0, this.bufA, 0, total * 4);
    } else {
      // ── Shared-memory IDCT (fallback for non-FFT-compatible sizes) ──
      const uTransNK = this._uniform(new Uint32Array([n, k]));
      const uTransKN = this._uniform(new Uint32Array([k, n]));
      tmp.push(uTransNK, uTransKN);

      const colChunks = [];
      for (let fs = 0; fs < n; fs += SMEM_FLOATS) {
        const u = this._uniform(new Uint32Array([k, n, fs, Math.min(fs + SMEM_FLOATS, n)]));
        colChunks.push(u); tmp.push(u);
      }
      const rowChunks = [];
      for (let fs = 0; fs < k; fs += SMEM_FLOATS) {
        const u = this._uniform(new Uint32Array([n, k, fs, Math.min(fs + SMEM_FLOATS, k)]));
        rowChunks.push(u); tmp.push(u);
      }

      // Transpose bufA (n×k) → bufB (k×n)
      onProgress?.('transpose');
      {
        const p = enc.beginComputePass();
        run(p, this.P.transpose, this._bg(this.P.transpose, [uTransNK, this.bufA, this.bufB]), d);
        p.end();
      }

      // Column IDCT: bufB → bufA
      onProgress?.('idct_cols');
      {
        const p = enc.beginComputePass();
        run(p, this.P.clear, this._bg(this.P.clear, [this.bufA]), d);
        for (const cp of colChunks) {
          run(p, this.P.idct, this._bg(this.P.idct, [cp, this.bufB, this.bufA]), [k, 1, 1]);
        }
        p.end();
      }

      // Transpose bufA (k×n) → bufB (n×k)
      onProgress?.('transpose');
      {
        const p = enc.beginComputePass();
        run(p, this.P.transpose, this._bg(this.P.transpose, [uTransKN, this.bufA, this.bufB]), d);
        p.end();
      }

      // Row IDCT: bufB → bufA
      onProgress?.('idct_rows');
      {
        const p = enc.beginComputePass();
        run(p, this.P.clear, this._bg(this.P.clear, [this.bufA]), d);
        for (const rp of rowChunks) {
          run(p, this.P.idct, this._bg(this.P.idct, [rp, this.bufB, this.bufA]), [n, 1, 1]);
        }
        p.end();
      }
    }
    // bufA = C

    // ── matmul L^T @ R → bufB ──
    onProgress?.('matmul');
    {
      const p = enc.beginComputePass();
      run(p, this.P.matmul, this._bg(this.P.matmul, [uMatmul, bL, bR, this.bufB]), d);
      p.end();
    }

    // ── combine ──
    onProgress?.('combine');
    {
      const p = enc.beginComputePass();
      run(p, this.P.combine, this._bg(this.P.combine, [this.bufA, this.bufB]), d);
      p.end();
    }

    // ── pack ──
    onProgress?.('pack');
    {
      const p = enc.beginComputePass();
      run(p, this.P.pack, this._bg(this.P.pack, [uPack, this.bufB, this.bufPacked]), dims(packedLen));
      p.end();
    }

    enc.copyBufferToBuffer(this.bufPacked, 0, this.staging, 0, packedLen * 4);

    onProgress?.('submit');
    dev.queue.submit([enc.finish()]);

    onProgress?.('readback');
    await this.staging.mapAsync(GPUMapMode.READ);
    const result = new Uint8Array(this.staging.getMappedRange()).slice(0, total);
    this.staging.unmap();

    for (const b of tmp) b.destroy();
    return result;
  }

  getAdapterDescription() {
    if (!this.adapterInfo) return 'Unknown GPU';
    return this.adapterInfo.description || this.adapterInfo.device || this.adapterInfo.vendor || 'WebGPU';
  }

  destroy() {
    this.bufA?.destroy(); this.bufB?.destroy();
    this.bufPacked?.destroy(); this.staging?.destroy();
    this.complexA?.destroy(); this.complexB?.destroy();
    this.device?.destroy();
  }
}
