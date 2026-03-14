// gpu-compress.js — GPU-accelerated compression: forward DCT, top-k selection, ALS solver
// Uses FFTEngine from gpu-fft.js for O(N log N) DCT/IDCT.

import { FFTEngine } from './gpu-fft.js';

const WG = 256;

function dims(total) {
  const g = Math.ceil(total / WG);
  if (g <= 65535) return [g, 1, 1];
  return [65535, Math.ceil(g / 65535), 1];
}

// ─── Compression Shaders ──────────────────────────────────────────────

// Load one channel from RGBA pixel data into f32 buffer
const LOAD_CHANNEL = /* wgsl */`
struct P { total: u32, channel: u32, width: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> rgba: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
  let idx = gid.y * (nwg.x * ${WG}u) + gid.x;
  if (idx >= p.total) { return; }
  let pixel = rgba[idx];
  let shift = p.channel * 8u;
  output[idx] = f32((pixel >> shift) & 0xFFu);
}`;

// Compute absolute magnitudes of DCT coefficients
const ABS_MAGNITUDE = /* wgsl */`
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
  let idx = gid.y * (nwg.x * ${WG}u) + gid.x;
  if (idx < arrayLength(&input)) {
    output[idx] = abs(input[idx]);
  }
}`;

// Build histogram of magnitudes (1024 bins, log scale)
const HISTOGRAM = /* wgsl */`
struct P { total: u32, max_val: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> magnitudes: array<f32>;
@group(0) @binding(2) var<storage, read_write> histogram: array<atomic<u32>>;

var<workgroup> local_hist: array<atomic<u32>, 1024>;

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
  // Clear local histogram
  for (var i = lid.x; i < 1024u; i += ${WG}u) {
    atomicStore(&local_hist[i], 0u);
  }
  workgroupBarrier();

  let idx = gid.y * (nwg.x * ${WG}u) + gid.x;
  if (idx < p.total) {
    let val = magnitudes[idx];
    var bin: u32;
    if (val <= 0.0 || p.max_val <= 0.0) {
      bin = 0u;
    } else {
      // Log-scale binning
      let normalized = val / p.max_val;
      let log_val = log2(normalized + 1.0);
      bin = min(u32(log_val * 1023.0), 1023u);
    }
    atomicAdd(&local_hist[bin], 1u);
  }
  workgroupBarrier();

  // Merge local to global
  for (var i = lid.x; i < 1024u; i += ${WG}u) {
    let count = atomicLoad(&local_hist[i]);
    if (count > 0u) {
      atomicAdd(&histogram[i], count);
    }
  }
}`;

// Threshold + compact: scatter surviving coefficients
// Uses a prefix-sum offset buffer to know where to write each survivor
const THRESHOLD_COMPACT = /* wgsl */`
struct P { total: u32, threshold: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> coeffs: array<f32>;
@group(0) @binding(2) var<storage, read> prefix: array<u32>;
@group(0) @binding(3) var<storage, read_write> out_indices: array<u32>;
@group(0) @binding(4) var<storage, read_write> out_values: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
  let idx = gid.y * (nwg.x * ${WG}u) + gid.x;
  if (idx >= p.total) { return; }
  if (abs(coeffs[idx]) >= p.threshold) {
    let pos = prefix[idx];
    out_indices[pos] = idx;
    out_values[pos] = coeffs[idx];
  }
}`;

// Compute survival mask (1 if |coeff| >= threshold, 0 otherwise)
const SURVIVAL_MASK = /* wgsl */`
struct P { total: u32, threshold: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> coeffs: array<f32>;
@group(0) @binding(2) var<storage, read_write> mask: array<u32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
  let idx = gid.y * (nwg.x * ${WG}u) + gid.x;
  if (idx >= p.total) { return; }
  mask[idx] = select(0u, 1u, abs(coeffs[idx]) >= p.threshold);
}`;

// Prefix sum (Blelloch scan) - per-workgroup phase
const PREFIX_SUM_LOCAL = /* wgsl */`
struct P { total: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> data: array<u32>;
@group(0) @binding(2) var<storage, read_write> block_sums: array<u32>;

var<workgroup> temp: array<u32, ${WG * 2}>;

@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  let block_start = wg.x * ${WG * 2}u;
  let i0 = block_start + lid.x;
  let i1 = block_start + lid.x + ${WG}u;

  temp[lid.x] = select(0u, data[i0], i0 < p.total);
  temp[lid.x + ${WG}u] = select(0u, data[i1], i1 < p.total);

  // Up-sweep (reduce)
  var offset = 1u;
  for (var d = ${WG}u; d > 0u; d >>= 1u) {
    workgroupBarrier();
    if (lid.x < d) {
      let ai = offset * (2u * lid.x + 1u) - 1u;
      let bi = offset * (2u * lid.x + 2u) - 1u;
      temp[bi] += temp[ai];
    }
    offset <<= 1u;
  }

  // Store block sum and clear last
  if (lid.x == 0u) {
    block_sums[wg.x] = temp[${WG * 2 - 1}u];
    temp[${WG * 2 - 1}u] = 0u;
  }

  // Down-sweep
  for (var d = 1u; d < ${WG * 2}u; d <<= 1u) {
    offset >>= 1u;
    workgroupBarrier();
    if (lid.x < d) {
      let ai = offset * (2u * lid.x + 1u) - 1u;
      let bi = offset * (2u * lid.x + 2u) - 1u;
      let t = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t;
    }
  }
  workgroupBarrier();

  if (i0 < p.total) { data[i0] = temp[lid.x]; }
  if (i1 < p.total) { data[i1] = temp[lid.x + ${WG}u]; }
}`;

// Add block sums to each element
const PREFIX_SUM_ADD = /* wgsl */`
struct P { total: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> data: array<u32>;
@group(0) @binding(2) var<storage, read> block_sums: array<u32>;
@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  if (wg.x == 0u) { return; }
  let idx = wg.x * ${WG * 2}u + lid.x;
  if (idx < p.total) { data[idx] += block_sums[wg.x]; }
  let idx2 = idx + ${WG}u;
  if (idx2 < p.total) { data[idx2] += block_sums[wg.x]; }
}`;

// ─── ALS Solver Shaders ───────────────────────────────────────────────

// Compute outer diagonal products: outer[a*layers+b, i] = L[a,i] * L[b,i]
// Output: layers*layers × N matrix (flattened)
const OUTER_DIAG = /* wgsl */`
struct P { N: u32, layers: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> diags: array<f32>;
@group(0) @binding(2) var<storage, read_write> outer: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
  let idx = gid.y * (nwg.x * ${WG}u) + gid.x;
  if (idx >= p.N) { return; }
  let i = idx;
  for (var a = 0u; a < p.layers; a++) {
    let la = diags[a * p.N + i];
    for (var b = 0u; b < p.layers; b++) {
      let lb = diags[b * p.N + i];
      outer[(a * p.layers + b) * p.N + i] = la * lb;
    }
  }
}`;

// ALS Gram matrix: EtE[a,b,j] = sum_i outer[a,b,i] * C[i,j]^2
// This is a GEMM: (layers^2, N) @ (N, K) but with C squared inline
const ALS_GRAM = /* wgsl */`
struct P { N: u32, K: u32, layers: u32, layers2: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> outer: array<f32>;
@group(0) @binding(2) var<storage, read> C: array<f32>;
@group(0) @binding(3) var<storage, read_write> EtE: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
  let idx = gid.y * (nwg.x * ${WG}u) + gid.x;
  let total = p.layers2 * p.K;
  if (idx >= total) { return; }
  let ab = idx / p.K;
  let j = idx % p.K;
  var sum = 0.0;
  // Kahan compensated summation for f32 precision
  var comp = 0.0;
  for (var i = 0u; i < p.N; i++) {
    let c_val = C[i * p.K + j];
    let term = outer[ab * p.N + i] * c_val * c_val;
    let y = term - comp;
    let t = sum + y;
    comp = (t - sum) - y;
    sum = t;
  }
  EtE[idx] = sum;
}`;

// ALS right-hand side: Etm[a,j] = sum_i diags[a,i] * C[i,j] * M[i,j]
const ALS_RHS = /* wgsl */`
struct P { N: u32, K: u32, layers: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> diags: array<f32>;
@group(0) @binding(2) var<storage, read> C: array<f32>;
@group(0) @binding(3) var<storage, read> M: array<f32>;
@group(0) @binding(4) var<storage, read_write> Etm: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
  let idx = gid.y * (nwg.x * ${WG}u) + gid.x;
  let total = p.layers * p.K;
  if (idx >= total) { return; }
  let a = idx / p.K;
  let j = idx % p.K;
  var sum = 0.0;
  var comp = 0.0;
  for (var i = 0u; i < p.N; i++) {
    let term = diags[a * p.N + i] * C[i * p.K + j] * M[i * p.K + j];
    let y = term - comp;
    let t = sum + y;
    comp = (t - sum) - y;
    sum = t;
  }
  Etm[idx] = sum;
}`;

// Batch solve layers×layers linear systems via Cholesky (one per column/row)
// Hardcoded for layers <= 12
const BATCH_SOLVE = /* wgsl */`
struct P { num_systems: u32, layers: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> EtE: array<f32>;
@group(0) @binding(2) var<storage, read> Etm: array<f32>;
@group(0) @binding(3) var<storage, read_write> result: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
  let sys = gid.y * (nwg.x * ${WG}u) + gid.x;
  if (sys >= p.num_systems) { return; }
  let L2 = p.layers * p.layers;

  // Load A and b into registers (max 12×12 = 144 + 12 = 156 floats)
  var A: array<f32, 144>;
  var b: array<f32, 12>;
  for (var ab = 0u; ab < L2; ab++) {
    A[ab] = EtE[ab * p.num_systems + sys];
  }
  for (var a = 0u; a < p.layers; a++) {
    b[a] = Etm[a * p.num_systems + sys];
  }

  // Add regularization
  for (var a = 0u; a < p.layers; a++) {
    A[a * p.layers + a] += 1e-6;
  }

  // Cholesky decomposition: A = L * L^T (in-place, lower triangle)
  for (var j = 0u; j < p.layers; j++) {
    var sum_sq = 0.0;
    for (var s = 0u; s < j; s++) {
      sum_sq += A[j * p.layers + s] * A[j * p.layers + s];
    }
    A[j * p.layers + j] = sqrt(max(A[j * p.layers + j] - sum_sq, 1e-10));
    let diag_inv = 1.0 / A[j * p.layers + j];
    for (var i = j + 1u; i < p.layers; i++) {
      var dot = 0.0;
      for (var s = 0u; s < j; s++) {
        dot += A[i * p.layers + s] * A[j * p.layers + s];
      }
      A[i * p.layers + j] = (A[i * p.layers + j] - dot) * diag_inv;
    }
  }

  // Forward substitution: L * y = b
  for (var i = 0u; i < p.layers; i++) {
    var s = b[i];
    for (var j = 0u; j < i; j++) {
      s -= A[i * p.layers + j] * b[j];
    }
    b[i] = s / A[i * p.layers + i];
  }

  // Back substitution: L^T * x = y
  for (var ii = 0u; ii < p.layers; ii++) {
    let i = p.layers - 1u - ii;
    var s = b[i];
    for (var j = i + 1u; j < p.layers; j++) {
      s -= A[j * p.layers + i] * b[j];
    }
    b[i] = s / A[i * p.layers + i];
  }

  // Write result
  for (var a = 0u; a < p.layers; a++) {
    result[a * p.num_systems + sys] = b[a];
  }
}`;

// Compute residual: sum of (clamp(C[i]*LTR[i], 0, 255) - M[i])^2
const COMPUTE_RESIDUAL = /* wgsl */`
@group(0) @binding(0) var<storage, read> C: array<f32>;
@group(0) @binding(1) var<storage, read> LTR: array<f32>;
@group(0) @binding(2) var<storage, read> M: array<f32>;
@group(0) @binding(3) var<storage, read_write> partials: array<f32>;

var<workgroup> wg_sums: array<f32, ${WG}>;

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wg: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
  let idx = gid.y * (nwg.x * ${WG}u) + gid.x;
  let total = arrayLength(&C);
  var val = 0.0;
  if (idx < total) {
    let pred = clamp(C[idx] * LTR[idx], 0.0, 255.0);
    let diff = pred - M[idx];
    val = diff * diff;
  }
  wg_sums[lid.x] = val;
  workgroupBarrier();

  // Tree reduction
  for (var s = ${WG / 2}u; s > 0u; s >>= 1u) {
    if (lid.x < s) {
      wg_sums[lid.x] += wg_sums[lid.x + s];
    }
    workgroupBarrier();
  }

  if (lid.x == 0u) {
    let wg_idx = wg.y * nwg.x + wg.x;
    partials[wg_idx] = wg_sums[0];
  }
}`;

// Final reduction: sum an array of partial sums
const REDUCE_SUM = /* wgsl */`
struct P { count: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> data: array<f32>;

var<workgroup> wg_sums: array<f32, ${WG}>;

@compute @workgroup_size(${WG})
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  var sum = 0.0;
  for (var i = lid.x; i < p.count; i += ${WG}u) {
    sum += data[i];
  }
  wg_sums[lid.x] = sum;
  workgroupBarrier();

  for (var s = ${WG / 2}u; s > 0u; s >>= 1u) {
    if (lid.x < s) {
      wg_sums[lid.x] += wg_sums[lid.x + s];
    }
    workgroupBarrier();
  }

  if (lid.x == 0u) {
    data[0] = wg_sums[0];
  }
}`;

// L^T @ R outer product matmul (same as decoder but separate for compression pipeline)
const MATMUL_LTR = /* wgsl */`
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

// ─── GPUCompressor class ──────────────────────────────────────────────

export class GPUCompressor {
  constructor(device) {
    this.device = device;
    this.fft = new FFTEngine(device);
    this._initPipelines();
  }

  _initPipelines() {
    const mk = code => {
      const m = this.device.createShaderModule({ code });
      return this.device.createComputePipeline({ layout: 'auto', compute: { module: m, entryPoint: 'main' } });
    };
    this.P = {
      loadChannel: mk(LOAD_CHANNEL),
      absMagnitude: mk(ABS_MAGNITUDE),
      histogram: mk(HISTOGRAM),
      survivalMask: mk(SURVIVAL_MASK),
      prefixSumLocal: mk(PREFIX_SUM_LOCAL),
      prefixSumAdd: mk(PREFIX_SUM_ADD),
      thresholdCompact: mk(THRESHOLD_COMPACT),
      outerDiag: mk(OUTER_DIAG),
      alsGram: mk(ALS_GRAM),
      alsRhs: mk(ALS_RHS),
      batchSolve: mk(BATCH_SOLVE),
      computeResidual: mk(COMPUTE_RESIDUAL),
      reduceSum: mk(REDUCE_SUM),
      matmulLTR: mk(MATMUL_LTR),
    };
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

  _uniformMixed(values, types) {
    // values: array of numbers, types: array of 'u32' or 'f32'
    const size = Math.ceil(values.length * 4 / 16) * 16;
    const buf = new ArrayBuffer(size);
    const dv = new DataView(buf);
    for (let i = 0; i < values.length; i++) {
      if (types[i] === 'f32') {
        dv.setFloat32(i * 4, values[i], true);
      } else {
        dv.setUint32(i * 4, values[i], true);
      }
    }
    return this._upload(new Uint8Array(buf), GPUBufferUsage.UNIFORM);
  }

  _bg(pipeline, entries) {
    return this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: entries.map((buf, i) => ({ binding: i, resource: { buffer: buf } }))
    });
  }

  _createBuf(size, usage = GPUBufferUsage.STORAGE) {
    return this.device.createBuffer({ size, usage: usage | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  }

  // Read back a GPU buffer to CPU
  async _readback(buf, size) {
    const staging = this.device.createBuffer({
      size,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(buf, 0, staging, 0, size);
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const data = new Uint8Array(staging.getMappedRange()).slice();
    staging.unmap();
    staging.destroy();
    return data;
  }

  // Load RGBA image pixels into a per-channel f32 buffer on GPU
  loadChannelToGPU(rgbaPixels, n, k, channel) {
    const total = n * k;
    // Pack RGBA as u32
    const u32View = new Uint32Array(rgbaPixels.buffer, rgbaPixels.byteOffset, total);
    const bRGBA = this._upload(u32View);
    const bOut = this._createBuf(total * 4);
    const uParams = this._uniformMixed([total, channel, k], ['u32', 'u32', 'u32']);

    const enc = this.device.createCommandEncoder();
    const p = enc.beginComputePass();
    p.setPipeline(this.P.loadChannel);
    p.setBindGroup(0, this._bg(this.P.loadChannel, [uParams, bRGBA, bOut]));
    p.dispatchWorkgroups(...dims(total));
    p.end();
    this.device.queue.submit([enc.finish()]);

    bRGBA.destroy();
    uParams.destroy();
    return bOut;
  }

  // Compute forward 2D DCT on GPU
  // Returns buffer with DCT coefficients (n×k, f32)
  async computeForwardDCT(channelBuf, n, k) {
    const total = n * k;
    const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC;

    const bufOut = this._createBuf(total * 4);
    const bufScratch = this._createBuf(total * 4);
    const maxDim = Math.max(n, k);
    const complexSize = total * 8; // vec2<f32>
    const complexA = this._createBuf(complexSize);
    const complexB = this._createBuf(complexSize);

    const tmpUniforms = [];
    const enc = this.device.createCommandEncoder();
    this.fft.encode2DDCT(enc, n, k, channelBuf, bufOut, bufScratch, complexA, complexB, tmpUniforms);
    this.device.queue.submit([enc.finish()]);

    // Cleanup scratch
    bufScratch.destroy();
    complexA.destroy();
    complexB.destroy();
    for (const u of tmpUniforms) u.destroy();

    return bufOut;
  }

  // Top-k selection via histogram-based thresholding
  // Returns { indices: Uint32Array, values: Float32Array, nnz: number }
  async selectTopK(dctBuf, total, targetNNZ) {
    // 1. Compute absolute magnitudes
    const absBuf = this._createBuf(total * 4);
    {
      const enc = this.device.createCommandEncoder();
      const p = enc.beginComputePass();
      p.setPipeline(this.P.absMagnitude);
      p.setBindGroup(0, this._bg(this.P.absMagnitude, [dctBuf, absBuf]));
      p.dispatchWorkgroups(...dims(total));
      p.end();
      this.device.queue.submit([enc.finish()]);
    }

    // 2. Find max value (read back and scan — small cost vs GPU complexity)
    const absData = new Float32Array((await this._readback(absBuf, total * 4)).buffer);
    let maxVal = 0;
    for (let i = 0; i < absData.length; i++) {
      if (absData[i] > maxVal) maxVal = absData[i];
    }

    if (maxVal === 0 || targetNNZ >= total) {
      absBuf.destroy();
      // Return all indices
      const indices = new Uint32Array(total);
      for (let i = 0; i < total; i++) indices[i] = i;
      const dctData = new Float32Array((await this._readback(dctBuf, total * 4)).buffer);
      return { indices, values: dctData, nnz: total };
    }

    // 3. Build histogram on GPU
    const histBuf = this._createBuf(1024 * 4);
    // Clear histogram
    {
      const zeros = new Uint32Array(1024);
      this.device.queue.writeBuffer(histBuf, 0, zeros);
    }
    const uHist = this._uniformMixed([total, maxVal], ['u32', 'f32']);
    {
      const enc = this.device.createCommandEncoder();
      const p = enc.beginComputePass();
      p.setPipeline(this.P.histogram);
      p.setBindGroup(0, this._bg(this.P.histogram, [uHist, absBuf, histBuf]));
      p.dispatchWorkgroups(...dims(total));
      p.end();
      this.device.queue.submit([enc.finish()]);
    }

    // 4. Read histogram, find threshold on CPU
    const histData = new Uint32Array((await this._readback(histBuf, 1024 * 4)).buffer);
    let cumCount = 0;
    let thresholdBin = 1023;
    // Scan from highest bin down to find threshold that keeps ~targetNNZ coefficients
    for (let b = 1023; b >= 0; b--) {
      cumCount += histData[b];
      if (cumCount >= targetNNZ) {
        thresholdBin = b;
        break;
      }
    }
    // Convert bin back to magnitude threshold
    const threshold = (Math.pow(2, thresholdBin / 1023) - 1) * maxVal;

    // 5. Compute survival mask
    const maskBuf = this._createBuf(total * 4);
    const uThresh = this._uniformMixed([total, threshold], ['u32', 'f32']);
    {
      const enc = this.device.createCommandEncoder();
      const p = enc.beginComputePass();
      p.setPipeline(this.P.survivalMask);
      p.setBindGroup(0, this._bg(this.P.survivalMask, [uThresh, dctBuf, maskBuf]));
      p.dispatchWorkgroups(...dims(total));
      p.end();
      this.device.queue.submit([enc.finish()]);
    }

    // 6. Prefix sum of mask (CPU for simplicity — mask is just u32 array)
    const maskData = new Uint32Array((await this._readback(maskBuf, total * 4)).buffer);
    const prefixData = new Uint32Array(total);
    let nnz = 0;
    for (let i = 0; i < total; i++) {
      prefixData[i] = nnz;
      nnz += maskData[i];
    }

    // 7. Compact on GPU
    const prefixBuf = this._upload(prefixData);
    const outIndices = this._createBuf(nnz * 4);
    const outValues = this._createBuf(nnz * 4);
    {
      const enc = this.device.createCommandEncoder();
      const p = enc.beginComputePass();
      p.setPipeline(this.P.thresholdCompact);
      p.setBindGroup(0, this._bg(this.P.thresholdCompact, [uThresh, dctBuf, prefixBuf, outIndices, outValues]));
      p.dispatchWorkgroups(...dims(total));
      p.end();
      this.device.queue.submit([enc.finish()]);
    }

    // Read back results
    const indices = new Uint32Array((await this._readback(outIndices, nnz * 4)).buffer);
    const values = new Float32Array((await this._readback(outValues, nnz * 4)).buffer);

    // Cleanup
    absBuf.destroy();
    histBuf.destroy();
    maskBuf.destroy();
    prefixBuf.destroy();
    outIndices.destroy();
    outValues.destroy();
    uHist.destroy();
    uThresh.destroy();

    return { indices, values, nnz };
  }

  // Run ALS iteration on GPU
  // C: f32 buffer (n×k), M: f32 buffer (n×k)
  // L: f32 buffer (layers×n), R: f32 buffer (layers×k)
  // Returns { L, R, residual } as GPU buffers (L, R) and scalar (residual)
  async alsIteration(C_buf, M_buf, L_buf, R_buf, n, k, layers) {
    const total = n * k;
    const layers2 = layers * layers;
    const tmp = [];

    // ─── Solve for R given L ───
    // 1. outer_diag(L)
    const outerL = this._createBuf(layers2 * n * 4);
    const uOuterL = this._uniform(new Uint32Array([n, layers]));
    tmp.push(uOuterL);
    {
      const enc = this.device.createCommandEncoder();
      const p = enc.beginComputePass();
      p.setPipeline(this.P.outerDiag);
      p.setBindGroup(0, this._bg(this.P.outerDiag, [uOuterL, L_buf, outerL]));
      p.dispatchWorkgroups(...dims(n));
      p.end();
      this.device.queue.submit([enc.finish()]);
    }

    // 2. ALS_GRAM(outerL, C) → EtE_R (layers2 × k)
    const EtE_R = this._createBuf(layers2 * k * 4);
    const uGramR = this._uniformMixed([n, k, layers, layers2], ['u32', 'u32', 'u32', 'u32']);
    tmp.push(uGramR);
    {
      const enc = this.device.createCommandEncoder();
      const p = enc.beginComputePass();
      p.setPipeline(this.P.alsGram);
      p.setBindGroup(0, this._bg(this.P.alsGram, [uGramR, outerL, C_buf, EtE_R]));
      p.dispatchWorkgroups(...dims(layers2 * k));
      p.end();
      this.device.queue.submit([enc.finish()]);
    }

    // 3. ALS_RHS(L, C, M) → Etm_R (layers × k)
    const Etm_R = this._createBuf(layers * k * 4);
    const uRhsR = this._uniform(new Uint32Array([n, k, layers]));
    tmp.push(uRhsR);
    {
      const enc = this.device.createCommandEncoder();
      const p = enc.beginComputePass();
      p.setPipeline(this.P.alsRhs);
      p.setBindGroup(0, this._bg(this.P.alsRhs, [uRhsR, L_buf, C_buf, M_buf, Etm_R]));
      p.dispatchWorkgroups(...dims(layers * k));
      p.end();
      this.device.queue.submit([enc.finish()]);
    }

    // 4. Batch solve → new R
    const newR = this._createBuf(layers * k * 4);
    const uSolveR = this._uniform(new Uint32Array([k, layers]));
    tmp.push(uSolveR);
    {
      const enc = this.device.createCommandEncoder();
      const p = enc.beginComputePass();
      p.setPipeline(this.P.batchSolve);
      p.setBindGroup(0, this._bg(this.P.batchSolve, [uSolveR, EtE_R, Etm_R, newR]));
      p.dispatchWorkgroups(...dims(k));
      p.end();
      this.device.queue.submit([enc.finish()]);
    }

    outerL.destroy();
    EtE_R.destroy();
    Etm_R.destroy();

    // ─── Solve for L given new R ───
    // 5. outer_diag(R)
    const outerR = this._createBuf(layers2 * k * 4);
    const uOuterR = this._uniform(new Uint32Array([k, layers]));
    tmp.push(uOuterR);
    {
      const enc = this.device.createCommandEncoder();
      const p = enc.beginComputePass();
      p.setPipeline(this.P.outerDiag);
      p.setBindGroup(0, this._bg(this.P.outerDiag, [uOuterR, newR, outerR]));
      p.dispatchWorkgroups(...dims(k));
      p.end();
      this.device.queue.submit([enc.finish()]);
    }

    // 6. ALS_GRAM for L: sum over j (columns/k dimension)
    // For L solve: EtE[a,b,i] = sum_j outer_R[a,b,j] * C[i,j]^2
    // This is (layers2, k) @ (k, n) transposed — but we need to handle differently
    // Actually the ALS for L is: we transpose the problem
    // EtE_L[a,b,i] = sum_j R_outer[a,b,j] * C_T[j,i]^2
    // We reuse ALS_GRAM with swapped N and K, and transposed C
    // For simplicity, read back C, transpose, and re-upload
    // Actually, the GEMM is (layers2, k) @ (k, n) where C is accessed as C[i,j] = C_flat[i*k+j]
    // For the L solve, we need sum_j outer_R[a,b,j] * C[i,j]^2 for each i
    // This is a different access pattern. Let's use a dedicated shader or transpose approach.

    // Simplest: transpose C, then use ALS_GRAM with swapped dims
    // C is n×k, C^T is k×n
    const C_T = this._createBuf(total * 4);
    const uTransNK = this._uniform(new Uint32Array([n, k]));
    tmp.push(uTransNK);
    {
      const enc = this.device.createCommandEncoder();
      const p = enc.beginComputePass();
      p.setPipeline(this.fft.pipelines.transpose);
      p.setBindGroup(0, this._bg(this.fft.pipelines.transpose, [uTransNK, C_buf, C_T]));
      p.dispatchWorkgroups(...dims(total));
      p.end();
      this.device.queue.submit([enc.finish()]);
    }

    // Similarly transpose M
    const M_T = this._createBuf(total * 4);
    {
      const enc = this.device.createCommandEncoder();
      const p = enc.beginComputePass();
      p.setPipeline(this.fft.pipelines.transpose);
      p.setBindGroup(0, this._bg(this.fft.pipelines.transpose, [uTransNK, M_buf, M_T]));
      p.dispatchWorkgroups(...dims(total));
      p.end();
      this.device.queue.submit([enc.finish()]);
    }

    const EtE_L = this._createBuf(layers2 * n * 4);
    const uGramL = this._uniformMixed([k, n, layers, layers2], ['u32', 'u32', 'u32', 'u32']);
    tmp.push(uGramL);
    {
      const enc = this.device.createCommandEncoder();
      const p = enc.beginComputePass();
      p.setPipeline(this.P.alsGram);
      p.setBindGroup(0, this._bg(this.P.alsGram, [uGramL, outerR, C_T, EtE_L]));
      p.dispatchWorkgroups(...dims(layers2 * n));
      p.end();
      this.device.queue.submit([enc.finish()]);
    }

    const Etm_L = this._createBuf(layers * n * 4);
    const uRhsL = this._uniform(new Uint32Array([k, n, layers]));
    tmp.push(uRhsL);
    {
      const enc = this.device.createCommandEncoder();
      const p = enc.beginComputePass();
      p.setPipeline(this.P.alsRhs);
      p.setBindGroup(0, this._bg(this.P.alsRhs, [uRhsL, newR, C_T, M_T, Etm_L]));
      p.dispatchWorkgroups(...dims(layers * n));
      p.end();
      this.device.queue.submit([enc.finish()]);
    }

    const newL = this._createBuf(layers * n * 4);
    const uSolveL = this._uniform(new Uint32Array([n, layers]));
    tmp.push(uSolveL);
    {
      const enc = this.device.createCommandEncoder();
      const p = enc.beginComputePass();
      p.setPipeline(this.P.batchSolve);
      p.setBindGroup(0, this._bg(this.P.batchSolve, [uSolveL, EtE_L, Etm_L, newL]));
      p.dispatchWorkgroups(...dims(n));
      p.end();
      this.device.queue.submit([enc.finish()]);
    }

    outerR.destroy();
    C_T.destroy();
    M_T.destroy();
    EtE_L.destroy();
    Etm_L.destroy();

    // ─── Compute residual ───
    // matmul L^T @ R
    const LTR = this._createBuf(total * 4);
    const uMat = this._uniform(new Uint32Array([n, k, layers]));
    tmp.push(uMat);
    {
      const enc = this.device.createCommandEncoder();
      const p = enc.beginComputePass();
      p.setPipeline(this.P.matmulLTR);
      p.setBindGroup(0, this._bg(this.P.matmulLTR, [uMat, newL, newR, LTR]));
      p.dispatchWorkgroups(...dims(total));
      p.end();
      this.device.queue.submit([enc.finish()]);
    }

    // Compute residual
    const numWG = Math.ceil(total / WG);
    const partials = this._createBuf(numWG * 4);
    {
      const enc = this.device.createCommandEncoder();
      const p = enc.beginComputePass();
      p.setPipeline(this.P.computeResidual);
      p.setBindGroup(0, this._bg(this.P.computeResidual, [C_buf, LTR, M_buf, partials]));
      p.dispatchWorkgroups(...dims(total));
      p.end();
      this.device.queue.submit([enc.finish()]);
    }

    // Final reduction
    const uReduce = this._uniform(new Uint32Array([numWG]));
    tmp.push(uReduce);
    {
      const enc = this.device.createCommandEncoder();
      const p = enc.beginComputePass();
      p.setPipeline(this.P.reduceSum);
      p.setBindGroup(0, this._bg(this.P.reduceSum, [uReduce, partials]));
      p.dispatchWorkgroups(1, 1, 1);
      p.end();
      this.device.queue.submit([enc.finish()]);
    }

    const residualData = new Float32Array((await this._readback(partials, 4)).buffer);
    const residual = residualData[0];

    LTR.destroy();
    partials.destroy();
    for (const u of tmp) u.destroy();

    return { L: newL, R: newR, residual };
  }

  // Full compression pipeline for one channel
  // Returns { indices, values, leftDiags, rightDiags, layers, ratio, nnz, psnr }
  async compressChannel(channelBuf, sourceBuf, n, k, layers, targetPSNR, maxIter, ratio, onProgress) {
    const total = n * k;

    // 1. Forward DCT
    onProgress?.('Forward DCT...');
    const dctBuf = await this.computeForwardDCT(channelBuf, n, k);

    // 2. Top-k selection
    const targetNNZ = Math.round(total * ratio);
    onProgress?.(`Top-k selection (${targetNNZ} coefficients)...`);
    const topk = await this.selectTopK(dctBuf, total, targetNNZ);

    // 3. Sparse IDCT: scatter coefficients on CPU, then upload and IDCT
    onProgress?.('Sparse IDCT...');
    const grid = new Float32Array(total);
    for (let i = 0; i < topk.nnz; i++) {
      grid[topk.indices[i]] = topk.values[i];
    }
    const sparseDCT = this._upload(grid);
    const idxBuf = this._upload(topk.indices);
    const valBuf = this._upload(topk.values);

    // IDCT to get C
    const C_buf = this._createBuf(total * 4);
    const scratchBuf = this._createBuf(total * 4);
    const complexA = this._createBuf(total * 8);
    const complexB = this._createBuf(total * 8);
    const tmpUniforms = [];
    {
      const enc = this.device.createCommandEncoder();
      this.fft.encode2DIDCT(enc, n, k, sparseDCT, C_buf, scratchBuf, complexA, complexB, tmpUniforms);
      this.device.queue.submit([enc.finish()]);
    }
    sparseDCT.destroy();
    scratchBuf.destroy();
    complexA.destroy();
    complexB.destroy();
    for (const u of tmpUniforms) u.destroy();

    // 4. Initialize L and R (random)
    onProgress?.('ALS initialization...');
    const L_init = new Float32Array(layers * n);
    const R_init = new Float32Array(layers * k);
    for (let i = 0; i < L_init.length; i++) L_init[i] = (Math.random() - 0.5) * 0.1;
    for (let i = 0; i < R_init.length; i++) R_init[i] = (Math.random() - 0.5) * 0.1;
    let L_buf = this._upload(L_init);
    let R_buf = this._upload(R_init);

    // 5. ALS iterations
    let lastResidual = Infinity;
    for (let iter = 0; iter < maxIter; iter++) {
      onProgress?.(`ALS iteration ${iter + 1}/${maxIter}...`);
      const result = await this.alsIteration(C_buf, sourceBuf, L_buf, R_buf, n, k, layers);

      L_buf.destroy();
      R_buf.destroy();
      L_buf = result.L;
      R_buf = result.R;

      const psnr = 10 * Math.log10(255 * 255 * total / result.residual);
      onProgress?.(`ALS iteration ${iter + 1}/${maxIter}: PSNR=${psnr.toFixed(2)} dB`);

      // Convergence check
      if (Math.abs(lastResidual - result.residual) / Math.max(result.residual, 1e-10) < 1e-4) {
        break;
      }
      lastResidual = result.residual;
    }

    // 6. Read back L, R
    const finalPSNR = 10 * Math.log10(255 * 255 * total / lastResidual);
    const leftDiags = new Float32Array((await this._readback(L_buf, layers * n * 4)).buffer);
    const rightDiags = new Float32Array((await this._readback(R_buf, layers * k * 4)).buffer);

    // Cleanup
    dctBuf.destroy();
    C_buf.destroy();
    L_buf.destroy();
    R_buf.destroy();
    idxBuf.destroy();
    valBuf.destroy();

    return {
      indices: topk.indices,
      values: topk.values,
      leftDiags,
      rightDiags,
      layers,
      ratio,
      nnz: topk.nnz,
      psnr: finalPSNR,
    };
  }
}
