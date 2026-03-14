// gpu-fft.js — FFT/DCT/IDCT shader code and dispatch logic
// Replaces O(N^2) direct IDCT with O(N log N) FFT-based approach.
//
// DCT-II via FFT:
//   1. Reorder: y[m] = x[2m] (m < N/2), y[m] = x[2(N-m)-1] (m >= N/2)
//   2. N-point complex FFT of y
//   3. Twiddle: X[k] = alpha[k] * Re(Y[k] * exp(-j*pi*k / 2N))
//
// IDCT (DCT-III) via IFFT:
//   1. Pre-twiddle: Y[k] = (1/alpha[k]) * X[k] * exp(j*pi*k / 2N)
//   2. N-point complex IFFT of Y
//   3. De-reorder: x[2m] = Re(y[m]) (m < N/2), x[2(N-m)-1] = Re(y[m]) (m >= N/2)

const WG = 256;

function dims(total) {
  const g = Math.ceil(total / WG);
  if (g <= 65535) return [g, 1, 1];
  return [65535, Math.ceil(g / 65535), 1];
}

// ─── Shaders ──────────────────────────────────────────────────────────

// Transpose (reused from gpu.js but defined locally for modularity)
const TRANSPOSE_SHADER = /* wgsl */`
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

// IDCT pre-twiddle: f32 DCT coefficients → vec2<f32> complex for IFFT
// For each row of N elements, produce N complex values:
//   Y[0] = X[0] / alpha0
//   Y[k] = X[k] / alpha * exp(j * pi * k / (2N))    for k > 0
// where alpha0 = sqrt(1/N), alpha = sqrt(2/N)
const IDCT_PRETWIDDLE = /* wgsl */`
struct P { num_rows: u32, N: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<vec2<f32>>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
  let idx = gid.y * (nwg.x * ${WG}u) + gid.x;
  let total = p.num_rows * p.N;
  if (idx >= total) { return; }
  let row = idx / p.N;
  let k = idx % p.N;
  let val = input[idx];
  let N_f = f32(p.N);
  let alpha0 = sqrt(1.0 / N_f);
  let alpha_k = sqrt(2.0 / N_f);
  let scale = select(val / alpha_k, val / alpha0, k == 0u);
  // Multiply by N for IFFT normalization (our IFFT doesn't divide by N)
  let scaled = scale * N_f;
  let angle = 3.14159265358979 * f32(k) / (2.0 * N_f);
  let re = scaled * cos(angle);
  let im = scaled * sin(angle);
  output[row * p.N + k] = vec2<f32>(re, im);
}`;

// IDCT de-reorder: complex IFFT output → f32 spatial values
// x[2m] = Re(y[m])        for m < N/2
// x[2(N-m)-1] = Re(y[m])  for m >= N/2
const IDCT_DEORDER = /* wgsl */`
struct P { num_rows: u32, N: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> input: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
  let idx = gid.y * (nwg.x * ${WG}u) + gid.x;
  let total = p.num_rows * p.N;
  if (idx >= total) { return; }
  let row = idx / p.N;
  let m = idx % p.N;
  let val = input[row * p.N + m].x;  // Real part
  let half = p.N / 2u;
  var out_pos: u32;
  if (m < half) {
    out_pos = 2u * m;
  } else {
    out_pos = 2u * (p.N - m) - 1u;
  }
  output[row * p.N + out_pos] = val;
}`;

// DCT reorder: f32 spatial values → complex for FFT
// y[m] = x[2m]           for m < N/2
// y[m] = x[2(N-m)-1]     for m >= N/2
const DCT_REORDER = /* wgsl */`
struct P { num_rows: u32, N: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<vec2<f32>>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
  let idx = gid.y * (nwg.x * ${WG}u) + gid.x;
  let total = p.num_rows * p.N;
  if (idx >= total) { return; }
  let row = idx / p.N;
  let m = idx % p.N;
  let half = p.N / 2u;
  var src_pos: u32;
  if (m < half) {
    src_pos = 2u * m;
  } else {
    src_pos = 2u * (p.N - m) - 1u;
  }
  output[row * p.N + m] = vec2<f32>(input[row * p.N + src_pos], 0.0);
}`;

// DCT post-twiddle: complex FFT output → f32 DCT coefficients
// X[k] = alpha[k] * Re(Y[k] * exp(-j*pi*k / 2N))
const DCT_TWIDDLE = /* wgsl */`
struct P { num_rows: u32, N: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> input: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
  let idx = gid.y * (nwg.x * ${WG}u) + gid.x;
  let total = p.num_rows * p.N;
  if (idx >= total) { return; }
  let row = idx / p.N;
  let k = idx % p.N;
  let c = input[row * p.N + k];
  let N_f = f32(p.N);
  let angle = -3.14159265358979 * f32(k) / (2.0 * N_f);
  // Y[k] * exp(-j*angle) = (re + j*im)(cos(a) + j*sin(a))
  let tw_re = c.x * cos(angle) - c.y * sin(angle);
  let alpha0 = sqrt(1.0 / N_f);
  let alpha_k = sqrt(2.0 / N_f);
  let alpha = select(alpha_k, alpha0, k == 0u);
  output[row * p.N + k] = alpha * tw_re;
}`;

// Stockham auto-sort FFT radix-2 stage (no bit-reversal needed)
// Natural-order input → natural-order output after all stages.
// direction: 1.0 for forward FFT, -1.0 for inverse FFT
const FFT_RADIX2 = /* wgsl */`
struct P { num_rows: u32, N: u32, half_N: u32, stride: u32, direction: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec2<f32>>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
  let idx = gid.y * (nwg.x * ${WG}u) + gid.x;
  let total = p.num_rows * p.half_N;
  if (idx >= total) { return; }
  let row = idx / p.half_N;
  let tid = idx % p.half_N;
  let base = row * p.N;

  // Stockham read: evens at tid, odds at tid + N/2
  let pos = tid % p.stride;
  let grp = tid / p.stride;

  let src_even = base + grp * p.stride + pos;
  let src_odd  = src_even + p.half_N;

  let a = src[src_even];
  let b = src[src_odd];

  // Twiddle
  let angle = p.direction * -6.28318530717959 * f32(pos) / f32(2u * p.stride);
  let tw = vec2<f32>(cos(angle), sin(angle));
  let bt = vec2<f32>(b.x * tw.x - b.y * tw.y, b.x * tw.y + b.y * tw.x);

  // Stockham write: deinterleave into butterfly groups
  let dst0 = base + grp * 2u * p.stride + pos;
  let dst1 = dst0 + p.stride;
  dst[dst0] = a + bt;
  dst[dst1] = a - bt;
}`;

// Stockham auto-sort FFT radix-3 stage for N with factor of 3 (e.g., 6144 = 3 * 2^11)
const FFT_RADIX3 = /* wgsl */`
struct P { num_rows: u32, N: u32, third_N: u32, stride: u32, direction: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec2<f32>>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
  let idx = gid.y * (nwg.x * ${WG}u) + gid.x;
  let total = p.num_rows * p.third_N;
  if (idx >= total) { return; }
  let row = idx / p.third_N;
  let tid = idx % p.third_N;
  let base = row * p.N;

  // Stockham read: 3 inputs spaced N/3 apart
  let pos = tid % p.stride;
  let grp = tid / p.stride;

  let src0 = base + grp * p.stride + pos;
  let src1 = src0 + p.third_N;
  let src2 = src1 + p.third_N;

  let a = src[src0];
  let b_raw = src[src1];
  let c_raw = src[src2];

  // Twiddle factors
  let angle1 = p.direction * -6.28318530717959 * f32(pos) / f32(3u * p.stride);
  let tw1 = vec2<f32>(cos(angle1), sin(angle1));
  let b = vec2<f32>(b_raw.x * tw1.x - b_raw.y * tw1.y, b_raw.x * tw1.y + b_raw.y * tw1.x);

  let angle2 = 2.0 * angle1;
  let tw2 = vec2<f32>(cos(angle2), sin(angle2));
  let c = vec2<f32>(c_raw.x * tw2.x - c_raw.y * tw2.y, c_raw.x * tw2.y + c_raw.y * tw2.x);

  // 3-point DFT
  let w3_re = -0.5;
  let w3_im = p.direction * -0.86602540378444;  // -sqrt(3)/2

  let sum_bc = b + c;
  let diff_bc = b - c;

  // Stockham write: deinterleave into 3 groups
  let out0 = base + grp * 3u * p.stride + pos;
  let out1 = out0 + p.stride;
  let out2 = out1 + p.stride;

  dst[out0] = a + sum_bc;
  dst[out1] = vec2<f32>(
    a.x + w3_re * sum_bc.x - w3_im * diff_bc.y,
    a.y + w3_re * sum_bc.y + w3_im * diff_bc.x
  );
  dst[out2] = vec2<f32>(
    a.x + w3_re * sum_bc.x + w3_im * diff_bc.y,
    a.y + w3_re * sum_bc.y - w3_im * diff_bc.x
  );
}`;

// ─── FFTEngine class ───────────────────────────────────────────────────

export class FFTEngine {
  constructor(device) {
    this.device = device;
    this.pipelines = {};
    this._init();
  }

  _init() {
    const mk = code => {
      const m = this.device.createShaderModule({ code });
      return this.device.createComputePipeline({ layout: 'auto', compute: { module: m, entryPoint: 'main' } });
    };
    this.pipelines = {
      idctPreTwiddle: mk(IDCT_PRETWIDDLE),
      idctDeorder: mk(IDCT_DEORDER),
      dctReorder: mk(DCT_REORDER),
      dctTwiddle: mk(DCT_TWIDDLE),
      fftRadix2: mk(FFT_RADIX2),
      fftRadix3: mk(FFT_RADIX3),
      transpose: mk(TRANSPOSE_SHADER),
    };
  }

  _uniform(u32arr) {
    const pad = new ArrayBuffer(Math.ceil(u32arr.byteLength / 16) * 16);
    new Uint8Array(pad).set(new Uint8Array(u32arr.buffer, u32arr.byteOffset, u32arr.byteLength));
    const b = this.device.createBuffer({
      size: pad.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(b, 0, new Uint8Array(pad));
    return b;
  }

  _uniformF32(data) {
    // data is array of [u32, u32, ..., f32] with the last being float
    const buf = new ArrayBuffer(Math.ceil(data.length * 4 / 16) * 16);
    const dv = new DataView(buf);
    for (let i = 0; i < data.length; i++) {
      if (typeof data[i] === 'number' && i === data.length - 1) {
        dv.setFloat32(i * 4, data[i], true);
      } else {
        dv.setUint32(i * 4, data[i], true);
      }
    }
    const b = this.device.createBuffer({
      size: buf.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(b, 0, new Uint8Array(buf));
    return b;
  }

  _bg(pipeline, entries) {
    return this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: entries.map((buf, i) => ({ binding: i, resource: { buffer: buf } }))
    });
  }

  // Factorize N into radix-3 and radix-2 stages
  _factorize(N) {
    const stages = [];
    let remaining = N;
    // Extract factors of 3 first
    while (remaining % 3 === 0) {
      stages.push(3);
      remaining /= 3;
    }
    // Then factors of 2
    while (remaining > 1) {
      if (remaining % 2 !== 0) throw new Error(`N=${N} is not of form 2^a * 3^b`);
      stages.push(2);
      remaining /= 2;
    }
    return stages;
  }

  // Run FFT stages on encoder. src/dst are complex buffers (vec2<f32>).
  // Returns the buffer containing the result (may be src or dst depending on parity).
  _encodeFFTStages(enc, numRows, N, src, dst, direction, tmpUniforms) {
    const stages = this._factorize(N);
    let curSrc = src;
    let curDst = dst;
    let stride = 1;

    for (const radix of stages) {
      const p = enc.beginComputePass();
      if (radix === 2) {
        const halfN = N / 2;
        const total = numRows * halfN;
        // uniform: num_rows, N, half_N, stride, direction
        const buf = new ArrayBuffer(32);
        const dv = new DataView(buf);
        dv.setUint32(0, numRows, true);
        dv.setUint32(4, N, true);
        dv.setUint32(8, halfN, true);
        dv.setUint32(12, stride, true);
        dv.setFloat32(16, direction, true);
        const uBuf = this.device.createBuffer({
          size: 32,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(uBuf, 0, new Uint8Array(buf));
        tmpUniforms.push(uBuf);

        p.setPipeline(this.pipelines.fftRadix2);
        p.setBindGroup(0, this._bg(this.pipelines.fftRadix2, [uBuf, curSrc, curDst]));
        p.dispatchWorkgroups(...dims(total));
        stride *= 2;
      } else {
        const thirdN = N / 3;
        const total = numRows * thirdN;
        const buf = new ArrayBuffer(32);
        const dv = new DataView(buf);
        dv.setUint32(0, numRows, true);
        dv.setUint32(4, N, true);
        dv.setUint32(8, thirdN, true);
        dv.setUint32(12, stride, true);
        dv.setFloat32(16, direction, true);
        const uBuf = this.device.createBuffer({
          size: 32,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(uBuf, 0, new Uint8Array(buf));
        tmpUniforms.push(uBuf);

        p.setPipeline(this.pipelines.fftRadix3);
        p.setBindGroup(0, this._bg(this.pipelines.fftRadix3, [uBuf, curSrc, curDst]));
        p.dispatchWorkgroups(...dims(total));
        stride *= 3;
      }
      p.end();
      // Ping-pong
      [curSrc, curDst] = [curDst, curSrc];
    }
    // Result is in curSrc (last written was curDst before swap)
    return curSrc;
  }

  // Encode IDCT of rows: input f32 buffer (numRows × N) → output f32 buffer
  // complexA and complexB are scratch vec2<f32> buffers of size numRows * N * 8 bytes
  encodeIDCT(enc, numRows, N, inputBuf, outputBuf, complexA, complexB, tmpUniforms) {
    const total = numRows * N;

    // 1. Pre-twiddle: f32 → complex
    const uPre = this._uniform(new Uint32Array([numRows, N]));
    tmpUniforms.push(uPre);
    {
      const p = enc.beginComputePass();
      p.setPipeline(this.pipelines.idctPreTwiddle);
      p.setBindGroup(0, this._bg(this.pipelines.idctPreTwiddle, [uPre, inputBuf, complexA]));
      p.dispatchWorkgroups(...dims(total));
      p.end();
    }

    // 2. IFFT (direction = -1.0)
    const resultBuf = this._encodeFFTStages(enc, numRows, N, complexA, complexB, -1.0, tmpUniforms);

    // 3. De-reorder: complex → f32
    const uPost = this._uniform(new Uint32Array([numRows, N]));
    tmpUniforms.push(uPost);
    {
      const p = enc.beginComputePass();
      p.setPipeline(this.pipelines.idctDeorder);
      p.setBindGroup(0, this._bg(this.pipelines.idctDeorder, [uPost, resultBuf, outputBuf]));
      p.dispatchWorkgroups(...dims(total));
      p.end();
    }
  }

  // Encode forward DCT of rows: input f32 buffer (numRows × N) → output f32 buffer
  encodeDCT(enc, numRows, N, inputBuf, outputBuf, complexA, complexB, tmpUniforms) {
    const total = numRows * N;

    // 1. Reorder: f32 → complex
    const uPre = this._uniform(new Uint32Array([numRows, N]));
    tmpUniforms.push(uPre);
    {
      const p = enc.beginComputePass();
      p.setPipeline(this.pipelines.dctReorder);
      p.setBindGroup(0, this._bg(this.pipelines.dctReorder, [uPre, inputBuf, complexA]));
      p.dispatchWorkgroups(...dims(total));
      p.end();
    }

    // 2. FFT (direction = 1.0)
    const resultBuf = this._encodeFFTStages(enc, numRows, N, complexA, complexB, 1.0, tmpUniforms);

    // 3. Post-twiddle: complex → f32
    const uPost = this._uniform(new Uint32Array([numRows, N]));
    tmpUniforms.push(uPost);
    {
      const p = enc.beginComputePass();
      p.setPipeline(this.pipelines.dctTwiddle);
      p.setBindGroup(0, this._bg(this.pipelines.dctTwiddle, [uPost, resultBuf, outputBuf]));
      p.dispatchWorkgroups(...dims(total));
      p.end();
    }
  }

  // Encode transpose: f32 buffer (rows × cols) → f32 buffer (cols × rows)
  encodeTranspose(enc, rows, cols, inputBuf, outputBuf, tmpUniforms) {
    const total = rows * cols;
    const uTrans = this._uniform(new Uint32Array([rows, cols]));
    tmpUniforms.push(uTrans);
    const p = enc.beginComputePass();
    p.setPipeline(this.pipelines.transpose);
    p.setBindGroup(0, this._bg(this.pipelines.transpose, [uTrans, inputBuf, outputBuf]));
    p.dispatchWorkgroups(...dims(total));
    p.end();
  }

  // Full 2D IDCT: separable row + column IDCT via transpose
  // input: f32 buffer n×k (row-major DCT coefficients)
  // output: f32 buffer n×k (row-major spatial values)
  // Needs: bufA, bufB (f32, n*k each), complexA, complexB (vec2<f32>, max(n,k)*batch each)
  encode2DIDCT(enc, n, k, inputBuf, outputBuf, scratchBuf, complexA, complexB, tmpUniforms) {
    // Step 1: Column IDCT via transpose
    // Transpose input (n×k) → scratchBuf (k×n)
    this.encodeTranspose(enc, n, k, inputBuf, scratchBuf, tmpUniforms);

    // IDCT rows of scratchBuf (k rows of n elements) → outputBuf
    this.encodeIDCT(enc, k, n, scratchBuf, outputBuf, complexA, complexB, tmpUniforms);

    // Transpose outputBuf (k×n) → scratchBuf (n×k)
    this.encodeTranspose(enc, k, n, outputBuf, scratchBuf, tmpUniforms);

    // Step 2: Row IDCT
    // IDCT rows of scratchBuf (n rows of k elements) → outputBuf
    this.encodeIDCT(enc, n, k, scratchBuf, outputBuf, complexA, complexB, tmpUniforms);
  }

  // Full 2D DCT: separable row + column DCT via transpose
  encode2DDCT(enc, n, k, inputBuf, outputBuf, scratchBuf, complexA, complexB, tmpUniforms) {
    // Step 1: Row DCT
    // DCT rows of inputBuf (n rows of k elements) → scratchBuf
    this.encodeDCT(enc, n, k, inputBuf, scratchBuf, complexA, complexB, tmpUniforms);

    // Transpose scratchBuf (n×k) → outputBuf (k×n)
    this.encodeTranspose(enc, n, k, scratchBuf, outputBuf, tmpUniforms);

    // Step 2: Column DCT (as rows on transposed data)
    // DCT rows of outputBuf (k rows of n elements) → scratchBuf
    this.encodeDCT(enc, k, n, outputBuf, scratchBuf, complexA, complexB, tmpUniforms);

    // Transpose scratchBuf (k×n) → outputBuf (n×k)
    this.encodeTranspose(enc, k, n, scratchBuf, outputBuf, tmpUniforms);
  }
}
