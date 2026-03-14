// gpu-utils.js — Shared GPU utilities: workgroup size, dispatch dims, transpose shader

export const WG = 256;

export function dims(total) {
  const g = Math.ceil(total / WG);
  if (g <= 65535) return [g, 1, 1];
  return [65535, Math.ceil(g / 65535), 1];
}

export const TRANSPOSE_SHADER = /* wgsl */`
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
