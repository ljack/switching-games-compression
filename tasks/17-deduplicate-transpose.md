# Task 17: Deduplicate TRANSPOSE Shader

## Status: completed

## Description
The TRANSPOSE WGSL shader is defined identically in two files:
- `gpu.js` line 39 as `TRANSPOSE`
- `gpu-fft.js` line 25 as `TRANSPOSE_SHADER`

Additionally, `gpu-compress.js` reaches into FFTEngine internals via `this.fft.pipelines.transpose`
to reuse the pipeline — a leaky abstraction.

## Steps
1. If Task 13 (shared gpu-utils.js) is done first, put TRANSPOSE_SHADER there
2. If standalone: remove TRANSPOSE from gpu.js, import from gpu-fft.js or expose via FFTEngine method
3. In gpu-compress.js, replace `this.fft.pipelines.transpose` access with `this.fft.encodeTranspose()`
   (the public method already exists and wraps the pipeline correctly)
4. Verify: demo decode + compression still work

## Dependency
- Best done as part of Task 13 (shared module). If Task 13 is skipped, can be done standalone.

## Files
- docs/gpu.js (remove TRANSPOSE, import or use FFTEngine)
- docs/gpu-fft.js (export TRANSPOSE_SHADER or keep as-is)
- docs/gpu-compress.js (use encodeTranspose() instead of pipelines.transpose)
