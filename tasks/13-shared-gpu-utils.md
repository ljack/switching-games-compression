# Task 13: Extract Shared GPU Utilities Module

## Status: completed

## Description
`dims()`, `WG`, `_upload()`, `_uniform()`, `_bg()` are triplicated across gpu.js, gpu-fft.js, and gpu-compress.js. The TRANSPOSE shader is duplicated in gpu.js and gpu-fft.js. Extract into a shared `gpu-utils.js` module.

## Steps
1. Create `docs/gpu-utils.js` with:
   - `export const WG = 256`
   - `export function dims(total)`
   - `export class GPUBase` with `_upload()`, `_uniform()`, `_uniformMixed()`, `_bg()`, `_createBuf()`, `_readback()` methods
   - `export const TRANSPOSE_SHADER` (WGSL source)
2. Have `GPUDecoder`, `FFTEngine`, `GPUCompressor` extend `GPUBase` or import utilities
3. Remove duplicate definitions from all three files
4. Remove duplicate TRANSPOSE shader from gpu.js (use FFTEngine's or shared)
5. Verify: demo decode, compression, and FFT all still work

## Risk
- Import order / circular dependency between gpu.js → gpu-fft.js → gpu-utils.js
- FFTEngine is used by both GPUDecoder and GPUCompressor — shared base must not create coupling

## Files
- docs/gpu-utils.js (NEW)
- docs/gpu.js (remove dims, WG, helpers, TRANSPOSE)
- docs/gpu-fft.js (remove dims, WG, helpers)
- docs/gpu-compress.js (remove dims, WG, helpers)
