# Task 16: GPU Parallel Reduction for selectTopK Max-Finding

## Status: pending

## Description
`selectTopK()` in gpu-compress.js reads back the entire absolute magnitude array to CPU
just to find the maximum value. For a 4096x4096 image, that's 64 MB transferred across
the PCIe bus. A GPU-side parallel max reduction would eliminate this readback.

## Current Flow
```javascript
// Line 605: reads back ALL magnitudes to CPU
const absData = new Float32Array((await this._readback(absBuf, total * 4)).buffer);
let maxMag = 0;
for (let i = 0; i < absData.length; i++) maxMag = Math.max(maxMag, absData[i]);
```

## Target Flow
```javascript
// GPU-side max reduction (similar to existing REDUCE_SUM)
const maxBuf = this.gpuMaxReduce(absBuf, total);
const maxData = new Float32Array((await this._readback(maxBuf, 4)).buffer);
const maxMag = maxData[0];
```

## Steps
1. Create `REDUCE_MAX` WGSL shader (similar to existing `REDUCE_SUM` but with `max()` instead of `+`)
   - Workgroup-level: `wg_vals[lid.x] = max(wg_vals[lid.x], wg_vals[lid.x + s])`
   - Final reduction: single workgroup reduces partial maxes
2. Create pipeline for REDUCE_MAX in `_initPipelines()`
3. Replace the readback + CPU loop in `selectTopK()` with GPU reduction
4. Only read back 4 bytes (the single max value) instead of `total * 4`

## Savings
- Readback: 64 MB → 4 bytes (for 4096x4096)
- CPU work: O(n*k) max-finding → 0
- GPU work: O(n*k / WG) + O(log(n*k/WG)) — negligible

## Also Consider
- The histogram shader (line 616) also needs the max value — same readback could feed both
- Could combine max reduction + histogram into a single pass

## Files
- docs/gpu-compress.js (selectTopK, new REDUCE_MAX shader)
