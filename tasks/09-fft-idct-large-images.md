# Task 09: FFT IDCT for Large Images via Row Batching

## Status: completed

## Description
The FFT-based IDCT requires complex buffers (2x f32 per element = 8 bytes/pixel).
For 8192×6144 images, that's 384MB per complex buffer — exceeds the 256MB maxBufferSize
on most GPUs. Currently falls back to shared-mem IDCT (~15s). Goal: sub-second decode
via batched FFT.

## Approach: Row-Batched FFT
Instead of allocating one complex buffer for the entire image, process rows in batches
that fit within maxBufferSize.

### Steps
1. In `gpu.js` `ensureBuffers()`, compute max rows that fit:
   ```
   maxComplexSize = device.limits.maxBufferSize
   bytesPerRow = k * 8  // vec2<f32> per element
   maxRows = floor(maxComplexSize / bytesPerRow)
   batchSize = min(n, maxRows)
   ```
2. Allocate complex buffers for `batchSize × k` instead of `n × k`

3. In `decodeChannel()`, process the 2D IDCT in batches:
   - **Column IDCT** (along n dimension): This is the hard part — columns span all rows.
     Options:
     a. Transpose to make columns into rows → batch FFT → transpose back
     b. Process column IDCT via shared-mem (already works) and only use FFT for row IDCT
   - **Row IDCT** (along k dimension): Easy to batch — each row is independent.
     Process `batchSize` rows at a time.

4. Hybrid approach (simplest):
   - Column IDCT: use existing shared-mem kernel (handles any size)
   - Row IDCT: use FFT in batches
   - This halves the decode time since row IDCT is half the work

5. Full FFT approach (optimal):
   - Column IDCT: transpose full image (n×k → k×n), then batch-FFT rows of transposed
   - But transpose itself needs n×k buffer... which fits in STORAGE (4 bytes/element, not 8)
   - So: transpose → batch-FFT-IDCT of k rows of n elements → transpose back → batch-FFT-IDCT of n rows of k elements

### Memory Budget (8192×6144 example)
- Real buffers (f32): 192MB each — fits in 256MB
- Complex buffers (vec2<f32>): 384MB — doesn't fit
- Batch of 2048 rows: 2048 × 6144 × 8 = 96MB — fits comfortably
- Transpose buffer (f32): 192MB — fits

## Success Criteria
- 8192×6144 demo decodes in < 3s (vs ~15s with shared-mem)
- No GPU OOM errors
- Badge shows "WebGPU (FFT)" for large images
- Exact same pixel output as shared-mem path

## Result
- 8192×6144 demo: 1041 ms decode (was ~15000 ms with shared-mem) — **14.4x speedup**
- Badge now shows "WebGPU (FFT)" for large images
- Batched path: complex buffers sized to 75% of maxStorageBufferBindingSize
- Verified: 8x8, 64x64 batched round-trip max error < 0.0002

## Files
- docs/gpu.js (ensureBuffers, decodeChannel — FFT enabled even without full complex buffers)
- docs/gpu-fft.js (_encodeIDCTSlice, _bgSlice, encode2DIDCT batched path)
