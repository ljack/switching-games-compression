# Task 02: Add Adaptive Layer Pruning on GPU

## Status: completed

## Description
Python has `algorithm1_adaptive` which prunes trailing layers that contribute less than 1% improvement.
The GPU compressor always uses all requested layers. Add adaptive pruning to reduce file size.

## Steps
1. After ALS converges in `compressChannel()`, compute per-layer contribution
2. For each layer count from 1..layers, compute residual with that many layers
3. If adding more layers improves less than `min_improvement` (1%), stop
4. Return only the pruned layers in the result
5. Update `encodeSWG3()` to write the actual layer count per channel

## Algorithm (from Python)
```python
for try_layers in range(1, max_layers):
    partial += outer(left[try_layers-1], right[try_layers-1])
    residual = M - C * partial
    err = norm(residual)
    if (err - full_err) / (norm_M + eps) < min_improvement:
        actual_layers = try_layers
        break
```

## Success Criteria
- Channels with simple content use fewer layers (e.g., 2-3 instead of 6)
- Compressed file size decreases
- PSNR remains within 0.5 dB of non-pruned version

## Files
- docs/gpu-compress.js (compressChannel, alsIteration)
- docs/app.js (encodeSWG3)
