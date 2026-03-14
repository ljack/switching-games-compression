# Task 03: Add Kahan Compensated Summation in GPU Shaders

## Status: completed

## Description
The ALS solver uses f32 on GPU (vs f64 in Python). For large images, accumulated rounding
in the Gram matrix (EtE) and RHS (Etm) GEMM-like operations can degrade PSNR.
Add Kahan compensated summation to improve f32 precision.

## Steps
1. Modify ALS_GRAM shader: accumulate `sum_j outer[a,b,j] * C[i,j]^2` with Kahan
2. Modify ALS_RHS shader: accumulate `sum_j diag[layer,j] * C[i,j] * M[i,j]` with Kahan
3. Modify COMPUTE_RESIDUAL shader: accumulate squared differences with Kahan
4. Test on 512x512 image — compare PSNR with and without Kahan
5. Test on larger images where f32 error is more pronounced

## Kahan Pattern
```wgsl
var sum: f32 = 0.0;
var comp: f32 = 0.0;
for (...) {
    let y = val - comp;
    let t = sum + y;
    comp = (t - sum) - y;
    sum = t;
}
```

## Success Criteria
- PSNR improves by >= 0.5 dB on large images (1000x1000+)
- No performance regression > 10%

## Files
- docs/gpu-compress.js (ALS_GRAM, ALS_RHS, COMPUTE_RESIDUAL shaders)
