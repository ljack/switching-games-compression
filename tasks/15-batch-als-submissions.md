# Task 15: Batch ALS GPU Submissions into Single Command Encoder

## Status: pending

## Description
`alsIteration()` calls `device.queue.submit()` 12 times per iteration — once per compute pass.
Each submit is a CPU-GPU synchronization point. These are all sequential dependencies that could
be encoded into a single command encoder with multiple compute passes, submitted once.

## Current Flow (12 submits per iteration)
1. outerDiag(L) → submit
2. alsGram(R) → submit
3. alsRhs(R) → submit
4. batchSolve(R) → submit
5. outerDiag(R) → submit
6. transpose(C) → submit
7. transpose(M) → submit
8. alsGram(L) → submit
9. alsRhs(L) → submit
10. batchSolve(L) → submit
11. matmulLTR → submit
12. computeResidual + reduceSum → submit

## Target Flow (1 submit per iteration)
```javascript
const enc = device.createCommandEncoder();
// All 12 compute passes encoded sequentially
const p1 = enc.beginComputePass(); /* outerDiag(L) */ p1.end();
const p2 = enc.beginComputePass(); /* alsGram(R) */ p2.end();
// ... etc
device.queue.submit([enc.finish()]);
// Only await at the end for residual readback
```

## Steps
1. Refactor `alsIteration()` to accept a single command encoder
2. Encode all compute passes sequentially on the same encoder
3. Submit once at the end
4. Only the residual readback needs to happen after submission
5. Verify: same PSNR results, faster wall-clock time

## Savings
- 12 submit calls → 1 per iteration
- 84 submits → 7 per channel (one per iteration)
- Eliminates ~77 CPU-GPU fence points per channel

## Risk
- WebGPU may have limits on number of passes per command encoder (unlikely for 12)
- Must ensure compute pass barriers are correct (WebGPU handles this automatically between passes)

## Files
- docs/gpu-compress.js (alsIteration)
