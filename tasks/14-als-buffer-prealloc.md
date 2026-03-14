# Task 14: Pre-allocate ALS Iteration Buffers

## Status: completed

## Description
`alsIteration()` in gpu-compress.js creates ~15 GPU buffers and ~8 uniform buffers per call.
With `maxIter=7`, that's ~105 buffer creates/destroys per channel, ~315 per image.
GPU buffer allocation involves kernel calls and memory mapping — expensive.

## Steps
1. Compute all buffer sizes from `n, k, layers` (they're deterministic):
   - `outerL/outerR`: layers^2 * max(n,k) * 4
   - `EtE_R/EtE_L`: layers^2 * max(n,k) * 4
   - `Etm_R/Etm_L`: layers * max(n,k) * 4
   - `newR/newL`: layers * max(n,k) * 4
   - `C_T/M_T`: n*k * 4
   - `LTR`: n*k * 4
   - `partials`: ceil(n*k/WG) * 4
2. Create an `ALSBuffers` object in `compressChannel()` before the iteration loop
3. Pass it into `alsIteration()` instead of creating/destroying each time
4. Destroy all buffers once after the loop completes
5. Uniform buffers with fixed content (e.g., `uTransNK`, `uOuterL`) can also be created once

## Savings
- ~315 buffer allocations → ~15 (one-time)
- ~315 buffer deallocations → ~15 (one-time)
- Uniform buffers: ~56 → ~8

## Risk
- Buffer reuse requires clearing between iterations (some buffers accumulate via atomics)
- Must ensure no read-after-write hazard between iterations

## Files
- docs/gpu-compress.js (alsIteration, compressChannel)
