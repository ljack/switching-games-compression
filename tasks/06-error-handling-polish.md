# Task 06: Error Handling and Edge Case Polish

## Status: completed

## Description
Harden the browser app against edge cases found during testing.

## Steps
1. Handle compression of very small images (< 32x32) — skip FFT, warn user
2. Handle compression cancellation (user switches tabs mid-compress)
3. Clear progress bar on all error paths (some errors leave it stuck)
4. Validate SWG3 header before attempting decode (magic bytes, version, reasonable dimensions)
5. Add "Cancel" button during compression
6. Handle WebGPU device loss during long operations (GPU timeout)

## Success Criteria
- No stuck progress bars on any error
- Small images compress without crashing
- Invalid .swg files show helpful error message
- Cancel button stops compression cleanly

## Files
- docs/app.js (error handling throughout)
- docs/gpu-compress.js (device loss handling)
- docs/index.html (cancel button)
