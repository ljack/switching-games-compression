# Task 04: Live Preview During ALS Iterations

## Status: completed

## Description
Show the reconstructed image updating after each ALS iteration so the user can
see quality improving in real-time. Currently only PSNR text updates.

## Steps
1. After each ALS iteration in `compressChannel()`, reconstruct the image on GPU
2. Read back the reconstructed channel to CPU
3. After all 3 channels complete an iteration, composite RGB and draw to preview canvas
4. Restructure compression loop: iterate all channels together instead of one-at-a-time
5. Add a small preview canvas in the compress results area

## Implementation Notes
- Reconstruction: `approx = C * (L^T @ R)` — reuse MATMUL_LTR shader + element-wise multiply
- Need to interleave channel iterations: iter1(R,G,B) -> preview -> iter2(R,G,B) -> preview
- This is a significant restructure of compressChannel flow

## Success Criteria
- Preview image updates visibly after each ALS iteration
- User can see quality improving from blurry to sharp
- Total compression time doesn't increase by more than 20%

## Files
- docs/app.js (compressImage, compress UI)
- docs/gpu-compress.js (compressChannel)
- docs/index.html (preview canvas element)
