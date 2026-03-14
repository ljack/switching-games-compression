# Task 01: Verify Browser Compression Quality

## Status: blocked

## Description
Test that browser compression achieves the target 35 dB PSNR with the default ratio of 0.30.
Compare browser-compressed output with Python-compressed output for the same image.

## Steps
1. Start local server: `cd docs && python3 -m http.server 8091`
2. Open browser to localhost:8091, switch to Compress tab
3. Load test.png (or take a camera photo), compress with defaults (35 dB, 6 layers, 7 iter)
4. Verify PSNR shown is >= 35 dB
5. Download the .swg file from browser
6. Decompress with Python: `python3 switching_compress.py decompress browser-output.swg`
7. Verify Python can read the browser-produced .swg without errors
8. Compare visual quality of browser vs Python compressed results

## Success Criteria
- Browser compression produces valid .swg files
- PSNR >= 35 dB for default settings
- Python can decompress browser-produced .swg files
- No console errors during compression

## Blocker
Multi-layer ALS solver diverges for layers > 1. With 1 layer, compression works (35+ dB on simple images).
The Gram matrix / Cholesky solve in BATCH_SOLVE shader likely has numerical issues with f32 for multiple layers.
Need to debug: OUTER_DIAG → ALS_GRAM → BATCH_SOLVE pipeline for layers=2+ on 512x512 images.

## Fixed So Far
- IDCT pretwiddle was completely wrong (multiplied by N instead of dividing, missing factor of 2 for AC terms)
- Fixed using Makhoul 1980 approach: V[k] = (X[k]/alpha[k]) * exp(+j*pi*k/(2N)) * factor / N
- 1D and 2D DCT↔IDCT round-trip now perfect (max error < 0.00001)

## Files
- docs/app.js (compressImage function)
- docs/gpu-compress.js (GPUCompressor, ALS shaders)
- docs/gpu-fft.js (IDCT pretwiddle — FIXED)
- switching_compress.py (reference implementation)
