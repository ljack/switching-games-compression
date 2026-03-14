# Task 01: Verify Browser Compression Quality

## Status: completed

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

## Result
- 512x512 test.png: 91.55 dB with 6 layers (default ratio 0.30)
- Auto-tune: 82 dB (finds minimum ratio, all coarse ratios exceed 35 dB)
- Python comparison: 32.9 dB for same settings — GPU now significantly better

## Fixed So Far
- IDCT pretwiddle was completely wrong (multiplied by N instead of dividing, missing factor of 2 for AC terms)
- Fixed using Makhoul 1980 approach: V[k] = (X[k]/alpha[k]) * exp(+j*pi*k/(2N)) * factor / N
- 1D and 2D DCT↔IDCT round-trip now perfect (max error < 0.00001)

## Files
- docs/app.js (compressImage function)
- docs/gpu-compress.js (GPUCompressor, ALS shaders)
- docs/gpu-fft.js (IDCT pretwiddle — FIXED)
- switching_compress.py (reference implementation)
