# Task 11: Verify Browser-Compressed SWG Decodes Correctly in Viewer

## Status: completed

## Description
After compression, `startCompression()` calls `decodeAndRender(result.data.buffer)` to
show the compressed result in the viewer. Verify this decode is pixel-correct — that
the viewer tab shows the same image the compressor computed, and that re-loading the
downloaded .swg file produces identical output.

## Steps

### 1. Self-consistency test
- Compress test.png in browser
- Note the displayed PSNR (e.g., 60 dB)
- Download the .swg file
- Switch to Viewer tab
- Load the downloaded .swg
- Compare: viewer should show identical image to what compress tab showed
- Info bar should show same dimensions, compressed size, decode time

### 2. Pixel-level verification (automated)
- In browser console, after compression:
  - Save `decodedImageData` from the auto-decode
  - Download .swg, re-load it, compare pixel-by-pixel
  - Max pixel difference should be 0 (identical decode)

### 3. Check decode path covers all SWG3 fields
- Verify `parseSWG3()` handles:
  - Variable layer counts per channel (from adaptive pruning)
  - All index encoding modes (bitmap=0, delta-u8=1, delta-u16=2, delta-u32=3)
  - Scale factor for dequantization
  - Float16 diagonal values

### 4. Test edge cases
- 1 layer compression → decode
- Very small image (32×32) → decode
- Very low ratio (0.005) → decode
- Maximum layers (12) → decode

## Success Criteria
- Downloaded .swg re-loads identically in viewer
- No decode errors for any valid compression settings
- All index encoding modes work
- Variable layer counts per channel decode correctly

## Result
- Compress → download → re-load in viewer: **pixel-identical** (maxDiff = 0)
- 512×512 dimensions preserved, no errors
- Adaptive layer pruning (variable layers per channel) decoded correctly

## Files
- docs/app.js (startCompression, decodeAndRender, parseSWG3)
- docs/gpu.js (decodeChannel)
