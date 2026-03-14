# Task 08: Browser ↔ Python Round-Trip Verification

## Status: completed

## Description
Verify that .swg files produced by the browser compressor can be decoded by the Python
decompressor, and that Python-compressed files decode correctly in the browser viewer.

## Steps
1. Browser → Python:
   - Compress test.png in browser (512x512, 6 layers, ratio 0.30)
   - Download the .swg file
   - Run `python3 switching_compress.py decompress browser.swg browser_out.png`
   - Compare browser_out.png with test.png — compute PSNR
   - If Python fails to parse, debug the SWG3 binary format differences

2. Python → Browser:
   - Compress test.png with Python: `python3 switching_compress.py compress test.png python.swg --layers 6 --dct-ratio 0.30`
   - Load python.swg in browser viewer
   - Verify it decodes without errors and image looks correct

3. Format compatibility checks:
   - Verify header fields match: magic, version, dimensions, channels, layers
   - Verify per-channel payload: layer count, ratio, nnz, index encoding mode, scale, delta-encoded values, float16 diags
   - Check endianness consistency (both should be little-endian)

## Potential Issues
- Browser uses f32 for diags before converting to f16; Python uses f64 → f16. Small precision differences expected.
- Index encoding: browser may choose different delta mode (u8/u16/u32) than Python for same data
- Zlib compression level may differ (browser CompressionStream vs Python zlib level 9)

## Success Criteria
- Python can decompress browser .swg without errors
- Browser can decode Python .swg without errors
- Visual quality is comparable (PSNR within 3 dB of reported value)

## Result
- Browser → Python: Decoded OK, PSNR 49.18 dB (3 pixel max diff from f16 quantization)
- Python → Browser: Decoded OK, rendered 512×512, no errors
- Both directions fully compatible

## Files
- docs/app.js (encodeSWG3)
- switching_compress.py (compress, decompress)
