# Task 12: Security Fixes

## Status: completed

## Description
Combined security findings from Claude audit + GitHub Copilot review.
Some fixes already applied, others pending.

## Fixed (Claude audit — commit a6aab28)
1. **CRITICAL: Decompression bomb** — 256MB limit on decompressed payload
2. **CRITICAL: OOB GPU scatter** — validate all indices < totalPixels
3. **HIGH: Integer overflow** — max dims 32768, totalPixels cap 256M
4. **HIGH: Payload bounds checking** — checkOffset() before every read
5. **MEDIUM: compressedSize vs file size** — validated before slicing
6. **MEDIUM: NaN/Infinity diagonals** — sanitized to 0

## Fixed (from Copilot PR — this session)
7. **MEDIUM: XSS in setInfo()** — fileName injected via innerHTML. Fix: use textContent + DOM construction
8. **LOW: Encoding consistency** — removed `sorted.length > 100` bitmap guard in JS encodeIndices() to match Python behaviour
9. **LOW: Python input validation** — compress() now rejects layers<1, ratio out of range, max_iter<1
10. **LOW: Python type annotations** — Optional[float] for target_psnr, return types added
11. **MEDIUM: Python decompression bomb** — zlib.decompress() replaced with _safe_decompress() using max_length cap (256 MB)
12. **LOW: Delta decode alignment** — existing use of data.slice(4) in decodeIndicesDelta already creates aligned copy; no change needed

## Files changed
- docs/app.js (7, 8)
- switching_compress.py (9, 10, 11)
