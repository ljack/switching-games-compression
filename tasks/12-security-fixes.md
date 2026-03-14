# Task 12: Security Fixes

## Status: pending

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

## Pending (from Copilot PR #1)
7. **MEDIUM: XSS in setInfo()** — fileName injected via innerHTML. Fix: use textContent + DOM construction
8. **LOW: Encoding consistency** — encodeIndices bitmap guard `sorted.length > 100` absent in Python. Remove guard.
9. **LOW: Python input validation** — compress() should reject layers<1, ratio out of range, max_iter<1
10. **LOW: Python type annotations** — Optional[float] for target_psnr, return types

## Pending (not yet fixed)
11. **MEDIUM: Python decompression bomb** — zlib.decompress() has no size limit. Need iterative decompression with cap.
12. **LOW: Delta decode alignment** — Uint16Array from subarray could fail if implementation changes from slice to subarray

## Plan
- Apply fixes 7-8 from Copilot PR (cherry-pick or manual)
- Apply fixes 9-10 to Python CLI
- Add fix 11 to Python decompressor
- Consider fix 12 (defensive)

## Files
- docs/app.js (7, 8)
- switching_compress.py (9, 10, 11)
