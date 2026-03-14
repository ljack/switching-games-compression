# Task 10: Optimize Compressed File Size

## Status: completed

## Description
With ratio 0.30, the SWG file is ~95% of raw size. The PSNR is 60-90 dB — way above the
35 dB target. The auto-tune should find a much lower ratio to reduce file size while
maintaining target quality. Also explore encoding improvements.

## Analysis
The issue is that even the smallest coarse ratio (0.005 = 0.5% of coefficients) already
exceeds 35 dB when the IDCT and ALS are working correctly. This means ratio 0.30 is
massive overkill — we're keeping 30% of coefficients when 0.5% suffices.

## Steps

### 1. Fix auto-tune to actually minimize file size
- The coarse ratios [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50] start at 0.005
- If 0.005 already hits 35 dB, auto-tune picks it — good
- But the file size is dominated by diagonals (2 × layers × (n+k) × 2 bytes for f16)
  plus indices and values for the sparse DCT coefficients
- With ratio 0.005 on 512×512: 0.005 × 262144 = 1310 coefficients
  = 1310 × 4 (indices) + 1310 × 2 (values) = 7860 bytes
  Plus 2 × 6 × (512+512) × 2 = 24576 bytes for diags
  Total ~32KB per channel, ~96KB for RGB — that's 12.5% of raw!

### 2. Lower default ratio when not auto-tuning
- Change default from 0.30 to 0.05 (5%)
- This is still generous but produces much smaller files
- User can always increase if needed

### 3. Encoding improvements
- Current: delta-encode indices, delta-encode int16 values
- Could add: run-length encoding for long zero runs in delta indices
- Could add: variable-length encoding for diagonals (many near zero)
- Biggest win: just lowering the ratio

### 4. Show estimated file size before compression
- After image is loaded, show estimate: "Estimated SWG3 size: ~X KB at ratio 0.30"
- Update when slider changes

## Success Criteria
- Default compression produces files < 20% of raw size at 35+ dB
- Auto-tune with 35 dB target produces files < 10% of raw size
- Estimated size shown before compression starts

## Files
- docs/app.js (compressImage default ratio, auto-tune, size estimation)
- docs/gpu-compress.js (compressChannel)
