# CLI Reference

All commands are run via `python3 switching_compress.py <command>`.

## compress

Compress an image to SWG3 format.

```bash
python3 switching_compress.py compress <input> <output> [options]
```

**Arguments:**
- `input` — Source image path (any format Pillow supports: JPEG, PNG, BMP, TIFF, WebP, etc.). Converted to RGB internally.
- `output` — Destination `.swg` file path.

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--target-psnr <dB>` | — | Target quality in decibels. Binary-searches DCT ratio per channel to hit this PSNR. Overrides `--dct-ratio`. Recommended for most use. |
| `--dct-ratio <0..1>` | 0.30 | Fraction of 2D DCT coefficients to keep (by magnitude). Lower = smaller file, lower quality. Ignored when `--target-psnr` is set. |
| `--layers <n>` | 6 | Maximum number of diagonal layer pairs (L, R) to solve. More layers can improve quality but increase file size and compress time. |
| `--max-iter <n>` | 7 | ALS (alternating least-squares) iterations per layer. More iterations refine the diagonal matrices but add compute time. 5-10 is typical. |
| `--no-adaptive` | off | Disable adaptive layer pruning. By default, layers that don't improve reconstruction are dropped. This flag forces all `--layers` to be kept. |

**How it works:**

1. Loads the image and splits into R, G, B channels.
2. For each channel:
   - Computes 2D DCT (type II, orthonormal) and keeps the top `dct_ratio` fraction of coefficients by magnitude. This gives the initial approximation C.
   - If `--target-psnr` is set, binary-searches across ratios (0.005 to 0.50) to find the smallest ratio that meets the target PSNR for this channel.
   - Runs Algorithm 1 (alternating least-squares) to find diagonal matrices L and R that minimize ‖M - L^T C R‖. Each iteration solves a batch linear system per row/column.
   - With adaptive mode (default), prunes layers whose contribution is below threshold.
3. Encodes per-channel data: sparse DCT indices (bitmap or delta-encoded, auto-selected for smallest size), delta-coded quantized values (int16), diagonal matrices (float16).
4. Compresses the full payload with zlib (level 9) and writes the SWG3 binary header.

**Output:**

Prints per-channel stats (ratio, layers, PSNR, time) and final compression ratio.

```
$ python3 switching_compress.py compress photo.jpg photo.swg --target-psnr 35
Image: 8192x6144, 3 channels
Max layers: 6, Target PSNR: 35.0 dB, Max iter: 7
  Channel R: ratio=0.0250, layers=5, PSNR=35.2 dB (28.4s)
  Channel G: ratio=0.0200, layers=5, PSNR=35.1 dB (31.2s)
  Channel B: ratio=0.0150, layers=4, PSNR=35.3 dB (38.1s)
Compressed in 97.7s
Original: 16,234,567 bytes
Compressed: 6,201,344 bytes (38.2%)
```

**Tips:**
- `--target-psnr 30` gives aggressive compression (small files, visible artifacts on close inspection).
- `--target-psnr 35` is a good balance for photos.
- `--target-psnr 40` produces near-lossless quality at larger file sizes.
- Fixed `--dct-ratio` is faster because it skips the binary search but doesn't guarantee a specific quality level.

---

## decompress

Decompress an SWG3 file back to an image.

```bash
python3 switching_compress.py decompress <input> <output>
```

**Arguments:**
- `input` — Path to `.swg` file.
- `output` — Destination image path. Format is inferred from extension (`.png`, `.jpg`, `.bmp`, etc.).

**How it works:**

1. Reads the 24-byte SWG3 header (dimensions, channel count, layer config).
2. Decompresses the zlib payload.
3. For each channel:
   - Decodes sparse DCT indices (bitmap or delta) and delta-coded quantized values.
   - Reconstructs the sparse DCT grid and applies inverse 2D DCT to get C.
   - Reads float16 diagonal matrices L and R.
   - Computes the final channel: M = L^T * C * R (element-wise multiplication of outer products with C).
4. Clamps to [0, 255], assembles RGB, saves.

**Output:**

```
$ python3 switching_compress.py decompress photo.swg photo_out.png
Image: 8192x6144, 3 channels, max_layers=6
Saved reconstructed image to photo_out.png
```

---

## evaluate

Compress, decompress, and measure quality in one step. Useful for testing parameters without keeping intermediate files.

```bash
python3 switching_compress.py evaluate <input> [options]
```

**Arguments:**
- `input` — Source image path.

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--target-psnr <dB>` | — | Same as in `compress`. |
| `--dct-ratio <0..1>` | 0.30 | Same as in `compress`. |
| `--layers <n>` | 6 | Same as in `compress`. |
| `--max-iter <n>` | 7 | Same as in `compress`. |
| `--output-dir <path>` | `.` | Directory for temporary and output files. |

**How it works:**

1. Compresses the image to `<basename>.swg` in the output directory.
2. Decompresses back to `<basename>_reconstructed.png`.
3. Computes PSNR (peak signal-to-noise ratio) and SSIM (structural similarity) between original and reconstructed images.
4. Creates a side-by-side comparison image `<basename>_comparison.png` with a gray divider strip.

**Requires:** `scikit-image` (for SSIM calculation).

**Output:**

```
$ python3 switching_compress.py evaluate photo.jpg --target-psnr 35
Image: 8192x6144, 3 channels
Max layers: 6, Target PSNR: 35.0 dB, Max iter: 7
  Channel R: ratio=0.0250, layers=5, PSNR=35.2 dB (28.4s)
  Channel G: ratio=0.0200, layers=5, PSNR=35.1 dB (31.2s)
  Channel B: ratio=0.0150, layers=4, PSNR=35.3 dB (38.1s)
Compressed in 97.7s
Original: 16,234,567 bytes
Compressed: 6,201,344 bytes (38.2%)
Image: 8192x6144, 3 channels, max_layers=6
Saved reconstructed image to photo_reconstructed.png

── Quality Metrics ──
PSNR: 34.82 dB
SSIM: 0.9412
Comparison image: photo_comparison.png
```

**Generated files:**
- `<basename>.swg` — compressed file
- `<basename>_reconstructed.png` — decoded image
- `<basename>_comparison.png` — original and reconstructed side by side

---

## sweep

Run compression at multiple DCT ratios and auto-tuned targets to produce a quality-vs-size table. Useful for understanding the tradeoff space for a specific image.

```bash
python3 switching_compress.py sweep <input> [options]
```

**Arguments:**
- `input` — Source image path.

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--layers <n>` | 6 | Same as in `compress`. |
| `--max-iter <n>` | 7 | Same as in `compress`. |
| `--output-dir <path>` | `.` | Directory for temporary files (cleaned up after each run). |

**How it works:**

1. Compresses the image at 8 fixed ratios: 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50.
2. Then compresses at 4 auto-tuned PSNR targets: 25, 30, 35, 40 dB.
3. For each, measures file size, PSNR, and SSIM (fixed ratios only).
4. Temporary `.swg` and `.png` files are deleted after each measurement.

**Requires:** `scikit-image` (for SSIM calculation).

**Output:**

```
$ python3 switching_compress.py sweep photo.jpg
Image: 8192x6144, 3 channels
PNG size: 16,234,567 bytes
Layers: 6, Max iter: 7

   Ratio       Size   vs PNG     PSNR     SSIM
──────── ────────── ──────── ──────── ────────
   0.005  1,204,531B     0.1x    24.3   0.8234
   0.010  2,589,012B     0.2x    28.0   0.8891
   0.020  4,102,445B     0.3x    31.5   0.9234
   0.050  7,698,321B     0.5x    37.2   0.9612
   0.100 12,045,678B     0.7x    41.1   0.9801
   0.200 18,234,567B     1.1x    44.8   0.9912
   0.300 23,456,789B     1.4x    47.0   0.9956
   0.500 32,100,456B     2.0x    50.2   0.9981

  Target       Size   vs PNG     PSNR
──────── ────────── ──────── ────────
    25dB  1,456,789B     0.1x    25.1
    30dB  3,456,789B     0.2x    30.2
    35dB  6,201,344B     0.4x    34.8
    40dB 11,234,567B     0.7x    40.1
```

---

## gen-photo

Generate a synthetic test image with gradients, colored circles, and Gaussian noise. Useful for quick testing without needing a real photo.

```bash
python3 switching_compress.py gen-photo <output> [options]
```

**Arguments:**
- `output` — Destination image path (e.g., `test.png`).

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--width <px>` | 512 | Image width in pixels. |
| `--height <px>` | same as width | Image height in pixels. If omitted, produces a square image. |

**How it works:**

1. Creates a smooth RGB gradient background (warm tones top-to-bottom, cool tones left-to-right).
2. Overlays randomly placed filled circles with blended colors. Number and size scale with image dimensions.
3. Adds Gaussian noise (sigma=12) for realistic texture.
4. Uses a fixed random seed (123) so the output is deterministic.

**Output:**

```
$ python3 switching_compress.py gen-photo test.png --width 1024 --height 768
Generated test photo: test.png (768x1024)
```
