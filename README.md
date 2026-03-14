# Switching Games Image Compression (SWG)

A proof-of-concept implementation of the image compression algorithm described in:

> **Switching Games for Image Compression**
> Marko Huhtanen, University of Oulu
> IEEE Signal Processing Letters, 2025
> DOI: [10.1109/LSP.2025.3543744](https://doi.org/10.1109/LSP.2025.3543744)

The method approximates an image matrix M as a sum of layers:

```
M ≈ Σ D_{2j-1} · C · D_{2j}
```

where C is a DCT-thresholded initial image and D's are diagonal matrices found via alternating least-squares. The name comes from its resemblance to Berlekamp's switching game in continuous form.

## Quick Start

```bash
# Compress with auto-tuned quality (recommended)
python3 switching_compress.py compress input.jpg output.swg --target-psnr 35

# Decompress
python3 switching_compress.py decompress output.swg reconstructed.png
```

See [CLI Reference](CLI.md) for full documentation of all commands and options.

## File Format (SWG3)

Binary format with zlib-compressed payload. Per-channel storage includes:
- Adaptive layer count and DCT ratio
- Delta-encoded or bitmap-encoded sparse DCT indices (auto-selected)
- Delta-coded quantized DCT values (int16)
- Diagonal matrices (float16)

## Requirements

```
numpy
scipy
Pillow
scikit-image  # for SSIM in evaluate/sweep
```

## Example Results (8192x6144 photo)

| Mode | Size | PSNR | Time |
|---|---|---|---|
| Original JPEG | 16.2 MB | — | — |
| `--target-psnr 35` | 6.2 MB (38%) | 34.8 dB | 98s |
| `--dct-ratio 0.05` | 7.7 MB (48%) | 37.2 dB | 14s |
| `--dct-ratio 0.01` | 2.6 MB (16%) | 28.0 dB | 13s |
