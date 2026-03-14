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

## Browser Demo

**[Live Demo on GitHub Pages](https://ljack.github.io/switching-games-compression/)**

The browser app runs entirely client-side using WebGPU compute shaders:

- **Viewer**: Load and decode .swg files with GPU-accelerated FFT-based IDCT
- **Compressor**: Compress images to .swg format with real-time preview
- **Camera**: Capture photos directly from device camera for compression
- **Comparison**: Side-by-side, swipe, and onion-skin comparison modes

Performance on 8192x6144 demo image: ~1 second decode via batched FFT IDCT.

### Browser Requirements

- Chrome 113+ or Edge 113+ (WebGPU required)
- Falls back to CPU Web Workers on browsers without WebGPU

## Python CLI

```bash
# Compress with auto-tuned quality (recommended)
python3 switching_compress.py compress input.jpg output.swg --target-psnr 35

# Decompress
python3 switching_compress.py decompress output.swg reconstructed.png
```

See [CLI Reference](CLI.md) for full documentation of all commands and options.

### Requirements

```
numpy
scipy
Pillow
scikit-image  # for SSIM in evaluate/sweep
```

## File Format (SWG3)

Binary format with zlib-compressed payload. Browser and Python implementations produce compatible files.

Per-channel storage includes:
- Adaptive layer count and DCT ratio
- Delta-encoded or bitmap-encoded sparse DCT indices (auto-selected)
- Delta-coded quantized DCT values (int16)
- Diagonal matrices (float16)

## Example Results (8192x6144 photo)

### Python CLI

| Mode | Size | PSNR | Time |
|---|---|---|---|
| Original JPEG | 16.2 MB | — | — |
| `--target-psnr 35` | 6.2 MB (38%) | 34.8 dB | 98s |
| `--dct-ratio 0.05` | 7.7 MB (48%) | 37.2 dB | 14s |
| `--dct-ratio 0.01` | 2.6 MB (16%) | 28.0 dB | 13s |

### Browser (WebGPU)

| Image | Decode Time | Method |
|---|---|---|
| 8192x6144 demo | ~1s | FFT IDCT (batched) |
| 512x512 test | ~12ms | FFT IDCT |
| Compression (512x512, 6 layers) | ~3s | GPU ALS + FFT DCT |
