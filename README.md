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

## Security

This project was built by AI (Claude) and has undergone multiple security review layers:

### Audit Results

| Reviewer | Findings | Status |
|----------|----------|--------|
| Claude security audit | 7 vulnerabilities (2 critical, 2 high, 2 medium, 1 low) | All fixed |
| GitHub Copilot review | XSS in setInfo(), encoding inconsistency, missing validation | All fixed |
| SWG3 fuzz testing | 20 deterministic + 50 random tests | All passing |

### Mitigations

- **Content Security Policy**: `script-src 'self'` prevents inline script injection
- **Decompression bomb protection**: 256 MB limit on decompressed payload (browser + Python)
- **SWG3 parser hardening**: bounds checking on all fields, dimension limits (32768x32768), index validation, NaN/Infinity sanitization
- **No server component**: static site with no authentication, no user data storage, no backend

### Threat Model

The primary attack surface is a crafted `.swg` file. Defenses:

| Attack | Defense |
|--------|---------|
| Decompression bomb | 256 MB payload limit |
| Out-of-bounds GPU write | All scatter indices validated < totalPixels |
| Integer overflow (n×k) | Max 32768 per dimension, 256M pixel cap |
| XSS via filename | All user text rendered via textContent, not innerHTML |
| Truncated/corrupt payload | Bounds-checked reads with descriptive errors |
| NaN/Infinity in diagonals | Sanitized to 0 before GPU upload |

### Known Limitations

- WebGPU shader correctness depends on GPU driver implementation
- Float32 precision in GPU compute may produce slightly different results than float64 Python reference
- The mathematical correctness of the ALS solver has not been independently verified by a domain expert
- No formal verification of the WGSL shader code

### Verification Approach

See the [blog post](https://ljack.github.io/switching-games-compression/blog.html) for a detailed discussion of how AI-generated code was verified without line-by-line human review.

Fuzz tests: open `test-fuzz.html` in browser to run 20+ parser security tests.
