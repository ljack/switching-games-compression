#!/usr/bin/env python3
"""
Switching Games Image Compression PoC

Based on "Switching Games for Image Compression" (Huhtanen, 2025).
Approximates image matrix M as sum_{j=1}^{l} D_{2j-1} * C * D_{2j}
where C is a DCT2 threshold-filtered initial image and D's are diagonal
matrices computed via alternating least-squares.

File format: SWG3 (.swg)
"""

import argparse
import io
import os
import struct
import time
import zlib

import numpy as np
from PIL import Image
from scipy.fft import dctn, idctn

MAGIC = b"SWG3"
QUANTIZATION_SCALE = 32767
CONVERGENCE_THRESHOLD = 1e-4
REGULARIZATION = 1e-12
COARSE_RATIOS = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]
BINARY_SEARCH_STEPS = 6


# ─── DCT2 initial image ─────────────────────────────────────────────

def _reconstruct_c_from_sparse_dct(indices: np.ndarray, values: np.ndarray,
                                    shape: tuple) -> np.ndarray:
    """Rebuild initial image C from sparse DCT coefficients."""
    sparse_coeffs = np.zeros(shape[0] * shape[1], dtype=np.float64)
    sparse_coeffs[indices] = values
    return idctn(sparse_coeffs.reshape(shape), type=2, norm="ortho")


def _compute_full_dct(M: np.ndarray):
    """Compute full 2D DCT once. Returns (flat_coeffs, magnitudes)."""
    coeffs = dctn(M, type=2, norm="ortho")
    flat = coeffs.ravel()
    return flat, np.abs(flat)


def _select_top_coefficients(flat: np.ndarray, magnitudes: np.ndarray,
                              ratio: float, shape: tuple):
    """
    Keep top `ratio` fraction of DCT coefficients by magnitude.

    Returns:
        C: reconstructed initial image (n x k)
        indices: flattened indices of kept coefficients
        values: kept coefficient values (float32)
    """
    nnz = max(1, int(ratio * flat.size))
    threshold_idx = np.argpartition(magnitudes, -nnz)[-nnz:]

    indices = threshold_idx.astype(np.int32)
    values = flat[indices].astype(np.float32)

    C = _reconstruct_c_from_sparse_dct(indices, values, shape)
    return C, indices, values


def generate_initial_image_dct2(M: np.ndarray, ratio: float):
    """Compute 2D DCT of M, keep top ratio fraction, inverse DCT to get C."""
    flat, magnitudes = _compute_full_dct(M)
    return _select_top_coefficients(flat, magnitudes, ratio, M.shape)


# ─── Algorithm 1 (vectorized) ────────────────────────────────────────

def algorithm1(M: np.ndarray, C: np.ndarray, l: int, max_iter: int = 7):
    """
    Algorithm 1 from the paper: find l layers of diagonal pairs.
    Vectorized — solves all rows/columns simultaneously via batch linear algebra.

    M ≈ sum_{j=1}^{l} D_{2j-1} C D_{2j}

    Returns:
        left_diags: (l, n) array
        right_diags: (l, k) array
    """
    n, k = M.shape

    rng = np.random.default_rng(42)
    left_diags = rng.standard_normal((l, n))
    right_diags = np.zeros((l, k))

    prev_residual_sq = None
    idx = np.arange(l)
    C2 = C ** 2
    CM = C * M

    for iteration in range(max_iter):
        # ── Solve for right diagonals (all columns simultaneously) ──
        L_outer = left_diags[:, None, :] * left_diags[None, :, :]  # (l, l, n)
        EtE = L_outer.reshape(l * l, n) @ C2  # (l*l, k)
        EtE = EtE.reshape(l, l, k)
        EtE[idx, idx, :] += REGULARIZATION
        Etm = left_diags @ CM  # (l, k)

        right_diags = np.linalg.solve(
            EtE.transpose(2, 0, 1),      # (k, l, l)
            Etm.T[:, :, np.newaxis]      # (k, l, 1)
        ).squeeze(-1).T  # (l, k)

        del L_outer, EtE, Etm

        # ── Solve for left diagonals (all rows simultaneously) ──
        R_outer = right_diags[:, None, :] * right_diags[None, :, :]  # (l, l, k)
        EtE = R_outer.reshape(l * l, k) @ C2.T  # (l*l, n)
        EtE = EtE.reshape(l, l, n)
        EtE[idx, idx, :] += REGULARIZATION
        Etm = right_diags @ CM.T  # (l, n)

        left_diags = np.linalg.solve(
            EtE.transpose(2, 0, 1),      # (n, l, l)
            Etm.T[:, :, np.newaxis]      # (n, l, 1)
        ).squeeze(-1).T  # (l, n)

        del R_outer, EtE, Etm

        # ── Convergence check (memory-efficient) ──
        LTR = left_diags.T @ right_diags  # (n, k)
        LTR *= C       # in-place: now = approx
        LTR -= M       # in-place: now = residual
        flat_r = LTR.ravel()
        residual_sq = flat_r @ flat_r
        del LTR

        if prev_residual_sq is not None:
            rel_change = abs(prev_residual_sq - residual_sq) / (prev_residual_sq + REGULARIZATION)
            if rel_change < CONVERGENCE_THRESHOLD:
                break
        prev_residual_sq = residual_sq

    return left_diags, right_diags


def algorithm1_adaptive(M: np.ndarray, C: np.ndarray, max_layers: int,
                        max_iter: int = 7, min_improvement: float = 0.01):
    """
    Run joint algorithm1 with max_layers, then prune trailing layers
    that contribute less than min_improvement relative quality.
    """
    left_diags, right_diags = algorithm1(M, C, max_layers, max_iter)

    full_residual = M - reconstruct(C, left_diags, right_diags)
    full_err = np.sqrt(full_residual.ravel() @ full_residual.ravel())
    M_norm = np.sqrt((M.ravel() @ M.ravel()))
    del full_residual
    actual_layers = max_layers

    # Incrementally check layer counts by accumulating outer products
    partial_weight = np.zeros_like(M)
    for try_layers in range(1, max_layers):
        partial_weight += np.outer(left_diags[try_layers - 1], right_diags[try_layers - 1])
        residual = M - C * partial_weight
        err = np.sqrt(residual.ravel() @ residual.ravel())
        if (err - full_err) / (M_norm + REGULARIZATION) < min_improvement:
            actual_layers = try_layers
            break

    return left_diags[:actual_layers], right_diags[:actual_layers], actual_layers


def reconstruct(C: np.ndarray, left_diags: np.ndarray, right_diags: np.ndarray) -> np.ndarray:
    """Reconstruct M_approx = sum_j diag(left_diags[j]) @ C @ diag(right_diags[j])"""
    return C * (left_diags.T @ right_diags)


# ─── Index encoding ──────────────────────────────────────────────────

def _encode_indices_delta(sorted_indices: np.ndarray) -> tuple:
    """Delta-encode sorted indices. Returns (mode, data_bytes)."""
    if len(sorted_indices) == 0:
        return 1, b""
    if len(sorted_indices) == 1:
        return 1, struct.pack("<I", int(sorted_indices[0]))

    first = int(sorted_indices[0])
    deltas = np.diff(sorted_indices)
    max_delta = int(deltas.max()) if len(deltas) > 0 else 0

    if max_delta <= 255:
        mode, delta_bytes = 1, deltas.astype(np.uint8).tobytes()
    elif max_delta <= 65535:
        mode, delta_bytes = 2, deltas.astype(np.uint16).tobytes()
    else:
        mode, delta_bytes = 3, deltas.astype(np.uint32).tobytes()

    return mode, struct.pack("<I", first) + delta_bytes


def _decode_indices_delta(mode: int, data: bytes, nnz: int) -> np.ndarray:
    """Decode delta-encoded indices."""
    if nnz == 0:
        return np.array([], dtype=np.int32)
    if nnz == 1:
        return np.array([struct.unpack("<I", data[:4])[0]], dtype=np.int32)

    first = struct.unpack("<I", data[:4])[0]
    delta_data = data[4:]
    dtypes = {1: np.uint8, 2: np.uint16, 3: np.uint32}
    deltas = np.frombuffer(delta_data, dtype=dtypes[mode]).astype(np.int64)

    indices = np.empty(nnz, dtype=np.int64)
    indices[0] = first
    indices[1:] = deltas
    return np.cumsum(indices).astype(np.int32)


def _encode_indices_bitmap(indices: np.ndarray, total_pixels: int) -> bytes:
    """Encode indices as packed bitmap."""
    bitmap = np.zeros(total_pixels, dtype=np.uint8)
    bitmap[indices] = 1
    return np.packbits(bitmap).tobytes()


def _decode_indices_bitmap(data: bytes, total_pixels: int) -> np.ndarray:
    """Decode bitmap-encoded indices."""
    packed = np.frombuffer(data, dtype=np.uint8)
    bitmap = np.unpackbits(packed)[:total_pixels]
    return np.nonzero(bitmap)[0].astype(np.int32)


def _encode_indices_auto(sorted_indices: np.ndarray, total_pixels: int) -> tuple:
    """Auto-select between bitmap and delta encoding based on estimated size."""
    nnz = len(sorted_indices)
    bitmap_size = (total_pixels + 7) // 8

    # Estimate delta size without computing it
    if nnz <= 1:
        return _encode_indices_delta(sorted_indices)
    max_delta = int(sorted_indices[-1] - sorted_indices[0]) // max(nnz - 1, 1)
    # Worst case: max delta determines dtype
    actual_max = int(np.diff(sorted_indices).max())
    if actual_max <= 255:
        delta_est = 4 + nnz - 1
    elif actual_max <= 65535:
        delta_est = 4 + (nnz - 1) * 2
    else:
        delta_est = 4 + (nnz - 1) * 4

    if bitmap_size <= delta_est:
        return 0, _encode_indices_bitmap(sorted_indices, total_pixels)
    else:
        return _encode_indices_delta(sorted_indices)


# ─── Delta coding for values ─────────────────────────────────────────

def _delta_encode_i16(values: np.ndarray) -> np.ndarray:
    """Delta-encode int16 values. First value kept as-is."""
    if len(values) <= 1:
        return values.copy()
    result = np.empty_like(values)
    result[0] = values[0]
    result[1:] = np.diff(values.astype(np.int32)).astype(np.int16)
    return result


def _delta_decode_i16(deltas: np.ndarray) -> np.ndarray:
    """Undo delta encoding of int16 values."""
    if len(deltas) <= 1:
        return deltas.copy()
    return np.cumsum(deltas.astype(np.int32)).astype(np.int16)


# ─── Auto-tune DCT ratio ─────────────────────────────────────────────

def _compress_channel_at_ratio(M: np.ndarray, dct_flat: np.ndarray,
                                dct_magnitudes: np.ndarray, ratio: float,
                                layers: int, max_iter: int,
                                adaptive: bool = False):
    """Compress a single channel at a given ratio using cached DCT."""
    C, indices, values = _select_top_coefficients(
        dct_flat, dct_magnitudes, ratio, M.shape)

    if adaptive:
        left_diags, right_diags, actual_layers = algorithm1_adaptive(
            M, C, layers, max_iter)
    else:
        left_diags, right_diags = algorithm1(M, C, layers, max_iter)
        actual_layers = layers

    # In-place clip for PSNR computation
    approx = reconstruct(C, left_diags, right_diags)
    np.clip(approx, 0, 255, out=approx)
    mse = np.mean((M - approx) ** 2)
    del approx
    ch_psnr = 10 * np.log10(255.0 ** 2 / mse) if mse > 0 else float("inf")

    return ch_psnr, C, indices, values, left_diags, right_diags, actual_layers


def _auto_tune_ratio(M: np.ndarray, target_psnr: float, layers: int,
                     max_iter: int, adaptive: bool = False,
                     verbose: bool = False):
    """
    Binary search for minimum DCT ratio achieving target PSNR.
    Computes full DCT once and reuses across ratio evaluations.
    """
    log = print if verbose else lambda *a, **kw: None

    # Compute full DCT once
    dct_flat, dct_mags = _compute_full_dct(M)

    # Evaluate coarse ratios to find straddling pair
    results = []
    for r in COARSE_RATIOS:
        log(f"    trying ratio={r:.3f}...", end=" ", flush=True)
        p, *data = _compress_channel_at_ratio(
            M, dct_flat, dct_mags, r, layers, max_iter, adaptive)
        results.append((r, p, data))
        log(f"PSNR={p:.1f} dB")
        if p >= target_psnr:
            break

    if results[-1][1] < target_psnr:
        r, p, data = results[-1]
        return (p, *data, r)
    if results[0][1] >= target_psnr:
        r, p, data = results[0]
        return (p, *data, r)

    # Find straddling pair
    lo_r, hi_r, hi_p, hi_data = None, None, None, None
    for i in range(len(results) - 1):
        if results[i][1] < target_psnr <= results[i + 1][1]:
            lo_r = results[i][0]
            hi_r, hi_p, hi_data = results[i + 1][0], results[i + 1][1], results[i + 1][2]
            break

    if lo_r is None:
        r, p, data = results[-1]
        return (p, *data, r)

    # Binary search
    best_r, best_p, best_data = hi_r, hi_p, hi_data
    for _ in range(BINARY_SEARCH_STEPS):
        mid_r = (lo_r + hi_r) / 2
        log(f"    refining ratio={mid_r:.4f}...", end=" ", flush=True)
        p, *data = _compress_channel_at_ratio(
            M, dct_flat, dct_mags, mid_r, layers, max_iter, adaptive)
        log(f"PSNR={p:.1f} dB")
        if p >= target_psnr:
            best_r, best_p, best_data = mid_r, p, data
            hi_r = mid_r
        else:
            lo_r = mid_r

    return (best_p, *best_data, best_r)


# ─── Compress / Decompress ───────────────────────────────────────────

def compress(image_path: str, output_path: str, layers: int = 6,
             dct_ratio: float = 0.30, max_iter: int = 7,
             target_psnr: float = None, adaptive: bool = True,
             quiet: bool = False):
    """Compress an image to .swg (SWG3) format."""
    if layers < 1:
        raise ValueError(f"layers must be >= 1, got {layers}")
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}")
    if not (0 < dct_ratio <= 1.0):
        raise ValueError(f"dct_ratio must be in (0, 1], got {dct_ratio}")
    img = Image.open(image_path).convert("RGB")
    M_full = np.array(img, dtype=np.float64)
    n, k, channels = M_full.shape

    auto_mode = target_psnr is not None
    if not auto_mode:
        target_psnr = 0.0

    log = print if not quiet else lambda *a, **kw: None

    log(f"Image: {n}x{k}, {channels} channels")
    if auto_mode:
        log(f"Max layers: {layers}, Target PSNR: {target_psnr} dB, Max iter: {max_iter}")
    else:
        log(f"Layers: {layers}, DCT ratio: {dct_ratio}, Max iter: {max_iter}")

    t0 = time.time()

    payload = io.BytesIO()
    total_pixels = n * k

    for ch in range(channels):
        ch_name = "RGB"[ch]
        ch_t0 = time.time()
        M = M_full[:, :, ch]

        if auto_mode:
            ch_psnr, C, indices, values, left_diags, right_diags, actual_layers, chosen_ratio = \
                _auto_tune_ratio(M, target_psnr, layers, max_iter, adaptive,
                                 verbose=not quiet)
            ch_elapsed = time.time() - ch_t0
            log(f"  Channel {ch_name}: ratio={chosen_ratio:.4f}, "
                f"layers={actual_layers}, PSNR={ch_psnr:.1f} dB ({ch_elapsed:.1f}s)")
        else:
            C, indices, values = generate_initial_image_dct2(M, dct_ratio)
            if adaptive:
                left_diags, right_diags, actual_layers = algorithm1_adaptive(
                    M, C, layers, max_iter)
            else:
                left_diags, right_diags = algorithm1(M, C, layers, max_iter)
                actual_layers = layers
            chosen_ratio = dct_ratio
            ch_elapsed = time.time() - ch_t0
            log(f"  Channel {ch_name}: ratio={chosen_ratio:.4f}, "
                f"layers={actual_layers} ({ch_elapsed:.1f}s)")

        nnz = len(indices)

        scale = np.float32(np.max(np.abs(values))) if nnz > 0 else np.float32(1.0)
        quantized = np.round(values / scale * QUANTIZATION_SCALE).astype(np.int16)

        sort_order = np.argsort(indices)
        sorted_indices = indices[sort_order]
        sorted_quantized = quantized[sort_order]

        delta_values = _delta_encode_i16(sorted_quantized)
        index_mode, index_data = _encode_indices_auto(sorted_indices, total_pixels)

        payload.write(struct.pack("<B", actual_layers))
        payload.write(struct.pack("<f", chosen_ratio))
        payload.write(struct.pack("<I", nnz))
        payload.write(struct.pack("<B", index_mode))
        payload.write(struct.pack("<I", len(index_data)))
        payload.write(index_data)
        payload.write(struct.pack("<f", scale))
        payload.write(delta_values.tobytes())
        payload.write(left_diags.astype(np.float16).tobytes())
        payload.write(right_diags.astype(np.float16).tobytes())

    compressed_payload = zlib.compress(payload.getvalue(), level=9)

    with open(output_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<II", n, k))
        f.write(struct.pack("<B", channels))
        f.write(struct.pack("<H", layers))
        f.write(struct.pack("<f", target_psnr))
        f.write(struct.pack("<B", max_iter))
        f.write(struct.pack("<I", len(compressed_payload)))
        f.write(compressed_payload)

    elapsed = time.time() - t0
    orig_size = os.path.getsize(image_path)
    comp_size = os.path.getsize(output_path)
    log(f"Compressed in {elapsed:.1f}s")
    log(f"Original: {orig_size:,} bytes")
    log(f"Compressed: {comp_size:,} bytes ({comp_size/orig_size:.1%})")
    return comp_size


def decompress(swg_path: str, output_path: str, quiet: bool = False):
    """Decompress a .swg (SWG3) file to an image."""
    log = print if not quiet else lambda *a, **kw: None
    with open(swg_path, "rb") as f:
        magic = f.read(4)
        if magic != MAGIC:
            raise ValueError(f"Invalid file: expected SWG3, got {magic}")

        n, k = struct.unpack("<II", f.read(8))
        if n == 0 or k == 0 or n > 32768 or k > 32768:
            raise ValueError(f"Invalid dimensions: {n}x{k} (max 32768x32768)")
        if n * k > 256 * 1024 * 1024:
            raise ValueError(f"Image too large: {n}x{k} = {n*k} pixels")
        channels = struct.unpack("<B", f.read(1))[0]
        if channels == 0 or channels > 4:
            raise ValueError(f"Invalid channel count: {channels}")
        layers = struct.unpack("<H", f.read(2))[0]
        if layers > 64:
            raise ValueError(f"Invalid layer count: {layers}")
        _target_psnr = struct.unpack("<f", f.read(4))[0]
        _max_iter = struct.unpack("<B", f.read(1))[0]

        log(f"Image: {n}x{k}, {channels} channels, max_layers={layers}")

        compressed_size = struct.unpack("<I", f.read(4))[0]
        MAX_DECOMPRESSED = 256 * 1024 * 1024
        raw = zlib.decompress(f.read(compressed_size), wbits=15, bufsize=MAX_DECOMPRESSED)
        if len(raw) > MAX_DECOMPRESSED:
            raise ValueError(f"Decompressed payload too large: {len(raw)} bytes")

    buf = io.BytesIO(raw)
    total_pixels = n * k
    M_full = np.zeros((n, k, channels), dtype=np.float64)

    for ch in range(channels):
        actual_layers = struct.unpack("<B", buf.read(1))[0]
        _ratio = struct.unpack("<f", buf.read(4))[0]
        nnz = struct.unpack("<I", buf.read(4))[0]
        index_mode = struct.unpack("<B", buf.read(1))[0]
        index_data_len = struct.unpack("<I", buf.read(4))[0]
        index_data = buf.read(index_data_len)

        if index_mode == 0:
            dct_indices = _decode_indices_bitmap(index_data, total_pixels)
        else:
            dct_indices = _decode_indices_delta(index_mode, index_data, nnz)

        scale = struct.unpack("<f", buf.read(4))[0]
        delta_values = np.frombuffer(buf.read(nnz * 2), dtype=np.int16).copy()
        quantized = _delta_decode_i16(delta_values)
        dct_values = (quantized.astype(np.float32) / QUANTIZATION_SCALE) * scale

        C = _reconstruct_c_from_sparse_dct(dct_indices, dct_values, (n, k))

        left_diags = np.frombuffer(
            buf.read(actual_layers * n * 2), dtype=np.float16
        ).reshape(actual_layers, n).astype(np.float64)
        right_diags = np.frombuffer(
            buf.read(actual_layers * k * 2), dtype=np.float16
        ).reshape(actual_layers, k).astype(np.float64)

        M_full[:, :, ch] = reconstruct(C, left_diags, right_diags)

    M_full = np.clip(M_full, 0, 255).astype(np.uint8)
    img = Image.fromarray(M_full, "RGB")
    img.save(output_path)
    log(f"Saved reconstructed image to {output_path}")


# ─── Evaluate ────────────────────────────────────────────────────────

def psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    mse = np.mean((original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(255.0 ** 2 / mse)


def _compress_and_measure(image_path: str, swg_path: str, recon_path: str,
                           original: np.ndarray, ssim_fn=None, **compress_kwargs):
    """Compress, decompress, measure quality. Returns (size, psnr, ssim_or_None)."""
    compress(image_path, swg_path, quiet=True, **compress_kwargs)
    decompress(swg_path, recon_path, quiet=True)

    recon = np.array(Image.open(recon_path).convert("RGB"))
    p = psnr(original, recon)
    s = ssim_fn(original, recon, channel_axis=2) if ssim_fn else None
    return os.path.getsize(swg_path), p, s


def evaluate(image_path: str, layers: int = 6, dct_ratio: float = 0.30,
             max_iter: int = 7, output_dir: str = ".", target_psnr: float = None):
    """Compress, decompress, and evaluate quality."""
    from skimage.metrics import structural_similarity as ssim

    base = os.path.splitext(os.path.basename(image_path))[0]
    swg_path = os.path.join(output_dir, f"{base}.swg")
    recon_path = os.path.join(output_dir, f"{base}_reconstructed.png")
    compare_path = os.path.join(output_dir, f"{base}_comparison.png")

    compress(image_path, swg_path, layers, dct_ratio, max_iter,
             target_psnr=target_psnr)
    decompress(swg_path, recon_path)

    original = np.array(Image.open(image_path).convert("RGB"))
    reconstructed = np.array(Image.open(recon_path).convert("RGB"))

    p = psnr(original, reconstructed)
    s = ssim(original, reconstructed, channel_axis=2)

    print(f"\n── Quality Metrics ──")
    print(f"PSNR: {p:.2f} dB")
    print(f"SSIM: {s:.4f}")

    h, w = original.shape[:2]
    comparison = np.zeros((h, w * 2 + 10, 3), dtype=np.uint8)
    comparison[:, :w] = original
    comparison[:, w + 10:] = reconstructed
    comparison[:, w:w + 10] = 128
    Image.fromarray(comparison).save(compare_path)
    print(f"Comparison image: {compare_path}")

    return p, s


# ─── Sweep ────────────────────────────────────────────────────────────

def sweep(image_path: str, layers: int = 6, max_iter: int = 7,
          output_dir: str = "."):
    """Evaluate image at multiple ratios, print comparison table."""
    from skimage.metrics import structural_similarity as ssim

    original = np.array(Image.open(image_path).convert("RGB"))
    png_size = os.path.getsize(image_path)
    n, k = original.shape[:2]

    print(f"Image: {n}x{k}, {original.shape[2]} channels")
    print(f"PNG size: {png_size:,} bytes")
    print(f"Layers: {layers}, Max iter: {max_iter}")
    print()
    print(f"{'Ratio':>8} {'Size':>10} {'vs PNG':>8} {'PSNR':>8} {'SSIM':>8}")
    print(f"{'─'*8} {'─'*10} {'─'*8} {'─'*8} {'─'*8}")

    base = os.path.splitext(os.path.basename(image_path))[0]

    for ratio in COARSE_RATIOS:
        swg_path = os.path.join(output_dir, f"{base}_sweep_{ratio:.3f}.swg")
        recon_path = os.path.join(output_dir, f"{base}_sweep_{ratio:.3f}.png")

        try:
            sz, p, s = _compress_and_measure(
                image_path, swg_path, recon_path, original, ssim_fn=ssim,
                layers=layers, dct_ratio=ratio, max_iter=max_iter)
            print(f"{ratio:>8.3f} {sz:>9,}B {sz/png_size:>7.1f}x {p:>7.1f} {s:>8.4f}")
        except Exception as e:
            print(f"{ratio:>8.3f} {'ERROR':>10} {str(e)[:40]}")
        finally:
            for f in (swg_path, recon_path):
                if os.path.exists(f):
                    os.remove(f)

    print()
    print(f"{'Target':>8} {'Size':>10} {'vs PNG':>8} {'PSNR':>8}")
    print(f"{'─'*8} {'─'*10} {'─'*8} {'─'*8}")

    for target in [25.0, 30.0, 35.0, 40.0]:
        swg_path = os.path.join(output_dir, f"{base}_sweep_auto_{target:.0f}.swg")
        recon_path = os.path.join(output_dir, f"{base}_sweep_auto_{target:.0f}.png")

        try:
            sz, p, _ = _compress_and_measure(
                image_path, swg_path, recon_path, original,
                layers=layers, max_iter=max_iter, target_psnr=target)
            print(f"{target:>6.0f}dB {sz:>9,}B {sz/png_size:>7.1f}x {p:>7.1f}")
        except Exception as e:
            print(f"{target:>6.0f}dB {'ERROR':>10} {str(e)[:40]}")
        finally:
            for f in (swg_path, recon_path):
                if os.path.exists(f):
                    os.remove(f)


# ─── Test photo generator ────────────────────────────────────────────

def generate_test_photo(output_path: str, width: int = 512, height: int = None):
    """Generate a synthetic photo-like test image (gradient + noise + shapes)."""
    if height is None:
        height = width
    rng = np.random.default_rng(123)
    img = np.zeros((height, width, 3), dtype=np.float64)

    y = np.linspace(0, 1, height)[:, None]
    x = np.linspace(0, 1, width)[None, :]
    img[:, :, 0] = 180 * y + 40 * x
    img[:, :, 1] = 100 * (1 - y) + 80 * x
    img[:, :, 2] = 60 + 140 * np.sin(np.pi * x) * np.cos(np.pi * y)

    area = width * height
    n_shapes = max(15, int(15 * area / (512 * 512)))
    max_radius = max(20, min(80, min(width, height) // 8))

    for _ in range(n_shapes):
        cx = rng.integers(max_radius, width - max_radius)
        cy = rng.integers(max_radius, height - max_radius)
        radius = rng.integers(max_radius // 4, max_radius)
        color = rng.integers(50, 255, 3)
        y0, y1 = max(0, cy - radius), min(height, cy + radius + 1)
        x0, x1 = max(0, cx - radius), min(width, cx + radius + 1)
        yy = np.arange(y0, y1)[:, None]
        xx = np.arange(x0, x1)[None, :]
        mask = ((xx - cx) ** 2 + (yy - cy) ** 2) < radius ** 2
        for c in range(3):
            patch = img[y0:y1, x0:x1, c]
            patch[mask] = patch[mask] * 0.3 + color[c] * 0.7

    noise = rng.standard_normal((height, width, 3)) * 12
    img += noise

    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img, "RGB").save(output_path)
    print(f"Generated test photo: {output_path} ({height}x{width})")
    return output_path


# ─── CLI ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Switching Games Image Compression"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_comp = sub.add_parser("compress", help="Compress image to .swg")
    p_comp.add_argument("input", help="Input image path")
    p_comp.add_argument("output", help="Output .swg path")
    p_comp.add_argument("--layers", type=int, default=6)
    p_comp.add_argument("--dct-ratio", type=float, default=0.30)
    p_comp.add_argument("--max-iter", type=int, default=7)
    p_comp.add_argument("--target-psnr", type=float, default=None,
                         help="Target PSNR in dB (overrides --dct-ratio)")
    p_comp.add_argument("--no-adaptive", action="store_true",
                         help="Disable adaptive layer count")

    p_dec = sub.add_parser("decompress")
    p_dec.add_argument("input", help="Input .swg path")
    p_dec.add_argument("output", help="Output image path")

    p_eval = sub.add_parser("evaluate")
    p_eval.add_argument("input", help="Input image path")
    p_eval.add_argument("--layers", type=int, default=6)
    p_eval.add_argument("--dct-ratio", type=float, default=0.30)
    p_eval.add_argument("--max-iter", type=int, default=7)
    p_eval.add_argument("--output-dir", default=".")
    p_eval.add_argument("--target-psnr", type=float, default=None)

    p_sweep = sub.add_parser("sweep", help="Evaluate at multiple ratios")
    p_sweep.add_argument("input", help="Input image path")
    p_sweep.add_argument("--layers", type=int, default=6)
    p_sweep.add_argument("--max-iter", type=int, default=7)
    p_sweep.add_argument("--output-dir", default=".")

    p_gen = sub.add_parser("gen-photo", help="Generate synthetic test photo")
    p_gen.add_argument("output", help="Output image path")
    p_gen.add_argument("--width", type=int, default=512)
    p_gen.add_argument("--height", type=int, default=None)

    args = parser.parse_args()

    if args.command == "compress":
        compress(args.input, args.output, args.layers, args.dct_ratio,
                 args.max_iter, target_psnr=args.target_psnr,
                 adaptive=not args.no_adaptive)
    elif args.command == "decompress":
        decompress(args.input, args.output)
    elif args.command == "evaluate":
        evaluate(args.input, args.layers, args.dct_ratio, args.max_iter,
                 args.output_dir, target_psnr=args.target_psnr)
    elif args.command == "sweep":
        sweep(args.input, args.layers, args.max_iter, args.output_dir)
    elif args.command == "gen-photo":
        generate_test_photo(args.output, args.width, args.height)


if __name__ == "__main__":
    main()
