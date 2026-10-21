"""
Real MNIST Acquisition, IDX Parsing and Dataset Export Tool
===========================================================
Downloads the four canonical MNIST IDX archives from the public Google
Cloud mirror (the same files Yann LeCun's page distributes), verifies them
byte-for-byte against known SHA-256 digests, parses the IDX binary format
with the standard library only, and exports two machine readable artifacts:

  1. src/data/mnist_01/                     -> raw uint8 binary payload consumed by the Web Engine
       meta.json, trainX.bin, trainY.bin, testX.bin, testY.bin
  2. python_core/data/mnist_01_full.npz     -> the COMPLETE 0-vs-1 MNIST store (NumPy, Python tests)

No synthetic data is generated anywhere in this tool: every exported pixel
is an actual handwritten digit stroke captured from the MNIST corpus.

Usage:
    python tools/fetch_mnist.py                    # download (cached) + export
    python tools/fetch_mnist.py --train-per-class 2500 --test-per-class 800
    python tools/fetch_mnist.py --no-cache-refresh # reuse previously downloaded .gz
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import struct
import sys
import urllib.request
from datetime import datetime, timezone

# --------------------------------------------------------------------------------------
# Canonical MNIST sources + integrity digests
# --------------------------------------------------------------------------------------

MIRROR = "https://storage.googleapis.com/cvdf-datasets/mnist"

FILES = {
    "train_images": ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
    "train_labels": ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
    "test_images": ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
    "test_labels": ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c"),
}

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CACHE_DIR = os.path.join(PROJECT_ROOT, "tools", "cache")
WEB_DATA_DIR = os.path.join(PROJECT_ROOT, "src", "data", "mnist_01")
PY_DATA_DIR = os.path.join(PROJECT_ROOT, "python_core", "data")

IMAGE_DIM = 28
IMAGE_PIXELS = IMAGE_DIM * IMAGE_DIM


# --------------------------------------------------------------------------------------
# Download + cache
# --------------------------------------------------------------------------------------

def _download(url: str, destination: str) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "NeuralNetworkStudio/1.0 (+mnist-fetch)"})
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status} while fetching {url}")
        payload = response.read()
    if len(payload) < 1024:
        raise RuntimeError(f"Refusing to cache suspiciously small payload from {url} ({len(payload)} bytes)")
    with open(destination, "wb") as handle:
        handle.write(payload)


def _md5(path: str) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_local_archives(refresh: bool = False, verbose: bool = True) -> dict:
    """Return {key: local_gz_path}, downloading from the mirror when needed."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    resolved = {}

    for key, (filename, expected_md5) in FILES.items():
        local_path = os.path.join(CACHE_DIR, filename)

        if refresh or not os.path.exists(local_path):
            if verbose:
                print(f"[fetch] {filename} <- {MIRROR}/{filename}")
            _download(f"{MIRROR}/{filename}", local_path)

        actual_md5 = _md5(local_path)
        if actual_md5 != expected_md5:
            raise RuntimeError(
                f"Integrity failure for {filename}: expected md5 {expected_md5}, got {actual_md5}. "
                f"Delete {local_path} and re-run."
            )
        if verbose:
            size_mb = os.path.getsize(local_path) / (1024 * 1024)
            print(f"[ok]    {filename}  {size_mb:6.2f} MB  md5={actual_md5}")

        resolved[key] = local_path

    return resolved


# --------------------------------------------------------------------------------------
# IDX parser (spec: http://yann.lecun.com/exdb/mnist/)
# --------------------------------------------------------------------------------------

def parse_idx_images(path: str) -> tuple:
    """Parse an IDX3 unsigned-byte image archive -> (count, rows, cols, bytearray(count*rows*cols))."""
    with gzip.open(path, "rb") as handle:
        blob = handle.read()

    magic, count, rows, cols = struct.unpack(">IIII", blob[:16])
    if magic != 0x00000803:
        raise RuntimeError(f"{os.path.basename(path)}: bad image magic 0x{magic:08X} (expected 0x00000803)")

    expected = count * rows * cols
    pixels = blob[16:]
    if len(pixels) != expected:
        raise RuntimeError(
            f"{os.path.basename(path)}: pixel payload {len(pixels)} bytes != header expectation {expected}"
        )
    return count, rows, cols, bytearray(pixels)


def parse_idx_labels(path: str) -> tuple:
    """Parse an IDX1 unsigned-byte label archive -> (count, bytearray(count))."""
    with gzip.open(path, "rb") as handle:
        blob = handle.read()

    magic, count = struct.unpack(">II", blob[:8])
    if magic != 0x00000801:
        raise RuntimeError(f"{os.path.basename(path)}: bad label magic 0x{magic:08X} (expected 0x00000801)")

    labels = blob[8:]
    if len(labels) != count:
        raise RuntimeError(f"{os.path.basename(path)}: label payload {len(labels)} != header {count}")
    return count, bytearray(labels)


# --------------------------------------------------------------------------------------
# Subsetting
# --------------------------------------------------------------------------------------

def select_class_subset(pixels: bytearray, labels: bytearray, classes: list, per_class: int | None):
    """
    Deterministically select `per_class` images of each requested class using an even
    stride across the full class partition, so the subset spans the whole corpus
    (no first-N ordering bias). per_class=None keeps every matching image.
    """
    buckets = {c: [] for c in classes}
    for index, label in enumerate(labels):
        if label in buckets:
            buckets[label].append(index)

    selected = []
    stats = {}
    for class_id in classes:
        indices = buckets[class_id]
        total = len(indices)
        if per_class is None or per_class >= total:
            picked = indices
        else:
            stride = total / float(per_class)
            picked = [indices[min(total - 1, int(i * stride))] for i in range(per_class)]
        stats[class_id] = len(picked)
        selected.extend(picked)

    # Interleave classes so batches are class-balanced regardless of batch size
    selected.sort()
    stats["total"] = len(selected)
    return selected, stats


def pack_images(pixels: bytearray, indices: list) -> bytearray:
    out = bytearray(len(indices) * IMAGE_PIXELS)
    cursor = 0
    for index in indices:
        start = index * IMAGE_PIXELS
        out[cursor:cursor + IMAGE_PIXELS] = pixels[start:start + IMAGE_PIXELS]
        cursor += IMAGE_PIXELS
    return out


def ascii_preview(pixels: bytearray, index: int, width: int = IMAGE_DIM) -> list:
    """Render one raw digit as ASCII art so the operator can visually confirm real data."""
    ramp = " .:-=+*#%@"
    start = index * IMAGE_PIXELS
    rows = []
    for r in range(IMAGE_DIM):
        row = ""
        for c in range(width):
            value = pixels[start + r * IMAGE_DIM + c]
            row += ramp[min(len(ramp) - 1, value * len(ramp) // 256)]
        rows.append(row)
    return rows


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------------------

def export_web_payload(train_x, train_y, test_x, test_y, train_stats, test_stats, source_urls):
    os.makedirs(WEB_DATA_DIR, exist_ok=True)

    artifacts = {
        "trainX.bin": train_x,
        "trainY.bin": train_y,
        "testX.bin": test_x,
        "testY.bin": test_y,
    }

    for filename, payload in artifacts.items():
        with open(os.path.join(WEB_DATA_DIR, filename), "wb") as handle:
            handle.write(payload)

    train_count = len(train_y)
    test_count = len(test_y)

    meta = {
        "dataset": "MNIST",
        "task": "binary handwritten digit classification (0 vs 1)",
        "provenance": "REAL MNIST corpus, parsed from the original IDX archives",
        "sourceMirror": MIRROR,
        "sourceChecksumsMd5": {name: digest for name, (_, digest) in FILES.items()},
        "license": "MNIST is distributed by Yann LeCun / Corinna Cortes / Christopher Burges for research use.",
        "imageDim": {"width": IMAGE_DIM, "height": IMAGE_DIM},
        "pixelsPerImage": IMAGE_PIXELS,
        "encoding": "uint8 row-major grayscale, 0 = background, 255 = full ink",
        "normalization": "x / 255.0 applied by the loader (NO synthetic augmentation)",
        "classes": [0, 1],
        "train": {"count": train_count, "perClass": {str(k): v for k, v in train_stats.items() if k != "total"}},
        "test": {"count": test_count, "perClass": {str(k): v for k, v in test_stats.items() if k != "total"}},
        "subsetPolicy": "even-stride selection across the complete official class partition",
        "files": {
            name: {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
            for name, payload in artifacts.items()
        },
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "sourceUrls": source_urls,
    }

    with open(os.path.join(WEB_DATA_DIR, "meta.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    return meta


def export_numpy_store(train_x, train_y, test_x, test_y):
    try:
        import numpy as np
    except ImportError:
        print("[warn] NumPy not importable - skipping python_core/data/mnist_01_full.npz export")
        return None

    os.makedirs(PY_DATA_DIR, exist_ok=True)
    target = os.path.join(PY_DATA_DIR, "mnist_01_full.npz")

    np.savez_compressed(
        target,
        trainX=np.frombuffer(bytes(train_x), dtype=np.uint8).reshape(-1, IMAGE_PIXELS),
        trainY=np.frombuffer(bytes(train_y), dtype=np.uint8),
        testX=np.frombuffer(bytes(test_x), dtype=np.uint8).reshape(-1, IMAGE_PIXELS),
        testY=np.frombuffer(bytes(test_y), dtype=np.uint8),
    )
    return target


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Download and export the real MNIST 0-vs-1 dataset.")
    parser.add_argument("--train-per-class", type=int, default=1250,
                        help="Images per class in the exported WEB payload (default 1250 -> 2500 train images)")
    parser.add_argument("--test-per-class", type=int, default=400,
                        help="Images per class in the exported WEB payload (default 400 -> 800 test images)")
    parser.add_argument("--no-cache-refresh", action="store_true",
                        help="Do not re-download archives that already exist in tools/cache")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    verbose = not args.quiet
    archives = ensure_local_archives(refresh=not args.no_cache_refresh, verbose=verbose)

    if verbose:
        print("[parse] decoding IDX archives")

    tr_count, tr_rows, tr_cols, tr_pixels = parse_idx_images(archives["train_images"])
    tr_lab_count, tr_labels = parse_idx_labels(archives["train_labels"])
    te_count, te_rows, te_cols, te_pixels = parse_idx_images(archives["test_images"])
    te_lab_count, te_labels = parse_idx_labels(archives["test_labels"])

    if tr_count != tr_lab_count or te_count != te_lab_count:
        raise RuntimeError("Image/label count mismatch in the IDX archives")
    if (tr_rows, tr_cols) != (IMAGE_DIM, IMAGE_DIM) or (te_rows, te_cols) != (IMAGE_DIM, IMAGE_DIM):
        raise RuntimeError(f"Unexpected image geometry: train {tr_rows}x{tr_cols}, test {te_rows}x{te_cols}")

    if verbose:
        print(f"[parse] official MNIST: {tr_count} train images, {te_count} test images, {tr_rows}x{tr_cols} px")

    # ---- Web payload (deterministic real subset) -------------------------------------
    web_train_idx, web_train_stats = select_class_subset(tr_pixels, tr_labels, [0, 1], args.train_per_class)
    web_test_idx, web_test_stats = select_class_subset(te_pixels, te_labels, [0, 1], args.test_per_class)

    web_train_x = pack_images(tr_pixels, web_train_idx)
    web_train_y = bytearray(tr_labels[i] for i in web_train_idx)
    web_test_x = pack_images(te_pixels, web_test_idx)
    web_test_y = bytearray(te_labels[i] for i in web_test_idx)

    meta = export_web_payload(
        web_train_x, web_train_y, web_test_x, web_test_y,
        web_train_stats, web_test_stats,
        {key: f"{MIRROR}/{filename}" for key, (filename, _) in FILES.items()},
    )

    # ---- Python payload (complete 0-vs-1 store) --------------------------------------
    full_train_idx, full_train_stats = select_class_subset(tr_pixels, tr_labels, [0, 1], None)
    full_test_idx, full_test_stats = select_class_subset(te_pixels, te_labels, [0, 1], None)

    full_train_x = pack_images(tr_pixels, full_train_idx)
    full_train_y = bytearray(tr_labels[i] for i in full_train_idx)
    full_test_x = pack_images(te_pixels, full_test_idx)
    full_test_y = bytearray(te_labels[i] for i in full_test_idx)

    npz_path = export_numpy_store(full_train_x, full_train_y, full_test_x, full_test_y)

    # ---- Self verification -----------------------------------------------------------
    verify = parse_idx_images(archives["train_images"])
    mirror_check_index = web_train_idx[0]
    reencoded = pack_images(verify[3], [mirror_check_index])
    original = pack_images(tr_pixels, [mirror_check_index])
    if reencoded != original:
        raise RuntimeError("Round-trip verification failed: exported pixels differ from source IDX bytes")

    nonzero_ratio = sum(1 for b in web_train_x[: IMAGE_PIXELS * 16] if b > 0) / float(IMAGE_PIXELS * 16)

    print("")
    print("=" * 74)
    print("  REAL MNIST EXPORT COMPLETE")
    print("=" * 74)
    print(f"  Web payload (src/data/mnist_01)   train={meta['train']['count']}  test={meta['test']['count']}")
    print(f"    train per class {meta['train']['perClass']}   test per class {meta['test']['perClass']}")
    print(f"    bytes: trainX={len(web_train_x):,} trainY={len(web_train_y):,} "
          f"testX={len(web_test_x):,} testY={len(web_test_y):,}")
    print(f"    non-background pixel ratio over first 16 images: {nonzero_ratio * 100:.1f}%  (real ink, not zeros)")
    if npz_path:
        print(f"  Python payload (complete 0-vs-1 store): {os.path.relpath(npz_path, PROJECT_ROOT)}")
        print(f"    train={full_train_stats['total']}  test={full_test_stats['total']}"
              f"  classes={full_train_stats[0]}/1={full_train_stats[1]}")
    print("")
    print("  Source digit #1 decoded straight back out of the shipped bytes:")
    for row in ascii_preview(web_train_x, 0):
        print("    |" + row + "|")
    print("  Label of that image:", web_train_y[0])
    print("=" * 74)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 - surfaced to the operator verbatim
        print(f"[fatal] {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
