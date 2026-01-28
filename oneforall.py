import os
import re
import json
import lzma
import random
import csv
import time
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from tqdm import tqdm

# =============================================================================
# Cache files
# =============================================================================
COMPRESSED_SIZE_CACHE_FILE = "compressed_size_cache.json"
NCD_CACHE_FILE = "ncd_cache.json"

compressed_size_cache = {}
ncd_cache = {}

# =============================================================================
# Cache load/save (MAIN PROCESS ONLY)
# =============================================================================
def load_caches():
    global compressed_size_cache, ncd_cache

    if os.path.exists(COMPRESSED_SIZE_CACHE_FILE):
        with open(COMPRESSED_SIZE_CACHE_FILE, "r", encoding="utf-8") as f:
            compressed_size_cache = json.load(f)
        print(f"✓ Loaded compressed size cache ({len(compressed_size_cache)})")

    if os.path.exists(NCD_CACHE_FILE):
        with open(NCD_CACHE_FILE, "r", encoding="utf-8") as f:
            ncd_cache = json.load(f)
        print(f"✓ Loaded NCD cache ({len(ncd_cache)})")


def save_caches():
    with open(COMPRESSED_SIZE_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(compressed_size_cache, f)

    with open(NCD_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(ncd_cache, f)

    print("✓ Caches saved")

# =============================================================================
# NCD core
# =============================================================================
def compress_size(data: bytes, key: str = None) -> int:
    if key and key in compressed_size_cache:
        return compressed_size_cache[key]

    size = len(lzma.compress(data, preset=9))
    if key:
        compressed_size_cache[key] = size
    return size


def ncd(x: bytes, y: bytes, x_key=None, y_key=None) -> float:
    key = f"{x_key}:{y_key}" if x_key and y_key else None
    if key and key in ncd_cache:
        return ncd_cache[key]

    Cx = compress_size(x, x_key)
    Cy = compress_size(y, y_key)
    Cxy = compress_size(x + y)

    d = (Cxy - min(Cx, Cy)) / max(Cx, Cy)
    if key:
        ncd_cache[key] = d
    return d

# =============================================================================
# File helpers
# =============================================================================
def load_file_bytes(folder, fname):
    try:
        with open(os.path.join(folder, fname), "rb") as f:
            return fname, f.read()
    except Exception:
        return None


def load_files_parallel(folder, files, max_workers=8):
    names, contents = [], []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for r in tqdm(
            ex.map(lambda f: load_file_bytes(folder, f), files),
            total=len(files),
            desc="Loading files"
        ):
            if r:
                names.append(r[0])
                contents.append(r[1])
    return names, contents

# =============================================================================
# Resume detection
# =============================================================================
def detect_last_completed_batch():
    files = [
        f for f in os.listdir(".")
        if re.match(r"incremental_prototypes_batch\d+\.json", f)
    ]
    if not files:
        return 1
    batches = [int(re.search(r"batch(\d+)", f).group(1)) for f in files]
    return max(batches) + 1

# =============================================================================
# Precompute compressed sizes (SKIP cached)
# =============================================================================
def precompute_compressed_sizes(folder, files, max_workers=12):
    to_compute = [f for f in files if f not in compressed_size_cache]

    if not to_compute:
        print("✓ All compressed sizes already cached — skipping precompute")
        return

    print(f"Precomputing compressed sizes for {len(to_compute)} new files")

    def worker(fname):
        r = load_file_bytes(folder, fname)
        if r:
            compress_size(r[1], key=fname)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        list(tqdm(ex.map(worker, to_compute), total=len(to_compute)))

# =============================================================================
# Assignment
# =============================================================================
def assign_chunk(args):
    chunk, prototypes, proto_keys, folder = args
    results = []

    for fname in chunk:
        r = load_file_bytes(folder, fname)
        if not r:
            results.append((fname, None, 1.0))
            continue

        _, data = r
        best_pid, best_d = None, float("inf")

        for pid, proto in prototypes.items():
            d = ncd(data, proto, fname, proto_keys[pid])
            if d < best_d:
                best_pid, best_d = pid, d

        results.append((fname, best_pid, best_d))

    return results


def assign_to_prototypes_parallel(
    files, prototypes, proto_keys, folder,
    dthreshold, max_workers=12, chunk_size=20
):
    assignments = {}
    chunks = [files[i:i + chunk_size] for i in range(0, len(files), chunk_size)]

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(assign_chunk, (chunk, prototypes, proto_keys, folder)): i
            for i, chunk in enumerate(chunks)
        }

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Assigning"):
            for fname, pid, d in fut.result():
                assignments[fname] = (pid, d)

    return assignments

# =============================================================================
# FPF (FIXED – ALWAYS RETURNS 3 VALUES)
# =============================================================================
def fpf_threshold_on_demand(
    files, folder, prototypes, proto_keys,
    dthreshold, seed=42, max_workers=8
):
    if not files:
        return {}, {}, max(prototypes.keys(), default=-1) + 1

    rng = random.Random(seed)
    rng.shuffle(files)

    fnames, contents = load_files_parallel(folder, files, max_workers)

    for f, c in zip(fnames, contents):
        compress_size(c, key=f)

    distances = []
    for i, data in enumerate(contents):
        if prototypes:
            d = min(
                ncd(data, p, fnames[i], proto_keys[pid])
                for pid, p in prototypes.items()
            )
        else:
            d = 1.0
        distances.append(d)

    new_p, new_k = {}
    next_pid = max(prototypes.keys(), default=-1) + 1

    while True:
        idx = int(np.argmax(distances))
        if distances[idx] < dthreshold:
            break

        new_p[next_pid] = contents[idx]
        new_k[next_pid] = fnames[idx]

        for i in range(len(contents)):
            distances[i] = min(
                distances[i],
                ncd(contents[i], contents[idx], fnames[i], fnames[idx])
            )

        next_pid += 1

    return new_p, new_k, next_pid

# =============================================================================
# Evaluation (NEXT batch only)
# =============================================================================
def evaluate_batch(files, prototypes, proto_keys, folder, dthreshold):
    assignments = assign_to_prototypes_parallel(
        files, prototypes, proto_keys, folder,
        dthreshold, max_workers=8
    )

    dists = [d for _, d in assignments.values()]
    return {
        "files": len(files),
        "outliers": sum(d > dthreshold for d in dists),
        "outlier_ratio": sum(d > dthreshold for d in dists) / len(dists),
        "mean_ncd": float(np.mean(dists)),
        "max_ncd": float(np.max(dists)),
    }

# =============================================================================
# MAIN PIPELINE
# =============================================================================
def incremental_clustering(
    folder="rendered_pages_parallel",
    batch_size=400,
    dthreshold=0.25,
    max_workers=12
):
    start_batch = detect_last_completed_batch()
    print(f"→ Auto-resuming from batch {start_batch}")

    prototypes, proto_keys = {}, {}

    if start_batch > 1:
        last = start_batch - 1
        with open(f"incremental_prototypes_batch{last}.json") as f:
            proto_keys = {int(k): v for k, v in json.load(f).items()}

        for pid, fname in proto_keys.items():
            r = load_file_bytes(folder, fname)
            if r:
                prototypes[pid] = r[1]

    all_files = sorted(f for f in os.listdir(folder) if f.endswith(".html"))

    precompute_compressed_sizes(folder, all_files, max_workers)

    batches = [
        all_files[i:i + batch_size]
        for i in range(0, len(all_files), batch_size)
    ]

    for idx, batch in enumerate(batches, start=start_batch):
        print(f"\n{'='*70}\nBATCH {idx} | {len(batch)} files\n{'='*70}")

        assignments = assign_to_prototypes_parallel(
            batch, prototypes, proto_keys, folder,
            dthreshold, max_workers
        )

        outliers = [f for f, (_, d) in assignments.items() if d > dthreshold]

        new_p, new_k, _ = fpf_threshold_on_demand(
            outliers, folder, prototypes, proto_keys, dthreshold
        )

        prototypes.update(new_p)
        proto_keys.update(new_k)

        with open(f"incremental_prototypes_batch{idx}.json", "w") as f:
            json.dump({str(k): v for k, v in proto_keys.items()}, f, indent=2)

        # ---- NEXT BATCH EVALUATION ----
        next_idx = idx - start_batch + 1
        if next_idx < len(batches):
            stats = evaluate_batch(
                batches[next_idx],
                prototypes, proto_keys, folder, dthreshold
            )
            print(
                f"[Preview Next Batch] "
                f"outliers={stats['outliers']} "
                f"({stats['outlier_ratio']:.2%}), "
                f"mean_ncd={stats['mean_ncd']:.3f}"
            )

    save_caches()
    print("\n✓ Incremental clustering finished")

# =============================================================================
if __name__ == "__main__":
    load_caches()
    incremental_clustering()
