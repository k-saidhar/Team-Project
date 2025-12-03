import os
import json
import lzma
import random
import csv
import time
import math
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib.pyplot as plt

# -------------------------------
# Optional GPU acceleration
# -------------------------------
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# -------------------------------
# Persistent caches
# -------------------------------
compressed_size_cache_file = "compressed_size_cache.json"
ncd_cache_file = "ncd_cache.json"

compressed_size_cache = {}
ncd_cache = {}
if os.path.exists(compressed_size_cache_file):
    with open(compressed_size_cache_file, "r") as f:
        compressed_size_cache = json.load(f)
if os.path.exists(ncd_cache_file):
    with open(ncd_cache_file, "r") as f:
        ncd_cache = json.load(f)

def compress_size(data: bytes, key: str = None) -> int:
    if key and key in compressed_size_cache:
        return compressed_size_cache[key]
    size = len(lzma.compress(data))
    if key:
        compressed_size_cache[key] = size
    return size

def ncd(x: bytes, y: bytes, x_key=None, y_key=None) -> float:
    key = f"{x_key}:{y_key}"
    if key in ncd_cache:
        return ncd_cache[key]

    # GPU accelerated compression (optional)
    if GPU_AVAILABLE:
        x_gpu = cp.array(list(x), dtype=cp.uint8)
        y_gpu = cp.array(list(y), dtype=cp.uint8)
        x_bytes = bytes(x_gpu.get().tolist())
        y_bytes = bytes(y_gpu.get().tolist())
    else:
        x_bytes, y_bytes = x, y

    Cx = compress_size(x_bytes, x_key)
    Cy = compress_size(y_bytes, y_key)
    Cxy = compress_size(x_bytes + y_bytes)
    dist = (Cxy - min(Cx, Cy)) / max(Cx, Cy) if max(Cx, Cy) > 0 else 0
    ncd_cache[key] = dist
    return dist

# -------------------------------
# Helper: sanitize filenames
# -------------------------------
def sanitize_filename(fname):
    fname = str(fname)
    fname = fname.replace("\\", "")
    fname = "".join(ch for ch in fname if ord(ch) >= 32)
    return fname.strip()

# -------------------------------
# Load file bytes (multi-threaded)
# -------------------------------
def load_file_bytes(folder, fname):
    try:
        path = os.path.join(folder, fname)
        with open(path, "rb") as f:
            return fname, f.read()
    except Exception as e:
        print(f"⚠️ Failed to load {fname}: {e}")
        return None

def load_files_parallel(folder, file_list, max_workers=8):
    files, contents = [], []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for result in tqdm(executor.map(lambda f: load_file_bytes(folder, f), file_list),
                           total=len(file_list),
                           desc="📂 Loading HTML files"):
            if result:
                fname, content = result
                files.append(fname)
                contents.append(content)
    return files, contents

# -------------------------------
# Assign chunk worker
# -------------------------------
def assign_chunk(args):
    chunk_files, prototypes, proto_keys, folder = args
    results = []
    for file in chunk_files:
        result = load_file_bytes(folder, file)
        if not result:
            results.append((file, None, float("inf")))
            continue
        data = result[1]
        min_dist, assigned_proto = float("inf"), None
        for pid, proto_data in prototypes.items():
            dist = ncd(data, proto_data, x_key=file, y_key=proto_keys[pid])
            if dist < min_dist:
                min_dist, assigned_proto = dist, pid
        results.append((file, assigned_proto, min_dist))
    return results

def assign_to_prototypes_parallel(files, prototypes, proto_keys, folder,
                                  dthreshold=0.25, max_workers=8,
                                  chunk_size=10,
                                  benchmark_csv="benchmark_stats.csv"):
    assignments = {}
    chunks = [files[i:i+chunk_size] for i in range(0, len(files), chunk_size)]
    if not os.path.exists(benchmark_csv):
        with open(benchmark_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["chunk_id", "chunk_size", "time_sec"])

    with ProcessPoolExecutor(max_workers=max_workers) as executor, \
         open(benchmark_csv, "a", newline="") as f:
        writer = csv.writer(f)
        futures = {executor.submit(assign_chunk, (chunk, prototypes, proto_keys, folder)): idx
                   for idx, chunk in enumerate(chunks)}
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="📌 Assigning to prototypes (parallel, chunked)"):
            chunk_id = futures[future]
            start = time.time()
            results = future.result()
            duration = time.time() - start
            writer.writerow([chunk_id, len(results), round(duration, 2)])
            for file, proto, dist in results:
                assignments[file] = (proto, dist)
    return assignments

# -------------------------------
# On-demand FPF for outliers
# -------------------------------
def compute_min_dist(args):
    i, outlier_contents, existing_prototypes, existing_proto_keys, loaded_files = args
    data = outlier_contents[i]
    min_dist = float("inf")
    for pid, proto_data in existing_prototypes.items():
        dist = ncd(data, proto_data, x_key=loaded_files[i], y_key=existing_proto_keys[pid])
        if dist < min_dist:
            min_dist = dist
    return i, min_dist

def update_distance(args):
    i, outlier_contents, new_proto_idx, distances, loaded_files, selected_indices = args
    if i in selected_indices:
        return 0
    data = outlier_contents[i]
    dist = ncd(data, outlier_contents[new_proto_idx],
               x_key=loaded_files[i], y_key=loaded_files[new_proto_idx])
    return min(distances[i], dist)

def fpf_threshold_on_demand(outlier_files, folder, existing_prototypes, existing_proto_keys,
                            dthreshold=0.25, seed=42, max_workers=8):
    rng = random.Random(seed)
    outlier_files = list(outlier_files)
    rng.shuffle(outlier_files)
    new_prototypes, new_proto_keys = {}, {}
    proto_id_counter = max(existing_prototypes.keys(), default=-1) + 1
    loaded_files, outlier_contents = load_files_parallel(folder, outlier_files, max_workers=max_workers)
    if not loaded_files:
        return new_prototypes, new_proto_keys, proto_id_counter

    for i, content in enumerate(outlier_contents):
        compress_size(content, key=loaded_files[i])

    args_list = [(i, outlier_contents, existing_prototypes, existing_proto_keys, loaded_files)
                 for i in range(len(outlier_contents))]
    distances = [0] * len(outlier_contents)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(compute_min_dist, args): args[0] for args in args_list}
        for future in tqdm(as_completed(futures), total=len(futures), desc="⚡ Computing initial distances"):
            i, dist = future.result()
            distances[i] = dist

    selected_indices = []
    while True:
        max_idx = int(np.argmax(distances))
        max_dist = distances[max_idx]
        if max_dist < dthreshold:
            break

        new_prototypes[proto_id_counter] = outlier_contents[max_idx]
        new_proto_keys[proto_id_counter] = loaded_files[max_idx]
        selected_indices.append(max_idx)
        proto_id_counter += 1

        args_list = [(i, outlier_contents, max_idx, distances, loaded_files, selected_indices)
                     for i in range(len(outlier_contents))]
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(update_distance, args): args[0] for args in args_list}
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc=f"⚡ Updating distances (FPF selected={len(selected_indices)})"):
                i = futures[future]
                distances[i] = future.result()

    return new_prototypes, new_proto_keys, proto_id_counter

# -------------------------------
# Save prototypes & cluster info
# -------------------------------
def save_prototypes(prototypes, proto_keys, batch_num):
    filename = f"incremental_prototypes_batch{batch_num}.json"
    with open(filename, "w") as f:
        json.dump({str(k): v for k, v in proto_keys.items()}, f, indent=2)
    print(f"✅ Saved prototypes → {filename}")

def save_cluster_info(full_assignments, batch_num):
    filename = f"cluster_info_batch{batch_num}.json"
    with open(filename, "w") as f:
        json.dump(full_assignments, f, indent=2)
    print(f"✅ Saved cluster info → {filename}")

# -------------------------------
# Pruning: Option 3 + Option 5
# -------------------------------
def prune_prototypes(prototypes, proto_keys, full_assignments, proto_last_used,
                     current_batch, X=4, max_prototypes=2000):
    # Option 3: Remove unused prototypes
    unused_protos = [pid for pid, last in proto_last_used.items() if last <= current_batch - X]
    for pid in unused_protos:
        prototypes.pop(pid, None)
        proto_keys.pop(pid, None)
        proto_last_used.pop(pid, None)
        full_assignments.pop(pid, None)
        print(f"🗑️ Removed unused prototype {pid} (last used batch {current_batch-X})")

    # Option 5: Limit total prototypes by utility
    if len(prototypes) > max_prototypes:
        utilities = {}
        for pid, files in full_assignments.items():
            if not files: continue
            ncd_vals = [ncd(prototypes[pid], load_file_bytes("rendered_pages_parallel", f)[1],
                            x_key=proto_keys[pid], y_key=f) for f in files]
            mean_ncd = np.mean(ncd_vals) if ncd_vals else 1
            utilities[pid] = len(files) / mean_ncd
        # Remove lowest utility prototypes
        sorted_protos = sorted(utilities.items(), key=lambda x: x[1])
        for pid, _ in sorted_protos[:len(prototypes)-max_prototypes]:
            prototypes.pop(pid, None)
            proto_keys.pop(pid, None)
            proto_last_used.pop(pid, None)
            full_assignments.pop(pid, None)
            print(f"🗑️ Removed low-utility prototype {pid} to maintain budget")

# -------------------------------
# Incremental clustering from batch 8
# -------------------------------
def incremental_clustering_continue(batch_start=8, batch_size=400,
                                    folder="rendered_pages_parallel",
                                    dthreshold=0.25, seed=42,
                                    csv_file="batch_stats.csv",
                                    max_workers=6, chunk_size=4):
    # Load previous state
    last_batch = batch_start - 1
    with open(f"incremental_prototypes_batch{last_batch}.json") as f:
        proto_keys = {int(k): v for k, v in json.load(f).items()}
    with open(f"cluster_info_batch{last_batch}.json") as f:
        full_assignments = defaultdict(list, json.load(f))

    # Initialize proto_last_used
    proto_last_used = {}
    for pid, files in full_assignments.items():
        if files:
            proto_last_used[pid] = last_batch
        else:
            proto_last_used[pid] = last_batch - 4

    # Load prototype content
    prototypes = {}
    for pid, fname in proto_keys.items():
        try:
            with open(f"{folder}/{fname}", "rb") as f:
                prototypes[pid] = f.read()
        except:
            print(f"⚠️ Prototype file missing: {fname}")

    all_files = sorted(f for f in os.listdir(folder) if f.endswith(".html"))
    # Only process new files
    processed_files = [f for files in full_assignments.values() for f in files]
    remaining_files = [f for f in all_files if f not in processed_files]
    batches = [remaining_files[i:i+batch_size] for i in range(0, len(remaining_files), batch_size)]

    # CSV logging
    if not os.path.exists(csv_file):
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["batch_num", "total_files", "outliers", "new_prototypes", "total_prototypes"])

    current_batch = batch_start
    prototype_counts = []

    for batch in batches:
        print(f"\n🚀 Processing batch {current_batch} with {len(batch)} files")

        assignments = assign_to_prototypes_parallel(batch, prototypes, proto_keys, folder,
                                                    dthreshold=dthreshold, max_workers=max_workers,
                                                    chunk_size=chunk_size)
        outliers = [f for f, (pid, dist) in assignments.items() if dist > dthreshold]
        matched_files = [f for f in batch if f not in outliers]

        # Update assignments and last used
        for f in matched_files:
            pid, _ = assignments[f]
            full_assignments[pid].append(f)
            proto_last_used[pid] = current_batch

        # FPF for outliers
        new_protos, new_proto_keys, proto_id_counter = fpf_threshold_on_demand(
            outliers, folder, prototypes, proto_keys,
            dthreshold=dthreshold, seed=seed, max_workers=max_workers
        )
        prototypes.update(new_protos)
        proto_keys.update(new_proto_keys)
        for pid in new_protos.keys():
            proto_last_used[pid] = current_batch

        # Apply pruning
        prune_prototypes(prototypes, proto_keys, full_assignments, proto_last_used,
                         current_batch=current_batch, X=4, max_prototypes=2000)

        # Save results
        save_prototypes(prototypes, proto_keys, current_batch)
        save_cluster_info(full_assignments, current_batch)

        prototype_counts.append(len(prototypes))
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([current_batch, len(batch), len(outliers), len(new_protos), len(prototypes)])

        print(f"✅ Batch {current_batch} done | Outliers: {len(outliers)} | New protos: {len(new_protos)} | Total protos: {len(prototypes)}")

        current_batch += 1

    # Save caches
    with open(compressed_size_cache_file, "w") as f:
        json.dump(compressed_size_cache, f)
    with open(ncd_cache_file, "w") as f:
        json.dump(ncd_cache, f)
    print("💾 Compression & NCD caches saved")

    # Plot prototype evolution
    plt.plot(range(batch_start, batch_start+len(prototype_counts)), prototype_counts, marker='o')
    plt.xlabel("Batch Number")
    plt.ylabel("Total Prototypes")
    plt.title("Prototype Evolution Over Batches")
    plt.grid(True)
    plt.show()

# -------------------------------
# Entry point
# -------------------------------
if __name__ == "__main__":
    incremental_clustering_continue(
        batch_start=8,
        batch_size=400,
        folder="rendered_pages_parallel",
        dthreshold=0.25,
        seed=42,
        csv_file="batch_stats.csv",
        max_workers=6,
        chunk_size=4
    )
