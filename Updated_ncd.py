import os
import json
import lzma
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

# ----------------------------
# Compression and NCD
# ----------------------------
compressed_size_cache = {}
ncd_cache = {}

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
    Cx = compress_size(x, x_key)
    Cy = compress_size(y, y_key)
    Cxy = compress_size(x + y)
    dist = (Cxy - min(Cx, Cy)) / max(Cx, Cy)
    ncd_cache[key] = dist
    return dist

# ----------------------------
# Load Files (Single-threaded here for clarity)
# ----------------------------
def load_single_file(path, fname):
    try:
        with open(path, "rb") as f:
            content = f.read()
        return fname, content
    except Exception:
        return None

def load_html_files(folder, file_list):
    files, contents = [], []
    for fname in tqdm(file_list, desc="📂 Loading HTML files"):
        result = load_single_file(os.path.join(folder, fname), fname)
        if result:
            fname, content = result
            files.append(fname)
            contents.append(content)
    return files, contents

# ----------------------------
# Compute + Cache NCD Matrix
# ----------------------------
def compute_ncd_matrix(contents, keys, cache_file=None, force_recompute=False):
    if cache_file and os.path.exists(cache_file) and not force_recompute:
        print(f"🧠 Loading cached NCD matrix from {cache_file}")
        return np.load(cache_file)

    n = len(contents)
    dist_matrix = np.zeros((n, n), dtype=np.float32)
    for i in tqdm(range(n), desc="🔁 Computing NCD matrix"):
        for j in range(i + 1, n):
            dist = ncd(contents[i], contents[j], keys[i], keys[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    if cache_file:
        np.save(cache_file, dist_matrix)
        print(f"💾 Saved NCD matrix to {cache_file}")

    return dist_matrix

# ----------------------------
# FPF Prototype Extraction
# ----------------------------
def fpf_from_matrix(dist_matrix, num_prototypes):
    n = dist_matrix.shape[0]
    if n == 0 or num_prototypes == 0:
        return []

    num_prototypes = min(num_prototypes, n)
    candidates = list(range(n))
    random.shuffle(candidates)
    prototypes = [candidates.pop()]  # Start with a random point

    while len(prototypes) < num_prototypes and candidates:
        max_dist = -1
        next_proto = None
        for i in candidates:
            min_dist = min(dist_matrix[i][p] for p in prototypes)
            if min_dist > max_dist:
                max_dist = min_dist
                next_proto = i
        if next_proto is not None:
            prototypes.append(next_proto)
            candidates.remove(next_proto)
        else:
            break

    return prototypes

# ----------------------------
# Parallel Prototype Assignment (with processes)
# ----------------------------
def assign_single(i, content, key, prototypes, proto_keys):
    """Worker function for assigning one file to the nearest prototype"""
    min_dist = float('inf')
    best_proto = None
    for pid, proto_content in prototypes.items():
        dist = ncd(content, proto_content, key, proto_keys[pid])
        if dist < min_dist:
            min_dist = dist
            best_proto = pid
    return i, best_proto, min_dist

def assign_to_prototypes_parallel(contents, keys, prototypes, proto_keys, threshold=0.25, max_workers=8):
    assignments = defaultdict(list)
    outliers = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(assign_single, i, contents[i], keys[i], prototypes, proto_keys)
            for i in range(len(contents))
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="📌 Assigning to prototypes (parallel processes)"):
            i, best_proto, min_dist = future.result()
            if min_dist <= threshold:
                assignments[best_proto].append(i)
            else:
                outliers.append(i)

    return assignments, outliers

# ----------------------------
# Save Info
# ----------------------------
def save_prototypes(prototypes, filename="incremental_prototypes.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(prototypes, f, indent=4)
    print(f"✅ Prototypes saved to {filename}")

def save_cluster_info(assignments, filename="cluster_info.json"):
    cluster_data = {}
    for pid, member_files in assignments.items():  # member_files are already filenames
        cluster_data[pid] = {
            "prototype": f"Prototype_{pid}.html",
            "members": member_files,
            "count": len(member_files)
        }
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(cluster_data, f, indent=4)
    print(f"📁 Cluster info saved to {filename}")

# ----------------------------
# Main Pipeline
# ----------------------------
def incremental_clustering(folder="rendered_pages_parallel", batch_size=2000, n_initial_prototypes=20, threshold=0.25, force_recompute=False, max_workers=8):
    all_files = sorted([f for f in os.listdir(folder) if f.endswith(".html")])
    total_files = len(all_files)
    batches = [all_files[i:i+batch_size] for i in range(0, total_files, batch_size)]

    prototypes = {}     # pid -> content
    proto_keys = {}     # pid -> filename
    proto_id_counter = 0
    full_assignments = defaultdict(list)

    for batch_num, batch_files in enumerate(batches):
        print(f"\n📦 Processing Batch {batch_num + 1}/{len(batches)} - {len(batch_files)} files")
        files, contents = load_html_files(folder, batch_files)
        keys = files

        # warm up compression cache
        for i, content in enumerate(contents):
            compress_size(content, keys[i])

        if batch_num == 0:
            print("✨ First batch: extracting initial prototypes via FPF...")
            matrix_path = f"ncd_cache_batch_{batch_num}.npy"
            dist_matrix = compute_ncd_matrix(contents, keys, cache_file=matrix_path, force_recompute=force_recompute)
            fpf_indices = fpf_from_matrix(dist_matrix, n_initial_prototypes)
            for idx in fpf_indices:
                if 0 <= idx < len(contents):
                    pid = str(proto_id_counter)
                    prototypes[pid] = contents[idx]
                    proto_keys[pid] = keys[idx]
                    proto_id_counter += 1
        else:
            # ✅ Use the parallel assignment with processes
            assignments, outliers = assign_to_prototypes_parallel(contents, keys, prototypes, proto_keys, threshold, max_workers=max_workers)
            for pid, indices in assignments.items():
                full_assignments[pid].extend([files[i] for i in indices])

            if outliers:
                print(f"⚠️ Found {len(outliers)} outliers — extracting local prototypes...")
                if len(outliers) == 1:
                    idx = outliers[0]
                    pid = str(proto_id_counter)
                    prototypes[pid] = contents[idx]
                    proto_keys[pid] = keys[idx]
                    proto_id_counter += 1
                elif len(outliers) > 1:
                    outlier_contents = [contents[i] for i in outliers]
                    outlier_keys = [keys[i] for i in outliers]
                    out_matrix_path = f"ncd_cache_batch_{batch_num}_outliers.npy"
                    dist_matrix = compute_ncd_matrix(outlier_contents, outlier_keys, cache_file=out_matrix_path, force_recompute=force_recompute)
                    num_protos = min(5, len(outliers))
                    new_proto_indices = fpf_from_matrix(dist_matrix, num_protos)
                    for idx in new_proto_indices:
                        if 0 <= idx < len(outlier_contents):
                            pid = str(proto_id_counter)
                            prototypes[pid] = outlier_contents[idx]
                            proto_keys[pid] = outlier_keys[idx]
                            proto_id_counter += 1
            else:
                print("✅ All pages matched existing prototypes.")

    # save everything
    proto_filenames = {pid: f"Prototype_{pid}.html" for pid in prototypes.keys()}
    save_prototypes(proto_filenames)
    save_cluster_info(full_assignments, filename="cluster_info.json")

# ----------------------------
# Entry Point
# ----------------------------
if __name__ == "__main__":
    incremental_clustering(
        folder="rendered_pages_parallel",
        batch_size=300,
        n_initial_prototypes=20,
        threshold=0.25,
        force_recompute=False,
        max_workers=8
    )
