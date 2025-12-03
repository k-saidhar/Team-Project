import os
import json
import lzma
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

# --------------------------------
# Compression and NCD (same as before)
# --------------------------------
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

# --------------------------------
# Load Files (multithreaded IO)
# --------------------------------
def load_single_file(path, fname):
    try:
        with open(path, "rb") as f:
            content = f.read()
        return fname, content
    except Exception:
        return None

def load_html_files(folder, file_list, max_workers=8):
    files, contents = [], []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(load_single_file, os.path.join(folder, fname), fname) for fname in file_list]
        for future in tqdm(as_completed(futures), total=len(futures), desc="📂 Loading HTML files"):
            result = future.result()
            if result:
                fname, content = result
                files.append(fname)
                contents.append(content)
    return files, contents

# --------------------------------
# Compute NCD matrix
# --------------------------------
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

# --------------------------------
# FPF Prototype Extraction
# --------------------------------
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

# --------------------------------
# Parallel assign_to_prototypes using ProcessPoolExecutor
# --------------------------------

def _assign_single(i, content, key, prototypes, proto_keys, threshold):
    min_dist = float('inf')
    best_proto = None
    for pid, proto_content in prototypes.items():
        dist = ncd(content, proto_content, key, proto_keys[pid])
        if dist < min_dist:
            min_dist = dist
            best_proto = pid
    if min_dist <= threshold:
        return (best_proto, i)
    else:
        return (None, i)

def assign_to_prototypes_parallel(contents, keys, prototypes, proto_keys, threshold=0.25, max_workers=4):
    assignments = defaultdict(list)
    outliers = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, (content, key) in enumerate(zip(contents, keys)):
            futures.append(executor.submit(_assign_single, i, content, key, prototypes, proto_keys, threshold))
        for future in tqdm(as_completed(futures), total=len(futures), desc="📌 Assigning to prototypes (parallel)"):
            proto, idx = future.result()
            if proto is not None:
                assignments[proto].append(idx)
            else:
                outliers.append(idx)

    return assignments, outliers

# --------------------------------
# Save / Load helpers
# --------------------------------
def save_prototypes(prototypes, filename="incremental_prototypes.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(prototypes, f, indent=4)
    print(f"✅ Prototypes saved to {filename}")

def save_cluster_info(assignments, file_map, filename="cluster_info.json"):
    cluster_data = {}
    for pid, indices in assignments.items():
        member_files = [file_map[i] for i in indices]
        cluster_data[pid] = {
            "prototype": f"Prototype_{pid}.html",
            "members": member_files,
            "count": len(member_files)
        }
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(cluster_data, f, indent=4)
    print(f"📁 Cluster info saved to {filename}")

def load_prototypes(filename):
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            proto_filenames = json.load(f)
        print(f"🧠 Loaded prototypes from {filename}")
        return proto_filenames
    return {}

def load_cluster_info(filename):
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            cluster_data = json.load(f)
        print(f"🧠 Loaded cluster info from {filename}")
        full_assignments = defaultdict(list)
        for pid, info in cluster_data.items():
            full_assignments[pid].extend(info.get("members", []))
        return full_assignments
    return defaultdict(list)

def find_last_processed_batch(prototype_folder="."):
    batches = []
    for f in os.listdir(prototype_folder):
        if f.startswith("prototypes_batch_") and f.endswith(".json"):
            try:
                num = int(f[len("prototypes_batch_"):-len(".json")])
                batches.append(num)
            except:
                pass
    if batches:
        return max(batches)
    return -1

# --------------------------------
# Main incremental clustering pipeline with autosave + resume + ProcessPoolExecutor
# --------------------------------
def incremental_clustering(
    folder="rendered_pages_parallel",
    batch_size=2000,
    n_initial_prototypes=50,
    threshold=0.25,
    force_recompute=False,
    max_workers_io=8,
    max_workers_proc=4,
    resume=False
):
    all_files = sorted([f for f in os.listdir(folder) if f.endswith(".html")])
    total_files = len(all_files)
    batches = [all_files[i:i+batch_size] for i in range(0, total_files, batch_size)]

    start_batch = 0
    prototypes = {}
    proto_keys = {}
    proto_id_counter = 0
    full_assignments = defaultdict(list)

    if resume:
        last_batch = find_last_processed_batch()
        if last_batch >= 0 and last_batch < len(batches):
            print(f"🔄 Resuming from batch {last_batch + 1}")
            start_batch = last_batch + 1

            proto_filenames = load_prototypes(f"prototypes_batch_{last_batch}.json")
            full_assignments = load_cluster_info(f"cluster_info_batch_{last_batch}.json")

            # Load prototype contents
            for pid, proto_file in proto_filenames.items():
                try:
                    with open(os.path.join(folder, proto_file), "rb") as f:
                        content = f.read()
                    prototypes[pid] = content
                    proto_keys[pid] = proto_file
                    proto_id_counter = max(proto_id_counter, int(pid) + 1)
                except Exception as e:
                    print(f"⚠️ Failed to load prototype file {proto_file}: {e}")

    for batch_num in range(start_batch, len(batches)):
        batch_files = batches[batch_num]
        print(f"\n📦 Processing Batch {batch_num + 1}/{len(batches)} - {len(batch_files)} files")
        files, contents = load_html_files(folder, batch_files, max_workers=max_workers_io)
        keys = files
        file_map = {i: files[i] for i in range(len(files))}

        # Cache compression sizes
        for i, content in enumerate(contents):
            compress_size(content, keys[i])

        if batch_num == 0 and not prototypes:
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
            assignments, outliers = assign_to_prototypes_parallel(
                contents, keys, prototypes, proto_keys, threshold, max_workers=max_workers_proc)

            for pid, indices in assignments.items():
                full_assignments[pid].extend([file_map[i] for i in indices])

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

                    matrix_path = f"ncd_cache_batch_{batch_num}_outliers.npy"
                    dist_matrix = compute_ncd_matrix(outlier_contents, outlier_keys, cache_file=matrix_path, force_recompute=force_recompute)
                    local_protos_idx = fpf_from_matrix(dist_matrix, max(1, int(n_initial_prototypes / 2)))

                    for idx in local_protos_idx:
                        pid = str(proto_id_counter)
                        prototypes[pid] = outlier_contents[idx]
                        proto_keys[pid] = outlier_keys[idx]
                        proto_id_counter += 1

                    # Assign outliers to local prototypes
                    local_assignments, local_outliers = assign_to_prototypes_parallel(
                        outlier_contents, outlier_keys, prototypes, proto_keys, threshold, max_workers=max_workers_proc)

                    for pid, indices in local_assignments.items():
                        full_assignments[pid].extend([outlier_keys[i] for i in indices])

                    if local_outliers:
                        print(f"⚠️ Warning: {len(local_outliers)} local outliers remain unassigned")

        # Autosave prototypes and clusters after each batch
        proto_files_map = {pid: proto_keys[pid] for pid in prototypes}
        save_prototypes(proto_files_map, filename=f"prototypes_batch_{batch_num}.json")

        # For saving clusters, we need to map keys back to indices in batch_files:
        # Here full_assignments has filenames, so we create a fake index map for saving
        cluster_map = {}
        for pid, members in full_assignments.items():
            cluster_map[pid] = {
                "prototype": proto_keys[pid],
                "members": members,
                "count": len(members)
            }
        with open(f"cluster_info_batch_{batch_num}.json", "w", encoding="utf-8") as f:
            json.dump(cluster_map, f, indent=4)
        print(f"💾 Autosaved prototypes and cluster info for batch {batch_num}")

    print("✅ Incremental clustering completed.")

if __name__ == "__main__":
    incremental_clustering()
