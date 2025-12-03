import os
import random
import lzma
import numpy as np
import multiprocessing
from collections import defaultdict
import json

# --- Compression & NCD ---
def compress_size(data: bytes) -> int:
    return len(lzma.compress(data))

def ncd(x: bytes, y: bytes) -> float:
    """Normalized Compression Distance (NCD) between two files"""
    Cx = compress_size(x)
    Cy = compress_size(y)
    Cxy = compress_size(x + y)
    return (Cxy - min(Cx, Cy)) / max(Cx, Cy)

# --- Parallelized Distance Computation ---
def compute_pairwise_ncd(args):
    i, j, contents = args
    return (i, j, ncd(contents[i], contents[j]))

def compute_distance_matrix_parallel(contents):
    """Compute pairwise NCD distances in parallel"""
    n = len(contents)
    index_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    with multiprocessing.get_context("spawn").Pool(processes=4) as pool:
        results = pool.map(compute_pairwise_ncd, [(i, j, contents) for i, j in index_pairs])
    
    dist_matrix = np.zeros((n, n), dtype=np.float32)
    for i, j, dist in results:
        dist_matrix[i, j] = dist
        dist_matrix[j, i] = dist  # Symmetric
    return dist_matrix

# --- Prototype Extraction: FPF (Furthest Point First) ---
def fpf_prototype_extraction(files, contents, n_prototypes):
    """Extract prototypes using FPF (Furthest Point First)"""
    prototypes = []
    initial_idx = random.choice(range(len(contents)))
    prototypes.append(initial_idx)
    
    while len(prototypes) < n_prototypes:
        max_dist = -1
        next_proto_idx = -1
        
        for i, content in enumerate(contents):
            if i not in prototypes:
                min_dist_to_proto = min(ncd(contents[i], contents[p]) for p in prototypes)
                if min_dist_to_proto > max_dist:
                    max_dist = min_dist_to_proto
                    next_proto_idx = i

        prototypes.append(next_proto_idx)
    return prototypes

# --- Clustering Based on Distance Threshold ---
def threshold_clustering(contents, dist_matrix, threshold, prototype_indices):
    """Assign each file to the closest prototype within the threshold"""
    clusters = defaultdict(list)
    prototype_map = {i: p_idx for i, p_idx in enumerate(prototype_indices)}

    for i in range(len(contents)):
        best_proto = None
        min_dist = float("inf")
        for cluster_id, proto_idx in prototype_map.items():
            dist = dist_matrix[i][proto_idx]
            if dist < threshold and dist < min_dist:
                best_proto = cluster_id
                min_dist = dist
        if best_proto is not None:
            clusters[best_proto].append(i)
        else:
            new_id = len(prototype_map)
            prototype_map[new_id] = i
            clusters[new_id].append(i)
    return clusters, prototype_map

# --- Extract Final Prototype from Each Cluster ---
def extract_cluster_prototypes(clusters, dist_matrix, files):
    final_prototypes = {}
    for cluster_id, indices in clusters.items():
        if len(indices) == 1:
            final_prototypes[cluster_id] = files[indices[0]]
            continue
        best_idx = min(indices, key=lambda i: np.mean([dist_matrix[i][j] for j in indices]))
        final_prototypes[cluster_id] = files[best_idx]
    return final_prototypes

# --- Save to JSON ---
def save_prototypes_to_json(prototypes, filename="prototypes_NCD.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(prototypes, f, ensure_ascii=False, indent=4)
    print(f"✅ Prototypes saved to {filename}")

def save_full_cluster_info(clusters, files, filename="clusters_with_prototypes_NCD.json"):
    full_info = {}
    for cluster_id, indices in clusters.items():
        members = [files[i] for i in indices]
        full_info[cluster_id] = {
            "prototype": members[0],
            "members": members,
            "count": len(members)
        }
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(full_info, f, ensure_ascii=False, indent=4)
    print(f"✅ Cluster info saved to {filename}")

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def load_single_file(path, fname):
    try:
        with open(path, "rb") as f:
            content = f.read()
        return fname, content
    except Exception as e:
        print(f"❌ Failed to load {fname}: {e}")
        return None

def load_html_files(folder="phishing_pages", max_workers=8):
    files = []
    contents = []
    counter = [0]  # Mutable counter in list for threading
    lock = threading.Lock()

    file_list = [fname for fname in os.listdir(folder) if fname.endswith(".html")]
    total = len(file_list)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(load_single_file, os.path.join(folder, fname), fname)
                   for fname in file_list]

        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result:
                fname, content = result
                files.append(fname)
                contents.append(content)

            # Thread-safe counter + progress
            with lock:
                counter[0] += 1
                if counter[0] % 100 == 0 or counter[0] == total:
                    print(f"🔄 Loaded {counter[0]} / {total} files...")

    return files, contents



# --- Main ---
def main():
    folder = "rendered_pages_parallel"
    print(f"📂 Loading files from {folder}...")
    files, contents = load_html_files(folder)
    print(f"✅ Loaded {len(files)} files.")

    print("⚙️ Extracting initial prototypes with FPF...")
    proto_indices = fpf_prototype_extraction(files, contents, n_prototypes=50)

    print("🔄 Computing NCD matrix in parallel...")
    dist_matrix = compute_distance_matrix_parallel(contents)

    print("📊 Clustering files using threshold...")
    threshold = 0.25
    clusters, prototype_map = threshold_clustering(contents, dist_matrix, threshold, proto_indices)

    print(f"📦 Total clusters formed: {len(clusters)}")

    print("🧠 Extracting final cluster prototypes...")
    final_prototypes = extract_cluster_prototypes(clusters, dist_matrix, files)

    # Save results
    save_prototypes_to_json(final_prototypes, "prototypes_NCD.json")
    save_full_cluster_info(clusters, files, "clusters_with_prototypes_NCD.json")

if __name__ == "__main__":
    main()
