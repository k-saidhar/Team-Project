import os
import zstandard as zstd
import numpy as np
import json
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from collections import defaultdict

# -------------------------
# Compression using Zstandard
# -------------------------
def compress_size(data: bytes) -> int:
    compressor = zstd.ZstdCompressor()
    return len(compressor.compress(data))

def ncd_zstd(x: bytes, y: bytes) -> float:
    Cx = compress_size(x)
    Cy = compress_size(y)
    Cxy = compress_size(x + y)
    return max(0.0,((Cxy - min(Cx, Cy)) / max(Cx, Cy)))

# -------------------------
# Load HTML files (batch-based)
# -------------------------
def load_html_files(folder, start=0, limit=500):
    all_files = sorted([f for f in os.listdir(folder) if f.endswith('.html')])
    selected_files = all_files[start:start+limit]
    contents = []
    actual_files = []
    for fname in selected_files:
        path = os.path.join(folder, fname)
        try:
            with open(path, 'rb') as f:
                contents.append(f.read())
            actual_files.append(fname)
        except Exception as e:
            print(f"Failed to load {fname}: {e}")
    return actual_files, contents

# -------------------------
# Distance Matrix for Initial Clustering
# -------------------------
def compute_distance_matrix(contents):
    n = len(contents)
    dist_matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            dist = ncd_zstd(contents[i], contents[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    return dist_matrix

# -------------------------
# Cluster Pages
# -------------------------
def cluster_pages(dist_matrix, threshold=0.3):
    condensed = squareform(dist_matrix)
    Z = linkage(condensed, method="average")
    return fcluster(Z, t=threshold, criterion='distance')

# -------------------------
# Extract Prototypes
# -------------------------
def extract_prototypes(files, contents, clusters):
    cluster_map = defaultdict(list)
    for idx, cluster_id in enumerate(clusters):
        cluster_map[cluster_id].append((files[idx], contents[idx]))

    prototypes = {}
    for cluster_id, items in cluster_map.items():
        if len(items) == 1:
            prototypes[str(cluster_id)] = {'filename': items[0][0], 'content': items[0][1]}
            continue
        min_avg_dist = float('inf')
        best_item = None
        for i, (file_i, content_i) in enumerate(items):
            avg_dist = 0
            for j, (file_j, content_j) in enumerate(items):
                if i != j:
                    avg_dist += ncd_zstd(content_i, content_j)
            avg_dist /= (len(items) - 1)
            if avg_dist < min_avg_dist:
                min_avg_dist = avg_dist
                best_item = (file_i, content_i)
        prototypes[str(cluster_id)] = {'filename': best_item[0], 'content': best_item[1]}
    return prototypes

# -------------------------
# Match New Files Incrementally
# -------------------------
def match_to_prototypes(new_file_content, prototypes, threshold=0.3):
    for pid, proto_data in prototypes.items():
        dist = ncd_zstd(new_file_content, proto_data['content'])
        if dist < threshold:
            return pid
    return None

# -------------------------
# Save Prototypes (excluding content)
# -------------------------
def save_prototypes(prototypes, path='prototypes_zstd.json'):
    light_protos = {k: v['filename'] for k, v in prototypes.items()}
    with open(path, 'w') as f:
        json.dump(light_protos, f, indent=2)

# -------------------------
# Main Training Function
# -------------------------
def train_initial_batch(folder, batch_size=700, threshold=0.3):
    files, contents = load_html_files(folder, start=0, limit=batch_size)
    dist_matrix = compute_distance_matrix(contents)
    clusters = cluster_pages(dist_matrix, threshold)
    prototypes = extract_prototypes(files, contents, clusters)
    save_prototypes(prototypes)
    return prototypes

# -------------------------
# Incremental Matching Function
# -------------------------
def process_incremental_batch(folder, prototypes, start, batch_size=700, threshold=0.3):
    new_files, new_contents = load_html_files(folder, start=start, limit=batch_size)
    assignments = {}
    new_cluster_id = max(map(int, prototypes.keys()), default=0) + 1

    for fname, content in zip(new_files, new_contents):
        match_id = match_to_prototypes(content, prototypes, threshold)
        if match_id is not None:
            assignments[fname] = match_id
        else:
            # Treat as new prototype
            prototypes[str(new_cluster_id)] = {'filename': fname, 'content': content}
            assignments[fname] = str(new_cluster_id)
            new_cluster_id += 1

    save_prototypes(prototypes)
    return assignments, prototypes

# -------------------------
# Entry Point
# -------------------------
def main():
    folder = "phishing_pages"
    batch_size = 700
    total_files = 42000

    print("⏳ Training on initial batch...")
    prototypes = train_initial_batch(folder, batch_size=batch_size)

    for start in range(batch_size, total_files, batch_size):
        print(f"\n📦 Processing batch from {start} to {start + batch_size}...")
        assignments, prototypes = process_incremental_batch(folder, prototypes, start, batch_size)
        print(f"✅ Assigned {len(assignments)} files in batch.")

    print(f"\n🎉 Incremental training complete. Total prototypes: {len(prototypes)}")

if __name__ == "__main__":
    main()
