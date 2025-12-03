import os
import json
import numpy as np
import zstandard as zstd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# --- Config ---
PHISHING_FOLDER = "phishing_pages"
PROTOTYPE_JSON_PATH = "prototypes.json"
CLUSTER_DISTANCE_THRESHOLD = 0.35
ZSTD_LEVEL = 3

# --- Compression & NCD ---
def compress_size(data: bytes, compressor) -> int:
    return len(compressor.compress(data))

def ncd_zstd(x: bytes, y: bytes, compressor) -> float:
    Cx = compress_size(x, compressor)
    Cy = compress_size(y, compressor)
    Cxy = compress_size(x + y, compressor)
    return max(0.0, (Cxy - min(Cx, Cy)) / max(Cx, Cy))

# --- File Loader ---
def load_html_files(folder, limit=None):
    files, contents = [], []
    count = 0
    for fname in os.listdir(folder):
        if fname.endswith(".html"):
            path = os.path.join(folder, fname)
            try:
                with open(path, "rb") as f:
                    content = f.read()
                files.append(fname)
                contents.append(content)
                count += 1
                if limit and count >= limit:
                    break
            except Exception as e:
                print(f"❌ Error reading {fname}: {e}")
    return files, contents

# --- Distance Matrix ---
def compute_distance_matrix(contents, compressor):
    n = len(contents)
    dist_matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            dist = ncd_zstd(contents[i], contents[j], compressor)
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    return dist_matrix

# --- Clustering ---
def cluster_phishing_pages(dist_matrix, threshold=0.35):
    condensed = squareform(dist_matrix)
    linkage_matrix = linkage(condensed, method="average")
    clusters = fcluster(linkage_matrix, t=threshold, criterion="distance")
    return clusters

# --- Prototype Selection ---
def extract_prototypes(files, contents, clusters, compressor):
    prototypes = {}
    cluster_ids = set(clusters)

    for cid in cluster_ids:
        indices = [i for i, c in enumerate(clusters) if c == cid]
        if len(indices) == 1:
            prototypes[str(cid)] = files[indices[0]]
            continue

        best_index = -1
        best_score = float('inf')
        for i in indices:
            avg_dist = np.mean([
                ncd_zstd(contents[i], contents[j], compressor)
                for j in indices if i != j
            ])
            if avg_dist < best_score:
                best_score = avg_dist
                best_index = i
        prototypes[str(cid)] = files[best_index]

    return prototypes

# --- Save Prototypes ---
def save_prototypes(prototypes, output_path=PROTOTYPE_JSON_PATH):
    with open(output_path, "w") as f:
        json.dump(prototypes, f, indent=2)
    print(f"✅ Prototypes saved to: {output_path}")

# --- Main Pipeline ---
def main(limit=None):
    print("📂 Loading HTML phishing pages...")
    files, contents = load_html_files(PHISHING_FOLDER, limit=limit)
    print(f"✅ Loaded {len(files)} files.")

    print("🧠 Computing Zstd-NCD distance matrix...")
    compressor = zstd.ZstdCompressor(level=ZSTD_LEVEL)
    dist_matrix = compute_distance_matrix(contents, compressor)

    print("🔗 Performing clustering...")
    clusters = cluster_phishing_pages(dist_matrix, threshold=CLUSTER_DISTANCE_THRESHOLD)
    print(f"✅ Found {len(set(clusters))} clusters.")

    print("📌 Extracting cluster prototypes...")
    prototypes = extract_prototypes(files, contents, clusters, compressor)
    print(f"✅ Extracted {len(prototypes)} prototypes.")

    print("💾 Saving prototype metadata...")
    save_prototypes(prototypes)

    print("🎉 All done.")

# --- Run ---
if __name__ == "__main__":
    main(limit=2000)  # You can change this limit as needed (e.g., 12000)
