import os
import requests
import json
import numpy as np
import zstandard as zstd
from sklearn.metrics.pairwise import cosine_similarity

# --- Config ---
PROTOTYPE_PATH = "prototypes_zstd.json"         # Precomputed prototype map: cluster_id → filename
PHISHING_FOLDER = "phishing_pages"         # Folder where prototype HTMLs are stored
SIMILARITY_THRESHOLD = 0.35                # Lower is more strict (NCD ∈ [0, 1])
ZSTD_LEVEL = 3

# --- Zstd Compression ---
def compress_size(data: bytes) -> int:
    cctx = zstd.ZstdCompressor(level=ZSTD_LEVEL)
    compressed = cctx.compress(data)
    return len(compressed)

def ncd_zstd(x: bytes, y: bytes) -> float:
    Cx = compress_size(x)
    Cy = compress_size(y)
    Cxy = compress_size(x + y)
    return (Cxy - min(Cx, Cy)) / max(Cx, Cy)

# --- Load Prototypes ---
def load_prototypes(prototype_file=PROTOTYPE_PATH, folder=PHISHING_FOLDER):
    with open(prototype_file, "r") as f:
        prototype_map = json.load(f)

    cluster_ids = list(prototype_map.keys())
    files = [prototype_map[cid] for cid in cluster_ids]
    htmls = []

    for fname in files:
        path = os.path.join(folder, fname)
        try:
            with open(path, "rb") as f:
                htmls.append(f.read())
        except Exception as e:
            print(f"⚠️ Failed to load {fname}: {e}")
            htmls.append(b"")  # empty fallback
    return cluster_ids, htmls

# --- Fetch Target HTML from URL ---
def fetch_html_bytes(url):
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return response.content
    except Exception as e:
        print(f"❌ Error fetching URL: {e}")
        return None

# --- NCD-Based Classification ---
def classify_url_with_ncd(url, prototype_ids, prototype_contents, threshold=SIMILARITY_THRESHOLD):
    test_html = fetch_html_bytes(url)
    if not test_html:
        return "error", 1.0, None

    min_ncd = float("inf")
    matched_cluster = None

    for idx, proto in zip(prototype_ids, prototype_contents):
        dist = ncd_zstd(test_html, proto)
        if dist < min_ncd:
            min_ncd = dist
            matched_cluster = idx

    if min_ncd <= threshold:
        return "phishing", min_ncd, matched_cluster
    else:
        return "legitimate", min_ncd, None

# --- Entry ---
if __name__ == "__main__":
    cluster_ids, proto_htmls = load_prototypes()
    input_url = input("🔗 Enter URL to classify: ").strip()

    result, score, cluster = classify_url_with_ncd(input_url, cluster_ids, proto_htmls)

    if result == "error":
        print("❌ Could not fetch or parse the URL.")
    elif result == "phishing":
        print(f"⚠️ Phishing detected. NCD = {score:.3f}, matched with cluster {cluster}")
    else:
        print(f"✅ Legitimate URL. NCD = {score:.3f}")
