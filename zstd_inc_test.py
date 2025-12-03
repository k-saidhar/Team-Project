import os
import requests
import json
import zstandard as zstd

# --- Config ---
PROTOTYPE_PATH = "prototypes_zstd.json"  # cluster_id -> prototype filename
PHISHING_FOLDER = "phishing_pages"        # folder containing prototype HTML files
SIMILARITY_THRESHOLD = 0.35                # NCD threshold; lower is stricter
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

# --- Load prototypes ---
def load_prototypes(prototype_file=PROTOTYPE_PATH, folder=PHISHING_FOLDER):
    with open(prototype_file, "r") as f:
        prototype_map = json.load(f)
    cluster_ids = list(prototype_map.keys())
    htmls = []
    for fname in prototype_map.values():
        path = os.path.join(folder, fname)
        try:
            with open(path, "rb") as f:
                htmls.append(f.read())
        except Exception as e:
            print(f"⚠️ Failed to load prototype file {fname}: {e}")
            htmls.append(b"")  # fallback empty bytes
    return cluster_ids, htmls

# --- Fetch live HTML ---
def fetch_html_bytes(url):
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return response.content
    except Exception as e:
        print(f"❌ Error fetching URL live: {e}")
        return None

# --- Classification ---
def classify_url_with_prototypes(url, prototype_ids, prototype_htmls, threshold=SIMILARITY_THRESHOLD):
    test_html = fetch_html_bytes(url)
    if not test_html:
        return "error", None, None
    
    min_ncd = float("inf")
    matched_cluster = None
    
    for cid, proto_html in zip(prototype_ids, prototype_htmls):
        dist = ncd_zstd(test_html, proto_html)
        if dist < min_ncd:
            min_ncd = dist
            matched_cluster = cid
    
    if min_ncd <= threshold:
        return "phishing", min_ncd, matched_cluster
    else:
        return "legitimate", min_ncd, None

# --- Main ---
if __name__ == "__main__":
    cluster_ids, proto_htmls = load_prototypes()
    url_to_test = input("🔗 Enter URL to classify: ").strip()
    
    result, score, cluster = classify_url_with_prototypes(url_to_test, cluster_ids, proto_htmls)
    
    if result == "error":
        print("❌ Could not fetch or parse the URL live.")
    elif result == "phishing":
        print(f"⚠️ Phishing detected! NCD = {score:.3f}, matched cluster = {cluster}")
    else:
        print(f"✅ URL classified as legitimate. NCD = {score:.3f}")
