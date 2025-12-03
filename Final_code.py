import os
import json
import lzma
from tqdm import tqdm

# -------------------------------
# Load caches (optional, speeds up NCD)
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
    Cx = compress_size(x, x_key)
    Cy = compress_size(y, y_key)
    Cxy = compress_size(x + y)
    dist = (Cxy - min(Cx, Cy)) / max(Cx, Cy) if max(Cx, Cy) > 0 else 0
    ncd_cache[key] = dist
    return dist


# -------------------------------
# Safe filename for Windows
# -------------------------------
def sanitize_filename(fname):
    fname = str(fname)
    fname = fname.replace("\\", "")
    fname = "".join(ch for ch in fname if ord(ch) >= 32)
    return fname.strip()


# -------------------------------
# Load all prototypes from batches 0-59
# -------------------------------
def load_all_prototypes(max_batch=59, folder="rendered_pages_parallel"):
    prototypes = {}
    proto_keys = {}
    for batch in range(max_batch + 1):
        proto_file = f"incremental_prototypes_batch{batch}.json"
        if not os.path.exists(proto_file):
            continue
        with open(proto_file) as f:
            proto_batch = {int(k): sanitize_filename(v) for k, v in json.load(f).items()}
        for pid, fname in proto_batch.items():
            path = os.path.join(folder, fname)
            if os.path.exists(path):
                with open(path, "rb") as f2:
                    prototypes[pid] = f2.read()
                proto_keys[pid] = fname
    return prototypes, proto_keys


# -------------------------------
# Load files in a batch
# -------------------------------
def load_batch_files(batch_folder):
    files = sorted(f for f in os.listdir(batch_folder) if f.endswith(".html"))
    contents = {}
    for f in files:
        safe_name = sanitize_filename(f)
        path = os.path.join(batch_folder, safe_name)
        try:
            with open(path, "rb") as ff:
                contents[f] = ff.read()
        except Exception as e:
            print(f"⚠️ Could not read {f}: {e}")
    return contents


# -------------------------------
# Test batch
# -------------------------------
def test_batch(batch_folder="rendered_pages_parallel", max_batch=59, dthreshold=0.25):
    print("🔄 Loading prototypes...")
    prototypes, proto_keys = load_all_prototypes(max_batch=max_batch, folder=batch_folder)

    print(f"Loaded {len(prototypes)} prototypes from batches 0-{max_batch}")

    batch_files = load_batch_files(batch_folder)
    print(f"Loaded {len(batch_files)} files in test batch")

    phishing_count = 0
    total_files = len(batch_files)

    print("⚡ Computing NCD distances for testing...")
    for fname, content in tqdm(batch_files.items()):
        min_dist = float("inf")
        assigned_proto = None
        for pid, proto_data in prototypes.items():
            dist = ncd(content, proto_data, x_key=fname, y_key=proto_keys[pid])
            if dist < min_dist:
                min_dist = dist
                assigned_proto = pid

        # Flag as phishing if distance < threshold
        if min_dist <= dthreshold:
            phishing_count += 1

    percentage = (phishing_count / total_files * 100) if total_files > 0 else 0
    print(f"\n✅ Batch test complete: {phishing_count}/{total_files} files flagged as phishing")
    print(f"📊 Phishing percentage: {percentage:.2f}%")
    return phishing_count, total_files, percentage


if __name__ == "__main__":
    test_batch(
        batch_folder="rendered_pages_parallel",
        max_batch=59,
        dthreshold=0.25
    )
