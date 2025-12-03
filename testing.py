import os
import json
import lzma
from tqdm import tqdm
from multiprocessing import Pool

# -------------------------------
# Load caches
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


# -------------------------------
# Fast LZMA compressor (preset=0)
# -------------------------------
def compress_size(data: bytes, key: str = None) -> int:
    if key and key in compressed_size_cache:
        return compressed_size_cache[key]

    size = len(lzma.compress(data, preset=0))

    if key:
        compressed_size_cache[key] = size

    return size


# -------------------------------
# Normalized Compression Distance
# -------------------------------
def ncd(x: bytes, y: bytes, x_key=None, y_key=None) -> float:
    pair_key = f"{x_key}:{y_key}"
    if pair_key in ncd_cache:
        return ncd_cache[pair_key]

    Cx = compress_size(x, x_key)
    Cy = compress_size(y, y_key)

    concat_key = f"{x_key}+{y_key}"
    if concat_key in compressed_size_cache:
        Cxy = compressed_size_cache[concat_key]
    else:
        Cxy = compress_size(x + y)
        compressed_size_cache[concat_key] = Cxy

    dist = (Cxy - min(Cx, Cy)) / max(Cx, Cy) if max(Cx, Cy) > 0 else 0
    ncd_cache[pair_key] = dist
    return dist


# -------------------------------
# Worker functions (TOP LEVEL)
# -------------------------------

# Batch HTML file loader
def load_file_worker(path):
    try:
        with open(path, "rb") as f:
            return os.path.basename(path), f.read()
    except Exception as e:
        print(f"⚠️ Skipped file {path}: {e}")
        return None

# Prototype loader
def prototype_worker(path_pid_tuple):
    path, pid = path_pid_tuple
    try:
        with open(path, "rb") as f:
            data = f.read()
        return pid, os.path.basename(path), data
    except Exception as e:
        print(f"⚠️ Skipped prototype {path}: {e}")
        return None

# NCD computation worker
def compute_min_dist(args):
    fname, content, prototypes, proto_keys, dthreshold = args

    min_dist = float("inf")
    for pid, proto_data in prototypes.items():
        dist = ncd(content, proto_data, x_key=fname, y_key=proto_keys[pid])
        if dist < min_dist:
            min_dist = dist
        if min_dist <= dthreshold:  # early exit
            break
    return fname, min_dist


# -------------------------------
# Load all prototypes in parallel (unique files only)
# -------------------------------
def load_all_prototypes(max_batch=59, folder="rendered_pages_parallel"):
    proto_files_set = set()
    proto_map = {}  # map full_path -> pid

    for batch in range(max_batch + 1):
        proto_file = f"incremental_prototypes_batch{batch}.json"
        if not os.path.exists(proto_file):
            continue

        with open(proto_file) as f:
            proto_batch = {int(k): v for k, v in json.load(f).items()}
        for pid, fname in proto_batch.items():
            full_path = os.path.join(folder, fname)
            proto_files_set.add(full_path)
            proto_map[full_path] = pid

    # Prepare list of tuples for multiprocessing
    proto_files = [(path, proto_map[path]) for path in proto_files_set]

    print(f"📂 Loading {len(proto_files)} unique prototypes in parallel using 8 workers...")

    prototypes = {}
    proto_keys = {}

    with Pool(processes=8) as pool:
        for result in pool.imap_unordered(prototype_worker, proto_files):
            if result:
                pid, fname, data = result
                prototypes[pid] = data
                proto_keys[pid] = fname

    print(f"✅ Loaded {len(prototypes)} prototypes from batches 0-{max_batch}")
    return prototypes, proto_keys


# -------------------------------
# Load only a specific batch of HTML files
# -------------------------------
def load_batch_files(batch_folder, batch_number, batch_size=300):
    # List all HTML files in folder (order in filesystem)
    all_files = [f for f in os.listdir(batch_folder) if f.endswith(".html")]

    # Determine start and end indices for the batch
    start_idx = batch_number * batch_size
    end_idx = start_idx + batch_size
    batch_files_list = all_files[start_idx:end_idx]

    batch_data = {}
    batch_paths = [os.path.join(batch_folder, f) for f in batch_files_list]

    print(f"📂 Loading {len(batch_paths)} HTML files for batch {batch_number} using 8 workers...")

    with Pool(processes=8) as pool:
        for result in pool.imap_unordered(load_file_worker, batch_paths):
            if result:
                fname, data = result
                batch_data[fname] = data

    return batch_data


# -------------------------------
# Main batch test
# -------------------------------
def test_batch(batch_folder="rendered_pages_parallel", batch_number=60, batch_size=300, max_batch=59, dthreshold=0.25):
    print("🔄 Loading prototypes...")
    prototypes, proto_keys = load_all_prototypes(max_batch=max_batch, folder=batch_folder)

    batch_files = load_batch_files(batch_folder, batch_number, batch_size)
    print(f"Loaded {len(batch_files)} files for testing batch {batch_number}")

    tasks = [
        (fname, content, prototypes, proto_keys, dthreshold)
        for fname, content in batch_files.items()
    ]

    print("⚡ Using max 8 CPU workers for parallel NCD calculation…")

    phishing_count = 0
    total_files = len(tasks)

    with Pool(processes=8) as pool:
        for fname, min_dist in tqdm(pool.imap_unordered(compute_min_dist, tasks),
                                    total=total_files):
            if min_dist <= dthreshold:
                phishing_count += 1

    percentage = (phishing_count / total_files * 100) if total_files else 0

    print(f"\n✅ Batch test complete: {phishing_count}/{total_files} files flagged as phishing")
    print(f"📊 Phishing percentage: {percentage:.2f}%")

    # Save caches
    with open(compressed_size_cache_file, "w") as f:
        json.dump(compressed_size_cache, f)
    with open(ncd_cache_file, "w") as f:
        json.dump(ncd_cache, f)

    return phishing_count, total_files, percentage


# -------------------------------
# Entry point
# -------------------------------
if __name__ == "__main__":
    test_batch(
        batch_folder="rendered_pages_parallel",
        batch_number=61,   
        batch_size=300,    
        max_batch=59,      
        dthreshold=0.25
    )
