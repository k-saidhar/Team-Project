import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Reuse your existing helpers (import them if they are in another file)
from Final_code import load_file_bytes, ncd_cache  

# -----------------------------
# Chunk assignment worker
# -----------------------------
def assign_chunk(files_chunk, prototypes, proto_keys, folder, dthreshold):
    results = []
    for file in files_chunk:
        path, data = load_file_bytes(folder, file)
        best_dist = float("inf")
        best_proto = None
        for pid, pdata in prototypes.items():
            dist = ncd_cache(data, pdata, key=(file, pid))
            if dist < best_dist:
                best_dist = dist
                best_proto = pid
        if best_dist > dthreshold:
            results.append((file, None, best_dist))  # outlier
        else:
            results.append((file, best_proto, best_dist))
    return results

# -----------------------------
# Parallel assignment with chunks
# -----------------------------
def assign_to_prototypes_parallel(files, prototypes, proto_keys, folder,
                                  dthreshold=0.4, max_workers=8, chunk_size=10):
    assignments = []
    chunks = [files[i:i+chunk_size] for i in range(0, len(files), chunk_size)]
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(assign_chunk, chunk, prototypes, proto_keys, folder, dthreshold)
                   for chunk in chunks]
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc=f"📌 Chunk size {chunk_size}"):
            assignments.extend(future.result())
    return assignments

# -----------------------------
# Benchmark runner
# -----------------------------
def benchmark_assignment(files, prototypes, proto_keys, folder, dthreshold=0.4, max_workers=8):
    chunk_sizes = [1, 5, 10, 20, 50]
    timings = {}
    for size in chunk_sizes:
        print(f"\n⏳ Testing chunk size {size} ...")
        t0 = time.time()
        assign_to_prototypes_parallel(files, prototypes, proto_keys,
                                      folder, dthreshold,
                                      max_workers=max_workers,
                                      chunk_size=size)
        elapsed = time.time() - t0
        timings[size] = elapsed
        print(f"✅ Chunk size {size}: {elapsed:.2f} sec")
    return timings

# -----------------------------
# Example usage (replace with your actual batch/prototypes)
# -----------------------------
if __name__ == "__main__":
    # Load one batch and prototypes as you normally would
    folder = "rendered_pages_parallel"
    files = os.listdir(folder)[:400]   # take 100 files for test run
    prototypes = {}    # <-- load from your clustering state
    proto_keys = []    # <-- load from your clustering state
    
    results = benchmark_assignment(files, prototypes, proto_keys, folder,
                                   dthreshold=0.4, max_workers=6)
    print("\n📊 Benchmark summary:", results)
