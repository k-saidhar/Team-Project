import os
import json
import lzma
import random
import csv
import time
import math
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib.pyplot as plt
import cProfile
import pstats
import io

# -------------------------------
# Persistent caches
# -------------------------------
COMPRESSED_SIZE_CACHE_FILE = "compressed_size_cache.json"
NCD_CACHE_FILE = "ncd_cache.json"

compressed_size_cache = {}
ncd_cache = {}

if os.path.exists(COMPRESSED_SIZE_CACHE_FILE):
    try:
        with open(COMPRESSED_SIZE_CACHE_FILE, "r", encoding="utf-8") as f:
            compressed_size_cache = json.load(f)
        print(f"✓ Loaded compressed size cache ({len(compressed_size_cache)} entries)")
    except Exception as e:
        print(f"⚠️ Failed to load compressed size cache: {e}")

if os.path.exists(NCD_CACHE_FILE):
    try:
        with open(NCD_CACHE_FILE, "r", encoding="utf-8") as f:
            ncd_cache = json.load(f)
        print(f"✓ Loaded NCD cache ({len(ncd_cache)} entries)")
    except Exception as e:
        print(f"⚠️ Failed to load NCD cache: {e}")

def compress_size(data: bytes, key: str = None) -> int:
    if key and key in compressed_size_cache:
        return compressed_size_cache[key]
    # Optimized preset from 9 to 6 for speed as per implementation plan
    size = len(lzma.compress(data, preset=6)) 
    if key:
        compressed_size_cache[key] = size
    return size

def ncd(x: bytes, y: bytes, x_key: str = None, y_key: str = None) -> float:
    key = f"{x_key}:{y_key}" if x_key and y_key else None
    if key and key in ncd_cache:
        return ncd_cache[key]

    Cx = compress_size(x, x_key)
    Cy = compress_size(y, y_key)
    Cxy = compress_size(x + y)

    if max(Cx, Cy) == 0:
        dist = 0.0
    else:
        dist = (Cxy - min(Cx, Cy)) / max(Cx, Cy)

    if key:
        ncd_cache[key] = dist
    return dist
def load_file_worker(args):
    folder, fname = args
    return load_file_bytes(folder, fname)

def load_files_parallel(folder, file_list, executor=None, max_workers=8):
    files, contents = [], []
    
    # Use provided executor or create a temporary one
    if executor is None:
        with ThreadPoolExecutor(max_workers=max_workers) as temp_executor:
            return _load_files_internal(folder, file_list, temp_executor)
    else:
        return _load_files_internal(folder, file_list, executor)

def _load_files_internal(folder, file_list, executor):
    files, contents = [], []
    args_list = [(folder, f) for f in file_list]
    for result in executor.map(load_file_worker, args_list):
        if result:
            fname, content = result
            files.append(fname)
            contents.append(content)
    return files, contents

# -------------------------------
# Helpers
# -------------------------------
def sanitize_filename(fname):
    fname = str(fname).replace("\\", "").strip()
    return "".join(ch for ch in fname if ord(ch) >= 32)

def load_file_bytes(folder: str, fname: str):
    try:
        path = os.path.join(folder, fname)
        with open(path, "rb") as f:
            return fname, f.read()
    except Exception as e:
        print(f"⚠️ Failed to load {fname}: {e}")
        return None

# -------------------------------
# Precompute all compressed sizes (very important!)
# -------------------------------
# -------------------------------
# Precompute all compressed sizes (very important!)
# -------------------------------
def precompute_worker(args):
    folder, fname = args
    result = load_file_bytes(folder, fname)
    if result:
        _, data = result
        compress_size(data, key=fname)

def precompute_compressed_sizes(folder: str, file_list: list, executor=None, max_workers: int = 12):
    print(f"Precomputing compressed sizes for {len(file_list)} files...")
    # Prepare args for top-level picklable worker
    args_list = [(folder, f) for f in file_list]
    
    if executor is None:
        with ThreadPoolExecutor(max_workers=max_workers) as temp_executor:
            # ThreadPool can handle lambdas, but for consistency we use the same structure
            list(tqdm(temp_executor.map(precompute_worker, args_list), total=len(file_list), desc="Precomputing C(x) sizes"))
    else:
        # If passed an executor (likely ProcessPool), we must use the picklable worker
        list(tqdm(executor.map(precompute_worker, args_list), total=len(file_list), desc="Precomputing C(x) sizes"))
    print("✓ Precomputation finished")


# -------------------------------
# Parallel assignment
# -------------------------------
def assign_chunk(args):
    chunk_id, chunk_files, prototypes, proto_keys, folder = args
    results = []
    for fname in chunk_files:
        result = load_file_bytes(folder, fname)
        if not result:
            results.append((fname, None, float("inf")))
            continue
        _, data = result
        min_dist = float("inf")
        best_pid = None
        for pid, proto_data in prototypes.items():
            dist = ncd(data, proto_data, x_key=fname, y_key=proto_keys[pid])
            if dist < min_dist:
                min_dist = dist
                best_pid = pid
        results.append((fname, best_pid, min_dist))
    return chunk_id, results, time.time() - 0 # Duration calc happens outside or we pass start time? 
    # Let's return just results and let caller measure wall time if needed, 
    # but strictly user wants to measure worker time.
    # We can measure inside.
    
def assign_chunk_timed(args):
    start_t = time.time()
    chunk_id, chunk_files, prototypes, proto_keys, folder = args
    results = []
    for fname in chunk_files:
        result = load_file_bytes(folder, fname)
        if not result:
            results.append((fname, None, float("inf")))
            continue
        _, data = result
        min_dist = float("inf")
        best_pid = None
        for pid, proto_data in prototypes.items():
            dist = ncd(data, proto_data, x_key=fname, y_key=proto_keys[pid])
            if dist < min_dist:
                min_dist = dist
                best_pid = pid
        results.append((fname, best_pid, min_dist))
    duration = time.time() - start_t
    return chunk_id, results, duration

def submit_assignment_tasks(executor, files, prototypes, proto_keys, folder, chunk_size=20):
    chunks = [files[i:i + chunk_size] for i in range(0, len(files), chunk_size)]
    futures = {}
    for i, chunk in enumerate(chunks):
        # We assume chunk_idx is getting unique across batches? 
        # Or relative to batch? Relative is fine for stats.
        fut = executor.submit(assign_chunk_timed, (i, chunk, prototypes, proto_keys, folder))
        futures[fut] = i
    return futures

def collect_assignment_results(futures, benchmark_csv=None, desc="Assigning files"):
    assignments = {}
    
    writer = None
    f_handle = None
    if benchmark_csv:
        if not os.path.exists(benchmark_csv):
             with open(benchmark_csv, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["chunk_id", "chunk_size", "time_sec"])
        f_handle = open(benchmark_csv, "a", newline="", encoding="utf-8")
        writer = csv.writer(f_handle)

    try:
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            chunk_idx = futures[future]
            try:
                c_id, results, duration = future.result()
                if writer:
                    writer.writerow([c_id, len(results), round(duration, 3)])
                for fname, pid, dist in results:
                    assignments[fname] = (pid, dist)
            except Exception as e:
                print(f" Chunk {chunk_idx} failed: {e}")
    finally:
        if f_handle:
            f_handle.close()
            
    return assignments

# Legacy wrapper if needed, but we will replace usages
def assign_to_prototypes_parallel(files, prototypes, proto_keys, folder, dthreshold=0.25, max_workers=12, chunk_size=20, benchmark_csv="benchmark_stats.csv", executor=None):
    if executor is None:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = submit_assignment_tasks(pool, files, prototypes, proto_keys, folder, chunk_size)
            return collect_assignment_results(futures, benchmark_csv)
    else:
        futures = submit_assignment_tasks(executor, files, prototypes, proto_keys, folder, chunk_size)
        return collect_assignment_results(futures, benchmark_csv)

# -------------------------------
# Farthest Point Sampling (FPF) for outliers
# -------------------------------
def compute_min_dist(args):
    i, data, fname, protos, proto_keys = args
    min_d = float("inf")
    for pid, proto in protos.items():
        d = ncd(data, proto, fname, proto_keys[pid])
        if d < min_d:
            min_d = d
    return i, min_d

def update_distance(args):
    i, contents, new_idx, distances, fnames, selected = args
    if i in selected:
        return 0.0
    d = ncd(contents[i], contents[new_idx], fnames[i], fnames[new_idx])
    return min(distances[i], d)

def fpf_threshold_on_demand(
    outlier_files: list,
    folder: str,
    existing_protos: dict,
    existing_keys: dict,
    dthreshold: float = 0.25,
    seed: int = 42,
    executor = None, # Passed executor
    max_workers: int = 12
):
    rng = random.Random(seed)
    files = list(outlier_files)
    rng.shuffle(files)

    new_protos, new_keys = {}, {}
    next_pid = max(existing_protos.keys(), default=-1) + 1

    _, contents = load_files_parallel(folder, files, executor=executor, max_workers=max_workers)
    if not contents:
        return new_protos, new_keys, next_pid

    # Pre-warm individual sizes
    for f, c in zip(files, contents):
        compress_size(c, key=f)

    distances = [0.0] * len(contents)

    # Helper function to submit tasks via executor
    def run_via_executor(func, args_iter):
        futures = {executor.submit(func, args): i for i, args in args_iter}
        results = {}
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()
        return results

    # Initial distances
    # Optimized: Pass only individual file content instead of entire contents list
    # This reduces pickling overhead from O(n) to O(1) per task
    futures = {
        executor.submit(compute_min_dist, (i, contents[i], files[i], existing_protos, existing_keys)): i
        for i in range(len(contents))
    }
    for fut in tqdm(as_completed(futures), total=len(futures), desc="Initial min distances"):
        i, d = fut.result()
        distances[i] = d

    selected = []
    while True:
        max_i = np.argmax(distances)
        max_d = distances[max_i]
        if max_d < dthreshold:
            break

        new_protos[next_pid] = contents[max_i]
        new_keys[next_pid] = files[max_i]
        selected.append(max_i)
        next_pid += 1

        futures = {
            executor.submit(update_distance, (i, contents, max_i, distances, files, selected)): i
            for i in range(len(contents))
        }
        for fut in as_completed(futures): # removed tqdm for inner loop to reduce noise? or can keep
             i = futures[fut]
             distances[i] = fut.result()

    return new_protos, new_keys, next_pid

# -------------------------------
# Save / Load helpers
# -------------------------------
def save_prototypes(prototypes: dict, proto_keys: dict, batch_num: int):
    fname = f"incremental_prototypes_batch{batch_num}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in proto_keys.items()}, f, indent=2)
    print(f"→ Saved prototypes: {fname}")

def save_cluster_info(assignments: dict, batch_num: int):
    fname = f"cluster_info_batch{batch_num}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(dict(assignments), f, indent=2)
    print(f"→ Saved cluster info: {fname}")

# -------------------------------
# Pruning
# -------------------------------
def prune_prototypes(
    prototypes: dict,
    proto_keys: dict,
    full_assignments: dict,
    proto_last_used: dict,
    current_batch: int,
    do_unused_pruning: bool = False,
    unused_threshold: int = 10,
    max_prototypes: int = 2000
):
    removed = 0

    # 1. Remove long-unused prototypes (only if assignment was slow)
    if do_unused_pruning:
        to_remove = [
            pid for pid, last in proto_last_used.items()
            if last <= current_batch - unused_threshold
        ]
        for pid in to_remove:
            prototypes.pop(pid, None)
            proto_keys.pop(pid, None)
            proto_last_used.pop(pid, None)
            full_assignments.pop(pid, None)
            removed += 1
        if removed:
            print(f"🗑 Removed {removed} unused prototypes (not used in last {unused_threshold} batches)")

    # 2. Hard limit by utility
    if len(prototypes) > max_prototypes:
        utilities = {}
        for pid, files in full_assignments.items():
            if not files:
                continue
            ncd_vals = []
            for f in files:
                result = load_file_bytes("rendered_pages_parallel", f)
                if result:
                    _, data = result
                    ncd_vals.append(ncd(prototypes[pid], data, proto_keys[pid], f))
            mean_ncd = np.mean(ncd_vals) if ncd_vals else 1.0
            utilities[pid] = len(files) / mean_ncd

        sorted_by_utility = sorted(utilities.items(), key=lambda x: x[1])
        to_keep = len(prototypes) - max_prototypes
        for pid, _ in sorted_by_utility[:to_keep]:
            prototypes.pop(pid, None)
            proto_keys.pop(pid, None)
            proto_last_used.pop(pid, None)
            full_assignments.pop(pid, None)
            removed += 1
        if removed:
            print(f"🗑 Removed {removed} low-utility prototypes (budget exceeded)")

# -------------------------------
# Refine Assignments (Delta Update)
# -------------------------------
def refine_assignments_delta(
    current_assignments: dict,
    batch_files: list,
    new_prototypes: dict,
    new_proto_keys: dict,
    folder: str,
    executor,
    dthreshold: float,
    chunk_size: int = 20
):
    """
    Checks if any files in batch_files are closer to the new_prototypes 
    than their currently assigned prototype.
    """
    if not new_prototypes:
        return current_assignments

    print(f"⚖️ Delta Check: Verifying {len(batch_files)} files against {len(new_prototypes)} NEW prototypes...")
    
    # We can reuse submit_assignment_tasks but we need to compare results
    # We treat new_prototypes as the only available prototypes for this run
    futures = submit_assignment_tasks(executor, batch_files, new_prototypes, new_proto_keys, folder, chunk_size)
    
    # We don't write to main benchmark CSV for this delta step to avoid skewing "main assignment" stats, 
    # or we can log it separately. For now, just collect.
    delta_assignments = collect_assignment_results(futures, benchmark_csv=None, desc="Delta checking")
    
    updates = 0
    for fname, (new_pid, new_dist) in delta_assignments.items():
        curr_pid, curr_dist = current_assignments.get(fname, (None, float("inf")))
        if new_dist < curr_dist:
            current_assignments[fname] = (new_pid, new_dist)
            updates += 1
            
    if updates:
        print(f"  ↳ Updated {updates} assignments to better match new prototypes")
    
    return current_assignments

# -------------------------------
# Main incremental clustering
# -------------------------------
def incremental_clustering(
    start_batch: int = None,
    batch_size: int = 400,
    folder: str = "rendered_pages_parallel",
    dthreshold: float = 0.25,
    seed: int = 42,
    csv_file: str = "batch_stats.csv",
    max_workers: int = 12,
    chunk_size: int = 20,
    unused_threshold: int = 10,
    max_prototypes_allowed: int = 2000
):
    # ── Automatically detect latest batch if start_batch is None ─────────
    if start_batch is None:
        batch_files = [f for f in os.listdir(".") if f.startswith("incremental_prototypes_batch") and f.endswith(".json")]
        if batch_files:
            batch_nums = [int(f.replace("incremental_prototypes_batch", "").replace(".json", "")) for f in batch_files]
            start_batch = max(batch_nums) + 1
            print(f"🔄 Auto-resume: Found batch {max(batch_nums)}, resuming from batch {start_batch}")
        else:
            start_batch = 1
            print("🆕 No previous batches found, starting from batch 1")

    # ── Load previous state if continuing ───────────────────────────────
    prototypes = {}
    proto_keys = {}
    full_assignments = defaultdict(list)
    proto_last_used = {}

    if start_batch > 1:
        last = start_batch - 1
        proto_file = f"incremental_prototypes_batch{last}.json"
        cluster_file = f"cluster_info_batch{last}.json"

        if os.path.exists(proto_file) and os.path.exists(cluster_file):
            with open(proto_file, encoding="utf-8") as f:
                proto_keys = {int(k): v for k, v in json.load(f).items()}
            with open(cluster_file, encoding="utf-8") as f:
                full_assignments = defaultdict(list, json.load(f))

            for pid, files in full_assignments.items():
                proto_last_used[pid] = last if files else last - unused_threshold

            print(f"→ Loaded previous state from batch {last} ({len(proto_keys)} prototypes)")

            # Load actual prototype content
            # Requires executor? We haven't created it yet. Load serially or simple IO threadpool.
            # Just use the simple parallel loader we refactored.
            # But we need to handle "missing file" gracefully. 
            print("Loading prototype contents...")
            loaded_files, loaded_contents = load_files_parallel(folder, list(proto_keys.values()), max_workers=max_workers)
            # Map back to PIDs requires care because load_files_parallel returns lists
            # Simpler: create a map of fname -> content
            content_map = dict(zip(loaded_files, loaded_contents))
            for pid, fname in proto_keys.items():
                if fname in content_map:
                    prototypes[pid] = content_map[fname]
                else:
                    print(f"  Missing prototype file: {fname}")
        else:
            print("→ Starting fresh - no previous state found")

    # ── Get all files & precompute ───────────────────────────────────────
    all_files = sorted(f for f in os.listdir(folder) if f.lower().endswith(".html"))
    print(f"Total HTML files found: {len(all_files)}")

    # We use a persistent executor for the main work loop to allow pipelining
    # PRECOMPUTE: We can use it too, or just let precompute use its own ThreadPool (IO bound? no lzma is cpu/gil-release)
    # The existing precompute used ThreadPool. Let's keep it simple and just run it. 
    # But wait, precompute might benefit from ProcessPool if we used valid multiprocessing? 
    # lzma releases GIL so ThreadPool is actually fine and less overhead than ProcessPool for simple compression.
    # Let's stick to ThreadPool for precompute as it was.
    precompute_compressed_sizes(folder, all_files, max_workers=max_workers)

    processed = set(f for files in full_assignments.values() for f in files)
    remaining = [f for f in all_files if f not in processed]
    print(f"Remaining unprocessed files: {len(remaining)}")
    batches = [remaining[i:i + batch_size] for i in range(0, len(remaining), batch_size)]

    if not os.path.exists(csv_file):
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "batch", "files", "outliers", "new_protos", "total_protos", "assign_sec", "next_batch_accuracy"
            ])

    # ──────────────────────────────────────────────────────────────────────────
    #  MAIN PIPELINE LOOP
    # ──────────────────────────────────────────────────────────────────────────
    
    # We maintain futures for the "Next Batch" to allow pipelining.
    next_batch_futures = None
    next_batch_files_ref = [] # keep track of which files correspond to next_batch_futures
    
    # Store "New Prototypes" from the PREVIOUS batch to perform delta-checks 
    # on the speculatively processed files.
    # Initial state: None
    prev_new_protos = {}
    prev_new_keys = {}

    current_batch = start_batch
    
    # Create persistent executor
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        
        for i, batch_files in enumerate(batches):
            if not batch_files:
                continue
                
            # Calculate current progress
            # Files processed so far (from previous batches)
            currently_processed = sum(len(files) for files in full_assignments.values())
            # Files that will be processed after this batch completes
            files_after_batch = currently_processed + len(batch_files)
            remaining_after_batch = len(all_files) - files_after_batch
            
            print(f"\n{'═' * 70}")
            print(f" BATCH {current_batch}  |  {len(batch_files)} files")
            print(f"{'═' * 70}")
            print(f"Progress: {currently_processed}/{len(all_files)} processed | After this batch: {files_after_batch}/{len(all_files)} | {remaining_after_batch} remaining")

            # 1. ACQUIRE FUTURES FOR CURRENT BATCH
            # ------------------------------------
            current_futures = None
            if next_batch_futures:
                print("⚡ Using speculatively scheduled tasks from previous batch...")
                current_futures = next_batch_futures
                next_batch_futures = None # consume them
            else:
                # First batch or no pipeline options
                print("Starting assignment phase...")
                current_futures = submit_assignment_tasks(executor, batch_files, prototypes, proto_keys, folder, chunk_size)

            # 2. SPECULATIVE SUBMISSION (Next Batch)
            # --------------------------------------
            # While workers are chewing on current_futures (or finishing them), 
            # we schedule the NEXT batch immediately if available.
            next_idx = i + 1
            if next_idx < len(batches):
                next_batch_files_ref = batches[next_idx]
                print(f"🚀 pipeline: Speculatively submitting Batch {current_batch + 1} ({len(next_batch_files_ref)} files)...")
                # Important: Uses CURRENT 'prototypes'. The delta will be fixed later.
                next_batch_futures = submit_assignment_tasks(executor, next_batch_files_ref, prototypes, proto_keys, folder, chunk_size)
            else:
                print("ℹ️ No next batch to pipeline (last batch).")
                next_batch_futures = None

            # 3. COLLECT RESULTS (Current Batch)
            # ----------------------------------
            start_assign = time.time() # We approximate start time or track it? 
            # The 'start_assign' metric is fuzzy in pipelining. 
            # We'll measure "wait time for results".
            assignments = collect_assignment_results(current_futures, csv_file, desc=f"Collecting Batch {current_batch}")
            assign_duration = time.time() - start_assign
            
            # 4. DELTA CORRECTION (If Speculative from Prev)
            # ----------------------------------------------
            # If current_futures came from speculation (i > 0 roughly), they missed 'prev_new_protos'.
            # We must check against them.
            if prev_new_protos:
                assignments = refine_assignments_delta(
                    assignments, batch_files, prev_new_protos, prev_new_keys,
                    folder, executor, dthreshold, chunk_size
                )
            
            # Reset prev news for this round
            prev_new_protos = {} 
            prev_new_keys = {}

            # 5. STATS & ACCURACY
            # -------------------
            accuracy = 0.0
            hits = 0
            if prototypes:
                hits = sum(1 for f, (pid, d) in assignments.items() if d <= dthreshold)
                accuracy = (hits / len(batch_files)) * 100
                print(f"📊 Accuracy: {accuracy:.2f}% ({hits}/{len(batch_files)})")
            else:
                print("📊 Accuracy: N/A (Initial batch)")

            print(f"→ Collection/Delta finished in {assign_duration:.1f}s")

            outliers = [f for f, (pid, d) in assignments.items() if d > dthreshold]
            matched = [f for f in batch_files if f not in outliers]

            # Update assignments & last used
            for f in matched:
                pid, _ = assignments[f]
                full_assignments[pid].append(f)
                proto_last_used[pid] = current_batch

            # 6. HANDLE OUTLIERS (FPF)
            # ------------------------
            # This step creates NEW prototypes.
            new_p, new_k, next_id = fpf_threshold_on_demand(
                outliers, folder, prototypes, proto_keys,
                dthreshold=dthreshold, seed=seed, 
                executor=executor, max_workers=max_workers
            )

            prototypes.update(new_p)
            proto_keys.update(new_k)
            for pid in new_p:
                proto_last_used[pid] = current_batch
            
            # Store for next batch's correction step
            prev_new_protos = new_p
            prev_new_keys = new_k

            # 6b. ASSIGN OUTLIERS TO NEW PROTOTYPES
            # -------------------------------------
            # Outliers were > threshold for ALL existing prototypes. 
            # We now assign these outliers to the closest NEW prototype.
            if new_p and outliers:
                 # Check against new_p only
                 outlier_futures = submit_assignment_tasks(executor, outliers, new_p, new_k, folder, chunk_size)
                 outlier_results = collect_assignment_results(outlier_futures, benchmark_csv=None, desc="Assigning outliers")
                 
                 reclaimed = 0
                 for f, (pid, d) in outlier_results.items():
                     if d <= dthreshold:
                         full_assignments[pid].append(f)
                         proto_last_used[pid] = current_batch
                         reclaimed += 1
                     else:
                         # Still an outlier? Should not happen if FPF worked correctly
                         pass
                 if reclaimed:
                     print(f"  ↳ Reclaimed {reclaimed}/{len(outliers)} outliers using new prototypes")

            # 7. PRUNING
            # ----------
            do_unused = assign_duration > 45 * 60
            prune_prototypes(
                prototypes, proto_keys, full_assignments, proto_last_used,
                current_batch=current_batch,
                do_unused_pruning=do_unused,
                unused_threshold=unused_threshold,
                max_prototypes=max_prototypes_allowed
            )

            # 8. SAVE
            # -------
            save_prototypes(prototypes, proto_keys, current_batch)
            save_cluster_info(full_assignments, current_batch)

            # Stats
            with open(csv_file, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    current_batch, len(batch_files), len(outliers),
                    len(new_p), len(prototypes), round(assign_duration, 1), round(accuracy, 2)
                ])

            print(f"→ Batch {current_batch} completed")
            print(f"  Outliers: {len(outliers):3d}  |  New protos: {len(new_p):3d}  |  Total protos: {len(prototypes):4d}")

            current_batch += 1

    # ── Final save caches ───────────────────────────────────────────────
    with open(COMPRESSED_SIZE_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(compressed_size_cache, f)
    with open(NCD_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(ncd_cache, f)
    print("\n✓ All caches saved")

    print("\nClustering finished. Final number of prototypes:", len(prototypes))

# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    incremental_clustering(
        start_batch=None,           # ← Set to None for auto-resume
        batch_size=400,
        folder="rendered_pages_parallel",
        dthreshold=0.25,
        seed=42,
        max_workers=12,          # adjust according to your CPU
        chunk_size=20,           # ← increased from previous versions
        unused_threshold=10,
        max_prototypes_allowed=2000
    )