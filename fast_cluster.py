<<<<<<< HEAD
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import json

def run_cpp_worker(file_list, proto_info, folder):
    file_str = ",".join(file_list)
    proto_str = ",".join(f"{fname}|{pid}" for pid, fname in proto_info)
    cmd = ["./ncd_worker", "assign", folder, file_str, proto_str, "0.3"]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Error:", result.stderr)
        return {}

    assignments = {}
    for line in result.stdout.strip().split("\n"):
        if not line: continue
        parts = line.split("|")
        if len(parts) != 3: continue
        fname, pid, dist = parts
        assignments[fname] = (int(pid), float(dist))
    return assignments

def assign_to_prototypes_fast(files, proto_keys, folder, chunk_size=60, workers=16):
    proto_info = [(fname, pid) for pid, fname in proto_keys.items()]
    chunks = [files[i:i+chunk_size] for i in range(0, len(files), chunk_size)]
    
    all_assignments = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_cpp_worker, chunk, proto_info, folder) for chunk in chunks]
        for future in tqdm(futures, desc="Blazing Fast NCD Assignment"):
            all_assignments.update(future.result())
    
    return all_assignments

# ———————— RUN THIS ————————
if __name__ == "__main__":
    # 1. Load your current prototypes
    with open("incremental_prototypes_batch7.json") as f:
        proto_keys = {int(k): v for k, v in json.load(f).items()}

    folder = "rendered_pages_parallel"

    # 2. Automatically find new files (not already prototypes)
    all_html = [f for f in os.listdir(folder) if f.endswith(".html")]
    existing = set(proto_keys.values())
    new_files = [f for f in all_html if f not in existing]

    print(f"Found {len(new_files)} new pages to classify using {len(proto_keys)} prototypes")

    # 3. Run ultra-fast assignment
    assignments = assign_to_prototypes_fast(
        new_files, proto_keys, folder,
        chunk_size=60,
        workers=os.cpu_count() or 12
    )

    # 4. Show results
    for fname, (cluster_id, ncd) in sorted(assignments.items(), key=lambda x: x[1][1]):
        print(f"{fname} → Cluster {cluster_id}  (NCD = {ncd:.4f})")
=======
import os
import subprocess
import json
import csv
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# -------------------------------
# C++ Worker Interface
# -------------------------------
def run_cpp_worker(file_list, proto_info, folder, threshold=0.3):
    """Call the C++ NCD worker for fast assignment"""
    file_str = ",".join(file_list)
    proto_str = ",".join(f"{fname}|{pid}" for pid, fname in proto_info)
    cmd = ["./ncd_worker", "assign", folder, file_str, proto_str, str(threshold)]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Error:", result.stderr)
        return {}

    assignments = {}
    for line in result.stdout.strip().split("\n"):
        if not line: continue
        parts = line.split("|")
        if len(parts) != 3: continue
        fname, pid, dist = parts
        assignments[fname] = (int(pid), float(dist))
    return assignments

def assign_to_prototypes_fast(files, proto_keys, folder, dthreshold=0.3, chunk_size=60, workers=16):
    """Fast parallel assignment using C++ worker"""
    proto_info = [(fname, pid) for pid, fname in proto_keys.items()]
    chunks = [files[i:i+chunk_size] for i in range(0, len(files), chunk_size)]
    
    all_assignments = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_cpp_worker, chunk, proto_info, folder, dthreshold) for chunk in chunks]
        for future in tqdm(futures, desc="⚡ Blazing Fast NCD Assignment"):
            all_assignments.update(future.result())
    
    return all_assignments

# -------------------------------
# Pruning: Track usage over 10 batches
# -------------------------------
def prune_prototypes(prototypes, proto_keys, full_assignments, proto_last_used,
                     current_batch, lookback_batches=10, max_prototypes=2000):
    """
    Prune prototypes based on:
    1. No files assigned in the last `lookback_batches` batches
    2. Limit total prototypes by utility (cluster_size / mean_NCD)
    
    This implements the pruning strategy from reiterate.py with a 10-batch lookback window.
    """
    # Option 1: Remove prototypes with no assignments in last 10 batches
    unused_protos = [pid for pid, last in proto_last_used.items() 
                     if last <= current_batch - lookback_batches]
    
    for pid in unused_protos:
        prototypes.pop(pid, None)
        proto_keys.pop(pid, None)
        proto_last_used.pop(pid, None)
        full_assignments.pop(str(pid), None)
        print(f"🗑️  Removed unused prototype {pid} (last used batch {proto_last_used.get(pid, 'N/A')}, current batch {current_batch})")

    # Option 2: Limit total prototypes by utility score
    if len(prototypes) > max_prototypes:
        utilities = {}
        for pid in prototypes.keys():
            files = full_assignments.get(str(pid), [])
            if not files:
                utilities[pid] = 0
            else:
                utilities[pid] = len(files)  # Simple utility: cluster size
        
        # Remove lowest utility prototypes
        sorted_protos = sorted(utilities.items(), key=lambda x: x[1])
        num_to_remove = len(prototypes) - max_prototypes
        for pid, _ in sorted_protos[:num_to_remove]:
            prototypes.pop(pid, None)
            proto_keys.pop(pid, None)
            proto_last_used.pop(pid, None)
            full_assignments.pop(str(pid), None)
            print(f"🗑️  Removed low-utility prototype {pid} to maintain budget")

# -------------------------------
# Save prototypes & cluster info
# -------------------------------
def save_prototypes(proto_keys, batch_num):
    filename = f"fast_cluster_prototypes_batch{batch_num}.json"
    with open(filename, "w") as f:
        json.dump({str(k): v for k, v in proto_keys.items()}, f, indent=2)
    print(f"✅ Saved prototypes → {filename}")

def save_cluster_info(full_assignments, batch_num):
    filename = f"fast_cluster_info_batch{batch_num}.json"
    with open(filename, "w") as f:
        json.dump(full_assignments, f, indent=2)
    print(f"✅ Saved cluster info → {filename}")

# -------------------------------
# Incremental clustering with pruning
# -------------------------------
def incremental_clustering_fast(batch_start=0, batch_size=400,
                                folder="rendered_pages_parallel",
                                dthreshold=0.3, 
                                csv_file="fast_cluster_stats.csv",
                                max_workers=16, chunk_size=60,
                                lookback_batches=10,
                                max_prototypes=2000):
    """
    Fast incremental clustering using C++ NCD worker with pruning.
    
    Args:
        batch_start: Starting batch number (0 for fresh start, >0 to continue)
        batch_size: Number of files per batch
        folder: Directory containing HTML files
        dthreshold: NCD distance threshold for outliers
        csv_file: CSV file to log batch statistics
        max_workers: Number of parallel workers
        chunk_size: Files per chunk for C++ worker
        lookback_batches: Prune prototypes unused for this many batches (default 10)
        max_prototypes: Maximum number of prototypes to maintain
    """
    
    # Initialize or load state
    if batch_start == 0:
        # Fresh start
        prototypes = {}
        proto_keys = {}
        full_assignments = defaultdict(list)
        proto_last_used = {}
        proto_id_counter = 0
        print("🆕 Starting fresh clustering")
    else:
        # Load previous state
        last_batch = batch_start - 1
        with open(f"fast_cluster_prototypes_batch{last_batch}.json") as f:
            proto_keys = {int(k): v for k, v in json.load(f).items()}
        with open(f"fast_cluster_info_batch{last_batch}.json") as f:
            full_assignments = defaultdict(list, json.load(f))
        
        # Initialize proto_last_used
        proto_last_used = {}
        for pid, files in full_assignments.items():
            pid = int(pid)
            if files:
                proto_last_used[pid] = last_batch
            else:
                proto_last_used[pid] = last_batch - lookback_batches
        
        # Load prototype content
        prototypes = {}
        for pid, fname in proto_keys.items():
            try:
                with open(f"{folder}/{fname}", "rb") as f:
                    prototypes[pid] = f.read()
            except:
                print(f"⚠️  Prototype file missing: {fname}")
        
        proto_id_counter = max(proto_keys.keys(), default=-1) + 1
        print(f"📂 Loaded state from batch {last_batch}")

    # Get all files and create batches
    all_files = sorted(f for f in os.listdir(folder) if f.endswith(".html"))
    processed_files = set()
    for files in full_assignments.values():
        processed_files.update(files)
    remaining_files = [f for f in all_files if f not in processed_files]
    batches = [remaining_files[i:i+batch_size] for i in range(0, len(remaining_files), batch_size)]

    # CSV logging
    if not os.path.exists(csv_file):
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["batch_num", "total_files", "outliers", "new_prototypes", 
                           "total_prototypes", "pruned_prototypes", "time_sec"])

    current_batch = batch_start

    for batch in batches:
        print(f"\n🚀 Processing batch {current_batch} with {len(batch)} files")
        start_time = time.time()

        # Assign files to existing prototypes
        assignments = assign_to_prototypes_fast(batch, proto_keys, folder,
                                               dthreshold=dthreshold,
                                               chunk_size=chunk_size,
                                               workers=max_workers)
        
        outliers = [f for f, (pid, dist) in assignments.items() if dist > dthreshold]
        matched_files = [f for f in batch if f not in outliers]

        # Update assignments and last used tracking
        for f in matched_files:
            pid, _ = assignments[f]
            full_assignments[str(pid)].append(f)
            proto_last_used[pid] = current_batch

        # Create new prototypes from outliers (simple: first outlier becomes prototype)
        new_protos_count = 0
        for outlier in outliers:
            # Make this outlier a new prototype
            prototypes[proto_id_counter] = open(f"{folder}/{outlier}", "rb").read()
            proto_keys[proto_id_counter] = outlier
            full_assignments[str(proto_id_counter)].append(outlier)
            proto_last_used[proto_id_counter] = current_batch
            proto_id_counter += 1
            new_protos_count += 1

        # Apply pruning
        initial_proto_count = len(prototypes)
        prune_prototypes(prototypes, proto_keys, full_assignments, proto_last_used,
                        current_batch=current_batch, 
                        lookback_batches=lookback_batches,
                        max_prototypes=max_prototypes)
        pruned_count = initial_proto_count - len(prototypes)

        # Save results
        save_prototypes(proto_keys, current_batch)
        save_cluster_info(full_assignments, current_batch)

        batch_time = time.time() - start_time

        # Log to CSV
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([current_batch, len(batch), len(outliers), new_protos_count,
                           len(prototypes), pruned_count, round(batch_time, 2)])

        print(f"✅ Batch {current_batch} done | Outliers: {len(outliers)} | "
              f"New protos: {new_protos_count} | Pruned: {pruned_count} | "
              f"Total protos: {len(prototypes)} | Time: {batch_time:.2f}s")

        current_batch += 1

    print(f"\n🎉 Clustering complete! Processed {len(batches)} batches")

# -------------------------------
# Entry point
# -------------------------------
if __name__ == "__main__":
    # Check if C++ worker is built
    if not os.path.exists("./ncd_worker") and not os.path.exists("./ncd_worker.exe"):
        print("❌ C++ worker not found!")
        print("Please run: bash build.sh")
        print("Or on Windows with WSL/MinGW: bash build.sh")
        exit(1)

    # Run incremental clustering
    incremental_clustering_fast(
        batch_start=0,  # Set to 0 for fresh start, or batch number to continue
        batch_size=400,
        folder="rendered_pages_parallel",
        dthreshold=0.3,
        csv_file="fast_cluster_stats.csv",
        max_workers=os.cpu_count() or 12,
        chunk_size=60,
        lookback_batches=10,  # Prune prototypes unused for 10 batches
        max_prototypes=2000
    )
>>>>>>> 45bc198a9 (updated)
