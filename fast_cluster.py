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