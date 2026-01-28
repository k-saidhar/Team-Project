import os
import json
import glob
import shutil
from collections import defaultdict

def retrieve_and_merge():
    print("🚀 Starting recovery/retrieve process...")

    # 1. Find all batch files
    cluster_files = sorted(glob.glob("cluster_info_batch*.json"))
    proto_files = sorted(glob.glob("incremental_prototypes_batch*.json"))
    
    # Extract batch numbers
    def get_batch_num(fname):
        try:
            return int(''.join(filter(str.isdigit, fname)))
        except:
            return -1

    # Filter to ensure we only get valid numbered batches
    cluster_files = [f for f in cluster_files if get_batch_num(f) > 0]
    proto_files = [f for f in proto_files if get_batch_num(f) > 0]
    
    # Sort by batch num
    cluster_files.sort(key=get_batch_num)
    proto_files.sort(key=get_batch_num)

    print(f"Found {len(cluster_files)} cluster info files and {len(proto_files)} prototype files.")

    # 2. Aggregation Structures
    # assignments: filename -> (prototype_id, batch_seen)
    # We keep the assignment from the LATEST batch if a file appears multiple times.
    assignments_map = {} 
    
    # prototypes: prototype_id -> key_filename
    prototypes_map = {}

    # 3. Process Prototypes (Incremental Prototypes)
    print("📦 Aggregating prototypes...")
    for pf in proto_files:
        batch_num = get_batch_num(pf)
        try:
            with open(pf, "r", encoding="utf-8") as f:
                data = json.load(f)
                # data is { pid: key_file }
                for pid, key_file in data.items():
                    # Just overwrite/update. Later batches might strictly add, but if they modify, we take latest.
                    prototypes_map[pid] = key_file
        except Exception as e:
            print(f"⚠️ Error reading {pf}: {e}")

    print(f"✓ Total unique prototypes recovered: {len(prototypes_map)}")

    # 4. Process Assignments (Cluster Info)
    print("📂 Aggregating assignments...")
    for cf in cluster_files:
        batch_num = get_batch_num(cf)
        try:
            with open(cf, "r", encoding="utf-8") as f:
                data = json.load(f)
                # data is { pid: [list of files] }
                for pid, files in data.items():
                    # We reverse map: file -> pid
                    for file in files:
                        assignments_map[file] = pid
        except Exception as e:
            print(f"⚠️ Error reading {cf}: {e}")

    print(f"✓ Total unique files recovered: {len(assignments_map)}")

    # 5. Reconstruct the Data Structures for Dump
    # We need to recreate the format expected by allforone.py
    # cluster_info: { pid: [files] }
    
    final_cluster_info = defaultdict(list)
    for fname, pid in assignments_map.items():
        # Ensure pid exists in prototypes? 
        # Ideally yes, but if we have an assignment to a PID we don't know, we should probably keep it (or warn).
        # But we aggregated all prototypes, so we should have it.
        final_cluster_info[pid].append(fname)
        
    # 6. Target the latest batch (86)
    # We will overwrite batch 86 files so allforone.py starts from 87 with FULL state.
    target_batch = 86
    
    target_cluster_file = f"cluster_info_batch{target_batch}.json"
    target_proto_file = f"incremental_prototypes_batch{target_batch}.json"
    
    # Backup existing
    if os.path.exists(target_cluster_file):
        shutil.copy(target_cluster_file, target_cluster_file + ".bak")
        print(f"Created backup: {target_cluster_file}.bak")
        
    if os.path.exists(target_proto_file):
        shutil.copy(target_proto_file, target_proto_file + ".bak")
        print(f"Created backup: {target_proto_file}.bak")

    # Write merged state
    print(f"💾 Saving merged state to {target_cluster_file}...")
    with open(target_cluster_file, "w", encoding="utf-8") as f:
        json.dump(final_cluster_info, f, indent=2)
        
    print(f"💾 Saving merged prototypes to {target_proto_file}...")
    with open(target_proto_file, "w", encoding="utf-8") as f:
        json.dump(prototypes_map, f, indent=2)
        
    # Also save recover_state.json as requested originally, just in case
    with open("recover_state.json", "w", encoding="utf-8") as f:
        json.dump({
            "total_files": len(assignments_map),
            "total_prototypes": len(prototypes_map),
            "last_batch": target_batch
        }, f, indent=2)

    print("✅ RETRIEVE COMPLETE.")
    print(f"Ready to resume allforone.py from Batch {target_batch + 1}")

if __name__ == "__main__":
    retrieve_and_merge()
