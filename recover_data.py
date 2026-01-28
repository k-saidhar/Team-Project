import json
import glob
import os
from collections import defaultdict

def recover_data():
    print("🔍 Scanning for lost data across all history files...")
    
    # 1. Find all history files
    cluster_files = sorted(glob.glob("cluster_info_batch*.json"), 
                          key=lambda x: int(x.replace("cluster_info_batch", "").replace(".json", "")))
    proto_files = sorted(glob.glob("incremental_prototypes_batch*.json"), 
                        key=lambda x: int(x.replace("incremental_prototypes_batch", "").replace(".json", "")))

    if not cluster_files:
        print("❌ No history files found!")
        return

    print(f"📂 Found {len(cluster_files)} batch files to analyze")
    
    # Master dictionaries to hold recovered data
    # format: {filename: prototype_id}
    best_assignments = {} 
    # format: {prototype_id: prototype_filename}
    all_prototypes = {}

    # 2. First pass: Collect all known prototypes from all batches
    print("\n📦 collecting all prototypes ever created...")
    for pfile in proto_files:
        try:
            with open(pfile, 'r', encoding='utf-8') as f:
                batch_protos = json.load(f)
                for pid, filename in batch_protos.items():
                    all_prototypes[int(pid)] = filename
        except Exception as e:
            print(f"⚠️ Error reading {pfile}: {e}")

    print(f"✅ Discovered {len(all_prototypes)} unique clusters across history")

    # 3. Second pass: Recover file assignments
    print("\n🧬 Recovering file assignments...")
    
    for cfile in cluster_files:
        try:
            with open(cfile, 'r', encoding='utf-8') as f:
                batch_assignments = json.load(f)
                
                for pid_str, files in batch_assignments.items():
                    pid = int(pid_str)
                    # Store assignments - later batches overwrite earlier ones which is what we want
                    for file in files:
                        best_assignments[file] = pid

        except Exception as e:
            print(f"⚠️ Error reading {cfile}: {e}")

    # 4. Reconstruct the logical state
    recovered_cluster_info = defaultdict(list)
    recovered_prototypes = {}
    
    orphaned_files = 0
    successfully_recovered = 0

    for filename, pid in best_assignments.items():
        if pid in all_prototypes:
            recovered_cluster_info[pid].append(filename)
            recovered_prototypes[pid] = all_prototypes[pid]
            successfully_recovered += 1
        else:
            orphaned_files += 1

    print("\n" + "="*50)
    print("RECOVERY REPORT")
    print("="*50)
    print(f"Total processed files found in history: {len(best_assignments)}")
    print(f"Files successfully recovered:         {successfully_recovered}")
    print(f"Prototypes recovered:                 {len(recovered_prototypes)}")
    print(f"Orphaned files (dropped):             {orphaned_files}")
    
    # 5. Create the Recovery Batch (Batch 88)
    recovery_batch = 88
    
    # Save Prototypes
    proto_out = f"incremental_prototypes_batch{recovery_batch}.json"
    with open(proto_out, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in recovered_prototypes.items()}, f, indent=2)
    
    # Save Cluster Info
    cluster_out = f"cluster_info_batch{recovery_batch}.json"
    with open(cluster_out, "w", encoding="utf-8") as f:
        # Convert defaultdict to dict for JSON serialization
        json_ready = {str(k): v for k, v in recovered_cluster_info.items()}
        json.dump(json_ready, f, indent=2)
        
    print(f"\n✅ SUCCESSFULLY RESTORED state to Batch {recovery_batch}")
    print(f"   Created: {proto_out}")
    print(f"   Created: {cluster_out}")
    print("\n🚀 You can now run 'python allforone.py' and it will auto-resume from this recovered state!")

if __name__ == "__main__":
    recover_data()
