import json
import os
import glob

# Find all cluster_info batch files
batch_files = sorted(glob.glob("cluster_info_batch*.json"), 
                     key=lambda x: int(x.replace("cluster_info_batch", "").replace(".json", "")))

print(f"Found {len(batch_files)} cluster_info batch files\n")
print("=" * 80)
print(f"{'Batch':<10} {'Files in Batch':<15} {'Cumulative Total':<20}")
print("=" * 80)

cumulative = 0
batch_details = []

for batch_file in batch_files:
    batch_num = int(batch_file.replace("cluster_info_batch", "").replace(".json", ""))
    
    try:
        with open(batch_file) as f:
            data = json.load(f)
        
        total_files = sum(len(files) for files in data.values())
        cumulative += total_files
        
        batch_details.append({
            'batch': batch_num,
            'files': total_files,
            'cumulative': cumulative
        })
        
        print(f"{batch_num:<10} {total_files:<15} {cumulative:<20}")
        
    except Exception as e:
        print(f"{batch_num:<10} ERROR: {e}")

print("=" * 80)
print(f"\nTotal files if we sum all batches: {cumulative}")
print(f"Latest batch file: {batch_files[-1]} has {batch_details[-1]['files']} files")
print(f"\nNote: If batches have overlapping files or if some batches overwrite others,")
print(f"the actual unique file count would be different.")

# Check for anomalies
print("\n" + "=" * 80)
print("ANOMALY CHECK:")
print("=" * 80)

# Batches with suspiciously few files
few_files = [b for b in batch_details if b['files'] < 10]
if few_files:
    print(f"\n⚠️  Batches with < 10 files (possible issue):")
    for b in few_files:
        print(f"   Batch {b['batch']}: {b['files']} files")

# Check if cumulative matches the last batch
if batch_details:
    last_batch_cumulative = batch_details[-1]['files']  # Last batch should contain ALL files
    sum_all = sum(b['files'] for b in batch_details)
    
    print(f"\n📊 Last batch contains: {last_batch_cumulative} files")
    print(f"📊 Sum of all batches: {sum_all} files")
