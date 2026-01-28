import csv

with open('batch_stats.csv') as f:
    rows = list(csv.DictReader(f))

print("Batch | Files | Outliers | Matched | New Protos")
print("-" * 60)

total_files = 0
total_outliers = 0
total_matched = 0

for r in rows[-20:]:  # Last 20 batches
    batch = r['batch']
    files = int(r['files'])
    outliers = int(r['outliers'])
    new_protos = r['new_protos']
    matched = files - outliers
    
    total_files += files
    total_outliers += outliers
    total_matched += matched
    
    print(f"{batch:5} | {files:5} | {outliers:8} | {matched:7} | {new_protos:10}")

print("-" * 60)
print(f"TOTALS: {total_files} files, {total_outliers} outliers, {total_matched} matched")
print(f"\nIf outliers aren't reclaimed, {total_outliers} files would be LOST!")
