import json
import os

for i in range(76, 81):
    fname = f"cluster_info_batch{i}.json"
    if os.path.exists(fname):
        with open(fname, 'r') as f:
            data = json.load(f)
        total = sum(len(files) for files in data.values())
        print(f"Batch {i}: {total} files across {len(data)} prototypes")
    else:
        print(f"Batch {i}: file not found")
