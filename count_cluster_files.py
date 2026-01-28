import json

with open('cluster_info_batch86.json') as f:
    data = json.load(f)

num_clusters = len(data)
total_files = sum(len(files) for files in data.values())

print(f"Number of prototypes (clusters): {num_clusters}")
print(f"Total files across all clusters: {total_files}")
print(f"\nFirst 5 clusters:")
for i, (pid, files) in enumerate(list(data.items())[:5]):
    print(f"  Prototype {pid}: {len(files)} files")
