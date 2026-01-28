# Fast Cluster - Build & Run Instructions

## Overview
`fast_cluster.py` uses a C++ worker (`ncd_worker.cpp`) for ultra-fast NCD (Normalized Compression Distance) computation. The workflow is:

1. **Build** the C++ worker using `build.sh`
2. **Run** the Python script which calls the C++ worker

---

## Prerequisites

### Linux/WSL (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install -y g++ liblzma-dev libxxhash-dev build-essential
```

### macOS
```bash
brew install xz xxhash
```

### Fedora/RHEL
```bash
sudo dnf install -y gcc-c++ xz-devel xxhash-devel
```

### Windows
You have **two options**:

#### Option 1: WSL (Recommended)
1. Install WSL: `wsl --install`
2. Inside WSL, follow Linux instructions above

#### Option 2: MinGW/MSYS2
1. Install MSYS2 from https://www.msys2.org/
2. Open MSYS2 terminal:
   ```bash
   pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-xz mingw-w64-x86_64-xxhash
   ```

---

## Build Instructions

### Step 1: Build the C++ Worker

```bash
# Make build.sh executable (Linux/macOS/WSL)
chmod +x build.sh

# Run the build script
bash build.sh
```

**Expected output:**
```
Building ultra-fast NCD worker...
Build successful! → ./ncd_worker
```

This creates an executable: `ncd_worker` (Linux/Mac) or `ncd_worker.exe` (Windows)

### Step 2: Verify the Build

```bash
# Check if the executable exists
ls -lh ncd_worker

# Test run (should show usage/error since no args provided)
./ncd_worker
```

---

## Run Instructions

### Fresh Start (Batch 0)

```bash
python fast_cluster.py
```

This will:
- Start from batch 0
- Process all HTML files in `rendered_pages_parallel/`
- Create prototypes and cluster assignments
- Save results to:
  - `fast_cluster_prototypes_batch0.json`
  - `fast_cluster_info_batch0.json`
  - `fast_cluster_stats.csv`

### Continue from Previous Batch

Edit `fast_cluster.py` line 253:
```python
incremental_clustering_fast(
    batch_start=5,  # Change this to continue from batch 5
    ...
)
```

Then run:
```bash
python fast_cluster.py
```

---

## Key Features Implemented

### 1. **Pruning Strategy** (from `reiterate.py`)
- **10-batch lookback**: Prototypes unused for 10 batches are removed
- **Utility-based pruning**: Limits total prototypes to 2000 by removing low-utility clusters

### 2. **Fast C++ Worker**
- Memory-mapped file I/O
- LZMA compression caching
- MinHash pre-filtering
- Multi-threaded processing

### 3. **Incremental Processing**
- Batch-based processing (default 400 files/batch)
- State persistence between runs
- CSV logging for analysis

---

## Configuration Parameters

Edit these in `fast_cluster.py` (line 253):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_start` | 0 | Starting batch (0 = fresh, >0 = continue) |
| `batch_size` | 400 | Files per batch |
| `folder` | `rendered_pages_parallel` | HTML files directory |
| `dthreshold` | 0.3 | NCD threshold for outliers |
| `max_workers` | CPU count | Parallel workers |
| `chunk_size` | 60 | Files per C++ worker call |
| `lookback_batches` | 10 | Prune if unused for X batches |
| `max_prototypes` | 2000 | Maximum prototypes to keep |

---

## Output Files

### Per Batch:
- `fast_cluster_prototypes_batch{N}.json` - Prototype ID → filename mapping
- `fast_cluster_info_batch{N}.json` - Cluster assignments

### Aggregate:
- `fast_cluster_stats.csv` - Batch statistics (outliers, pruning, time)

---

## Troubleshooting

### Build Errors

**Error: `lzma.h: No such file or directory`**
```bash
# Install LZMA development headers
sudo apt install liblzma-dev  # Ubuntu/Debian
brew install xz               # macOS
```

**Error: `xxhash.h: No such file or directory`**
```bash
# Install xxHash development headers
sudo apt install libxxhash-dev  # Ubuntu/Debian
brew install xxhash             # macOS
```

### Runtime Errors

**Error: `C++ worker not found!`**
```bash
# Rebuild the worker
bash build.sh
```

**Error: `FileNotFoundError: rendered_pages_parallel`**
```bash
# Create the directory or update the folder parameter
mkdir -p rendered_pages_parallel
```

**Error: `No such file or directory: 'fast_cluster_prototypes_batch7.json'`**
- You're trying to continue from a batch that doesn't exist
- Set `batch_start=0` to start fresh

---

## Performance Tips

1. **Increase workers**: Set `max_workers` to your CPU core count
2. **Adjust chunk size**: Larger chunks = fewer subprocess calls (try 100-200)
3. **Use SSD**: NCD computation is I/O intensive
4. **Monitor memory**: Large batches may require more RAM

---

## Comparison: `reiterate.py` vs `fast_cluster.py`

| Feature | `reiterate.py` | `fast_cluster.py` |
|---------|----------------|-------------------|
| NCD Computation | Python (slow) | C++ (fast) |
| Parallelization | ProcessPoolExecutor | C++ worker + Python orchestration |
| Pruning | 4-batch lookback | **10-batch lookback** |
| FPF for outliers | Yes (complex) | Simplified (first outlier = prototype) |
| Speed | ~10-20 files/sec | ~100-500 files/sec |

---

## Example Workflow

```bash
# 1. Build C++ worker
bash build.sh

# 2. Run first batch
python fast_cluster.py

# 3. Check results
cat fast_cluster_stats.csv

# 4. Continue from batch 1 (edit fast_cluster.py first)
python fast_cluster.py

# 5. Analyze cluster distribution
python -c "
import json
with open('fast_cluster_info_batch0.json') as f:
    clusters = json.load(f)
    for cid, files in clusters.items():
        print(f'Cluster {cid}: {len(files)} files')
"
```

---

## Questions?

- **How does pruning work?** See `prune_prototypes()` function (line 51-78)
- **How to change threshold?** Edit `dthreshold` parameter (line 256)
- **How to add FPF?** Integrate `fpf_threshold_on_demand()` from `reiterate.py`
