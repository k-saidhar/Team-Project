
import os
import json
import glob

FOLDER = "rendered_pages_parallel"
CLUSTER_FILE_PATTERN = "cluster_info_batch*json"
REPORT_FILE = "status_report.txt"

def check_status():
    report = []
    
    if not os.path.exists(FOLDER):
        report.append(f"Error: {FOLDER} does not exist.")
        with open(REPORT_FILE, "w") as f:
            f.write("\n".join(report))
        return

    all_files = [f for f in os.listdir(FOLDER) if f.lower().endswith(".html")]
    total_files = len(all_files)
    report.append(f"Total HTML files in {FOLDER}: {total_files}")
    
    # 2. Find latest batch
    cluster_files = glob.glob(CLUSTER_FILE_PATTERN)
    if not cluster_files:
        report.append("No cluster info files found.")
    else:
        batch_nums = []
        for cf in cluster_files:
            try:
                bn = int(cf.replace("cluster_info_batch", "").replace(".json", ""))
                batch_nums.append(bn)
            except:
                pass
        
        if not batch_nums:
             report.append("No valid batch numbers found.")
        else:
            latest_batch = max(batch_nums)
            latest_file = f"cluster_info_batch{latest_batch}.json"
            report.append(f"Latest batch info: {latest_file}")
            
            try:
                with open(latest_file, "r", encoding="utf-8") as f:
                    assignments = json.load(f)
                
                processed_files = set()
                # Assignments is PID -> [list of files] OR just simple dict if format changed?
                # Based on previous code: full_assignments = defaultdict(list, json.load(f))
                # So it IS a dict of PID -> list of files.
                for pid, files in assignments.items():
                    processed_files.update(files)
                
                report.append(f"Total processed files (in batch {latest_batch} info): {len(processed_files)}")
                
                unprocessed = set(all_files) - processed_files
                report.append(f"Unprocessed files: {len(unprocessed)}")
                if len(unprocessed) > 0:
                    report.append(f"First 10 unprocessed: {list(unprocessed)[:10]}")
                
            except Exception as e:
                report.append(f"Error reading {latest_file}: {e}")

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
        
if __name__ == "__main__":
    check_status()
