<<<<<<< HEAD
import os
import shutil
from multiprocessing import Process, cpu_count
from bs4 import BeautifulSoup, Comment

# Folder with rendered HTML files
DIR = "rendered_pages_parallel"

# Backup folder for removed files
REMOVED_DIR = os.path.join(DIR, "removed_pages")
os.makedirs(REMOVED_DIR, exist_ok=True)


# --------------------------------------------------------
# 1. Safer Empty/Error Page Detection
# --------------------------------------------------------
def is_empty_or_error(html_text: str) -> bool:
    # Only remove fully empty pages
    return len(html_text.strip()) == 0


# --------------------------------------------------------
# 2. Remove Text + Comments (PhishSim structural-only)
# --------------------------------------------------------
def strip_text_and_comments(soup):
    for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
        c.extract()
    for t in soup.find_all(string=True):
        t.extract()


# --------------------------------------------------------
# 3. Worker Function (each process handles a chunk)
# --------------------------------------------------------
def process_files(file_list, worker_id):
    for file_name in file_list:
        file_path = os.path.join(DIR, file_name)

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                html = f.read()
        except:
            continue

        # Empty/error removal (backup instead of deleting)
        if is_empty_or_error(html):
            backup_path = os.path.join(REMOVED_DIR, file_name)
            shutil.move(file_path, backup_path)
            print(f"[Worker {worker_id}] MOVED (empty): {file_name}")
            continue

        # Structural cleaning
        soup = BeautifulSoup(html, "html.parser")
        strip_text_and_comments(soup)
        cleaned = str(soup)

        # Overwrite file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(cleaned)

        print(f"[Worker {worker_id}] CLEANED: {file_name}")


# --------------------------------------------------------
# 4. Split list into chunks
# --------------------------------------------------------
def split_list(lst, num_chunks):
    avg = len(lst) // num_chunks
    return [lst[i * avg:(i + 1) * avg] for i in range(num_chunks - 1)] + [lst[(num_chunks - 1) * avg:]]


# --------------------------------------------------------
# MAIN
# --------------------------------------------------------
def main(num_workers=None):
    if num_workers is None:
        num_workers = max(2, cpu_count() - 1)  # use CPU efficiently

    files = [f for f in os.listdir(DIR) if f.endswith(".html")]
    file_chunks = split_list(files, num_workers)

    processes = []
    for i in range(num_workers):
        p = Process(target=process_files, args=(file_chunks[i], i))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print("\n✔ DONE — In-place preprocessing complete. Empty files moved to removed_pages/.")


if __name__ == "__main__":
    main(num_workers=8)   # Adjust number of workers depending on CPU cores
=======
import os
import shutil
from multiprocessing import Process, cpu_count
from bs4 import BeautifulSoup, Comment

# Folder with rendered HTML files defaults
DEFAULT_DIR = "legit_pages"

# --------------------------------------------------------
# 1. Safer Empty/Error Page Detection
# --------------------------------------------------------
def is_empty_or_error(html_text: str) -> bool:
    # Only remove fully empty pages
    if len(html_text.strip()) == 0:
        return True
    
    # Check for common error signatures
    error_keywords = [
        "404 Not Found", "403 Forbidden", "Access Denied", 
        "Bad Gateway", "Service Unavailable", "Domain For Sale",
        "This domain is pending"
    ]
    
    # A simple check: if the text is very short and contains an error keyword
    # We strip tags first to check text content is safer but here we check raw string for speed
    # as error pages usually have these in title or main headers.
    
    # Heuristic: if content is small (<500 bytes) and has error keyword
    if len(html_text) < 1000:
        for keyword in error_keywords:
            if keyword.lower() in html_text.lower():
                return True
                
    return False


# --------------------------------------------------------
# 2. Remove Text + Comments (PhishSim structural-only)
# --------------------------------------------------------
def strip_text_and_comments(soup):
    for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
        c.extract()
    for t in soup.find_all(string=True):
        t.extract()


# --------------------------------------------------------
# 3. Worker Function (each process handles a chunk)
# --------------------------------------------------------
def process_files(file_list, base_dir, worker_id):
    removed_dir = os.path.join(base_dir, "removed_pages")
    os.makedirs(removed_dir, exist_ok=True) # Ensure it exists

    for file_name in file_list:
        file_path = os.path.join(base_dir, file_name)

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                html = f.read()
        except:
            continue

        # Empty/error removal (backup instead of deleting)
        if is_empty_or_error(html):
            backup_path = os.path.join(removed_dir, file_name)
            try:
                shutil.move(file_path, backup_path)
                print(f"[Worker {worker_id}] MOVED (empty): {file_name}")
            except Exception as e:
                print(f"[Worker {worker_id}] Failed to move {file_name}: {e}")
            continue

        # Structural cleaning
        soup = BeautifulSoup(html, "html.parser")
        strip_text_and_comments(soup)
        cleaned = str(soup)

        # Overwrite file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(cleaned)

        print(f"[Worker {worker_id}] CLEANED: {file_name}")


# --------------------------------------------------------
# 4. Split list into chunks
# --------------------------------------------------------
def split_list(lst, num_chunks):
    if num_chunks <= 0: return []
    avg = len(lst) // num_chunks
    return [lst[i * avg:(i + 1) * avg] for i in range(num_chunks - 1)] + [lst[(num_chunks - 1) * avg:]]


# --------------------------------------------------------
# MAIN
# --------------------------------------------------------
def main(target_dir=None, num_workers=None):
    if num_workers is None:
        num_workers = max(2, cpu_count() - 1)  # use CPU efficiently
    
    if target_dir is None:
        import sys
        if len(sys.argv) > 1:
            target_dir = sys.argv[1]
        else:
            target_dir = DEFAULT_DIR
            
    if not os.path.exists(target_dir):
        print(f"Directory {target_dir} does not exist.")
        return

    print(f"Processing files in: {target_dir}")
    
    # Create backup dir early
    removed_dir = os.path.join(target_dir, "removed_pages")
    os.makedirs(removed_dir, exist_ok=True)

    files = [f for f in os.listdir(target_dir) if f.endswith(".html")]
    if not files:
        print("No HTML files found.")
        return
        
    file_chunks = split_list(files, num_workers)

    processes = []
    for i in range(num_workers):
        if i < len(file_chunks):
            p = Process(target=process_files, args=(file_chunks[i], target_dir, i))
            processes.append(p)
            p.start()

    for p in processes:
        p.join()

    print(f"\n✔ DONE — In-place preprocessing complete for {target_dir}. Empty files moved to removed_pages/.")

    print(f"\n✔ DONE — In-place preprocessing complete for {target_dir}. Empty files moved to removed_pages/.")


if __name__ == "__main__":
    main(num_workers=8)   # Adjust number of workers depending on CPU cores
>>>>>>> 45bc198a9 (updated)
