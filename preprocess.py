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
