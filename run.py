import requests
import csv
from io import StringIO
import os
import hashlib
import time
import schedule
from bs4 import BeautifulSoup

def fetch_phishing_urls():
    url = "http://data.phishtank.com/data/online-valid.csv"
    try:
        print("Downloading phishing URLs list...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        csv_file = StringIO(response.text)
        reader = csv.DictReader(csv_file)
        phishing_urls = [row['url'] for row in reader]
        print(f"Fetched {len(phishing_urls)} phishing URLs.")
        return phishing_urls
    except Exception as e:
        print(f"Failed to download phishing URLs: {e}")
        return []

# Download HTML content of phishing page
def download_phishing_page(url):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200 and "text/html" in response.headers.get("Content-Type", ""):
            soup = BeautifulSoup(response.text, 'html.parser')
            return soup.prettify()
    except Exception as e:
        print(f"Failed to download page {url}: {e}")
    return None

# Save HTML content to disk with filename as MD5 hash of URL
def save_html_content(url, html, base_dir="phishing_pages"):
    os.makedirs(base_dir, exist_ok=True)
    filename = hashlib.md5(url.encode()).hexdigest() + ".html"
    path = os.path.join(base_dir, filename)
    
    # Skip if file already exists
    if os.path.exists(path):
        print(f"Already downloaded: {url} -> {filename}")
        return False

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved phishing page: {url} -> {filename}")
    return True

# Main function to collect and save phishing pages
def collect_phishing_data():
    urls = fetch_phishing_urls()
    for url in urls:
        filename = hashlib.md5(url.encode()).hexdigest() + ".html"
        filepath = os.path.join("phishing_pages", filename)
        if os.path.exists(filepath):
            print(f"Skipping already downloaded URL: {url}")
            continue
        html = download_phishing_page(url)
        if html:
            save_html_content(url, html)

# Schedule the task every 3 hours
schedule.every(3).hours.do(collect_phishing_data)

'''if __name__ == "__main__":
    collect_phishing_data()  # run once immediately
    while True:
        schedule.run_pending()
        time.sleep(60)'''


'''
import os
import lzma
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

def compress_size(data: bytes) -> int:
    return len(lzma.compress(data))

def ncd(x: bytes, y: bytes) -> float:
    Cx = compress_size(x)
    Cy = compress_size(y)
    Cxy = compress_size(x + y)
    return (Cxy - min(Cx, Cy)) / max(Cx, Cy)

def load_html_files(folder="phishing_pages"):
    files = []
    contents = []
    for fname in os.listdir(folder):
        if fname.endswith(".html"):
            clean_fname = fname.strip()
            path = os.path.join(folder, clean_fname)
            print("Trying file:", repr(clean_fname))
            try:
                with open(path, "rb") as f:
                    content = f.read()
                files.append(clean_fname)
                contents.append(content)
                print("Success:", clean_fname)
            except Exception as e:
                print("❌ Failed:", clean_fname)
                print("   Full path:", repr(path))
                print("   Error:", e)
    return files, contents


def compute_distance_matrix(contents):
    n = len(contents)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = ncd(contents[i], contents[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    return dist_matrix

def cluster_phishing_pages(dist_matrix, threshold=0.3):
    condensed = squareform(dist_matrix)
    Z = linkage(condensed, method='average')
    clusters = fcluster(Z, t=threshold, criterion='distance')
    return clusters

def extract_prototypes(files, contents, clusters):
    prototypes = {}
    unique_clusters = set(clusters)
    for c in unique_clusters:
        indices = [i for i, cl in enumerate(clusters) if cl == c]
        if len(indices) == 1:
            prototypes[c] = files[indices[0]]
            continue
        min_avg_dist = float('inf')
        proto_index = -1
        for i in indices:
            dist_sum = 0
            for j in indices:
                if i != j:
                    dist_sum += ncd(contents[i], contents[j])
            avg_dist = dist_sum / (len(indices) - 1)
            if avg_dist < min_avg_dist:
                min_avg_dist = avg_dist
                proto_index = i
        prototypes[c] = files[proto_index]
    return prototypes

if __name__ == "__main__":
    files, contents = load_html_files("phishing_pages")
    print(f"Loaded {len(files)} phishing pages.")
    dist_matrix = compute_distance_matrix(contents)
    print("Computed NCD distance matrix.")
    clusters = cluster_phishing_pages(dist_matrix, threshold=0.3)
    print(f"Formed {len(set(clusters))} clusters.")
    prototypes = extract_prototypes(files, contents, clusters)
    print("Extracted prototypes for each cluster:")
    for cluster_id, proto_file in prototypes.items():
        print(f"Cluster {cluster_id}: Prototype file = {proto_file}")
'''
import os
import lzma
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import multiprocessing

# --- Global contents list (will be shared in each subprocess) ---
shared_contents = []

# --- Compression and NCD using LZMA ---

def compress_size(data: bytes) -> int:
    return len(lzma.compress(data))

def ncd(x: bytes, y: bytes) -> float:
    Cx = compress_size(x)
    Cy = compress_size(y)
    Cxy = compress_size(x + y)
    return (Cxy - min(Cx, Cy)) / max(Cx, Cy)

# --- File Loader ---

def load_html_files(folder="phishing_pages"):
    files = []
    contents = []
    count=0
    for fname in os.listdir(folder):
        if fname.endswith(".html"):
            clean_fname = fname.strip()
            path = os.path.join(folder, clean_fname)
            print("Trying file:", repr(clean_fname))
            count+=1
            try:
                with open(path, "rb") as f:
                    content = f.read()
                files.append(clean_fname)
                contents.append(content)
                print("Success:", clean_fname)
            except Exception as e:
                print("❌ Failed:", clean_fname)
                print("   Full path:", repr(path))
                print("   Error:", e)
        if count==200:
            return files,contents
    return files, contents

# --- Worker Init for Multiprocessing ---

def init_worker(contents):
    global shared_contents
    shared_contents = contents

def compute_pairwise_ncd_by_index(pair):
    i, j = pair
    return (i, j, ncd(shared_contents[i], shared_contents[j]))

def compute_distance_matrix_parallel(contents):
    n = len(contents)
    dist_matrix = np.zeros((n, n), dtype=np.float32)
    index_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    # Use spawn context for Windows compatibility
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(initializer=init_worker, initargs=(contents,)) as pool:
        for i, j, dist in pool.map(compute_pairwise_ncd_by_index, index_pairs):
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    return dist_matrix

# --- Clustering ---

def cluster_phishing_pages(dist_matrix, threshold=0.3):
    condensed = squareform(dist_matrix)
    Z = linkage(condensed, method='average')
    clusters = fcluster(Z, t=threshold, criterion='distance')
    return clusters

# --- Prototype Extraction ---

def extract_prototypes(files, contents, clusters):
    prototypes = {}
    unique_clusters = set(clusters)
    for c in unique_clusters:
        indices = [i for i, cl in enumerate(clusters) if cl == c]
        if len(indices) == 1:
            prototypes[c] = files[indices[0]]
            continue
        min_avg_dist = float('inf')
        proto_index = -1
        for i in indices:
            dist_sum = 0
            for j in indices:
                if i != j:
                    dist_sum += ncd(contents[i], contents[j])
            avg_dist = dist_sum / (len(indices) - 1)
            if avg_dist < min_avg_dist:
                min_avg_dist = avg_dist
                proto_index = i
        prototypes[c] = files[proto_index]
    return prototypes

# --- Main ---

def main():
    folder = "phishing_pages"
    files, contents = load_html_files(folder)
    print(f"Loaded {len(files)} phishing pages.")

    print("Computing NCD distance matrix using LZMA compression in parallel...")
    dist_matrix = compute_distance_matrix_parallel(contents)
    print("NCD matrix computation completed.")

    clusters = cluster_phishing_pages(dist_matrix, threshold=0.3)
    print(f"Formed {len(set(clusters))} clusters.")

    prototypes = extract_prototypes(files, contents, clusters)
    print("Extracted prototypes for each cluster:")
    for cluster_id, proto_file in prototypes.items():
        print(f"Cluster {cluster_id}: Prototype file = {proto_file}")

if __name__ == "__main__":
    main()

