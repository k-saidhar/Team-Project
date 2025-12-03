'''import os

folder = "phishing_pages"

for fname in os.listdir(folder):
    if fname.endswith(".html"):
        full_path = os.path.join(folder, fname)
        print(f"Trying: {full_path}")
        try:
            with open(full_path, "rb") as f:
                content = f.read()
            print(f"Success: {fname}")
        except Exception as e:
            print(f"Failed: {fname} - {e}")
'''
import os
'''folder = "phishing_pages"

for fname in os.listdir(folder):
    if fname.endswith(".html"):
        print("Filename repr:", repr(fname))  # Check for hidden characters
        full_path = os.path.join(folder, fname)
        try:
            with open(full_path, "rb") as f:
                content = f.read()
            print(f"Success: {fname}")
        except Exception as e:
            print(f"Failed: {fname} - {e}")
'''
'''import os

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
    return files, contents'''
import os
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import normalize


# --- Load HTML files ---
def load_html_files(folder="phishing_pages", limit=None):
    files = []
    contents = []
    for i, fname in enumerate(os.listdir(folder)):
        if fname.endswith(".html"):
            try:
                with open(os.path.join(folder, fname), "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                files.append(fname)
                contents.append(content)
            except Exception as e:
                print(f"Skipping {fname}: {e}")
        if limit and len(files) >= limit:
            break
    return files, contents


# --- Compute TF-IDF Cosine Distance Matrix ---
def compute_tfidf_cosine_distance_matrix(contents):
    vectorizer = TfidfVectorizer(input="content", stop_words="english", max_features=10000)
    tfidf_matrix = vectorizer.fit_transform(contents)
    tfidf_matrix = normalize(tfidf_matrix)
    distance_matrix = cosine_distances(tfidf_matrix)
    return distance_matrix


# --- Clustering ---
def cluster_documents(distance_matrix, threshold=0.3):
    condensed = distance_matrix[np.triu_indices(len(distance_matrix), k=1)]
    Z = linkage(condensed, method='average')
    clusters = fcluster(Z, t=threshold, criterion='distance')
    return clusters


# --- Extract Prototypes ---
def extract_prototypes(files, contents, clusters, distance_matrix):
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
            avg_dist = np.mean([distance_matrix[i][j] for j in indices if j != i])
            if avg_dist < min_avg_dist:
                min_avg_dist = avg_dist
                proto_index = i
        prototypes[c] = files[proto_index]
    return prototypes


# --- Main Function ---
def main(folder="phishing_pages", limit=None, threshold=0.3):
    files, contents = load_html_files(folder, limit=limit)
    print(f"Loaded {len(files)} files.")

    print("Computing TF-IDF + cosine similarity distance matrix...")
    distance_matrix = compute_tfidf_cosine_distance_matrix(contents)

    print("Clustering documents...")
    clusters = cluster_documents(distance_matrix, threshold=threshold)
    print(f"Identified {len(set(clusters))} clusters.")

    print("Extracting prototypes...")
    prototypes = extract_prototypes(files, contents, clusters, distance_matrix)

    print("Prototype summary:")
    for cid, proto in prototypes.items():
        print(f"Cluster {cid}: {proto}")
    with open("tfidf_clusters.json", "w") as f:
        json.dump({"files": files, "clusters": clusters.tolist()}, f)
    with open("tfidf_prototypes.json", "w") as f:
        json.dump({int(k): v for k, v in prototypes.items()}, f)



if __name__ == "__main__":
    main(limit=11000)  # Adjust the limit as needed

