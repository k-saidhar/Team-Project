import os
import json
import requests
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# --- Config ---
PROTOTYPE_PATH = "tfidf_prototypes.json"          # Cluster ID -> filename
PHISHING_FOLDER = "phishing_pages"          # Folder with original files
SIMILARITY_THRESHOLD = 0.7                  # Tune as needed

# --- Load prototypes ---
def load_prototypes(prototype_file=PROTOTYPE_PATH, folder=PHISHING_FOLDER):
    with open(prototype_file, "r") as f:
        prototype_map = json.load(f)

    cluster_ids = list(prototype_map.keys())
    files = [prototype_map[cid] for cid in cluster_ids]
    htmls = []
    for fname in files:
        path = os.path.join(folder, fname)
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                htmls.append(f.read())
        except Exception as e:
            print(f"❌ Error reading {fname}: {e}")
            htmls.append("")  # Keep index alignment

    return cluster_ids, htmls

# --- Fetch HTML from URL ---
def fetch_html(url):
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        return r.text
    except Exception as e:
        print(f"❌ Failed to fetch URL: {e}")
        return None

# --- Classify using cosine similarity to prototypes ---
def classify_url(url, prototypes_html, cluster_ids, threshold=SIMILARITY_THRESHOLD):
    new_html = fetch_html(url)
    if not new_html:
        return "error", 0.0

    # TF-IDF vectorization
    tfidf = TfidfVectorizer(max_features=10000, stop_words="english")
    tfidf_matrix = tfidf.fit_transform(prototypes_html + [new_html])
    tfidf_matrix = normalize(tfidf_matrix)

    new_vec = tfidf_matrix[-1]
    proto_vecs = tfidf_matrix[:-1]

    sims = cosine_similarity(new_vec, proto_vecs)[0]
    best_score = np.max(sims)
    best_cluster = cluster_ids[np.argmax(sims)]

    is_phishing = best_score >= threshold
    return ("phishing", best_score, best_cluster) if is_phishing else ("legitimate", best_score, None)

# --- Example usage ---
if __name__ == "__main__":
    cluster_ids, prototypes_html = load_prototypes()
    test_url = input("Enter URL to classify: ").strip()
    label, score, cluster = classify_url(test_url, prototypes_html, cluster_ids)

    if label == "error":
        print("❌ Could not classify the URL.")
    elif label == "phishing":
        print(f"⚠️ Phishing detected! Similarity = {score:.3f}, matched with cluster {cluster}")
    else:
        print(f"✅ Legitimate URL. Similarity = {score:.3f}")
