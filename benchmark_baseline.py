import os
import random
from detection import PhishSimDetector

LEGIT_DIR = "legit_pages"
PHISHING_DIR = "phishing_pages"
RATIO_LEGIT = 140
RATIO_PHISHING = 1

def load_files(directory, limit=None):
    files = []
    if not os.path.exists(directory):
        return []
    for f in os.listdir(directory):
        if f.endswith(".html"):
            path = os.path.join(directory, f)
            files.append(path)
    if limit:
        return files[:limit]
    return files

def main():
    print("Initializing Detector...")
    detector = PhishSimDetector()
    
    # Load available files
    legit_files = load_files(LEGIT_DIR)
    phishing_files = load_files(PHISHING_DIR)
    
    if not legit_files:
        print(f"No legitimate files found in {LEGIT_DIR}. Please run collect_legit_data.py first.")
        # Create dummy file for testing correctness if needed, but better to fail.
        return

    # Calculate counts based on ratio
    # If we have N legit files, we should use N/140 phishing files?
    # Or if we have M phishing files, we need M*140 legit files.
    # Usually we are limited by the smaller set relative to the ratio.
    # We want to maximize test size.
    
    n_phishing = len(phishing_files)
    n_legit = len(legit_files)
    
    # Target: 1 phishing : 140 legit
    # If we determine based on phishing count:
    needed_legit = n_phishing * 140
    
    if needed_legit > n_legit:
        # We are limited by legits
        used_legit = n_legit
        used_phishing = max(1, n_legit // 140)
        print(f"Limited by legitimate files. Using {used_legit} legit and {used_phishing} phishing.")
    else:
        # Limited by phishing (or plenty of both)
        used_phishing = n_phishing
        used_legit = used_phishing * 140
        print(f"Using {used_legit} legit and {used_phishing} phishing.")
        
    test_set = []
    
    # Select files
    selected_legit = random.sample(legit_files, used_legit)
    selected_phishing = random.sample(phishing_files, used_phishing)
    
    for p in selected_legit:
        test_set.append((p, False)) # Path, IsPhishing
    for p in selected_phishing:
        test_set.append((p, True))
        
    random.shuffle(test_set)
    
    print(f"Starting Benchmark on {len(test_set)} files...")
    
    fp = 0
    tp = 0
    fn = 0
    tn = 0
    
    for path, is_phishing_actual in test_set:
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except:
            continue
            
        is_phishing_pred, reason, details = detector.detect(path, content)
        
        if is_phishing_actual:
            if is_phishing_pred:
                tp += 1
            else:
                fn += 1
                # Trigger incremental learning here?
                # detector.feedback(path, content, True)
        else:
            if is_phishing_pred:
                fp += 1
                print(f"False Positive: {path}. Reason: {reason}, Details: {details}")
            else:
                tn += 1
                
    print("\n--- Results ---")
    print(f"Total: {len(test_set)}")
    print(f"True Positives (Phishing caught): {tp}")
    print(f"False Negatives (Phishing missed): {fn}")
    print(f"True Negatives (Legit safe): {tn}")
    print(f"False Positives (Legit flagged): {fp}")
    
    if (tp + fn) > 0:
        tpr = tp / (tp + fn)
        print(f"TPR (Recall): {tpr:.2%}")
    if (fp + tn) > 0:
        fpr = fp / (fp + tn)
        print(f"FPR: {fpr:.2%}")
        
    print("Benchmark Complete.")

if __name__ == "__main__":
    main()
