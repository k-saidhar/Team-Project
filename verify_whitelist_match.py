
import os
import random
import json
import hashlib
from bs4 import BeautifulSoup
from preprocess import strip_text_and_comments

def main():
    whitelist_file = "whitelist.json"
    if not os.path.exists(whitelist_file):
        print("whitelist.json missing")
        return
        
    with open(whitelist_file, 'r') as f:
        whitelist = json.load(f)
        
    print(f"Loaded whitelist with {len(whitelist)} domains.")
    
    legit_dir = "legit_pages"
    files = [f for f in os.listdir(legit_dir) if f.endswith(".html")]
    
    if not files:
        print("No files in legit_pages")
        return
        
    # Test 5 random files
    for i in range(5):
        fname = random.choice(files)
        domain = fname.replace(".html", "")
        path = os.path.join(legit_dir, fname)
        
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        # Preprocess
        soup = BeautifulSoup(content, "html.parser")
        strip_text_and_comments(soup)
        processed = str(soup)
        
        h = hashlib.md5(processed.encode('utf-8')).hexdigest()
        
        if domain in whitelist:
            if h in whitelist[domain]:
                print(f"[PASS] {domain} found in whitelist with correct hash.")
            else:
                print(f"[FAIL] {domain} found but HASH MISMATCH.")
                print(f"Computed: {h}")
                print(f"Stored: {whitelist[domain]}")
        else:
            print(f"[FAIL] {domain} NOT found in whitelist keys.")

if __name__ == "__main__":
    main()
