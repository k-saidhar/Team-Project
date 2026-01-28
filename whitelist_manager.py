import os
import hashlib
import json
from bs4 import BeautifulSoup
from preprocess import is_empty_or_error, strip_text_and_comments

class WhitelistManager:
    def __init__(self, whitelist_file="whitelist.json"):
        self.whitelist_file = whitelist_file
        self.whitelist = {} # format: {domain: {html_hash, ...}}
        self.load_whitelist()

    def load_whitelist(self):
        if os.path.exists(self.whitelist_file):
            try:
                with open(self.whitelist_file, 'r') as f:
                    self.whitelist = json.load(f)
                print(f"Loaded whitelist with {len(self.whitelist)} domains.")
            except Exception as e:
                print(f"Failed to load whitelist: {e}")
                self.whitelist = {}

    def save_whitelist(self):
        try:
            with open(self.whitelist_file, 'w') as f:
                json.dump(self.whitelist, f)
            print("Whitelist saved.")
        except Exception as e:
            print(f"Failed to save whitelist: {e}")

    def compute_hash(self, html_content):
        """Computes hash of the processed HTML content."""
        return hashlib.md5(html_content.encode('utf-8')).hexdigest()

    def add_to_whitelist(self, domain, html_content):
        """Adds a domain and its HTML structure hash to the whitelist."""
        h = self.compute_hash(html_content)
        if domain not in self.whitelist:
            self.whitelist[domain] = []
        
        if h not in self.whitelist[domain]:
            self.whitelist[domain].append(h)

    def is_safe(self, domain, html_content):
        """
        Checks if the domain and content match a known legitimate site.
        """
        if domain not in self.whitelist:
            return False
        
        h = self.compute_hash(html_content)
        return h in self.whitelist[domain]

    def build_from_directory(self, directory):
        """
        Compiles legitimate HTML files from a directory into the whitelist.
        Filename format expected: domain.html
        """
        print(f"Building whitelist from {directory}...")
        for fname in os.listdir(directory):
            if fname.endswith(".html"):
                domain = fname.replace(".html", "")
                path = os.path.join(directory, fname)
                try:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # Preprocess content to match detection pipeline
                    if is_empty_or_error(content):
                        continue
                        
                    soup = BeautifulSoup(content, "html.parser")
                    strip_text_and_comments(soup)
                    processed_content = str(soup)
                    
                    self.add_to_whitelist(domain, processed_content)
                except Exception as e:
                    print(f"Error processing {fname}: {e}")
        self.save_whitelist()
