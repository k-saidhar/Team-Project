import os
import requests
import zipfile
import io
import csv
import json
import time
import gzip
from concurrent.futures import ThreadPoolExecutor

# Configuration
TRANCO_URL = "https://tranco-list.eu/top-1m.csv.zip"
TARGET_DOMAINS_COUNT = 4000
DATA_DIR = "legit_pages"
CC_INDEX_SERVER = "http://index.commoncrawl.org/CC-MAIN-2024-51-index" # Use a recent index
# Note: You might need to update the CC Index URL to the latest one found at https://commoncrawl.org/the-data/get-started/

os.makedirs(DATA_DIR, exist_ok=True)

def download_tranco_list():
    """Download and extract the top domains from Tranco."""
    print(f"Downloading Tranco list from {TRANCO_URL}...")
    try:
        response = requests.get(TRANCO_URL, timeout=60)
        response.raise_for_status()
        
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # The zip usually contains a single CSV file like 'top-1m.csv'
            csv_filename = z.namelist()[0]
            with z.open(csv_filename) as f:
                content = f.read().decode('utf-8')
                
        lines = content.strip().split('\n')
        domains = []
        for line in lines[:TARGET_DOMAINS_COUNT]:
            parts = line.split(',')
            if len(parts) >= 2:
                domains.append(parts[1].strip())
        
        print(f"Extracted {len(domains)} domains.")
        return domains
    except Exception as e:
        print(f"Failed to download/parse Tranco list: {e}")
        return []

def fetch_from_common_crawl(domain):
    """
    Try to fetch the latest HTML for a domain from Common Crawl.
    Returns HTML string or None.
    """
    print(f"Searching Common Crawl for: {domain}")
    try:
        # 1. Search existing index
        params = {
            'url': domain,
            'output': 'json',
            'limit': 1,
            'matchType': 'domain', 
            'filter': 'status:200' # Only successful captures
        }
        r = requests.get(CC_INDEX_SERVER, params=params, timeout=15)
        if r.status_code != 200:
            return None
        
        # Parse JSON line (sometimes multiple lines, we took limit=1)
        data = r.text.strip().split('\n')[0]
        if not data:
            return None
            
        record = json.loads(data)
        
        # 2. Fetch the content from the WARC file
        # Common Crawl uses AWS public datasets. URL format:
        # https://data.commoncrawl.org/{filename}
        # We need to use Range header to get just the record
        
        offset = int(record['offset'])
        length = int(record['length'])
        filename = record['filename']
        
        s3_url = f"https://data.commoncrawl.org/{filename}"
        headers = {'Range': f'bytes={offset}-{offset + length - 1}'}
        
        warc_resp = requests.get(s3_url, headers=headers, timeout=30)
        if warc_resp.status_code == 206: # Partial Content
            # WARC record is usually gzipped
            # But the 'length' in index might be the compressed length.
            # Usually Common Crawl stores data gzipped.
            
            # Simple approach: content is usually the HTTP response (header + body)
            # We need to decompress if it's gzipped, then strip headers.
            
            raw_data = warc_resp.content
            
            # Try decompressing (Common Crawl WARC files are concatenations of gzip records)
            try:
                content = gzip.decompress(raw_data)
            except:
                content = raw_data # Maybe not gzipped?
                
            # Content is a WARC record: info + http headers + html
            # We need to split to find the HTML.
            
            # Very naive WARC parsing:
            # Look for double newline which separates headers from body
            # The WARC record has WARC headers, then HTTP headers, then body.
            
            text = content.decode('utf-8', errors='ignore')
            
            # Find the start of the HTML (<!DOCTYPE or <html)
            # Or just split by \r\n\r\n twice?
            
            parts = text.split('\r\n\r\n')
            if len(parts) >= 2:
                # The last part is likely the body, or the largest part.
                return parts[-1] 
            return text
            
    except Exception as e:
        print(f"Error fetching {domain} from CC: {e}")
        
    return None

def fetch_live(domain):
    """Fallback: Fetch live from the web."""
    url = f"http://{domain}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.text
    except:
        try:
            url = f"https://{domain}"
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                return r.text
        except:
            pass
    return None

def process_domain(domain):
    filename = os.path.join(DATA_DIR, f"{domain}.html")
    if os.path.exists(filename):
        return
        
    # Priority 1: Common Crawl
    html = fetch_from_common_crawl(domain)
    
    # Priority 2: Live Fetch (Fallback)
    if not html:
        print(f"Common Crawl failed for {domain}, trying live...")
        html = fetch_live(domain)
        
    if html:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"Saved {domain}")
    else:
        print(f"Failed to collect {domain}")

def main():
    domains = download_tranco_list()
    if not domains:
        print("No domains found.")
        return

    # Process in parallel
    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(process_domain, domains)

if __name__ == "__main__":
    main()
