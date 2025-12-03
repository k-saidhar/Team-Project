import os
import time
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
from tqdm import tqdm

INPUT_DIR = 'phishing_pages'
OUTPUT_DIR = 'rendered_pages_parallel'
FAILED_FILE_LIST = 'failed_files.txt'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def render_html_with_long_timeout(html_path, timeout_ms=20000, wait_seconds=3):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        
        try:
            # Use a longer timeout
            page.goto(f'file://{os.path.abspath(html_path)}', wait_until='domcontentloaded', timeout=timeout_ms)

            # Wait longer for JS to settle
            time.sleep(wait_seconds)

            # Get rendered HTML
            rendered_html = page.content()

        except PlaywrightTimeoutError:
            print(f"[Timeout] {html_path}")
            rendered_html = None
        except Exception as e:
            print(f"[Error] {html_path}: {e}")
            rendered_html = None
        finally:
            browser.close()

        return rendered_html

# Load the failed file list
with open(FAILED_FILE_LIST, 'r') as f:
    failed_files = [line.strip() for line in f if line.strip()]

for file_name in tqdm(failed_files, desc="Retrying failed files"):
    try:
        input_path = os.path.join(INPUT_DIR, file_name)
        output_path = os.path.join(OUTPUT_DIR, file_name)

        rendered_html = render_html_with_long_timeout(input_path)

        if rendered_html:
            with open(output_path, 'w', encoding='utf-8') as f_out:
                f_out.write(rendered_html)
        else:
            print(f"Still failed: {file_name}")

    except Exception as e:
        print(f"Retry failed for {file_name}: {e}")
