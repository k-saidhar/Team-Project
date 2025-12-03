import os
import time
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
from tqdm import tqdm

INPUT_DIR = 'phishing_pages'
OUTPUT_DIR = 'rendered_pages'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def render_html_with_browser(html_path):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        
        try:
            # Open local file with a safe timeout
            page.goto(f'file://{os.path.abspath(html_path)}', wait_until='domcontentloaded', timeout=10000)

            # Optional: wait a bit more for JS to finish (if needed)
            time.sleep(1.5)  # adjust if needed

            # Try to get page content
            rendered_html = page.content()

        except PlaywrightTimeoutError:
            print(f"Timeout while loading {html_path}")
            rendered_html = None
        except Exception as e:
            print(f"Unexpected error in {html_path}: {e}")
            rendered_html = None
        finally:
            browser.close()

        return rendered_html

# Process each file
html_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.html')]

for file_name in tqdm(html_files, desc="Rendering HTML files"):
    try:
        input_path = os.path.join(INPUT_DIR, file_name)
        output_path = os.path.join(OUTPUT_DIR, file_name)

        rendered_html = render_html_with_browser(input_path)

        if rendered_html:
            with open(output_path, 'w', encoding='utf-8') as f_out:
                f_out.write(rendered_html)
        else:
            print(f"Skipped writing {file_name} due to rendering failure")

    except Exception as e:
        print(f"Error rendering {file_name}: {e}")
