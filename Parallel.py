import os
import time
from multiprocessing import Process, current_process, cpu_count
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
from tqdm import tqdm

INPUT_DIR = 'phishing_pages'
OUTPUT_DIR = 'rendered_pages_parallel'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def render_html_files(file_list, process_id, timeout_ms=10000, wait_seconds=1.5):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        for file_name in tqdm(file_list, desc=f"Process {process_id}", position=process_id):
            input_path = os.path.join(INPUT_DIR, file_name)
            output_path = os.path.join(OUTPUT_DIR, file_name)

            try:
                page.goto(f'file://{os.path.abspath(input_path)}', wait_until='domcontentloaded', timeout=timeout_ms)
                time.sleep(wait_seconds)
                rendered_html = page.content()

                with open(output_path, 'w', encoding='utf-8') as f_out:
                    f_out.write(rendered_html)

            except PlaywrightTimeoutError:
                print(f"[Timeout] {file_name}")
            except Exception as e:
                print(f"[Error] {file_name}: {e}")

        browser.close()

def split_list(full_list, num_chunks):
    avg = len(full_list) // num_chunks
    return [full_list[i * avg: (i + 1) * avg] for i in range(num_chunks - 1)] + [full_list[(num_chunks - 1) * avg:]]

def main(num_processes=4):
    html_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith('.html')])
    file_chunks = split_list(html_files, num_processes)

    processes = []
    for i in range(num_processes):
        p = Process(target=render_html_files, args=(file_chunks[i], i))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

if __name__ == '__main__':
    main(num_processes=4)  # Change this depending on your CPU cores
