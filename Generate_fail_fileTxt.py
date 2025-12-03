import os

input_dir = 'phishing_pages'
output_dir = 'rendered_pages_parallel'
failed_file_list = 'failed_files.txt'

input_files = set(f for f in os.listdir(input_dir) if f.endswith('.html'))
rendered_files = set(f for f in os.listdir(output_dir) if f.endswith('.html'))

# Files that were not rendered
failed_files = input_files - rendered_files

# Save to file
with open(failed_file_list, 'w') as f:
    for file_name in sorted(failed_files):
        f.write(file_name + '\n')

print(f"Identified {len(failed_files)} failed files. Saved to {failed_file_list}")
