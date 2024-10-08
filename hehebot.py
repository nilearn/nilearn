import os
import re
from pathlib import Path


def replace_os_path_with_pathlib(file_path):
    # Read the contents of the Python file
    with open(file_path, 'r') as file:
        content = file.read()

    # Regex to match os.path.join cases
    pattern = r'os\.path\.join\((.*?)\)'

    # Function to transform os.path.join into Path style
    def replace_join(match):
        # Split the arguments passed to os.path.join
        args = match.group(1).split(',')
        # Remove any spaces around the arguments and convert them into Path style
        args = [arg.strip() for arg in args]
        return ' / '.join(args)

    # Replace occurrences in the file content
    new_content = re.sub(pattern, replace_join, content)

    # Replace 'import os' with 'from pathlib import Path'
    if 'import os' in new_content:
        new_content = new_content.replace('import os', 'from pathlib import Path', 1)

    # Write the updated content back to the file
    with open(file_path, 'w') as file:
        file.write(new_content)


def process_folder(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Only process Python files
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                replace_os_path_with_pathlib(file_path)


if __name__ == "__main__":
    # Specify the root folder containing all the Python files (e.g., 'nilearn')
    folder_to_process = "nilearn"
    process_folder(folder_to_process)
