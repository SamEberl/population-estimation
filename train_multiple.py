import subprocess

# Define the different file paths in a list
file_paths = [
    "fixMatch_1",
    "fixMatch_2",
    "fixMatch_3",
    "fixMatch_4",
    "fixMatch_5",
]

# Loop through the file paths and call train.py with each one
for path in file_paths:
    subprocess.run(["python", "train.py", "--file_path", f'configs/{path}.yaml'])
