import subprocess

# Define the different file paths in a list
file_paths = [
    "fixMatch_l2_0p1",
    "fixMatch_l2_1",
    "fixMatch_drop_1",
    "fixMatch_drop_2",
    "fixMatch_drop_3",
    "fixMatch_drop_4",
    "fixMatch_drop_5",
]

# Loop through the file paths and call train.py with each one
for path in file_paths:
    subprocess.run(["python", "train.py", "--file_path", str(path)])
