from utils import parse_yaml
from train import train_main


# Define the different file paths in a list
file_paths = [
    "fixMatch_1",
    "fixMatch_2",
    "fixMatch_3",
    "fixMatch_4",
    "fixMatch_5",
]

configs = {}

for i, path in enumerate(file_paths):
    configs[i] = parse_yaml(path)

for config in configs.values():
    train_main(config)