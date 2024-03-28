from utils import parse_yaml
from train import train_main


file_paths = [
    "configs/fixMatch_0_1.yaml",
    "configs/fixMatch_0_2.yaml",
    # "configs/fixMatch_0_3.yaml",
]

# file_paths = [
#     "configs/fixMatch_1.yaml",
    #"configs/fixMatch_2.yaml",
    #"configs/fixMatch_3.yaml",
    #"configs/fixMatch_4.yaml",
    #"configs/fixMatch_5.yaml",
# ]

configs = {}

for i, path in enumerate(file_paths):
    configs[i] = parse_yaml(path)

for config in configs.values():
    train_main(config)