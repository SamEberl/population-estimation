from utils import parse_yaml
from train import train_main

path = "configs/fixMatch.yaml"
config = parse_yaml(path)
train_main(config)
