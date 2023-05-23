#!/bin/bash

# Activate Conda environment
source activate dlr_torch

# Run Python script
python - << END
from utils import *
open_tensorboard("/home/sam/Desktop/DLR/Data/Data_100GB/trained_models/")
END

