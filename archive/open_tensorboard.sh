#!/bin/bash

# Activate Conda environment
source activate dino

# Run Python script
python - << END
from utils import *
open_tensorboard("path")
END

