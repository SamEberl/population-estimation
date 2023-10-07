#!/bin/bash

# Start the training process in the background
python train_student_teacher.py &

# Start TensorBoard in the background
tensorboard --logdir=/home/sameberl/logs/ --host=0.0.0.0 --port=6006