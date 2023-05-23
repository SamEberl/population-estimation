import os
import yaml
import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import aeDataset
# from autoencoder.models.ae import Autoencoder_basic
from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils import *
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from models import *
from trainFuncs import *

print('---Training model---')

ae_config = parse_yaml('configs/aeResNet.yaml')
reg_config = parse_yaml('configs/regBasic.yaml')

log_dir = ae_config['logging_params']['save_dir']

# ae_model = train_ae(ae_config, log_dir)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ae_model = ae_models[ae_config['model_params']['name']](**ae_config['model_params']).to(device)
model_path = os.path.join(ae_config['logging_params']['save_dir'], ae_config['logging_params']['name'])
ae_model.load_state_dict(torch.load(model_path))

train_reg(reg_config, log_dir, ae_model)

# open_tensorboard(log_dir)


