import os
import yaml
import argparse
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import aeDataset
from autoencoder.models import *
# from autoencoder.models.ae import Autoencoder_basic
from tqdm import tqdm

print('---Training model---')

parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config', '-c',
                    dest="filename",
                    metavar='FILE',
                    help='path to the config file',
                    default='configs/aeMNIST.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set to utilize tensor cores of GPU
torch.set_float32_matmul_precision('medium')

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the dataset
print('Load data')
# path_data = "/home/sam/Desktop/DLR/Data/Data_100GB/So2Sat_POP_Part1/train/"
# path_data = "/home/sam/Desktop/DLR/Data/Data_testing/AE/"
# train_dataset = CustomDataset(path_data, transform=transform)
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)

mnistTrainSet = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_dataloader = torch.utils.data.DataLoader(mnistTrainSet, batch_size=16,shuffle=True, num_workers=2)

# Initialize the model
model = vae_models[config['model_params']['name']](**config['model_params'])
model.to(device)

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=config['exp_params']['LR'])

# Train the model
print('Start training')
for epoch in tqdm(range(config['trainer_params']['max_epochs'])):
    for i, data in enumerate(train_dataloader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass
        model_outs = model(inputs)
        loss = model.loss_function(*model_outs, M_N=0.005)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print loss
        if i+1 % 50 == 0:
            print(f" Epoch [{epoch+1}/{config['trainer_params']['max_epochs']}], Step [{i+1}/{len(train_dataloader)}], Loss: {loss.item():.6f}")

# Save the model
#TODO: give each model unique name
torch.save(model.state_dict(), os.path.join(config['logging_params']['save_dir'], config['logging_params']['name']))
