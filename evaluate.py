import os
import torch
from dataset import aeDataset
import torchvision
import yaml
import argparse
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import CustomDataset
from autoencoder.models import *
import matplotlib.pyplot as plt


def eval_model():
    # Define the path to the saved model
    model_path = '/home/sam/Desktop/DLR/Data/Data_100GB/test_conv_model.pth'

    # instantiate the model and load saved state dictionary
    model = Autoencoder()
    model.load_state_dict(torch.load(model_path))

    # Set the model to evaluation mode
    model.eval()

    data = "/home/sam/Desktop/DLR/Data/Data_100GB/So2Sat_POP_Part1/train/"
    # Define the test dataset
    test_dataset = CustomDataset(data_dir=data,
                                 transform=transforms.ToTensor())

    # Define the test dataloader
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=custom_collate)

    # Define the loss function (MSE)
    criterion = nn.MSELoss()

    # Initialize the running loss
    running_loss = 0.0

    # Loop over the test dataset
    for data in test_loader:
        # Get the inputs and labels
        inputs, labels = data

        # Pass the inputs through the model
        outputs = model(inputs)

        # Compute the loss between the inputs and outputs
        loss = criterion(outputs, inputs)

        # Add the batch loss to the running loss
        running_loss += loss.item() * inputs.size(0)

    # Calculate the average loss
    test_loss = running_loss / len(test_loader.dataset)

    print('Test Loss: {:.6f}'.format(test_loss))

import numpy as np
def display_input_output(model, input_tensor):
    def show_images():
        output_np = model(input_tensor)[0].detach().numpy()
        array_questionable = output_np[0, 0, :, :]
        array_comparison = torch.full((100, 100), 0.00217692).numpy()
        diff = np.where(array_questionable != array_comparison)
        print(diff)
        for c in range(n_channels):
            # Show input image
            axes[c, 0].imshow(input_np[idx][c], cmap='gray', vmin=0, vmax=1)
            axes[c, 0].set_title(f'In: {c} | mean: {input_tensor[idx][c].mean().item():.4f}')
            axes[c, 0].axis('off')

            # Show output image
            axes[c, 1].imshow(output_np[idx][c], cmap='gray', vmin=0, vmax=1)
            print(f'shape: {output_np[idx][c].shape} | mean: {output_np[idx][c].mean():.4f} | type: {type(output_np[idx][c])}')
            axes[c, 1].set_title(f'Out: {c} | mean: {output_np[idx][c].mean().item():.4f}')
            axes[c, 1].axis('off')

        plt.tight_layout()
        plt.show(block=True)

    def on_key_press(event):
        nonlocal idx
        if event.key == 'right':
            idx = (idx + 1) % input_tensor.shape[0]
            show_images()
        if event.key == 'left':
            idx = (idx - 1) % input_tensor.shape[0]
            show_images()
        print('-----')
        # print(str(input_tensor[idx][0].mean()) + '    ' + str(output_tensor[idx][0].mean()))
        # print(str(input_tensor[idx][1].mean()) + '    ' + str(output_tensor[idx][1].mean()))
        # print(str(input_tensor[idx][2].mean()) + '    ' + str(output_tensor[idx][2].mean()))
        # print(str(input_tensor[idx][3].mean()) + '    ' + str(output_tensor[idx][3].mean()))
        # print(str(input_tensor[idx].mean()) + '    ' + str(output_tensor[idx].mean()))
        #
        # print(str(output_tensor[idx][0]) + '    ' + str(output_tensor[idx][3]))



    # output_tensor = model(input_tensor)[0]
    input_np = input_tensor.detach().cpu().numpy()
    # output_np = output_tensor.detach().cpu().numpy()
    idx = 0
    n_channels = input_np[idx].shape[0]
    fig, axes = plt.subplots(nrows=n_channels, ncols=2, figsize=(7, 10))
    fig.canvas.mpl_connect('key_press_event', on_key_press)

    show_images()


parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config', '-c',
                    dest="filename",
                    metavar='FILE',
                    help='path to the config file',
                    default='configs/aeResNet.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)


# data = aeDataset(**config["data_params"])
data = aeDataset("/home/sam/Desktop/DLR/Data/Data_testing/AE", 64, 64, 100, 4)
dataloader = data.val_dataloader()

# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])
# mnistTrainSet = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# train_dataloader = torch.utils.data.DataLoader(mnistTrainSet, batch_size=16, shuffle=True, num_workers=2)


model = ae_models[config['model_params']['name']](**config['model_params'])
print('loading model')
model_path = os.path.join(config['logging_params']['save_dir'], config['logging_params']['name'])
model.load_state_dict(torch.load(model_path))
print('done')
model.eval()

input_tensor = next(iter(dataloader))[0]
print(input_tensor.shape)
display_input_output(model, input_tensor)
