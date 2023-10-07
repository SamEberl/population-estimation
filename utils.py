import subprocess
import webbrowser
import argparse
import torchvision.transforms
import yaml
import os
import torch

import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import pandas as pd

from tqdm import tqdm
from ssl_dataset import sslDataset


def display_input_output(model, input_tensor):
    def show_images():
        output_np = model(input_tensor)[0].detach().numpy()
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

    input_np = input_tensor.detach().cpu().numpy()
    idx = 0
    n_channels = input_np[idx].shape[0]
    fig, axes = plt.subplots(nrows=n_channels, ncols=2, figsize=(7, 10))
    fig.canvas.mpl_connect('key_press_event', on_key_press)

    show_images()


def open_tensorboard(log_dir):
    # Start the TensorBoard server
    tensorboard_process = subprocess.Popen(['tensorboard', '--logdir', log_dir])

    # Open the browser window
    webbrowser.open_new_tab('http://localhost:6006')

    # Wait for the TensorBoard server to stop
    tensorboard_process.wait()


def parse_yaml(config_path='configs/aeBasic.yaml'):
    parser = argparse.ArgumentParser(description='Generic runner for VAE models')
    parser.add_argument('--config', '-c',
                        dest="filename",
                        metavar='FILE',
                        help='path to the config file',
                        default=config_path)

    args = parser.parse_args()
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    return config


def filter_images_by_mean(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.tif'):
                file_list.append(os.path.join(root, file))

    counter_removed = 0
    for image_path in tqdm(file_list, desc="Filtering Images"):
        image = tiff.imread(image_path)
        image_mean = np.mean(image, axis=None)

        if np.all(image_mean >= 0.009):
            pass
            # tqdm.write(f"Image '{image_path}' has mean >= 0.01. Keeping it.")
        else:
            # tqdm.write(f"Image '{image_path}' has mean < 0.01. Removing it.")
            os.remove(image_path)
            counter_removed += 1
    print(f'number of images removed: {counter_removed}')


def browse_images_with_mean(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.tif') or file.lower().endswith('.tiff'):
                file_list.append(os.path.join(root, file))

    for image_path in file_list:
        image = imageio.imread(image_path)
        print(f'image_mean: {np.mean(image, axis=None)}')
        n_channels = image.shape[2]

        fig, axes = plt.subplots(n_channels, 2, figsize=(8, 8))

        for c in range(n_channels):
            channel = image[:, :, c]
            channel_mean = np.mean(channel)

            # Show input channel
            axes[c, 0].imshow(channel, cmap='gray')
            axes[c, 0].set_title(f'Input channel {c + 1} | Mean: {channel_mean:.4f}')
            axes[c, 0].axis('off')

        plt.tight_layout()
        plt.show()


def create_feature_csv(reg_config, ssl_model):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set to utilize tensor cores of GPU
    torch.set_float32_matmul_precision('medium')

    data = sslDataset(**reg_config["data_params"])
    #train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()


    df_train = pd.DataFrame(columns=range(reg_config['model_params']['in_size']))
    df_train['PopCount'] = None

    df_val = pd.DataFrame(columns=range(reg_config['model_params']['in_size']))
    df_val['PopCount'] = None

    for param in ssl_model.parameters():
        param.requires_grad = False

    with torch.no_grad():
        for i, data in enumerate(val_dataloader):
            inputs, labels, name = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            ssl_features = ssl_model(inputs)
            df_train_tmp = pd.DataFrame(columns=range(reg_config['model_params']['in_size']))
            df_train_tmp['PopCount'] = None
            for j, feature in tqdm(enumerate(ssl_features)):
                feature = feature.tolist()
                feature.append(labels[j].item())
                df_feature = pd.DataFrame([feature], columns=df_train.columns, index=[(i+1)*(j+1)])
                df_train_tmp = pd.concat([df_train_tmp, df_feature], ignore_index=False)
            df_train = pd.concat([df_train, df_train_tmp], ignore_index=False)

    df_train.to_csv('/home/sam/Desktop/val_features_sen2spring_full.csv', index=False)


def save_data_as_jpg(reg_config, save_dir):
    # Set to utilize tensor cores of GPU
    torch.set_float32_matmul_precision('medium')

    data = sslDataset(**reg_config["data_params"])
    dataloader = data.train_dataloader()
    # dataloader = data.val_dataloader()


    df_train = pd.DataFrame(columns=range(reg_config['model_params']['in_size']))
    df_train['PopCount'] = None

    df_val = pd.DataFrame(columns=range(reg_config['model_params']['in_size']))
    df_val['PopCount'] = None

    with torch.no_grad():
        for data in tqdm(dataloader):
            inputs, labels, names = data

            for i, input in enumerate(inputs):
                image = torchvision.transforms.ToPILImage()(input)
                image.save(os.path.join(save_dir, names[i] + '.jpg'), quality=100)


def get_img_stats():
    reg_config = parse_yaml('configs/regBasicDINOv2.yaml')

    data = sslDataset(**reg_config["data_params"])
    train_dataloader = data.train_dataloader()

    channel_sums = np.zeros(3)
    channel_sums_squared = np.zeros(3)
    total_samples = 0

    for batch in train_dataloader:
        # Convert batch tensor to numpy array
        batch_np = batch[0].numpy()
        batch_size = batch_np.shape[0]

        # Sum pixel values for each channel
        channel_sums += np.sum(batch_np, axis=(0, 2, 3))
        channel_sums_squared += np.sum(batch_np ** 2, axis=(0, 2, 3))
        total_samples += batch_size

    # Calculate mean and standard deviation for each channel
    channel_means = channel_sums / (total_samples * 9604)
    channel_stddevs = np.sqrt((channel_sums_squared / (total_samples * 98 * 98)) - (channel_means ** 2))

    print("Mean of each channel (RGB):")
    print(channel_means)

    print("\nStandard deviation of each channel (RGB):")
    print(channel_stddevs)


def count_decimal_places(number):
    # Convert the float to a string
    number_str = str(number)

    # Check if there is a decimal point in the string
    if '.' in number_str:
        # Split the string into integer and decimal parts
        integer_part, decimal_part = number_str.split('.')

        # Count the number of characters in the decimal part
        return len(decimal_part)
    else:
        # If there is no decimal point, return 0
        return 0

