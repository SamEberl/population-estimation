import subprocess
import webbrowser
import argparse
import torchvision.transforms
import yaml
import os
import torch
import random
import pickle

import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import pandas as pd

from tqdm import tqdm
from ssl_dataset import sslDataset

from torchvision import transforms, datasets
from dataset import studentTeacherDataset
from pathlib import Path

from models import *
from dataset import get_dataloader, get_transforms


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


def create_feature_csv(config, model):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    student_model = ssl_models[config['model_params']['architecture']](**config['model_params']).to(device)
    teacher_model = ssl_models[config['model_params']['architecture']](**config['model_params']).to(device)
    for param in student_model.parameters():
        param.requires_grad = False
    for param in teacher_model.parameters():
        param.requires_grad = False

    student_transform, teacher_transform = get_transforms(config)
    train_dataloader, val_dataloader = get_dataloader(config, student_transform, teacher_transform)

    df_train = pd.DataFrame(columns=range(config['model_params']['in_size']))
    df_train['PopCount'] = None

    df_val = pd.DataFrame(columns=range(config['model_params']['in_size']))
    df_val['PopCount'] = None

    with torch.no_grad():
        for loader in [train_dataloader, val_dataloader]:
            for i, data in enumerate(loader):
                student_data, teacher_data, label, datapoint_name = data
                teacher_data = teacher_data.to(device)
                label = label.to(device)

                teacher_features = teacher_model.model(teacher_data)
                df_train_tmp = pd.DataFrame(columns=range(config['model_params']['in_size']))
                df_train_tmp['PopCount'] = None
                for j, feature in tqdm(enumerate(teacher_features)):
                    feature = feature.tolist()
                    feature.append(label[j].item())
                    df_feature = pd.DataFrame([feature], columns=df_train.columns, index=[(i+1)*(j+1)])
                    df_train_tmp = pd.concat([df_train_tmp, df_feature], ignore_index=False)
                df_train = pd.concat([df_train, df_train_tmp], ignore_index=False)

                df_val_tmp = pd.DataFrame(columns=range(config['model_params']['in_size']))
                df_val_tmp['PopCount'] = None
                for j, feature in tqdm(enumerate(teacher_features)):
                    feature = feature.tolist()
                    feature.append(label[j].item())
                    df_feature = pd.DataFrame([feature], columns=df_val.columns, index=[(i+1)*(j+1)])
                    df_val_tmp = pd.concat([df_val_tmp, df_feature], ignore_index=False)
                df_val = pd.concat([df_val, df_val_tmp], ignore_index=False)

    df_train.to_csv('/home/sam/Desktop/so2sat_test/train_features_23-12-18.csv', index=False)
    df_train.to_csv('/home/sam/Desktop/val_features_23-12-18.csv', index=False)


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
        # Find the position of the decimal point
        decimal_index = number_str.index('.')
        return len(number_str) - decimal_index - 1
    elif 'e-' in number_str:
        # Split the string at 'e'
        parts = number_str.split('-')
        return int(parts[1])
    else:
        # If there is no decimal point, return 0
        return 0


def derangement_shuffle(tensor):
    """
    Shuffle a tensor such that no element remains in its original position.

    :param tensor: A PyTorch tensor to be shuffled.
    :return: A shuffled tensor.
    """
    n = tensor.size(0)
    indices = list(range(n))
    while any(i == indices[i] for i in range(n)):
        random.shuffle(indices)

    return tensor[torch.tensor(indices)]


def calc_stats_dataset():
    data_path = "/home/sam/Desktop/so2sat_test/So2Sat_POP_Part1"
    # data_path = "/home/sam/Desktop/so2sat_test/tmp_train"
    nbr_channels = 20

    dataset = studentTeacherDataset(data_path, split='train', use_teacher=False, drop_labels=False, student_transform=None, teacher_transform=None, nbr_channels=nbr_channels)
    # dataset = studentTeacherDataset(data_path, split='test', use_teacher=False, drop_labels=False, student_transform=None, teacher_transform=None, nbr_channels=nbr_channels)

    # Initialize variables
    n_samples = 0
    channel_sum = torch.tensor([0.0] * nbr_channels)
    channel_sum_squared = torch.tensor([0.0] * nbr_channels)
    channel_min = torch.tensor([float('inf')] * nbr_channels)
    channel_max = torch.tensor([float('-inf')] * nbr_channels)

    value_counts = [{} for _ in range(nbr_channels)]

    # Loop through the dataset
    for data, _, _, _ in tqdm(dataset):
        # Convert numpy array to PyTorch tensor if necessary
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)

        # Assuming data is a CxHxW tensor
        C, H, W = data.shape
        n_samples += H * W  # HxW
        channel_sum += torch.sum(data, dim=[1, 2])
        # channel_sum += data.sum(dim=[1, 2])  # Sum over HxW for each channel
        channel_sum_squared += (data ** 2).sum(dim=[1, 2])  # Sum of squares

        # Update min and max
        channel_min = torch.min(channel_min, data.view(data.size(0), -1).min(dim=1).values)
        channel_max = torch.max(channel_max, data.view(data.size(0), -1).max(dim=1).values)

        for c in range(C):
            for value in data[c].view(-1).tolist():
                value_counts[c][value] = value_counts[c].get(value, 0) + 1

    # Compute mean and std
    mean = channel_sum / n_samples
    std = (channel_sum_squared / n_samples - mean ** 2).sqrt()

    torch.set_printoptions(precision=3, sci_mode=False)
    # Print statistics
    print(f'shape:\n{mean.shape}')
    print(f"Mean: \n{mean}")
    print(f"Standard Deviation: \n{std}")
    print(f"Minimum: \n{channel_min}")
    print(f"Maximum: \n{channel_max}")

    # Calculate percentiles
    percentiles = [0.001, 0.01, 0.25, 0.5, 0.75, 0.99, 0.999]
    percentile_values = torch.zeros((nbr_channels, len(percentiles)))

    for c in tqdm(range(nbr_channels)):
        sorted_values = sorted(value_counts[c].items())  # Sort by value
        total_counts = sum(value_counts[c].values())
        cumulative_count = 0
        percentile_indices = [int(p * total_counts) for p in percentiles]
        current_index = 0

        for value, count in sorted_values:
            cumulative_count += count
            while current_index < len(percentiles) and cumulative_count >= percentile_indices[current_index]:
                percentile_values[c, current_index] = value
                current_index += 1

    print("Approximate Percentiles:\n", percentile_values)


    # Directory to save histograms
    histogram_dir = Path("/home/sam/Desktop/so2sat_test/histograms")
    # After value_counts has been computed
    value_counts_file = histogram_dir/'value_counts.pkl'

    # Check if the directory exists, if not create it
    if not os.path.exists(histogram_dir):
        os.makedirs(histogram_dir)

    # Save value_counts to a Pickle file
    with open(value_counts_file, 'wb') as f:
        pickle.dump(value_counts, f)

    # with open(value_counts_file, 'rb') as f:
    #     value_counts = pickle.load(f)

    # Create and save histograms
    for c in range(nbr_channels):
        values = list(value_counts[c].keys())
        counts = list(value_counts[c].values())

        # Determine bin width based on the range of values
        min_value, max_value = min(values), max(values)
        range_of_values = max_value - min_value
        bin_width = range_of_values / 50  # for example, create 50 bins

        plt.figure()
        plt.hist(values, bins=np.arange(min_value, max_value + bin_width, bin_width), weights=counts)
        plt.title(f'Histogram for Channel {c}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.savefig(f'{histogram_dir}/channel_{c}_histogram_custom.png')
        plt.close()


def calc_r2_denominator():
    data_path = "/home/sam/Desktop/so2sat_test/So2Sat_POP_Part1"
    # data_path = "/home/sam/Desktop/so2sat_test/tmp_train"
    nbr_channels = 20
    dataset = studentTeacherDataset(data_path, split='train', use_teacher=False, drop_labels=False,
                                    student_transform=None,
                                    teacher_transform=None, nbr_channels=nbr_channels)
    # dataset = studentTeacherDataset(data_path, split='test', use_teacher=False, drop_labels=False, student_transform=None, teacher_transform=None, nbr_channels=nbr_channels)

    train_mean = 1074.556447 # 883259408384.0     # 7373152.5
    # val_mean = 1158.546960     # 180515012608.0 # 9868522.0

    total_sum = 0

    for _, _, label, _ in tqdm(dataset):
        total_sum += np.square(label - train_mean)

    print(f'total_sum: {total_sum}')
    print(len(dataset))
    print(f'specific_sum: {total_sum/len(dataset)}')



