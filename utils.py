import subprocess
import webbrowser
import argparse
import yaml
import os
import numpy as np
import tifffile as tiff
from tqdm import tqdm
import matplotlib.pyplot as plt
import imageio.v2 as imageio


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


def check_sizes():
    pass


# open_tensorboard("/home/sam/Desktop/DLR/Data/Data_100GB/trained_models/")


# Specify the directory to search for .tiff images
directory = '/home/sam/Desktop/DLR/Data/Data_100GB/So2Sat_POP_Part1/train/'

# filter_images_by_mean(directory)
# browse_images_with_mean(directory)
