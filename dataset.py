import csv
import random

import numpy as np
import torch
import os
import rasterio
import math

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from logging_utils import logger
from scipy.stats import rankdata

# import albumentations as A


def get_data(data_dir, split, reduce_zeros, reduce_zeros_percent):
    data = []
    nbr_not_found = 0
    nbr_found = 0
    if split == 'valid':  # Because on server the val set is called test :(
        split = 'test'
    data_sub_dir = os.path.join(data_dir, split)
    for city_folder in os.listdir(data_sub_dir):
        # Load the csv file that maps datapoint names to folder names
        with open(os.path.join(data_sub_dir, f'{city_folder}/{city_folder}.csv'), 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                datapoint_name = row[0]
                label = row[2]

                if label == 'POP':  # skip first row which is header of file
                    continue
                elif int(label) == 0:
                    class_nbr = 'Class_0'
                else:
                    class_nbr = f'Class_{math.ceil(math.log(int(label), 2) + 0.00001)}'

                modality = 'sen2spring'
                file_name = datapoint_name + '_' + modality + '.tif'
                input_path = os.path.join(data_sub_dir, city_folder, modality, class_nbr, file_name)
                if os.path.isfile(input_path):
                    nbr_found += 1
                    data.append((input_path, label))
                else:
                    nbr_not_found += 1
    if split == 'train':
        for i, data_point in enumerate(data):
            _, label = data_point
            if int(label) == 0 and reduce_zeros:
                del data[i]
    print(f'In: {data_sub_dir} \n  #found: {nbr_found} \n  #notFound: {nbr_not_found}')
    return data


def reduce_data_func(data, percent, seed):
    random.seed(seed)
    random.shuffle(data)
    keep_size = int((1-percent) * len(data))
    reduced_data = data[:keep_size]
    return reduced_data


def create_dataloader(data, split, transform_params, use_channels, bs, num_workers):
    dataset = PopDataset(data, split, transform_params, use_channels=use_channels)
    dataloader = DataLoader(dataset,
                            batch_size=bs,
                            shuffle=True,
                            num_workers=num_workers,
                            pin_memory=True)
    return dataloader


def get_dataloaders(config):
    data_dir = config['data_params']['data_dir']
    reduce_zeros = config['data_params']['reduce_zeros']
    reduce_zeros_percent = config['data_params']['reduce_zeros_percent']
    reduce_data = config['data_params']['reduce_data']
    reduce_data_percent = config['data_params']['reduce_data_percent']
    unlabeled_data = config['data_params']['unlabeled_data']
    reduce_supervised = config['data_params']['reduce_supervised']
    reduce_supervised_percent = config['data_params']['reduce_supervised_percent']
    seed = config['data_params']['seed']

    bs_train = config['data_params']['train_batch_size']
    bs_valid = config['data_params']['val_batch_size']
    num_workers = config['data_params']['num_workers']

    use_channels = {'use_spring_rgb': config['data_params']['use_spring_rgb'],
                    'use_lu': config['data_params']['use_lu'],
                    'use_lcz': config['data_params']['use_lcz'],
                    'use_dem': config['data_params']['use_dem'],
                    'use_viirs': config['data_params']['use_viirs']}

    transform_params = config['transform_params']

    data = {'train': [], 'valid': []}

    # fill data
    for split in data.keys():
        data[split] = get_data(data_dir, split, reduce_zeros, reduce_zeros_percent)

    # drop data
    if reduce_data:
        data['train'] = reduce_data_func(data['train'], reduce_data_percent, seed)

    # create dataloaders
    train_dataloader_unlabeled = None
    if unlabeled_data:
        train_dataloader_unlabeled = create_dataloader(data['train'], 'train_unlabeled', transform_params, use_channels, bs_train, num_workers)
    if reduce_supervised:
        data['train'] = reduce_data_func(data['train'], reduce_supervised_percent, seed)
    train_dataloader = create_dataloader(data['train'], 'train', transform_params, use_channels, bs_train, num_workers)
    valid_dataloader = create_dataloader(data['valid'], 'valid', transform_params, use_channels, bs_valid, num_workers)

    return train_dataloader, valid_dataloader, train_dataloader_unlabeled


class PopDataset(Dataset):
    def __init__(self,
                 data,
                 split,
                 transform_params,
                 use_channels):
        self.data = data
        self.split = split
        self.transform_params = transform_params

        self.nbr_channels = 0
        self.channel_functions = []
        if use_channels['use_spring_rgb']:
            self.channel_functions.append(self.generate_spring_rgb)
            self.nbr_channels += 3
        if use_channels['use_lu']:
            self.channel_functions.append(self.generate_lu)
            self.nbr_channels += 4
        if use_channels['use_lcz']:
            self.channel_functions.append(self.generate_lcz)
            self.nbr_channels += 1
        if use_channels['use_dem']:
            self.channel_functions.append(self.generate_dem)
            self.nbr_channels += 1
        if use_channels['use_viirs']:
            self.channel_functions.append(self.generate_viirs)
            self.nbr_channels += 1

        logger.info(f'dataset length: {len(self.data)}')
        # TODO: Maybe try Quantile normalization for targets if z-score normalization doesnt work well

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = self.data[idx][0]
        label = torch.tensor(float(self.data[idx][1]))

        data = np.empty((self.nbr_channels, 100, 100), dtype=np.float32)

        # Iterate over the channel functions and populate the data array
        counter = 0
        for func in self.channel_functions:
            if func == self.generate_spring_rgb:
                data[counter:counter + 3, :, :] = func(file_path)  # RGB occupies 3 channels
                counter += 3
            elif func == self.generate_lu:
                data[counter:counter+4, :, :] = func(file_path)
                counter += 4
            else:
                data[counter, :, :] = func(file_path)
                counter += 1
        if self.split == 'train':
            return apply_transforms(data, self.transform_params), label
        elif self.split == 'valid' or self.split == 'test':
            return data, label
        elif self.split == 'train_unlabeled':
            return data, apply_transforms(data, self.transform_params), label


    def generate_spring_rgb(self, file_path):
        try:
            with rasterio.open(file_path, 'r') as data:
                image_bands = data.read()[[3, 2, 1], :, :]#.astype(np.float16)
                image_bands = np.clip(image_bands, 0, 4000)
                image_bands = image_bands / 4000
                return image_bands
        except rasterio.RasterioIOError:
            print(f'Image could not be created from: {file_path}')
            return None

    def generate_winter_rgb(self, file_path):
        file_path = file_path.replace('sen2spring', 'sen2winter')
        try:
            with rasterio.open(file_path, 'r') as data:
                image_bands = data.read()[[3, 2, 1], :, :]#.astype(np.float16)
                image_bands = np.clip(image_bands, 0, 4000)  # * (1 / self.clip_max)
                image_bands /= 4000
                return image_bands
        except rasterio.RasterioIOError:
            print(f'Image could not be created from: {file_path}')
            return None

    def generate_lu(self, file_path):
        file_path = file_path.replace('sen2spring', 'lu')
        try:
            with rasterio.open(file_path, 'r') as data:
                image_bands = data.read()#.astype(np.float16)
                # image_bands[image_bands > 1] = 1
                return image_bands
        except rasterio.RasterioIOError:
            print(f'Image could not be created from: {file_path}')
            return None

    def generate_viirs(self, file_path):
        file_path = file_path.replace('sen2spring', 'viirs')
        try:
            with rasterio.open(file_path, 'r') as data:
                image_bands = data.read()#.astype(np.float16)
                image_bands[image_bands > 50] = 50
                image_bands /= 50
                return image_bands
        except rasterio.RasterioIOError:
            print(f'Image could not be created from: {file_path}')
            return None

    def generate_lcz(self, file_path):
        file_path = file_path.replace('sen2spring', 'lcz')
        try:
            with rasterio.open(file_path, 'r') as data:
                image_bands = data.read()
                image_bands[np.logical_and(image_bands > 0, image_bands <= 9)] = (image_bands[np.logical_and(image_bands > 0, image_bands <= 9)] + 1) / 10
                image_bands[image_bands > 9] = 0
                return image_bands
        except rasterio.errors.RasterioIOError as e:
            print(f'Image could not be created from: {file_path}')
            return None

    def generate_dem(self, file_path):
        file_path = file_path.replace('So2Sat_POP_Part1', 'So2Sat_POP_Part2')
        file_path = file_path.replace('sen2spring', 'dem')
        try:
            with rasterio.open(file_path, 'r') as data:
                image_bands = data.read()#.astype(np.float16)
                image_bands[image_bands < -2] = -2
                image_bands[image_bands > 10] = 10

                image_bands += 2
                image_bands /= 12

                height, width = image_bands.shape[1], image_bands.shape[2]
                if height != 100 or width != 100:
                    # Calculate how much padding is needed
                    padding_height = max(0, 100 - height)
                    padding_width = max(0, 100 - width)

                    image_bands = np.pad(
                        image_bands,
                        ((0, 0), (0, padding_height), (0, padding_width)),
                        mode='constant',
                        constant_values=image_bands.mean()
                    )
                return image_bands
        except rasterio.RasterioIOError:
            print(f'Image could not be created from: {file_path}')
            return None


def apply_transforms(image_bands, transform_params):
    """
    Apply a series of transforms to the input image.

    Args:
    - image_bands (np.ndarray): Input image tensor with shape (channels, height, width).
    - noise_params (dict): Parameters for adding Gaussian noise.
    - brightness_params (dict): Parameters for adjusting brightness.
    - contrast_params (dict): Parameters for adjusting contrast.
    - patch_params (dict): Parameters for blocking out a patch.

    Returns:
    - np.ndarray: Transformed image.
    """

    if transform_params['apply_rot_and_flip']:
        # Apply rotation and flipping
        image_bands = apply_flip_and_rotate_transforms(
            image_bands,
        )

    if transform_params['apply_noise']:
        # Apply gaussian noise
        image_bands = add_gaussian_noise(
            image_bands,
            mean=transform_params['noise_mean'],
            std_min=transform_params['noise_std_min'],
            std_max=transform_params['noise_std_max'],
            p=transform_params['noise_p']
        )

    if transform_params['apply_brightness']:
        # Adjust brightness
        image_bands = adjust_brightness(
            image_bands,
            brightness_range=transform_params['brightness_range'],
            p=transform_params['brightness_p']
        )

    if transform_params['apply_contrast']:
        # Adjust contrast
        image_bands = adjust_contrast(
            image_bands,
            contrast_range=transform_params['contrast_range'],
            p=transform_params['contrast_p']
        )

    image_bands = np.clip(image_bands, 0, 1)

    return image_bands



def apply_flip_and_rotate_transforms(image_bands, probability=0.5):
    """
    Apply horizontal flip, vertical flip, and 90-degree rotation transforms to the image.
    Each transform has a 0.5 probability of being applied.

    This function can handle images with more than three dimensions, assuming the last three dimensions
    are channels, height, and width.

    Args:
    - image_bands (np.ndarray): Input image with shape (..., channels, height, width).
    - probability (float): Probability of applying each transform.

    Returns:
    - np.ndarray: Transformed image.
    """

    # Apply the transformations only on the last two dimensions (height and width)
    if np.random.rand() < probability:
        # Flipping horizontally
        image_bands = np.flip(image_bands, axis=-2).copy()

    if np.random.rand() < probability:
        # Flipping vertically
        image_bands = np.flip(image_bands, axis=-3).copy()

    if np.random.rand() < probability:
        # Rotating by 90 degrees
        # This rotation swaps the last two axes (height and width)
        image_bands = np.rot90(image_bands, axes=(-1, -2)).copy()

    return image_bands


def add_gaussian_noise(image_bands, mean=0, std_min=0.01, std_max=0.07, p=0.5):
    """
    Add Gaussian noise to the input channels and clip the values to be within [0, 1].

    Args:
    - channels (np.ndarray): Input tensor with shape (channels, height, width).
    - mean (float): Mean of the Gaussian noise (default is 0).
    - std (float): Standard deviation of the Gaussian noise (default is 0.05).

    Returns:
    - torch.Tensor: Output tensor with Gaussian noise, clipped to be within [0, 1].
    """
    apply_noise = np.random.rand() < p

    if apply_noise:
        # Randomly choose a standard deviation value between std_min and std_max
        std = np.random.uniform(std_min, std_max)

        # Generate Gaussian noise
        noise = np.random.normal(loc=mean, scale=std, size=image_bands.shape).astype(np.float32)

        # Add noise to the input channels
        noisy_channels = image_bands + noise

        return noisy_channels
    else:
        return image_bands


def adjust_brightness(image_bands, brightness_range=(0.8, 1.2), p=0.5):
    """
    Randomly adjust the brightness of the input channels.

    Args:
    - image_bands (np.array): Input array with shape (3, height, width).
    - brightness_range (tuple): A tuple containing min and max multiplier for brightness adjustment.
    - p (float): Probability of applying the brightness adjustment.

    Returns:
    - np.array: Output array with adjusted brightness.
    """

    apply_brightness_adjustment = np.random.rand() < p

    if apply_brightness_adjustment:
        # Randomly choose a brightness multiplier
        brightness_multiplier = np.random.uniform(*brightness_range)

        # Adjust brightness
        adjusted_channels = image_bands * brightness_multiplier

        return adjusted_channels
    else:
        return image_bands


def adjust_contrast(image_bands, contrast_range=(0.8, 1.2), p=0.5):
    """
    Randomly adjust the contrast of each channel of the image.

    Args:
    - image_bands (np.array): Input array with shape (num_channels, height, width).
    - contrast_range (tuple): A tuple containing min and max contrast adjustment factors.
    - p (float): Probability of applying the contrast adjustment.

    Returns:
    - np.array: Output array with adjusted contrast.
    """

    apply_contrast_adjustment = np.random.rand() < p

    if apply_contrast_adjustment:
        num_channels = image_bands.shape[0]

        for i in range(num_channels):
            contrast_factor = np.random.uniform(*contrast_range)
            channel_mean = np.mean(image_bands[i])

            # Adjust contrast
            image_bands[i] = (image_bands[i] - channel_mean) * contrast_factor + channel_mean

        return image_bands
    else:
        return image_bands


def block_out_patch(image_bands, patch_size=(16, 16), probability=0.5):
    """
    Replace a small patch in the input image bands with its mean with a specified probability.

    Args:
    - image_bands (np.ndarray): Input array with shape (height, width, 3).
    - patch_size (tuple): Size of the patch to be replaced (default is (16, 16)).
    - probability (float): Probability of applying the patch replacement (default is 0.5).

    Returns:
    - np.ndarray: Output array with patch replacement (if applied).
    """
    # Check if the input array has the correct shape
    if image_bands.shape[2] != 3:
        raise ValueError("Input array must have 3 channels.")

    # Determine whether to apply patch replacement based on the specified probability
    apply_patch = np.random.rand() < probability

    if apply_patch:
        # Randomly select the top-left corner of the patch
        h, w = image_bands.shape[0], image_bands.shape[1]
        top_left_h = np.random.randint(0, h - patch_size[0] + 1)
        top_left_w = np.random.randint(0, w - patch_size[1] + 1)

        # Extract the patch
        patch = image_bands[top_left_h:top_left_h + patch_size[0], top_left_w:top_left_w + patch_size[1], :]

        # Calculate the mean of the patch
        patch_mean = np.mean(patch, axis=(0, 1))

        # Replace the patch with its mean
        image_bands[top_left_h:top_left_h + patch_size[0], top_left_w:top_left_w + patch_size[1], :] = patch_mean

    return image_bands


def normalize_labels(labels, mean=1068, std=1792):
    """
    Normalizes the labels by applying Z-score normalization.
    This assumes we drop 90% of the zeros and ignore values above 10k to calc the mean and std
    """
    mean = 5.37
    std = 2.32
    log_labels = torch.log(labels)
    return (log_labels - mean) / std


def unnormalize_preds(preds, mean=1068, std=1792):
    """
    UnNormalizes the predictions by reversing Z-score normalization.
    """
    mean = 5.37
    std = 2.32
    exp_preds = torch.exp(preds)
    return (exp_preds * std) + mean


def quantile_normalize_labels(labels):
    """
    Performs quantile normalization on the given labels using PyTorch.

    Parameters:
    labels (Tensor): The labels to be normalized.

    Returns:
    Tensor: The quantile normalized labels.
    """
    if not torch.is_tensor(labels):
        labels = torch.tensor(labels)

    # Sort labels and get the original indices
    sorted_labels, sorted_indices = torch.sort(labels)

    # Calculate ranks (average rank for ties)
    # This method calculates the ranks by counting the occurrences of each value
    unique_values, inverse_indices, counts = torch.unique_consecutive(sorted_labels, return_inverse=True, return_counts=True)
    ranks = torch.cumsum(counts, dim=0) - counts / 2.0
    ranks = ranks[inverse_indices]

    # Normalize ranks to get quantiles
    quantiles = ranks / (labels.numel() - 1)

    # Create an empty tensor to hold the normalized values
    normalized_labels = torch.empty_like(labels)

    # Place quantiles back to their original indices
    normalized_labels.scatter_(0, sorted_indices, quantiles)

    return normalized_labels