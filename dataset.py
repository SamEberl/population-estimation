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


class studentTeacherDataset(Dataset):
    def __init__(self,
                 data_dir,
                 split: str = "train",
                 use_teacher=False,
                 drop_labels=False,
                 drop_data=False,
                 student_transform=None,
                 teacher_transform=None,
                 percentage_unlabeled=0.0,
                 nbr_channels=7):
        self.student_transform = student_transform
        self.teacher_transform = teacher_transform
        self.split = split
        self.data_sub_dir = os.path.join(data_dir, split)
        self.data = []
        self.clip_min = 0
        self.clip_max = 4000
        self.use_teacher = use_teacher
        self.percentage_unlabeled = percentage_unlabeled
        self.nbr_channels = nbr_channels

        splits = ['train', 'valid', 'test']
        if split not in splits:
            raise ValueError(f'split: "{split}" not in {splits}')
        else:
            self.get_data_paths()
            if drop_labels and split == 'train':
                self.split_labeled_unlabeled()
            if drop_data and split == 'train':
                self.drop_data()
        logger.info(f'{split}-dataset length: {len(self.data)}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = self.data[idx][0]
        label = torch.tensor(float(self.data[idx][1]))
        datapoint_name = self.data[idx][2]

        data = np.empty((self.nbr_channels, 100, 100), dtype=np.float32)

        # All values are expected to be between 0 and 1
        data[0:3, :, :] = self.generate_rgb_img(file_path)  # sen2spring_rgb
        # data[3:7, :, :] = self.generate_lu(file_path)  # lu
        # data[7, :, :] = self.generate_lcz(file_path)
        # data[8, :, :] = self.generate_dem(file_path)
        # data[6:9, :, :] = self.generate_winter_img(file_path)

        # data[15, :, :] = self.generate_viirs(file_path)  # viirs

        if self.student_transform is not None:
            student_data = self.student_transform(image=data.transpose(1, 2, 0))['image'].transpose(2, 0, 1)
            student_data = self.add_gaussian_noise(student_data)
            student_data = self.adjust_brightness(student_data)
            student_data = self.adjust_contrast(student_data)
            # Clip the values to ensure they are within a valid range
            student_data = np.clip(student_data, 0, 1)
        else:
            student_data = data

        if self.use_teacher == True:
            if self.teacher_transform is not None:
                teacher_data = self.teacher_transform(image=data.transpose(1, 2, 0))['image'].transpose(2, 0, 1)
            else:
                teacher_data = data
        else:
            teacher_data = 0

        return student_data, teacher_data, label, datapoint_name


    def get_data_paths(self):
        nbr_not_found = 0
        nbr_found = 0
        for city_folder in os.listdir(self.data_sub_dir):
            # Load the csv file that maps datapoint names to folder names
            with open(os.path.join(self.data_sub_dir, f'{city_folder}/{city_folder}.csv'), 'r') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    datapoint_name = row[0]
                    label = row[2]
                    if label == 'POP':  # skip first row which is header of file
                        continue
                    elif int(label) == 0:
                        Class_nbr = 'Class_0'
                    else:
                        Class_nbr = f'Class_{math.ceil(math.log(int(label), 2)+0.00001)}'

                    modality = 'sen2spring'
                    file_name = datapoint_name + '_' + modality + '.tif'
                    file_path = os.path.join(self.data_sub_dir, city_folder, modality, Class_nbr, file_name)
                    if os.path.isfile(file_path):
                        nbr_found += 1
                        self.data.append((file_path, label, datapoint_name))
                    else:
                        nbr_not_found += 1
                        # print(f'Could not find file: {file_path}')
        # print(f'In: {self.data_sub_dir} \n  #found: {nbr_found} \n  #notFound: {nbr_not_found}')

    def split_labeled_unlabeled(self):
        # Simply remove 80% of labels -> Loss has to be adjusted. Sometimes batch has no labels.
        # Alternataive option would be: 2 different datasets. One with and one without labels
        random.seed(42)
        random.shuffle(self.data)
        total_images = len(self.data)
        unlabeled_size = int(self.percentage_unlabeled * total_images)

        labeled_data = self.data[unlabeled_size:]
        unlabeled_data = [(path, -1, name) for path, _, name in self.data[:unlabeled_size]]

        self.data = unlabeled_data + labeled_data

    def drop_data(self):
        # Drop 80% of datapoints
        random.seed(42)
        random.shuffle(self.data)
        total_images = len(self.data)
        # Calculate the number of images to keep (20% of the total)
        keep_size = int(0.2 * total_images)
        # Keep only 20% of the data
        self.data = self.data[:keep_size]

    def generate_rgb_img(self, file_path):
        try:
            with rasterio.open(file_path, 'r') as data:
                image_bands = data.read()[[3, 2, 1], :, :].astype(np.float16)
                image_bands = np.clip(image_bands, self.clip_min, self.clip_max)
                image_bands /= self.clip_max
                return image_bands
        except rasterio.RasterioIOError:
            print(f'Image could not be created from: {file_path}')
            return None

    def generate_winter_img(self, file_path):
        file_path = file_path.replace('sen2spring', 'sen2winter')
        try:
            with rasterio.open(file_path, 'r') as data:
                image_bands = data.read()[[3, 2, 1], :, :].astype(np.float16)
                image_bands = np.clip(image_bands, self.clip_min, self.clip_max)  # * (1 / self.clip_max)
                image_bands /= self.clip_max
                return image_bands
        except rasterio.RasterioIOError:
            print(f'Image could not be created from: {file_path}')
            return None

    def generate_lu(self, file_path):
        file_path = file_path.replace('sen2spring', 'lu')
        try:
            with rasterio.open(file_path, 'r') as data:
                image_bands = data.read().astype(np.float16)
                # image_bands[image_bands > 1] = 1
                return image_bands
        except rasterio.RasterioIOError:
            print(f'Image could not be created from: {file_path}')
            return None

    def generate_viirs(self, file_path):
        file_path = file_path.replace('sen2spring', 'viirs')
        try:
            with rasterio.open(file_path, 'r') as data:
                image_bands = data.read().astype(np.float16)
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
                image_bands = data.read().astype(np.float16)
                image_bands /= 17
                return image_bands
        except rasterio.RasterioIOError:
            print(f'Image could not be created from: {file_path}')
            return None

    def generate_dem(self, file_path):
        file_path = file_path.replace('So2Sat_POP_Part1', 'So2Sat_POP_Part2')
        file_path = file_path.replace('sen2spring', 'dem')
        try:
            with rasterio.open(file_path, 'r') as data:
                image_bands = data.read().astype(np.float16)
                image_bands[image_bands < -2] = -2
                image_bands[image_bands > 10] = 10

                image_bands += 2
                image_bands /= 12

                height, width = image_bands.shape[1], image_bands.shape[2]
                # Calculate how much padding is needed
                padding_height = max(0, 100 - height)
                padding_width = max(0, 100 - width)

                padded_image = np.pad(
                    image_bands,
                    ((0, 0), (0, padding_height), (0, padding_width)),
                    mode='constant',
                    constant_values=image_bands.mean()
                )
                return padded_image
        except rasterio.RasterioIOError:
            print(f'Image could not be created from: {file_path}')
            return None

    def add_gaussian_noise(self, image_bands, mean=0, std_min=0.01, std_max=0.07, p=0.5):
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
            noise = np.random.normal(loc=mean, scale=std, size=image_bands.shape).astype(np.float16)

            # Add noise to the input channels
            noisy_channels = image_bands + noise

            return noisy_channels
        else:
            return image_bands

    import numpy as np

    def adjust_brightness(self, image_bands, brightness_range=(0.8, 1.2), p=0.5):
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
            brightness_multiplier = np.random.uniform(*brightness_range).astype(np.float16)

            # Adjust brightness
            adjusted_channels = image_bands * brightness_multiplier

            return adjusted_channels
        else:
            return image_bands

    def adjust_contrast(self, image_bands, contrast_range=(0.8, 1.2), p=0.5):
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
                contrast_factor = np.random.uniform(*contrast_range).astype(np.float16)
                channel_mean = np.mean(image_bands[i]).astype(np.float16)

                # Adjust contrast
                image_bands[i] = (image_bands[i] - channel_mean) * contrast_factor + channel_mean

            return image_bands
        else:
            return image_bands


    def block_out_patch(self, image_bands, patch_size=(16, 16), probability=0.5):
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

def get_dataloader(config, student_transform, teacher_transform):
    data_path = config["data_params"]["data_path"]
    train_bs = config["data_params"]["train_batch_size"]
    #val_bs = config["data_params"]["val_batch_size"]
    num_workers = config["data_params"]["num_workers"]
    use_teacher = config['train_params']['use_teacher']
    drop_labels = config['data_params']['drop_labels']
    drop_data = config['data_params']['drop_data']
    seed = config['train_params']['seed']
    percentage_unlabeled = config['data_params']['percentage_unlabeled']
    nbr_channels = config['model_params']['in_channels']

    train_dataset = studentTeacherDataset(data_path, split='train', use_teacher=use_teacher, drop_labels=drop_labels, drop_data=drop_data, student_transform=student_transform, teacher_transform=teacher_transform, percentage_unlabeled=percentage_unlabeled, nbr_channels=nbr_channels)
    val_dataset = studentTeacherDataset(data_path, split='test', use_teacher=use_teacher, drop_labels=drop_labels, drop_data=drop_data, student_transform=None, teacher_transform=None, percentage_unlabeled=percentage_unlabeled, nbr_channels=nbr_channels)

    # Use adapted val batch sizes to accommodate different amounts of data
    data_ratio = len(train_dataset) / len(val_dataset)

    shuffle = True
    # train_sampler = None
    # val_sampler = None
    # if config['hparam_search']['active']:
    #     shuffle = False
    #     train_sampler = SequentialSampler(train_dataset)
    #     val_sampler = SequentialSampler(val_dataset)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=train_bs,
                                  shuffle=shuffle,
                                  #sampler=train_sampler,
                                  num_workers=num_workers,
                                  pin_memory=True,
                                  )
    val_dataloader = DataLoader(val_dataset,
                                batch_size=int(train_bs//data_ratio),
                                shuffle=shuffle,
                                #sampler=val_sampler,
                                num_workers=num_workers,
                                pin_memory=True,
                                )

    return train_dataloader, val_dataloader