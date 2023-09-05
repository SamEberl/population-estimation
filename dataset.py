import csv
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from rasterio.enums import Resampling
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms

from logging_utils import logger

import os
import rasterio
import math


class CustomDataset(Dataset):
    def __init__(self,
                 data_dir,
                 split: str = "train",
                 transform=None):
        self.transform = transform
        self.data_sub_dir = os.path.join(data_dir, split)
        self.data = []

        self.clip_min = 0
        self.clip_max = 4000

        splits = ['train', 'valid', 'test']

        # if split == 'all':
        #     self.get_data(os.path.join(data_dir, splits[0]))
        #     self.get_data(os.path.join(data_dir, splits[0]))
        #     self.get_data(os.path.join(data_dir, splits[0]))
        if split not in splits:
            raise ValueError(f'split: "{split}" not in {splits}')
        else:
            self.get_data()

        logger.info(f'{split}-dataset length: {len(self.data)}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = self.data[idx][0]
        label = self.data[idx][1]
        datapoint_name = self.data[idx][2]
        try:
            with rasterio.open(file_path, 'r') as data:
                # Read all bands/channels
                #image_bands = data.read(out_shape=(data.count, 100, 100), resampling=Resampling.cubic)[[3, 2, 1], :, :]
                image_bands = data.read(out_shape=(data.count, 100, 100))[[3, 2, 1], :, :].astype(np.float16)

                image_bands = np.clip(image_bands, self.clip_min, self.clip_max)  # * (1 / self.clip_max)

                # Set empty parts to mean to avoid skewing the normalization
                mask_0 = (image_bands == 0)
                image_bands[mask_0] = image_bands.mean()

                # normalize channelwise
                image_bands[0, :, :] = (image_bands[0, :, :] - image_bands[0, :, :].min()) / (image_bands[0, :, :].max() - image_bands[0, :, :].min())
                image_bands[1, :, :] = (image_bands[1, :, :] - image_bands[1, :, :].min()) / (image_bands[1, :, :].max() - image_bands[1, :, :].min())
                image_bands[2, :, :] = (image_bands[2, :, :] - image_bands[2, :, :].min()) / (image_bands[2, :, :].max() - image_bands[2, :, :].min())

                # reset empty parts back to 0
                image_bands[mask_0] = 0

                # pil_image = Image.fromarray((image_bands * 255).astype(np.uint8), mode="RGB")
                # pil_image = Image.fromarray(image_bands)
                tensor_data = torch.from_numpy(image_bands)
                tensor_data = self.transform(tensor_data)
                return tensor_data, label, datapoint_name
                #return {'inputs': tensor_data, 'label': label, 'name': datapoint_name}
        except rasterio.RasterioIOError:
            print(f'Image not found at: {file_path}')
            return None, None, None
            #return {'data': None, 'label': None, 'name': None}

    # def __getitem__(self, idx):
    #     try:
    #         # Open the satellite data file
    #         with rasterio.open(self.data[idx][0][0]) as lcz:
    #             # Read all bands/channels
    #             image_bands = lcz.read().astype(np.float32) / 35
    #             # Convert the data to a PyTorch tensor
    #             tensor_data = torch.from_numpy(image_bands)
    #
    #             with rasterio.open(self.data[idx][0][1]) as lu:
    #                 # Read all bands/channels
    #                 image_bands = lu.read().astype(np.float32)
    #                 # Convert the data to a PyTorch tensor
    #                 tensor_data = torch.cat((tensor_data, torch.from_numpy(image_bands)), dim=0)
    #
    #                 label = self.data[idx][1]
    #
    #                 return tensor_data, label
    #     except rasterio.RasterioIOError:
    #         # Handle the case where the image file cannot be opened
    #         # Return None or a sentinel value to indicate the failure
    #         return None, None


    def get_data(self):
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
                    #print(f'Class {Class_nbr}')
                    if os.path.isfile(file_path):
                        self.data.append((file_path, label, datapoint_name))
                    else:
                        print(f'Could not find file: {file_path}')


    def get_data_archive(self):
        for city_folder in os.listdir(self.data_sub_dir):

            # get list of all files in directory and its subdirectories
            path_list = []
            # for root, dirs, files in os.walk(os.path.join(self.data_sub_dir, f'{city_folder}/lcz')):
            #     for file in files:
            #         file_list.append(os.path.join(root, file))
            # for root, dirs, files in os.walk(os.path.join(self.data_sub_dir, f'{city_folder}/lu')):
            #     for file in files:
            #         file_list.append(os.path.join(root, file))
            # for root, dirs, files in os.walk(os.path.join(self.data_sub_dir, f'{city_folder}/viirs')):
            #     for file in files:
            #         file_list.append(os.path.join(root, file))
            for root, dirs, files in os.walk(os.path.join(self.data_sub_dir, f'{city_folder}/sen2spring')):
                for file in files:
                    path_list.append(os.path.join(root, file))

            # Load the csv file that maps datapoint names to folder names
            with open(os.path.join(self.data_sub_dir, f'{city_folder}/{city_folder}.csv'), 'r') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    datapoint_name = row[0]
                    label = row[2]

                    # Filter the file list to include only files with the datapoint name
                    for path in path_list:
                        file_path = None
                        basename = os.path.basename(path)
                        if basename.startswith(datapoint_name) and basename.endswith('.tif'):
                            print(f'basename: {basename}    datapoint: {datapoint_name}')
                            print(path)
                            file_path = path
                            break
                    #filtered_files = [f for f in path_list if datapoint_name in os.path.basename(f)]
                    #print(filtered_files)

                    if file_path != None:
                        self.data.append((file_path, label, datapoint_name))


class aeDataset():
    """
    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
            self,
            data_path: str,
            train_batch_size: int = 8,
            val_batch_size: int = 8,
            patch_size: int = 100,
            num_workers: int = 0,
            pin_memory: bool = False,
            **kwargs):

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        train_transforms = transforms.Compose([#transforms.RandomHorizontalFlip(),
                                               transforms.Pad(6, 0, padding_mode='constant'),
                                               #transforms.CenterCrop(98),
                                               #transforms.Resize(self.patch_size),
                                               #transforms.ToTensor(),
                                               #transforms.Normalize((0.34234527, 0.38267878, 0.41151279), (0.12574408, 0.07536756, 0.0596833))
                                               ])

        val_transforms = transforms.Compose([#transforms.RandomHorizontalFlip(),
                                             transforms.Pad(6, 0, padding_mode='constant'),
                                             #transforms.CenterCrop(98),
                                             #transforms.Resize(self.patch_size),
                                             #transforms.ToTensor(),
                                             #transforms.Normalize((0.01, 0.01, 0.01), (1, 1, 1))
                                             ])

        self.train_dataset = CustomDataset(
            self.data_dir,
            split='train',
            transform=train_transforms,
        )

        self.val_dataset = CustomDataset(
            self.data_dir,
            split='test',
            transform=val_transforms,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
            collate_fn=self.custom_collate
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
            collate_fn=self.custom_collate
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
            collate_fn=self.custom_collate
        )


    def custom_collate(self, batch):
        # Get the maximum number of channels in the batch
        max_channels = max([x[0].shape[0] for x in batch])

        # Create an empty tensor with the maximum number of channels
        x_batch = torch.zeros((len(batch), max_channels, batch[0][0].shape[1], batch[0][0].shape[2]))

        # Create an empty list for the labels
        y_batch = []
        names = []

        # Loop through the batch and fill in the tensor and label list
        for i, (x, y, name) in enumerate(batch):
            x_batch[i, :x.shape[0], :, :] = x
            # x_batch[i, :x.shape[0], :100, :100] = x
            y_batch.append(int(y))
            names.append(batch[i][2])

        # Convert the label list to a tensor
        y_batch = torch.reshape(torch.tensor(y_batch, dtype=torch.float32), (-1, 1))

        return x_batch, y_batch, names


