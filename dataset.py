import csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms

import os
import rasterio


class CustomDataset(Dataset):
    def __init__(self,
                 data_dir,
                 split: str = "train",
                 transform=None):
        self.transform = transform
        self.data_sub_dir = os.path.join(data_dir, split)
        self.data = []

        splits = ['train', 'valid', 'test']

        # if split == 'all':
        #     self.get_data(os.path.join(data_dir, splits[0]))
        #     self.get_data(os.path.join(data_dir, splits[0]))
        #     self.get_data(os.path.join(data_dir, splits[0]))
        if split not in splits:
            raise ValueError(f'split: "{split}" not in {splits}')
        else:
            self.get_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            # Open the satellite data file
            with rasterio.open(self.data[idx][0][0]) as lu:
                # Read all bands/channels
                image_bands = lu.read().astype(np.float32)
                # Convert the data to a PyTorch tensor
                tensor_data = torch.from_numpy(image_bands)

                label = self.data[idx][1]

                return tensor_data, label
        except rasterio.RasterioIOError:
            # Handle the case where the image file cannot be opened
            # Return None or a sentinel value to indicate the failure
            return None, None

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

            # get list of all files in directory and its subdirectories
            file_list = []
            # for root, dirs, files in os.walk(os.path.join(self.data_sub_dir, f'{city_folder}/lcz')):
            #     for file in files:
            #         file_list.append(os.path.join(root, file))
            for root, dirs, files in os.walk(os.path.join(self.data_sub_dir, f'{city_folder}/lu')):
                for file in files:
                    file_list.append(os.path.join(root, file))
            # for root, dirs, files in os.walk(os.path.join(self.data_sub_dir, f'{city_folder}/viirs')):
            #     for file in files:
            #         file_list.append(os.path.join(root, file))

            # Load the csv file that maps datapoint names to folder names
            with open(os.path.join(self.data_sub_dir, f'{city_folder}/{city_folder}.csv'), 'r') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    datapoint_name = row[0]
                    label = row[2]

                    if '1kmN' not in datapoint_name:
                        continue

                    # Filter the file list to include only files with the datapoint name
                    filtered_files = [f for f in file_list if datapoint_name in os.path.basename(f)]

                    # Read the data from the files for this datapoint
                    # datapoint_data = []
                    # for filename in filtered_files:
                    #     file_path = os.path.join(self.data_sub_dir, city_folder, filename)
                    #     datapoint_data.append(file_path)

                    # if len(datapoint_data) != 0:
                    #     self.data.append((datapoint_data[0], label))

                    if len(filtered_files) == 1:
                        self.data.append((filtered_files, label))


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

        train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                               transforms.CenterCrop(148),
                                               transforms.Resize(self.patch_size),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                               ])

        val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                             transforms.CenterCrop(148),
                                             transforms.Resize(self.patch_size),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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

        # Loop through the batch and fill in the tensor and label list
        for i, (x, y) in enumerate(batch):
            x_batch[i, :x.shape[0], :, :] = x
            # x_batch[i, :x.shape[0], :100, :100] = x
            y_batch.append(int(y))

        # Convert the label list to a tensor
        y_batch = torch.reshape(torch.tensor(y_batch, dtype=torch.float32), (-1, 1))

        return x_batch, y_batch


