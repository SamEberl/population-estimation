import csv
import random

import numpy as np
import torch
import os
import rasterio
import math

from torch.utils.data import Dataset
from logging_utils import logger


class studentTeacherDataset(Dataset):
    def __init__(self,
                 data_dir,
                 split: str = "train",
                 use_teacher=False,
                 semi_supervised=False,
                 student_transform=None,
                 teacher_transform=None,
                 percentage_unlabeled=0.0):
        self.student_transform = student_transform
        self.teacher_transform = teacher_transform
        self.split = split
        self.data_sub_dir = os.path.join(data_dir, split)
        self.data = []
        self.clip_min = 0
        self.clip_max = 4000
        self.use_teacher = use_teacher
        self.percentage_unlabeled = percentage_unlabeled

        splits = ['train', 'valid', 'test']
        if split not in splits:
            raise ValueError(f'split: "{split}" not in {splits}')
        else:
            self.get_data_paths()
            if semi_supervised and split == 'train':
                self.split_labeled_unlabeled()
        logger.info(f'{split}-dataset length: {len(self.data)}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = self.data[idx][0]
        label = torch.tensor(float(self.data[idx][1]))
        datapoint_name = self.data[idx][2]

        data = np.empty((7, 100, 100), dtype=np.float32)

        # All values are expected to be between 0 and 1
        data[0:3, :, :] = self.generate_rgb_img(file_path)  # sen2spring_rgb
        data[3:7, :, :] = self.generate_lu(file_path)  # lu
        # data[7, :, :] = self.generate_viirs(file_path)  # viirs

        # tensor_data = torch.cat((sen2spring_rgb, lu_tensor, viirs_tensor), dim=0)
        # data = np.concatenate((sen2spring_rgb, lu, viirs))
        # transformed_data = self.student_transform(image=data)["image"]
        # print(f'tens: {tensor_data.shape}')

        if self.student_transform is not None:
            student_data = self.student_transform(image=data.transpose(1, 2, 0))['image'].transpose(2, 0, 1)
        else:
            student_data = data

        if self.use_teacher == True:
            if self.teacher_transform is not None:
                teacher_data = self.teacher_transform(image=data.transpose(1, 2, 0))['image'].transpose(2, 0, 1)
            else:
                teacher_data = data
        else:
            teacher_data = 0

        # print(f'student_data[0]: \n{student_data[0, :, :]}')
        # print(f'student_data[1]: \n{student_data[1, :, :]}')
        # print(f'student_data[2]: \n{student_data[2, :, :]}')
        # print(f'student_data[3]: \n{student_data[3, :, :]}')
        # print(f'student_data[4]: \n{student_data[4, :, :]}')
        # print(f'student_data[5]: \n{student_data[5, :, :]}')
        # print(f'student_data[6]: \n{student_data[6, :, :]}')
        # print(f'student_data[7]: \n{student_data[7, :, :]}')

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

    def generate_rgb_img(self, file_path):
        try:
            with rasterio.open(file_path, 'r') as data:
                # image_bands = data.read(out_shape=(data.count, 100, 100), resampling=Resampling.cubic)[[3, 2, 1], :, :]
                image_bands = data.read()[[3, 2, 1], :, :].astype(np.float16)
                image_bands = np.clip(image_bands, self.clip_min, self.clip_max)  # * (1 / self.clip_max)
                # Set empty parts to mean to avoid skewing the normalization
                image_bands[(image_bands == 0)] = image_bands.mean()
                # normalize channelwise
                image_bands -= image_bands.min(axis=(1, 2), keepdims=True)
                image_bands /= (image_bands.max(axis=(1, 2), keepdims=True) - image_bands.min(axis=(1, 2), keepdims=True))
                #image_bands = torch.from_numpy(image_bands)
                return image_bands
        except rasterio.RasterioIOError:
            print(f'Image could not be created from: {file_path}')
            return None

    def generate_lu(self, file_path):
        file_path = file_path.replace('sen2spring', 'lu')
        try:
            with rasterio.open(file_path, 'r') as data:
                image_bands = data.read().astype(np.float16)
                #image_bands = torch.from_numpy(image_bands)
                return image_bands
        except rasterio.RasterioIOError:
            print(f'Image could not be created from: {file_path}')
            return None

    def generate_viirs(self, file_path):
        file_path = file_path.replace('sen2spring', 'viirs')
        try:
            with rasterio.open(file_path, 'r') as data:
                image_bands = data.read().astype(np.float16)
                image_bands[image_bands < 0] = 0
                image_bands = image_bands / 20
                #image_bands = torch.from_numpy(image_bands)
                return image_bands
        except rasterio.RasterioIOError:
            print(f'Image could not be created from: {file_path}')
            return None
