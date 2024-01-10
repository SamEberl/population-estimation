def get_transforms(config):
    student_transforms_list = []
    # Loop through the dictionary and add augmentations to the list
    for student_params in config['student_transforms']:
        student_aug_fn = getattr(A, list(student_params.keys())[0])(**list(student_params.values())[0])
        student_transforms_list.append(student_aug_fn)
    # Create an augmentation pipeline using the list of augmentation functions
    student_transform = A.Compose(student_transforms_list)

    teacher_transfroms_list = []
    if config['train_params']['use_teacher']:
        # Loop through the dictionary and add augmentations to the list
        for teacher_params in config['teacher_transforms']:
            teacher_aug_fn = getattr(A, list(teacher_params.keys())[0])(**list(teacher_params.values())[0])
            teacher_transfroms_list.append(teacher_aug_fn)
        # Create an augmentation pipeline using the list of augmentation functions
    teacher_transform = A.Compose(teacher_transfroms_list)

    return student_transform, teacher_transform


def do_transforms():
    if self.student_transform is not None and self.split == 'train':
        student_data = self.student_transform(image=data.transpose(1, 2, 0))['image'].transpose(2, 0, 1)
        student_data = self.add_gaussian_noise(student_data)  # TODO: Move transforms outside of dataloader
        student_data = self.adjust_brightness(student_data)
        student_data = self.adjust_contrast(student_data)
        # Clip the values to ensure they are within a valid range
        student_data = np.clip(student_data, 0, 1)
    else:
        student_data = data

    if self.use_teacher:
        if self.teacher_transform is not None and self.split == 'train':
            teacher_data = self.teacher_transform(image=data.transpose(1, 2, 0))['image'].transpose(2, 0, 1)
        else:
            teacher_data = data
    else:
        teacher_data = np.float32(0)

    student_data = student_data.astype(np.float32)
    teacher_data = teacher_data.astype(np.float32)

    return student_data, teacher_data, label


def check_inputs():
    # TODO: Implement checks to make sure inputs are valid
    splits = ['train', 'valid']
    if split not in splits:
        raise ValueError(f'split: "{split}" not in {splits}')
    # Ensure inputs and labels have the same length
    assert len(inputs) == len(labels)



# nbr_channels = config['model_params']['in_channels']  # TODO: Model params ???
# use_teacher = config['train_params']['use_teacher']  # TODO: Move elsewhere



# TODO: Can probably be removed for good
def batch_generator(dataloader):
    dataloader_iter = iter(dataloader)
    while True:
        try:
            yield next(dataloader_iter)
        except StopIteration:
            yield None

