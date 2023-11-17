import random
import datetime
import albumentations as A
import torch
from torch import optim
from torch.utils.data import DataLoader
from dataset import studentTeacherDataset
from tensorboardX import SummaryWriter
from utils import *
from datetime import datetime
from logging_utils import *
from torch.optim.lr_scheduler import CosineAnnealingLR
from models.losses import maskedL1Loss

import matplotlib.pyplot as plt


def forward_pass(student_model,
                 teacher_model,
                 student_inputs,
                 teacher_inputs,
                 labels,
                 config,
                 split,
                 writer,
                 step_nbr,
                 save_img=False):
    hparam_search = config['hparam_search']['active']

    if split == 'train':
        student_model.train()
    else:
        student_model.eval()

    student_preds, student_features, student_data_uncertainty = student_model(student_inputs)
    mask_labeled = labels != -1
    supervised_loss = 0
    if torch.sum(mask_labeled) > 0:
        supervised_loss = student_model.loss_supervised_w_uncertainty(student_preds, labels, student_data_uncertainty)

    if not hparam_search:

        supervised_loss_name = config['model_params']['supervised_criterion']
        if supervised_loss != -1:
            writer.add_scalar(f'Loss-{supervised_loss_name}/{split}', supervised_loss, step_nbr)
        # loss_mae = torch.nn.functional.l1_loss(student_preds, labels)
        loss_mae = maskedL1Loss(student_preds, labels)
        if loss_mae != -1:
            writer.add_scalar(f'Loss-L1-Compare/{split}', loss_mae, step_nbr)
        writer.add_scalar(f'Percentage-Labeled/{split}', torch.sum(mask_labeled)/len(mask_labeled), step_nbr)

        # print(f'supervised_loss: {supervised_loss}')
        # print(f'L1-Compare: {loss_mae}')
        # print(f'Percentage: {torch.sum(mask_labeled)/len(mask_labeled)}')

    unsupervised_loss = 0
    if split == 'train' and config['train_params']['use_teacher']:
        num_samples = config['train_params']['num_samples_teacher']

        # Ensure dropout is active during evaluation
        teacher_model.train()

        # Store all predictions
        n_teacher_preds = []
        n_teacher_features = []
        n_teacher_data_uncertainties = []

        with torch.no_grad():  # Ensure no gradients are computed
            for _ in range(num_samples):
                teacher_preds, teacher_features, teacher_data_uncertainty = teacher_model(teacher_inputs)
                n_teacher_preds.append(teacher_preds)
                n_teacher_features.append(teacher_features)
                n_teacher_data_uncertainties.append(teacher_data_uncertainty)

        n_teacher_preds = torch.stack(n_teacher_preds)
        n_teacher_features = torch.stack(n_teacher_features)
        n_teacher_data_uncertainties = torch.stack(n_teacher_data_uncertainties)

        # Compute model uncertainty
        teacher_model_uncertainty = n_teacher_preds.var(dim=0)

        # Compute feature spread
        teacher_features_mean = n_teacher_features.mean(dim=0)  # Get averaged features
        squared_diffs = (n_teacher_features - teacher_features_mean) ** 2  # Calculate the squared differences
        euclidean_distances = torch.sqrt(squared_diffs.sum(dim=-1))  # Sum along the feature dimension and take root
        teacher_features_spread = euclidean_distances.std(dim=0)  # Get spread

        # Compute data uncertainty
        teacher_data_uncertainty = n_teacher_data_uncertainties.var(dim=0)

        writer.add_scalar(f'Teacher_model_uncertainty/{split}', torch.mean(teacher_model_uncertainty), step_nbr)
        writer.add_scalar(f'Teacher_feature_spread/{split}', torch.mean(teacher_features_spread), step_nbr)
        writer.add_scalar(f'Teacher_data_uncertainty/{split}', torch.mean(teacher_data_uncertainty), step_nbr)

        if config['train_params']['use_teacher']:
            # pseudo_label_mask = (np.sqrt(teacher_model_uncertainty) / n_teacher_preds.mean(dim=0)) > 0.15  # Use CV as threshold
            pseudo_label_mask = teacher_features_spread < 0.1
            writer.add_scalar(f'Percentage-used-unsupervised', torch.sum(pseudo_label_mask)/len(pseudo_label_mask), step_nbr)
            #print(f'Percentage-used-unsupervised: {torch.sum(pseudo_label_mask)/len(pseudo_label_mask)}')
            # pseudo_label_mask = teacher_data_uncertainty < ?
            if torch.sum(pseudo_label_mask) > 0:
                unsupervised_loss = student_model.loss_unsupervised(student_features, teacher_features_mean, pseudo_label_mask)
                writer.add_scalar(f'Loss-Unsupervised/{split}', unsupervised_loss.item(), step_nbr)
                #print(f'unsupervised_loss: {unsupervised_loss.item()}')

        check2 = False
        if check2:  # TODO unsupervised loss on prediction instead of features
            pass
            # teacher_preds_loss = (teacher_preds_mean - labels)**2

    loss = supervised_loss + unsupervised_loss
    writer.add_scalar(f'Loss-Total/{split}', loss, step_nbr)
    # print(f'Loss-total: {loss}')

    # if save_img:
    #     #sample_nbr = random.randint(0, len(student_inputs[:, 0, 0, 0]-1))
    #     writer.add_images(f'Panel_Images', student_inputs[0, :3, :, :], global_step=step_nbr, dataformats='CHW')
    #     log_regression_plot_to_tensorboard(writer, step_nbr, labels.cpu().flatten(),
    #                                        student_preds.cpu().flatten())

    return loss


def batch_generator(dataloader):
    dataloader_iter = iter(dataloader)
    while True:
        try:
            yield next(dataloader_iter)
        except StopIteration:
            yield None


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


def get_dataloader(config, student_transform, teacher_transform):
    data_path = config["data_params"]["data_path"]
    train_bs = config["data_params"]["train_batch_size"]
    #val_bs = config["data_params"]["val_batch_size"]
    num_workers = config["data_params"]["num_workers"]
    use_teacher = config['train_params']['use_teacher']
    semi_supervised = config['data_params']['semi_supervised']
    seed = config['train_params']['seed']
    percentage_unlabeled = config['data_params']['percentage_unlabeled']

    train_dataset = studentTeacherDataset(data_path, split='train', use_teacher=use_teacher, semi_supervised=semi_supervised, student_transform=student_transform, teacher_transform=teacher_transform, percentage_unlabeled=percentage_unlabeled)
    val_dataset = studentTeacherDataset(data_path, split='test', use_teacher=use_teacher, semi_supervised=semi_supervised, student_transform=student_transform, teacher_transform=teacher_transform, percentage_unlabeled=percentage_unlabeled)

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


def train_fix_match(config, writer, student_model, teacher_model, train_dataloader, val_dataloader):
    # Get params from config
    ema_alpha = config["train_params"]["ema_alpha"]  # Exponential moving average decay factor
    num_epochs = config['train_params']['max_epochs']
    use_teacher = config['train_params']['use_teacher']

    optimizer = optim.Adam(student_model.parameters(),
                           lr=config['train_params']['LR'],
                           betas=(config['train_params']['beta1'], config['train_params']['beta2']),
                           weight_decay=config['train_params']['L2_reg'] * 2)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_dataloader)*num_epochs, eta_min=0.00001)

    pbar = tqdm(total=len(train_dataloader), ncols=140)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    total_train_loss = 0
    total_val_loss = 0
    # Train the model
    for epoch in range(num_epochs):
        val_generator = batch_generator(val_dataloader)
        total_train_loss = 0
        total_val_loss = 0
        #config['train_params']['use_teacher'] = (epoch == (num_epochs - 1))
        for i, train_data in enumerate(train_dataloader):
            step_nbr = epoch * len(train_dataloader) + i

            student_inputs, teacher_inputs, labels, datapoint_name = train_data
            student_inputs = student_inputs.to(device)
            if use_teacher:
                teacher_inputs = teacher_inputs.to(device)
            labels = labels.to(device)

            train_loss = forward_pass(
                student_model=student_model,
                teacher_model=teacher_model,
                student_inputs=student_inputs,
                teacher_inputs=teacher_inputs,
                labels=labels,
                config=config,
                split='train',
                writer=writer,
                step_nbr=step_nbr)
            total_train_loss += train_loss

            # Backward pass and optimization
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            scheduler.step()

            # Update teacher model using exponential moving average
            for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
                teacher_param.data.mul_(ema_alpha).add_(student_param.data * (1 - ema_alpha))

            val_data = next(val_generator)
            if val_data is not None:
                step_nbr = epoch * len(val_dataloader) + i

                save_img = False
                if step_nbr % (num_epochs * len(val_dataloader) / 10) == 0:
                    save_img = True

                student_inputs, teacher_inputs, labels, datapoint_name = val_data
                student_inputs = student_inputs.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    val_loss = forward_pass(
                            student_model=student_model,
                            teacher_model=teacher_model,
                            student_inputs=student_inputs,
                            teacher_inputs=teacher_inputs,
                            labels=labels,
                            config=config,
                            split='valid',
                            writer=writer,
                            step_nbr=step_nbr)
                    total_val_loss += val_loss

                if config['hparam_search']['active'] and ((i+1) % config['hparam_search']['nbr_batches']) == 0:
                    hparam_name = config['hparam_search']['hparam_name']
                    writer.add_scalar(f'Search_Hparam/{hparam_name}-Train-Loss', train_loss, config['train_params']['LR'])
                    writer.add_scalar(f'Search_Hparam/{hparam_name}-Val-Loss', val_loss, config['train_params']['LR'])
                    return

            # pbar.set_description(f"{i}/{len(train_dataloader)} | Train Loss: {train_loss.item():.2f} | Val Loss: {val_loss.item():.2f}")
            pbar.set_description(f"{i}/{len(train_dataloader)} | Train Loss: {train_loss.item():.2f} | Val Loss: {val_loss.item():.2f}")
            pbar.update(1)
        print(f'Epoch: [{epoch + 1}/{num_epochs}] Total_Val_Loss: {total_val_loss / len(val_dataloader):.2f}')
    pbar.close()

    return (total_val_loss / len(val_dataloader)), (total_train_loss / len(train_dataloader))
