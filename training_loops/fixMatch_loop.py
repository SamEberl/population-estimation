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
from models.losses import maskedBias, maskedL1Loss, maskedMSELoss, maskedRMSELoss

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
    supervised_loss = torch.tensor(0, dtype=torch.float32)
    r2_numerator = torch.tensor(0, dtype=torch.float32)
    bias = torch.tensor(0, dtype=torch.float32)
    if torch.cuda.is_available():
        supervised_loss = supervised_loss.cuda()
    if torch.sum(mask_labeled) > 0:
        supervised_loss = student_model.loss_supervised_w_uncertainty(student_preds, labels, student_data_uncertainty)
        r2_numerator = maskedMSELoss(student_preds, labels)
        bias = maskedBias(student_preds, labels)
        # total_step = 140 * 119794 + (0 + 1) * 256
        # supervised_loss = student_model.loss_supervised_w_uncertainty_decay(student_preds, labels, student_data_uncertainty, step_nbr, total_step)

    if not hparam_search:
        supervised_loss_name = config['model_params']['supervised_criterion']
        if supervised_loss != -1:
            writer.add_scalar(f'Loss-{supervised_loss_name}/{split}', supervised_loss, step_nbr)
        # loss_mae = torch.nn.functional.l1_loss(student_preds, labels)
        loss_mae = maskedL1Loss(student_preds, labels)
        loss_rmse = maskedRMSELoss(student_preds, labels)
        if loss_mae != -1:
            writer.add_scalar(f'Loss-L1-Compare/{split}', loss_mae, step_nbr)
            writer.add_scalar(f'Loss-RMSE-Compare/{split}', loss_rmse, step_nbr)
        writer.add_scalar(f'Percentage-Labeled/{split}', torch.sum(mask_labeled)/len(mask_labeled), step_nbr)

        # print(f'supervised_loss: {supervised_loss}')
        # print(f'L1-Compare: {loss_mae}')
        # print(f'Percentage: {torch.sum(mask_labeled)/len(mask_labeled)}')

    unsupervised_loss = torch.tensor(0, dtype=torch.float32)
    if torch.cuda.is_available():
        unsupervised_loss = unsupervised_loss.cuda()
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
        writer.add_scalar(f'Teacher_feature_euclidean/{split}', torch.mean(euclidean_distances), step_nbr)

        # pseudo_label_mask = (np.sqrt(teacher_model_uncertainty) / n_teacher_preds.mean(dim=0)) > 0.15  # Use CV as threshold
        pseudo_label_mask = teacher_features_spread < 1.0
        writer.add_scalar(f'Percentage-used-unsupervised', torch.sum(pseudo_label_mask)/len(pseudo_label_mask), step_nbr)
        #print(f'Percentage-used-unsupervised: {torch.sum(pseudo_label_mask)/len(pseudo_label_mask)}')
        # pseudo_label_mask = teacher_data_uncertainty < ?
        if torch.sum(pseudo_label_mask) > 0:
            dearanged_teacher_features = derangement_shuffle(teacher_features)
            unsupervised_loss = student_model.loss_unsupervised(student_features, teacher_features, dearanged_teacher_features, pseudo_label_mask)
        writer.add_scalar(f'Loss-Unsupervised/{split}', unsupervised_loss.item(), step_nbr)

        check2 = False
        if check2:  # TODO unsupervised loss on prediction instead of features
            pass
            # teacher_preds_loss = (teacher_preds_mean - labels)**2

    loss = supervised_loss + unsupervised_loss
    # print(f'sup: {supervised_loss}')
    # print(f'unsup: {unsupervised_loss}')
    # print(f'loss: {loss}')
    writer.add_scalar(f'Loss-Total/{split}', loss, step_nbr)

    # if save_img:
    #     #sample_nbr = random.randint(0, len(student_inputs[:, 0, 0, 0]-1))
    #     writer.add_images(f'Panel_Images', student_inputs[0, :3, :, :], global_step=step_nbr, dataformats='CHW')
    #     # log_regression_plot_to_tensorboard(writer, step_nbr, labels.cpu().flatten(), student_preds.cpu().flatten())

    return loss, r2_numerator, bias


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
    drop_labels = config['data_params']['drop_labels']
    seed = config['train_params']['seed']
    percentage_unlabeled = config['data_params']['percentage_unlabeled']
    nbr_channels = config['model_params']['in_channels']

    train_dataset = studentTeacherDataset(data_path, split='train', use_teacher=use_teacher, drop_labels=drop_labels, student_transform=student_transform, teacher_transform=teacher_transform, percentage_unlabeled=percentage_unlabeled, nbr_channels=nbr_channels)
    val_dataset = studentTeacherDataset(data_path, split='test', use_teacher=use_teacher, drop_labels=drop_labels, student_transform=None, teacher_transform=None, percentage_unlabeled=percentage_unlabeled, nbr_channels=nbr_channels)

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
        r2_numerators_train = []
        r2_numerators_val = []
        biases_train = []
        biases_val = []
        #config['train_params']['use_teacher'] = (epoch == (num_epochs - 1))
        for i, train_data in enumerate(train_dataloader):
            step_nbr = epoch * len(train_dataloader.dataset) + (i + 1) * train_dataloader.batch_size

            student_inputs, teacher_inputs, labels, datapoint_name = train_data
            student_inputs = student_inputs.to(device)
            if use_teacher:
                teacher_inputs = teacher_inputs.to(device)
            labels = labels.to(device)

            train_loss, r2_numerator_train, bias_train = forward_pass(
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
            r2_numerators_train.append(r2_numerator_train)
            biases_train.append(bias_train)

            # Backward pass and optimization
            if train_loss != 0:
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
            scheduler.step()

            # Update teacher model using exponential moving average
            for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
                teacher_param.data.mul_(ema_alpha).add_(student_param.data * (1 - ema_alpha))

            val_data = next(val_generator)
            if val_data is not None:
                # step_nbr = epoch * len(val_dataloader.dataset) + (i+1) * val_dataloader.batch_size

                student_inputs, teacher_inputs, labels, datapoint_name = val_data
                student_inputs = student_inputs.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    val_loss, r2_numerator_val, bias_val = forward_pass(
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
                    r2_numerators_val.append(r2_numerator_val)
                    biases_val.append(bias_val)

                if config['hparam_search']['active'] and ((i+1) % config['hparam_search']['nbr_batches']) == 0:
                    hparam_name = config['hparam_search']['hparam_name']
                    writer.add_scalar(f'Search_Hparam/{hparam_name}-Train-Loss', train_loss, config['train_params']['LR'])
                    writer.add_scalar(f'Search_Hparam/{hparam_name}-Val-Loss', val_loss, config['train_params']['LR'])
                    return
            pbar.set_description(f"Train Loss: {train_loss.item():.2f} | Val Loss: {val_loss.item():.2f}")
            pbar.update(1)
        print(f'  Epoch: [{epoch + 1}/{num_epochs}] Total_Val_Loss: {(total_val_loss.item() / len(val_dataloader)):.2f}')
        writer.add_scalar(f'R2/train', calc_r2(r2_numerators_train, 'train'), epoch)
        writer.add_scalar(f'R2/val', calc_r2(r2_numerators_val, 'val'), epoch)
        writer.add_scalar(f'Bias/train', calc_bias(biases_train), epoch)
        writer.add_scalar(f'Bias/val', calc_bias(biases_val), epoch)
    pbar.close()

    return (total_val_loss / len(val_dataloader)), (total_train_loss / len(train_dataloader))
