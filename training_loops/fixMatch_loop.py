import random
import datetime
import albumentations as A
import torch
from torch import optim
from dataset import studentTeacherDataset
from tensorboardX import SummaryWriter
from utils import *
from datetime import datetime
from logging_utils import *
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from models.losses import maskedBias, maskedL1Loss, maskedMSELoss, maskedRMSELoss
from MetricsLogger import MetricsLogger

import matplotlib.pyplot as plt


def forward_pass(student_model,
                 teacher_model,
                 student_inputs,
                 teacher_inputs,
                 labels,
                 config,
                 split,
                 logger):
    if split == 'train':
        student_model.train()
    else:
        student_model.eval()

    for i in range(len(student_inputs)):
        student_path = f'/home/sameberl/img_logs/student_{i}.png'
        student_array = student_inputs[i, :, :, :].permute(1, 2, 0)
        if student_array.is_cuda:
            student_array = student_array.cpu()
        student_array = student_array.numpy()
        plt.imsave(student_path, student_array)

        teacher_path = f'/home/sameberl/img_logs/teacher_{i}.png'
        teacher_array = teacher_inputs[i, :, :, :].permute(1, 2, 0)
        if teacher_array.is_cuda:
            teacher_array = teacher_array.cpu()
        teacher_array = teacher_array.numpy()
        plt.imsave(teacher_path, teacher_array)
    print('saved student and teacher imgs')
    exit()

    student_preds, student_features, student_data_uncertainty = student_model(student_inputs)
    mask_labeled = labels != -1
    supervised_loss = torch.tensor(0, dtype=torch.float32)
    if torch.cuda.is_available():
        supervised_loss = supervised_loss.cuda()
    if torch.sum(mask_labeled) > 0:
        supervised_loss = student_model.loss_supervised(student_preds, labels)
        # supervised_loss = student_model.loss_supervised_w_uncertainty(student_preds, labels, student_data_uncertainty)
        r2_numerator = maskedMSELoss(student_preds, labels)
        logger.add_metric('Observe-R2', split, r2_numerator)
        bias = maskedBias(student_preds, labels)
        logger.add_metric('Observe-Bias', split, bias)
        # supervised_loss = student_model.loss_supervised_w_uncertainty_decay(student_preds, labels, student_data_uncertainty, step_nbr, total_step)

    supervised_loss_name = config['model_params']['supervised_criterion']
    if supervised_loss != -1:
        logger.add_metric(f'Loss-Supervised-{supervised_loss_name}', split, supervised_loss)
    # loss_mae = torch.nn.functional.l1_loss(student_preds, labels)
    loss_mae = maskedL1Loss(student_preds, labels)
    loss_rmse = maskedRMSELoss(student_preds, labels)
    if loss_mae != -1:
        logger.add_metric('Loss-Compare-L1', split, loss_mae)
        logger.add_metric('Loss-Compare-RMSE', split, loss_rmse)
    logger.add_metric('Observe-Percent-Labeled', split, torch.sum(mask_labeled)/len(mask_labeled))

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
        l2_distances = torch.sqrt(((n_teacher_features - teacher_features_mean) ** 2).sum(dim=-1))  # Calculate the squared differences
        l2_distances_var = l2_distances.var(dim=0)  # Get spread

        # Compute data uncertainty
        teacher_data_uncertainty = n_teacher_data_uncertainties.var(dim=0)

        logger.add_metric('Uncertainty_Var_Regression', split, torch.mean(teacher_model_uncertainty))
        logger.add_metric('Uncertainty-Features-L2-Var', torch.mean(l2_distances_var))
        logger.add_metric('Uncertainty_Predicted', split, torch.mean(teacher_data_uncertainty))
        logger.add_metric('Uncertainty-Features-L2', split, torch.mean(l2_distances))

        # pseudo_label_mask = (np.sqrt(teacher_model_uncertainty) / n_teacher_preds.mean(dim=0)) > 0.15  # Use CV as threshold
        pseudo_label_mask = l2_distances_var < 0.5
        logger.add_metric(f'Observe-Percent-used-unsupervised', split, torch.sum(pseudo_label_mask)/len(pseudo_label_mask))
        # pseudo_label_mask = teacher_data_uncertainty < ?
        if torch.sum(pseudo_label_mask) > 0:
            dearanged_teacher_features = derangement_shuffle(teacher_features_mean)
            unsupervised_loss = student_model.loss_unsupervised(student_features, teacher_features_mean, dearanged_teacher_features, pseudo_label_mask)
        logger.add_metric(f'Loss-Unsupervised', split, unsupervised_loss.item())

    loss = supervised_loss + unsupervised_loss
    logger.add_metric('Loss-All', split, loss)

    # if save_img:
    #     #sample_nbr = random.randint(0, len(student_inputs[:, 0, 0, 0]-1))
    #     writer.add_images(f'Panel_Images', student_inputs[0, :3, :, :], global_step=step_nbr, dataformats='CHW')
    #     # log_regression_plot_to_tensorboard(writer, step_nbr, labels.cpu().flatten(), student_preds.cpu().flatten())

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


def train_fix_match(config, writer, student_model, teacher_model, train_dataloader, val_dataloader):
    logger = MetricsLogger(writer)

    # Get params from config
    ema_alpha = config["train_params"]["ema_alpha"]  # Exponential moving average decay factor
    num_epochs = config['train_params']['max_epochs']
    use_teacher = config['train_params']['use_teacher']
    info = config['info']['info']

    optimizer = optim.AdamW(student_model.parameters(),
                            lr=config['train_params']['LR'],
                            betas=(config['train_params']['beta1'], config['train_params']['beta2']),
                            weight_decay=config['train_params']['L2_reg'] * 2)
    # scheduler = CosineAnnealingLR(optimizer, T_max=len(train_dataloader)*num_epochs, eta_min=0.00001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    pbar = tqdm(total=len(train_dataloader), ncols=140)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    total_train_loss = 0
    total_val_loss = 0
    # Train the model
    for epoch in range(num_epochs):
        val_generator = batch_generator(val_dataloader)
        total_train_loss = 0
        total_val_loss = 0
        writer.add_scalar(f'Observe-LR', optimizer.defaults['lr'], epoch)
        for i, train_data in enumerate(train_dataloader):
            # step_nbr = epoch * len(train_dataloader.dataset) + (i + 1) * train_dataloader.batch_size

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
                logger=logger)
            total_train_loss += train_loss

            # Backward pass and optimization
            if train_loss != 0:
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

            # Update teacher model using exponential moving average
            for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
                teacher_param.data.mul_(ema_alpha).add_(student_param.data * (1 - ema_alpha))

            val_data = next(val_generator)
            if val_data is not None:
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
                            logger=logger)
                    total_val_loss += val_loss

            if i % 10 == 0:
                pbar.set_description(f"Epoch: [{epoch + 1}/{num_epochs}] | Info: {info}")
                pbar.update(1)
        scheduler.step(total_val_loss/len(val_dataloader))
        logger.write(epoch+1)
        # writer.add_scalar(f'R2/train', calc_r2(r2_numerators_train, 'train'), epoch)
        # writer.add_scalar(f'R2/val', calc_r2(r2_numerators_val, 'val'), epoch)
        # writer.add_scalar(f'Bias/train', calc_bias(biases_train), epoch)
        # writer.add_scalar(f'Bias/val', calc_bias(biases_val), epoch)
    pbar.close()

    return (total_val_loss / len(val_dataloader)), (total_train_loss / len(train_dataloader))
