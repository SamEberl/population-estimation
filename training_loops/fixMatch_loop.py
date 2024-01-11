import torch
import random
import statistics
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau  # CosineAnnealingLR
from models.losses import MaskedBias, MaskedL1Loss, MaskedRMSELoss, MaskedMSELoss
from MetricsLogger import MetricsLogger
from tqdm import tqdm


def derangement_shuffle(tensor):
    """
    Shuffle a tensor such that no element remains in its original position.

    :param tensor: A PyTorch tensor to be shuffled.
    :return: A shuffled tensor.
    """
    n = tensor.size(0)
    indices = list(range(n))
    while any(i == indices[i] for i in range(n)):
        random.shuffle(indices)

    return tensor[torch.tensor(indices)]


def forward_supervised(student_model,
                       student_inputs,
                       labels,
                       supervised_loss_name,
                       split,
                       logger):
    # Set mode to disable dropout for eval
    if split == 'train':
        student_model.train()
    else:
        student_model.eval()

    # Pass inputs through model
    student_preds, student_features, student_data_uncertainty = student_model(student_inputs)

    # Calc Supervised Loss
    supervised_loss = student_model.loss_supervised(student_preds, labels)
    # supervised_loss = student_model.loss_supervised_w_uncertainty(student_preds, labels, student_data_uncertainty)
    # supervised_loss = student_model.loss_supervised_w_uncertainty_decay(student_preds, labels, student_data_uncertainty, step_nbr, total_step)

    # Log Metrics
    logger.add_metric(f'Loss-Supervised-{supervised_loss_name}', split, supervised_loss)
    logger.add_metric('Observe-R2', split, MaskedMSELoss.forward(student_preds, labels))  # TODO: Switch out all the losses for non masked
    logger.add_metric('Observe-Bias', split, MaskedBias.forward(student_preds, labels))
    logger.add_metric('Loss-Compare-L1', split, MaskedL1Loss.forward(student_preds, labels))  # loss_mae = torch.nn.functional.l1_loss(student_preds, labels)
    logger.add_metric('Loss-Compare-RMSE', split, MaskedRMSELoss.forward(student_preds, labels))
    # logger.add_metric('Observe-Percent-Labeled', split, torch.sum(mask_labeled) / len(mask_labeled))

    return supervised_loss



def forward_unsupervised(student_model,
                         teacher_model,
                         student_inputs,
                         teacher_inputs,
                         num_samples_teacher,
                         logger):

    student_model.train()
    # Pass inputs through model
    student_preds, student_features, student_data_uncertainty = student_model(student_inputs)

    # Store all predictions
    n_teacher_preds = []
    n_teacher_features = []
    n_teacher_data_uncertainties = []

    teacher_model.train()  # Ensure dropout is active during evaluation
    with torch.no_grad():  # Ensure no gradients are computed
        for _ in range(num_samples_teacher):
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

    # TODO: Use pseudo_label_mask
    # pseudo_label_mask = (np.sqrt(teacher_model_uncertainty) / n_teacher_preds.mean(dim=0)) > 0.15  # Use CV as threshold
    # pseudo_label_mask = l2_distances_var < 0.5
    # pseudo_label_mask = teacher_data_uncertainty < ?
    # if torch.sum(pseudo_label_mask) > 0:
    pseudo_label_mask = None
    dearanged_teacher_features = derangement_shuffle(teacher_features_mean)
    unsupervised_loss = student_model.loss_unsupervised(student_features, teacher_features_mean, dearanged_teacher_features, mask=pseudo_label_mask)

    split = 'train'
    logger.add_metric('Uncertainty_Var_Regression', split, torch.mean(teacher_model_uncertainty))
    logger.add_metric('Uncertainty-Features-L2-Var', split, torch.mean(l2_distances_var))
    logger.add_metric('Uncertainty_Predicted', split, torch.mean(teacher_data_uncertainty))
    logger.add_metric('Uncertainty-Features-L2', split, torch.mean(l2_distances))
    logger.add_metric(f'Observe-Percent-used-unsupervised', split, torch.sum(pseudo_label_mask)/len(pseudo_label_mask))
    logger.add_metric(f'Loss-Unsupervised', split, unsupervised_loss.item())

    # if save_img:
    #     #sample_nbr = random.randint(0, len(student_inputs[:, 0, 0, 0]-1))
    #     writer.add_images(f'Panel_Images', student_inputs[0, :3, :, :], global_step=step_nbr, dataformats='CHW')
    #     # log_regression_plot_to_tensorboard(writer, step_nbr, labels.cpu().flatten(), student_preds.cpu().flatten())

    return unsupervised_loss


def train_fix_match(config, writer, student_model, teacher_model, train_dataloader, valid_dataloader, train_dataloader_unlabeled):
    logger = MetricsLogger(writer)

    # Get params from config
    ema_alpha = config["train_params"]["ema_alpha"]  # Exponential moving average decay factor
    num_epochs = config['train_params']['max_epochs']
    unlabeled_data = config['data_params']['unlabeled_data']
    num_samples_teacher = config['train_params']['num_samples_teacher']
    info = config['info']['info']

    supervised_loss_name = config['model_params']['supervised_criterion']

    optimizer = optim.AdamW(student_model.parameters(),
                            lr=config['train_params']['LR'],
                            betas=(config['train_params']['beta1'], config['train_params']['beta2']),
                            weight_decay=config['train_params']['L2_reg'] * 2)
    # scheduler = CosineAnnealingLR(optimizer, T_max=len(train_dataloader)*num_epochs, eta_min=0.00001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    pbar = tqdm(total=len(train_dataloader), ncols=140)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Train the model
    for epoch in range(num_epochs):
        for train_data in (train_dataloader):
            inputs, labels = train_data
            inputs = inputs.to(device)
            labels = labels.to(device)

            supervised_loss = forward_supervised(student_model,
                                                 inputs,
                                                 labels,
                                                 supervised_loss_name,
                                                 split='train',
                                                 logger=logger)

            # Backward pass and optimization
            optimizer.zero_grad()
            supervised_loss.backward()
            optimizer.step()

            # Update teacher model using exponential moving average
            if unlabeled_data:
                for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
                    teacher_param.data.mul_(ema_alpha).add_(student_param.data * (1 - ema_alpha))

        if unlabeled_data:
            for train_data_unlabeled in train_dataloader_unlabeled:
                inputs, inputs_transformed = train_data_unlabeled
                inputs = inputs.to(device)
                inputs_transformed = inputs_transformed.to(device)
                unsupervised_loss = forward_unsupervised(student_model,
                                                         teacher_model,
                                                         student_inputs=inputs_transformed,
                                                         teacher_inputs=inputs,
                                                         num_samples_teacher=num_samples_teacher,
                                                         logger=logger)

                # Backward pass and optimization
                optimizer.zero_grad()
                unsupervised_loss.backward()
                optimizer.step()

        for i, valid_data in enumerate(valid_dataloader):
            inputs, labels = valid_data
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                val_loss = forward_supervised(
                        student_model=student_model,
                        student_inputs=inputs,
                        labels=labels,
                        supervised_loss_name=supervised_loss_name,
                        split='valid',
                        logger=logger)

            if i % 100 == 0:
                pbar.set_description(f"Epoch: [{epoch + 1}/{num_epochs}] | Info: {info}")
                pbar.update(1)

        writer.add_scalar(f'Observe-LR', optimizer.defaults['lr'], epoch)
        scheduler.step(statistics.mean(logger.metrics[f'Loss-Supervised-{supervised_loss_name}']))
        logger.write(epoch+1)
        logger.clear()
    pbar.close()
