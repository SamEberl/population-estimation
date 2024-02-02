import torch
import random
import statistics
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau  # CosineAnnealingLR
from models.losses import CalcBias, MaskedL1Loss, MaskedRMSELoss, MaskedMSELoss
from MetricsLogger import MetricsLogger
from tqdm import tqdm
from dataset import normalize_labels, unnormalize_preds, quantile_normalize_labels


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

    #labels_normalized = normalize_labels(labels)

    # Calc Supervised Loss
    supervised_loss = student_model.loss_supervised(student_preds, labels)
    # uncertainty_loss = student_model.loss_uncertainty(student_preds, labels_normalized, student_data_uncertainty)
    # supervised_loss = student_model.loss_supervised_w_uncertainty(student_preds, labels_normalized, student_data_uncertainty)
    # supervised_loss = student_model.loss_supervised_w_uncertainty_decay(student_preds, labels_normalized, student_data_uncertainty, step_nbr, total_step)

    #student_preds = unnormalize_preds(student_preds)

    # Log Metrics
    logger.add_metric(f'Loss-Supervised-{supervised_loss_name}', split, supervised_loss)
    logger.add_metric('Observe-R2', split, F.mse_loss(student_preds, labels, reduction='mean'))
    logger.add_metric('Observe-Bias', split, CalcBias.forward(student_preds, labels))
    logger.add_metric('Loss-Compare-L1', split, F.l1_loss(student_preds, labels, reduction='mean'))  # loss_mae = torch.nn.functional.l1_loss(student_preds, labels)
    logger.add_metric('Loss-Compare-RMSE', split, torch.sqrt(F.mse_loss(student_preds, labels, reduction='mean')))
    # logger.add_metric('Loss-Uncertainty', split, uncertainty_loss)
    # logger.add_metric('Observe-Percent-Labeled', split, torch.sum(mask_labeled) / len(mask_labeled))

    # if True:
    #     image_tensor = student_inputs[0, :3, :, :]
    #     if image_tensor.is_cuda:
    #         image_tensor = image_tensor.cpu()
    #     to_pil = ToPILImage()
    #     image = to_pil(image_tensor)
    #     rand_int = random.randint(0, 100)
    #     image.save(f"/home/sameberl/img_logs/output_image_{rand_int}.png")

    return supervised_loss


def forward_unsupervised(student_model,
                         teacher_model,
                         student_inputs,
                         teacher_inputs,
                         num_samples_teacher,
                         labels,
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
    teacher_data_uncertainty = n_teacher_data_uncertainties.mean(dim=0)  # TODO: before var() but should probably be mean

    # TODO: Use pseudo_label_mask
    # pseudo_label_mask = (np.sqrt(teacher_model_uncertainty) / n_teacher_preds.mean(dim=0)) > 0.15  # Use CV as threshold
    # pseudo_label_mask = l2_distances_var < 0.5
    # pseudo_label_mask = teacher_data_uncertainty < ?
    # if torch.sum(pseudo_label_mask) > 0:
    pseudo_label_mask = None
    dearanged_teacher_features = derangement_shuffle(teacher_features_mean)
    unsupervised_loss = student_model.loss_unsupervised(student_features, teacher_features_mean, dearanged_teacher_features, mask=pseudo_label_mask)

    split = 'train'
    logger.add_metric('Uncertainty_Predicted', split, torch.mean(teacher_data_uncertainty))
    logger.add_metric('Uncertainty_Calculated', split, torch.mean(teacher_model_uncertainty))
    logger.add_metric('Uncertainty-Features-L2-Var', split, torch.mean(l2_distances_var))
    logger.add_metric('Uncertainty-Features-L2', split, torch.mean(l2_distances))
    # logger.add_metric(f'Observe-Percent-used-unsupervised', split, torch.sum(pseudo_label_mask)/len(pseudo_label_mask))
    logger.add_metric(f'Loss-Unsupervised', split, unsupervised_loss.item())

    # # Uncertainties to log
    predictions = n_teacher_preds.mean(dim=0)
    logger.add_uncertainty('pred', predictions)
    logger.add_uncertainty('label', labels)
    logger.add_uncertainty('loss', student_model.loss_supervised(predictions, labels))
    logger.add_uncertainty('pred_var', teacher_data_uncertainty)
    logger.add_uncertainty('calc_var', teacher_model_uncertainty)
    logger.add_uncertainty('features_l2', l2_distances_var)

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
        print(f"\n Start Epoch: [{epoch + 1}/{num_epochs}] | {info}")
        for train_data in tqdm(train_dataloader):
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

        # TODO: Interweave unsupervised training with supervised training
        if unlabeled_data:
            for train_data_unlabeled in tqdm(train_dataloader_unlabeled):
                inputs, inputs_transformed, labels = train_data_unlabeled
                inputs = inputs.to(device)
                inputs_transformed = inputs_transformed.to(device)
                unsupervised_loss = forward_unsupervised(student_model,
                                                         teacher_model,
                                                         student_inputs=inputs_transformed,
                                                         teacher_inputs=inputs,
                                                         num_samples_teacher=num_samples_teacher,
                                                         labels=labels,
                                                         logger=logger)

                # Backward pass and optimization
                optimizer.zero_grad()
                unsupervised_loss.backward()
                optimizer.step()

        for i, valid_data in tqdm(enumerate(valid_dataloader)):
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

        #writer.add_scalar(f'Observe-LR', optimizer.defaults['lr'], epoch)
        scheduler.step(statistics.mean(logger.metrics[f'Loss-Supervised-{supervised_loss_name}/train']))
        logger.write(epoch+1)
        logger.save_uncertainties()
        logger.clear()
    pbar.close()
