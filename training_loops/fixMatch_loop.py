import torch
import random
import yaml
import numpy as np
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau  # CosineAnnealingLR
from models.losses import CalcBias, MaskedL1Loss, MaskedRMSELoss, MaskedMSELoss
from MetricsLogger import MetricsLogger
from UncertaintyJudge import UncertaintyJudge
from datetime import datetime
from sklearn.utils import shuffle
from dataset import normalize_labels, unnormalize_preds


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
    # Pass inputs through model
    student_preds, student_features, student_data_uncertainty = student_model(student_inputs)

    labels_normalized = normalize_labels(labels)

    # Calc Supervised Loss
    # supervised_loss = student_model.loss_supervised(student_preds, labels)
    # uncertainty_loss = student_model.loss_uncertainty(student_preds, labels, student_data_uncertainty)
    supervised_loss = student_model.loss_supervised_w_uncertainty(student_preds, labels_normalized,
                                                                  student_data_uncertainty)
    # supervised_loss = student_model.loss_supervised_w_uncertainty_decay(student_preds, labels_normalized, student_data_uncertainty, step_nbr, total_step)

    student_preds = unnormalize_preds(student_preds)

    # Log Metrics
    logger.add_metric(f'Loss-Supervised-{supervised_loss_name}', split, supervised_loss)
    logger.add_metric('Observe-R2', split, F.mse_loss(student_preds, labels, reduction='mean'))
    logger.add_metric('Observe-Bias', split, CalcBias.forward(student_preds, labels))
    logger.add_metric('Loss-Compare-L1', split, F.l1_loss(student_preds, labels, reduction='mean'))  # loss_mae = torch.nn.functional.l1_loss(student_preds, labels)
    logger.add_metric('Loss-Compare-RMSE', split, torch.sqrt(F.mse_loss(student_preds, labels, reduction='mean')))
    logger.add_metric('Uncertainty_Predicted', split, torch.mean(student_data_uncertainty))
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
                         judge,
                         logger):
    if judge.use_judge and judge.threshold_func is None:
        # In the first epoch don't do SSL just get threshold func
        teacher_preds, teacher_features, teacher_data_uncertainty = teacher_model(teacher_inputs)
        judge.add_pred_var_pair(teacher_preds, teacher_data_uncertainty)
        return None
    else:
        student_preds, student_features, student_data_uncertainty = student_model(student_inputs)
        teacher_preds, teacher_features, teacher_data_uncertainty = teacher_model(teacher_inputs)
        if judge.use_judge:
            judge.add_pred_var_pair(teacher_preds, teacher_data_uncertainty)
            pseudo_label_mask = judge.evaluate_threshold_func(teacher_preds, teacher_data_uncertainty)
        else:
            pseudo_label_mask = torch.ones(student_inputs.shape[0]).to(student_inputs.device)

        if torch.sum(pseudo_label_mask) > 0:
            dearanged_teacher_features = derangement_shuffle(teacher_features)
            unsupervised_loss = student_model.loss_unsupervised(student_features, teacher_features,
                                                                dearanged_teacher_features, mask=pseudo_label_mask)
            logger.add_metric(f'Loss-Unsupervised', 'train', unsupervised_loss.item())
            return unsupervised_loss
        else:
            return None


def train_fix_match(config, writer, student_model, teacher_model, train_dataloader, valid_dataloader,
                    train_dataloader_unlabeled):
    logger = MetricsLogger(writer)
    judge = UncertaintyJudge(config["train_params"]["use_judge"])

    # Get params from config
    ema_alpha = config["train_params"]["ema_alpha"]  # Exponential moving average decay factor
    num_epochs = config['train_params']['max_epochs']
    unlabeled_data = config['data_params']['unlabeled_data']
    # num_samples_teacher = config['train_params']['num_samples_teacher']
    info = config['info']['info']

    supervised_loss_name = config['model_params']['supervised_criterion']

    optimizer = optim.AdamW(student_model.parameters(),
                            lr=config['train_params']['LR'],
                            betas=(config['train_params']['beta1'], config['train_params']['beta2']),
                            weight_decay=config['train_params']['L2_reg'] * 2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Train the model
    for epoch in range(num_epochs):
        print(f"\nStart Epoch: [{epoch + 1}/{num_epochs}] | {datetime.now().strftime('%H:%M:%S')} | {info}")
        if epoch == (num_epochs - 1):
            logger.last_epoch = True

        for train_data in train_dataloader:
            inputs, labels = train_data
            inputs = inputs.to(device)
            labels = labels.to(device)
            student_model.train()
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

        print(f"Start Valid: [{epoch + 1}/{num_epochs}] | {datetime.now().strftime('%H:%M:%S')} | {info}")
        for i, valid_data in enumerate(valid_dataloader):
            inputs, labels = valid_data
            inputs = inputs.to(device)
            labels = labels.to(device)
            student_model.eval()
            with torch.no_grad():
                val_loss = forward_supervised(
                        student_model=student_model,
                        student_inputs=inputs,
                        labels=labels,
                        supervised_loss_name=supervised_loss_name,
                        split='valid',
                        logger=logger)

        if unlabeled_data:
            print(f"Start SSL  : [{epoch + 1}/{num_epochs}] | {datetime.now().strftime('%H:%M:%S')} | {info}")
            # Update teacher model using exponential moving average
            for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
                teacher_param.data.mul_(ema_alpha).add_(student_param.data * (1 - ema_alpha))
            for train_data_unlabeled in train_dataloader_unlabeled:
                student_model.train()
                teacher_model.eval()
                inputs_ssl, inputs_ssl_transformed = train_data_unlabeled
                inputs_ssl = inputs_ssl.to(device)
                inputs_ssl_transformed = inputs_ssl_transformed.to(device)
                unsupervised_loss = forward_unsupervised(student_model,
                                                         teacher_model,
                                                         student_inputs=inputs_ssl_transformed,
                                                         teacher_inputs=inputs_ssl,
                                                         judge=judge,
                                                         logger=logger)
                if unsupervised_loss is not None:
                    optimizer.zero_grad()
                    unsupervised_loss.backward()
                    optimizer.step()
            if judge.use_judge:
                judge.calc_threshold_func()

        logger.write(epoch + 1)
        if epoch == 0:
            param_yaml_str = yaml.dump(config, default_flow_style=False)
            param_yaml_str = param_yaml_str.replace('\n', '<br>')
            writer.add_text('Parameters', param_yaml_str, 0)
        if logger.last_epoch:
            logger.save_uncertainties(config)
            logger.print_final_stats()
            logger.print_final_stats_latex()
        logger.clear()
