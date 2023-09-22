import random
import albumentations as A
from itertools import zip_longest
from torch import optim
from torch.utils.data import DataLoader
from dataset import studentTeacherDataset
from tensorboardX import SummaryWriter
from utils import *
from datetime import datetime
from logging_utils import *
from torch.optim.lr_scheduler import CosineAnnealingLR


def forward_pass(model,
                 data,
                 split,
                 device,
                 writer,
                 supervised_loss_name,
                 step_nbr,
                 save_img=False):
    student_inputs, labels, datapoint_name = data
    student_inputs = student_inputs.to(device)
    # teacher_inputs = teacher_transform(inputs).to(device)
    labels = labels.to(device)

    student_preds, student_features = model(student_inputs)
    # with torch.no_grad():
    #     teacher_outputs = teacher_model(teacher_inputs)

    loss = model.loss_supervised(student_preds, labels)

    with torch.no_grad():
        writer.add_scalar(f'Loss-{supervised_loss_name}/{split}', loss.item(), step_nbr)
        loss_mae = torch.nn.functional.l1_loss(student_preds, labels)
        writer.add_scalar(f'Loss-L1-Compare/{split}', loss_mae, step_nbr)

    if save_img:
        sample_nbr = random.randint(0, len(student_inputs[:, 0, 0, 0]-1))
        log_images_scores_to_tensorboard(writer, step_nbr, student_inputs[sample_nbr, :3, :, :])
        log_regression_plot_to_tensorboard(writer, step_nbr, labels.cpu().flatten(),
                                           student_preds.cpu().flatten())

    return loss


def train_fix_match(config, log_dir, student_model, teacher_model):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_path = config["data_params"]["data_path"]
    train_bs = config["data_params"]["train_batch_size"]
    #val_bs = config["data_params"]["val_batch_size"]
    num_workers = config["data_params"]["num_workers"]

    ema_alpha = config["train_params"]["ema_alpha"]  # Exponential moving average decay factor
    num_epochs = config['train_params']['max_epochs']
    supervised_loss_name = config['model_params']['supervised_criterion']

    student_transforms_list = []
    teacher_transfroms_list = []
    # Loop through the dictionary and add augmentations to the list
    for student_params, teacher_params in zip(config['student_transforms'], config['teacher_transforms']):
        student_aug_fn = getattr(A, list(student_params.keys())[0])(**list(student_params.values())[0])
        student_transforms_list.append(student_aug_fn)
        teacher_aug_fn = getattr(A, list(teacher_params.keys())[0])(**list(teacher_params.values())[0])
        teacher_transfroms_list.append(teacher_aug_fn)
    # Create an augmentation pipeline using the list of augmentation functions
    student_transform = A.Compose(student_transforms_list)
    teacher_transform = A.Compose(teacher_transfroms_list)

    train_dataset = studentTeacherDataset(data_path, split='train', student_transform=student_transform, teacher_transform=teacher_transform)
    val_dataset = studentTeacherDataset(data_path, split='test', student_transform=student_transform, teacher_transform=teacher_transform)

    data_ratio = len(train_dataset) / len(val_dataset)

    train_dataloader = DataLoader(train_dataset, batch_size=train_bs, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=int(train_bs//data_ratio), shuffle=True, num_workers=num_workers)

    for param in teacher_model.model.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(student_model.parameters(),
                           lr=config['train_params']['LR'],
                           betas=(config['train_params']['beta1'], config['train_params']['beta2']),
                           weight_decay=config['train_params']['L2_reg'] * 2)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_dataloader)*num_epochs, eta_min=0.00001)

    # Create a SummaryWriter for TensorBoard
    writer = SummaryWriter(
        logdir=log_dir + config['model_params']['name'] + '_' + datetime.now().strftime("%Y%m%d_%H%M%S"))

    print(student_model.model.default_cfg)
    param_yaml_str = yaml.dump(config, default_flow_style=False)
    param_yaml_str = param_yaml_str.replace('\n', '<br>')
    writer.add_text('Parameters', param_yaml_str, 0)
    model_yaml_str = yaml.dump(student_model.model.default_cfg, default_flow_style=False)
    model_yaml_str = model_yaml_str.replace('false ', '<br>')
    writer.add_text('Model Specs', model_yaml_str, 0)


    # Train the model
    for epoch in range(num_epochs):
        total_val_loss = 0
        for i, (train_data, val_data) in enumerate(zip_longest(train_dataloader, val_dataloader)):
            step_nbr = epoch * len(train_dataloader) + i

            train_loss = forward_pass(
                model=student_model,
                data=train_data,
                split='train',
                device=device,
                writer=writer,
                supervised_loss_name=supervised_loss_name,
                step_nbr=step_nbr)

            # Backward pass and optimization
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            # Update teacher model using exponential moving average
            # for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
            #     teacher_param.data.mul_(ema_alpha).add_(student_param.data * (1 - ema_alpha))

            if i % 10 == 0:
                print(f'Epoch: [{epoch + 1}/{num_epochs}] - {i} - Train_Loss: {train_loss.item():.3f}')
            scheduler.step()

            if val_data is not None:
                step_nbr = epoch * len(val_dataloader) + i

                save_img = False
                if step_nbr % (num_epochs * len(val_dataloader) / 10) == 0:
                    save_img = True

                with torch.no_grad():
                    val_loss = forward_pass(
                        model=student_model,
                        data=val_data,
                        split='valid',
                        device=device,
                        writer=writer,
                        supervised_loss_name=supervised_loss_name,
                        step_nbr=step_nbr,
                        save_img=save_img)
                    total_val_loss += val_loss

        print(f'Epoch: [{epoch + 1}/{num_epochs}] Val_Loss: {total_val_loss / len(val_dataloader):.3f}')

        # if epoch % 10 == 0:
        #     torch.save(reg_model.state_dict(),
        #                os.path.join(reg_config['logging_params']['save_dir'],
        #                             'ep_' + str(epoch) + '_' + reg_config['logging_params']['name']))

    # Close the SummaryWriter after training
    writer.close()

    # TODO: give each model unique name
    torch.save(student_model.state_dict(),
               os.path.join(config['logging_params']['save_dir'], config['model_params']['model_name']))



