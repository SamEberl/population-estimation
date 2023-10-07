import random
import datetime
import albumentations as A
from torch import optim
from torch.utils.data import DataLoader
from dataset import studentTeacherDataset
from tensorboardX import SummaryWriter
from utils import *
from datetime import datetime
from logging_utils import *
from torch.optim.lr_scheduler import CosineAnnealingLR


def forward_pass(student_model,
                 teacher_model,
                 use_teacher,
                 data,
                 split,
                 device,
                 writer,
                 supervised_loss_name,
                 step_nbr,
                 save_img=False):
    student_inputs, teacher_inputs, labels, datapoint_name = data
    student_inputs = student_inputs.to(device)
    labels = labels.to(device)
    if split == 'train':
        student_model.train()
    else:
        student_model.eval()

    student_preds, student_features = student_model(student_inputs)
    supervised_loss = student_model.loss_supervised(student_preds, labels)

    writer.add_scalar(f'Loss-{supervised_loss_name}/{split}', supervised_loss.item(), step_nbr)
    loss_mae = torch.nn.functional.l1_loss(student_preds, labels)
    writer.add_scalar(f'Loss-L1-Compare/{split}', loss_mae, step_nbr)

    unsupervised_loss = 0
    if use_teacher and split == 'train':
        teacher_inputs = teacher_inputs.to(device)
        teacher_preds, teacher_features = teacher_model(teacher_inputs)
        unsupervised_loss = student_model.unsupervised_loss(student_features, teacher_features)
        #TODO scale unsupervised_loss to be similar to supervised_loss

        writer.add_scalar(f'Loss-{unsupervised_loss}/{split}', unsupervised_loss.item(), step_nbr)

    loss = supervised_loss + unsupervised_loss

    if save_img:
        #sample_nbr = random.randint(0, len(student_inputs[:, 0, 0, 0]-1))
        log_images_scores_to_tensorboard(writer, step_nbr, student_inputs[0, :3, :, :])
        log_regression_plot_to_tensorboard(writer, step_nbr, labels.cpu().flatten(),
                                           student_preds.cpu().flatten())

    return loss


def batch_generator(dataloader):
    dataloader_iter = iter(dataloader)
    while True:
        try:
            yield next(dataloader_iter)
        except StopIteration:
            yield None


def train_fix_match(config, log_dir, student_model, teacher_model):
    current_datetime = datetime.now().strftime("%Y%m%d%H%M%S")
    print(f'starting training at {current_datetime}')
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_path = config["data_params"]["data_path"]
    train_bs = config["data_params"]["train_batch_size"]
    #val_bs = config["data_params"]["val_batch_size"]
    num_workers = config["data_params"]["num_workers"]

    ema_alpha = config["train_params"]["ema_alpha"]  # Exponential moving average decay factor
    num_epochs = config['train_params']['max_epochs']
    supervised_loss_name = config['model_params']['supervised_criterion']
    use_teacher = config['train_params']['use_teacher']

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

    train_dataset = studentTeacherDataset(data_path, split='train', use_teacher=use_teacher, student_transform=student_transform, teacher_transform=teacher_transform)
    val_dataset = studentTeacherDataset(data_path, split='test', use_teacher=use_teacher, student_transform=student_transform, teacher_transform=teacher_transform)

    # Use adapted val batch sizes to accommodate different amounts of data
    data_ratio = len(train_dataset) / len(val_dataset)

    train_dataloader = DataLoader(train_dataset, batch_size=train_bs, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=int(train_bs//data_ratio), shuffle=True, num_workers=num_workers, pin_memory=True)

    # Remove dropout from teacher
    teacher_model.eval()
    # Make sure no grad is calculated for teacher
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

    pbar = tqdm(total=config['train_params']['max_epochs'], ncols=120)

    # Train the model
    for epoch in range(num_epochs):
        val_generator = batch_generator(val_dataloader)
        total_val_loss = 0
        for i, train_data in enumerate(train_dataloader):
            step_nbr = epoch * len(train_dataloader) + i

            train_loss = forward_pass(
                student_model=student_model,
                teacher_model=teacher_model,
                use_teacher=use_teacher,
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
            scheduler.step()

            # Update teacher model using exponential moving average
            # for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
            #     teacher_param.data.mul_(ema_alpha).add_(student_param.data * (1 - ema_alpha))

            if i % 10 == 0:
                print(f'\n Epoch: [{epoch + 1}/{num_epochs}] - {i} - Train_Loss: {train_loss.item():.3f}')

            val_data = next(val_generator)
            if val_data is not None:
                step_nbr = epoch * len(val_dataloader) + i

                save_img = False
                if step_nbr % (num_epochs * len(val_dataloader) / 10) == 0:
                    save_img = True

                with torch.no_grad():
                    val_loss = forward_pass(
                        student_model=student_model,
                        teacher_model=None,
                        use_teacher=False,
                        data=val_data,
                        split='valid',
                        device=device,
                        writer=writer,
                        supervised_loss_name=supervised_loss_name,
                        step_nbr=step_nbr,
                        save_img=save_img)
                    total_val_loss += val_loss

            pbar.set_description(f"Train Loss: {train_loss.item():.4f} | Val Loss: {val_loss.item():.4f}")
            pbar.update(1)

        print(f'Epoch: [{epoch + 1}/{num_epochs}] Val_Loss: {total_val_loss / len(val_dataloader):.3f}')

        # if epoch % 10 == 0:
        #     torch.save(reg_model.state_dict(),
        #                os.path.join(reg_config['logging_params']['save_dir'],
        #                             'ep_' + str(epoch) + '_' + reg_config['logging_params']['name']))

    # Close the SummaryWriter after training
    writer.close()
    pbar.close()

    # TODO: give each model unique name
    save_path = os.path.join(config['logging_params']['save_dir'],
                             f"{config['model_params']['model_name']}_{current_datetime}.pt")
    print(f'saving model under {save_path}')
    torch.save(student_model.state_dict(), save_path)



