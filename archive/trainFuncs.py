import torch
from torch import optim
from torch.utils.data import DataLoader
from dataset import sslDataset, studentTeacherDataset
from models import *
from tensorboardX import SummaryWriter
from utils import *
from datetime import datetime
from logging_utils import *
import random
import torchvision.transforms as transforms



def train_ae(ae_config, log_dir):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set to utilize tensor cores of GPU
    torch.set_float32_matmul_precision('medium')

    data = sslDataset(**ae_config["data_params"])
    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()

    pbar = tqdm(total=ae_config['train_params']['max_epochs'], ncols=120)

    # Initialize the model
    ae_model = ssl_models[ae_config['model_params']['name']](**ae_config['model_params'])
    ae_model.to(device)

    # Define the optimizer
    optimizer = optim.Adam(ae_model.parameters(),
                           lr=ae_config['train_params']['LR'],
                           betas=(ae_config['train_params']['beta1'], ae_config['train_params']['beta2']),
                           weight_decay=ae_config['train_params']['L2_reg']*2)

    # Create a SummaryWriter for TensorBoard
    writer = SummaryWriter(logdir=log_dir + ae_config['model_params']['name'] + '_' + datetime.now().strftime("%Y%m%d_%H%M%S"))

    yaml_str = yaml.dump(ae_config, default_flow_style=False)
    yaml_str = yaml_str.replace('\n', '<br>')
    writer.add_text('Parameters', yaml_str, 0)

    train_loss = 0
    val_loss = 0

    # Train the model
    for epoch in range(ae_config['train_params']['max_epochs']):
        for i, data in enumerate(train_dataloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            model_outs = ae_model(inputs)
            train_loss = ae_model.loss_function(*model_outs, M_N=0.005)

            # Backward pass and optimization
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            writer.add_scalar('Loss-ae/train', train_loss.item(), epoch * len(train_dataloader) + i + 1)

        with torch.no_grad():
            for j, data in enumerate(val_dataloader):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                model_outs = ae_model(inputs)
                val_loss = ae_model.loss_function(*model_outs, M_N=0.005)

                writer.add_scalar('Loss-ae/valid', val_loss.item(), epoch * len(val_dataloader) + j + 1)

        # for name, param in model.named_parameters():
        #     if 'weight' in name:
        #         layer_mean = torch.mean(param.data)
        #         layer_variance = torch.var(param.data)
        #         writer.add_scalar(f'{name}_mean', layer_mean.item(), epoch * len(train_dataloader) + i + 1)
        #         writer.add_scalar(f'{name}_variance', layer_variance.item(), epoch * len(train_dataloader) + i + 1)

        pbar.set_description(f"Train Loss: {train_loss.item():.4f} | Val Loss: {val_loss.item():.4f}")
        # pbar.set_postfix_str(f"Progress: {pbar.n+1}/{pbar.total}")
        pbar.update(1)

    # Close the SummaryWriter after training
    writer.close()
    pbar.close()

    torch.save(ae_model.state_dict(), os.path.join(ae_config['logging_params']['save_dir'], ae_config['logging_params']['name']))

    return ae_model


def train_reg(reg_config, log_dir, ae_model):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set to utilize tensor cores of GPU
    torch.set_float32_matmul_precision('medium')

    data = sslDataset(**reg_config["data_params"])
    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()

    pbar = tqdm(total=reg_config['train_params']['max_epochs'], ncols=120)

    # Initialize the model
    reg_model = reg_models[reg_config['model_params']['name']](**reg_config['model_params'])
    # reg_model = reg_models['regBasic'](in_size=128)
    reg_model.to(device)

    # Define the optimizer
    optimizer = optim.Adam(reg_model.parameters(),
                           lr=reg_config['train_params']['LR'],
                           betas=(reg_config['train_params']['beta1'], reg_config['train_params']['beta2']),
                           weight_decay=reg_config['train_params']['L2_reg'] * 2)

    # Create a SummaryWriter for TensorBoard
    writer = SummaryWriter(logdir=log_dir + reg_config['model_params']['name'] + '_' + datetime.now().strftime("%Y%m%d_%H%M%S"))

    yaml_str = yaml.dump(reg_config, default_flow_style=False)
    yaml_str = yaml_str.replace('\n', '<br>')
    writer.add_text('Parameters', yaml_str, 0)

    train_loss = 0
    val_loss = 0

    for param in ae_model.parameters():
        param.requires_grad = False

    # Train the model
    for epoch in range(reg_config['train_params']['max_epochs']):
        for i, data in enumerate(train_dataloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            ae_features = ae_model.encode(inputs)
            prediction = reg_model(ae_features)
            # print(f'pred-labels: {(prediction-labels)**2}')
            train_loss = reg_model.loss_function(prediction, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            writer.add_scalar('Loss-reg/train', train_loss.item(), epoch * len(train_dataloader) + i + 1)
            with torch.no_grad():
                loss_mae = torch.nn.functional.l1_loss(prediction, labels)
                writer.add_scalar('Loss-reg-MAE/train', loss_mae, epoch * len(train_dataloader) + i + 1)

        with torch.no_grad():
            for i, data in enumerate(val_dataloader):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                ae_features = ae_model.encode(inputs)
                prediction = reg_model(ae_features)
                val_loss = reg_model.loss_function(prediction, labels)

                writer.add_scalar('Loss-reg/valid', val_loss.item(), epoch * len(val_dataloader) + i + 1)

                with torch.no_grad():
                    loss_mae = torch.nn.functional.l1_loss(prediction, labels)
                    writer.add_scalar('Loss-reg-MAE/valid', loss_mae, epoch * len(val_dataloader) + i + 1)

        # for name, param in model.named_parameters():
        #     if 'weight' in name:
        #         layer_mean = torch.mean(param.data)
        #         layer_variance = torch.var(param.data)
        #         writer.add_scalar(f'{name}_mean', layer_mean.item(), epoch * len(train_dataloader) + i + 1)
        #         writer.add_scalar(f'{name}_variance', layer_variance.item(), epoch * len(train_dataloader) + i + 1)

        pbar.set_description(f"Train Loss: {train_loss.item():.4f} | Val Loss: {val_loss.item():.4f}")
        pbar.update(1)

    # Close the SummaryWriter after training
    writer.close()
    pbar.close()

    torch.save(reg_model.state_dict(), os.path.join(reg_config['logging_params']['save_dir'], reg_config['logging_params']['name']))



def train_reg_from_dino(reg_config, log_dir, ae_model):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set to utilize tensor cores of GPU
    torch.set_float32_matmul_precision('medium')

    data = sslDataset(**reg_config["data_params"])
    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()

    pbar = tqdm(total=reg_config['train_params']['max_epochs'], ncols=120)

    # Initialize the model
    reg_model = reg_models[reg_config['model_params']['name']](**reg_config['model_params'])
    reg_model_path = '/home/sam/Desktop/logs/regBasicDINOv2_finetuned.pth'
    # reg_model.load_state_dict(torch.load(reg_model_path))
    reg_model.to(device)

    # Define the optimizer
    # optimizer = optim.Adam(reg_model.parameters(),
    #                        lr=reg_config['train_params']['LR'],
    #                        betas=(reg_config['train_params']['beta1'], reg_config['train_params']['beta2']),
    #                        weight_decay=reg_config['train_params']['L2_reg'] * 2)

    optimizer = optim.AdamW(reg_model.parameters(),
                            lr=reg_config['train_params']['LR'],
                            betas=(reg_config['train_params']['beta1'], reg_config['train_params']['beta2']),
                            weight_decay=reg_config['train_params']['L2_reg'] * 2)



    # Create a SummaryWriter for TensorBoard
    writer = SummaryWriter(logdir=log_dir + reg_config['model_params']['name'] + '_' + datetime.now().strftime("%Y%m%d_%H%M%S"))

    yaml_str = yaml.dump(reg_config, default_flow_style=False)
    yaml_str = yaml_str.replace('\n', '<br>')
    writer.add_text('Parameters', yaml_str, 0)

    train_loss = 0
    val_loss = 0

    for param in ae_model.parameters():
        param.requires_grad = True

    # for param in ae_model.layer_name.parameters():
    #     param.requires_grad = True
    #
    # for name, module in ae_model.named_modules():
    #     print(name)

    # Train the model
    for epoch in range(reg_config['train_params']['max_epochs']):
        for i, data in enumerate(train_dataloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # print(f'labels: {labels}')

            # Forward pass
            ae_features = ae_model(inputs)
            # print(f'mean: {ae_features.mean()},  std: {ae_features.std()}')
            predictions = reg_model(ae_features)
            # print(f'pred-labels: {(prediction-labels)**2}')
            train_loss = reg_model.loss_function(predictions, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            step_nbr = epoch * len(train_dataloader) + i + 1

            writer.add_scalar('Loss-reg/train', train_loss.item(), step_nbr)
            with torch.no_grad():
                loss_mae = torch.nn.functional.l1_loss(predictions, labels)
                writer.add_scalar('Loss-reg-MAE/train', loss_mae, step_nbr)

        with torch.no_grad():
            for i, data in enumerate(val_dataloader):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                ae_features = ae_model(inputs)
                predictions = reg_model(ae_features)
                val_loss = reg_model.loss_function(predictions, labels)

                step_nbr = epoch * len(val_dataloader) + i + 1

                writer.add_scalar('Loss-reg/valid', val_loss.item(), step_nbr)
                with torch.no_grad():
                    loss_mae = torch.nn.functional.l1_loss(predictions, labels)
                    writer.add_scalar('Loss-reg-MAE/valid', loss_mae, step_nbr)

                if i == 0:
                    sample_nbr = random.randint(0, reg_config['data_params']['val_batch_size']-2)
                    log_images_scores_to_tensorboard(writer, step_nbr, inputs[sample_nbr, :, :, :])
                    log_regression_plot_to_tensorboard(writer, step_nbr, labels.cpu().flatten(), predictions.cpu().flatten())

        # for name, param in model.named_parameters():
        #     if 'weight' in name:
        #         layer_mean = torch.mean(param.data)
        #         layer_variance = torch.var(param.data)
        #         writer.add_scalar(f'{name}_mean', layer_mean.item(), epoch * len(train_dataloader) + i + 1)
        #         writer.add_scalar(f'{name}_variance', layer_variance.item(), epoch * len(train_dataloader) + i + 1)

        pbar.set_description(f"Train Loss: {train_loss.item():.4f} | Val Loss: {val_loss.item():.4f}")
        pbar.update(1)

        if epoch % 10 == 0:
            torch.save(reg_model.state_dict(),
                       os.path.join(reg_config['logging_params']['save_dir'],
                                    'ep_' + str(epoch) + '_' + reg_config['logging_params']['name']))

    # Close the SummaryWriter after training
    writer.close()
    pbar.close()

    # TODO: give each model unique name
    torch.save(reg_model.state_dict(), os.path.join(reg_config['logging_params']['save_dir'], reg_config['logging_params']['name']))

