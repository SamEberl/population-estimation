import argparse
from training_loops.fixMatch_loop import *
from dataset import get_dataloaders
from tensorboardX import SummaryWriter
from datetime import datetime
from logging_utils import *
from models import *
from utils import parse_yaml

def main(file_path):
    # Your existing code that uses the file_path
    print(f"--- Training on file: {file_path} ---")
    config = parse_yaml(f'configs/fixMatch_l2_0p1.yaml')
    #config = parse_yaml('configs/fixMatch_local.yaml')

    seed = config['train_params']['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Choose device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Init models
    student_model = ssl_models[config['model_params']['architecture']](**config['model_params']).to(device)
    teacher_model = ssl_models[config['model_params']['architecture']](**config['model_params']).to(device)

    # Retrain from checkpoint
    if config['model_params']['retrain']:
        retrain_from = config['model_params']['retrain_from']
        student_model.load_state_dict(torch.load(os.path.join(config['save_dirs']['model_save_dir'], retrain_from)))
        teacher_model.load_state_dict(torch.load(os.path.join(config['save_dirs']['model_save_dir'], retrain_from)))

    # Set logging directory
    log_dir = config['save_dirs']['log_save_dir']
    current_datetime = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    print(f'--- Start training at {current_datetime} ---')

    # Create a SummaryWriter for TensorBoard
    writer = SummaryWriter(logdir=log_dir + config['model_params']['architecture'] + '-' + current_datetime)

    train_dataloader, valid_dataloader, train_dataloader_unlabeled = get_dataloaders(config)

    train_fix_match(config,
                    writer,
                    student_model,
                    teacher_model,
                    train_dataloader,
                    valid_dataloader,
                    train_dataloader_unlabeled)

    save_path = os.path.join(config['save_dirs']['model_save_dir'],
                             f"{config['model_params']['pretrained_weights']}_{current_datetime}.pt")
    student_model.eval()
    print(f'Saving model under {save_path}')
    torch.save(student_model.state_dict(), save_path)

    # writer.add_hparams({
    #     'in_channels': config['model_params']['in_channels'],
    #     'retrain': config['model_params']['retrain'],
    #     'drop_rate': config['model_params']['drop_rate'],
    #     'train_batch_size': config['data_params']['train_batch_size'],
    #     'LR': config['train_params']['LR'],
    #     'L2_reg': config['train_params']['L2_reg'],
    #     'beta1': config['train_params']['beta1'],
    #     'beta2': config['train_params']['beta2'],
    #     'ema_alpha': config['train_params']['ema_alpha']},
    #     {'train_loss': train_loss,
    #      'val_loss': val_loss}, name='')

    # Close the SummaryWriter after training
    writer.close()


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='Train a model with a specified file path.')

    # Add an argument for the file path
    parser.add_argument('--file_path', type=str, required=True, help='The path to the file to train on.')

    # Parse the arguments
    args = parser.parse_args()

    # Call the main function with the file path
    main(args.file_path)