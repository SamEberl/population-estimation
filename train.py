import copy
from models import *
from training_loops.fixMatch_loop import *
from dataset import get_dataloader

print('--- Loading model ---')

config = parse_yaml('configs/fixMatch.yaml')
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
param_yaml_str = yaml.dump(config, default_flow_style=False)
param_yaml_str = param_yaml_str.replace('\n', '<br>')
writer.add_text('Parameters', param_yaml_str, 0)
model_yaml_str = yaml.dump(student_model.model.default_cfg, default_flow_style=False)
model_yaml_str = model_yaml_str.replace('false ', '<br>')
writer.add_text('Model Specs', model_yaml_str, 0)

student_transform, teacher_transform = get_transforms(config)
train_dataloader, val_dataloader = get_dataloader(config, student_transform, teacher_transform)


train_loss, val_loss = train_fix_match(config, writer, student_model, teacher_model, train_dataloader, val_dataloader)

save_path = os.path.join(config['save_dirs']['model_save_dir'],
                         f"{config['model_params']['pretrained_weights']}_{current_datetime}.pt")
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
