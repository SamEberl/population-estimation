import copy
from models import *
from training_loops.fixMatch_loop import *

print('--- Loading model ---')

config = parse_yaml('configs/fixMatch.yaml')

seed = config['train_params']['seed']
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Create model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
student_model = ssl_models[config['model_params']['architecture']](**config['model_params']).to(device)
teacher_model = ssl_models[config['model_params']['architecture']](**config['model_params']).to(device)
# Load previous checkpoint
# student_model.load_state_dict(torch.load(os.path.join(config['logging_params']['save_dir'], config['logging_params']['name'])))
# teacher_model.load_state_dict(torch.load(os.path.join(config['logging_params']['save_dir'], config['logging_params']['name'])))
# Make sure no grad is calculated for teacher & remove things like dropout
teacher_model.eval()

log_dir = config['save_dirs']['log_save_dir']
current_datetime = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
print(f'--- Start training at {current_datetime} ---')

# Create a SummaryWriter for TensorBoard
writer = SummaryWriter(logdir=log_dir + config['model_params']['architecture'] + '-' + current_datetime)

writer.add_hparams({
    'in_channels': config['model_params']['in_channels'],
    'drop_rate': config['model_params']['drop_rate'],
    'train_batch_size': config['data_params']['train_batch_size'],
    'LR': config['train_params']['LR'],
    'L2_reg': config['train_params']['L2_reg'],
    'beta1': config['train_params']['beta1'],
    'beta2': config['train_params']['beta2'],
    'ema_alpha': config['train_params']['ema_alpha']}, {})

param_yaml_str = yaml.dump(config, default_flow_style=False)
param_yaml_str = param_yaml_str.replace('\n', '<br>')
writer.add_text('Parameters', param_yaml_str, 0)
model_yaml_str = yaml.dump(student_model.model.default_cfg, default_flow_style=False)
model_yaml_str = model_yaml_str.replace('false ', '<br>')
writer.add_text('Model Specs', model_yaml_str, 0)

student_transform, teacher_transform = get_transforms(config)
train_dataloader, val_dataloader = get_dataloader(config, student_transform, teacher_transform)

if config['hparam_search']['active']:
    hparam_name = config['hparam_search']['hparam_name']
    n = config['hparam_search']['nbr_values']
    lowest = (config['hparam_search']['lowest'])
    highest = (config['hparam_search']['highest'])
    param_list = np.geomspace(lowest, highest, n)
    decimal_places = count_decimal_places(lowest) + 1
    rounded_param_list = [round(value, decimal_places) for value in param_list]
    rounded_param_list = [5e-05, 5e-05, 5e-05, 5e-05, 5e-05, 5e-05, 5e-05, 5e-05, 5e-05, 5e-05, 5e-05, 5e-05, 5e-05,
                          5e-05, 5e-05, 5e-05, 5e-05, 5e-05, 5e-05, 5e-05, 5e-05, 5e-05, 5e-05, 5e-05, 5e-05, 5e-05]
    print(f'param_list: {rounded_param_list}')
    for i in range(n):
        student_model_temp = copy.deepcopy(student_model)
        teacher_model_temp = copy.deepcopy(teacher_model)
        config['train_params'][hparam_name] = rounded_param_list[i]
        train_fix_match(config, writer, student_model_temp, teacher_model_temp, train_dataloader, val_dataloader)
else:
    train_fix_match(config, writer, student_model, teacher_model, train_dataloader, val_dataloader)

    save_path = os.path.join(config['save_dirs']['model_save_dir'],
                             f"{config['model_params']['pretrained_weights']}_{current_datetime}.pt")
    print(f'Saving model under {save_path}')
    torch.save(student_model.state_dict(), save_path)
