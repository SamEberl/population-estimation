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
student_model = ssl_models[config['model_params']['architecture']](**config['model_params'])
teacher_model = ssl_models[config['model_params']['architecture']](**config['model_params'])

# Load previous checkpoint
# student_model.load_state_dict(torch.load(os.path.join(config['logging_params']['save_dir'], config['logging_params']['name'])))
# teacher_model.load_state_dict(torch.load(os.path.join(config['logging_params']['save_dir'], config['logging_params']['name'])))

log_dir = config['save_dirs']['log_save_dir']
current_datetime = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
print(f'--- Start training at {current_datetime} ---')

# Create a SummaryWriter for TensorBoard
writer = SummaryWriter(logdir=log_dir + config['model_params']['architecture'] + '-' + current_datetime)

if config['hparam_search']['active']:
    print(f'ACTIVE!')
    hparam_name = config['hparam_search']['hparam_name']
    n = config['hparam_search']['nbr_values']
    lowest = config['hparam_search']['lowest']
    highest = config['hparam_search']['highest']
    param_list = np.linspace(lowest, highest, n)
    decimal_places = count_decimal_places(lowest) + 1
    rounded_param_list = [round(value, decimal_places) for value in param_list]
    for i in range(n):
        student_model_temp = student_model
        teacher_model_temp = teacher_model
        config['train_params'][hparam_name] = rounded_param_list[i]
        print('rounded param list')
        print(rounded_param_list)
        print('hparam value: ')
        print(config['train_params'][hparam_name])
        train_fix_match(config, writer, student_model_temp, teacher_model_temp)
else:
    train_fix_match(config, writer, student_model, teacher_model)

    save_path = os.path.join(config['save_dirs']['model_save_dir'],
                             f"{config['model_params']['pretrained_weights']}_{current_datetime}.pt")
    print(f'Saving model under {save_path}')
    torch.save(student_model.state_dict(), save_path)
