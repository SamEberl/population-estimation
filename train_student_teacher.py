from trainFuncs import *

print('---Start Training---')

config = parse_yaml('configs/fixMatch.yaml')

log_dir = config['logging_params']['save_dir']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
student_model = ssl_models[config['model_params']['name']](**config['model_params']).to(device)
teacher_model = ssl_models[config['model_params']['name']](**config['model_params']).to(device)
student_path = os.path.join(config['logging_params']['save_dir'], config['logging_params']['name'])
teacher_path = os.path.join()
student_model.load_state_dict(torch.load(student_path))
teacher_model.load_state_dict(torch.load(teacher_path))

print('---Training Regression model---')
train_fix_match(config, log_dir, student_model, teacher_model)
