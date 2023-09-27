from models import *
from training_loops.fixMatch_loop import *

print('---Student Teacher Model---')

config = parse_yaml('configs/fixMatch.yaml')

log_dir = config['logging_params']['save_dir']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create model
student_model = ssl_models[config['model_params']['name']](**config['model_params']).to(device)
teacher_model = ssl_models[config['model_params']['name']](**config['model_params']).to(device)

# Load previous checkpoint
# student_model.load_state_dict(torch.load(os.path.join(config['logging_params']['save_dir'], config['logging_params']['name'])))
# teacher_model.load_state_dict(torch.load(os.path.join(config['logging_params']['save_dir'], config['logging_params']['name'])))

print('---Training model---')
train_fix_match(config, log_dir, student_model, teacher_model)
