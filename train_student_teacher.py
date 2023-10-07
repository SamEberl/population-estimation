from models import *
from training_loops.fixMatch_loop import *

print('--- Loading model ---')

config = parse_yaml('configs/fixMatch.yaml')

# Create model
student_model = ssl_models[config['model_params']['name']](**config['model_params'])
teacher_model = ssl_models[config['model_params']['name']](**config['model_params'])

# Load previous checkpoint
# student_model.load_state_dict(torch.load(os.path.join(config['logging_params']['save_dir'], config['logging_params']['name'])))
# teacher_model.load_state_dict(torch.load(os.path.join(config['logging_params']['save_dir'], config['logging_params']['name'])))

log_dir = config['logging_params']['save_dir']
current_datetime = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
print(f'--- Start training at {current_datetime} ---')

# Create a SummaryWriter for TensorBoard
writer = SummaryWriter(logdir=log_dir + config['model_params']['name'] + '-' + current_datetime)

# Train model
train_fix_match(config, writer, student_model, teacher_model)

save_path = os.path.join(config['logging_params']['save_dir'],
                         f"{config['model_params']['model_name']}_{current_datetime}.pt")
print(f'Saving model under {save_path}')
torch.save(student_model.state_dict(), save_path)
