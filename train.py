from trainFuncs import *

print('---Start Training---')

ae_config = parse_yaml('configs/aeResNet.yaml')
reg_config = parse_yaml('configs/regBasic.yaml')

log_dir = ae_config['logging_params']['save_dir']

# print('---Training AE model---')
# ae_model = train_ae(ae_config, log_dir)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ae_model = ae_models[ae_config['model_params']['name']](**ae_config['model_params']).to(device)
model_path = os.path.join(ae_config['logging_params']['save_dir'], ae_config['logging_params']['name'])
ae_model.load_state_dict(torch.load(model_path))

print('---Training Regression model---')
train_reg(reg_config, log_dir, ae_model)
