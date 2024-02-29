import math
import csv
import numpy as np
import os
from models import *
from dataset import PopDataset, unnormalize_preds
from torch.utils.data import DataLoader
from tqdm import tqdm


class UncertaintyLogger:
    def __init__(self,
                 retrain_from='convnextv2_atto.fcmae_2024_02_07-19_30_14',
                 data_dir="/home/pop-dens/data/So2Sat_POP_Part1"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.retrain_from = retrain_from
        self.teacher_model = None
        self.data_dir = data_dir
        self.dataloader = None
        self.num_samples_teacher = 50
        self.uncertainties = {}

        self.create_model()
        self.create_dataloader()

    def create_model(self):
        self.teacher_model = ssl_models['fixMatch'](pretrained_weights='convnextv2_atto.fcmae',
                                                    pretrained=False,
                                                    in_channels=3,
                                                    nbr_outputs=1,
                                                    supervised_criterion='Aleatoric',
                                                    unsupervised_criterion='triplet',
                                                    unsupervised_factor=1,
                                                    drop_rate=0).to(self.device)
        # Retrain from checkpoint
        self.teacher_model.load_state_dict(torch.load(os.path.join('/home/sameberl/models', self.retrain_from + '.pt')))

    def create_dataloader(self):
        data = []
        nbr_not_found = 0
        nbr_found = 0
        #for split in ['train', 'test']:
        for split in ['test']:
            data_sub_dir = os.path.join(self.data_dir, split)
            for city_folder in os.listdir(data_sub_dir):
                # Load the csv file that maps datapoint names to folder names
                with open(os.path.join(data_sub_dir, f'{city_folder}/{city_folder}.csv'), 'r') as csvfile:
                    reader = csv.reader(csvfile)
                    for row in reader:
                        datapoint_name = row[0]
                        label = row[2]

                        if label == 'POP':  # skip first row which is header of file
                            continue
                        elif int(label) == 0:
                            class_nbr = 'Class_0'
                        else:
                            class_nbr = f'Class_{math.ceil(math.log(int(label), 2) + 0.00001)}'

                        modality = 'sen2spring'
                        file_name = datapoint_name + '_' + modality + '.tif'
                        input_path = os.path.join(data_sub_dir, city_folder, modality, class_nbr, file_name)
                        if os.path.isfile(input_path):
                            nbr_found += 1
                            data.append((input_path, label))
                        else:
                            nbr_not_found += 1
        print(f'In: {data_sub_dir} \n  #found: {nbr_found} \n  #notFound: {nbr_not_found}')

        use_channels = {'use_spring_rgb': True,
                        'use_lu': False,
                        'use_lcz': False,
                        'use_dem': False,
                        'use_viirs': False}
        dataset = PopDataset(data, split='valid', transform_params=None, use_channels=use_channels)
        self.dataloader = DataLoader(dataset,
                                     batch_size=32,
                                     shuffle=True,
                                     num_workers=32,
                                     pin_memory=True)

    def add_uncertainty(self, uncertainty_name, uncertainty):
        uncertainty = uncertainty.to('cpu').detach().numpy()
        if uncertainty_name not in self.uncertainties:
            self.uncertainties[uncertainty_name] = np.array([])
        self.uncertainties[uncertainty_name] = np.concatenate((self.uncertainties[uncertainty_name], uncertainty),
                                                              axis=0)

    def get_uncertainties(self):
        for data in tqdm(self.dataloader):
            teacher_inputs, labels = data
            teacher_inputs = teacher_inputs.to(self.device)
            labels = labels.to(self.device)

            n_teacher_preds = []
            n_teacher_features = []
            self.teacher_model.train()  # Ensure dropout is active for approximating Bayesian-NN
            with torch.no_grad():  # Ensure no gradients are computed
                for _ in range(self.num_samples_teacher):
                    teacher_preds, teacher_features, teacher_data_uncertainty = self.teacher_model(teacher_inputs)
                    print(f'teacher_preds: {teacher_preds}')
                    print(f'teacher_features: {teacher_features}')
                    print(f'teacher_data_uncert: {teacher_data_uncertainty}')
                    n_teacher_preds.append(unnormalize_preds(teacher_preds))
                    n_teacher_features.append(teacher_features)
            n_teacher_preds = torch.stack(n_teacher_preds)
            n_teacher_features = torch.stack(n_teacher_features)

            # Compute model uncertainty
            teacher_model_uncertainty = n_teacher_preds.var(dim=0)

            # Compute spread: Variance of L2-Distances between features
            spread = torch.sqrt((n_teacher_features - n_teacher_features.mean(dim=0)).pow(2).sum(dim=-1)).var(dim=0)

            # Compute data uncertainty
            self.teacher_model.eval()
            with torch.no_grad():
                teacher_preds, _, teacher_data_uncertainty = self.teacher_model(teacher_inputs)
                teacher_preds = unnormalize_preds(teacher_preds)

            self.add_uncertainty('pred', teacher_preds)
            self.add_uncertainty('label', labels)
            self.add_uncertainty('loss', (teacher_preds - labels) ** 2)
            self.add_uncertainty('pred_var', teacher_data_uncertainty)
            self.add_uncertainty('calc_var', teacher_model_uncertainty)
            self.add_uncertainty('spread', spread)

        if bool(self.uncertainties):
            path = f"/home/sameberl/computed_numpy/{self.retrain_from}"
            print(f'saving to: {path}')
            if not os.path.exists(path):
                os.makedirs(path)
            for uncertainty_name, uncertainty in self.uncertainties.items():
                np.save(os.path.join(path, f'{uncertainty_name}.npy'), uncertainty)


retrain_from = 'convnextv2_atto.fcmae_2024_02_28-14_18_29'
print(f'Get uncertainties from {retrain_from}')
uncertainty_logger = UncertaintyLogger(retrain_from=retrain_from)
uncertainty_logger.get_uncertainties()
