import torch.nn as nn
from timm import create_model
from losses import RMSELoss, RMSLELoss


# Define the student and teacher models
class Student(nn.Module):
    def __init__(self, model_name='resnet18', num_classes=10):
        super(Student, self).__init__()
        self.model = create_model(model_name, pretrained=True)
        self.model.fc = nn.Linear(self.model.num_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x

    def loss_function(self, predictions, labels):
        # criterion = RMSLELoss()
        criterion = RMSELoss()
        # criterion = nn.MSELoss()
        # criterion = nn.L1Loss()
        loss = criterion(predictions, labels)
        return loss


class Teacher(nn.Module):
    def __init__(self, model_name='resnet18', num_classes=10):
        super(Teacher, self).__init__()
        self.model = create_model(model_name, pretrained=True)
        self.model.fc = nn.Linear(self.model.num_features, num_classes)

        #self.ema = nn.Sequential()
        #for name, module in self.model.named_children():
        #    if name != 'head':
        #        self.ema.add_module(name, module)

    def forward(self, x):
        x = self.model(x)
        return x