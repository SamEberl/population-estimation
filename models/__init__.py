from .base import *
from .aeBasic import *
from .vaeBasic import *
from .aeMNIST import *
from .regBasic import *
from .regBasicDINOv2 import *
from .aeResNet import *

# Aliases
ssl_models = {'aeBasic': aeBasic,
             'vaeBasic': vaeBasic,
             'aeResNet': aeResNet,
             'aeMNIST': aeMNIST}

reg_models = {'regBasic': regBasic,
              'regBasicDINOv2': regBasicDINOv2}
