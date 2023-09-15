from .base import *
from .aeBasic import *
from .vaeBasic import *
from .aeMNIST import *
from .regBasic import *
from .regBasicDINOv2 import *
from .aeResNet import *
from .fixMatch import *

# Aliases
ssl_models = {'aeBasic': aeBasic,
              'vaeBasic': vaeBasic,
              'aeResNet': aeResNet,
              'aeMNIST': aeMNIST,
              'fixMatch': fixMatch}

reg_models = {'regBasic': regBasic,
              'regBasicDINOv2': regBasicDINOv2}
