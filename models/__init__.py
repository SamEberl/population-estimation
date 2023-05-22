from .base import *
from .aeBasic import *
from .vaeBasic import *
from .aeMNIST import *
from .regBasic import *
from .aeResNet import *

# Aliases
ae_models = {'aeBasic': aeBasic,
             'vaeBasic': vaeBasic,
             'aeResNet': aeResNet,
             'aeMNIST': aeMNIST}

reg_models = {'regBasic': regBasic}
