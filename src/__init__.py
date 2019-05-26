
# flake8: noqa
from catalyst.dl import registry

from .experiment import Experiment
from .runner import ModelRunner as Runner
from .callbacks import MyLossCallback, IterCheckpointCallback
from .models import Net, FewShotModel
from .losses import *

registry.Model(Net)
registry.Model(FewShotModel)
registry.Callback(MyLossCallback)
registry.Callback(IterCheckpointCallback)
registry.Criterion(FocalLoss)