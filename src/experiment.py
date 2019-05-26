from collections import OrderedDict
from catalyst.dl.experiments import ConfigExperiment
from src.augmentation import *
from src.dataset import *

from catalyst.dl.registry import \
    MODELS, CRITERIONS, OPTIMIZERS, SCHEDULERS, CALLBACKS
from torch import nn, optim
from catalyst.dl.fp16 import Fp16Wrap
from catalyst.dl import utils

_Model = nn.Module
_Criterion = nn.Module
_Optimizer = optim.Optimizer
# noinspection PyProtectedMember
_Scheduler = optim.lr_scheduler._LRScheduler


class Experiment(ConfigExperiment):

    # def get_optimizer(self, stage: str, model: nn.Module) -> _Optimizer:
    #
    #     # print(model)
    #
    #     optimizer_params = \
    #         self.stages_config[stage].get("optimizer_params", {})
    #
    #     extractor_params = list(map(id, model.module.extractor.parameters()))
    #     classifier_params = filter(lambda p: id(p) not in extractor_params, model.parameters())
    #     optimizer = optim.Adam([
    #         {'params': model.module.extractor.parameters()},
    #         {'params': classifier_params, 'lr': optimizer_params['lr'] * 10}
    #     ], lr=optimizer_params['lr'], weight_decay=optimizer_params['weight_decay'])
    #
    #     return optimizer

    def get_datasets(self, stage: str, **kwargs):
        datasets = OrderedDict()

        train_csv = kwargs.get('train_csv', None)
        valid_csv = kwargs.get('valid_csv', None)
        datapath = kwargs.get('datapath', None)

        trainset = LandmarkDataset(
            df=train_csv,
            root=datapath,
            mode='train',
            transform=train_aug(224),
        )
        testset = LandmarkDataset(
            df=valid_csv,
            root=datapath,
            mode='valid',
            transform=valid_aug(224),
        )

        datasets["train"] = trainset
        datasets["valid"] = testset

        return datasets
