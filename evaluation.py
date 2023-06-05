from collections import OrderedDict

import torch
from torch import nn, optim

from ignite.engine import *
from ignite.handlers import *
from ignite.metrics import *
from ignite.utils import *
from ignite.contrib.metrics.regression import *
from ignite.contrib.metrics import *

# create default evaluator for doctests

def eval_step(engine, batch):
    return batch

default_evaluator = Engine(eval_step)

# create default optimizer for doctests

param_tensor = torch.zeros([1], requires_grad=True)
default_optimizer = torch.optim.SGD([param_tensor], lr=0.1)

# create default trainer for doctests
# as handlers could be attached to the trainer,
# each test must define his own trainer using `.. testsetup:`

def get_default_trainer():

    def train_step(engine, batch):
        return batch

    return Engine(train_step)

# create default model for doctests

default_model = nn.Sequential(OrderedDict([
    ('base', nn.Linear(4, 2)),
    ('fc', nn.Linear(2, 1))
]))

manual_seed(666)
if __name__=='__main__':
    # metric = InceptionScore()
    # metric.attach(default_evaluator, "is")
    # y = torch.rand(10, 3, 299, 299)
    # state = default_evaluator.run([y])
    # print(state.metrics["is"])
    # metric = FID(num_features=1, feature_extractor=default_model)
    # metric.attach(default_evaluator, "fid")
    # y_true = torch.ones(10, 4)
    # y_pred = torch.randn(10, 4)
    # state = default_evaluator.run([[y_pred, y_true]])
    # print(state.metrics["fid"])
    # metric = InceptionScore(num_features=1, feature_extractor=default_model)
    # metric.attach(default_evaluator, "is")
    # y = torch.zeros(10, 4)
    # state = default_evaluator.run([y])
    # print(state.metrics["is"])
    metric = SSIM(data_range=1.0)
    metric.attach(default_evaluator, 'ssim')
    psnr = PSNR(data_range=1.0)
    psnr.attach(default_evaluator, 'psnr')
    preds = torch.rand([4, 3, 16, 16])
    target = preds * 0.75
    state = default_evaluator.run([[preds, target]])
    print(state.metrics['ssim'])
    print(state.metrics['psnr'])
