#!/usr/bin/python
# -*- coding:utf-8 -*-
from .abs_trainer import TrainConfig
from .AbFlow_trainer import AbFlowTrainer
from .AbFlowOpt_trainer import AbFlowOptTrainer

isMEANTrainer = AbFlowTrainer
isMEANOptTrainer = AbFlowOptTrainer
