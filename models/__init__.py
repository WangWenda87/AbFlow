#!/usr/bin/python
# -*- coding:utf-8 -*-
from .AbFlow.AbFlow_model import AbFlowModel
from .AbFlow.AbFlowStruct_model import AbFlowStructModel
from .AbFlow.AbFlowOpt_model import AbFlowOptModel
from . import AbFlow

import sys
sys.modules['models.isMEAN'] = AbFlow
sys.modules['models.dyMEAN'] = AbFlow
sys.modules['models.isMEAN.isMEAN_model'] = AbFlow.AbFlow_model
sys.modules['models.isMEAN.isMEANOpt_model'] = AbFlow.AbFlowOpt_model
sys.modules['models.dyMEAN.dyMEAN_model'] = AbFlow.AbFlow_model
