# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F

try:
    from torch_sparse import spmm, coalesce, spspmm
    import dgl
    import dgl.function as fn
except:
    pass
from utils import *

  