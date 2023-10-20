from torch import nn
import torch.nn.functional as F


ACT2FN = dict(
    gelu=F.gelu,
    relu=F.relu,
    tanh=F.tanh,
)

ACT2NN = dict(
    gelu=nn.GELU,
    relu=nn.ReLU,
    tanh=nn.Tanh,
)

