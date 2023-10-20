import torch
from torch import nn
from torch import Tensor, FloatTensor
from torch.nn import functional as F

from ..utils.activations import ACT2NN

from typing import *

SrcIdxT = List[int]
DstIdxT = List[int]
TargetsT = Tuple[SrcIdxT, DstIdxT]


class BalaurHeads(nn.Module):
    def __init__(self,
                 src_features: int,
                 dst_features: int,
                 head_dim: int,
                 num_heads: int,
                 src_bias: bool = True,
                 dst_bias: bool = True,
                 rel_bias: bool = False,
                 eps: float = 1e-7,
                 activation: str = "gelu",
                 vocab_dim: int = 50265,
                 ):
        super().__init__()
        self.head_dim = head_dim
        self.balaur_dim = num_heads * head_dim
        self.num_heads = num_heads
        self.vocab_dim = vocab_dim

        self.src_proj = nn.Linear(src_features, self.balaur_dim, bias=src_bias)
        self.dst_proj = nn.Linear(dst_features, self.balaur_dim, bias=dst_bias)
        self.src_rel_proj = nn.Linear(self.balaur_dim, self.balaur_dim, bias=rel_bias)
        self.dst_rel_proj = self.src_rel_proj
        self.logit_layer_norm = nn.LayerNorm(self.vocab_dim, eps=eps)

        self.activation = ACT2NN[activation]()

    def split_heads(self, x: Tensor):
        return x.view(-1, self.num_heads, self.head_dim)

    def merge_heads(self, x: Tensor):
        return x.view(-1, self.balaur_dim)

    def compute_logits(self, src: Tensor, dst: Tensor):
        # project source embeddings into relation space for each head
        src = self.src_proj(src)
        src = self.activation(src)
        src = self.split_heads(self.src_rel_proj(src))

        # project destination embeddings into relation space for each head
        dst = self.dst_proj(dst)
        dst = self.activation(dst)
        dst = self.split_heads(self.dst_rel_proj(dst))

        # compute logits, i.e. the inner product between src and dst embeddings
        # (broadcast across each head and corresponding relation subspace)
        src = src.permute(1, 0, 2)  # (num_heads, -1, head_dim)
        dst = dst.permute(1, 2, 0)  # (num_heads, head_dim, -1)

        # prevent numerical underflow observed in large models
        if src.dtype == torch.float16:
            with torch.autocast(device_type=src.device.type, dtype=torch.float32):
                logits = torch.matmul(src, dst)
        else:
            logits = torch.matmul(src, dst)
        logits = self.logit_layer_norm(logits)

        return logits

    def forward(self, src: Tensor, dst: Tensor):
        return self.compute_logits(src, dst)

    def compute_loss(self, src: Tensor, dst: Tensor, targets: List[TargetsT]):
        logits = self.forward(src, dst)
        nll = -F.log_softmax(logits, dim=-1)
        total_loss = 0
        losses = []
        for rel_idx, rel_targets in enumerate(targets):
            # skip empty targets
            if rel_targets[0].shape[0]:
                loss = nll[rel_idx][rel_targets].mean()
                losses.append(loss.item())
                total_loss += loss
            else:
                losses.append(None)
        total_loss /= len(targets)
        return total_loss, losses