# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modified from DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
from typing import Optional

import torch
from torch import Tensor, nn
from functools import partial
import copy
import torch.nn.functional as F
import math
from typing import Callable, List, Optional, Tuple
from torch.nn.parameter import Parameter
import einops
import numpy as np
from torch.nn.init import xavier_uniform_, constant_


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


NORM_DICT = {
    "bn1d": nn.BatchNorm1d,
    "id": nn.Identity,
    "ln": nn.LayerNorm,
}


ACTIVATION_DICT = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "leakyrelu": partial(nn.LeakyReLU, negative_slope=0.1),
}


WEIGHT_INIT_DICT = {
    "xavier_uniform": nn.init.xavier_uniform_,
}


# def linear(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
#     r"""
#     Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.
#     This operator supports :ref:`TensorFloat32<tf32_on_ampere>`.
#     Shape:
#         - Input: :math:`(*, in\_features)` where `*` means any number of
#           additional dimensions, including none
#         - Weight: :math:`(out\_features, in\_features)` or :math:`(in\_features)`
#         - Bias: :math:`(out\_features)` or :math:`()`
#         - Output: :math:`(*, out\_features)` or :math:`(*)`, based on the shape of the weight
#     """
#     return torch._C._nn.linear(input, weight, bias)


class MultiheadAttention(nn.Module):
    def __init__(self, dim, heads, attn_drop=0.0, proj_drop=0.0, rela_q=True, rela_k=True, rela_v=True):
        super().__init__()
        self.heads = heads
        assert dim % heads == 0
        head_dim = dim // heads
        self.in_proj_weight = Parameter(torch.empty((3 * dim, dim)))
        self.in_proj_bias = Parameter(torch.empty(3 * dim))
        self.scale = head_dim ** -0.5
        self.out_proj = nn.Linear(dim, dim)
        self.rela_q = nn.Sequential(nn.Linear(3, dim // 2), nn.ReLU(), nn.Linear(dim // 2, dim)) if rela_q else None
        self.rela_k = nn.Sequential(nn.Linear(3, dim // 2), nn.ReLU(), nn.Linear(dim // 2, dim)) if rela_k else None
        self.rela_v = nn.Sequential(nn.Linear(3, dim // 2), nn.ReLU(), nn.Linear(dim // 2, dim)) if rela_v else None
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)
        if self.rela_q is not None:
            nn.init.constant_(self.rela_q[-1].bias, 0.)
        if self.rela_k is not None:
            nn.init.constant_(self.rela_k[-1].bias, 0.)
        if self.rela_v is not None:
            nn.init.constant_(self.rela_v[-1].bias, 0.)

    def _in_projection_packed(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        w: Tensor,
        b: Optional[Tensor] = None,
    ) -> List[Tensor]:
        r"""
        Performs the in-projection step of the attention operation, using packed weights.
        Output is a triple containing projection tensors for query, key and value.
        Args:
            q, k, v: query, key and value tensors to be projected. For self-attention,
                these are typically the same tensor; for encoder-decoder attention,
                k and v are typically the same tensor. (We take advantage of these
                identities for performance if they are present.) Regardless, q, k and v
                must share a common embedding dimension; otherwise their shapes may vary.
            w: projection weights for q, k and v, packed into a single tensor. Weights
                are packed along dimension 0, in q, k, v order.
            b: optional projection biases for q, k and v, packed into a single tensor
                in q, k, v order.
        Shape:
            Inputs:
            - q: :math:`(..., E)` where E is the embedding dimension
            - k: :math:`(..., E)` where E is the embedding dimension
            - v: :math:`(..., E)` where E is the embedding dimension
            - w: :math:`(E * 3, E)` where E is the embedding dimension
            - b: :math:`E * 3` where E is the embedding dimension
            Output:
            - in output list :math:`[q', k', v']`, each output tensor will have the
                same shape as the corresponding input tensor.
        """
        qkv_same = torch.equal(q, k) and torch.equal(k, v)
        kv_same = torch.equal(k, v)
        E = q.size(-1)
        if qkv_same:
            # self-attention
            return F.linear(q, w, b).chunk(3, dim=-1)
        elif kv_same:
            # encoder-decoder attention
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            return (F.linear(q, w_q, b_q),) + F.linear(k, w_kv, b_kv).chunk(2, dim=-1)
        else:
            w_q, w_k, w_v = w.chunk(3)
            if b is None:
                b_q = b_k = b_v = None
            else:
                b_q, b_k, b_v = b.chunk(3)
            return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)

    def forward(self, q, k, v, qk_xyz):
        """
        Args:
            q: (L, B, C) 
            k: (S, B, C)
            v: (S, B, C)
            qk_xyz: (L, S, B, 3)
        """
        q, k, v = self._in_projection_packed(q, k, v, self.in_proj_weight, self.in_proj_bias)  # (T, B, 3C)
        q, k, v = [einops.rearrange(t, 't b (h c1) -> b h t c1', h=self.heads) for t in [q, k, v]]
        q = q * self.scale
        attn = torch.einsum('b h m c, b h n c -> b h m n', q, k)  # (B, head, L, S)
        attn_bias = 0
        if self.rela_q:
            q_pos = einops.rearrange(self.rela_q(qk_xyz), 'm n b (h c1) -> b h m n c1', h=self.heads)
            attn_bias += torch.einsum('b h m c, b h m n c -> b h m n', q, q_pos)
        if self.rela_k:
            k_pos = einops.rearrange(self.rela_k(qk_xyz), 'm n b (h c1) -> b h m n c1', h=self.heads)
            attn_bias += torch.einsum('b h n c, b h m n c -> b h m n', k * self.scale, k_pos)
        attn = self.attn_drop((attn + attn_bias).softmax(dim=-1))
        x = torch.einsum('b h m n, b h n c -> b h m c', attn, v)
        if self.rela_v:
            v_pos = einops.rearrange(self.rela_v(qk_xyz), 'm n b (h c1) -> b h m n c1', h=self.heads)
            x += torch.einsum('b h m n, b h m n c -> b h m c', attn, v_pos)
        x = einops.rearrange(x, 'b h t c1 -> t b (h c1)')
        x = self.proj_drop(self.out_proj(x))  # (L, B, C)
        return x


class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", norm_name="ln", weight_init_name="xavier_uniform"):
        super().__init__()
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, norm_name)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers)
        self._reset_parameters(weight_init_name)

    def _reset_parameters(self, weight_init_name):
        func = WEIGHT_INIT_DICT[weight_init_name]
        for p in self.parameters():
            if p.dim() > 1:
                func(p)
        for m in self.modules():
            if isinstance(m, MultiheadAttention):
                m._reset_parameters()

    def forward(self, src, tgt, pos_res):
        out = self.decoder(tgt, src, pos_res)
        return out


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt, memory, pos_res):
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, pos_res)
        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead=4, dim_feedforward=256,
                 dropout=0.1, activation="relu", norm_name="ln"):
        super().__init__()
        self.cross_attn = MultiheadAttention(d_model, nhead, attn_drop=dropout, proj_drop=0.0)
        self.norm2 = NORM_DICT[norm_name](d_model)
        self.norm3 = NORM_DICT[norm_name](d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = ACTIVATION_DICT[activation]()

    def forward(self, tgt, memory, pos_res):
        tgt = self.norm2(tgt + self.dropout2(self.cross_attn(tgt, memory, memory, pos_res)))
        tgt = self.norm3(tgt + self.dropout3(self.linear2(self.dropout(self.activation(self.linear1(tgt))))))
        return tgt

