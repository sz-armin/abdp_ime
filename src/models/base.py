import logging
import math
from copy import deepcopy
from typing import List, Optional, Type

import numpy as np
import lightning.pytorch as pl
import torch
from torch import nn, optim
from torch._tensor import Tensor

# TODO make dropout (and friends) layers seperate


class LearnedPosEmbedding(nn.Module):
    def __init__(
        self, num_embeddings: int = 1024, embedding_dim: int = 512, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings + 1
        self.internal_pad_idx = 0
        self.embedding = nn.Embedding(
            self.num_embeddings,
            embedding_dim,
            padding_idx=self.internal_pad_idx,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_idx: int = 3):
        mask = x.ne(padding_idx)
        pos = torch.cumsum(mask, dim=1)

        assert (
            pos.max() < self.num_embeddings
        ), f"max pos {pos.max()} is larger than num_embeddings {self.num_embeddings}"

        return self.embedding(pos)


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    # TODO test

    def __init__(self, d_model: int, dropout, max_len: int = 5000) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x, fixed_pos: Optional[int]=None):
        """
        x: BxLxD
        fixed_pos: right boundry
        """
        if fixed_pos is None:
            x = x + self.pe[:, : x.size(1), :].requires_grad_(False)  # type: ignore
        else:
            x = x + self.pe[:, fixed_pos - 1 : fixed_pos, :].requires_grad_(False)  # type: ignore

        return self.dropout(x)


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, nheads, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % nheads == 0
        self.d_model = d_model
        self.nheads = nheads
        # assuming d_k = d_v
        self.d_k = d_model // nheads
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(self.dropout)

        self.linears = nn.ModuleList(
            deepcopy(nn.Linear(self.d_model, self.d_model)) for _ in range(4)
        )
        for l in self.linears:
            nn.init.xavier_uniform_(l.weight, gain=1 / math.sqrt(2))  # type: ignore
            nn.init.constant_(l.bias, 0)  # type: ignore
        nn.init.xavier_uniform_(self.linears[-1].weight, gain=1 / math.sqrt(2))  # type: ignore

    def forward(self, q: Tensor, k, v, mask:Optional[Tensor]=None, requires_attention: bool = False):
        bsz = q.size(0)
        q, k, v = (
            linear(x).view(bsz, -1, self.nheads, self.d_k).transpose(1, 2)
            for linear, x in zip(self.linears, (q, k, v))
        )

        x, attn = self._attention(q, k, v, mask)
        x = x.transpose(1, 2).reshape(bsz, -1, self.d_model)
        x = self.linears[-1](x)

        if requires_attention:
            return x, attn
        else:
            return x, None

    def _attention(self, q: Tensor, k, v, mask: Optional[Tensor]=None):
        q = q / math.sqrt(self.d_k)
        scores = torch.matmul(q, k.transpose(-2, -1))  # (bsz, nheads, q_len, k_len)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        p_attn = scores.softmax(-1)
        # local attention can cause <pad> rows to be all nan
        p_attn = torch.nan_to_num(p_attn, nan=0.0)
        p_attn = self.dropout_layer(p_attn)

        return torch.matmul(p_attn, v), p_attn  # TODO


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        dropout: float = 0.1,
        activation: Type[torch.nn.modules.activation.ReLU] = nn.ReLU,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout
        self.dim_feedforward = self.d_model * 4
        self.activation = activation

        self.dropout_layer = nn.Dropout(self.dropout)
        self.layer_norms = nn.ModuleList(
            deepcopy(nn.LayerNorm(self.d_model)) for _ in range(2)
        )

        self.self_attn = MultiheadAttention(
            self.d_model, self.nhead, dropout=self.dropout
        )

        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, self.dim_feedforward),
            self.activation(),
            self.dropout_layer,
            nn.Linear(self.dim_feedforward, self.d_model),
        )
        for p in self.ffn.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor]=None,
        requires_attention: bool = False,
        inc_cache: Optional[List[Tensor]]=None,
        cache_idx: int=1, # hack for TorchScript, shoud be None
    ):
        """
        x: BxLxD, or 1x1xD if incremental
        """
        if inc_cache is None:
            x2, attn = self.self_attn(
                x, x, x, mask=mask, requires_attention=requires_attention
            )
        else:
            assert x.size(0) == 1 and x.size(1) == 1
            # TODO avoid clones
            k_v = torch.cat([inc_cache[cache_idx], x], dim=1)
            inc_cache[cache_idx] = torch.cat([inc_cache[cache_idx], x], dim=1)
            x2, attn = self.self_attn(
                x, k_v, k_v, mask=mask, requires_attention=requires_attention
            )

        x = x + self.dropout_layer(x2)
        x = self.layer_norms[0](x)

        x2 = self.ffn(x)
        x = x + self.dropout_layer(x2)
        x = self.layer_norms[1](x)

        return x, attn


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        dropout: float = 0.1,
        activation: Type[torch.nn.modules.activation.ReLU] = nn.ReLU,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout
        self.dim_feedforward = self.d_model * 4
        self.activation = activation

        self.dropout_layer = nn.Dropout(self.dropout)
        self.layer_norms = nn.ModuleList(
            deepcopy(nn.LayerNorm(self.d_model)) for _ in range(3)
        )

        self.self_attn = MultiheadAttention(
            self.d_model, self.nhead, dropout=self.dropout
        )
        self.cross_attn = MultiheadAttention(
            self.d_model, self.nhead, dropout=self.dropout
        )
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, self.dim_feedforward),
            self.activation(),
            self.dropout_layer,
            nn.Linear(self.dim_feedforward, self.d_model),
        )
        for p in self.ffn.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        x,
        memory,
        tgt_mmask=None,
        memory_mmask=None,
        requires_attention: bool = False,
        self_inc_cache=None,
        cache_idx=None,
    ):
        if self_inc_cache is None:
            x2, attn_self = self.self_attn(
                x, x, x, mask=tgt_mmask, requires_attention=requires_attention
            )
        else:
            assert x.size(0) == 1 and x.size(1) == 1
            # TODO avoid clones
            k_v = torch.cat([self_inc_cache[cache_idx], x], dim=1)
            self_inc_cache[cache_idx] = torch.cat([self_inc_cache[cache_idx], x], dim=1)
            x2, attn_self = self.self_attn(
                x, k_v, k_v, mask=tgt_mmask, requires_attention=requires_attention
            )

        x = x + self.dropout_layer(x2)
        x = self.layer_norms[0](x)
        x2, attn_cross = self.cross_attn(
            x, memory, memory, mask=memory_mmask, requires_attention=requires_attention
        )  # TODO mask
        x = x + self.dropout_layer(x2)
        x = self.layer_norms[1](x)

        x2 = self.ffn(x)
        x = x + self.dropout_layer(x2)
        x = self.layer_norms[2](x)

        return x, attn_self, attn_cross

class Frontier:
    def __init__(
        self,
        tokenizer,
        path=[1],
        scores=[1],
    ) -> None:
        self.path = path
        self.scores = scores
        self.tokenizer = tokenizer

        self.next_candidates = []

    @property
    def score(self):
        return math.prod(self.scores) ** (
            1 / len(self.path)
        )  # TODO is normalizing needed?

    @score.setter
    def score(self, value):
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"Frontier(path={self.path}, score={self.score})"

    def __eq__(self, __o: object) -> bool:
        return self.tokenizer.decode(self.path).replace(" ", "").replace(
            "</w>", ""
        ) == self.tokenizer.decode(__o.path).replace(" ", "").replace("</w>", "")

    def __hash__(self) -> int:
        return hash(
            self.tokenizer.decode(self.path).replace(" ", "").replace("</w>", "")
        )