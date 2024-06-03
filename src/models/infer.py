# import collections
# import copy
# import itertools
# import logging
# import math
# import pickle
from copy import deepcopy
import json
from pathlib import Path
from typing import (Any, Callable, Dict, List, Optional, Sequence, Tuple, Type,
                    Union)

import lightning.pytorch as pl
from tokenizers import Tokenizer
# import numpy as np
import torch
# from tokenizers import Tokenizer
# from torch._tensor import Tensor
# from torch.optim.adamw import AdamW
# from torch.optim.lr_scheduler import LambdaLR
# from torch.optim.optimizer import Optimizer
# from torchtext.functional import to_tensor
# from torchtext.vocab import Vocab

# from src.models.enc_dec import TransformerEncDec

from .base import *
# from .utils import Frontier


class TransformerEncDec(pl.LightningModule):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        activation: Callable = nn.ReLU,
        dropout: float = 0.1,
        max_len: int = 1024,
        padding_idx: int = 3,
        lr: float = 5e-4,
        lr_factor: int = 1,
        warmup: int = 2500,
        weight_decay: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.98),
        eps: float = 1e-9,
        max_steps: int = 54300,
        lerned_pos_enc: bool = False,
        causal_encoder: bool = False,
        enc_attn_window: int = -1,
        aligned_cross_attn: bool = False,
        aligned_cross_attn_noise: float = 0.0,
        wait_k_cross_attn: int = -1,
        modified_wait_k=False,
        requires_alignment: bool = False,
        alignment_loss_weight: float = 20.0,
        align_cor_layer_width: int = 4,
        requires_attention: bool = False,
        extract_alignment: bool = False,
    ) -> None:
        super().__init__()
        # self.save_hyperparameters()

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout
        self.activation = activation
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.max_len = max_len
        self.warmup = warmup
        self.padding_idx = padding_idx
        self.lr = lr
        self.lr_factor = lr_factor
        self.max_steps = max_steps
        self.weight_decay = weight_decay
        self.betas = betas
        self.eps = eps
        self.lerned_pos_enc = lerned_pos_enc

        self.causal_encoder = causal_encoder
        self.enc_attn_window = enc_attn_window
        self.aligned_cross_attn = aligned_cross_attn
        self.aligned_cross_attn_noise = aligned_cross_attn_noise
        self.wait_k_cross_attn = wait_k_cross_attn
        self.modified_wait_k = modified_wait_k
        self.requires_alignment = requires_alignment
        self.alignment_loss_weight = alignment_loss_weight
        self.align_cor_layer_width = align_cor_layer_width

        self.test_preds = []
        self.test_labels = []
        self.extracted_agns = []

        self.example_input_array = (
            torch.randint(low=1, high=self.src_vocab_size - 10, size=(16, 200)),
            torch.randint(low=1, high=self.tgt_vocab_size - 10, size=(16, 200)),
            torch.randint(low=1, high=200, size=(16, 200)),
        )

        causal_mmask = torch.triu(
            torch.ones(self.max_len, self.max_len, dtype=torch.bool), diagonal=1
        )
        self.register_buffer("causal_mmask", causal_mmask)

        self.src_embedding = nn.Embedding(
            src_vocab_size, self.d_model, padding_idx=self.padding_idx
        )
        self.tgt_embedding = nn.Embedding(
            tgt_vocab_size, self.d_model, padding_idx=self.padding_idx
        )

        self.pos_encoding = PositionalEncoding(
                self.d_model, self.dropout, self.max_len
            )
        self.encoder = nn.ModuleList(
            deepcopy(
                TransformerEncoderLayer(
                    d_model=self.d_model,
                    nhead=self.nhead,
                    dropout=self.dropout,
                    activation=self.activation,
                )
            )
            for _ in range(self.num_encoder_layers)  # TODO
        )
        self.decoder = nn.ModuleList(
            deepcopy(
                TransformerDecoderLayer(
                    d_model=self.d_model,
                    nhead=self.nhead,
                    dropout=self.dropout,
                    activation=self.activation,
                )
            )
            for _ in range(self.num_decoder_layers)
        )

        self.align = nn.Linear(self.d_model, 1)
        self.align_cor = nn.Linear(self.d_model * self.align_cor_layer_width, 1)


        ####

        # self.tokenizer = Tokenizer.from_file("data/vocabs/train_kanji.json")


        self.enc_outs = torch.tensor([])
        self.aligns = torch.tensor([], dtype=torch.long)
        self.enc_inc_cache = [torch.tensor([])] * (self.num_encoder_layers)
        self.dec_self_inc_cache = [torch.tensor([])] * (
            self.num_decoder_layers
        )
        self.kana_i = 0

        self.preds: List[int] = [1]
        self.sub_c: List[int] = [1] # hack, for some reason we can't use empty lists in init

        with open("data/vocabs/train_kanji.json", "r") as f:
            data = json.load(f)
            self.eow = torch.tensor(list(map(lambda x: 1 if x.endswith("</w>") else 0, list(data["model"]["vocab"]))))
                    
    def forward(self, x: int) -> List[int]:
        if x == 1: #torch.empty
            self.enc_outs = torch.tensor(torch.jit.annotate(List[float], []))
            self.aligns = torch.tensor(torch.jit.annotate(List[int], []), dtype=torch.long)
            self.enc_inc_cache = [torch.tensor(torch.jit.annotate(List[float], []))] * (self.num_encoder_layers)
            self.dec_self_inc_cache = [torch.tensor(torch.jit.annotate(List[float], []))] * (
                self.num_decoder_layers
            )
            self.kana_i = 0
            self.preds = torch.jit.annotate(List[int], [1])
            self.sub_c = torch.jit.annotate(List[int], [])

        ###

        self.kana_i += 1
        last_enc_inps = [9999]

        # encoder
        enc_inp = torch.tensor(
            [[x]], dtype=torch.long
        )
        enc_out = self.pos_encoding(
            self.src_embedding(enc_inp) * math.sqrt(self.d_model),
            fixed_pos=self.kana_i,
        )
        for i, layer in enumerate(self.encoder):
            enc_out, _ = layer(enc_out, inc_cache=self.enc_inc_cache, cache_idx=i)
        self.enc_outs = torch.cat((self.enc_outs, enc_out[:, -1, :]), dim=0)

        align = self.align(enc_out).sigmoid() > 0.5
        self.aligns = torch.hstack((self.aligns, align))

        # handle alignemt correction
        cor_i = self.kana_i - self.align_cor_layer_width
        if cor_i  >= 0:
            align_cor = (
                self.align_cor(
                    self.enc_outs[cor_i:self.kana_i, :].reshape(1, -1)
                ).sigmoid()
                > 0.5
                    )
            
            if align_cor[0] != self.aligns[0, cor_i, 0]:
                steps_diff = (
                    self.aligns[:, -self.align_cor_layer_width : -1] == torch.tensor(True)
                ).sum()

                self.aligns[0, cor_i] = align_cor
                aligns_i = torch.jit.annotate(List[int],(self.aligns.flatten().nonzero().squeeze(-1) + 1).tolist())

                n_mis_out_tokens = sum(self.sub_c[-steps_diff:])

                if steps_diff == 0:
                    n_mis_out_tokens = -9999
                else:
                    self.sub_c = self.sub_c[:-steps_diff]

                self.preds = self.preds[:-n_mis_out_tokens]
            #     last_enc_inps = aligns_i[
            #         self.eow[self.preds].sum() :
            #     ] #TODO
                
            #     for idx in range(self.num_decoder_layers):
            #         if self.dec_self_inc_cache[idx].dim() > 3:
            #             self.dec_self_inc_cache[idx] = self.dec_self_inc_cache[idx][
            #                 :, :-n_mis_out_tokens, :
            #             ]

            # if align != 1 and last_enc_inps == 9999:  # i.e. no corrections
            #     return torch.jit.annotate(List[int], [])

            # if self.preds[-1] == 2:
            #     return torch.jit.annotate(List[int], [])

            # while len(last_enc_inps) > 0:
            #     self.last_enc_inp = last_enc_inps.pop(0)

                # self.__decode_step()
                # self.sub_c.append(1)

                # for _ in range(self.max_subword_len):
                #     if self.preds[-1] == 2 or self.tokenizer.decode(
                #         [self.preds[-1]]
                #     ).endswith("</w>"):
                #         break
                #     self.sub_c[-1] += 1
                #     self.__decode_step()

        # self.preds = torch.jit.annotate(List[int], [1])
        return self.preds
