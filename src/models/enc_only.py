import logging
import math
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import lightning.pytorch as pl
import torch
from base import *
from torch import nn, optim
from torch._tensor import Tensor
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer


class TransformerEncOnly(pl.LightningModule):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        activation: Type[torch.nn.modules.activation.ReLU] = nn.ReLU,
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
        requires_attention: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout
        self.activation = activation
        self.num_encoder_layers = num_encoder_layers
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

        self.requires_attention = requires_attention

        self.test_preds = []
        self.test_labels = []

        causal_mask = torch.triu(
            torch.ones(self.max_len, self.max_len, dtype=torch.bool), diagonal=1
        )
        self.register_buffer("causal_mask", causal_mask)

        self.src_embedding = nn.Embedding(
            src_vocab_size, self.d_model, padding_idx=self.padding_idx
        )
        for p in self.src_embedding.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.tgt_embedding = nn.Embedding(
            tgt_vocab_size, self.d_model, padding_idx=self.padding_idx
        )
        for p in self.tgt_embedding.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        if self.lerned_pos_enc:
            self.pos_embedding = LearnedPosEmbedding(
                self.max_len, self.d_model, self.dropout
            )
        else:
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

    def forward(
        self,
        x: Tensor,
        src_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        src_padding_mask = x == self.padding_idx

        # casual enc-self-attn
        if self.causal_encoder:
            # Bx1xNxN`
            enc_mask = torch.logical_or(
                src_padding_mask.unsqueeze(1),
                self.causal_mask[: x.shape[1], : x.shape[1]],
            ).unsqueeze(1)
        else:
            enc_mask = src_padding_mask.reshape(x.shape[0], 1, 1, -1)

        # local enc-self-attn
        if self.enc_attn_window > 0:
            if self.causal_encoder:
                c = self.enc_attn_window - 1
            else:
                assert self.enc_attn_window % 2 == 1
                c = (self.enc_attn_window - 1) / 2
            c = int(c)
            tmp_attn = torch.zeros(
                (x.shape[1], x.shape[1]), dtype=torch.bool, device=enc_mask.device
            )
            for i in range(-c, c + 1):
                diag = tmp_attn.diagonal(i)
                diag += True
            enc_mask = torch.logical_or(enc_mask, ~tmp_attn)

        if self.lerned_pos_enc:
            x = self.src_embedding(x) + self.src_pos_embedding(x)
        else:
            x = self.pos_encoding(self.src_embedding(x) * math.sqrt(self.d_model))

        enc_self_attns = []
        for layer in self.encoder:
            x, enc_self_attn = layer(x, mask=enc_mask)
            if self.requires_attention:
                enc_self_attns.append(np.array(enc_self_attn))
        if self.requires_attention:
            np.save("enc_self_attns.npy", enc_self_attns)

        # if self.lerned_pos_enc:
        #     y = self.tgt_embedding(y) + self.pos_embedding(y)
        # else:
        #     y = self.pos_encoding(self.tgt_embedding(y) * math.sqrt(self.d_model))

        y = torch.matmul(x, self.tgt_embedding.weight.T)

        return y

    # def _pre_alloc(self):
    #     # pre-allocate memory for training
    #     pre_x = torch.fill(
    #         torch.zeros((48, self.max_len), dtype=torch.long, device=self.device), 3
    #     )
    #     pre_y = torch.fill(
    #         torch.zeros((48, self.max_len), dtype=torch.long, device=self.device), 3
    #     )
    #     y_hat = self(pre_x, pre_y)
    #     loss = nn.functional.cross_entropy(
    #         y_hat.reshape(-1, self.tgt_vocab_size),
    #         pre_y.reshape(-1),
    #         ignore_index=3,
    #     )
    #     loss.backward()
    #     self.zero_grad()
    #     del pre_x, pre_y, y_hat, loss

    def training_step(self, batch, batch_idx) -> Tensor:
        x, y = batch
        y_hat = self(x)

        loss = nn.functional.cross_entropy(
            y_hat.reshape(-1, self.tgt_vocab_size),
            y.reshape(-1),
            ignore_index=self.padding_idx,
            label_smoothing=0.1,
        )

        self.log("train_loss", loss, logger=True)
        self.log("bsz", float(x.shape[0]), logger=True)
        self.log(
            "wpb", float(x.shape[0] * x.shape[1] + y.shape[0] * y.shape[1]), logger=True
        )

        self.log(
            "train_loss_epoch",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch, batch_idx) -> Tensor:
        x, y = batch
        y_hat = self(x)

        loss = nn.functional.cross_entropy(
            y_hat.reshape(-1, self.tgt_vocab_size),
            y.reshape(-1),
            ignore_index=self.padding_idx,
        )

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return loss

    def test_step(self, batch, batch_idx) -> None:
        x, y = batch

        y_hat = self(x)

        self.test_preds.append(y_hat.detach().argmax(-1))
        self.test_labels.append(y.detach())

    def configure_optimizers(
        self,
    ) -> Tuple[List[AdamW], List[Dict[str, Union[str, LambdaLR]]]]:
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.98),
            eps=self.eps,
        )

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer, lr_lambda=self.rate_cosine
        )

        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    def rate_trans(self, current_step: int) -> float:
        if current_step == 0:
            current_step = 1
        # current_step += 200
        return self.lr_factor * (
            self.d_model ** (-0.5)
            * min(current_step ** (-0.5), current_step * self.warmup ** (-1.5))
        )

    def rate_cosine(self, current_step) -> float:
        if current_step < self.warmup:
            return float(current_step) / float(self.warmup)
        else:
            return (
                math.cos(
                    (10 * (current_step - self.warmup))
                    / (math.pi * (self.max_steps - self.warmup))
                )
                + 1
            ) / 2

    def optimizer_zero_grad(
        self, epoch: int, batch_idx: int, optimizer: Optimizer, optimizer_idx: int
    ) -> None:
        optimizer.zero_grad(set_to_none=True)
