import copy
import itertools
import logging
import math
import pickle
from copy import deepcopy
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import lightning.pytorch as pl
import torch
from torch._tensor import Tensor
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer
from torchtext.functional import to_tensor
from torchtext.vocab import Vocab

from .base import *


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
        self.save_hyperparameters()

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

        self.requires_attention = requires_attention

        self.extract_alignment = extract_alignment

        self.test_preds = []
        self.test_labels = []
        self.extracted_agns = []

        assert self.requires_alignment == self.aligned_cross_attn
        if self.wait_k_cross_attn != -1:
            assert not self.aligned_cross_attn

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
            self.src_pos_embedding = LearnedPosEmbedding(
                self.max_len, self.d_model, self.dropout
            )
            self.tgt_pos_embedding = LearnedPosEmbedding(
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
        if self.requires_alignment:
            self.align = nn.Linear(self.d_model, 1)
            for p in self.align.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            self.align_cor = nn.Linear(self.d_model * self.align_cor_layer_width, 1)
            for p in self.align_cor.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
                    
    def forward(
        self,
        x: Tensor,
        y: Tensor,
        a=None,
        src_padding_mmask=None,
        require_state: bool = False,
        has_state: bool = False,
    ):
        # encoder

        if not has_state:
            src_padding_mmask = x == self.padding_idx

            # casual enc-self-attn
            if self.causal_encoder:
                # Bx1xNxN`
                # enc_mmask.resize_as_(self.causal_mmask[: x.shape[1], : x.shape[1]])
                enc_mmask = src_padding_mmask.unsqueeze(1).logical_or(
                    self.causal_mmask[: x.shape[1], : x.shape[1]]
                ).unsqueeze(1)
            else:
                enc_mmask = src_padding_mmask.reshape(x.shape[0], 1, 1, -1)

            # local enc-self-attn
            if self.enc_attn_window > 0:
                if self.causal_encoder:
                    c = self.enc_attn_window - 1
                else:
                    assert self.enc_attn_window % 2 == 1
                    c = (self.enc_attn_window - 1) / 2
                    c = int(c)
                tmp_attn = torch.zeros(
                    (x.shape[1], x.shape[1]), dtype=torch.bool, device=enc_mmask.device
                )
                for i in range(-c, c + 1):
                    diag = tmp_attn.diagonal(i)
                    diag += True
                enc_mmask = enc_mmask.logical_or(~tmp_attn)

            if self.lerned_pos_enc:
                x = self.src_embedding(x) + self.src_pos_embedding(x)
            else:
                x = self.pos_encoding(self.src_embedding(x) * math.sqrt(self.d_model))

            enc_self_attns = []
            for layer in self.encoder:
                x, enc_self_attn = layer(
                    x, mask=enc_mmask, requires_attention=self.requires_attention
                )
                if self.requires_attention:
                    enc_self_attns.append(np.array(enc_self_attn))
            if self.requires_attention:
                logging.info("Saving encoder self attention weights...")
                np.save("enc_self_attns.npy", enc_self_attns)

        # decoder

        tgt_padding_mmask = y == self.padding_idx
        tgt_eos_mmask = y == 2

        causal_mmask = self.causal_mmask[: y.shape[1], : y.shape[1]]
        if self.lerned_pos_enc:
            y = self.tgt_embedding(y) + self.tgt_pos_embedding(y)
        else:
            y = self.pos_encoding(self.tgt_embedding(y) * math.sqrt(self.d_model))
        memory_mmask = src_padding_mmask.reshape(x.shape[0], 1, 1, -1)
        if self.aligned_cross_attn:
            a = a[:, : y.size(1)]
            r = torch.arange(memory_mmask.size(-1), device=a.device)
            align_mmask = (a.unsqueeze(-1) < r).unsqueeze(1)

            if self.aligned_cross_attn_noise > 0:
                random_mmask = (
                    torch.rand(align_mmask.shape, device=align_mmask.device)
                    < self.aligned_cross_attn_noise
                )
                align_mmask[random_mmask] = ~align_mmask[random_mmask]

            memory_mmask = memory_mmask.logical_or(align_mmask)
        if self.wait_k_cross_attn > -1:
            if not self.modified_wait_k:
                wait_k_mmask = torch.triu(
                    torch.ones(
                        y.shape[1],
                        memory_mmask.size(-1),
                        device=x.device,
                        dtype=torch.bool,
                    ),
                    self.wait_k_cross_attn,
                )
            else:  # already considers BOS
                a2 = torch.tensor(
                    [[x * self.wait_k_cross_attn for x in range(1, y.size(1) + 1)]],
                    device=x.device,
                )
                a2 = a2[:, : y.size(1)]
                r2 = torch.arange(memory_mmask.size(-1), device=a2.device)
                wait_k_mmask = (a2.unsqueeze(-1) < r2).unsqueeze(1)
            # memory_mmask.resize_as_(wait_k_mmask)
            memory_mmask = memory_mmask.logical_or(wait_k_mmask)

        dec_self_attns = []
        dec_cross_attns = []
        for layer in self.decoder:
            y, dec_self_attn, dec_cross_attn = layer(
                y,
                x,
                tgt_mmask=causal_mmask,
                memory_mmask=memory_mmask,
                requires_attention=(self.requires_attention or self.extract_alignment),
            )
            if self.requires_attention:
                dec_self_attns.append(np.array(dec_self_attn))
                dec_cross_attns.append(np.array(dec_cross_attn))
            elif self.extract_alignment:
                dec_cross_attns.append(dec_cross_attn)
        if self.requires_attention:
            np.save("dec_self_attns.npy", dec_self_attns)
            np.save("dec_cross_attns.npy", dec_cross_attns)
        if self.extract_alignment:
            # LxBxHxNtxNs
            dec_cross_attns = torch.stack(dec_cross_attns)

            pad_eos_mmask = tgt_padding_mmask.logical_or(tgt_eos_mmask)

            dec_cross_attns[
                pad_eos_mmask.reshape(1, tgt_padding_mmask.size(0), 1, -1, 1).expand(
                    dec_cross_attns.shape
                )
            ] = 0

            extracted_agn = dec_cross_attns[-2] + dec_cross_attns[-3]
            extracted_agn = extracted_agn.mean(1)
            extracted_agn[:, :, 0] = 0
            extracted_agn = extracted_agn.diff(dim=-1).argmax(-1)
            last_agn = (~src_padding_mmask).sum(-1).reshape(
                extracted_agn.size(0), -1
            ) - 2  # TODO test
            extracted_agn = extracted_agn[:, 1:]

            pad_agn = torch.tensor([[9999, 9999]], device=extracted_agn.device).expand(
                extracted_agn.size(0), -1
            )
            pred_tgt_mmask = torch.cat(
                (
                    pad_eos_mmask,
                    torch.tensor([True], device=pad_eos_mmask.device)
                    .repeat(pad_eos_mmask.size(0))
                    .unsqueeze(1),
                ),
                dim=-1,
            )  # one off
            extracted_agn[pred_tgt_mmask[:, 2:]] = -2
            extracted_agn = torch.cat(
                (
                    extracted_agn,
                    last_agn,
                    pad_agn,
                ),
                dim=-1,
            )

            self.extracted_agns.append(extracted_agn)

        y = torch.matmul(y, self.tgt_embedding.weight.T)

        # alignment

        align = torch.tensor([], device=x.device)
        align_cor = torch.tensor([], device=x.device)

        if self.requires_alignment:
            masked_x = x.masked_fill(
                mask=src_padding_mmask.unsqueeze(-1), value=0
            )  # BxNxD

            align = self.align(masked_x)  # TODO

            masked_x_for_big = torch.cat(
                (
                    masked_x,
                    torch.zeros(
                        (
                            masked_x.size(0),
                            self.align_cor_layer_width - 1,
                            masked_x.size(2),
                        ),
                        device=masked_x.device,
                    ),
                ),
                dim=1,
            )
            masked_x_for_big = masked_x_for_big.unfold(
                1, self.align_cor_layer_width, 1
            ).transpose(
                -1, -2
            )  # BxNxWxD
            masked_x_for_big = masked_x_for_big.reshape(
                masked_x_for_big.size(0), masked_x_for_big.size(1), -1
            )  # BxNxWD
            align_cor = self.align_cor(masked_x_for_big)

        if require_state:
            return y, x, src_padding_mmask, align_cor
        else:
            return y, align, align_cor

    def training_step(self, batch, batch_idx) -> Tensor:
        x, y, a = batch
        if a is not None:
            y_hat, align, align_cor = self(x, y[:, :-1], a[:, :-1]) #TODO what?
        else:
            y_hat, align, align_cor = self(x, y[:, :-1])

        if self.requires_alignment:
            assert torch.numel(align) > 0
            a_temp = a.clone()
            a_temp[a_temp == 9999] = 0
            bin_alignment = torch.zeros(x.size(0), x.size(1) + 0, device=x.device)
            bin_alignment = bin_alignment.scatter(-1, a_temp, value=1)
            bin_alignment[..., 0] = 0

            alignment_loss = nn.functional.binary_cross_entropy_with_logits(
                align.reshape(-1), bin_alignment.reshape(-1)
            )  # TODO padding?
            self.log("alignment_loss", alignment_loss, logger=True)

            alignment_cor_loss = nn.functional.binary_cross_entropy_with_logits(
                align_cor.reshape(-1), bin_alignment.reshape(-1)
            )  # TODO padding?
            self.log("alignment_cor_loss", alignment_cor_loss, logger=True)

        trans_loss = nn.functional.cross_entropy(
            y_hat.reshape(-1, self.tgt_vocab_size),
            y[:, 1:].reshape(-1),
            ignore_index=self.padding_idx,
            label_smoothing=0.0,
        )

        bsz = x.shape[0]

        self.log("trans_loss", trans_loss, logger=True, batch_size=bsz)

        if self.requires_alignment:
            loss = (
                self.alignment_loss_weight * (alignment_loss + alignment_cor_loss)
            ) + trans_loss
        else:
            loss = trans_loss

        self.log("train_loss", loss, logger=True, batch_size=bsz)
        self.log("bsz", float(bsz), logger=True, batch_size=bsz)
        self.log(
            "wpb",
            float(x.shape[0] * x.shape[1] + y.shape[0] * y.shape[1]),
            logger=True,
            batch_size=bsz,
        )

        self.log(
            "train_loss_epoch",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
            batch_size=bsz,
        )

        return loss

    # def validation_step(self, batch, batch_idx) -> Tensor:
    #     x, y, a = batch
    #     bsz = x.size(0)

    #     if a is not None:
    #         y_hat, align, align_cor = self(x, y[:, :-1], a[:, :-1])
    #     else:
    #         y_hat, align, align_cor = self(x, y[:, :-1])

    #     if self.requires_alignment:
    #         assert torch.numel(align) > 0
    #         a_temp = a.clone()
    #         a_temp += 1
    #         a_temp[a_temp == 10000] = 0
    #         bin_alignment = torch.zeros(x.size(0), x.size(1), device=x.device)
    #         bin_alignment = bin_alignment.scatter(-1, a_temp, value=1)
    #         bin_alignment[..., 0] = 0

    #         alignment_loss = nn.functional.binary_cross_entropy_with_logits(
    #             align.reshape(-1), bin_alignment.reshape(-1)
    #         )  # TODO padding?
    #         self.log(
    #             "val_alignment_loss",
    #             alignment_loss,
    #             on_step=False,
    #             on_epoch=True,
    #             logger=True,
    #             sync_dist=True,
    #         )

    #     trans_loss = nn.functional.cross_entropy(
    #         y_hat.reshape(-1, self.tgt_vocab_size),
    #         y[:, 1:].reshape(-1),
    #         ignore_index=self.padding_idx,
    #     )

    #     self.log(
    #         "val_trans_loss",
    #         trans_loss,
    #         on_step=False,
    #         on_epoch=True,
    #         logger=True,
    #         sync_dist=True,
    #         batch_size=bsz,
    #     )

    #     if self.requires_alignment:
    #         loss = alignment_loss + trans_loss
    #     else:
    #         loss = trans_loss

    #     self.log(
    #         "val_loss",
    #         loss,
    #         on_step=False,
    #         on_epoch=True,
    #         prog_bar=True,
    #         logger=True,
    #         sync_dist=True,
    #         batch_size=bsz,
    #     )

    #     return loss

    def test_step(self, batch: Tuple[Tensor,Tensor, Tensor], batch_idx) -> None:
        x, y, a = batch
        y_hat = torch.ones((x.shape[0], 1), dtype=torch.long, device=x.device)

        last, x, mask, align_cor = self(x, y_hat, require_state=True)
        last = last.argmax(dim=-1)[:, -1]
        y_hat = torch.cat((y_hat, last.unsqueeze(1)), -1)

        for step in range(200 - 2):
            last = self(x, y_hat, src_padding_mmask=mask, has_state=True)[0].argmax(
                dim=-1
            )[:, -1]
            y_hat = torch.cat((y_hat, last.unsqueeze(1)), -1)
            if (y_hat == 2).any(dim=1).all():
                break

        self.test_preds.append(y_hat.detach())
        self.test_labels.append(y.detach())

    def predict_sample(
        self,
        inp: str,
        tgt_v: Vocab,
        src_v: Optional[Vocab] = None,
    ) -> str:
        # TODO
        assert self.requires_alignment

        if src_v is not None:
            inp = list(inp)
            inp = ["<s>"] + inp + ["</s>"]  # no eos
            inp = src_v.lookup_indices(inp)
        else:
            inp = list(map(int, inp.split()))

        pred = [1]
        with torch.no_grad():
            enc_inp = torch.tensor(inp, dtype=torch.long, device=self.device).unsqueeze(
                0
            )
            mask = self.causal_mmask[: enc_inp.shape[1], : enc_inp.shape[1]]
            enc_out = self.pos_encoding(
                self.src_embedding(enc_inp) * math.sqrt(self.d_model)
            )
            for layer in self.encoder:
                enc_out, _ = layer(enc_out, mask=mask)
            align = (
                (self.align_cor(enc_out).sigmoid() > 0.5)
                .flatten()
                .nonzero()
                .squeeze(-1)
            )

            for i in align:
                if pred[-1] == 2:
                    pass
                dec_out = self.pos_encoding(
                    self.tgt_embedding(torch.tensor([pred], device=self.device))
                    * math.sqrt(self.d_model)
                )
                for layer in self.decoder:
                    dec_out, _, _ = layer(dec_out, enc_out[:, :i, :])
                dec_out = torch.matmul(dec_out, self.tgt_embedding.weight.T)
                pred.append(dec_out.argmax(dim=-1)[0, -1].item())
            return pred
            return "".join(
                filter(lambda x: x not in ["<s>", "</s>"], tgt_v.lookup_tokens(pred))
            )

    def predict_offline(self, inp: str, tgt_v, src_v=None, beam_w=1) -> str:
        assert not self.requires_alignment

        if src_v is not None:
            inp = [src_v.encode(inp).ids]
        else:  # TODO
            pass
            # inp = list(map(int, inp.split()))

        frontiers = [Frontier(tgt_v)]
        buffer = []

        with torch.no_grad():
            enc_inp = torch.tensor(inp, dtype=torch.long, device=self.device)
            enc_out = self.pos_encoding(
                self.src_embedding(enc_inp) * math.sqrt(self.d_model)
            )
            for layer in self.encoder:
                enc_out, _ = layer(enc_out, mask=None)

            front_end_state = [0] * beam_w
            for _ in range(200 - 2):
                if sum(front_end_state) == beam_w:
                    break
                else:
                    front_end_state = [0] * beam_w

                if len(frontiers) == 0:
                    # rebuild frontiers if not at initial state
                    buffer.sort(key=lambda x: x.score, reverse=True)
                    buffer = list(set(buffer))  # TODO avoid recalculation
                    buffer.sort(key=lambda x: x.score, reverse=True)
                    frontiers = buffer[:beam_w]
                    buffer = []

                c = -1
                while len(frontiers) > 0:
                    frontier = frontiers.pop(0)
                    c += 1

                    if frontier.path[-1] == 2:  # TODO meaningless
                        buffer.append(frontier)
                        front_end_state[c] = 1
                        continue

                    dec_out = self.pos_encoding(
                        self.tgt_embedding(
                            torch.tensor([frontier.path], device=self.device)
                        )
                        * math.sqrt(self.d_model)
                    )
                    causal_mmask = self.causal_mmask[
                        : dec_out.shape[1], : dec_out.shape[1]
                    ]
                    for layer in self.decoder:
                        dec_out, _, _ = layer(dec_out, enc_out, tgt_mmask=causal_mmask)
                    dec_out = torch.matmul(
                        dec_out, self.tgt_embedding.weight.T
                    ).softmax(-1)

                    top_k = dec_out.topk(k=beam_w, dim=-1)
                    top_k_ids = top_k[1][0, -1, :].tolist()
                    top_k_scores = top_k[0][0, -1, :].tolist()

                    for f_id, f_score in zip(top_k_ids, top_k_scores):
                        frontier.next_candidates.append(
                            Frontier(
                                tgt_v,
                                frontier.path + [f_id],
                                copy.deepcopy(frontier.scores) + [f_score],
                            )
                        )
                    for candidate_idx, candidate in enumerate(frontier.next_candidates):
                        for _ in range(0):  # TODO what to do?
                            if candidate.path[-1] == 2 or tgt_v.decode(
                                [candidate.path[-1]]
                            ).endswith("</w>"):
                                break

                            dec_out = self.pos_encoding(
                                self.tgt_embedding(
                                    torch.tensor([candidate.path], device=self.device)
                                )
                                * math.sqrt(self.d_model),
                            )
                            causal_mmask = self.causal_mmask[
                                : dec_out.shape[1], : dec_out.shape[1]
                            ]
                            for idx, layer in enumerate(self.decoder):
                                dec_out, _, _ = layer(
                                    dec_out, enc_out, tgt_mmask=causal_mmask
                                )
                            dec_out = torch.matmul(
                                dec_out, self.tgt_embedding.weight.T
                            ).softmax(-1)
                            top_k = dec_out.topk(k=1, dim=-1)
                            top_k_id = top_k[1][0, -1, :].item()
                            top_k_score = top_k[0][0, -1, :].item()
                            candidate.path.append(top_k_id)
                            candidate.scores.append(top_k_score)
                        buffer.append(candidate)

            # return [
            #     tgt_v.decode(x.path).replace("</w>", "").replace(" ", "")
            #     for x in buffer[:beam_w]
            # ]
            return [
                x.path for x in buffer[:beam_w]
            ]

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
        if current_step > self.max_steps:
            current_step = self.max_steps

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
        self, epoch: int, batch_idx: int, optimizer: Optimizer,
    ) -> None:
        optimizer.zero_grad(set_to_none=True)

    def _pre_alloc(self) -> None:
        # pre-allocate memory for training
        pre_x = torch.fill(
            torch.zeros((48, self.max_len), dtype=torch.long, device=self.device), 3
        )
        pre_y = torch.fill(
            torch.zeros((48, self.max_len), dtype=torch.long, device=self.device), 3
        )
        y_hat = self(pre_x, pre_y)
        loss = nn.functional.cross_entropy(
            y_hat.reshape(-1, self.tgt_vocab_size),
            pre_y.reshape(-1),
            ignore_index=3,
        )
        loss.backward()
        self.zero_grad()
        del pre_x, pre_y, y_hat, loss

    @staticmethod
    def balign_to_salign(balign):
        # TODO
        pass

    @staticmethod
    def salign_to_balign(salign, x):
        """
        Convert sequential alignment data to binary word alignment.
        salign: (batch_size, align_len)
        x: (batch_size, src_len), only used for sanity checks
        """
        a_temp = salign.clone()
        a_temp += 1
        a_temp[a_temp == 10000] = 0
        bin_alignment = torch.zeros(x.size(0), x.size(1) + 0, device=x.device)
        bin_alignment = bin_alignment.scatter(-1, a_temp, value=1)
        bin_alignment[..., 0] = 0

        return bin_alignment
