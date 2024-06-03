import collections
import copy
import math
from typing import Any, Sequence

import torch
from out_agent.base import OutAgentBase
from tokenizers import Tokenizer

from src.models.enc_dec import TransformerEncDec

from .utils import Frontier


class SemiIncAlignDecOutAgent(OutAgentBase):
    def __init__(self) -> None:
        self.model = (
            TransformerEncDec(500, 16000)
            .load_from_checkpoint(
                "logs/agn_attn_wa_6_6/checkpoints/last.ckpt",
                hparams_file=f"logs/agn_attn_wa_6_6/hparams.yaml",
            )
            .cuda()
        )
        self.model.eval()

        self.tokenizer = Tokenizer.from_file("data/vocabs/train_kanji.json")

    def put(self, batch: Sequence[Any]) -> list[int]:
        x, align_truth = batch
        align_truth = (
            torch.tensor(align_truth[0], dtype=torch.long, device=self.model.device) + 1
        )
        pred = [1]
        with torch.no_grad():
            enc_inp = torch.tensor(x, dtype=torch.long, device=self.model.device)
            mask = self.model.causal_mask[: enc_inp.shape[1], : enc_inp.shape[1]]
            enc_out = self.model.pos_encoding(
                self.model.src_embedding(enc_inp) * math.sqrt(self.model.d_model)
            )
            for layer in self.model.encoder:
                enc_out, _ = layer(enc_out, mask=mask)

            masked_x = torch.cat(
                (
                    enc_out,
                    torch.zeros(
                        (enc_out.size(0), 3, enc_out.size(2)), device=enc_out.device
                    ),
                ),
                dim=1,
            )
            masked_x = masked_x.unfold(1, 4, 1).transpose(-1, -2)  # BxNx4xD
            masked_x = masked_x.reshape(masked_x.size(0), masked_x.size(1), -1)
            align_cor = (
                self.model.align_cor(masked_x).sigmoid() > 0.5
            ).flatten().nonzero().squeeze(-1) + 1
            # align = align_truth

            for i in align_cor:
                causal_mask = self.model.causal_mask[: len(pred), : len(pred)]
                if pred[-1] == 2:
                    break
                dec_out = self.model.pos_encoding(
                    self.model.tgt_embedding(
                        torch.tensor([pred], device=self.model.device)
                    )
                    * math.sqrt(self.model.d_model)
                )
                for layer in self.model.decoder:
                    dec_out, _, _ = layer(
                        dec_out, enc_out[:, :i, :], tgt_mask=causal_mask
                    )
                dec_out = torch.matmul(dec_out, self.model.tgt_embedding.weight.T)
                pred.append(dec_out.argmax(dim=-1)[0, -1].cpu().item())

                for _ in range(10):
                    causal_mask = self.model.causal_mask[: len(pred), : len(pred)]
                    if pred[-1] == 2 or self.tokenizer.decode([pred[-1]]).endswith(
                        "</w>"
                    ):
                        break
                    dec_out = self.model.pos_encoding(
                        self.model.tgt_embedding(
                            torch.tensor([pred], device=self.model.device)
                        )
                        * math.sqrt(self.model.d_model)
                    )
                    for layer in self.model.decoder:
                        dec_out, _, _ = layer(
                            dec_out, enc_out[:, :i, :], tgt_mask=causal_mask
                        )
                    dec_out = torch.matmul(dec_out, self.model.tgt_embedding.weight.T)
                    pred.append(dec_out.argmax(dim=-1)[0, -1].cpu().item())

            return pred


class IncAlignDecOutAgent(OutAgentBase):
    def __init__(self) -> None:
        self.model = (
            TransformerEncDec(1150, 16000)
            .load_from_checkpoint(
                "logs/agn_attn_wa_6_6/checkpoints/last.ckpt",
                hparams_file=f"logs/agn_attn_wa_6_6/hparams.yaml",
            )
            .cuda()
        )
        self.model.eval()

        self.tokenizer = Tokenizer.from_file("data/vocabs/train_kanji.json")

    def put(self, batch: Sequence[Any]) -> list[int]:
        x, align_truth = batch
        align_truth = (
            torch.tensor(align_truth[0], dtype=torch.long, device=self.model.device) + 1
        )
        pred = [1]
        with torch.no_grad():
            enc_outs = torch.tensor([], device=self.device)
            aligns = torch.tensor([], dtype=torch.long, device=self.device)
            enc_inc_cache = [torch.tensor([])] * (self.num_encoder_layers)
            dec_self_inc_cache = [torch.tensor([])] * (self.num_encoder_layers)
            for i, kana in enumerate(x[0], start=1):
                enc_inp = torch.tensor([[kana]], dtype=torch.long, device=self.device)
                enc_out = self.pos_encoding(
                    self.src_embedding(enc_inp) * math.sqrt(self.d_model), fixed_pos=i
                )
                for i2, layer in enumerate(self.encoder):
                    enc_out, _ = layer(enc_out, inc_cache=enc_inc_cache, cache_idx=i2)
                enc_outs = torch.cat((enc_outs, enc_out[:, -1, :]), dim=0)
                align = self.align(enc_out).sigmoid() > 0.5
                aligns = torch.hstack((aligns, align))

                if align != 1:
                    continue

                if pred[-1] == 2:
                    break

                dec_out = self.pos_encoding(
                    self.tgt_embedding(torch.tensor([[pred[-1]]], device=self.device))
                    * math.sqrt(self.d_model)
                )
                for i2, layer in enumerate(self.decoder):
                    dec_out, _, _ = layer(
                        dec_out,
                        enc_outs,
                        self_inc_cache=dec_self_inc_cache,
                        cache_idx=i2,
                    )
                dec_out = torch.matmul(dec_out, self.tgt_embedding.weight.T)
                pred.append(dec_out.topk(k=3, dim=-1)[1][0, -1, 0].item())

                for _ in range(10):
                    if pred[-1] == 2 or self.tokenizer.decode([pred[-1]]).endswith(
                        "</w>"
                    ):
                        break
                    dec_out = self.pos_encoding(
                        self.tgt_embedding(
                            torch.tensor([[pred[-1]]], device=self.device)
                        )
                        * math.sqrt(self.d_model)
                    )
                    for layer in self.decoder:
                        dec_out, _, _ = layer(
                            dec_out,
                            enc_outs,
                            self_inc_cache=dec_self_inc_cache,
                            cache_idx=i2,
                        )
                    dec_out = torch.matmul(dec_out, self.tgt_embedding.weight.T)

                    pred.append(dec_out.topk(k=3, dim=-1)[1][0, -1, 0].item())

                return pred


class IncAlignDecWCOutAgent(OutAgentBase):
    def __init__(
        self,
        model_path: str,
        hparam_path: str,
        correction: bool = True,
        beam_w=1,
        retrun_aligns=False,
        use_ref_align=False,
    ) -> None:
        self.model = TransformerEncDec(500, 16000).load_from_checkpoint(
            model_path, hparams_file=hparam_path, map_location=torch.device("cpu")
        )
        self.model.eval()

        self.tokenizer = Tokenizer.from_file("data/vocabs/train_kanji.json")

        self.max_subword_len = 10
        self.correction = correction

        self.beam_w = beam_w

        self.return_aligns = retrun_aligns
        self.use_ref_align = use_ref_align

    def put(self, batch: Sequence[Any]) -> list[int]:
        input_ids, ref_aligns = batch
        ref_aligns = ref_aligns[0]

        with torch.no_grad():
            if input_ids[0] == 1:
                self.enc_outs = torch.tensor([], device=self.model.device)
                self.aligns = torch.tensor(
                    [], dtype=torch.long, device=self.model.device
                )
                self.enc_inc_cache = [torch.tensor([])] * (
                    self.model.num_encoder_layers
                )

                self.frontiers = [
                    Frontier(
                        self.tokenizer, num_decoder_layers=self.model.num_decoder_layers
                    )
                ]
                self.buffer = []
                if self.return_aligns or self.use_ref_align:
                    self.ref_aligns_bin = []
                    self.aligns_bin = []
                    self.cor_aligns_bin = []

                self.kana_i = 0

            # keep the loop for compatibility with AllInAgent
            for kana in input_ids:
                # handle alignment correction for the last k
                if kana == 2:
                    for _ in range(self.model.align_cor_layer_width - 3):
                        input_ids.append(-1)
                    input_ids.append(-2)
                    input_ids.append(-1)

                self.kana_i += 1
                last_enc_inps = [9999]

                # encoder
                if kana >= 0:
                    enc_inp = torch.tensor(
                        [[kana]], dtype=torch.long, device=self.model.device
                    )
                    enc_out = self.model.pos_encoding(
                        self.model.src_embedding(enc_inp)
                        * math.sqrt(self.model.d_model),
                        fixed_pos=self.kana_i,
                    )
                    for i, layer in enumerate(self.model.encoder):
                        enc_out, _ = layer(
                            enc_out, inc_cache=self.enc_inc_cache, cache_idx=i
                        )
                    self.enc_outs = torch.cat((self.enc_outs, enc_out[:, -1, :]), dim=0)

                    align = self.model.align(enc_out).sigmoid() > 0.5
                    self.aligns = torch.hstack((self.aligns, align))
                    if self.return_aligns or self.use_ref_align:
                        self.aligns_bin.append(int(align.flatten().item()))
                        ref_align = (
                            1 if (self.kana_i - 1 in ref_aligns) or (kana == 2) else 0
                        )  # TODO 2?
                        self.ref_aligns_bin.append(ref_align)

                        if self.use_ref_align:
                            assert self.correction == False
                            align = torch.tensor(
                                [[[ref_align]]], device=self.model.device
                            )
                else:
                    self.enc_outs = torch.cat(
                        (
                            self.enc_outs,
                            torch.zeros(
                                (1, self.model.d_model), device=self.model.device
                            ),
                        ),
                        dim=0,
                    )
                    align = torch.tensor([[[0]]], device=self.model.device)
                    self.aligns = torch.hstack((self.aligns, align))

                # handle alignemt correction
                if (
                    cor_i := (self.kana_i - self.model.align_cor_layer_width)
                ) >= 0 and self.correction:
                    align_cor = (
                        self.model.align_cor(
                            self.enc_outs[cor_i : self.kana_i, :].reshape(1, -1)
                        ).sigmoid()
                        > 0.5
                    )
                    if self.return_aligns:
                        self.cor_aligns_bin.append(int(align_cor.flatten().item()))
                    if kana == -2:
                        align_cor = torch.tensor([[1]], device=self.model.device)

                    if align_cor[0] != self.aligns[0, cor_i, 0]:
                        steps_diff = (
                            self.aligns[:, -self.model.align_cor_layer_width : -1]
                            == True
                        ).sum()

                        self.aligns[0, cor_i] = align_cor
                        aligns_i = (
                            self.aligns.flatten().nonzero().squeeze(-1) + 1
                        ).tolist()

                        last_enc_inps = aligns_i[
                            self.aligns[
                                :, : -self.model.align_cor_layer_width, :
                            ].sum() :
                        ]

                        for frontier in self.buffer:
                            n_mis_out_tokens = sum(frontier.sub_c[-steps_diff:])

                            if steps_diff == 0:
                                n_mis_out_tokens = -9999
                            else:
                                frontier.sub_c = frontier.sub_c[:-steps_diff]

                            frontier.path = frontier.path[:-n_mis_out_tokens]
                            frontier.scores = frontier.scores[:-n_mis_out_tokens]

                            for idx, layer in enumerate(frontier.dec_self_inc_cache):
                                if layer.dim() < 3:
                                    break
                                frontier.dec_self_inc_cache[idx] = layer[
                                    :, :-n_mis_out_tokens, :
                                ]

                if align != 1 and last_enc_inps == [9999]:  # i.e. no corrections
                    continue

                # if self.preds[-1] == 2:
                #     break

                while len(last_enc_inps) > 0:
                    self.last_enc_inp = last_enc_inps.pop(0)

                    if len(self.frontiers) == 0:
                        # rebuild frontiers if not at initial state
                        self.buffer.sort(key=lambda x: x.score, reverse=True)
                        self.buffer = list(set(self.buffer))  # TODO avoid recalculation
                        self.buffer.sort(key=lambda x: x.score, reverse=True)
                        self.frontiers = self.buffer[: self.beam_w]
                        self.buffer = []

                    while len(self.frontiers) > 0:
                        frontier = self.frontiers.pop(0)

                        if frontier.path[-1] == 2:  # TODO meaningless
                            break

                        self.__decode_step(frontier)

            self.buffer.sort(key=lambda x: x.score, reverse=True)
            # self.frontiers = self.buffer[: self.beam_w]

            if self.return_aligns and kana < 0:
                return (
                    [x.path for x in self.buffer[: self.beam_w]],
                    self.ref_aligns_bin,
                    self.aligns_bin,
                    self.cor_aligns_bin,
                )
            else:
                return [x.path for x in self.buffer[: self.beam_w]]

    def __decode_step(self, frontier):
        dec_out = self.model.pos_encoding(
            self.model.tgt_embedding(
                torch.tensor([[frontier.path[-1]]], device=self.model.device)
            )
            * math.sqrt(self.model.d_model),
            fixed_pos=len(frontier.path),
        )
        for i, layer in enumerate(self.model.decoder):
            dec_out, _, _ = layer(
                dec_out,
                self.enc_outs[: self.last_enc_inp, :],
                self_inc_cache=frontier.dec_self_inc_cache,
                cache_idx=i,
            )
        dec_out = torch.matmul(dec_out, self.model.tgt_embedding.weight.T).softmax(-1)

        top_k = dec_out.topk(k=self.beam_w, dim=-1)
        top_k_ids = top_k[1][0, -1, :].tolist()
        top_k_scores = top_k[0][0, -1, :].tolist()

        for f_id, f_score in zip(top_k_ids, top_k_scores):
            frontier.sub_c.append(1)
            frontier.next_candidates.append(
                Frontier(
                    self.tokenizer,
                    frontier.path + [f_id],
                    copy.deepcopy(frontier.scores) + [f_score],
                    # frontier1.score * f_score,
                    copy.deepcopy(frontier.dec_self_inc_cache),
                    copy.deepcopy(frontier.sub_c),
                    num_decoder_layers=self.model.num_decoder_layers,
                )
            )

        for candidate_idx, candidate in enumerate(frontier.next_candidates):
            for _ in range(10):
                if candidate.path[-1] == 2 or self.tokenizer.decode(
                    [candidate.path[-1]]
                ).endswith("</w>"):
                    break

                candidate.sub_c[-1] += 1

                dec_out = self.model.pos_encoding(
                    self.model.tgt_embedding(
                        torch.tensor([[candidate.path[-1]]], device=self.model.device)
                    )
                    * math.sqrt(self.model.d_model),
                    fixed_pos=len(candidate.path),
                )
                for i2, layer in enumerate(self.model.decoder):
                    dec_out, _, _ = layer(
                        dec_out,
                        self.enc_outs[: self.last_enc_inp, :],
                        self_inc_cache=candidate.dec_self_inc_cache,
                        cache_idx=i2,
                    )
                dec_out = torch.matmul(
                    dec_out, self.model.tgt_embedding.weight.T
                ).softmax(-1)
                top_k = dec_out.topk(k=1, dim=-1)
                top_k_id = top_k[1][0, -1, :].item()
                top_k_score = top_k[0][0, -1, :].item()
                candidate.path.append(top_k_id)
                candidate.scores.append(top_k_score)
            self.buffer.append(candidate)
