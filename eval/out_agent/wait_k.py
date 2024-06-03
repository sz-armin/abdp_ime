import math
from typing import Any, Sequence

import torch
from out_agent.base import OutAgentBase
from tokenizers import Tokenizer

from src.models.enc_dec import TransformerEncDec

from .utils import Frontier


class WaitKOutAgent(OutAgentBase):
    def __init__(
        self, model_path: str, hparam_path: str, k: int = 4, beam_w=1
    ) -> None:
        self.model = TransformerEncDec(500, 16000).load_from_checkpoint(
            model_path, hparams_file=hparam_path, map_location=torch.device("cpu")
        )
        self.model.eval()

        self.tokenizer = Tokenizer.from_file("vocabs/train_kanji.json")

        self.k = k
        self.beam_w = beam_w

    def put(self, batch: Sequence[Any]) -> list[int]:
        # TODO fix after final inp
        input_ids, align_truth = batch

        with torch.no_grad():
            if input_ids[0] == [1]:
                self.enc_outs = torch.tensor([], device=self.model.device)
                self.enc_inc_cache = [torch.tensor([], device=self.model.device)] * (
                    self.model.num_encoder_layers
                )
                self.dec_self_inc_cache = [
                    torch.tensor([], device=self.model.device)
                ] * (self.model.num_decoder_layers)

                self.buffer = [1]

                self.kana_i = 0

            if self.buffer[-1] == 2:
                return self.buffer

            if (
                len(input_ids[0]) < self.k and input_ids[0][-1] != 2
            ):  # TODO sentences shorter than k
                self.kana_i += 1

                enc_inp = torch.tensor(
                    [[input_ids[0][-1]]], dtype=torch.long, device=self.model.device
                )
                enc_out = self.model.pos_encoding(
                    self.model.src_embedding(enc_inp) * math.sqrt(self.model.d_model),
                    fixed_pos=self.kana_i,
                )
                for i2, layer in enumerate(self.model.encoder):
                    enc_out, _ = layer(
                        enc_out, inc_cache=self.enc_inc_cache, cache_idx=i2
                    )
                self.enc_outs = torch.cat((self.enc_outs, enc_out[:, -1, :]), dim=0)

                return [1]

            # keep the loop for compatibility with AllInAgent
            kana = input_ids[0][-1]
            self.kana_i += 1

            enc_inp = torch.tensor([[kana]], dtype=torch.long, device=self.model.device)
            enc_out = self.model.pos_encoding(
                self.model.src_embedding(enc_inp) * math.sqrt(self.model.d_model),
                fixed_pos=self.kana_i,
            )
            for i2, layer in enumerate(self.model.encoder):
                enc_out, _ = layer(enc_out, inc_cache=self.enc_inc_cache, cache_idx=i2)
            self.enc_outs = torch.cat((self.enc_outs, enc_out[:, -1, :]), dim=0)

            if self.buffer[-1] == 2:
                return self.buffer  # TODO

            dec_out = self.model.pos_encoding(
                self.model.tgt_embedding(
                    torch.tensor([[self.buffer[-1]]], device=self.model.device)
                )
                * math.sqrt(self.model.d_model),
                fixed_pos=len(self.buffer),
            )
            for i2, layer in enumerate(self.model.decoder):
                dec_out, _, _ = layer(
                    dec_out,
                    self.enc_outs,
                    self_inc_cache=self.dec_self_inc_cache,
                    cache_idx=i2,
                )
            dec_out = torch.matmul(dec_out, self.model.tgt_embedding.weight.T)

            self.buffer.append(dec_out.topk(k=1, dim=-1)[1][0, -1, 0].item())

            if kana == 2:
                dec_c = 0
                while self.buffer[-1] != 2:
                    if dec_c > 25:  # TODO max
                        break
                    dec_out = self.model.pos_encoding(
                        self.model.tgt_embedding(
                            torch.tensor([[self.buffer[-1]]], device=self.model.device)
                        )
                        * math.sqrt(self.model.d_model),
                        fixed_pos=len(self.buffer),
                    )
                    for i2, layer in enumerate(self.model.decoder):
                        dec_out, _, _ = layer(
                            dec_out,
                            self.enc_outs,
                            self_inc_cache=self.dec_self_inc_cache,
                            cache_idx=i2,
                        )
                    dec_out = torch.matmul(dec_out, self.model.tgt_embedding.weight.T)

                    self.buffer.append(dec_out.topk(k=1, dim=-1)[1][0, -1, 0].item())

                    dec_c += 1

            return self.buffer
