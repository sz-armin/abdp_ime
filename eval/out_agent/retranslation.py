from typing import Any, Sequence

import torch
from out_agent.base import OutAgentBase
from tokenizers import Tokenizer

from src.models.enc_dec import TransformerEncDec


class RetranOutAgent(OutAgentBase):
    def __init__(self, model_path: str, hparam_path: str) -> None:
        self.model = (
            TransformerEncDec(500, 16000).load_from_checkpoint(
                model_path,
                hparams_file=hparam_path,
                map_location=torch.device("cpu"),
            )
            # .cuda()
        )
        self.model.eval()

        self.tokenizer = Tokenizer.from_file("vocabs/train_kanji.json")

    def put(self, batch: Sequence[Any]) -> list[int]:
        x, align_truth = batch
        align_truth = (
            torch.tensor(align_truth, dtype=torch.long, device=self.model.device) + 1
        )

        result = self.model.predict_offline(
            x, self.tokenizer, beam_w=1
        )  # TODO two same finals

        return result
