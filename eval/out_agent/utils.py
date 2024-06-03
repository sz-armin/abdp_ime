import math

import torch


class Frontier:
    def __init__(
        self,
        tokenizer,
        path=[1],
        scores=[1],
        dec_self_inc_cache=None,
        sub_c=[],
        num_decoder_layers=6,
    ) -> None:
        self.path = path
        self.scores = scores
        self.tokenizer = tokenizer

        if dec_self_inc_cache is None:
            self.dec_self_inc_cache = [torch.tensor([])] * num_decoder_layers
        else:
            self.dec_self_inc_cache = dec_self_inc_cache
        self.next_candidates = []

        if sub_c is None:
            self.sub_c = []
        else:
            self.sub_c = sub_c

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
