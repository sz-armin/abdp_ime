from __future__ import annotations

from argparse import ArgumentParser
from typing import Optional

import numpy as np
import pylcs
from nptyping import Bool, Float, Int, NDArray, Shape


class Scorer:
    def __init__(self) -> None:
        pass

    def _lcs(
        self, src: str, tgt: str, normalize: bool = True
    ) -> NDArray[Shape["3"], Float]:
        lcs = pylcs.lcs_sequence_length(src, tgt)

        if normalize:
            prec = lcs / len(tgt)
            recall = lcs / len(src)  ####????
            if (prec + recall) == 0:
                return np.array([prec, recall, 0])
            # F2
            return np.array([prec, recall, (2 * prec * recall) / (prec + recall)])
        else:
            return np.array([lcs, lcs, lcs])

    def _cer(self, src: str, tgt: str, normalize: bool = True) -> float:
        cer = pylcs.edit_distance(src, tgt)

        if normalize:
            return cer / (len(tgt))
        else:
            return cer

    def score_paralell(
        self, src_list: list[str], tgt_list: list[str]
    ) -> dict[str, float]:
        sample_count = 0
        acc = 0
        src_cum_len = 0
        tgt_cum_len = 0
        lcs = np.zeros(3)
        cer = 0
        for src_line, tgt_line in zip(src_list, tgt_list):
            src_line = src_line.strip()
            tgt_line = tgt_line.strip()
            src_line = src_line.replace(" ", "")
            tgt_line = tgt_line.replace(" ", "")

            src_len = len(src_line)
            tgt_len = len(tgt_line)
            if tgt_len == 0:
                raise ValueError(f"Empty target line, src: {src_line}, tgt: {tgt_line}")
            src_cum_len += src_len
            tgt_cum_len += tgt_len

            if src_len != 0:
                lcs += self._lcs(src_line, tgt_line)
            cer += self._cer(src_line, tgt_line)
            if src_line == tgt_line:
                acc += 1

            sample_count += 1

        dic = {
            "P": lcs[0] / sample_count,
            "R": lcs[1] / sample_count,
            "F": lcs[2] / sample_count,
            "Acc": acc / sample_count,
            "CER": cer / sample_count,
            "TgtLen": tgt_cum_len / sample_count,
            "SrcLen": src_cum_len / sample_count,
            "Count": sample_count,
        }

        return dic


if __name__ == "__main__":
    # Example usage
    # python utils/score.py --prediction-path preds.txt --label-path labels.txt
    parser = ArgumentParser()
    parser.add_argument("--prediction-path", type=str)
    parser.add_argument("--label-path", type=str)
    args = parser.parse_args()

    with open(args.prediction_path, encoding="utf-8") as src_file, open(
        args.label_path, encoding="utf-8"
    ) as tgt_file:
        src_list = src_file.readlines()
        tgt_list = tgt_file.readlines()

    scorer = Scorer()
    score = scorer.score_paralell(src_list, tgt_list)
    print(score)
