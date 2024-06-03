from pathlib import Path

import numpy as np


class AlignmentExtractor:
    def __init__(self):
        pass

    def align_from_sample(self, sample):
        """
        There are two ways to interpret the outputed alignment:
        - Considering boundaries, it refers to the right boundary
        of the last source element of the corresponding token.
        - Considering elemnts, it refers to the first source
        element corresponding to the next token.

        Output requires two right paddings, set as "9999".
        Besides that, we do not take BOS and EOS into account.
        """
        source, target = sample

        alignment = np.cumsum([len(item) for item in source]) - 0
        alignment = np.hstack((alignment, [9999, 9999]))
        alignment = alignment.tolist()

        assert len(target) > 0
        assert len(alignment) == len(target) + 2, f"{source} != {target} + 2"

        alignment = " ".join([str(item) for item in alignment])

        return alignment

    def align_from_file(self, src_file_path, tgt_file_path):
        src_file_path = Path(src_file_path)
        with open(src_file_path, "r", encoding="utf-8") as src_inp_file, open(
            tgt_file_path, "r", encoding="utf-8"
        ) as tgt_inp_file, open(
            src_file_path.with_suffix(".align"), "w", encoding="utf-8"
        ) as out_file:
            for src_line, tgt_file in zip(src_inp_file, tgt_inp_file):
                src_line = src_line.strip().split(" ")
                tgt_line = tgt_file.strip().split(" ")
                out_file.write(self.align_from_sample((src_line, tgt_line)) + "\n")


if __name__ == "__main__":
    # Example run:
    # python preprocess/mec2alignment.py data/train.kana data/train.kanji
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("kana_path", help="Path to the kana file")
    parser.add_argument("kanji_path", help="Path to the kanji file")
    args = parser.parse_args()

    exctractor = AlignmentExtractor()
    exctractor.align_from_file(args.kana_path, args.kanji_path)