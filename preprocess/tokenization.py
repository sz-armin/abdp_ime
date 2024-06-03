import logging
import pathlib
import shutil
from typing import Union

import sentencepiece as spm
from tokenizers import Tokenizer, normalizers
from tokenizers.models import BPE, Unigram, WordLevel
from tokenizers.normalizers import Strip
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordLevelTrainer

logging.basicConfig(level=logging.INFO)

# TODO merge with pre-tokenizers
# TODO move to a class


def fix_sample_bpe_alignemnt(sample: str, inp_alignment: str):
    sample = sample.strip().split()
    sample = list(filter(lambda x: x != "<s>" and x != "</s>", sample))
    inp_alignment = list(map(int, inp_alignment.strip().split()))[:-2]

    out_alignment = []
    i = 0
    for t in sample:
        out_alignment.append(inp_alignment[i])
        if t.endswith("</w>"):
            i += 1

    assert len(sample) == len(out_alignment)

    out_alignment.extend(["9999", "9999"])
    out_alignment = " ".join(map(str, out_alignment))

    return out_alignment


def train_tokenizer(
    files: list[str], vocab_size: int, algorithm: str = "wordlevel"
) -> Tokenizer:
    if algorithm == "sp_unigram":
        # TODO complete this
        training_data = []
        for file in files:
            with open(file, "r", encoding="utf-8") as f:
                training_data.extend(f.readlines())
        spm.SentencePieceTrainer.train(
            input=training_data,
            model="unigram",
            model_prefix="spm",
            vocab_size=vocab_size,
            pad_id=3,
            character_coverage=1.0,
            add_dummy_prefix=False,
            allow_whitespace_only_pieces=False,
            eos_piece="</s>",
            normalization_rule_name="identity",
            remove_extra_whitespaces=True,
            split_digits=True,
        )

        shutil.move("spm.model", "data/vocabs/spm.model")
        shutil.move("spm.vocab", "data/vocabs/spm.vocab")

        sp_tokenizer = spm.SentencePieceProcessor(model_file="data/vocabs/spm.model")

        return sp_tokenizer

    if algorithm == "wordlevel":
        tokenizer = Tokenizer(WordLevel(unk_token="<unk>"))
        tokenizer.pre_tokenizer = WhitespaceSplit()

        trainer = WordLevelTrainer(
            special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
            vocab_size=vocab_size,
        )
    elif algorithm == "bpe":
        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = WhitespaceSplit()

        trainer = BpeTrainer(
            special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
            vocab_size=vocab_size,
            end_of_word_suffix="</w>",
        )
    elif algorithm == "unigram":
        tokenizer = Tokenizer(Unigram())

        trainer = UnigramTrainer(
            special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
            vocab_size=vocab_size,
            unk_token="<unk>",
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        special_tokens=[("<s>", 1), ("</s>", 2)],
    )
    tokenizer.normalizer = normalizers.Sequence([Strip()])

    # add_token is buggy!
    # tokenizer.add_tokens(["▁"])

    print(files)
    tokenizer.train(files, trainer)

    return tokenizer


def convert_sample_to_ids(tokenizer: Tokenizer, samples: list[str]) -> None:
    return tokenizer.encode_batch(samples)


def convert_files(files: list[str], tokenizer: Union[Tokenizer, str]) -> None:
    for file in files:
        logging.info("Converting %s", file)

        file = pathlib.Path(file)
        id_out_path = file.with_name(file.stem + "_ids" + file.suffix)
        piece_out_path = file.with_name(file.stem + "_pieces" + file.suffix)

        if isinstance(tokenizer, Tokenizer):
            pass
        elif isinstance(tokenizer, str):
            logging.info("Loading tokenizer from %s", tokenizer)
            tokenizer = Tokenizer.from_file(tokenizer)
        else:
            raise ValueError("tokenizer must be a Tokenizer or path to a Tokenizer.")

        with open(file, "r", encoding="utf-8") as f:
            samples = f.readlines()
        samples = convert_sample_to_ids(tokenizer, samples)
        with open(id_out_path, "w", encoding="utf-8") as id_f, open(
            piece_out_path, "w", encoding="utf-8"
        ) as piece_f:
            for s in samples:
                id_f.write(" ".join(map(str, s.ids)) + "\n")
                piece_f.write(" ".join(map(str, s.tokens)) + "\n")


def train_convert_files(
    tokenizer_path: str,
    train_files: list[str],
    files_to_conv: list[str],
    vocab_size: int,
    algorithm: str = "wordlevel",
) -> None:
    tokenizer_path = pathlib.Path(tokenizer_path)
    if tokenizer_path.exists():
        tokenizer = str(tokenizer_path)
    else:
        tokenizer = train_tokenizer(train_files, vocab_size, algorithm)
        tokenizer.save(str(tokenizer_path))
    convert_files(files_to_conv, tokenizer)

    # correct alignment for bpe
    if algorithm == "bpe":
        for path in files_to_conv:
            path = pathlib.Path(path)
            with open(
                path.with_stem(path.stem + "_pieces"), encoding="utf-8"
            ) as inp, open(
                path.with_suffix(".align"), encoding="utf-8"
            ) as inp_align, open(
                path.with_suffix(".align").with_stem(path.stem + "_pieces"),
                "w",
                encoding="utf-8",
            ) as out_align_f:
                for sample, smaple_alignment in zip(inp, inp_align):
                    out_align = fix_sample_bpe_alignemnt(sample, smaple_alignment)
                    out_align_f.write(out_align + "\n")


if __name__ == "__main__":
    # Example run:
    # python preprocess/tokenization.py --tokenizer_path data/vocabs/kana.json --train_files data/train.kana --files_to_conv data/train.kana --vocab_size 500 --algorithm wordlevel
    # python preprocess/tokenization.py --tokenizer_path data/vocabs/train_kanji.json --train_files data/train.kanji --files_to_conv data/train.kanji --vocab_size 16000 --algorithm bpe

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", help="Path to the tokenizer")
    parser.add_argument("--train_files", help="Path to the training file", nargs="+")
    parser.add_argument("--files_to_conv", help="Files to convert", nargs="+")
    parser.add_argument("--vocab_size", help="Vocabulary size", type=int)
    parser.add_argument(
        "--algorithm", help="Algorithm to use: wordlevel, bpe", default="wordlevel"
    )
    args = parser.parse_args()

    train_convert_files(
        args.tokenizer_path,
        args.train_files,
        args.files_to_conv,
        args.vocab_size,
        args.algorithm,
    )
