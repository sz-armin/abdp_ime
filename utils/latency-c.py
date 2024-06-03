from pathlib import Path
import pickle
import numpy as np
from tokenizers import Tokenizer

v_tgt = Tokenizer.from_file("data/vocabs/train_kanji.json")
v_tkana = Tokenizer.from_file("data/vocabs/kana.json")


def process_file(file_path):
    file_path = Path(file_path)
    with file_path.open("rb") as file:
        data = pickle.load(file)

    _, _, _, times = list(zip(*data))

    # total
    times_list = [np.array(x, dtype=object).sum() for x in times]
    total_time = np.array(times_list, dtype=object).mean()
    print(f"Total time: {total_time*1000}")

    # mean
    times_list = [np.array(x, dtype=object).mean() for x in times]
    total_time = np.array(times_list, dtype=object).mean()
    print(f"Mean time: {total_time*1000}")

    # max
    times_list = [np.array(x, dtype=object).max() for x in times]
    total_time = np.array(times_list, dtype=object).mean()
    print(f"Max time: {total_time*1000}")


if __name__ == "__main__":
    # Example usage
    # python utils/latency-c.py output.pkl
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("pkl_path", type=str)
    args = parser.parse_args()

    process_file(args.pkl_path)
