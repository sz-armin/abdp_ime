import pickle
from tokenizers import Tokenizer
import numpy as np
import pandas as pd

def decode(pkl_path, v_tgt, v_test, policy):
    v_tgt = Tokenizer.from_file(v_tgt)
    v_test = Tokenizer.from_file(v_test)

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    preds, ins, labels, times = list(zip(*data))

    with open(f"labels.txt", "w") as f:
        for s in labels:
            label = s[0][0].split()
            label = list(map(int, label))
            label = v_test.decode(label)
            label = label.replace("</w>", "")
            label = label.replace(" ", "")
            f.write(label + "\n")

    with open(f"preds.txt", "w") as o_f:
        for s in preds:
            if policy == "ours" or policy == "retrans":
                h = v_tgt.decode_batch(s[-1]) # agn/retrans
            else:
                h = v_tgt.decode_batch([s[-1]]) # wait/mod
            if type(h) == list and len(h) == 1:
                h = h[0]
                
            if len(h) > 0:
                h = h.replace("</w>", "")
                h = h.replace(" ", "")

                o_f.write(h + "\n")
            else:
                o_f.write("\n")

if __name__ == "__main__":
    # Example usage
    # python utils/unpickler.py --policy ours --pkl-path output.pkl --train-vocab-path /vocabs/train_kanji.json --test-vocab-path /vocabs/test_kanji.json
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, default="ours")
    parser.add_argument("--pkl-path", type=str)
    parser.add_argument("--train-vocab-path", type=str)
    parser.add_argument("--test-vocab-path", type=str)
    args = parser.parse_args()

    decode(args.pkl_path, args.train_vocab_path, args.test_vocab_path, args.policy)

