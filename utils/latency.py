import pickle

import numpy as np
from tokenizers import Tokenizer


def measure_sample(
    input_steps, prediction_steps, ref=None, char_level=True, policy="ours"
):
    steps = []
    prev = None

    match policy:
        case "ours":
            final_pred_step = list(
                filter(lambda x: x != 1 and x != 2, prediction_steps[-1])
            )
            final_inp_step = list(
                filter(lambda x: x != 1 and x != 2, input_steps[-1][0])
            )
        case "retrans":
            final_pred_step = list(
                filter(lambda x: x != 1 and x != 2, prediction_steps[-1])
            )
            final_inp_step = list(
                filter(lambda x: x != 1 and x != 2, input_steps[-1][0][0])
            )
        case "wait-k":
            final_pred_step = list(
                filter(lambda x: x != 1 and x != 2, prediction_steps[-1])
            )
            final_inp_step = list(
                filter(lambda x: x != 1 and x != 2, input_steps[-1][0][-1])
            )
        case "mod-wait-k":
            final_pred_step = list(
                filter(lambda x: x != 1 and x != 2, prediction_steps[-1])
            )
            final_inp_step = list(
                filter(lambda x: x != 1 and x != 2, input_steps[-1][0][-1])
            )
        case _:
            raise ValueError(f"Unknown policy: {policy}")

    if char_level:
        final_pred_step = v_tgt.decode(final_pred_step)
        final_pred_step = final_pred_step.replace("</w>", "")
        final_pred_step = final_pred_step.replace(" ", "")
        final_pred_step = list(final_pred_step)

        final_inp_step = v_tkana.decode(final_inp_step)
        final_inp_step = final_inp_step.replace(" ", "")
        final_inp_step = list(final_inp_step)

    final_pred_len = len(final_pred_step)
    final_inp_len = len(final_inp_step)

    if ref:
        ref = ref.strip()
        ref = ref.replace(" ", "")
        ref = ref.replace("</w>", "")
        ref = ref.replace("</s>", "")
        ref = ref.replace("<s>", "")

        if len(ref) > final_pred_len:
            pass

        final_pred_len = max(final_pred_len, len(ref))

    ratio = final_pred_len / final_inp_len

    if final_pred_len == 0:
        return np.NAN
    for input_step, prediction_step in zip(input_steps, prediction_steps):
        if policy == "ours":
            input_step = input_step[0]
        elif policy == "retrans":
            input_step = input_step[0][0]
        elif policy == "wait-k" or policy == "mod-wait-k":
            input_step = input_step[0][-1]
        else:
            raise ValueError(f"Unknown policy: {policy}")

        input_step = list(filter(lambda x: x != 1 and x != 2, input_step))
        prediction_step = list(filter(lambda x: x != 1 and x != 2, prediction_step))

        if char_level:
            input_step = v_tkana.decode(input_step)
            prediction_step = v_tgt.decode(prediction_step)
            input_step = input_step.replace(" ", "")
            input_step = list(input_step)
            prediction_step = prediction_step.replace("</w>", "")
            prediction_step = prediction_step.replace(" ", "")
            prediction_step = list(prediction_step)
        else:
            pass

        if (
            prediction_step == prev
            or input_step == []
            or prediction_step == []
            or input_step == [""]
            or prediction_step == [""]
        ):
            continue

        steps.append(
            (
                input_step,
                prediction_step + [None] * (final_pred_len - len(prediction_step)),
            )
        )
        prev = prediction_step

    if len(steps) > 1:
        r = min(
            len(list(filter(lambda x: x is not None, steps[-2][-1]))) + 1,
            len(list(filter(lambda x: x is not None, steps[-1][-1]))),
        )
    else:
        r = 1

    lrs = []
    prev = None
    for t in range(final_pred_len):
        prev = None
        for input_step, prediction_step in steps:
            tok = prediction_step[t]
            if tok == prev:
                continue
            lr = len(input_step) - 0
            prev = tok
        lrs.append(lr)

    lrs.sort()

    sum = 0
    for t in range(0, r):
        tmp = lrs[t] - ((t - 0) / ratio)
        sum += tmp

    sum = sum / r

    return sum


def process_file(file_name, policy, label_path):
    with open(file_name, "rb") as failure_count:
        data = pickle.load(failure_count)

    predictions, inputs, _, _ = list(zip(*data))

    with open(label_path, "r") as f:
        ref = f.readlines()

    if policy != "retrans":
        make_prefix(inputs)

    latencies = []
    failure_count = 0
    count = 0
    for input, prediction in zip(inputs, predictions):
        temp_p = []
        for prediction_step in prediction:
            if policy == "wait-k" or policy == "mod-wait-k":
                prediction_step = [prediction_step]
            if prediction_step != []:
                temp_p.append(prediction_step[0])
            else:
                temp_p.append([1])
        prediction = temp_p
        try:
            latencies.append(
                measure_sample(input, prediction, ref[count], policy=policy)
            )
        except Exception as _:
            failure_count += 1
        count += 1

    print(f"NaN count: {latencies.count(np.NAN)}")
    latencies = list(filter(lambda x: not np.isnan(x), latencies))
    mean = np.array(latencies).mean()
    print(mean)
    print(f"Failure count: {failure_count}")


def make_prefix(inputs):
    for i, inp1 in enumerate(inputs):
        inp1.reverse()
        for i, inp2 in enumerate(inp1):
            inp2 = inp2[0]
            for inp3 in inp1[i + 1 :]:
                inp2[:0] = inp3[0]
        inp1.reverse()


def test():
    a = [[[3]], [[3, 4]], [[3, 4, 5]], [[3, 4, 5, 6]]]
    b = [[3], [3, 4], [3, 4, 5], [3, 4, 5, 6]]
    print(measure_sample(a, b))  # 1

    a = [[[3]], [[3, 4]], [[3, 4, 5]], [[3, 4, 5, 6]]]
    b = [[3], [3, 4], [3, 4, 5], [3, 4, 5, 6, 7, 8, 9, 10]]
    print(measure_sample(a, b))

    a = [[[3]], [[3, 4]], [[3, 4, 5]], [[3, 4, 5, 6]]]
    b = [[], [], [3], [3, 4], [3, 4, 5, 6]]
    print(measure_sample(a, b))  # 3

    a = [[[3]], [[3, 4]], [[3, 4, 5]], [[3, 4, 5, 6]]]
    b = [[], [3], [3], [3, 4, 5, 6]]
    print(measure_sample(a, b))

    a = [
        [[3]],
        [[3, 4]],
        [[3, 4, 5]],
        [[3, 4, 5, 6]],
        [[3, 4, 5, 6, 7]],
        [[3, 4, 5, 6, 7, 8]],
        [[3, 4, 5, 6, 7, 8, 9]],
        [[3, 4, 5, 6, 7, 8, 9, 10]],
    ]
    b = [
        [3],
        [3, 4, 5],
        [3, 4, 5, 6],
        [3, 4, 5, 6],
        [3, 4, 5, 6],
        [3, 4, 5, 6],
        [3, 4, 5, 6],
        [3, 4, 5, 6, 7],
    ]
    print(measure_sample(a, b))


if __name__ == "__main__":
    # Example usage
    # python utils/latency-c.py output.pkl --policy ours --kanji-vocab-path vocabs/train_kanji.json --kana-vocab-path vocabs/kana.json --label-path label.text
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("pkl_path", type=str)
    parser.add_argument("--policy", type=str, default="ours")
    parser.add_argument("--kanji-vocab-path", type=str)
    parser.add_argument("--kana-vocab-path", type=str)
    parser.add_argument("--label-path", type=str)
    args = parser.parse_args()

    global v_tgt, v_tkana
    v_tgt = Tokenizer.from_file(args.kanji_vocab_path)
    v_tkana = Tokenizer.from_file(args.kana_vocab_path)

    process_file(args.pkl_path, args.policy, args.label_path)
