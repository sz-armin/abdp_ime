*Model checkpoints will be made available in the near future*

---

# Alignment-Based Decoding Policy for Low-Latency and Anticipation-Free Neural Japanese Input Method Editors

This repository contains the implementation of our proposed approach and the main baselines introduced in our paper in Pytorch. The code and checkpoints will be released under an open license upon acceptance.

## Overview of Our Approach

The following image outlines an example conversion done by our model:

<img width="1192" alt="Outline of our model." src="https://user-images.githubusercontent.com/77587091/214112249-7d432fb1-b7ca-47ba-bf93-8d59bb8311b6.png">

To put it simply, our model achieves an anticipation-free conversion by decoding only on word-boundaries predicted by a linear classifier on top of a deep encoder stack which is trained in a multi-task settings. This is possible because the many-to-one and monotonic nature of kana-kanji alignment means that word-boundaries are all that we need to obtain kana-kanji alignments, which is a necessity for anticipation-free conversions.
Furthermore, we also use an additional linear classifier trained in a wait-k fashion to obtain more accurate boundary predictions and trigger a correction if there is a mismatch.

## Outline of the Repository

The model implementations can be found in `src/models`. Particularly, `src/models/enc_dec.py` contains the main model implementations.

The decoding policies can be found in `eval/out_agent`. The class `IncAlignDecWCOutAgent` in `eval/out_agent/aligned_dec.py` contains the implemntation of our proposed policy.
 

## Training and Evaluation
### Setting Up The Environemnt
You can install all the requirements using [Conda](https://docs.conda.io/en/latest/):
```
conda env create --name envname --file=env.yml
```
Then activate the virtual environment:
```
conda activate ime
```

### Data Preparation
You need two separate text files containing the kana sequences (separated by a new line) tokenized as characters, and the corresponding kanji sequences tokenized by words (which can be achieved by any morphological analyzer such as [mecab](https://taku910.github.io/mecab/))
> Although BCCWJ is not available under an open license due to containing copy-righted material, it can still be obtained even by individuals. Please refer to [here](https://clrd.ninjal.ac.jp/bccwj/en/subscription.html) for more information.

An example `data.kana` file:
```
こ の じ て ん で わ れ わ れ は か み の こ と ば 、
「 わ が お も い は な ん じ の お も い で は な い 。
```
An example `data.kanji` file:
```
この 時点 で われわれ は 神 の 言葉 、
「 我が 思い は 汝 の 思い で は ない 。
```

Next, you should extract alignments for training:
```
python preprocess/mec2alignment.py data/train.kana data/train.kanji
```

Then create vocabularies and and tokenize kana and kanji files (note that .align file should be present in the same directory):
```
python preprocess/tokenization.py --tokenizer_path data/vocabs/kana.json --train_files data/train.kana --files_to_conv data/train.kana data/val.kana data/test_full.kana --vocab_size 500 --algorithm wordlevel
python preprocess/tokenization.py --tokenizer_path data/vocabs/train_kanji.json --train_files data/train.kanji --files_to_conv data/train.kanji --vocab_size 16000 --algorithm bpe
```

### Training
#### Our model
```
python src/train.py --train-data-path data --name ours --causal-encoder --enc-attn-window -1 --aligned-cross-attn --requires-alignment --num-encoder-layers 10 --num-decoder-layers 2 --wait-k-cross-attn 4 --no-modified-wait-k
```
> The data path refers to the base path for files produced in the tokenization step, e.g. data if your files include `data_ids.kana`

#### Wait-k
```
python src/train.py --train-data-path data --name wait-3 --causal-encoder --enc-attn-window -1 --no-aligned-cross-attn --no-requires-alignment --num-encoder-layers 10 --num-decoder-layers 2 --wait-k-cross-attn 4 --no-modified-wait-k
```
> Note that `wait-k-cross-attn` should be set as $k+1$

#### Modified Wait-k
```
python src/train.py --train-data-path data --name wait-3 --causal-encoder --enc-attn-window -1 --no-aligned-cross-attn --no-requires-alignment --num-encoder-layers 10 --num-decoder-layers 2 --wait-k-cross-attn 4 --modified-wait-k
```

#### Re-translation
```
python src/train.py --name retranslation --no-causal-encoder --enc-attn-window -1 --no-aligned-cross-attn --no-requires-alignment --num-encoder-layers 10 --num-decoder-layers 2 --wait-k-cross-attn -1 --no-modified-wait-k
```

### Evaluation
#### Predictions
First, run the models to get prediction data as a pickle file. The training vocabulary file is expected to exist in `vocabs/train_kanji.json`.
```
python eval/evaluator.py --policy ours --test-data-path data --model-path model.ckpt --hparam-path hparams.yaml
```
> The data path refers to the base path for files produced in the tokenization step, e.g. data if your files include `data_ids.kana`.

> For baselines, set the policy to either of `wait-k`, `mod-wait-k` or `retrans`.

> In the case of wait-k and mod-wait-k, set the test-time k via --k (Note that in the case of vanilla wait-k you have to set k to k+1)

Then decode the pickle file to get the predictions and corresponding labels:
```
python utils/unpickler.py --policy ours --pkl-path output.pkl --train-vocab-path /vocabs/train_kanji.json --test-vocab-path /vocabs/test_kanji.json
```
> For baselines, set the policy to either of `wait-k`, `mod-wait-k` or `retrans`.

#### Calculate the Metrics
**Conversion Quality:**
```
python utils/score.py --prediction-path preds.txt --label-path labels.txt
```

**Computational latency:**
```
python utils/latency-c.py output.pkl
```

**Non-computational latency:**
```
python utils/latency.py output.pkl --policy ours --kanji-vocab-path vocabs/train_kanji.json --kana-vocab-path vocabs/kana.json --label-path label.text
```
> For baselines, set the policy to either of `wait-k`, `mod-wait-k` or `retrans`.
