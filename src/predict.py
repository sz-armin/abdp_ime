import glob
import logging
import re
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import pytorch_lightning as pl

import datasets
import models.enc_dec as enc_dec

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = ArgumentParser()
    parser.add_argument("--name", type=str, default="base")
    parser.add_argument("--causal_encoder", type=bool, default=False)
    args: Namespace = parser.parse_args()

    main_dm = datasets.IMEDataModule(
        num_workers=16,
    )

    ckpt_path = glob.glob(f"logs/{args.name}/checkpoints/*.ckpt")
    ckpt_path.sort()
    ckpt_path = ckpt_path[-1]
    logging.info(f"Loading checkpoint {ckpt_path}")
    enc_dec = enc_dec.TransformerEncDec(500, 16000, d_model=512).load_from_checkpoint(
        ckpt_path,
        hparams_file=f"logs/{args.name}/hparams.yaml",
    )

    decoder_trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision=16,
    )

    decoder_trainer.test(
        model=enc_dec,
        datamodule=main_dm,
    )

    Path(f"outputs/{args.name}").mkdir(parents=True, exist_ok=True)

    with open(f"outputs/{args.name}/preds_id.txt", "w+") as f1, open(
        f"outputs/{args.name}/labels_id.txt", "w+"
    ) as f2:
        for x, y in zip(enc_dec.test_preds, enc_dec.test_labels):
            np.savetxt(f1, x.cpu(), delimiter=" ", fmt="%s")
            np.savetxt(f2, y.cpu(), delimiter=" ", fmt="%s")
        for f in (f1, f2):
            f.seek(0)
            content = f.read()
            content_new = re.sub(" 2 .*", " 2", content)
            f.seek(0)
            f.write(content_new)
            f.truncate()
