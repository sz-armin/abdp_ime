import glob
import logging
import re
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import lightning.pytorch as pl

import datasets
import models.enc_dec as enc_dec

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = ArgumentParser()
    parser.add_argument("--name", type=str, default="base")
    args: Namespace = parser.parse_args()

    main_dm = datasets.IMEDataModule(
        num_workers=8,
        extract_alignment=True,
    )

    ckpt_path = glob.glob(f"lightning_logs/{args.name}/checkpoints/*.ckpt")
    ckpt_path.sort()
    ckpt_path = ckpt_path[-1]
    logging.info(f"Loading checkpoint {ckpt_path}")
    enc_dec = enc_dec.TransformerEncDec(1250, 64000, d_model=512).load_from_checkpoint(
        ckpt_path,
        hparams_file=f"lightning_logs/{args.name}/hparams.yaml",
    )
    enc_dec.extract_alignment=True

    decoder_trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision=16,
    )

    decoder_trainer.validate(
        model=enc_dec,
        datamodule=main_dm,
    )

    Path(f"outputs/{args.name}").mkdir(parents=True, exist_ok=True)

    with open(f"outputs/{args.name}/extracted_agns.txt", "w+") as f:
        # print(enc_dec.extracted_agns)
        for x in enc_dec.extracted_agns:
            np.savetxt(f, x.cpu(), delimiter=" ", fmt="%s")

        f.seek(0)
        content = f.read()
        content = re.sub(" *-2 *", " ", content)
        # content = re.sub(" *-1 *", " ", content)
        content = re.sub(" +", " ", content)
        content = re.sub(" $", "", content)
        content = re.sub("^ ", "", content)
        f.seek(0)
        f.write(content)
        f.truncate()