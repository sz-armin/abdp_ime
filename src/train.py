import argparse
import logging
from argparse import ArgumentParser

import lightning.pytorch as pl
import torch
import torch._dynamo.config
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import (
    DeviceStatsMonitor,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelPruning,
)
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import PyTorchProfiler
from lightning.pytorch.strategies.ddp import DDPStrategy
from torch import nn

import datasets
import models.enc_dec as enc_dec

torch.set_float32_matmul_precision("medium")

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    pl.seed_everything(50, workers=True)

    parser = ArgumentParser()
    parser.add_argument("--train-data-path", type=str, default="data/train_ids")
    parser.add_argument("--name", type=str, default="k4")
    parser.add_argument("--num-encoder-layers", type=int, default=6)
    parser.add_argument("--num-decoder-layers", type=int, default=6)
    parser.add_argument("--wait-k-cross-attn", type=int, default=-1)
    parser.add_argument(
        "--modified-wait-k",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--causal-encoder",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument("--enc-attn-window", type=int, default=-1)
    parser.add_argument(
        "--aligned-cross-attn",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--requires-alignment",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    args: argparse.Namespace = parser.parse_args()

    main_dm = datasets.IMEDataModule(
        args.train_data_path,
        train_bsz=10000,
        num_workers=8,
    )

    enc_dec = enc_dec.TransformerEncDec(
        300,
        16000,
        d_model=512,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        causal_encoder=args.causal_encoder,
        enc_attn_window=args.enc_attn_window,
        aligned_cross_attn=args.aligned_cross_attn,
        requires_alignment=args.requires_alignment,
        wait_k_cross_attn=args.wait_k_cross_attn,
        modified_wait_k=args.modified_wait_k,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    device_stats = DeviceStatsMonitor()
    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor="val_loss",
        save_last=True,
        save_on_train_epoch_end=True,
        every_n_epochs=1,
    )
    profiler = PyTorchProfiler(filename="perf-logs")
    tb_logger = pl_loggers.TensorBoardLogger(
        name="logs", save_dir=".", version=args.name, log_graph=False
    )

    decoder_trainer = pl.Trainer(
        max_epochs=9,
        accelerator="gpu",
        devices=-1,
        strategy=DDPStrategy(find_unused_parameters=True, static_graph=False),
        callbacks=[
            lr_monitor,
            checkpoint_callback,
        ],
        log_every_n_steps=100,
        precision="16-mixed",
        gradient_clip_val=1,
        enable_checkpointing=True,
        deterministic=True,
        use_distributed_sampler=False,
        logger=tb_logger,
    )

    decoder_trainer.fit(
        model=enc_dec,
        datamodule=main_dm,
    )
