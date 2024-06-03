from __future__ import annotations

import ctypes
import logging
import os
import pickle
import sys
import time

from in_agent.character import *

sys.path.insert(0, os.path.abspath("./"))
import multiprocessing as mp
import os
from abc import ABC, abstractmethod
from multiprocessing import freeze_support, sharedctypes
from pathlib import Path
from typing import Any, Callable, Generator, Optional, Sequence, Union

import tqdm
from out_agent.aligned_dec import *
from out_agent.retranslation import *
from out_agent.wait_k import *
from out_agent.mod_wait_k import *
from in_agent.all import *
from utils import *
import resource


class EvaluatorWorker:
    def __init__(
        self,
        in_agent: InAgentBase,
        out_agent: OutAgentBase,
        pred_data: Sequence[Any],
        eval_data: Sequence[Any],
        res_container: list[Any],
        counter: Optional[sharedctypes.SynchronizedBase[ctypes.c_int]],
    ) -> None:
        self.in_agent = in_agent
        self.out_agent = out_agent

        self.pred_data = pred_data
        self.eval_data = eval_data

        self.res_container = res_container
        self.counter = counter

    def get_batch(
        self, batch_size: int = 1
    ) -> Generator[tuple[Sequence[Any], Sequence[Any]], None, None]:
        assert len(self.pred_data) == len(
            self.eval_data
        ), f"The number of prediction data and evaluation data must be the same., {len(self.pred_data)} != {len(self.eval_data)}"

        s = 0
        e = batch_size
        while s < len(self.pred_data):
            yield self.pred_data[s:e], self.eval_data[s:e]
            s += batch_size
            e += batch_size

    def evaluate(self) -> None:
        # TODO bsz
        for b in self.get_batch(1):
            pred_batch, eval_batch = b

            batch_interm_outs = []
            batch_interm_ins = []
            batch_interm_times = []
            for incream_inp in self.in_agent.get(pred_batch):
                batch_interm_ins.append(copy.deepcopy(incream_inp))
                t1 = time.monotonic()
                incream_out = self.out_agent.put(incream_inp)
                t2 = time.monotonic()

                batch_interm_outs.append(copy.deepcopy(incream_out))
                batch_interm_times.append(t2 - t1)

            self.res_container.append(
                (batch_interm_outs, batch_interm_ins, eval_batch, batch_interm_times)
            )
            if self.counter is not None:
                with self.counter.get_lock():
                    self.counter.value += 1  # type: ignore


class Evaluator:
    def __init__(
        self,
        in_agent: Callable,  # TODO add config support
        out_agent: Callable,
        pred_data_paths: Sequence[Union[str, Path]],
        eval_data_paths: Sequence[Union[str, Path]],
        num_workers: int = 1,
        output_path: Optional[str] = "./output.pkl",
        in_agent_args: Optional[Sequence[Any]] = None,
        out_agent_args: Optional[Sequence[Any]] = None,
    ) -> None:
        self.in_agent = in_agent
        self.out_agent = out_agent
        self.in_agent_args = in_agent_args
        self.out_agent_args = out_agent_args

        self.pred_data, self.eval_data = self.read_data(
            pred_data_paths, eval_data_paths
        )

        self.num_processes = num_workers

        self.output_path = output_path

        self.evaluate()

    def evaluate(self) -> None:
        mp.set_start_method("spawn", force=True)
        chunk_size = len(self.pred_data) // self.num_processes
        s = 0
        e = chunk_size
        manager = mp.Manager()
        res_container = manager.list()
        procs = []
        c = mp.Value("i", 0)
        for i in range(self.num_processes):
            if i == self.num_processes - 1:
                e = len(self.pred_data)
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = f"{i}"
            p = mp.Process(
                target=self._eval_loop,
                args=(
                    self.in_agent,
                    self.out_agent,
                    self.pred_data[s:e],
                    self.eval_data[s:e],
                    res_container,
                    c,
                ),
            )
            procs.append(p)
            p.start()

            s += chunk_size
            e += chunk_size

        prog_p = mp.Process(target=self._progress_bar, args=(c, len(self.pred_data)))
        prog_p.start()
        for p in procs:
            p.join()
        prog_p.join()
        # print(res_container)

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "wb") as f:
            pickle.dump(list(res_container), f, protocol=pickle.HIGHEST_PROTOCOL)

    def _eval_loop(
        self, in_agent, out_agent, pred_data, eval_data, res_container, counter=None
    ) -> None:
        eval_worker = EvaluatorWorker(
            in_agent(**self.in_agent_args) if self.in_agent_args else in_agent(),
            out_agent(*self.out_agent_args) if self.out_agent_args else out_agent(),
            pred_data,
            eval_data,
            res_container,
            counter=counter,
        )
        eval_worker.evaluate()

    def _progress_bar(self, current: int, total: int) -> None:
        pbar = tqdm.tqdm(range(total))
        last_value = 0
        stall_count = 0
        while (i := current.value) < total:  # type: ignore
            if i == last_value:
                stall_count += 1
                if stall_count > 1000:
                    logging.error(
                        "Exiting due to timeout, parent process is likely dead."
                    )
                    break
            else:
                stall_count = 0

            pbar.n = i
            pbar.last_print_n = i
            pbar.refresh()
            time.sleep(0.25)
        pbar.n = i
        pbar.last_print_n = i
        pbar.refresh()

    def read_data(
        self,
        pred_data_paths: Sequence[Union[str, Path]],
        eval_data_paths: Sequence[Union[str, Path]],
    ) -> Sequence[Any]:
        pred_data = []
        for path in pred_data_paths:
            with open(path, "r") as f:
                pred_data.append(f.read().splitlines())

        eval_data = []
        for path in eval_data_paths:
            with open(path, "r") as f:
                eval_data.append(f.read().splitlines())
        return [list(zip(*pred_data)), list(zip(*eval_data))]

    def on_after_read_data(self, data: Sequence[Any]) -> Sequence[Any]:
        return data


if __name__ == "__main__":
    # Example usage
    # python eval/evaluator.py --policy ours --test-data-path data/test --model-path agn_attn_wa_ref.ckpt --hparam-path /hparams.yaml
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, default="ours")
    parser.add_argument("--test-data-path", type=str)
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--hparam-path", type=str)
    parser.add_argument("--output-path", type=str, default="eval.pkl")
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))

    match args.policy:
        case "ours":
            Evaluator(
                CharInAgent,
                IncAlignDecWCOutAgent,
                [
                    f"{args.test_data_path}_ids.kana",
                    f"{args.test_data_path}_pieces.align",
                ],
                [f"{args.test_data_path}_ids.kanji"],
                output_path=args.output_path,
                out_agent_args=[args.model_path, args.hparam_path],
            )
        case "wait-k":
            Evaluator(
                CharSuffixInAgent,
                WaitKOutAgent,
                [
                    f"{args.test_data_path}_ids.kana",
                    f"{args.test_data_path}_pieces.align",
                ],
                [f"{args.test_data_path}_ids.kanji"],
                output_path=args.output_path,
                out_agent_args=[args.model_path, args.hparam_path, args.k],
            )
        case "retrans":
            Evaluator(
                CharSuffixInAgent,
                RetranOutAgent,
                [
                    f"{args.test_data_path}_ids.kana",
                    f"{args.test_data_path}_pieces.align",
                ],
                [f"{args.test_data_path}_ids.kanji"],
                output_path=args.output_path,
                in_agent_args={"inc_eos": True},
                out_agent_args=[args.model_path, args.hparam_path],
            )
        case "mod-wait-k":
            Evaluator(
                CharSuffixInAgent,
                ModWaitKOutAgent,
                [
                    f"{args.test_data_path}_ids.kana",
                    f"{args.test_data_path}_pieces.align",
                ],
                [f"{args.test_data_path}_ids.kanji"],
                output_path=args.output_path,
                out_agent_args=[args.model_path, args.hparam_path, args.k],
            )
        case _:
            raise ValueError(f"Policy {args.policy} not found")
