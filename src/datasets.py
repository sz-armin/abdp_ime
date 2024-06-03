import random, logging
from typing import List, Optional
from torchtext.functional import to_tensor
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl
from more_itertools import sort_together


def _batchify(data, labels, aligns, num_tokens) -> tuple[tuple, ...]:
    if aligns is None:
        data = list(map(lambda x: list(map(int, x.split())), data))
        labels = list(map(lambda x: list(map(int, x.split())), labels))

        data, labels = sort_together(
            [data, labels],
            key_list=[0],
            key=lambda x: len(x),
            reverse=True,
        )
        dbatches = []
        lbatches = []
        tmp_dbatch = []
        tmp_lbatch = []
        tmp_msize = 0
        tmp_cumsize = 0
        for d, l in zip(data, labels):
            if tmp_cumsize <= num_tokens - tmp_msize:
                if tmp_msize == 0:
                    tmp_msize = len(d)
                tmp_dbatch.append(d)
                tmp_lbatch.append(l)
                tmp_cumsize += tmp_msize
            else:
                dbatches.append(tmp_dbatch)
                lbatches.append(tmp_lbatch)
                tmp_dbatch = []
                tmp_lbatch = []
                tmp_msize = 0
                tmp_cumsize = 0

                tmp_msize = len(d)
                tmp_dbatch.append(d)
                tmp_lbatch.append(l)
                tmp_cumsize += tmp_msize
        dbatches.append(tmp_dbatch)
        lbatches.append(tmp_lbatch)

        tmp_list = list(zip(dbatches, lbatches))
        random.shuffle(tmp_list)

        dbatches, lbatches = zip(*tmp_list)

        return dbatches, lbatches

    else:
        data = list(map(lambda x: list(map(int, x.split())), data))
        labels = list(map(lambda x: list(map(int, x.split())), labels))
        aligns = list(map(lambda x: list(map(int, x.split())), aligns))

        data, labels, aligns = sort_together(
            [data, labels, aligns],
            key_list=[0],
            key=lambda x: len(x),
            reverse=True,
        )
        dbatches = []
        lbatches = []
        abatches = []
        tmp_dbatch = []
        tmp_lbatch = []
        tmp_abatch = []
        tmp_msize = 0
        tmp_cumsize = 0
        for d, l, a in zip(data, labels, aligns):
            if tmp_cumsize <= num_tokens - tmp_msize:
                if tmp_msize == 0:
                    tmp_msize = len(d)
                tmp_dbatch.append(d)
                tmp_lbatch.append(l)
                tmp_abatch.append(a)
                tmp_cumsize += tmp_msize
            else:
                dbatches.append(tmp_dbatch)
                lbatches.append(tmp_lbatch)
                abatches.append(tmp_abatch)
                tmp_dbatch = []
                tmp_lbatch = []
                tmp_abatch = []
                tmp_msize = 0
                tmp_cumsize = 0

                tmp_msize = len(d)
                tmp_dbatch.append(d)
                tmp_lbatch.append(l)
                tmp_abatch.append(a)
                tmp_cumsize += tmp_msize
        dbatches.append(tmp_dbatch)
        lbatches.append(tmp_lbatch)
        abatches.append(tmp_abatch)

        tmp_list = list(zip(dbatches, lbatches, abatches))
        random.shuffle(tmp_list)

        dbatches, lbatches, abatches = zip(*tmp_list)

        return dbatches, lbatches, abatches


class IMETokenDataset(Dataset):
    def __init__(
        self, data_path, labels_path, align_path, train_bsz, rank, num_devices: int
    ) -> None:
        self.align_path = align_path

        with open(data_path, "r") as f, open(labels_path, "r") as g:
            self.data = f.readlines()
            self.labels = g.readlines()
        if self.align_path is not None:
            with open(align_path, "r") as h:
                self.align = h.readlines()

        if self.align_path is not None:
            self.data, self.labels, self.align = _batchify(
                self.data, self.labels, self.align, train_bsz
            )
        else:
            self.data, self.labels = _batchify(self.data, self.labels, None, train_bsz)

        chunk_size = len(self.data) // num_devices

        if self.align_path is not None:
            self.align = [
                self.align[i : i + chunk_size]
                for i in range(0, len(self.align), chunk_size)
            ]
            self.align = self.align[rank]

        self.data = [
            self.data[i : i + chunk_size] for i in range(0, len(self.data), chunk_size)
        ]
        self.data = self.data[rank]

        self.labels = [
            self.labels[i : i + chunk_size]
            for i in range(0, len(self.labels), chunk_size)
        ]
        self.labels = self.labels[rank]

        logging.info(f"Loaded {sum(map(len, self.data))} samples on rank {rank}")

    def __getitem__(self, index):
        if self.align_path is not None:
            return [self.data[index], self.labels[index], self.align[index]]
        else:
            return [self.data[index], self.labels[index]]

    def __len__(self) -> int:
        return len(self.data)


class IMEDataset(Dataset):
    def __init__(
        self, data_path, labels_path, align_path, is_test: bool = False
    ) -> None:
        self.align_path = align_path

        with open(data_path, "r") as f, open(labels_path, "r") as g:
            self.data = f.readlines()
            # self.data = [x[:-3] for x in self.data]
            self.labels = g.readlines()
        if self.align_path is not None:
            with open(align_path, "r") as h:
                self.align = h.readlines()

        if is_test:
            if self.align_path is not None:
                self.data, self.labels, self.align = sort_together(
                    [self.data, self.labels, self.align],
                    key_list=[0],
                    key=lambda x: len(x),
                    reverse=True,
                )

            else:
                self.data, self.labels = sort_together(
                    [self.data, self.labels],
                    key_list=[0],
                    key=lambda x: len(x),
                    reverse=True,
                )

    def __getitem__(self, index) -> List[List[int]]:
        if self.align_path is not None:
            return [
                list(map(int, self.data[index].split())),
                list(map(int, self.labels[index].split())),
                list(map(int, self.align[index].split())),
            ]
        else:
            return [
                list(map(int, self.data[index].split())),
                list(map(int, self.labels[index].split())),
            ]

    def __len__(self) -> int:
        return len(self.data)


class IMEDataModule(pl.LightningDataModule):
    def __init__(self, file_path: str ,num_workers: int = 4, train_bsz: int = 256, extract_alignment: bool = False) -> None:
        super().__init__()
        self.train_bsz = train_bsz
        self.num_workers = num_workers
        self.extract_alignment = extract_alignment
        self.file_path = file_path

    def setup(self, stage: Optional[str] = None) -> None:
        self.dataset_train = IMETokenDataset(
            f"{self.file_path}_ids.kana",
            f"{self.file_path}_ids.kanji",
            f"{self.file_path}_pieces.align",
            self.train_bsz,
            rank=self.trainer.global_rank, # type: ignore
            num_devices=self.trainer.num_devices, # type: ignore
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=1,
            collate_fn=self._token_collate_wrapper,
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=True,
            prefetch_factor=4,
            pin_memory=True,
        )


    def on_after_batch_transfer(self, batch, dataloader_idx: int):
        return batch[0], batch[1], batch[2]

    def _collate_wrapper(self, batch):
        batch = list(zip(*batch))
        data = to_tensor(list(batch[0]), padding_value=3)
        labels = to_tensor(list(batch[1]), padding_value=3)

        if self.dataset_train.align_path is not None and self.extract_alignment is None:
            align = to_tensor(list(batch[2]), padding_value=9999)
            return [data, labels, align]
        else:
            return [data, labels, None]

    def _token_collate_wrapper(self, batch):
        data = to_tensor(batch[0][0], padding_value=3)
        labels = to_tensor(batch[0][1], padding_value=3)

        if self.dataset_train.align_path is not None:
            align = to_tensor(batch[0][2], padding_value=9999)
            return [data, labels, align]
        else:
            return [data, labels, None]
