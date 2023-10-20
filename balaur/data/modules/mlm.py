from dataclasses import dataclass, field
import os
from itertools import chain
from pathlib import Path
import random

import datasets as ds
import pytorch_lightning as pl
import transformers as tr
import torch
from torch.utils.data import DataLoader

from balaur import config as balaur_conf
from balaur.utils.logging import get_logger
from balaur.utils.fingerprint import fingerprint_dict
from balaur.data.datasets import get_raw_dataset, TEXT_COL
from balaur.data.tokenizers import get_tokenizer

from typing import *
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

CACHE_DIR = os.path.join(balaur_conf.DEFAULT_BALAUR_DATA_CACHE, "mlm")
logger = get_logger(__name__)


@dataclass
class MlmDataModuleConfig:
    dataset_name: str = field(
        default='wikipedia',
        metadata={'help': 'Huggingface dataset name.'}
    )
    tokenizer: str = field(
        default='roberta-base',
        metadata={'help': 'Name of tokenizer.'}
    )
    complete_docs: bool = field(
        default=False,
        metadata={'help': 'Prevent examples crossing document boundaries.'}
    )
    short_seq_prob: float = field(
        default=None,
        metadata={'help': 'Probability of training example being shortened'}
    )
    valid_split: float = field(
        default=0.005,
        metadata={'help': 'Fraction of dataset to reserve for validation.'}
    )
    mlm_probability: float = field(
        default=0.15,
        metadata={'help': 'Masking probability for masked language modelling.'}
    )
    dataset_seq_len: int = field(
        default=128,
        metadata={'help': 'Maximum sequence length in tokens in dataset.'}
    )
    split_seed: int = field(
        default=0xBAD5EED,
        metadata={'help': 'Seed for dataset splitting.'}
    )
    shorten_seed: int = field(
        default=0xBAD5EED,
        metadata={'help': 'Seed for dataset preprocessing.'}
    )
    mask_seed: int = field(
        default=0xBAD5EED,
        metadata={'help': 'Seed for MLM masking.'}
    )
    num_preprocess_workers: int = field(
        default=8,
        metadata={'help': 'Number of workers for dataset preprocessing.'}
    )
    preprocess_map_bsz: int = field(
        default=128,
        metadata={'help': 'Batch size for preprocessing examples.'}
    )
    preprocess_write_bsz: int = field(
        default=128,
        metadata={'help': 'Batch size for preprocessing examples.'}
    )
    shuffle_seed: int = field(
        default=None,
        metadata={'help': "Seed for shuffling dataset."}
    )
    per_device_bsz: int = field(
        default=256,
        metadata={'help': 'Batch size (per device).'}
    )
    num_dataloader_workers: int = field(
        default=0,
        metadata={'help': 'Number of workers for dataloader.'}
    )
    cache_dir: str = field(
        default=CACHE_DIR,
        metadata={'help': f"Directory for caching preprocessed dataset.\n"
                          f"Defaults to {CACHE_DIR}"}
    )
    overwrite: bool = field(
        default=False,
        metadata={'help': "Rerun preprocessing and overwrite cache."}
    )

    def __post_init__(self):
        self.fingerprint = self._init_fingerprint()

    def _init_fingerprint(self):
        KEYS_TO_HASH = [
            'dataset_name',
            'tokenizer',
            'complete_docs',
            'valid_split',
            'split_seed',
            'dataset_seq_len'
        ]
        state = self.__dict__
        state = {k: state[k] for k in KEYS_TO_HASH}
        return fingerprint_dict(state)


class MlmDataModule(pl.LightningDataModule):
    def __init__(self, config: MlmDataModuleConfig):
        super().__init__()
        self.config = config
        self.cache_file_path = self._get_cache_file_path()

    def _get_cache_file_path(self):
        fp = self.config.fingerprint
        return os.path.join(self.config.cache_dir, fp)

    def prepare_data(self) -> None:
        if Path(self.cache_file_path).exists() and not self.config.overwrite:
            logger.info(f"Preprocessed dataset already cached in "
                        f"{self.cache_file_path}, skipping `prepare_data`.")
            return

        datasets = get_raw_dataset(
            self.config.dataset_name,
            split=self.config.valid_split,
            seed=self.config.split_seed
        )

        datasets = [self._preprocess_raw_datasets(d) for d in datasets]
        dataset = ds.DatasetDict()
        dataset['train'] = ds.concatenate_datasets([d['train'] for d in datasets])
        if self.config.valid_split is not None:
            dataset['validation'] = ds.concatenate_datasets([d['validation'] for d in datasets])
        logger.info(f"Caching processed dataset to {self.cache_file_path}")
        dataset.save_to_disk(self.cache_file_path)

    def _preprocess_raw_datasets(self, d: ds.DatasetDict):
        c = self.config
        # for bos/eos tokens (added in collate_fn)
        block_size = c.dataset_seq_len - 2
        tokenizer = get_tokenizer(c.tokenizer)

        def batch_preprocess(examples):
            # filter out empty lines
            examples[TEXT_COL] = [
                x.strip() for x in examples[TEXT_COL]
                if len(x) > 0 and not x.isspace()
            ]
            tokenized_docs = tokenizer(
                examples['text'],
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )['input_ids']

            if self.config.complete_docs:
                tokenized_examples = []
                for doc in tokenized_docs:
                    tokenized_examples.extend(
                        [doc[i: i + block_size] for i in range(0, len(doc), block_size)]
                    )
            else:
                tokenized_examples = list(chain(*tokenized_docs))
                total_length = len(tokenized_examples)
                if total_length >= block_size:
                    total_length = (total_length // block_size) * block_size
                tokenized_examples = [tokenized_examples[i: i+block_size]
                                      for i in range(0, total_length, block_size)]
            return {'input_ids': tokenized_examples}

        preproc_datasets = d.map(
            batch_preprocess,
            batched=True,
            batch_size=c.preprocess_map_bsz,
            writer_batch_size=c.preprocess_write_bsz,
            num_proc=c.num_preprocess_workers,
            remove_columns=d['train'].column_names,
            desc="Running tokenizer on dataset",
        )

        return preproc_datasets

    def setup(self, stage: Optional[str] = None) -> None:
        self.dataset = ds.load_from_disk(self.cache_file_path)
        self.tokenizer = tr.AutoTokenizer.from_pretrained(self.config.tokenizer,  use_fast=False)
        self.mask_id = self.tokenizer.mask_token_id
        # NOTE: this does not ensure replicability, but allows to study variance
        self.mask_rng = torch.Generator().manual_seed(self.config.mask_seed)
        self.shorten_rng = random.Random(self.config.shorten_seed)

    def randomly_shorten_batch(self, input_ids):
        for i, x in enumerate(input_ids):
            if self.shorten_rng.random() < self.config.short_seq_prob:
                max_tokens = len(x)
                seq_len = self.shorten_rng.randint(1, max_tokens)
                offset = self.shorten_rng.randint(0, max_tokens - seq_len)
                input_ids[i] = x[offset:offset + seq_len]
        return input_ids

    def collate_fn(self, examples: List[Dict[str, Any]]):
        input_ids = [x['input_ids'] for x in examples]
        if self.config.short_seq_prob is not None:
            input_ids = self.randomly_shorten_batch(input_ids)

        batch = self.tokenizer.batch_encode_plus(
            input_ids,
            padding='longest',
            add_special_tokens=True,
            return_special_tokens_mask=True,
            is_split_into_words=True,  # handle pre-tokenized input
            return_tensors="pt",
        )
        self._mask_batch(batch)
        return batch

    def _mask_batch(self, batch):
        if self.config.mlm_probability:
            batch['labels'] = batch['input_ids'].clone()
            prob_matrix = torch.full(batch["labels"].shape, self.config.mlm_probability)
            prob_matrix.masked_fill_(batch['special_tokens_mask'], value=0.0)
            masked_indices = torch.bernoulli(prob_matrix, generator=self.mask_rng).bool()
            batch["labels"][~masked_indices] = -100
            batch["input_ids"][masked_indices] = self.tokenizer.mask_token_id

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        c = self.config
        dataset = self.dataset['train']
        if c.shuffle_seed is not None:
            dataset = dataset.shuffle(seed=c.shuffle_seed)
        return DataLoader(
            dataset,
            batch_size=c.per_device_bsz,
            collate_fn=self.collate_fn,
            num_workers=c.num_dataloader_workers,
            shuffle=False,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        c = self.config
        return DataLoader(
            self.dataset['validation'],
            batch_size=c.per_device_bsz,
            collate_fn=self.collate_fn,
            num_workers=c.num_dataloader_workers,
            shuffle=False,
        )


if __name__ == '__main__':
    c = MlmDataModuleConfig(dataset_name='wikitext-2', complete_docs=True)
    dm = MlmDataModule(c)
    dm.prepare_data()
    dm.setup()
    tdl = dm.train_dataloader()
    print(next(iter(tdl)))
