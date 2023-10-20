from dataclasses import dataclass
import os

import torch

from balaur import config as balaur_conf
from balaur.utils.logging import get_logger
from balaur.data.modules.mlm import MlmDataModuleConfig, MlmDataModule
from balaur.wordnet.relation_extractor import WordNetRelationExtractor as WNRE

from typing import *
SeqIdxT = int


CACHE_DIR = os.path.join(balaur_conf.DEFAULT_BALAUR_DATA_CACHE, "mlm")
logger = get_logger(__name__)

@dataclass
class MlmWnreDataModuleConfig(MlmDataModuleConfig):
    rel_depth: int = 3
    wnre: bool = True
    wnre_only_mask: bool = False


class MlmWnreDataModule(MlmDataModule):
    def __init__(self, config: MlmWnreDataModuleConfig):
        super().__init__(config)
        self.wnre = WNRE(config.tokenizer, rel_depth=config.rel_depth) if config.wnre else None
        self.relations = [r for r in self.wnre.relations] if self.wnre else []

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
        if self.config.wnre:
            self._wnre_batch(batch)
        if self.config.wnre_only_mask:
            self._wnre_only_mask_batch(batch)
        else:
            self._mask_batch(batch)
        return batch

    def _wnre_only_mask_batch(self, batch):
        if self.config.mlm_probability:
            batch['labels'] = batch['input_ids'].clone()
            adjusted_mlm_prob = self.config.mlm_probability * batch['input_ids'].numel() / batch['wnre_mask'].numel()
            adjusted_mlm_prob = min(adjusted_mlm_prob, 1.0)
            prob_matrix = torch.zeros(batch["labels"].shape, dtype=torch.float)
            prob_matrix.view(-1)[batch['wnre_mask']] = adjusted_mlm_prob
            masked_indices = torch.bernoulli(prob_matrix, generator=self.mask_rng).bool()
            batch["labels"][~masked_indices] = -100
            batch["input_ids"][masked_indices] = self.tokenizer.mask_token_id

    def _mask_batch(self, batch):
        if self.config.mlm_probability:
            batch['labels'] = batch['input_ids'].clone()
            prob_matrix = torch.full(batch["labels"].shape, self.config.mlm_probability)
            prob_matrix.masked_fill_(batch['special_tokens_mask'], value=0.0)
            masked_indices = torch.bernoulli(prob_matrix, generator=self.mask_rng).bool()
            batch["labels"][~masked_indices] = -100
            batch["input_ids"][masked_indices] = self.tokenizer.mask_token_id

    def _wnre_batch(self, batch, *args, **kwargs):
        # indices to subspample tokens with relations (in flattened hidden states)
        batch['wnre_mask'] = []
        # indices for related token-synset pairs in the logits resulting from subsampled hidden states
        for rel in self.relations:
            batch[f'wnre_{rel}'] = ([], [])

        bidx = 0  # index/counter for first dimension of logits resulting from subsampled hidden states
        for idx, token in enumerate(batch['input_ids'].view(-1).tolist()):
            rel_synsets = self.wnre.tok2rel2syn[token]
            if not rel_synsets:
                continue
            batch['wnre_mask'].append(idx)
            for rel, synsets in rel_synsets.items():
                batch[f'wnre_{rel}'][0].extend([bidx] * len(synsets))
                batch[f'wnre_{rel}'][1].extend(synsets)
            bidx += 1

        batch['wnre_mask'] = torch.LongTensor(batch['wnre_mask'])
        for rel in self.relations:
            batch[f'wnre_{rel}'] = (
                torch.LongTensor(batch[f'wnre_{rel}'][0]),
                torch.LongTensor(batch[f'wnre_{rel}'][1])
            )


if __name__ == '__main__':
    c = MlmDataModuleConfig(dataset_name='wikipedia')
    dm = MlmDataModule(c)
    dm.prepare_data()
    dm.setup()
    tdl = dm.train_dataloader()
    print(next(iter(tdl)))