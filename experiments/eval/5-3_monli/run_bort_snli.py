from argparse import ArgumentParser
from pathlib import Path
import os

import torch
import wandb
from torch.utils.data import DataLoader
from torch.optim import AdamW

import transformers as tr
import datasets as ds
import pytorch_lightning as pl
from pytorch_lightning.trainer.supporters import CombinedLoader

import sys
sys.path.append("/home/mila/m/mirceara/experiments/pretrain")
from run_bort import MlmModel, MlmModelConfig
from balaur.modeling.backbone.bort import BortForSequenceClassification
from balaur import ROOT_DIR

class BortSnliFinetune(pl.LightningModule):
    def __init__(self,
                 model_ckpt: str,
                 ft_regime: str,
                 bsz: int = 32,
                 tokenizer: str = 'roberta-base',
                 num_preproc_workers: int = 1,
                 num_labels: int = 3,
                 adam_wd: float = 0.1,
                 lr: float = 1e-5,
                 wu_proportion: float = 0.06,
                 num_epochs: int = 2,
                 seed: int = 1337,
                 wnre_factor: float = 1.0,
                 ):
        super().__init__()
        self.model_ckpt = model_ckpt
        self.ft_regime = ft_regime
        assert ft_regime in ["snli", "monli", "snli+monli"]
        self.bsz = bsz
        self.tokenizer = tokenizer
        self.num_preproc_workers = num_preproc_workers
        self.num_labels = num_labels
        self.adam_wd = adam_wd
        self.lr = lr
        self.wu_proportion = wu_proportion
        self.num_epochs = num_epochs
        self.seed = seed
        self.wnre_factor = wnre_factor

        self.save_hyperparameters()
        pl.seed_everything(self.seed)

        self._setup_model()
        self._setup_snli_dataset()
        self._setup_monli_dataset()
        self._setup_train_dataset()
        self.slow_tokenizer = None

    def _setup_model(self):
        ckpt = torch.load(self.model_ckpt)
        config = MlmModelConfig()
        config.__dict__.update(ckpt['hyper_parameters'])
        config.__dict__['wnre_factor'] = self.wnre_factor
        pl_model = MlmModel(config=config)
        pl_model.load_state_dict(ckpt['state_dict'])
        model = BortForSequenceClassification(pl_model.bort_config, num_labels=self.num_labels)
        model.bort = pl_model.model
        self.model = model

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.adam_wd,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,lr=self.lr, weight_decay=self.adam_wd,)

        num_total_steps = len(self.train_dataloader()) * self.num_epochs
        num_warmup_steps = num_total_steps * self.wu_proportion
        lr_scheduler = tr.get_scheduler(
            name='linear',
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_total_steps,
        )
        lr_dict = dict(
            scheduler=lr_scheduler,
            interval="step",
        )
        return dict(optimizer=optimizer, lr_scheduler=lr_dict)

    def training_step(self, batch, batch_idx):
        out = self.model(**batch)
        loss = out['loss']
        acc = (out['logits'].argmax(-1) == batch['labels']).float().mean()
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self.model(**batch)
        loss = out['loss']
        acc = (out['logits'].argmax(-1) == batch['labels']).float().mean()
        self.log('valid_loss', loss)
        self.log('valid_acc', acc)
        return loss

    def _setup_snli_dataset(self):
        d = ds.load_dataset('snli')
        d = d.filter(lambda x: x['label'] != -1)
        tknzr = tr.AutoTokenizer.from_pretrained(self.tokenizer)
        d = d.map(
            lambda x: tknzr(x['premise'], x['hypothesis']),
            batched=True, num_proc=self.num_preproc_workers
        )
        self.snli_dataset = d

    def _setup_monli_dataset(self):
        d = ds.DatasetDict(dict(
            nmonli = ds.load_dataset("json", data_files=str(ROOT_DIR / "eval/monli/nmonli_train.jsonl"), split='train'),
            nmonli_test = ds.load_dataset("json", data_files=str(ROOT_DIR / "eval/monli/nmonli_test.jsonl"), split='train'),
            pmonli = ds.load_dataset("json", data_files=str(ROOT_DIR / "eval/monli/pmonli.jsonl"), split='train')
        ))
        snli_class_label = self.snli_dataset['train'].features['label']
        d = d.map(
            lambda x: {'label': [snli_class_label.str2int(l) for l in x['gold_label']]},
            batched=True, num_proc=self.num_preproc_workers
        )
        d = d.cast_column('label', snli_class_label)
        tknzr = tr.AutoTokenizer.from_pretrained(self.tokenizer)
        d = d.map(
            lambda x: tknzr(x['sentence1'], x['sentence2']),
            batched=True, num_proc=self.num_preproc_workers
        )
        self.monli_dataset = d

    def _setup_train_dataset(self):
        if self.ft_regime == "snli":
            self.train_dataset = self.snli_dataset['train'].shuffle(seed=self.seed)
        elif self.ft_regime == "monli":
            self.train_dataset = ds.concatenate_datasets(
                [self.monli_dataset['pmonli'], self.monli_dataset['nmonli']]
            ).shuffle(seed=self.seed)
        elif self.ft_regime == "snli+monli":
            keep_cols = ['input_ids', 'label']
            d1 = self.snli_dataset['train']
            d1 = d1.remove_columns([c for c in d1.column_names if c not in keep_cols])
            d2 = ds.concatenate_datasets([self.monli_dataset['pmonli'], self.monli_dataset['nmonli']])
            d2 = d2.remove_columns([c for c in d2.column_names if c not in keep_cols])
            self.train_dataset = ds.concatenate_datasets([d1, d2]).shuffle(seed=self.seed)
        else:
            raise ValueError(f"Invalid ft_regime: {self.ft_regime}")

    def _get_slow_tokenizer(self):
        if self.slow_tokenizer is None:
            self.slow_tokenizer = tr.AutoTokenizer.from_pretrained(self.tokenizer, use_fast=False)
        return self.slow_tokenizer

    def _collate_fn(self, examples):
        tknzr = self._get_slow_tokenizer()
        batch = tknzr.batch_encode_plus(
            [x['input_ids'] for x in examples],
            padding='longest',
            add_special_tokens=True,
            return_special_tokens_mask=False,
            return_token_type_ids=False,
            is_split_into_words=True,  # handle pre-tokenized input
            return_tensors="pt",
        )
        batch['labels'] = torch.LongTensor([e['label'] for e in examples])
        return batch

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.bsz,
            collate_fn=self._collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.snli_dataset['validation'],
            batch_size=self.bsz,
            collate_fn=self._collate_fn
        )

    def test_dataloader(self):
        self.test_dataloader_map = ['snli', 'pmonli', 'nmonli', 'nmonli_test']
        get_dl = lambda d: DataLoader(d, batch_size=self.bsz, collate_fn=self._collate_fn)
        dls = [
            get_dl(self.snli_dataset['test']),
            get_dl(self.monli_dataset['pmonli']),
            get_dl(self.monli_dataset['nmonli']),
            get_dl(self.monli_dataset['nmonli_test']),
        ]
        return dls

    def test_step(self, batch, batch_idx, dataloader_idx):
        task = self.test_dataloader_map[dataloader_idx]
        out = self.model(**batch)
        loss = out['loss']
        acc = (out['logits'].argmax(-1) == batch['labels']).float().mean()
        self.log(f'{task}_loss', loss)
        self.log(f'{task}_acc', acc)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model_name", default="balaur_mlm_only_large")
    parser.add_argument("--tokenizer", default="roberta-base")
    parser.add_argument("--model_step", type=int, default=25000)
    parser.add_argument("--model_epoch", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--experiment_id", required=True, default=None)
    parser.add_argument("--ft_regime", default="monli")
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--wnre_factor", type=float, default=1.0)
    args = parser.parse_args()

    EXPERIMENT_DIR = f"/home/mila/m/mirceara/scratch/.cache/balaur/runs/{args.model_name}/snli_sbatch/"

    model_ckpt = f"/home/mila/m/mirceara/scratch/.cache/balaur/runs/{args.model_name}/" \
                 f"balaur/{args.model_name}/checkpoints/" \
                 f"epoch={args.model_epoch}-step={args.model_step}.ckpt"

    os.environ["PL_FAULT_TOLERANT_TRAINING"] = "automatic"
    plugins = [pl.plugins.environments.SLURMEnvironment(auto_requeue=True)]
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        every_n_epochs=1,
        verbose=True,
        save_top_k=-1,
        save_on_train_epoch_end=True,  # prevent saving every eval step
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(
        logging_interval='step',
    )
    trainer_callbacks = [checkpoint_callback, lr_monitor]
    logger = pl.loggers.WandbLogger(
        project="balaur",
        entity="amr-amr",
        name=args.experiment_id,
        id=args.experiment_id,
        save_dir=EXPERIMENT_DIR,
    )

    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        log_every_n_steps=100,
        callbacks=trainer_callbacks,
        precision=16,
        accelerator='gpu',
        devices=1,
        num_nodes=1,
        logger=logger,
        default_root_dir=EXPERIMENT_DIR,
        plugins=plugins,
    )

    model = BortSnliFinetune(
        model_ckpt=model_ckpt,
        ft_regime=args.ft_regime,
        wnre_factor=args.wnre_factor,
        tokenizer=args.tokenizer,
        seed=args.seed,
    )
    trainer.fit(model=model)
    trainer.test(model)
    wandb.finish()
