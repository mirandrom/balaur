from dataclasses import dataclass, field
from datetime import datetime as dt
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.optim import AdamW
from torch import nn
from torch.nn import functional as F
import transformers as tr
import wandb

from balaur.modeling.backbone.bort import BortBackbone, BortMLMHead, BortConfig
from balaur.modeling.utils.activations import ACT2NN
from balaur.modeling.heads.balaur import BalaurHeads
from balaur.data.modules.mlm_wnre import MlmWnreDataModule, MlmWnreDataModuleConfig
from balaur.config import BALAUR_RUN_CACHE
from balaur.utils.log_scale_val_callback import LogScaleValCheckCallback
from balaur.wordnet.relation_extractor import WordNetRelationExtractor as WNRE

from torch import Tensor, FloatTensor
from typing import List, Tuple

SrcIdxT = torch.LongTensor
DstIdxT = torch.LongTensor
TargetsT = Tuple[SrcIdxT, DstIdxT]

@dataclass
class MlmModelConfig:
    wnre_factor: float = 1.0
    layer_norm_eps: float = 1e-7
    large: bool = False
    model_seed: int = field(
        default=1337,
        metadata={'help': 'Seed to be used by `pl.seed_everything`.'}
    )
    learning_rate: float = field(
        default=5e-5,
        metadata={'help': 'Learning rate (peak if used with scheduler).'}
    )
    lr_scheduler_name: str = field(
        default='cosine',
        metadata={'help': 'Name of learning rate scheduler from huggingface '
                          'transformers library. Valid options are '
                          f'{[x.value for x in tr.optimization.SchedulerType]}'
                  }
    )
    num_warmup_steps: int = field(
        default=3000,
        metadata={'help': 'Number of warmup steps for `lr_scheduler_name`.'}
    )
    num_warmup_total_steps: int = field(
        default=None,
        metadata={'help': 'Number of total steps for `lr_scheduler_name`.'}
    )
    gradient_clip_val: float = field(
        default=0,
        metadata={'help': 'Gradient clipping (by L2 norm); '
                          'for use by pytorch-lightning `Trainer`. '}
    )
    adam_eps: float = field(
        default=1e-8,
        metadata={'help': 'Hyperparameter for AdamW optimizer.'}
    )
    adam_wd: float = field(
        default=0.0,
        metadata={'help': 'Weight decay for AdamW optimizer.'}
    )
    adam_beta_1: float = field(
        default=0.9,
        metadata={'help': 'Hyperparameter for AdamW optimizer.'}
    )
    adam_beta_2: float = field(
        default=0.999,
        metadata={'help': 'Hyperparameter for AdamW optimizer.'}
    )

    def __post_init__(self):
        self.model_name = 'bort'

    def set_tokenizer(self, tokenizer: str):
        if tokenizer == 'bert-base-uncased':
            self.bort_kwargs = dict(vocab_size=30522,
                                    mask_token_id=103,
                                    pad_token_id=0,
                                    layer_norm_eps=self.layer_norm_eps)
        elif tokenizer == 'bert-base-cased':
            self.bort_kwargs = dict(vocab_size=28996,
                                    mask_token_id=103,
                                    pad_token_id=0,
                                    layer_norm_eps=self.layer_norm_eps)
        elif tokenizer == 'roberta-base':
            self.bort_kwargs = dict(vocab_size=50265,
                                    mask_token_id=50264,
                                    pad_token_id=1,
                                    layer_norm_eps=self.layer_norm_eps)
        else:
            raise ValueError(f'Unknown tokenizer: {tokenizer}')

        if self.large:
            self.bort_kwargs.update(
                dict(
                    num_attention_heads=16,
                    num_hidden_layers=24,
                    hidden_size=1024,
                    intermediate_size=4096,
                )
            )

    def set_wnre_balaur(self, wnre: WNRE, use_wnre: bool = False):
        if use_wnre:
            self.wnre = True
            self.wnre_embed_num = len(wnre.synset_vocab)
            self.wnre_relations = wnre.relations
        else:
            self.wnre = False


class MlmModel(pl.LightningModule):
    def __init__(self, config: MlmModelConfig):
        super().__init__()
        self.config = config
        self.bort_config = BortConfig(**self.config.bort_kwargs)
        self.save_hyperparameters(config.__dict__)
        pl.seed_everything(self.config.model_seed)
        self.init_model()

    def init_model(self):
        self.model = BortBackbone(self.bort_config)
        self.model.apply(self.init_weights)
        self.head = BortMLMHead(self.bort_config)
        self.head.apply(self.init_weights)
        self.head.decoder.weight = self.model.embeddings.word_embeddings.weight
        # TODO: look into initializations for embeds
        if self.config.wnre:
            wnre_dim = int(self.bort_config.hidden_size * self.config.wnre_factor)
            self.wnre_embeds = nn.Embedding(self.config.wnre_embed_num, wnre_dim)
            self.wnre_embeds.apply(self.init_weights)
            self.wnre_head = BalaurHeads(
                src_features=self.bort_config.hidden_size,
                dst_features=wnre_dim,
                head_dim=wnre_dim,
                num_heads=len(self.config.wnre_relations),
                vocab_dim=self.config.wnre_embed_num,
                eps=self.config.layer_norm_eps,
            )
            self.wnre_head.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.bort_config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.adam_wd,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.config.learning_rate,
                          betas=(self.config.adam_beta_1,
                                 self.config.adam_beta_2),
                          eps=self.config.adam_eps,
                          weight_decay=self.config.adam_wd,
                          )
        lr_scheduler = tr.get_scheduler(
            name=self.config.lr_scheduler_name,
            optimizer=optimizer,
            num_warmup_steps=self.config.num_warmup_steps,
            num_training_steps=self.config.num_warmup_total_steps,
        )
        lr_dict = dict(
            scheduler=lr_scheduler,
            interval="step",
        )
        return dict(optimizer=optimizer, lr_scheduler=lr_dict)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        # HACK: default hook does not handle wnre targets
        for k,v in batch.items():
            if k in ['input_ids', 'attention_mask', 'labels']:
                batch[k] = v.to(device)
        return batch

    def forward(self, batch):
        out = self.model(batch['input_ids'])
        return out

    def training_step(self, batch, batch_idx):
        src = self.forward(batch)
        src = src.view(-1, src.shape[-1])
        mask_unmasked = batch['input_ids'].view(-1) == self.bort_config.mask_token_id
        mlm_src = src[mask_unmasked]
        labels = batch['labels'].view(-1)[mask_unmasked]
        logits = self.head(mlm_src)
        loss = F.cross_entropy(logits, labels)
        mlm_loss = loss.item()

        if self.config.wnre:
            wnre_src = src[batch['wnre_mask']]
            wnre_dst = self.wnre_embeds.weight
            wnre_targets = [batch[f'wnre_{rel}'] for rel in self.config.wnre_relations]
            wnre_loss, wnre_losses = self.wnre_head.compute_loss(wnre_src, wnre_dst, wnre_targets)
            loss += wnre_loss
            for rel, rel_loss in zip(self.config.wnre_relations, wnre_losses):
                self.log(f'train_{rel}_loss', rel_loss)

        self.log('train_loss', loss.item())
        self.log('train_mlm_loss', mlm_loss)
        # manually log global_step to prevent issues with resuming from ckpt
        # see: https://github.com/PyTorchLightning/pytorch-lightning/issues/13163
        self.log('global_step', self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        src = self.forward(batch)
        src = src.view(-1, src.shape[-1])
        mask_unmasked = batch['input_ids'].view(-1) == self.bort_config.mask_token_id
        mlm_src = src[mask_unmasked]
        labels = batch['labels'].view(-1)[mask_unmasked]
        logits = self.head(mlm_src)
        loss = F.cross_entropy(logits, labels)
        mlm_loss = loss.item()

        if self.config.wnre:
            wnre_src = src[batch['wnre_mask']]
            wnre_dst = self.wnre_embeds.weight
            wnre_targets = [batch[f'wnre_{rel}'] for rel in self.config.wnre_relations]
            wnre_loss, wnre_losses = self.wnre_head.compute_loss(wnre_src, wnre_dst, wnre_targets)
            loss += wnre_loss
            for rel, rel_loss in zip(self.config.wnre_relations, wnre_losses):
                self.log(f'eval_{rel}_loss', rel_loss)

        self.log('eval_loss', loss.item())
        self.log('eval_mlm_loss', mlm_loss)
        self.log('global_step', self.global_step)
        return loss


@dataclass
class TrainerConfig:
    experiment_name: str = field(
        default=None,
        metadata={'help': 'Identifier for experiment.'}
    )
    timestamp_id: bool = field(
        default=True,
        metadata={'help': 'Add timestamp to experiment id.'}
    )
    log_dir: str = field(
        default=BALAUR_RUN_CACHE,
        metadata={'help': 'Parent directory where model checkpoints, logs, '
                          'and other artefacts will be stored. Note that '
                          'these are saved in a subdirectory named according '
                          'to the experiment identifier.'}
    )
    num_training_steps: int = field(
        default=int(1e5),
        metadata={'help': 'Maximum number of training steps for experiment.'}
    )
    save_every_n_steps: int = field(
        default=None,
        metadata={'help': 'Save a model checkpoint every n steps.'}
    )
    save_last: bool = field(
        default=True,
        metadata={'help': 'Save a model checkpoint at the end of training.'}
    )
    save_on_eval: bool = False
    log_every_n_steps: int = field(
        default=100,
        metadata={'help': 'Log metrics during training every n steps.'}
    )
    eval_every_n_steps: int = field(
        default=100,
        metadata={'help': 'Run validation step during training every n steps.'}
    )
    eval_every_log_steps: bool = field(
        default=False,
        metadata={'help': 'Eval on steps 1,2,...10,20,...100,200,...1000, etc.'}
    )
    limit_val_batches: int = field(
        default=10,
        metadata={'help': 'Limit validation batches. Set to 0.0 to skip.'}
    )
    skip_eval: bool = field(
        default=False,
        metadata={'help': 'Skip eval loop when training.'}
    )
    # HACK: HfArgumentParser does not support union types which we would need
    #       to handle "bf16" as a valid option for precision. Instead we
    #       include bf16 as a separate bool flag that overrides precision.
    bf16: bool = field(
        default=False,
        metadata={'help': 'Use bf16 training (overrides `precision`).'}
    )
    precision: int = field(
        default=32,
        metadata={'help': 'Precision flag for pytorch-lightning Trainer.'}
    )
    total_bsz: int = field(
        default=None,
        metadata={'help': 'Total batch size (use with `per_device_bsz` to '
                          'automatically determine gradient accumulation '
                          'steps consistent with number of nodes/devices). '
                          'If None, is set to `per_device_bsz` times the '
                          'total number of devices across nodes.'}
    )
    accelerator: str = field(
        default='gpu',
        metadata={'help': 'Accelerator flag for pytorch-lightning Trainer.'}
    )
    devices: str = field(
        default="1",
        metadata={'help': 'Number of accelerator devices or list of device ids.'
                          'Must be a str that will evaluate to a python '
                          'expression. E.g. "1" or "[0,1]".'}
    )
    num_nodes: int = field(
        default=1,
        metadata={'help': 'Number of nodes.'}
    )
    strategy: str = field(
        default=None,
        metadata={'help': 'Set to `ddp` for multiple devices or nodes. \n'
                          'Can also be set to e.g. `deepspeed_stage_3_offload` \n'
                          'or any other keyword strategy supported by '
                          'pytorch-lightning. '}
    )
    fault_tolerant: bool = field(
        default=True,
        metadata={'help': 'Whether to run pytorch-lightning in fault-tolerant mode. \n'
                          'See: https://pytorch-lightning.readthedocs.io/en/1.6.3/advanced/fault_tolerant_training.html?highlight=fault%%20tolerant \n'
                          'and: https://github.com/PyTorchLightning/pytorch-lightning/blob/1.6.3/pl_examples/fault_tolerant/automatic.py'}
    )
    slurm_auto_requeue: bool = field(
        default=True,
        metadata={'help': 'Whether to run pytorch-lightning with auto-requeue. \n'
                          'See: https://pytorch-lightning.readthedocs.io/en/1.6.3/clouds/cluster.html?highlight=slurm#wall-time-auto-resubmit'}
    )
    wandb_project: str = field(
        default='balaur',
        metadata={'help': 'Weights and bias project to log to.'}
    )
    wandb_entity: str = field(
        default='amr-amr',
        metadata={'help': 'Weights and bias entity to log to.'}
    )

    experiment_id: str = field(init=False)
    experiment_dir: Path = field(init=False)

    def __post_init__(self):
        self.experiment_id = self.init_experiment_id()
        self.experiment_dir = self.init_experiment_dir()
        self.devices = eval(self.devices)
        self.precision = "bf16" if self.bf16 else self.precision
        if self.skip_eval:
            self.limit_val_batches = 0.0
            self.eval_every_n_steps = 1.0
        if isinstance(self.devices, list):
            self.num_devices = len(self.devices)
        elif isinstance(self.devices, int):
            self.num_devices = self.devices
        else:
            raise ValueError("devices must be an int or a list of ints")

    def init_experiment_id(self):
        ts = dt.utcnow().isoformat(timespec='seconds')
        if self.experiment_name and self.timestamp_id:
            return f"{self.experiment_name}_{ts}".replace(":", "")
        if self.experiment_name:
            return self.experiment_name
        if self.timestamp_id:
            return ts.replace(":", "")
        else:
            raise ValueError("If `experiment_name` is not specified, "
                             "`timestamp_id` must be True.")

    def init_experiment_dir(self):
        p = Path(self.log_dir) / self.experiment_id
        p.mkdir(parents=True, exist_ok=True)
        return p


def main():
    # parse arguments
    dc: MlmWnreDataModuleConfig
    mc: MlmModelConfig
    tc: TrainerConfig
    configs = (MlmWnreDataModuleConfig, MlmModelConfig, TrainerConfig)
    parser = tr.HfArgumentParser(configs)
    dc, mc, tc = parser.parse_args_into_dataclasses()
    mc.set_tokenizer(dc.tokenizer)

    # setup fault tolerant / auto-requeue
    if tc.fault_tolerant:
        os.environ["PL_FAULT_TOLERANT_TRAINING"] = "automatic"
    if tc.slurm_auto_requeue:
        plugins = [pl.plugins.environments.SLURMEnvironment(auto_requeue=True)]
    else:
        plugins = [pl.plugins.environments.SLURMEnvironment(auto_requeue=False)]

    # setup gradient accumulation
    if tc.total_bsz is None:
        tc.total_bsz = dc.per_device_bsz * tc.num_devices * tc.num_nodes
    accumulate_grad_batches = (tc.total_bsz // dc.per_device_bsz
                               // tc.num_devices // tc.num_nodes)
    assert (tc.total_bsz == dc.per_device_bsz * tc.num_devices
            * accumulate_grad_batches * tc.num_nodes)
    tc.eval_every_n_steps *= accumulate_grad_batches

    # setup warmup steps
    mc.num_warmup_total_steps = mc.num_warmup_total_steps or tc.num_training_steps

    # callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        every_n_train_steps=tc.save_every_n_steps,
        save_last=tc.save_last,
        verbose=True,
        save_top_k=-1,
        save_on_train_epoch_end=not tc.save_on_eval,  # prevent saving every eval step
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(
        logging_interval='step',
    )
    trainer_callbacks = [checkpoint_callback, lr_monitor]
    if tc.eval_every_log_steps:
        trainer_callbacks.append(LogScaleValCheckCallback(accumulate_grad_batches))

    # logger
    if tc.wandb_project:
        logger = pl.loggers.WandbLogger(
            project=tc.wandb_project,
            entity=tc.wandb_entity,
            name=tc.experiment_id,
            id=tc.experiment_id,
            save_dir=str(tc.experiment_dir),
        )
    else:
        # will use tensorboard logger by default
        logger = True

    trainer = pl.Trainer(
        max_steps=tc.num_training_steps,
        max_epochs=None,
        log_every_n_steps=tc.log_every_n_steps,
        val_check_interval=tc.eval_every_n_steps,
        callbacks=trainer_callbacks,
        precision=tc.precision,
        strategy=tc.strategy,
        accelerator=tc.accelerator,
        devices=tc.devices,
        num_nodes=tc.num_nodes,
        accumulate_grad_batches=accumulate_grad_batches,
        logger=logger,
        default_root_dir=str(tc.experiment_dir),
        gradient_clip_val=mc.gradient_clip_val,
        limit_val_batches=tc.limit_val_batches,
        plugins=plugins,
    )
    datamodule = MlmWnreDataModule(dc)
    mc.set_wnre_balaur(datamodule.wnre, dc.wnre)
    model = MlmModel(mc)
    trainer.fit(model=model, datamodule=datamodule)
    wandb.finish()


if __name__ == '__main__':
    main()
