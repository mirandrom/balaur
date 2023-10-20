# Balaur
Contains code for "Balaur: Language Model Pretraining with Lexical Semantic Relations", accepted at Findings of EMNLP 2023. 

## Setup
```bash
git clone https://github.com/mirandrom/balaur.git
cd balaur
pip install -e .
```

## Reproducibility
### Pretraining baseline
See `experiments/pretrain/mlm_only.sh` or 

```bash
python experiments/pretrain/run_bort.py \
--experiment_name='mlm_only' \
--dataset_name='wikibooks' \
--large \
--no_wnre \
--tokenizer='roberta-base' \
--short_seq_prob=0.1 \
--complete_docs \
--devices=1 \
--num_nodes=8 \
--strategy='ddp' \
--per_device_bsz=128 \
--total_bsz=4096 \
--learning_rate=2e-3 \
--adam_wd=0.01 \
--lr_scheduler_name='linear' \
--num_warmup_steps=1500 \
--num_training_steps=25000 \
--save_every_n_steps=1000 \
--skip_eval \
--log_every_n_steps=10 \
--slurm_auto_requeue \
--no_timestamp_id \
--precision=16 \
--save_last \
--num_dataloader_workers=1 \
```

### Pretraining with Balaur
See `experiments/pretrain/mlm_wnre.sh` or 

```bash
python experiments/pretrain/run_bort.py \
--experiment_name='mlm_wnre' \
--dataset_name='wikibooks' \
--large \
--wnre \
--wnre_factor=0.75 \
--tokenizer='roberta-base' \
--short_seq_prob=0.1 \
--complete_docs \
--devices=1 \
--num_nodes=8 \
--strategy='ddp' \
--per_device_bsz=128 \
--total_bsz=4096 \
--learning_rate=2e-3 \
--adam_wd=0.01 \
--lr_scheduler_name='linear' \
--num_warmup_steps=1500 \
--num_training_steps=25000 \
--save_every_n_steps=1000 \
--skip_eval \
--log_every_n_steps=10 \
--slurm_auto_requeue \
--no_timestamp_id \
--precision=16 \
--save_last \
--num_dataloader_workers=1 \
```

### Evaluations
See `experiments/eval` and [HypCC dataset](https://github.com/mirandrom/balaur/blob/master/experiments/eval/5-2_hypcc/hypcc.csv)