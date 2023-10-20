#!/bin/bash

#SBATCH --array=1-5
#SBATCH --job-name=balaur_mlm_wnre_large_joint
#SBATCH --output=log-balaur_mlm_wnre_large_joint.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --time=12:00:00
#SBATCH --mem=48Gb
#SBATCH --cpus-per-task=8

# setup vars
conda_env="balaur"
home_dir="/home/mila/m/mirceara"
exp_dir="${home_dir}/balaur/experiments/eval/5-3_monli"
script_path="${exp_dir}/run_bort_snli.py"
run_from_path="${exp_dir}"
bashrc_path="${home_dir}/.bashrc"

seed=$SLURM_ARRAY_TASK_ID

# script args
args=" \
--model_name=2022_12_20_mlm_wnre_large \
--experiment_id=2022_12_20_mlm_wnre_large_joint_snli-seed-${seed} \
--ft_regime='snli+monli' \
--model_step=25000 \
--seed=$seed \
--wnre_factor=0.75 \
"

# setup environment
cd $run_from_path
pwd; hostname; date
source ${bashrc_path}
module load anaconda/3
activate $conda_env
export PYTHONFAULTHANDLER=1
export BALAUR_CACHE=/network/scratch/m/mirceara/.cache/balaur

# run experiment
cmd="srun python $script_path $args"
echo $cmd
eval $cmd