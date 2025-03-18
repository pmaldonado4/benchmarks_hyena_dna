#!/bin/bash
#SBATCH --job-name="metabolic_classification"
#SBATCH --output="benchmark.out.%j.%N.out"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2          # One task per GPU
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=2            # Using all 4 GPUs on the node
#SBATCH --gpu-bind=closest
#SBATCH --account=bdhi-delta-gpu
#SBATCH -t 10:00:00
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out

export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /projects/bdhi/pmaldonadocatala/poetry_envs/hyena-pretrained-CQmhwyTM-py3.11/bin/activate

torchrun --nproc_per_node=2 metabolic_hyena_classification_dpp_freeze_02.py