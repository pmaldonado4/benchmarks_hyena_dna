#!/bin/bash
#SBATCH --job-name="hyena_benchmark"
#SBATCH --output="benchmark.out.%j.%N.out"
#SBATCH --partition=gpuA100x4
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=3
#SBATCH --gpu-bind=closest
#SBATCH --account=bdhi-delta-gpu
#SBATCH -t 48:00:00
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out

# Set environment variables for distributed training
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Set up distributed training environment variables
export MASTER_PORT=29500
export MASTER_ADDR=$(hostname)
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

# Activate your conda environment
source /projects/bdhi/pmaldonadocatala/poetry_envs/hyena-dna-CQmhwyTM-py3.8/bin/activate
# Run the training script with srun
srun --mpi=pmi2 \
     --cpu-bind=cores \
     --distribution=block:block \
     --hint=nomultithread \
     python3 benchmarks_multidataset.py