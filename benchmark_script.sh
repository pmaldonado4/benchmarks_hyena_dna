#!/bin/bash
#SBATCH --job-name="hyena_benchmark"
#SBATCH --output="benchmark.out.%j.%N.out"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bdhi-delta-gpu
#SBATCH -t 01:00:00
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out

export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  

source /projects/bdhi/pmaldonadocatala/poetry_envs/hyena-pretrained-CQmhwyTM-py3.11/bin/activate


srun --mpi=pmi2 python3 benchmark_simple.py