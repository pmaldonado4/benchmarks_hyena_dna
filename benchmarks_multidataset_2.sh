#!/bin/bash
#SBATCH --job-name="hyena_benchmark"
#SBATCH --output="benchmark.out.%j.%N.out"
#SBATCH --partition=gpuA100x4
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bdhi-delta-gpu
#SBATCH -t 48:00:00
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out

# Load Conda
echo "Loading Conda..."
source /sw/external/python/anaconda3/etc/profile.d/conda.sh || eval "$(conda shell.bash hook)"

# Activate Conda Environment
echo "Activating Conda environment: hyena-dna-env"
conda activate hyena-dna-env

# Debug: Check Python
echo "Python path:"
which python
echo "Checking Torch version..."
python -c "import torch; print('Torch version:', torch.__version__)"

# Ensure logs are immediately written
export PYTHONUNBUFFERED=1

# Set up distributed training environment variables
export MASTER_PORT=29500
export MASTER_ADDR=$(hostname)
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

# Set NCCL environment variables
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=eth1

# Run the script
echo "Running script..."

# Run the training script with srun
srun --mpi=pmi2 \
     --cpu-bind=cores \
     --distribution=block:block \
     --hint=nomultithread \
     --gpu-bind=closest \
     python3 benchmarks_multidataset.py