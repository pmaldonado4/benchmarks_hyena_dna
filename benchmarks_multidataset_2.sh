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

# Run the script
echo "Running script..."

# Set up distributed training environment variables
export MASTER_PORT=29500
export MASTER_ADDR=$(hostname)
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

# Run the training script with srun
srun --mpi=pmi2 \
     --cpu-bind=cores \
     --distribution=block:block \
     --hint=nomultithread \
     python3 benchmarks_multidataset.py