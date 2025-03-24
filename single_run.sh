#!/bin/bash
#SBATCH --job-name="hyena_single"
#SBATCH --output="single_benchmark.%j.%N.out"
#SBATCH --partition=gpuA100x4
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bdhi-delta-gpu
#SBATCH -t 24:00:00
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out

# Load Conda
echo "Loading Conda..."
source /sw/external/python/anaconda3/etc/profile.d/conda.sh || eval "$(conda shell.bash hook)"

# Activate Conda Environment
echo "Activating Conda environment: hyena-dna-env"
conda activate hyena-dna-env

# Debug: Check Python and CUDA
echo "Python path:"
which python
echo "Checking CUDA availability and version..."
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('GPU device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

# Ensure logs are immediately written
export PYTHONUNBUFFERED=1

# Set up CUDA environment variables
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1

# Run the script
echo "Running single dataset training script..."
python3 single_dataset.py