#!/bin/bash
#SBATCH --job-name="hyena_benchmark"
#SBATCH --output="benchmark_out.%j.%N.out"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=64G
#SBATCH --nodes=1                     # Request 1 node for this benchmark
#SBATCH --ntasks-per-node=1           # 1 task per node
#SBATCH --cpus-per-task=4             # Allocate 4 CPUs per task
#SBATCH --gpus-per-node=1             # 1 GPU per node
#SBATCH --gpu-bind=closest            # Bind CPUs closest to the GPUs
#SBATCH --account=bdhi-delta-gpu      # Replace with your actual account name
#SBATCH -t 01:00:00                   # Adjust time as needed
#SBATCH -e slurm-%j.err               # Standard error log
#SBATCH -o slurm-%j.out               # Standard output log

# Set OpenMP to use 1 thread per process
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Optimizing memory allocation

# Load necessary modules
module load nvidia/24.5
module load python/3.11.6  # Ensure the correct Python version
module load cuda/11.8.0   # Ensure the correct CUDA version

# Activate the environment
source $(poetry env info --path)/bin/activate 

# Run the benchmarking script
srun python3 inference_test.py  # Replace with your script name