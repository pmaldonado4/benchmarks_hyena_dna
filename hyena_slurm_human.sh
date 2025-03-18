#!/bin/bash
#SBATCH --job-name="hyena_pretrain_test"
#SBATCH --output="a.out.%j.%N.out"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=64G
#SBATCH --nodes=4                     # Request 2 nodes
#SBATCH --ntasks-per-node=1            # 1 task per node (since we use 1 GPU per node)
#SBATCH --cpus-per-task=4              # Allocate 4 CPUs per task
#SBATCH --gpus-per-node=1              # 1 GPU per node
#SBATCH --gpu-bind=closest             # Bind CPUs closest to the GPUs
#SBATCH --account=bdhi-delta-gpu       # Replace with your actual account name
#SBATCH -t 01:00:00                    # Adjust time as needed
#SBATCH -e slurm-%j.err                # Standard error log
#SBATCH -o slurm-%j.out                # Standard output log

# Set OpenMP to use 1 thread per process
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Optimizing memory allocation

# Load necessary modules 
module load nvidia/24.5
module load python/3.11.6  # Ensure the correct Python version
module load cuda/11.8.0   # Ensure the correct CUDA version, adjust if necessary

# Activate the environment
source $(poetry env info --path)/bin/activate 

# Login to WandB if not done globally
wandb login 7194520caeea29f37cac5c0c29b0432cb3f9b59c
# Run the training script using srun with the required number of GPUs and CPUs
srun --mpi=pmi2 python3 -m train \
   wandb.project=hyena-dna-human-1k-test \
   wandb.entity=gregg_lab \
   experiment=hg38/hg38_hyena \
   model.d_model=128 \
   model.n_layer=2 \
   dataset.batch_size=32 \
   train.global_batch_size=128 \
   dataset.max_length=450002 \
   optimizer.lr=3e-4 \
   trainer.devices=1 \
   trainer.num_nodes=4 \
   trainer.accelerator=gpu \
# Notes:
# 1. Ensure you replace `your-wandb-api-key` with your actual WandB API key.
# 2. Replace `your-wandb-entity` with your actual WandB entity or team name.
# 3. The `trainer.devices=4` flag ensures that all 4 GPUs are used.
# 4. Adjust the script’s `-t` (time) parameter based on the expected duration of your job.
# 5. Make sure the modules and environment are set up correctly before running.