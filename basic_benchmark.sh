#!/bin/bash
#SBATCH --job-name="hyena_finetune"            # Job name
#SBATCH --output="hyena_finetune.%j.out"       # Standard output log
#SBATCH --error="hyena_finetune.%j.err"        # Standard error log
#SBATCH --partition=gpuA40x4                   # Partition to use, adjust as needed
#SBATCH --nodes=1                              # Number of nodes
#SBATCH --ntasks-per-node=1                    # 1 task per node (adjust if necessary)
#SBATCH --cpus-per-task=4                      # Number of CPUs per task
#SBATCH --mem=128G                             # Total memory per node
#SBATCH --gpus-per-node=1                      # Number of GPUs per node
#SBATCH --time=02:00:00                        # Maximum run time
#SBATCH --account=bdhi-delta-gpu       # Replace with your actual account name
#SBATCH -t 2:00:00                    # Adjust time as needed
#SBATCH -e slurm-%j.err                # Standard error log
#SBATCH -o slurm-%j.out       # Your email for notifications

# Load necessary modules (adjust based on your environment)
module load python/3.11.6 
module load cuda/11.8.0 
module load nvidia/24.5

# Activate your environment (assuming you're using Poetry)
source $(poetry env info --path)/bin/activate 

# Set up WandB for tracking the experiment
wandb login 7194520caeea29f37cac5c0c29b0432cb3f9b59c

# Run the training script using srun
srun --mpi=pmi2 python3 -m train \
    wandb.project=hyena-dna-genomic-benchmark \
    +wandb.entity=gregg_lab \
    experiment=hg38/genomic_benchmark \
    +dataset_name=human_enhancers_cohn \
    train.pretrained_model_path=/u/pmaldonadocatala/hyena-dna/weights/basic.ckpt \
    dataset.max_length=500 \
    ++model.layer.l_max=1024 \
    optimizer.lr=1e-4 \
    trainer.devices=1 \
    trainer.num_nodes=1 \
    trainer.accelerator=gpu