#!/bin/bash
#SBATCH --job-name="genomic_benchmark_finetune"
#SBATCH --output="a.out.%j.%N.out"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=60G
#SBATCH --nodes=1                     # Request 1 node
#SBATCH --ntasks-per-node=1            # 1 task per node (since we use 1 GPU per node)
#SBATCH --cpus-per-task=4              # Allocate 4 CPUs per task
#SBATCH --gpus-per-node=1              # Request 2 GPUs per node
#SBATCH --gpu-bind=closest             # Bind CPUs closest to the GPUs
#SBATCH --account=bdhi-delta-gpu       # Replace with your actual account name
#SBATCH -t 00:10:00                    # Adjust time as needed
#SBATCH -e slurm-%j.err                # Standard error log
#SBATCH -o slurm-%j.out                # Standard output log

# Set OpenMP to use 1 thread per process
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Optimizing memory allocation

# Activate the environment
source $(poetry env info --path)/bin/activate 

# Login to WandB if not done globally
wandb login 7194520caeea29f37cac5c0c29b0432cb3f9b59c

# Run the fine-tuning script for GenomicBenchmarks

srun --mpi=pmi2 python3 -m train \
   wandb.project=hyena-dna-genomic-benchmark-demo_coding_vs_intergenomic_seqs \
   +wandb.entity=gregg_lab \
   experiment=hg38/genomic_benchmark \
   +dataset_name=demo_coding_vs_intergenomic_seqs \
   train.pretrained_model_path=//u/pmaldonadocatala/benchmarks_hyena_dna/checkpoints/hyenadna-tiny-1k-seqlen/weights.ckpt \
   dataset.max_length=500 \
   ++model.layer.l_max=32004 \
   ++model.d_model=256 \
   ++model.n_layer=4 \
   ++model.resid_dropout=0.2 \
   ++model.embed_dropout=0.3 \
   ++dataset.batch_size=256 \
   ++optimizer.weight_decay=0.001 \
   optimizer.lr=1e-5 \
   trainer.devices=1 \
   trainer.num_nodes=1 \
   trainer.accelerator=gpu
# Notes:
# 1. Ensure your WandB API key is set up and your environment is configured properly. 
# 2. Change `train.pretrained_model_path` to point to your actual pretrained model checkpoint.
# 3. Adjust dataset.max_length and model.layer.l_max as required for different datasets.