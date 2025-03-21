#!/bin/bash
#SBATCH --job-name="hyena_pretrain_test"
#SBATCH --output="a.out.%j.%N.out"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=64G
#SBATCH --nodes=1                     # Request 4 nodes
#SBATCH --ntasks-per-node=1            # 1 task per node (since we use 1 GPU per node)
#SBATCH --cpus-per-task=4              # Allocate 4 CPUs per task
#SBATCH --gpus-per-node=1              # 1 GPU per node
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

# Run the training script using srun with the required number of GPUs and CPUs
srun --mpi=pmi2 python3 -m inference_test_ver_2.py \
   wandb.project=hyena-dna-multispecies-inference-human_enhancers_cohn \
   +wandb.entity=gregg_lab \
   experiment=hg38/genomic_benchmark \
   +dataset_name=human_enhancers_cohn \
   train.pretrained_model_path=/u/pmaldonadocatala/hyena-dna/weights/weights.ckpt \
   dataset.max_length=200 \
   model.layer.l_max=1026 \
   +model.layer.embed_size=128 \
   +model.layer.mlp_size=1026 \
   trainer.devices=1 \
   trainer.num_nodes=1 \
   trainer.accelerator=gpu 


    
    ## Main entry point for training.  Select the dataset name and metadata, as
    ## well as model and training args, and you're off to the genomic races!

    ### GenomicBenchmarks Metadata
    # there are 8 datasets in this suite, choose 1 at a time, with their corresponding settings
    # name                                num_seqs        num_classes     median len    std
    # dummy_mouse_enhancers_ensembl       1210            2               2381          984.4
    # demo_coding_vs_intergenomic_seqs    100_000         2               200           0
    # demo_human_or_worm                  100_000         2               200           0
    # human_enhancers_cohn                27791           2               500           0
    # human_enhancers_ensembl             154842          2               269           122.6
    # human_ensembl_regulatory            289061          3               401           184.3
    # human_nontata_promoters             36131           2               251           0
    # human_ocr_ensembl                   174756          2               315           108.1

