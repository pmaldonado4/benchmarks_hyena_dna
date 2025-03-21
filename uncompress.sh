#!/bin/bash
#SBATCH --account=bdhi-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --job-name=decompress_data
#SBATCH --output=decompress.out.%j
#SBATCH --error=decompress.err.%j
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32          # Use multiple cores for pigz
#SBATCH --time=24:00:00
#SBATCH --mem=16G

echo "Starting decompression job at $(date)"
echo "Running on node: $(hostname)"

# Optionally, load pigz if it's provided as a module.
# For example: module load pigz
# If pigz is in your PATH, this may not be necessary.

# Run tar with pigz as the decompression program.
tar --use-compress-program=pigz -xf /work/hdd/bdhi/pmaldonadocatala/genomic_benchmarks/benchmark_data.tar.gz \
    -C /work/hdd/bdhi/pmaldonadocatala/genomic_benchmarks/

echo "Decompression finished at $(date)"