#!/usr/bin/env python3
import os
from pathlib import Path

# Adjust the import based on where download_dataset and is_downloaded are defined.
# Here we assume they're in genomic_benchmarks/download.py
from genomic_benchmarks.loc2seq import download_dataset
from genomic_benchmarks.data_check import is_downloaded

# Define the local destination directory where datasets will be downloaded.
local_dest_path = "/work/hdd/bdhi/pmaldonadocatala/genomic_benchmarks"

# List of dataset names to download.
datasets = [
    'dummy_mouse_enhancers_ensembl',
    'demo_coding_vs_intergenomic_seqs',
    'demo_human_or_worm',
    'human_enhancers_cohn',
    'human_enhancers_ensembl',
    'human_ensembl_regulatory',
    'human_nontata_promoters',
    'human_ocr_ensembl',
]

def main():
    for ds in datasets:
        # Check if the dataset is already downloaded locally.
        if not is_downloaded(ds, cache_path=str(local_dest_path)):
            print(f"Downloading dataset '{ds}' to {local_dest_path}...")
            download_dataset(ds, version=0, dest_path=str(local_dest_path))
        else:
            print(f"Dataset '{ds}' is already downloaded.")

if __name__ == '__main__':
    main()