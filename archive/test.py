import torch
from transformers import AutoTokenizer
from dataloaders.datasets.multispecies_dataset import SpeciesDataset  # TODO make registry
def test_multispecies_dataset():
    # Define species and directory paths
    species = ['human', 'mouse', 'lemur', 'squirrel']
    species_dir = "/u/pmaldonadocatala/hyena-dna/processed"  # Ensure this directory contains subfolders for each species with their fasta files
    split = 'train'  # Use 'train', 'valid', or 'test'
    
    # Load a tokenizer (you can use any tokenizer; for simplicity, I'll use a BPE tokenizer here)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    
    # Define dataset parameters
    max_length = 100  # Define max length for sequences
    add_eos = True  # Whether to add an end-of-sequence token
    rc_aug = True  # Enable reverse complement augmentation
    replace_N_token = True  # Replace 'N' tokens if needed
    
    # Create the dataset instance
    dataset = SpeciesDataset(
        species=species,
        species_dir=species_dir,
        split=split,
        max_length=max_length,
        tokenizer=tokenizer,
        tokenizer_name='char',  # You can change to 'char' if using character-level tokenization
        add_eos=add_eos,
        rc_aug=rc_aug,
        replace_N_token=replace_N_token
    )
    
    # Print some information about the dataset
    print(f"Dataset size: {len(dataset)} samples")
    
    # Sample some sequences from the dataset
    num_samples = 2  # Number of samples to display
    for i in range(num_samples):
        data, target = dataset[i]  # Get a sample
        #print(f"\nSample {i + 1}:")
        #print(f"Data (input sequence): {data}")
        #print(f"Target (shifted sequence): {target}")
    for i in range(2):  # Test a few samples
        data, target = dataset[i]  # Get a sample
        middle_index = len(data) // 2  # Find the middle index
        middle_part = data[middle_index - 50:middle_index + 250]  # Get 100 tokens from the middle
        
        #print(f"Sample {i}")
        #print(f"Data shape: {data.shape}, Target shape: {target.shape}")
        print(f"Middle part of tokenized sequence: {middle_part}")  # Print the middle part of the sequence
        print('-' * 80)
    # Test dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
    for batch_idx, (data, target) in enumerate(dataloader):
        print(f"\nBatch {batch_idx + 1} - Data shape: {data.shape}, Target shape: {target.shape}")
        if batch_idx == 1:  # Stop after showing a couple of batches
            break
    # Print a sample of data and target
    # Modify the __getitem__ method in the dataset to print the raw sequence
    
    for i in range(2):  # Checking first 5 samples
        data, target = dataset[i]
        print(f"Sample {i+1}")
        print(f"Data shape: {data.shape}, Target shape: {target.shape}")
        print(f"Data: {data}")  # Printing the tokenized sequence
        print(f"Target: {target}\n")  # Printing the target
    
if __name__ == "__main__":
    test_multispecies_dataset()

