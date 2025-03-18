# Initialize your tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
from src.dataloaders.datasets.species_dataset import SpeciesDataset  # TODO make registry
from torch.utils.data import DataLoader

# Instantiate the dataset
dataset = SpeciesDataset(
    split='train',
    species_list=['human', 'mouse', 'lemur', 'squirrel'],
    species_bed_files={
        'human': "/u/pmaldonadocatala/hyena-dna/processed/human/human_genome_intervals.bed",
        'mouse': "/u/pmaldonadocatala/hyena-dna/processed/mouse/mouse_genome_intervals.bed",
        'lemur': "/u/pmaldonadocatala/hyena-dna/processed/lemur/lemur_genome_intervals.bed",
        'squirrel': "/u/pmaldonadocatala/hyena-dna/processed/squirrel/squirrel_genome_intervals.bed"
    },
    species_fasta_files={
        'human': "/u/pmaldonadocatala/hyena-dna/processed/human/human_genome.fna",
        'mouse': "/u/pmaldonadocatala/hyena-dna/processed/mouse/mouse_genome.fna",
        'lemur': "/u/pmaldonadocatala/hyena-dna/processed/lemur/lemur_genome.fna",
        'squirrel': "/u/pmaldonadocatala/hyena-dna/processed/squirrel/squirrel_genome.fna"
    },
    max_length=1000,
    tokenizer=tokenizer,
    tokenizer_name='char',
    add_eos=True,
    rc_aug=False,
    pad_interval=False,
    replace_N_token=False,
)
# Verify dataset length
print(f"Dataset length: {len(dataset)}")

# Instantiate the DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=12
)

# Iterate over a few batches
for i, (data, target) in enumerate(dataloader):
    print(f"Batch {i}: data shape {data.shape}, target shape {target.shape}")
    if i >= 2:  # Test the first few batches
        break