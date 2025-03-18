from pathlib import Path
from pyfaidx import Fasta
import pandas as pd
import torch
from random import randrange, random, choice
import numpy as np

# Helper functions

def exists(val):
    return val is not None

def coin_flip():
    return random() > 0.5

# Augmentations

string_complement_map = {
    'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A',
    'a': 't', 'c': 'g', 'g': 'c', 't': 'a'
}

def string_reverse_complement(seq):
    rev_comp = ''
    for base in seq[::-1]:
        rev_comp += string_complement_map.get(base, base)
    return rev_comp

class FastaInterval():
    # Same as your original FastaInterval class
    # No changes needed here

    def __init__(
        self,
        *,
        fasta_file,
        return_seq_indices=False,
        shift_augs=None,
        rc_aug=False,
        pad_interval=False,
    ):
        fasta_file = Path(fasta_file)
        assert fasta_file.exists(), 'path to fasta file must exist'

        self.seqs = Fasta(str(fasta_file))
        self.return_seq_indices = return_seq_indices
        self.shift_augs = shift_augs
        self.rc_aug = rc_aug
        self.pad_interval = pad_interval

        # Calculate lengths of each chromosome/scaffold
        self.chr_lens = {chr_name: len(self.seqs[chr_name]) for chr_name in self.seqs.keys()}

    def __call__(self, chr_name, start, end, max_length):
        # Same as your original __call__ method
        # No changes needed here

        interval_length = end - start
        chromosome = self.seqs[chr_name]
        chromosome_length = self.chr_lens[chr_name]

        if exists(self.shift_augs):
            min_shift, max_shift = self.shift_augs
            max_shift += 1

            min_shift = max(start + min_shift, 0) - start
            max_shift = min(end + max_shift, chromosome_length) - end

            rand_shift = randrange(min_shift, max_shift)
            start += rand_shift
            end += rand_shift

        left_padding = right_padding = 0

        if interval_length < max_length:
            extra_seq = max_length - interval_length
            extra_left_seq = extra_seq // 2
            extra_right_seq = extra_seq - extra_left_seq
            start -= extra_left_seq
            end += extra_right_seq

        if start < 0:
            left_padding = -start
            start = 0

        if end > chromosome_length:
            right_padding = end - chromosome_length
            end = chromosome_length

        if interval_length > max_length:
            end = start + max_length

        seq = str(chromosome[start:end])

        if self.rc_aug and coin_flip():
            seq = string_reverse_complement(seq)

        if self.pad_interval:
            seq = ('.' * left_padding) + seq + ('.' * right_padding)

        return seq
from src.utils import registry
class SequenceDataset:
    registry = {}

    @classmethod
    def register(cls, name):
        def inner_wrapper(wrapped_class):
            cls.registry[name] = wrapped_class
            return wrapped_class
        return inner_wrapper

@SequenceDataset.register('multispecies')
class MultispeciesDataset(torch.utils.data.Dataset):
    '''
    Dataset class for multiple species.
    '''

    def __init__(
        self,
        split,
        species,
        bed_files,
        fasta_files,
        max_length,
        pad_max_length=None,
        tokenizer=None,
        tokenizer_name=None,
        add_eos=False,
        return_seq_indices=False,
        shift_augs=None,
        rc_aug=False,
        return_augs=False,
        replace_N_token=False,
        pad_interval=False,
    ):
        self.max_length = max_length
        self.pad_max_length = pad_max_length if pad_max_length is not None else max_length
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer
        self.return_augs = return_augs
        self.add_eos = add_eos
        self.replace_N_token = replace_N_token
        self.pad_interval = pad_interval

        # Load FASTA files for all species
        self.fastas = {}
        
        for species_name in species:
            fasta_file = Path(fasta_files[species_name])
            assert fasta_file.exists(), f'FASTA file for {species_name} must exist at {fasta_file}'
            self.fastas[species_name] = FastaInterval(
                fasta_file=fasta_file,
                return_seq_indices=return_seq_indices,
                shift_augs=shift_augs,
                rc_aug=rc_aug,
                pad_interval=pad_interval,
            )

        # Load and combine BED files
        dfs = []
        for species_name in species:
            bed_file = Path(bed_files[species_name])
            assert bed_file.exists(), f'BED file for {species_name} must exist at {bed_file}'
            df = pd.read_csv(
                str(bed_file),
                sep='\t',
                names=['chr_name', 'start', 'end', 'split']
            )
            df = df[df['split'] == split]
            df['species'] = species_name  # Add species column
            dfs.append(df)

        # Combine all dataframes
        self.df = pd.concat(dfs, ignore_index=True)
        self.df.reset_index(drop=True, inplace=True)
        
    def __len__(self):
        #print(f"Length of dataset: {len(self.df)}") 
        return len(self.df)

    def replace_value(self, x, old_value, new_value):
        return torch.where(x == old_value, new_value, x)

    def __getitem__(self, idx):
        """Returns a sequence of specified len"""
        row = self.df.iloc[idx]
        species = row['species']
        chr_name = row['chr_name']
        start = row['start']
        end = row['end']
        
        # Retrieve sequence using the appropriate FASTA file
        seq = self.fastas[species](
            chr_name, start, end, max_length=self.max_length
        )
        #print(f"Tokenized sequence: {seq}")
        if seq is None or len(seq) == 0:
            raise ValueError(f"Failed to retrieve sequence for species: {species}, chr: {chr_name}, start: {start}, end: {end}")
        # Tokenization
        if self.tokenizer_name == 'char':
            seq = self.tokenizer(
                seq,
                add_special_tokens=True if self.add_eos else False,
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
            )
            seq = seq["input_ids"]  # get input_ids

        elif self.tokenizer_name == 'bpe':
            seq = self.tokenizer(
                seq,
                padding="max_length",
                max_length=self.pad_max_length,
                truncation=True,
            )
            # get input_ids
            if self.add_eos:
                seq = seq["input_ids"][1:]  # remove the bos, keep the eos token
            else:
                seq = seq["input_ids"][1:-1]  # remove both special tokens
        if seq is None or len(seq) == 0:
            raise ValueError("Tokenization failed, resulting in an empty sequence.")

        #print(f"Tokenized sequence: {seq}", flush=True)  # Add debug statement

        # Convert to tensor
        seq = torch.LongTensor(seq)

        if self.replace_N_token:
            # Replace 'N' token with pad token
            n_token_id = self.tokenizer._vocab_str_to_int.get('N', None)
            pad_token_id = self.tokenizer.pad_token_id
            if n_token_id is not None:
                seq = self.replace_value(seq, n_token_id, pad_token_id)

        data = seq[:-1].clone()  # Remove last token
        target = seq[1:].clone()  # Offset by 1

        return data, target