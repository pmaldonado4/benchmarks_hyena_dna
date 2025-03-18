from pathlib import Path
from pyfaidx import Fasta
import torch
from random import randrange, random
import numpy as np

"""

Dataset for sampling arbitrary intervals from the human genome.

"""

# helper functions
def exists(val):
    return val is not None

def coin_flip():
    return random() > 0.5

# augmentations
string_complement_map = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'a': 't', 'c': 'g', 'g': 'c', 't': 'a'}

def string_reverse_complement(seq):
    rev_comp = ''
    for base in seq[::-1]:
        if base in string_complement_map:
            rev_comp += string_complement_map[base]
        else:
            rev_comp += base
    return rev_comp


class FastaInterval():
    def __init__(
        self,
        fasta_file,
        rc_aug=False,
        pad_interval=False,
    ):
        fasta_file = Path(fasta_file)
        assert fasta_file.exists(), 'Path to fasta file must exist'

        self.seqs = Fasta(str(fasta_file))
        self.rc_aug = rc_aug
        self.pad_interval = pad_interval

        # Store lengths of each chromosome
        self.chr_lens = {chr_name: len(self.seqs[chr_name]) for chr_name in self.seqs.keys()}

    def __call__(self, chr_name, start, end, max_length):
        interval_length = end - start
        chromosome = self.seqs[chr_name]
        chromosome_length = self.chr_lens[chr_name]

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


class HG38Dataset(torch.utils.data.Dataset):
    '''
    Loop through chromosomes, retrieve random intervals, query fasta file for sequence.
    '''

    def __init__(
        self,
        split,
        fasta_file,
        max_length,
        tokenizer=None,
        tokenizer_name=None,
        add_eos=False,
        rc_aug=False,
        return_augs=False,
        replace_N_token=False,
        pad_interval=False,
    ):
        self.max_length = max_length
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer
        self.return_augs = return_augs
        self.add_eos = add_eos
        self.replace_N_token = replace_N_token
        self.pad_interval = pad_interval

        self.fasta = FastaInterval(
            fasta_file=fasta_file,
            rc_aug=rc_aug,
            pad_interval=pad_interval,
        )

        # Define training, validation, and test splits
        self.chr_splits = {
            'train': [f'chr{i}' for i in range(1, 23)] + ['chrX'],
            'valid': ['chr1', 'chr2'],
            'test': ['chr3', 'chr4'],
        }
        self.chromosomes = self.chr_splits[split]

    def __len__(self):
        return len(self.chromosomes) * 1000  # Arbitrary large number of samples

    def replace_value(self, x, old_value, new_value):
        return torch.where(x == old_value, new_value, x)

    def __getitem__(self, idx):
        # Randomly select a chromosome and an interval within it
        chr_name = random.choice(self.chromosomes)
        chromosome_length = self.fasta.chr_lens[chr_name]

        start = randrange(0, chromosome_length - self.max_length)
        end = start + self.max_length

        seq = self.fasta(chr_name, start, end, max_length=self.max_length)

        # Tokenize the sequence
        if self.tokenizer_name == 'char':
            seq = self.tokenizer(seq, add_special_tokens=True if self.add_eos else False, padding="max_length", max_length=self.max_length, truncation=True)
            seq = seq["input_ids"]

        # Convert to tensor
        seq = torch.LongTensor(seq)

        if self.replace_N_token:
            seq = self.replace_value(seq, self.tokenizer._vocab_str_to_int['N'], self.tokenizer.pad_token_id)

        data = seq[:-1].clone()  # remove eos
        target = seq[1:].clone()  # offset by 1, includes eos

        return data, target