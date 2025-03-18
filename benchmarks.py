# Required Imports
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from genomic_benchmarks.loc2seq import download_dataset
from genomic_benchmarks.data_check import is_downloaded
from tokenizer import CharacterTokenizer
# Helper Functions
def exists(val):
    return val is not None

def coin_flip():
    return random() > 0.5

string_complement_map = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'a': 't', 'c': 'g', 'g': 'c', 't': 'a'}
def string_reverse_complement(seq):
    rev_comp = ''
    for base in seq[::-1]:
        if base in string_complement_map:
            rev_comp += string_complement_map[base]
        else:
            rev_comp += base
    return rev_comp

# Dataset Class
class GenomicBenchmarkDataset(torch.utils.data.Dataset):
    def __init__(self, split, max_length, dataset_name, tokenizer, rc_aug=False, use_padding=True, add_eos=False):
        self.max_length = max_length
        self.use_padding = use_padding
        self.tokenizer = tokenizer
        self.rc_aug = rc_aug
        self.add_eos = add_eos

        # Download dataset if not available
        if not is_downloaded(dataset_name, cache_path="./datasets"):
            print(f"Downloading {dataset_name}...")
            download_dataset(dataset_name, version=0, dest_path="./datasets")
        else:
            print(f"Dataset {dataset_name} is already downloaded.")

        # Load data paths and labels
        base_path = Path("./datasets") / dataset_name / split
        self.all_paths = []
        self.all_labels = []
        label_mapper = {x.stem: i for i, x in enumerate(base_path.iterdir())}
        for label_type in label_mapper.keys():
            for x in (base_path / label_type).iterdir():
                self.all_paths.append(x)
                self.all_labels.append(label_mapper[label_type])

    def __len__(self):
        return len(self.all_paths)

    def __getitem__(self, idx):
        txt_path = self.all_paths[idx]
        with open(txt_path, "r") as f:
            content = f.read()
        x = content
        y = self.all_labels[idx]

        # Apply reverse complement augmentation
        if self.rc_aug and coin_flip():
            x = string_reverse_complement(x)

        seq = self.tokenizer(x, add_special_tokens=False, padding="max_length" if self.use_padding else None, 
                             max_length=self.max_length, truncation=True)["input_ids"]

        # Add end-of-sequence token
        if self.add_eos:
            seq.append(self.tokenizer.sep_token_id)

        seq = torch.LongTensor(seq)  # Convert to tensor
        target = torch.LongTensor([y])
        return seq, target


# Train Function
def train(model, device, train_loader, optimizer, epoch, loss_fn, log_interval=10):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target.squeeze())
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                  f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")


# Test Function
def test(model, device, test_loader, loss_fn):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target.squeeze()).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} "
          f"({accuracy:.2f}%)\n")
    return accuracy


# Main Function to Perform Benchmarking
def run_benchmark(task_name):
    # Task-specific settings
    tasks = {
        "human_enhancers": {
            "dataset_name": "human_enhancers_cohn",
            "max_length": 500,
            "num_classes": 2,
        },
        "worm_vs_human": {
            "dataset_name": "demo_human_or_worm",
            "max_length": 200,
            "num_classes": 2,
        },
    }

    if task_name not in tasks:
        raise ValueError(f"Task {task_name} not recognized. Available tasks: {list(tasks.keys())}")

    task_config = tasks[task_name]

    # Training settings
    num_epochs = 10
    batch_size = 128
    learning_rate = 6e-4
    weight_decay = 0.1
    rc_aug = True
    add_eos = False

    # Initialize tokenizer
    tokenizer = CharacterTokenizer(
        characters=["A", "C", "G", "T", "N"],
        model_max_length=task_config["max_length"] + 2,
        add_special_tokens=False,
        padding_side="left",
    )

    # Load datasets
    train_dataset = GenomicBenchmarkDataset(
        split="train",
        max_length=task_config["max_length"],
        dataset_name=task_config["dataset_name"],
        tokenizer=tokenizer,
        rc_aug=rc_aug,
        add_eos=add_eos,
    )

    test_dataset = GenomicBenchmarkDataset(
        split="test",
        max_length=task_config["max_length"],
        dataset_name=task_config["dataset_name"],
        tokenizer=tokenizer,
        rc_aug=rc_aug,
        add_eos=add_eos,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = HyenaDNAPreTrainedModel.from_pretrained(
        "./checkpoints",
        "hyenadna-tiny-1k-seqlen",
        download=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_head=True,
        n_classes=task_config["num_classes"],
    )

    # Optimizer and Loss Function
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    # Training Loop
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, optimizer, epoch, loss_fn)
        test(model, device, test_loader, loss_fn)


# Run Benchmarks
run_benchmark("human_enhancers")  # For Human Enhancers
# run_benchmark("worm_vs_human")  # Uncomment for Worm vs. Human