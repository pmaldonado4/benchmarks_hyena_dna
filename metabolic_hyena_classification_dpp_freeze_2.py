import os

def is_main_process():
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True

# Remove the SLURM environment variable handling since we're using torchrun
# torchrun will set RANK, LOCAL_RANK, and WORLD_SIZE automatically

def setup_distributed():
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group(backend='nccl')
        print(f"Initialized process {dist.get_rank()} / {dist.get_world_size()}")

def run_metabolic_training():
    # Set hyperparameters first
    num_epochs = 50
    warmup_epochs = 5
    max_length = 4400
    batch_size = 64
    learning_rate = 6e-4
    weight_decay = 0.1
    rc_aug = True
    use_padding = True
    add_eos = False
    pretrained_model_name = 'hyenadna-metabolic-32k'
    use_head = True
    n_classes = 4

    # Initialize distributed setup first
    setup_distributed()
    
    if dist.is_initialized():
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f'cuda:{local_rank}')
        print(f"Process {dist.get_rank()} using device: {device}")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Single process using device: {device}")

    # Initialize wandb only on rank 0
    if is_main_process():
        print("Initializing wandb on main process")
        wandb.init(
            project="hyena-dna-metabolic-metahyena-classification",
            name="HyenaDNA-Training",
            config={
                "num_epochs": num_epochs,
                "max_length": max_length,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "rc_aug": rc_aug,
                "use_padding": use_padding,
                "add_eos": add_eos,
                "warmup_epochs": warmup_epochs,
                "device": str(device)
            })

    if pretrained_model_name:
        model = HyenaDNAPreTrainedModel.from_pretrained(
            '/projects/bdhi/benchmarks_hyena_dna/checkpoints',
            pretrained_model_name,
            download=False,
            config=None,
            device=device,
            use_head=use_head,
            n_classes=n_classes,
        )
    else:
        model = HyenaDNAModel(use_head=use_head, n_classes=n_classes)
    tokenizer = CharacterTokenizer(
        characters=['A', 'C', 'G', 'T', 'N'],
        model_max_length=max_length + 2,
        padding_side='left'
    )

    dataset = MetabolicDataset(df=df, max_length=max_length, tokenizer=tokenizer,
                               use_padding=use_padding, rc_aug=rc_aug, add_eos=add_eos)
    torch.manual_seed(seed)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # Use DistributedSampler if in distributed mode
    if dist.is_initialized():
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
        print(f"Process {dist.get_rank()} using device {torch.cuda.current_device()}")
    else:
        train_sampler = None
        test_sampler = None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
                              sampler=train_sampler, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, sampler=test_sampler, drop_last=False)

    class_weights = calculate_class_weights(df).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights) # Calculate loss function taking class imbalance into account
    model.to(device)

    # Freeze early layers (check model config file for # of layers or blocks present)
    freeze_early_layers(model, freeze_up_to=2)

    # Wrap model in DDP if distributed is initialized
    if dist.is_initialized():
        model = DDP(model, device_ids=[device.index] if device.type == 'cuda' else None)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Calculate total training steps for scheduler: epochs * steps per epoch
    total_steps = num_epochs * len(train_loader)
    warmup_steps = warmup_epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                  num_training_steps=total_steps)
    scaler = GradScaler()

    wandb.watch(model, log="all", log_freq=10)

    # CSV logging 
    csv_file = 'metrics_log.csv'
    best_f1 = 0
    best_epoch = -1

    for epoch in range(num_epochs):
        # If using DistributedSampler, set epoch for reproducibility
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train(model, device, train_loader, optimizer, epoch, loss_fn, scaler)
        test_loss, test_accuracy, f1, mcc, conf_matrix = test(model, device, test_loader, loss_fn)
        scheduler.step()

        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch
            if not dist.is_initialized() or dist.get_rank() == 0:
                best_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
                best_model_path = 'best_model.pth'
                torch.save(best_state, best_model_path)

                # create artifact and save the model to WANDB
                artifact = wandb.Artifact('best_model', type = 'model')
                artifact.add_file(best_model_path)
                wandb.log_artifact(artifact)
                print(f"Saved best model at epoch {epoch} with F1: {best_f1:.4f}")

        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, None, None, test_loss, test_accuracy, f1, mcc, str(conf_matrix)])
        print(f"Epoch {epoch} - Best F1 so far: {best_f1:.4f} (Epoch {best_epoch})")

    torch.save(model.state_dict(), "hyena_dna_metabolic_32k.pth")
    wandb.save("hyena_dna_metabolic_32k.pth")
    
    if dist.is_initialized():
        cleanup_distributed()
    
    wandb.finish()

csv_file = 'metrics_log.csv'
if not os.path.exists(csv_file):
    with open(csv_file, 'w', newline='') as f:
         writer = csv.writer(f)
         writer.writerow(["epoch", "train_loss", "train_accuracy", "test_loss", "test_accuracy", "f1_score", "mcc", "confusion_matrix"])

run_metabolic_training()