import copy
import os
import random
import time
from functools import partial, wraps
from typing import Callable, List, Sequence

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only, rank_zero_warn
from pytorch_lightning.strategies.ddp import DDPStrategy
from tqdm.auto import tqdm
from pytorch_lightning.strategies.ddp import DDPStrategy

import src.models.nn.utils as U
import src.utils as utils
import src.utils.train
from src.dataloaders.datasets.species_dataset import SpeciesDataset
from src.tasks import decoders, encoders, tasks
from src.utils import registry
from src.utils.optim_groups import add_optimizer_hooks
import sys
log = src.utils.train.get_logger(__name__)
sys.setrecursionlimit(6000)

import torch
print("CUDA available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())

# Turn on TensorFloat32 (speeds up large model training substantially)
import torch.backends
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

OmegaConf.register_new_resolver('eval', eval)
OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)

# Dummy WandB setup to handle retries
class DummyExperiment:
    """Dummy experiment."""
    def nop(self, *args, **kw):
        pass

    def __getattr__(self, _):
        return self.nop

    def __getitem__(self, idx) -> "DummyExperiment":
        return self

    def __setitem__(self, *args, **kwargs) -> None:
        pass

def rank_zero_experiment(fn: Callable) -> Callable:
    """Returns the real experiment on rank 0 and otherwise the DummyExperiment."""
    @wraps(fn)
    def experiment(self):
        @rank_zero_only
        def get_experiment():
            return fn(self)
        return get_experiment() or DummyExperiment()
    return experiment

class CustomWandbLogger(WandbLogger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    @rank_zero_experiment
    def experiment(self):
        if self._experiment is None:
            if self._offline:
                os.environ["WANDB_MODE"] = "dryrun"

            attach_id = getattr(self, "_attach_id", None)
            if wandb.run is not None:
                rank_zero_warn("There is a wandb run already in progress.")
                self._experiment = wandb.run
            elif attach_id is not None and hasattr(wandb, "_attach"):
                self._experiment = wandb._attach(attach_id)
            else:
                while True:
                    try:
                        self._experiment = wandb.init(**self._wandb_init)
                        break
                    except Exception as e:
                        print("wandb Exception:\n", e)
                        t = random.randint(30, 60)
                        print(f"Sleeping for {t} seconds")
                        time.sleep(t)

                if getattr(self._experiment, "define_metric", None):
                    self._experiment.define_metric("trainer/global_step")
                    self._experiment.define_metric("*", step_metric="trainer/global_step", step_sync=True)

        return self._experiment

class SequenceLightningModule(pl.LightningModule):
    def __init__(self, config):
        try:
            torch._C._jit_set_profiling_executor(False)
            torch._C._jit_set_profiling_mode(False)
        except AttributeError:
            pass

        super().__init__()
        self.save_hyperparameters(config, logger=False)

        # Dataset arguments
        self.dataset = SpeciesDataset(
            species=config.dataset.species,
            species_dir=config.dataset.species_dir,
            split=config.dataset.split,
            max_length=config.dataset.max_length,
            total_size=config.dataset.total_size,
            tokenizer_name=config.dataset.tokenizer_name,
            pad_max_length=config.dataset.pad_max_length,
            add_eos=config.dataset.add_eos,
            rc_aug=config.dataset.rc_aug,
            replace_N_token=config.dataset.replace_N_token,
            task=config.task._name_,
        )

        #self._check_config()

        self._has_setup = False
        self.setup()

    def setup(self, stage=None):
        if not self.hparams.train.disable_dataset:
            #self.dataset.setup()
            print('no setup')

        if self._has_setup:
            return
        else:
            self._has_setup = True

        encoder_cfg = utils.to_list(self.hparams.encoder) + utils.to_list(self.hparams.model.pop("encoder", None))
        decoder_cfg = utils.to_list(self.hparams.model.pop("decoder", None)) + utils.to_list(self.hparams.decoder)

        self.model = utils.instantiate(registry.model, self.hparams.model)
        self.task = utils.instantiate(tasks.registry, self.hparams.task, dataset=self.dataset, model=self.model)

        encoder = encoders.instantiate(encoder_cfg, dataset=self.dataset, model=self.model)
        decoder = decoders.instantiate(decoder_cfg, model=self.model, dataset=self.dataset)

        self.encoder = U.PassthroughSequential(self.task.encoder, encoder)
        self.decoder = U.PassthroughSequential(decoder, self.task.decoder)
        self.loss = self.task.loss
        self.loss_val = self.task.loss
        self.metrics = self.task.metrics
        self.train_torchmetrics = self.task.train_torchmetrics
        self.val_torchmetrics = self.task.val_torchmetrics
        self.test_torchmetrics = self.task.test_torchmetrics

    def forward(self, batch):
        return self.task.forward(batch, self.encoder, self.model, self.decoder, self._state)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, batch_size=self.hparams.dataset.batch_size, shuffle=True, num_workers=self.hparams.dataset.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, batch_size=self.hparams.dataset.batch_size, shuffle=False, num_workers=self.hparams.dataset.num_workers)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, batch_size=self.hparams.dataset.batch_size, shuffle=False, num_workers=self.hparams.dataset.num_workers)

### pytorch-lightning utils and entrypoint ###

def create_trainer(config, **kwargs):
    callbacks: List[pl.Callback] = []
    logger = None

    if config.get("wandb") is not None:
        logger = CustomWandbLogger(
            config=utils.to_dict(config, recursive=True),
            settings=wandb.Settings(start_method="fork"),
            **config.wandb,
        )

    if "callbacks" in config:
        for _name_, callback in config.callbacks.items():
            log.info(f"Instantiating callback <{registry.callbacks[_name_]}>")
            callback._name_ = _name_
            callbacks.append(utils.instantiate(registry.callbacks, callback))

    n_devices = config.trainer.get('devices', 1)
    if isinstance(n_devices, Sequence):
        n_devices = len(n_devices)
    if n_devices > 1 and config.trainer.get('strategy', None) is None:
        config.trainer.strategy = dict(
            _target_='pytorch_lightning.strategies.DDPStrategy',
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )

    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer = hydra.utils.instantiate(config.trainer, callbacks=callbacks, logger=logger)    

    return trainer

def train(config):
    if config.train.seed is not None:
        pl.seed_everything(config.train.seed, workers=True)
    trainer = create_trainer(config)
    model = SequenceLightningModule(config)

    if config.train.get("pretrained_model_path", None) is not None:
        model = SequenceLightningModule.load_from_checkpoint(
            config.train.pretrained_model_path,
            config=config,
            strict=config.train.pretrained_model_strict_load,
        )

    if config.train.validate_at_start:
        trainer.validate(model)

    if config.train.ckpt is not None:
        trainer.fit(model, ckpt_path=config.train.ckpt)
    else:
        trainer.fit(model)
    if config.train.test:
        trainer.test(model)

@hydra.main(config_path="configs", config_name="config.yaml")
def main(config: OmegaConf):
    config = utils.train.process_config(config)
    utils.train.print_config(config, resolve=True)
    train(config)

if __name__ == "__main__":
    main()