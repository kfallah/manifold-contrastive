"""
Experiment wrapper that loads and validates config, and then connects to
experiment tracker.

@Filename    experiment.py
@Author      Kion
@Created     08/31/22
"""

import logging
import os
import warnings
from dataclasses import MISSING, dataclass
from typing import Tuple

import hydra
import numpy as np
import torch
import wandb
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from dataloader.base import Dataset
from dataloader.config import DataLoaderConfig
from dataloader.ssl_dataloader import get_dataset
from eval.config import EvaluatorConfig
from eval.evaluator import Evaluator
from model.config import ModelConfig
from model.model import Model
from train.config import TrainerConfig
from train.trainer import Trainer

warnings.filterwarnings("ignore")


@dataclass
class ExperimentConfig:
    # Hierarchical configurations used for experiment
    dataloader_cfg: DataLoaderConfig = MISSING
    model_cfg: ModelConfig = MISSING
    trainer_cfg: TrainerConfig = MISSING
    evaluator_cfg: EvaluatorConfig = MISSING

    # Experimental settings
    exp_name: str = MISSING
    exp_dir: str = MISSING
    devices: Tuple[int] = MISSING
    enable_wandb: bool = True
    seed: int = 0


def register_configs() -> None:
    cs.store(name="exp_cfg", node=ExperimentConfig)


log = logging.getLogger(__name__)
cs = ConfigStore.instance()
register_configs()


def run_eval(
    epoch: int,
    current_best: float,
    trainer: Trainer,
    evaluator: Evaluator,
    dataset: Dataset,
    last_epoch: bool = False,
):
    # Run evaluation metrics on the model
    eval_out = evaluator.run_eval(epoch, dataset, last_epoch)
    # Only not equal to None when an evaluation was run
    if eval_out is not None:
        save_metric, eval_metadata = eval_out
        format_eval = [
            f"{key}: {eval_metadata[key]:.3E}" for key in eval_metadata.keys() if isinstance(eval_metadata[key], float)
        ]
        log.info(f"[Evaluation epoch {epoch}]: " + ", ".join(format_eval))
        # If the current model has a better key metric than previous models, save its weights.
        if current_best > save_metric:
            trainer.save_model(epoch, os.getcwd() + "/checkpoints/", save_best=True)
            current_best = save_metric
        if wandb.run is not None:
            wandb.log(eval_metadata)


def run_experiment(
    cfg: ExperimentConfig,
    trainer: Trainer,
    evaluator: Evaluator,
    dataset: Dataset,
) -> None:
    log.info("Running experiment...")
    # Used to save the
    current_best = 1e99
    for epoch in range(cfg.trainer_cfg.num_epochs):
        # Run eval metrics on the model
        if epoch > 0:
            run_eval(epoch, current_best, trainer, evaluator, dataset)

        # Perform a training epoch
        _ = trainer.train_epoch(epoch, dataset.train_dataloader)
        # Save the model every few epochs
        if epoch % cfg.trainer_cfg.save_interval == 0:
            trainer.save_model(epoch, os.getcwd() + "/checkpoints/")

    # Save the model and force all evals after training is complete
    run_eval(
        epoch,
        current_best,
        trainer,
        evaluator,
        dataset,
        last_epoch=True,
    )
    trainer.save_model(epoch, os.getcwd() + "/checkpoints/")
    log.info("...Experiment complete!")


@hydra.main(version_base=None, config_path="../config", config_name="simclr_cifar10")
def initialize_experiment(cfg: DictConfig) -> None:
    wandb.init(
        project="manifold-contrastive",
        mode="online" if cfg.enable_wandb else "disabled",
        settings=wandb.Settings(start_method="thread"),
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
    )
    log.info(f"Initializing experiment {wandb.run.name}...")

    assert len(cfg.devices) > 0
    default_device = torch.device(cfg.devices[0])

    # Initialize train_loader and eval_loader
    log.info("Initializing dataloaders for " + cfg.dataloader_cfg.dataset_cfg.dataset_name + " dataset...")
    dataset = get_dataset(cfg.dataloader_cfg)

    # Set random seeds
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Initialize the model
    log.info(f"Initializing model with {cfg.model_cfg.backbone_cfg.hub_model_name} backbone...")
    model = Model.initialize_model(cfg.model_cfg, cfg.dataloader_cfg.dataset_cfg.dataset_name, cfg.devices)

    # Initialize the trainer and evaluator
    trainer = Trainer.initialize_trainer(
        cfg.trainer_cfg, model, cfg.trainer_cfg.num_epochs * len(dataset.train_dataloader), default_device
    )
    evaluator = Evaluator.initialize_evaluator(cfg.evaluator_cfg, trainer.get_model(), default_device)

    # Run experiment
    run_experiment(cfg, trainer, evaluator, dataset)


if __name__ == "__main__":
    initialize_experiment()
