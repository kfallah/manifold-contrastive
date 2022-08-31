"""
Experiment wrapper that loads and validates config, and then connects to
experiment tracker.

@Filename    experiment.py
@Author      Kion
@Created     08/31/22
"""

import logging
from dataclasses import MISSING, dataclass
from typing import Tuple

import hydra
import wandb
from hydra.core.config_store import ConfigStore

import model.contrastive.config as header_config
from dataloader.config import DataLoaderConfig
from dataloader.contrastive_dataloader import get_dataloader
from eval.config import EvaluatorConfig
from eval.evaluator import Evaluator
from model.config import ModelConfig
from model.model import Model
from train.config import TrainerConfig
from train.trainer import Trainer


@dataclass
class ExperimentConfig:
    # Hierarchical configurations used for experiment
    train_dataloader_cfg: DataLoaderConfig = MISSING
    eval_dataloader_cfg: DataLoaderConfig = MISSING
    model_cfg: ModelConfig = MISSING
    trainer_cfg: TrainerConfig = MISSING
    eval_cfg: EvaluatorConfig = MISSING

    # Experimental settings
    exp_name: str = MISSING
    exp_dir: str = MISSING
    devices: Tuple[int] = MISSING


def register_configs() -> None:
    cs.store(name="exp_cfg", node=ExperimentConfig)
    header_config.register_configs()


log = logging.getLogger(__name__)
cs = ConfigStore.instance()
register_configs()


def run_experiment(
    cfg: ExperimentConfig,
    trainer: Trainer,
    evaluator: Evaluator,
    train_dataloader: torch.utils.data.DataLoader,
    eval_dataloader: torch.utils.data.DataLoader,
) -> None:
    log.info("Running experiment...")
    for epoch in range(cfg.trainer_cfg.num_epochs):
        train_metadata = trainer.train_epoch(epoch, train_dataloader)

        if epoch % cfg.eval_cfg.eval_frequency == 0:
            eval_metadata = evaluator.run_eval(epoch, eval_dataloader)


@hydra.main(version_base=None, config_path="../config", config_name="base")
def initialize_experiment(cfg: ExperimentConfig) -> None:
    log.info("Initializing experiment...")
    # wandb.config = OmegaConf.to_container(
    #     cfg, resolve=True, throw_on_missing=True
    # )
    # wandb.init(project="manifold-contrastive", entity="kfallah",
    #            settings=wandb.Settings(start_method="thread"))

    # Initialize train_loader and eval_loader
    log.info("Initializing dataloaders for " + cfg.train_dataloader_cfg.dataset_cfg.dataset_name + " dataset...")
    train_dataset, train_dataloder = get_dataloader(cfg.train_dataloader_cfg)
    eval_dataset, eval_dataloader = get_dataloader(cfg.eval_dataloader_cfg)

    # Initialize the model
    log.info(
        f"Initializing model with {cfg.model_cfg.backbone_cfg.hub_model_name} backbone"
        + f"and {cfg.model_cfg.header_cfg.header_name} header..."
    )
    model = Model.initialize_model(cfg.model_cfg)

    # Initialize the trainer and evaluator
    trainer = Trainer.initialize_trainer(cfg, model)
    evaluator = Evaluator.initialize_evaluator(cfg, model)

    # Run experiment
    log.info("...Starting experiment.")
    run_experiment(cfg, trainer, evaluator)


if __name__ == "__main__":
    initialize_experiment()
