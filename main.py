import os

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import wandb


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    from experiment import experiment
    best_nor_ret = experiment(cfg)
    wandb.finish()
    return best_nor_ret

if __name__ == "__main__":
    main() 