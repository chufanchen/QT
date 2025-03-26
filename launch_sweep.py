import os
import hydra
from omegaconf import DictConfig
import wandb
from hydra.core.hydra_config import HydraConfig

@hydra.main(config_path="configs", config_name="sweep/halfcheetah_sweep", version_base="1.3")
def launch_sweep(cfg: DictConfig):
    from experiment import experiment
    best_nor_ret = experiment(cfg)
    wandb.finish()
    return best_nor_ret

if __name__ == "__main__":
    launch_sweep() 