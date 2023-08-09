import isaacgym

import hydra
from omegaconf import DictConfig, OmegaConf
from common.util import omegaconf_to_dict, print_dict, fix_wandb, update_dict

from compose import CompositionAgent
from sac import SACAgent

import torch
import numpy as np

import wandb


@hydra.main(config_name="config", config_path="./cfg")
def launch_rlg_hydra(cfg: DictConfig):
    cfg_dict = omegaconf_to_dict(cfg)

    wandb.init()
    wandb_dict = fix_wandb(wandb.config)

    print_dict(wandb_dict)
    update_dict(cfg_dict, wandb_dict)

    cfg_dict["buffer"]["n_env"] = cfg_dict["env"]["num_envs"]
    print_dict(cfg_dict)

    torch.manual_seed(cfg_dict["seed"])
    np.random.seed(cfg_dict["seed"])

    if "sac" in cfg_dict["agent"]["name"].lower():
        agent = SACAgent(cfg=cfg_dict)
    else:
        agent = CompositionAgent(cfg_dict)

    agent.run()
    wandb.finish()


if __name__ == "__main__":
    launch_rlg_hydra()
