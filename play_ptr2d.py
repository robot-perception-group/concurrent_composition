import isaacgym

import hydra
from omegaconf import DictConfig, OmegaConf
from common.util import omegaconf_to_dict, print_dict, fix_wandb, update_dict

from compose import CompositionAgent
from sac import SACAgent

import torch
import numpy as np

import wandb

from tkinter import *

model_path = "../logs/dacgpi/Pointer2D/2023-08-08-15-19-48/model100/"

@hydra.main(config_name="config", config_path="./cfg")
def launch_rlg_hydra(cfg: DictConfig):
    cfg_dict = omegaconf_to_dict(cfg)

    wandb.init(mode="disabled")
    wandb_dict = fix_wandb(wandb.config)

    print_dict(wandb_dict)
    update_dict(cfg_dict, wandb_dict)

    cfg_dict["buffer"]["n_env"] = cfg_dict["env"]["num_envs"]
    print_dict(cfg_dict)

    torch.manual_seed(456)
    np.random.seed(456)

    if "sac" in cfg_dict["agent"]["name"].lower():
        agent = SACAgent(cfg=cfg_dict)
    else:
        agent = CompositionAgent(cfg_dict)

    agent.load_torch_model(model_path)

    root = Tk()
    root.title("test")
    root.geometry("300x400")

    def update_pos(v):
        agent.task.w_eval_nav[0] = float(v)
        agent.task.w_eval_hov[0] = float(v)
        agent.task._w_eval_init = torch.tile(agent.task.w_eval_nav, (agent.task.n_env, 1))  # [N, F]
        agent.task.w_eval = agent.task._w_eval_init.clone()  # [N, F]

    def update_vel(v):
        agent.task.w_eval_nav[1] = float(v)
        agent.task.w_eval_hov[1] = float(v)
        agent.task._w_eval_init = torch.tile(agent.task.w_eval_nav, (agent.task.n_env, 1))  # [N, F]
        agent.task.w_eval = agent.task._w_eval_init.clone()  # [N, F]

    def update_ang(v):
        agent.task.w_eval_nav[2] = float(v)
        agent.task.w_eval_hov[2] = float(v)
        agent.task._w_eval_init = torch.tile(agent.task.w_eval_nav, (agent.task.n_env, 1))  # [N, F]
        agent.task.w_eval = agent.task._w_eval_init.clone()  # [N, F]

    def update_suc(v):
        agent.task.w_eval_nav[4] = float(v)
        agent.task.w_eval_hov[4] = float(v)
        agent.task._w_eval_init = torch.tile(agent.task.w_eval_nav, (agent.task.n_env, 1))  # [N, F]
        agent.task.w_eval = agent.task._w_eval_init.clone()  # [N, F]

    pos_slide = Scale(
        root,
        from_=0,
        to=1,
        digits=3,
        resolution=0.01,
        label="pos",
        orient=HORIZONTAL,
        command=update_pos,
    )
    pos_slide.pack()

    ang_slide = Scale(
        root,
        from_=0,
        to=1,
        digits=3,
        resolution=0.01,
        label="ang",
        orient=HORIZONTAL,
        command=update_ang,
    )
    ang_slide.pack()

    vel_slide = Scale(
        root,
        from_=0,
        to=1,
        digits=3,
        resolution=0.01,
        label="vel",
        orient=HORIZONTAL,
        command=update_vel,
    )
    vel_slide.pack()

    suc_slide = Scale(
        root,
        from_=0,
        to=1,
        digits=3,
        resolution=0.01,
        label="suc",
        orient=HORIZONTAL,
        command=update_suc,
    )
    suc_slide.pack()

    rew = DoubleVar(name="reward")  # instantiate the IntVar variable class
    rew.set(0.0)  # set it to 0 as the initial value

    # the label's textvariable is set to the variable class instance
    Label(root, textvariable=rew).pack()

    # root.mainloop()

    while True:
        agent.evaluate(root, rew)
        if agent.steps > agent.total_timesteps:
            break

    wandb.finish()


if __name__ == "__main__":
    launch_rlg_hydra()
