import os
import time

import numpy as np
from raisimGymTorch.env.bin import (
    pointmass2d,
)  # pointmassXd, quadcopter_task0
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from ruamel.yaml import YAML, RoundTripDumper, dump

ENV = pointmass2d  # pointmassXd, quadcopter_task0
task_path = os.path.dirname(os.path.realpath(__file__))
raisim_unity_Path = task_path + "/../raisim_drone/raisimUnity/raisimUnity.x86_64"
model_path = task_path + "/../raisim_drone"

env_cfg = {
    "simulation_dt": 0.01,
    "control_dt": 0.04,
    "max_time": 15.0,
    "render": True,
    "num_envs": 240,
    "vis_every_n": 50,
    "visualize_eval": True,
    "num_threads": 40,
    "random_robot_state": False,
    "random_target_state": True,
    "reward": {
        "success": {"coeff": 1},
    },
}

env = VecEnv(
    ENV.RaisimGymEnv(model_path, dump(env_cfg, Dumper=RoundTripDumper)),
    env_cfg,
)
n_env = env_cfg["num_envs"]

for _ in range(10000):
    s = env.reset()

    if s is None:
        s = np.zeros((n_env, env.num_obs))

    for T in range(100000):
        # print(f"============ time step: {T} ==============")

        acts = 2 * np.random.random((n_env, env.num_acts)) - 1
        # acts = 0.0 * np.ones((n_env, env.num_acts))
        r, done = env.step(acts.astype(np.float32))
        s_next = env.observe()
        s = s_next

        print(f"{s.shape=}")
        print(f"{acts.shape=}")
        print(f"{r.shape=}")
        print(f"{done.shape=}")

        time.sleep(0.01)
