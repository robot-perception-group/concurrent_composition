from env import env_map
from common.feature import pm_feature, ptr_feature
import torch


class MultiTaskEnv:
    def __init__(self, env_cfg):
        self.env_cfg = env_cfg
        self.env = env_map[env_cfg["env_name"]](env_cfg)

        if env_cfg.get("single_task", False):
            task_weight = env_cfg["task"]["single"]
        else:
            task_weight = env_cfg["task"]["multi"]

        self.w_nav = task_weight["w_nav"]
        self.w_hov = task_weight["w_hov"]
        self.w_eval_nav = task_weight["w_eval_nav"]
        self.w_eval_hov = task_weight["w_eval_hov"]

        self.success_threshold = env_cfg["feature"]["success_threshold"]

    def define_tasks(self, env_cfg, combination):
        def get_w(c, d, w):
            # feature order: [pos, pos_norm, vel, vel_norm, ang, angvel, success]

            if "pointmass" in env_cfg["env_name"].lower():
                w_pos = c[0] * d * [w[0]]
                w_pos_norm = c[1] * [w[1]]
                w_vel = c[2] * d * [w[2]]
                w_vel_norm = c[3] * [w[3]]
                w_success = c[4] * [w[4]]
                return w_pos+ w_pos_norm+ w_vel+ w_vel_norm+ w_success

            elif "pointer" in env_cfg["env_name"].lower():
                w_pos_norm = c[0] * [w[0]]
                w_vel_norm = c[1] * [w[1]]
                w_ang_norm = c[2] * [w[2]]
                w_angvel_norm = c[3] * [w[3]]
                w_success = c[4] * [w[4]]
                return w_pos_norm + w_vel_norm + w_ang_norm + w_angvel_norm + w_success
            else:
                raise NotImplementedError(f"env name {env_cfg['env_name']} invalid")

        dim = env_cfg["dim"]

        w_nav = get_w(combination, dim, self.w_nav)
        w_hov = get_w(combination, dim, self.w_hov)
        w_eval_nav = get_w(combination, dim, self.w_eval_nav)
        w_eval_hov = get_w(combination, dim, self.w_eval_hov)

        task_w = Task_Weights(w_nav, w_hov, w_eval_nav, w_eval_hov, env_cfg)

        return task_w

    def getEnv(self):
        feature_type = self.env_cfg["feature"]["type"]
        combination = self.env_cfg["feature"][feature_type]
        task = self.define_tasks(self.env_cfg, combination)
        if "pointer" in self.env_cfg["env_name"].lower():
            feature = ptr_feature(self.env_cfg, combination, self.success_threshold)
        elif "pointmass" in self.env_cfg["env_name"].lower():
            feature = pm_feature(self.env_cfg, combination, self.success_threshold)
        else:
            raise NotImplementedError(f'no such env {self.env_cfg["env_name"]}')

        return self.env, task, feature

class Task_Weights:
    def __init__(self, w_nav, w_hov, w_eval_nav, w_eval_hov, env_cfg) -> None:
        self.dim = env_cfg["dim"]
        self.n_env = env_cfg["num_envs"]
        self.pos_index = env_cfg["feature"]["pos_index"]
        self.switch_threshold = env_cfg["task"]["switch_threshold"]
        self.device = env_cfg["sim"]["sim_device"]

        self.w_nav = torch.tensor(w_nav, device=self.device).float()  # [F]
        self.w_hov = torch.tensor(w_hov, device=self.device).float()  # [F]
        self.w_eval_nav = torch.tensor(w_eval_nav, device=self.device).float()  # [F]
        self.w_eval_hov = torch.tensor(w_eval_hov, device=self.device).float() # [F]

        self._w_init = torch.tile(self.w_nav, (self.n_env, 1))  # [N, F]
        self._w_eval_init = torch.tile(self.w_eval_nav, (self.n_env, 1))  # [N, F]

        self.w = self._w_init.clone()  # [N, F]
        self.w_eval = self._w_eval_init.clone()  # [N, F]


    def reset(self):
        self.w = self._w_init.clone()  # [N, F]
        self.w_eval = self._w_eval_init.clone()  # [N, F]

    def update_task(self, s, eval=False):
        dist = torch.linalg.norm(s[:, self.pos_index : self.pos_index + self.dim], axis=1)
        if eval:
            self.w_eval[torch.where(dist <= self.switch_threshold)[0], :] = self.w_eval_hov.clone()
            self.w_eval[torch.where(dist > self.switch_threshold)[0], :] = self.w_eval_nav.clone()
        else:
            self.w[torch.where(dist <= self.switch_threshold)[0], :] = self.w_hov.clone()
            self.w[torch.where(dist > self.switch_threshold)[0], :] = self.w_nav.clone()
        