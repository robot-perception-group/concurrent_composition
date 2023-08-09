import torch
from scipy.spatial.transform import Rotation as R


class pm_feature:
    def __init__(
        self,
        env_cfg,
        combination=[True, False, False, True, True], # p, |p|, v, |v|, success
        success_threshold=(1, 1),
    ) -> None:
        self.envdim = env_cfg["dim"]

        self._pos_err = combination[0]
        self._pos_norm = combination[1]
        self._vel_err = combination[2]
        self._vel_norm = combination[3]
        self._success = combination[4]

        self._st_pos = success_threshold[0]
        self._st_vel = success_threshold[1]

        self.dim = (
            self.envdim * combination[0]  # pos
            + combination[1]  # pos_norm
            + self.envdim * combination[2]  # vel
            + combination[3]  # vel_norm
            + combination[4]  # suc
        )

    def extract(self, s):
        features = []

        pos = s[:, 0 : self.envdim]
        if self._pos_err:
            features.append(-torch.abs(pos))
        if self._pos_norm:
            features.append(-torch.linalg.norm(pos, axis=1, keepdims=True))

        vel = s[:, self.envdim : 2 * self.envdim]
        if self._vel_err:
            features.append(-torch.abs(vel))
        if self._vel_norm:
            vel_norm = s[:, 2 * self.envdim, None] # |vr| - |vg|
            features.append(-torch.abs(vel_norm))

        if self._success:
            features.append(self.success_position(pos))

        return torch.cat(features, 1)

    def success_position(self, pos):
        dist = torch.linalg.norm(pos, axis=1, keepdims=True)
        suc = torch.zeros_like(dist)
        suc[torch.where(dist < self._st_pos)] = 1.0
        return suc


class ptr_feature:
    def __init__(
        self,
        env_cfg,
        combination=[True, True, True, True, True],
        success_threshold=(1, 1, 1, 1),
    ) -> None:
        self.envdim = env_cfg["dim"]

        self._pos_norm = combination[0]
        self._vel_norm = combination[1]
        self._ang_norm = combination[2]
        self._angvel_norm = combination[3]
        self._success = combination[4]

        self._st_pos = success_threshold[0]
        self._st_vel = success_threshold[1]
        self._st_ang = success_threshold[2]
        self._st_angvel = success_threshold[3]

        self.dim = (
            combination[0]  # pos_norm
            + combination[1]  # vel_norm
            + combination[2]  # ang_norm
            + combination[3]  # angvel_norm
            + combination[4]  # suc
        )

    def extract(self, s):
        features = []

        pos = s[:, 3:5]
        pos_norm = torch.linalg.norm(pos, axis=1, keepdims=True)

        if self._pos_norm:
            features.append(-pos_norm)

        vel = s[:, 5:7]
        vel_norm = torch.linalg.norm(vel, axis=1, keepdims=True)
        if self._vel_norm:
            features.append(-vel_norm)

        ang = s[:, 0:1]
        if self._ang_norm:
            features.append(-torch.linalg.norm(ang, axis=1, keepdims=True))

        angvel = s[:, 7:8]
        if self._angvel_norm:
            features.append(-torch.linalg.norm(angvel, axis=1, keepdims=True))

        if self._success:
            features.append(self.success_position(pos_norm))

        return torch.concatenate(features, 1)

    def success_position(self, pos_norm):
        suc = torch.zeros_like(pos_norm)
        suc[torch.where(pos_norm < self._st_pos)] = 1.0
        return suc
