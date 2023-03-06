import numpy as np
from scipy.spatial.transform import Rotation as R


class pm_feature:
    def __init__(
        self,
        combination=[True, False, False, False, False, False, False],
        success_threshold=(1, 1, 1, 1),
    ) -> None:
        self._pos_err = combination[0]
        self._pos_norm = combination[1]
        self._vel_err = combination[2]
        self._vel_norm = combination[3]
        self._ang_err = combination[4]
        self._angvel_err = combination[5]
        self._success = combination[6]

        self._st_pos = success_threshold[0]
        self._st_vel = success_threshold[1]
        self._st_ang = success_threshold[2]
        self._st_angvel = success_threshold[3]

        self.dim = (
            3 * combination[0]  # pos
            + combination[1]  # pos_norm
            + 3 * combination[2]  # vel
            + combination[3]  # vel_norm
            + 3 * combination[4]  # ang
            + 3 * combination[5]  # angvel
            + combination[6]  # suc
        )

    def extract(self, s):
        features = []

        pos = s[:, 0:3]
        if self._pos_err:
            features.append(-np.abs(pos))
        if self._pos_norm:
            features.append(-np.linalg.norm(pos, axis=1, keepdims=True))

        vel = s[:, 12:15]
        if self._vel_err:
            features.append(-np.abs(vel))
        if self._vel_norm:
            features.append(-np.linalg.norm(vel, axis=1, keepdims=True))

        if self._success:
            features.append(self.success_position(pos))

        return np.concatenate(features, 1)

    def success_position(self, pos):
        dist = np.linalg.norm(pos, axis=1, keepdims=True)
        suc = np.zeros_like(dist)
        suc[np.where(dist < self._st_pos)] = 1.0
        return suc


class prism_feature:
    def __init__(
        self,
        combination=[True, True, True, True, True, True],
        success_threshold=(1, 1, 1, 1),
    ) -> None:
        self._pos_err = combination[0]
        self._pos_norm = combination[1]
        self._vel_err = combination[2]
        self._vel_norm = combination[3]
        self._ang_err = combination[4]
        self._angvel_err = combination[5]
        self._success = combination[6]

        self._st_pos = success_threshold[0]
        self._st_vel = success_threshold[1]
        self._st_ang = success_threshold[2]
        self._st_angvel = success_threshold[3]

        self.dim = (
            3 * combination[0]  # pos
            + combination[1]  # pos_norm
            + 3 * combination[2]  # vel
            + combination[3]  # vel_norm
            + 3 * combination[4]  # ang
            + 3 * combination[5]  # angvel
            + combination[6]  # suc
        )

    def extract(self, s):
        features = []

        pos = s[:, 0:3]
        if self._pos_err:
            features.append(-np.abs(pos))
        if self._pos_norm:
            features.append(-np.linalg.norm(pos, axis=1, keepdims=True))

        vel = s[:, 12:15]
        if self._vel_err:
            features.append(-np.abs(vel))
        if self._vel_norm:
            features.append(-np.linalg.norm(vel, axis=1, keepdims=True))

        angvel = s[:, 15:18]
        if self._angvel_err:
            features.append(-np.abs(angvel))

        if self._success:
            features.append(self.success_position(pos))

        return np.concatenate(features, 1)

    def success_position(self, pos):
        dist = np.linalg.norm(pos, axis=1, keepdims=True)
        suc = np.zeros_like(dist)
        suc[np.where(dist < self._st_pos)] = 1.0
        return suc


class quadcopter_feature:
    def __init__(
        self,
        combination=[True, True, True, True, True, True, True],
        success_threshold=(1, 1, 1, 1),
    ) -> None:
        self._pos_err = combination[0]
        self._pos_norm = combination[1]
        self._vel_err = combination[2]
        self._vel_norm = combination[3]
        self._ang_err = combination[4]
        self._angvel_err = combination[5]
        self._success = combination[6]

        self._st_pos = success_threshold[0]
        self._st_vel = success_threshold[1]
        self._st_ang = success_threshold[2]
        self._st_angvel = success_threshold[3]

        self.dim = (
            3 * combination[0]  # pos
            + combination[1]  # pos_norm
            + 3 * combination[2]  # vel
            + combination[3]  # vel_norm
            + 3 * combination[4]  # ang
            + 3 * combination[5]  # angvel
            + combination[6]  # suc
        )

    def extract(self, s):
        features = []

        pos = s[:, 0:3]
        if self._pos_err:
            features.append(-np.abs(pos))
        if self._pos_norm:
            features.append(-np.linalg.norm(pos, axis=1, keepdims=True))

        ang = R.from_matrix(s[:, 3:12].reshape(-1, 3, 3)).as_euler("xyz", degrees=True)
        if self._ang_err:
            features.append(-np.abs(ang))

        vel = s[:, 12:15]
        if self._vel_err:
            features.append(-np.abs(vel))
        if self._vel_norm:
            features.append(-np.linalg.norm(vel, axis=1, keepdims=True))

        angvel = s[:, 15:18]
        if self._angvel_err:
            features.append(-np.abs(angvel))

        if self._success:
            features.append(self.success_position(pos))

        return np.concatenate(features, 1)

    def success_position(self, pos):
        dist = np.linalg.norm(pos, axis=1, keepdims=True)
        suc = np.zeros_like(dist)
        suc[np.where(dist < self._st_pos)] = 1.0
        return suc
