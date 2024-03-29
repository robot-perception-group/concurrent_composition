from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *

from .base.vec_env import VecEnv

from common.torch_jit_utils import *

import torch
import math


class Pointer(VecEnv):
    def __init__(self, cfg):
        # task-specific parameters
        self.num_obs = 10  # pole_angle + pole_vel + cart_vel + cart_pos
        self.num_act = 2  # force applied on the pole (-1 to 1)
        self.max_push_effort = 5.0  # the range of force applied to the pointer
        self.init_hight = 2

        self.reset_dist = cfg["goal"]["reset_dist"]  # when to reset x^2+y^2
        self.goal_lim = cfg["goal"]["goal_lim"] # list [position, velocity, angle] range
        self.goal_lim[2] *= math.pi
        self.rand_vel_goal = cfg["goal"]["rand_vel_goal"]
        self.rand_rot_goal = cfg["goal"]["rand_rot_goal"]

        super().__init__(cfg=cfg)

        # task specific buffers
        self.goal_pos = (
            (torch.rand((self.num_envs, 3), device=self.sim_device) - 0.5)
            * 2
            * self.goal_lim[0]
        )
        self.goal_pos[..., 2] = self.init_hight

        if self.rand_vel_goal:
            self.goal_vel = (
                (torch.rand((self.num_envs, 3), device=self.sim_device) - 0.5)
                * 2
                * self.goal_lim[1]
            )
            self.goal_vel[..., 1] = 0
            self.goal_vel[..., 2] = 0
        else:
            self.goal_vel = torch.zeros((self.num_envs, 3), device=self.sim_device)
            self.goal_vel[..., 0] = 2 # default velocity (2,0,0)

        if self.rand_rot_goal:
            self.goal_rot = (
                (torch.rand((self.num_envs, 3), device=self.sim_device) - 0.5)
                * 2
                * self.goal_lim[2]
            )
            self.goal_rot[..., 0] = 0
            self.goal_rot[..., 1] = 0
        else:
            self.goal_rot = torch.zeros((self.num_envs, 3), device=self.sim_device)

        ####
        self.goal_rot_w = torch.zeros((self.num_envs, 3), device=self.sim_device) 
        self.goal_rot_w[..., 2] = 0.5

        # initialise envs and state tensors
        self.envs, self.num_bodies = self.create_envs()

        rb_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_states = gymtorch.wrap_tensor(rb_tensor)
        self.rb_pos = self.rb_states[:, 0:3].view(self.num_envs, self.num_bodies, 3)
        self.rb_rot = self.rb_states[:, 3:7].view(self.num_envs, self.num_bodies, 4)
        self.rb_lvels = self.rb_states[:, 7:10].view(self.num_envs, self.num_bodies, 3)
        self.rb_avels = self.rb_states[:, 10:13].view(self.num_envs, self.num_bodies, 3)

        # storing tensors for visualisations
        self.force_tensor = torch.zeros(
            (self.num_envs, self.num_bodies, 3),
            device=self.sim_device,
            dtype=torch.float,
        )
        self.torques_tensor = torch.zeros(
            (self.num_envs, self.num_bodies, 3),
            device=self.sim_device,
            dtype=torch.float,
        )

        # step simulation to initialise tensor buffers
        self.gym.prepare_sim(self.sim)
        self.reset()

    def create_envs(self):
        # add ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.restitution = 1
        plane_params.static_friction = 0
        plane_params.dynamic_friction = 0
        self.gym.add_ground(self.sim, plane_params)

        # define environment space (for visualisation)
        spacing = self.goal_lim[0]
        lower = gymapi.Vec3(0.5 * -spacing, -spacing, 0.0)
        upper = gymapi.Vec3(0.5 * spacing, spacing, spacing)
        num_per_row = int(np.sqrt(self.num_envs))

        # add pointer asset
        asset_root = "assets"
        asset_file = "urdf/pointer.urdf"
        asset_options = gymapi.AssetOptions()
        point_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options
        )
        num_bodies = self.gym.get_asset_rigid_body_count(point_asset)

        # define pointer pose
        pose = gymapi.Transform()
        pose.p.z = self.init_hight  # generate the pointer 1m from the ground
        pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # generate environments
        self.pointer_handles = []
        envs = []
        print(f"Creating {self.num_envs} environments.")
        for i in range(self.num_envs):
            # create env
            env = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # add pointer here in each environment
            point_handle = self.gym.create_actor(
                env, point_asset, pose, "pointmass", i, 1, 0
            )

            envs.append(env)
            self.pointer_handles.append(point_handle)

        return envs, num_bodies

    def get_obs(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)

        # refreshes the rb state tensor with new values
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        roll, pitch, yaw = get_euler_xyz(self.rb_rot[env_ids, 0, :])

        # maps rpy from -pi to pi
        pitch = torch.where(pitch > torch.pi, pitch - 2 * torch.pi, pitch)
        yaw = torch.where(yaw > torch.pi, yaw - 2 * torch.pi, yaw)
        roll = torch.where(roll > torch.pi, roll - 2 * torch.pi, roll)

        # fix for orientation bug
        self.reset_buf = torch.where(
            abs(pitch) > 0.0873, torch.ones_like(self.reset_buf), self.reset_buf
        )
        self.reset_buf = torch.where(
            abs(roll) > 0.0873, torch.ones_like(self.reset_buf), self.reset_buf
        )

        # angles
        self.obs_buf[env_ids, 0] = yaw - self.goal_rot[env_ids, 2]

        # rotations
        sine_z = torch.sin(yaw)
        cosine_z = torch.cos(yaw)

        self.obs_buf[env_ids, 1] = sine_z[env_ids]
        self.obs_buf[env_ids, 2] = cosine_z[env_ids]

        # relative xyz pos
        pos = self.goal_pos[env_ids] - self.rb_pos[env_ids, 0]

        xp, yp, zp = globalToLocalRot(
            roll, pitch, yaw, pos[env_ids, 0], pos[env_ids, 1], pos[env_ids, 2]
        )
        self.obs_buf[env_ids, 3] = xp
        self.obs_buf[env_ids, 4] = yp

        # relative xyz vel
        vel = self.rb_lvels[env_ids, 0]

        xv, yv, zv = globalToLocalRot(
            roll, pitch, yaw, vel[env_ids, 0], vel[env_ids, 1], vel[env_ids, 2]
        )
        self.obs_buf[env_ids, 5] = xv - self.goal_vel[env_ids, 0]
        self.obs_buf[env_ids, 6] = yv - self.goal_vel[env_ids, 1]

        # angular velocities
        ang_vel = self.rb_avels[env_ids, 0]

        xw, yw, zw = globalToLocalRot(
            roll,
            pitch,
            yaw,
            ang_vel[env_ids, 0],
            ang_vel[env_ids, 1],
            ang_vel[env_ids, 2],
        )
        # self.obs_buf[env_ids, 7] = zw
        self.obs_buf[env_ids, 7] = zw - self.goal_rot_w[env_ids, 2] ####

        # absolute Z
        self.obs_buf[env_ids, 8] = self.rb_pos[env_ids, 0, 2]

    def get_reward(self):
        # retrieve environment observations from buffer
        x = self.obs_buf[:, 3]
        y = self.obs_buf[:, 4]
        # z = self.obs_buf[:, 11]

        z_abs = self.obs_buf[:, 8]

        yaw = self.obs_buf[:, 0]
        # pitch = self.obs_buf[:, 1]
        # roll = self.obs_buf[:, 0]

        # omegax = self.obs_buf[:, 15]
        # omegaz = self.obs_buf[:, 16]
        omegaz = self.obs_buf[:, 7]

        (
            self.reward_buf[:],
            self.reset_buf[:],
            self.return_buf[:],
            self.truncated_buf[:],
        ) = compute_point_reward(
            x,
            y,
            z_abs,
            yaw,
            omegaz,
            self.reset_dist,
            self.reset_buf,
            self.progress_buf,
            self.return_buf,
            self.truncated_buf,
            self.max_episode_length,
        )

    def reset(self):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(env_ids) == 0:
            return

        # randomise initial positions and velocities
        positions = torch.zeros(
            (len(env_ids), self.num_bodies, 3), device=self.sim_device
        )
        positions[:, :, 2] = self.init_hight

        velocities = 2 * (
            torch.rand((len(env_ids), self.num_bodies, 6), device=self.sim_device) - 0.5
        )

        velocities[..., 2:5] = 0

        rotations = (
            (torch.rand((len(env_ids), 3), device=self.sim_device) - 0.5) * 2 * math.pi
        )

        rotations[..., 0:2] = 0

        pos_goals = (
            (torch.rand((len(env_ids), 3), device=self.sim_device) - 0.5)
            * 2
            * self.goal_lim[0]
        )
        pos_goals[:, 2] = self.init_hight

        if self.rand_vel_goal:
            vel_goals = (
                (torch.rand((len(env_ids), 3), device=self.sim_device) - 0.5)
                * 2
                * self.goal_lim[1]
            )
            vel_goals[..., 1] = 0
            vel_goals[..., 2] = 0
        else:
            vel_goals = torch.zeros((self.num_envs, 3), device=self.sim_device)
            vel_goals[..., 0] = 2 # default velocity (2,0,0)

        if self.rand_rot_goal:
            rot_goals = (
                (torch.rand((len(env_ids), 3), device=self.sim_device) - 0.5)
                * 2
                * self.goal_lim[2]
            )
            rot_goals[..., 0] = 0
            rot_goals[..., 1] = 0
        else:
            rot_goals = torch.zeros((len(env_ids), 3), device=self.sim_device)

        # set random pos, rot, vels
        self.rb_pos[env_ids, :] = positions[:]

        self.rb_rot[env_ids, 0, :] = quat_from_euler_xyz(
            rotations[:, 0], rotations[:, 1], rotations[:, 2]
        )

        self.rb_lvels[env_ids, :] = velocities[..., 0:3]
        self.rb_avels[env_ids, :] = velocities[..., 3:6]

        self.goal_pos[env_ids, :] = pos_goals[:]
        self.goal_vel[env_ids, :] = vel_goals[:]
        self.goal_rot[env_ids, :] = rot_goals[:]

        # selectively reset the environments
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.rb_states[::2].contiguous()),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

        # clear relevant buffers
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.return_buf[env_ids] = 0
        self.truncated_buf[env_ids] = 0

        # refresh new observation after reset
        self.get_obs()

    def step(self, actions):
        actions = (
            actions.to(self.sim_device).reshape((self.num_envs, self.num_act))
            * self.max_push_effort
        )

        # viscosity multiplier
        Q = 3

        # friction = kv^2
        # Ks for forces :
        FK_X = -Q * 0.0189 * torch.sign(self.obs_buf[:, 5]) * self.obs_buf[:, 5] ** 2
        FK_Y = -Q * 0.0472 * torch.sign(self.obs_buf[:, 6]) * self.obs_buf[:, 6] ** 2
        # FK_Z = -Q * 0.0943 * torch.sign(self.obs_buf[:, 14]) * self.obs_buf[:, 14] ** 2

        # Ks for torques
        # TK_X = (
        #     -Q * 0.0004025 * torch.sign(self.obs_buf[:, 15]) * self.obs_buf[:, 15] ** 2
        # )
        # TK_Y = (
        #     -Q * 0.003354 * torch.sign(self.obs_buf[:, 16]) * self.obs_buf[:, 16] ** 2
        # )
        TK_Z = -Q * 0.003354 * torch.sign(self.obs_buf[:, 7]) * self.obs_buf[:, 7] ** 2

        self.force_tensor[:, 0, 0] = 0.5 * FK_X + actions[:, 0]
        self.force_tensor[:, 1, 0] = 0.5 * FK_X + actions[:, 0]

        self.force_tensor[:, 0, 1] = 0.5 * FK_Y
        self.force_tensor[:, 1, 1] = 0.5 * FK_Y

        # self.force_tensor[:, 0, 2] = 0.5 * FK_Z
        # self.force_tensor[:, 1, 2] = 0.5 * FK_Z

        # self.torques_tensor[:, 0, 0] = TK_X + actions[:, 1]
        # self.torques_tensor[:, 0, 1] = TK_Y + actions[:, 2]
        self.torques_tensor[:, 0, 2] = TK_Z + actions[:, 1]

        # unwrap tensors
        forces = gymtorch.unwrap_tensor(self.force_tensor)
        torques = gymtorch.unwrap_tensor(self.torques_tensor)

        # apply actions
        self.gym.apply_rigid_body_force_tensors(
            self.sim, forces, torques, gymapi.LOCAL_SPACE
        )

        # simulate and render
        self.simulate()
        if not self.headless:
            self.render()

        # reset environments if required
        self.progress_buf += 1

        self.get_obs()
        self.get_reward()

    def _generate_lines(self):
        num_lines = 3 + 4
        line_colors = [
            [0, 0, 0],
            [0, 0, 0],
            [200, 0, 0],
            [0, 200, 0],
            [0, 200, 0],
            [0, 200, 0],
            [0, 200, 0],
        ]

        roll, pitch, yaw = get_euler_xyz(self.rb_rot[:, 0, :])
        xa, ya, za = localToGlobalRot(
            roll,
            pitch,
            yaw,
            self.force_tensor[:, 0, 0],
            self.force_tensor[:, 0, 1],
            self.force_tensor[:, 0, 2],
        )

        line_vertices = []

        for i in range(self.num_envs):
            vertices = [
                [self.goal_pos[i, 0].item(), self.goal_pos[i, 1].item(), 0],
                [
                    self.goal_pos[i, 0].item(),
                    self.goal_pos[i, 1].item(),
                    self.goal_pos[i, 2].item(),
                ],
                [
                    self.goal_pos[i, 0].item(),
                    self.goal_pos[i, 1].item(),
                    self.goal_pos[i, 2].item(),
                ],
                [
                    self.goal_pos[i, 0].item() + math.cos(self.goal_rot[i, 0].item()),
                    self.goal_pos[i, 1].item() + math.sin(self.goal_rot[i, 0].item()),
                    self.goal_pos[i, 2].item(),
                ],
                [
                    self.rb_pos[i, 0, 0].item(),
                    self.rb_pos[i, 0, 1].item(),
                    self.rb_pos[i, 0, 2].item(),
                ],
                [
                    self.rb_pos[i, 0, 0].item() - xa[i].item(),
                    self.rb_pos[i, 0, 1].item() - ya[i].item(),
                    self.rb_pos[i, 0, 2].item() - za[i].item(),
                ],
                [self.goal_pos[i, 0].item() + 3, self.goal_pos[i, 1].item() + 3, 0.1],
                [self.goal_pos[i, 0].item() + 3, self.goal_pos[i, 1].item() - 3, 0.1],
                [self.goal_pos[i, 0].item() + 3, self.goal_pos[i, 1].item() + 3, 0.1],
                [self.goal_pos[i, 0].item() - 3, self.goal_pos[i, 1].item() + 3, 0.1],
                [self.goal_pos[i, 0].item() - 3, self.goal_pos[i, 1].item() - 3, 0.1],
                [self.goal_pos[i, 0].item() - 3, self.goal_pos[i, 1].item() + 3, 0.1],
                [self.goal_pos[i, 0].item() - 3, self.goal_pos[i, 1].item() - 3, 0.1],
                [self.goal_pos[i, 0].item() + 3, self.goal_pos[i, 1].item() - 3, 0.1],
            ]

            line_vertices.append(vertices)

        return line_vertices, line_colors, num_lines


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_point_reward(
    x_pos,
    y_pos,
    z_abs,
    yaw,
    wz,
    reset_dist,
    reset_buf,
    progress_buf,
    return_buf,
    truncated_buf,
    max_episode_length,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor,Tensor, Tensor,float) -> Tuple[Tensor, Tensor, Tensor, Tensor]

    # # square distance from goal
    # sqr_dist = (x_pos) ** 2 + (y_pos) ** 2 + (z_pos) ** 2

    # # Proximity reward
    # A1 = 0.55
    # B1 = (2 + torch.log(A1)) / (6**2)

    # # proximity_rew_gauss = (1 / torch.exp(-B1)) * torch.exp(-B1 * sqr_dist)
    # # proximity_rew = torch.where(sqr_dist > 1, proximity_rew_gauss, 1)
    # proximity_rew = (torch.exp(-B1 * sqr_dist) + torch.exp(-3 * sqr_dist)) / 2

    # # Angle reward
    # A2 = 0.3
    # B2 = 0.5

    # angle_rew = proximity_rew * torch.exp(-B2 * (pitch**2 + roll**2 + yaw**2))

    # # Rotation reward
    # A3 = 0.15
    # B3 = 0.01

    # rot_rew = (
    #     0.8 * torch.exp(-B3 * wz**2)
    #     + 0.1 * torch.exp(-B3 * wy**2)
    #     + 0.1 * torch.exp(-B3 * wx**2)
    # )

    # # Total
    # reward = A1 * proximity_rew + A2 * angle_rew + A3 * rot_rew

    ###################################################################################

    sqr_dist = (x_pos) ** 2 + (y_pos) ** 2

    # Proximity reward
    A1 = 0.55
    B1 = (2 + torch.log(A1)) / (6**2)

    proximity_rew_gauss = (A1 / torch.exp(-B1)) * torch.exp(-B1 * sqr_dist)
    proximity_rew = torch.where(sqr_dist > 1, proximity_rew_gauss, A1)

    # Angle reward
    # angle_rew = - (torch.abs(yaw) + torch.abs(pitch) + torch.abs(roll)) * proximity_rew / (A1 * 2)
    B4 = 0.5
    A4 = 0.3

    angle_rew = proximity_rew * torch.exp(-B4 * (yaw**2))
    # angle = torch.acos(2*(0*qx + 0*qy + 0*qz + 1*qw)**2 - 1)
    # angle_rew = torch.exp(- B4 * angle)

    # print(angle_rew[0])

    # Rotation reward
    WZ_LIM = 30

    A2 = 0.15
    # B2 = torch.log(10/(A2) + 1)/WZ_LIM**2
    B2 = 0.01

    # rot_rew = - (torch.exp(B2 * wz**2) - 1) - A2 * (torch.exp(B2 * wy**2) - 1) - A2 * (torch.exp(B2 * wx**2) - 1)
    rot_rew = 1 * torch.exp(-B2 * wz**2)

    # Thrust Reward
    A3 = 0.00
    B3 = 0.2

    # thrust_rew = -(torch.exp(-B3 * x_action) - 1)

    # Total
    reward = proximity_rew + A4 * angle_rew + A2 * rot_rew

    # print(proximity_rew[0], angle_rew[0], rot_rew[0])
    # print(reward[0])

    # adjust reward for reset agents
    reward = torch.where(z_abs < 0.75, torch.ones_like(reward) * -200.0, reward)
    # reward = torch.where(torch.abs(x_pos) > reset_dist, torch.ones_like(reward) * -200.0, reward)
    # reward = torch.where(torch.abs(y_pos) > reset_dist, torch.ones_like(reward) * -200.0, reward)
    # reward = torch.where(torch.abs(wz) > 45, torch.ones_like(reward) * -200.0, reward)
    # reward = torch.where(x_action < 0, reward - 0.1, reward)
    # reward = torch.where((torch.abs(x_pos) < 0.1) & (torch.abs(y_pos) < 0.1), reward + 1, reward)

    return_buf += reward

    reset = torch.where(
        torch.abs(sqr_dist) > reset_dist, torch.ones_like(reset_buf), reset_buf
    )
    reset = torch.where(
        (z_abs < 1.75) | (z_abs > 2.25), torch.ones_like(reset_buf), reset
    )  # limit the range and prevent orientation drift bug
    reset = torch.where(torch.abs(wz) > 70, torch.ones_like(reset_buf), reset)
    # reset = torch.where(torch.abs(wy) > 70, torch.ones_like(reset_buf), reset)
    # reset = torch.where(torch.abs(wx) > 70, torch.ones_like(reset_buf), reset)
    reset = torch.where(
        progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset
    )

    truncated_buf = torch.where(
        progress_buf >= max_episode_length - 1,
        torch.ones_like(reset_buf),
        truncated_buf,
    )

    return reward, reset, return_buf, truncated_buf
