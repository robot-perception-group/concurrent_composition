from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *

from env.base.vec_env import VecEnv

from common.torch_jit_utils import *

import torch
import os


class PointMass2D(VecEnv):
    def __init__(self, cfg):
        # task-specific parameters
        self.num_obs = 7
        self.num_act = 2
        self.max_push_effort = 5.0  # the range of force applied to the pointer
        self.ball_height = 2
        
        self.reset_dist = cfg["goal"]["reset_dist"]  # when to reset x^2+y^2
        self.goal_lim = cfg["goal"]["goal_lim"] # list [position, velocity] range
        self.rand_vel_goal = cfg["goal"]["rand_vel_goal"]

        super().__init__(cfg=cfg)

        # task specific buffers
        self.goal_pos = (
            (torch.rand((self.num_envs, 3), device=self.sim_device) - 0.5)
            * 2
            * self.goal_lim[0]
        )
        self.goal_pos[..., 2] = self.ball_height

        if self.rand_vel_goal:
            self.goal_vel = (
                (torch.rand((self.num_envs, 3), device=self.sim_device) - 0.5)
                * 2
                * self.goal_lim[1]
            )
            self.goal_vel[..., 2] = 0
        else:
            self.goal_vel = torch.zeros([self.num_envs, 3], device=self.sim_device)

        # initialise envs and state tensors
        self.envs, self.num_bodies = self.create_envs()

        rb_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_states = gymtorch.wrap_tensor(rb_tensor)
        self.rb_pos = self.rb_states[:, 0:3].view(self.num_envs, self.num_bodies, 3)
        self.rb_rot = self.rb_states[:, 3:7].view(self.num_envs, self.num_bodies, 4)
        self.rb_lvels = self.rb_states[:, 7:10].view(self.num_envs, self.num_bodies, 3)
        self.rb_avels = self.rb_states[:, 10:13].view(self.num_envs, self.num_bodies, 3)

        # storing tensors for visualisations
        self.actions_tensor = torch.zeros(
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
        asset_root = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../assets"
        )
        asset_file = "urdf/ball.urdf"
        asset_options = gymapi.AssetOptions()
        point_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options
        )
        num_bodies = self.gym.get_asset_rigid_body_count(point_asset)

        # define pointer pose
        pose = gymapi.Transform()
        pose.p.z = self.ball_height  # generate the pointer 1m from the ground
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

        # relative pos
        pos = self.goal_pos[env_ids] - self.rb_pos[env_ids, 0]
        self.obs_buf[env_ids, 0] = pos[env_ids, 0]
        self.obs_buf[env_ids, 1] = pos[env_ids, 1]

        # relative vel
        lvel = self.rb_lvels[env_ids, 0]
        goal_lvel = self.goal_vel[env_ids]

        rel_vel = goal_lvel - lvel
        self.obs_buf[env_ids, 2] = rel_vel[env_ids, 0]
        self.obs_buf[env_ids, 3] = rel_vel[env_ids, 1]

        # vel norm
        goal_vel_norm = torch.linalg.norm(goal_lvel, axis=1, keepdims=False)
        rob_vel_norm = torch.linalg.norm(lvel, axis=1, keepdims=False)
        rel_vel_norm = goal_vel_norm - rob_vel_norm
        self.obs_buf[env_ids, 4] = rel_vel_norm

        # linear vel 
        self.obs_buf[env_ids, 5] = lvel[env_ids, 0]
        self.obs_buf[env_ids, 6] = lvel[env_ids, 1]


    def get_reward(self):
        # retrieve environment observations from buffer
        x = self.obs_buf[:, 0]
        y = self.obs_buf[:, 1]
        (
            self.reward_buf[:],
            self.reset_buf[:],
            self.return_buf[:],
            self.truncated_buf[:],
        ) = compute_point_reward(
            x,
            y,
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
        positions[:, :, 2] = self.ball_height

        velocities = 2 * (
            torch.rand((len(env_ids), self.num_bodies, 3), device=self.sim_device) - 0.5
        )

        velocities[..., 2] = 0

        rotations = torch.zeros((len(env_ids), 3), device=self.sim_device)

        pos_goals = (
            (torch.rand((len(env_ids), 3), device=self.sim_device) - 0.5)
            * 2
            * self.goal_lim[0]
        )
        pos_goals[:, 2] = self.ball_height

        if self.rand_vel_goal:
            vel_goals = (
                (torch.rand((len(env_ids), 3), device=self.sim_device) - 0.5)
                * 2
                * self.goal_lim[1]
            )
            vel_goals[:, 2] = 0
        else:
            vel_goals = torch.zeros((len(env_ids), 3), device=self.sim_device)

        # set random pos, rot, vels
        self.rb_pos[env_ids, :] = positions[:]

        self.rb_rot[env_ids, 0, :] = quat_from_euler_xyz(
            rotations[:, 0], rotations[:, 1], rotations[:, 2]
        )

        self.rb_lvels[env_ids, :] = velocities[:]
        self.goal_pos[env_ids, :] = pos_goals[:]
        self.goal_vel[env_ids, :] = vel_goals[:]

        # selectively reset the environments
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.rb_states),
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

        self.actions_tensor[:, 0, 0] = actions[:, 0]
        self.actions_tensor[:, 0, 1] = actions[:, 1]

        # unwrap tensors
        forces = gymtorch.unwrap_tensor(self.actions_tensor)

        # apply actions
        self.gym.apply_rigid_body_force_tensors(
            self.sim, forces, None, gymapi.ENV_SPACE
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
        num_lines = 4
        line_colors = [[0, 0, 0], [200, 0, 0], [200, 0, 0], [200, 0, 0]]

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
                    self.rb_pos[i, 0, 0].item(),
                    self.rb_pos[i, 0, 1].item(),
                    self.rb_pos[i, 0, 2].item(),
                ],
                [
                    self.rb_pos[i, 0, 0].item()
                    - 0.4 * self.actions_tensor[i, 0, 0].item(),
                    self.rb_pos[i, 0, 1].item(),
                    self.rb_pos[i, 0, 2].item(),
                ],
                [
                    self.rb_pos[i, 0, 0].item(),
                    self.rb_pos[i, 0, 1].item(),
                    self.rb_pos[i, 0, 2].item(),
                ],
                [
                    self.rb_pos[i, 0, 0].item(),
                    self.rb_pos[i, 0, 1].item()
                    - 0.4 * self.actions_tensor[i, 0, 1].item(),
                    self.rb_pos[i, 0, 2].item(),
                ],
                [
                    self.rb_pos[i, 0, 0].item(),
                    self.rb_pos[i, 0, 1].item(),
                    self.rb_pos[i, 0, 2].item(),
                ],
                [
                    self.rb_pos[i, 0, 0].item(),
                    self.rb_pos[i, 0, 1].item(),
                    self.rb_pos[i, 0, 2].item()
                    - 0.4 * self.actions_tensor[i, 0, 2].item(),
                ],
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
    reset_dist,
    reset_buf,
    progress_buf,
    return_buf,
    truncated_buf,
    max_episode_length,
):
    # type: (Tensor, Tensor, float, Tensor, Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor, Tensor, Tensor]

    # square distance from goal
    sqr_dist = (x_pos) ** 2 + (y_pos) ** 2

    # Proximity reward
    A1 = 1
    B1 = (2 + torch.log(A1)) / (8**2)

    proximity_rew = (torch.exp(-B1 * sqr_dist) + torch.exp(-3 * sqr_dist)) / 2
    reward = A1 * proximity_rew

    # Total
    return_buf += reward

    reset = torch.where(
        torch.abs(sqr_dist) > reset_dist, torch.ones_like(reset_buf), reset_buf
    )
    reset = torch.where(
        progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset
    )

    truncated_buf = torch.where(
        progress_buf >= max_episode_length - 1,
        torch.ones_like(reset_buf),
        truncated_buf,
    )

    return reward, reset, return_buf, truncated_buf
