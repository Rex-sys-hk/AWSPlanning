# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))    
sys.path.append('plan_utils')

import gym
from gym import spaces
import numpy as np
import math
from plan_utils.robot2actuator import relative_v_and_omega_control


class AGVEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        skip_frame=1,
        physics_dt=1.0 / 60.0,
        rendering_dt=1.0 / 60.0,
        max_episode_length=256,
        seed=0,
        headless=True,
    ) -> None:
        from omni.isaac.kit import SimulationApp

        self.headless = headless
        self._simulation_app = SimulationApp({"headless": self.headless, "anti_aliasing": 0})
        self._skip_frame = skip_frame
        self._dt = physics_dt * self._skip_frame
        self._max_episode_length = max_episode_length
        self._steps_after_reset = int(rendering_dt / physics_dt)
        from omni.isaac.core import World
        from omni.isaac.core.articulations import Articulation
        from omni.isaac.core.utils.stage import add_reference_to_stage, create_new_stage
        from omni.isaac.wheeled_robots.robots import WheeledRobot
        # from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
        from omni.isaac.core.utils.types import ArticulationAction
        from omni.isaac.core.objects import VisualCuboid
        from omni.isaac.core.utils.nucleus import get_assets_root_path

        self._my_world = World(physics_dt=physics_dt, rendering_dt=rendering_dt, stage_units_in_meters=1.0)
        path_to_robot_usd = "/workspace/4WS_control/AGV/AGV_centered.usd"
        self.v_joints = ['FLWalk', 'FRWalk', 'RLWalk', 'RRWalk']
        self.p_joints = ['FLSteer', 'FRSteer', 'RLSteer', 'RRSteer']
        self.v_zeros = np.zeros(len(self.v_joints))
        self.p_zeros = np.zeros(len(self.p_joints))

        add_reference_to_stage(path_to_robot_usd, '/World')
        # self._articulation = Articulation('/World/AGV/Body')
        # self._my_world.scene.add_articulation(self._articulation)
        self.robot = self._my_world.scene.add(
            WheeledRobot(
                prim_path='/World/AGV/Body',
                name="AGV",
                wheel_dof_names=self.v_joints+self.p_joints,
                create_robot=False,
                # usd_path=path_to_robot_usd,
                # position=np.array([0, 0.0, 0.3]),
                # orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            )
        )
        # self.robot.initialize()
        self.controller = relative_v_and_omega_control
        # self.jetbot_controller = DifferentialController(name="simple_control", wheel_radius=0.0325, wheel_base=0.1125)
        self.goal = self._my_world.scene.add(
            VisualCuboid(
                prim_path="/new_cube_1",
                name="visual_cube",
                position=np.array([0.60, 0.30, 0.05]),
                size=0.1,
                color=np.array([1.0, 0, 0]),
            )
        )
        self.seed(seed)
        self.reward_range = (-float("inf"), float("inf"))
        gym.Env.__init__(self)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=float("inf"), high=float("inf"), shape=(24,), dtype=np.float32)

        self.max_velocity = 2
        self.max_angular_velocity = 1*math.pi
        self.reset_counter = 0
        ### TODO customized
        # self.actions = np.random.rand(3)*2-1
        self.robot_move_state = np.zeros(3)
        self.pre_dis_dot = None
        self.reward_order = 1
        return

    def get_dt(self):
        return self._dt
    
    def vel_world_to_robot(self, vel_world, yaw_world):
        v_x = vel_world[0]
        v_y = vel_world[1]
        dyaw = vel_world[2]
        robot_rpy_orien = yaw_world
        dx_r = np.sin(robot_rpy_orien[2])*v_y+np.cos(robot_rpy_orien[2])*v_x
        dy_r = np.cos(robot_rpy_orien[2])*v_y-np.sin(robot_rpy_orien[2])*v_x
        return np.array([dx_r,dy_r,dyaw])

    
    def set_move_state(self, state):
        self.robot_move_state = state

    def step(self, action):
        # from omni.isaac.core.utils.types import ArticulationAction
        previous_robot_position, _ = self.robot.get_world_pose()
        self.robot_move_state = action*self._dt+self.robot_move_state # TODO action diff ratio
        # action forward velocity , angular velocity on [-1, 1]
        raw_forward_x = self.robot_move_state[0]
        raw_forward_y = self.robot_move_state[1]
        raw_angular = self.robot_move_state[2]

        forward_velocity_x = raw_forward_x * self.max_velocity
        forward_velocity_y = raw_forward_y * self.max_velocity

        angular_velocity = raw_angular * self.max_angular_velocity
        # we apply our actions to the jetbot
        for i in range(self._skip_frame):
            v,s = self.controller(forward_velocity_x, forward_velocity_y, angular_velocity, wheel_rotate_velocity=True)
            self.robot.set_wheel_velocities(np.concatenate([v,self.p_zeros]))
            self.robot.set_wheel_positions(np.concatenate([self.v_zeros,s]))
            self._my_world.step(render=True) # TODO render was False
        observations = self.get_observations()
        info = {}
        done = False
        if self._my_world.current_time_step_index - self._steps_after_reset >= self._max_episode_length:
            done = True
        goal_world_position, _ = self.goal.get_world_pose()
        current_robot_position, _ = self.robot.get_world_pose()
        previous_dist_to_goal = np.linalg.norm(goal_world_position[:2] - previous_robot_position[:2])
        current_dist_to_goal = np.linalg.norm(goal_world_position[:2] - current_robot_position[:2])
        dis_dot = previous_dist_to_goal - current_dist_to_goal
        if self.pre_dis_dot is None:
            self.pre_dis_dot = dis_dot
        dis_dot_dot = self.pre_dis_dot - dis_dot
        self.pre_dis_dot = dis_dot
        reward = dis_dot if self.reward_order == 1 else dis_dot_dot
        if current_dist_to_goal < 0.1:
            done = True
        return observations, reward, done, info

    def reset(self):
        self._my_world.reset()
        self.reset_counter = 0
        self.robot_move_state = np.zeros(3)
        # randomize goal location in circle around robot
        alpha = 2 * math.pi * np.random.rand()
        r = 1.00 * math.sqrt(np.random.rand()) + 0.20
        self.goal.set_world_pose(np.array([math.sin(alpha) * r, math.cos(alpha) * r, 0.3]))
        observations = self.get_observations()
        return observations

    def get_observations(self):
        self._my_world.render()
        robot_world_position, robot_world_orientation = self.robot.get_world_pose()
        robot_linear_velocity = self.robot.get_linear_velocity()
        robot_angular_velocity = self.robot.get_angular_velocity()
        goal_world_position, _ = self.goal.get_world_pose()
        velocities = self.robot.get_wheel_velocities()[:len(self.v_joints)]
        positions = self.robot.get_wheel_positions()[len(self.v_joints):]
        return np.concatenate(
            [
                robot_world_position,
                robot_world_orientation,
                robot_linear_velocity,
                robot_angular_velocity,
                goal_world_position,
                velocities,
                positions
            ]
        )

    def render(self, mode="human"):
        return

    def close(self):
        self._simulation_app.close()
        return

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        np.random.seed(seed)
        return [seed]


if __name__ == "__main__":
    env = AGVEnv(headless=False)
    env.reset()
    for i in range(10000):
        action = np.random.rand(3) * 2 - 1
        for t in range(100):
            obs, reward, done, info = env.step(action)
            print("step: ", i, " reward: ", reward)
            if done:
                print('Reach goal in step: ', t)
                env.reset()
    env.close()