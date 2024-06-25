from re import X
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, TypeVar
import os
import sys
from matplotlib import patches
from sklearn.metrics import det_curve
print(os.path.dirname(__file__))
sys.path.append(os.path.dirname(__file__))

from pygame import init
from common import pi2pi, Casadi_pi2pi, xyyaw2v

from constants import VehicleConst
from robot2actuator import dxdydyaw2ICM, k_theta_v_to_control, ktv2ICM, relative_v_and_omega_control
Pose = Tuple[float, float, float]  # (x, y, yaw)
from casadi import (DM, Opti, OptiSol, cos, diff, sin, 
                    sumsqr, vertcat, pi, fmod, norm_2, 
                    atan2, if_else, tan , horzcat, log, 
                    fabs, acos, sqrt)
import numpy as np
import logging


def AWS_wheel_constrain(v, s, omega, L, W):
    # v,s are the wheel speed and steering angle
    # v = [FL, FR, RL, RR]
    # s = [FL, FR, RL, RR]
    # L is the wheel base
    # W is the wheel width
    # omega is the robot's angular velocity
    # v = R*omega
    # R_F*sin(s_FL)-R_R*sin(s_RL) = L*omega
    # R_L*sin(s_RL)-R_R*sin(s_RR) = L*omega
    constrain = []
    constrain.append(v[1]*cos(s[1])-v[0]*cos(s[0]) - omega*W)
    constrain.append(v[3]*cos(s[3])-v[2]*cos(s[2]) - omega*W)
    constrain.append(v[0]*sin(s[0])-v[2]*sin(s[2]) - omega*L)
    constrain.append(v[1]*sin(s[1])-v[3]*sin(s[3]) - omega*L)
    return constrain # set this parameter bound==0

def wheel2robot(v, s, L, W):
    # v,s are the wheel speed and steering angle
    # v = [FL, FR, RL, RR]
    # s = [FL, FR, RL, RR]
    # L is the wheel base
    # W is the wheel width
    # omega is the robot's angular velocity
    v_FL = v[0]
    v_RL = v[2]
    s_FL = s[0]
    s_RL = s[2]
    RL_y = -L/(tan(s_FL)-tan(s_RL))
    RL_x = -RL_y*tan(s_FL)
    R_RL = norm_2(vertcat(RL_x,RL_y))
    omega = v_RL/R_RL
    R_O_x = L/2-RL_x
    R_O_y = -W/2-RL_y
    R_O = norm_2(vertcat(R_O_x,R_O_y))
    v_O = R_O*omega
    robot_v_x = v_O*(-R_O_y/R_O)
    robot_v_y = v_O*(R_O_x/R_O)
    return robot_v_x, robot_v_y, omega

def dxdydyaw2wheelstate(dx,dy,dyaw,vs=False):
    wheel_positions = VehicleConst.wheel_positions
    wheel_vx = [None, None, None, None]
    wheel_vy = [None, None, None, None]
    for i in range(4):
        wheel_vx[i] = dx-wheel_positions[i][1]*dyaw
        wheel_vy[i] = dy+wheel_positions[i][0]*dyaw
    if not vs:
        return vertcat(wheel_vx[0], wheel_vx[1], wheel_vx[2], wheel_vx[3]), \
            vertcat(wheel_vy[0], wheel_vy[1], wheel_vy[2], wheel_vy[3])
    if vs:
        wheel_v = [None, None, None, None]
        wheel_s = [None, None, None, None]
        for i in range(4):
            wheel_v[i] = sqrt(wheel_vx[i]**2+wheel_vy[i]**2)
            wheel_s[i] = atan2(wheel_vy[i],wheel_vx[i])
        return vertcat(wheel_v[0], wheel_v[1], wheel_v[2], wheel_v[3]), \
            vertcat(wheel_s[0], wheel_s[1], wheel_s[2], wheel_s[3])

def d_dxdydyaw2wheelstate(ddx,ddy,ddyaw,wheel_vx=None,wheel_vy=None,vs=False):
    # robot heading is x axis, wheel sequence FL, FR, RL, RR
    wheel_positions = VehicleConst.wheel_positions
    wheel_num = len(VehicleConst.wheel_positions)
    d_wheel_vx = [None]*wheel_num
    d_wheel_vy = [None]*wheel_num
    for i in range(wheel_num):
        d_wheel_vx[i] = ddx-wheel_positions[i][1]*ddyaw
        d_wheel_vy[i] = ddy+wheel_positions[i][0]*ddyaw
    if vs:
        d_wheel_v = [None, None, None, None]
        d_wheel_s = [None, None, None, None]
        for i in range(wheel_num):
            d_wheel_v[i] = 1/sqrt(wheel_vx[i]**2+wheel_vy[i]**2)*(wheel_vx[i]*d_wheel_vx[i]+wheel_vy[i]*d_wheel_vy[i])
            d_wheel_s[i] = 1/(1+(wheel_vy[i]/wheel_vx[i])**2)*(d_wheel_vy[i]/wheel_vx[i]-wheel_vy[i]*wheel_vx[i]/d_wheel_vx[i]**2)
        return vertcat(d_wheel_vx[0], d_wheel_vx[1], d_wheel_vx[2], d_wheel_vx[3]), \
                vertcat(d_wheel_vy[0], d_wheel_vy[1], d_wheel_vy[2], d_wheel_vy[3]), \
                vertcat(d_wheel_v[0], d_wheel_v[1], d_wheel_v[2], d_wheel_v[3]), \
                vertcat(d_wheel_s[0], d_wheel_s[1], d_wheel_s[2], d_wheel_s[3])
    return vertcat(d_wheel_vx[0], d_wheel_vx[1], d_wheel_vx[2], d_wheel_vx[3]), \
        vertcat(d_wheel_vy[0], d_wheel_vy[1], d_wheel_vy[2], d_wheel_vy[3])

def RotationMatrix(theta):
    return vertcat(horzcat(cos(theta),-sin(theta)),horzcat(sin(theta),cos(theta)))

def dRotationMatrix(theta):
    return vertcat(horzcat(-sin(theta),-cos(theta)),horzcat(cos(theta),-sin(theta)))

def ddRotationMatrix(theta):
    return vertcat(horzcat(-cos(theta),sin(theta)),horzcat(-sin(theta),-cos(theta)))

def global_dxdydyaw2wheelstate(dx,dy,dyaw,v_rel_tehta = 0):
    wheel_positions = VehicleConst.wheel_positions
    wheel_num = len(VehicleConst.wheel_positions)
    wheel_vx = horzcat([None]*wheel_num)
    wheel_vy = horzcat([None]*wheel_num)
    wheel_v = vertcat(wheel_vx, wheel_vy)
    for i in range(wheel_num):
        wheel_v[:,i] = vertcat(dx,dy) + RotationMatrix(v_rel_tehta)*vertcat(wheel_positions[i][0],wheel_positions[i][1])

    return wheel_vx.T, wheel_vy.T

def global_d_dxdydyaw2wheelstate(ddx, ddy, ddyaw, dyaw, v_rel_tehta = 0):
    wheel_positions = VehicleConst.wheel_positions
    wheel_num = len(VehicleConst.wheel_positions)
    d_wheel_vx = horzcat([None]*wheel_num)
    d_wheel_vy = horzcat([None]*wheel_num)
    d_wheel_v = vertcat(d_wheel_vx, d_wheel_vy)
    for i in range(wheel_num):
        d_wheel_v[:,i] = \
            vertcat(ddx,ddy) \
            + dRotationMatrix(v_rel_tehta)*ddyaw*vertcat(wheel_positions[i][0],wheel_positions[i][1]) \
            + ddRotationMatrix(v_rel_tehta)*dyaw**2*vertcat(wheel_positions[i][0],wheel_positions[i][1])

    return d_wheel_vx.T, d_wheel_vy.T

class AWSMassPointOptimizer:
    """
    Smoothing a set of xy observations with a vehicle dynamics model.
    Solved with direct multiple-shooting.

    :param trajectory_len: trajectory length
    :param dt: timestep (sec)
    """

    def __init__(self, trajectory_len: int, dt: float, wheel_constrain=False, current_index=0):
        """
        :param trajectory_len: the length of trajectory to be optimized.
        :param dt: the time interval between trajectory points.
        """
        self.name = "AWSMassPointOptimizer"
        self.dt = dt
        self.trajectory_len = trajectory_len
        self.wheel_constrain = wheel_constrain
        self.current_index = current_index
        # Use a array of dts to make it compatible to situations with varying dts across different time steps.
        self._dts = np.asarray([[dt] * trajectory_len])
        self.init_guess_dim = 6 if not self.wheel_constrain else 6+8
        self.nx = 6 if not self.wheel_constrain else 6+8 # state dim
        self.nu = 3  # control dim
        self._init_optimization()

    def _init_optimization(self) -> None:
        """
        Initialize related variables and constraints for optimization.
        """
        self._optimizer = Opti()  # Optimization problem
        self._create_decision_variables()
        self._create_parameters()
        self._set_dynamic_constraints()
        self._set_state_constraints()
        self._set_control_constraints()
        self._set_objective()

        # Set default solver options (quiet)
        self._optimizer.solver("ipopt", {"ipopt.print_level": 2, "print_time": 1, "ipopt.sb": "no", "verbose":0})

    def set_reference_trajectory(self, initial_guess: Sequence[Pose], dts: Sequence[float] = None) -> None:
        """
        Set the reference trajectory that the smoother is trying to loosely track.

        :param x_curr: current state of size nx (x, y, yaw, speed)
        :param reference_trajectory: N+1 x 3 reference, where the second dim is for (x, y, yaw)
        """
        self._check_inputs(initial_guess[0,:], initial_guess)

        self._optimizer.set_value(self.x_curr, DM(initial_guess[0,:]))
        self._optimizer.set_value(self.ref_traj, DM(initial_guess[:,:3]).T)
        self._set_initial_guess(initial_guess)

    def solve(self) -> OptiSol:
        """
        Solve the optimization problem. Assumes the reference trajectory was already set.

        :return Casadi optimization class
        """
        return self._optimizer.solve()

    def _create_decision_variables(self) -> None:
        """
        Define the decision variables for the trajectory optimization.
        """
        # State trajectory (x, y, yaw, speed)
        self.state = self._optimizer.variable(self.nx, self.trajectory_len+1)
        self.position_x = self.state[0, :]
        self.position_y = self.state[1, :]
        self.yaw = self.state[2, :]
        self.d_x = self.state[3, :]
        self.d_y = self.state[4, :]
        self.d_yaw = self.state[5, :]
        if self.wheel_constrain:
            self.wheel_vx = self.state[6:10, :]
            self.wheel_vy = self.state[10:14, :]
            self.d_wheel_vx = diff(self.wheel_vx.T).T/self.dt# vertcat(self._dts,self._dts,self._dts,self._dts)
            self.d_wheel_vy = diff(self.wheel_vy.T).T/self.dt# vertcat(self._dts,self._dts,self._dts,self._dts)

            # self.wheel_s = atan2(self.wheel_vy,fabs(self.wheel_vx)+1e-9)# limit 2 PI/2 only
            self.wheel_s = atan2(self.wheel_vy,self.wheel_vx)# limit 2 PI/2 only
            # self.wheel_v = sqrt(self.wheel_vx**2+self.wheel_vy**2+1e-9)/VehicleConst.WHEEL_RADIUS
            self.wheel_v = sqrt(self.wheel_vx**2+self.wheel_vy**2)#/VehicleConst.WHEEL_RADIUS
            self.d_wheel_v = diff(self.wheel_v.T).T/self.dt#/vertcat(self._dts,self._dts,self._dts,self._dts)
            self.d_wheel_s = diff(self.wheel_s.T).T/self.dt#vertcat(self._dts,self._dts,self._dts,self._dts)

        # Control trajectory (curvature, accel)
        self.control = self._optimizer.variable(self.nu, self.trajectory_len)
        self.accel_x = self.control[0, :]
        self.accel_y = self.control[1, :]
        self.dd_yaw = self.control[2, :]

        # Derived control and state variables, dt[:, 1:] becuases state vector is one step longer than action.
        self.jerk_x = diff(self.accel_x) / self._dts[:, :-1]
        self.jerk_y = diff(self.accel_y) / self._dts[:, :-1]
        self.jerk_yaw = diff(self.dd_yaw) / self._dts[:, :-1]

    def _create_parameters(self) -> None:
        """
        Define the expert trjactory and current position for the trajectory optimizaiton.
        """
        self.ref_traj = self._optimizer.parameter(3, self.trajectory_len+1)  # (x, y, yaw)
        self.x_curr = self._optimizer.parameter(self.nx, 1)

    def _set_dynamic_constraints(self) -> None: # TODO
        r"""
        Set the system dynamics constraints as following:
          dx/dt = f(x,u)
          \dot{x} = speed * cos(yaw)
          \dot{y} = speed * sin(yaw)
          \dot{yaw} = speed * curvature
          \dot{speed} = accel
        """
        state = self.state
        control = self.control
        dt = self.dt

        def process(x: Sequence[float], u: Sequence[float]) -> Any:
            """Process for state propagation."""
            # return vertcat(x[3] * cos(x[2]), x[3] * sin(x[2]), x[3] * u[0], u[1])
            yaw = x[2]
            vx = x[3]
            vy = x[4]
            if self.wheel_constrain:
                wheel_ddx, wheel_ddy = d_dxdydyaw2wheelstate(u[0], u[1], u[2])

                return vertcat(vx*cos(yaw)-vy*sin(yaw),
                            vx*sin(yaw)+vy*cos(yaw),
                            x[5],
                            u,
                            wheel_ddx,
                            wheel_ddy,
                            )
            return vertcat(vx*cos(yaw)-vy*sin(yaw),
                            vx*sin(yaw)+vy*cos(yaw),
                            x[5],
                            u,
                            )

        for k in range(self.trajectory_len):  # loop over control intervals
            # Runge-Kutta 4 integration
            k1 = process(state[:, k], control[:, k])
            k2 = process(state[:, k] + dt / 2 * k1, control[:, k])
            k3 = process(state[:, k] + dt / 2 * k2, control[:, k])
            k4 = process(state[:, k] + dt * k3, control[:, k])
            next_state = state[:, k] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            self._optimizer.subject_to(state[:, k + 1] == next_state)  # close the gaps

    def _set_control_constraints(self) -> None:
        """Set the hard control constraints."""
        curvature_limit = VehicleConst.MAX_ddyaw # 1/m
        self._optimizer.subject_to(self._optimizer.bounded(-curvature_limit, self.dd_yaw, curvature_limit))
        accel_limit = VehicleConst.MAX_ACC # m/s^2
        self._optimizer.subject_to(self._optimizer.bounded(0, self.accel_x**2+self.accel_y**2, accel_limit**2))
        # self._optimizer.subject_to(self._optimizer.bounded(-accel_limit, self.accel_x, accel_limit))
        # self._optimizer.subject_to(self._optimizer.bounded(-accel_limit, self.accel_y, accel_limit))

    def _set_state_constraints(self) -> None:
        """Set the hard state constraints."""
        # Constrain the current time -- NOT start of history
        self._optimizer.subject_to(self.state[:, self.current_index] == self.x_curr)  # initial boundary condition
        if not self.wheel_constrain:
            max_speed = VehicleConst.MAX_V # m/s # TODO
            speed = self.d_x**2 + self.d_y**2
            self._optimizer.subject_to(self._optimizer.bounded(0.0, speed, max_speed**2))  # only forward
            max_yaw_rate = VehicleConst.MAX_dyaw # rad/s #TODO
            self._optimizer.subject_to(self._optimizer.bounded(-max_yaw_rate, self.d_yaw, max_yaw_rate))

        if self.wheel_constrain:
            self._optimizer.subject_to(self._optimizer.bounded(0, self.wheel_vx**2+self.wheel_vy**2, VehicleConst.MAX_WHEEL_V**2))
            # self._optimizer.subject_to(self._optimizer.bounded(-VehicleConst.MAX_WHEEL_ACC, 
            #                                                    self.d_wheel_vx/VehicleConst.WHEEL_RADIUS, 
            #                                                    VehicleConst.MAX_WHEEL_ACC))
            # self._optimizer.subject_to(self._optimizer.bounded(-VehicleConst.MAX_WHEEL_ACC, 
            #                                                    self.d_wheel_vy/VehicleConst.WHEEL_RADIUS, 
            #                                                    VehicleConst.MAX_WHEEL_ACC))
            # self._optimizer.subject_to(self._optimizer.bounded(0.0, self.wheel_vx, VehicleConst.MAX_WHEEL_V))
            # self._optimizer.subject_to(self._optimizer.bounded(0.0, self.wheel_vy, VehicleConst.MAX_WHEEL_V))
            # self._optimizer.subject_to(self._optimizer.bounded(0.0, 
            #                                                    self.d_wheel_vx**2+self.d_wheel_vy**2/(1e-9+self.wheel_vx[:,1:]**2+self.wheel_vy[:,1:]**2), 
            #                                                    VehicleConst.MAX_WHEEL_ACC**2/VehicleConst.WHEEL_RADIUS**2))
            # steer = diff(atan2(self.wheel_vy,self.wheel_vx).T).T
            # self._optimizer.subject_to(self._optimizer.bounded(-VehicleConst.MAX_STEER, steer, VehicleConst.MAX_STEER))
            # self._optimizer.subject_to(self._optimizer.bounded(-VehicleConst.MAX_dSTEER, d_steer, VehicleConst.MAX_dSTEER))
            # self._optimizer.subject_to(
            #     self._optimizer.bounded(0.,
            #                             self.wheel_vy**2/(0.02/60+self.wheel_vx**2), # inplace arbitrary rotation, speed threshold = 0.02 m/s
            #                             60**2)) # tan(s) < tan(89deg) = 57.28996, 
            # max_wheel_speed = VehicleConst.MAX_WHEEL_V
            # max_wheel_acc = VehicleConst.MAX_WHEEL_ACC
            # self._optimizer.subject_to(self._optimizer.bounded(-max_wheel_speed, self.wheel_v, max_wheel_speed))
            # self._optimizer.subject_to(self._optimizer.bounded(-max_wheel_acc, self.d_wheel_v, max_wheel_acc))

            # max_steering_angle = VehicleConst.MAX_STEER
            # max_steering_rate = VehicleConst.MAX_dSTEER
            # print("max_steering_angle", max_steering_angle)
            # self._optimizer.subject_to(self._optimizer.bounded(-max_steering_angle, self.wheel_s, max_steering_angle))
            # self._optimizer.subject_to(self._optimizer.bounded(-max_steering_rate, self.d_wheel_s, max_steering_rate))

    def _set_objective(self) -> None: # TODO
        """Set the objective function. Use care when modifying these weights."""
        # Follow reference, minimize control rates and absolute inputs
        alpha_xy = 0.1
        alpha_yaw = 0.5
        alpha_dd = 0.2
        alpha_dd_yaw = 0.2
        alpha_jerk = 0.02
        alpha_jerk_yaw = 0.02
        cost_stage = (0.
            + alpha_xy * sumsqr(self.ref_traj[:2, :] - vertcat(self.position_x, self.position_y))
            + alpha_yaw * sumsqr(Casadi_pi2pi(self.yaw) - self.ref_traj[2, :])
            # + alpha_yaw * sumsqr(self.yaw-self.ref_traj[2, :])
            # + alpha_yaw * sumsqr(self.yaw - self.ref_traj[2, :])
            + alpha_jerk * (sumsqr(self.jerk_x) + sumsqr(self.jerk_y))
            + alpha_jerk_yaw * sumsqr(self.jerk_yaw)
            + alpha_dd * (sumsqr(self.accel_x) + sumsqr(self.accel_y))
            + alpha_dd_yaw * sumsqr(self.dd_yaw)
        )

        # Take special care with the final state
        alpha_terminal_xy = 20.0
        alpha_terminal_yaw = 20.0  # really care about final heading to help with lane changes
        cost_terminal = (
            alpha_terminal_xy * sumsqr(self.ref_traj[:2, -1] - vertcat(self.position_x[-1], self.position_y[-1])) 
            + alpha_terminal_yaw * sumsqr(self.yaw[-1] - self.ref_traj[2, -1])
        )

        self._optimizer.minimize(cost_stage + cost_terminal) #+ self.trajectory_len / 4.0 *

    def _set_initial_guess(self, initial_guess: Sequence[Pose]) -> None:
        """Set a warm-start for the solver based on the reference trajectory."""
        self._check_inputs(initial_guess[0,:], initial_guess)

        # Initialize state guess based on reference
        self._optimizer.set_initial(self.state[:self.init_guess_dim,:], DM(initial_guess[:,:self.init_guess_dim]).T)  # (x, y, yaw)
        # self._optimizer.set_initial(self.state[3:6, :], DM(x_curr[3:6]))  # speed

        # I think initializing the controls would be quite noisy, so using default zero init

    def _check_inputs(self, x_curr: Sequence[float], reference_trajectory: Sequence[Pose]) -> None:
        """Raise ValueError if inputs are not of proper size."""
        if len(x_curr) != self.nx:
            raise ValueError(f"x_curr length {len(x_curr)} must be equal to state dim {self.nx}")

        if len(reference_trajectory) != self.trajectory_len + 1:
            raise ValueError(
                f"reference traj length {len(reference_trajectory)} must be equal to {self.trajectory_len + 1}"
            )




def traj_smooth(initial_guess: np.ndarray, _optimizer=None, dts: np.ndarray = None, result_dict:dict=None):
    """_summary_
    dt = 0.1
    th = 50
    Args:
        initial_guess (np.ndarray): [th,(x,y,yaw...)]
    Returns:
        _type_: _description_
    """
    # dimension check
    # ss = np.linalg.norm(np.diff(fut_traj,axis = -2),axis=-1)
    # if np.sum(ss)<=3:
    #     logging.error("Traj too short! Use G.T. instead")
    #     return fut_traj
    
    # TODO : correct this in DIPP, the init v should be the last state of history
    # ego_velocity = np.linalg.norm(init_state[:, 3:5], axis=1)/0.1 # didn't divided by dt before

    ## Set reference and solve
    # init state may inclute wheel state, velocity
    # ego_trajectory only include x,y,yaw
    if result_dict is not None:
        result_dict['failed'] = True
    _optimizer.set_reference_trajectory(initial_guess, dts)
    out_control = np.zeros((initial_guess[1:].shape[0],4))
    try:
        time_start = time.time()
        sol = _optimizer.solve()
        time_end = time.time()
        if result_dict is not None:
            result_dict['failed'] = False
            result_dict['opt_time'] = time_end - time_start
        print(f"Time cost of optimization: {time_end - time_start}")
    except RuntimeError:
        logging.error("Smoothing crashed! Use G.T. instead" )
        print("Smoothing crashed! Use G.T. instead")
        return initial_guess[1:], out_control
    if not sol.stats()['success']:
        if result_dict is not None:
            result_dict['failed'] = True
        logging.error(f"Smoothing failed with status {sol.stats()['return_status']}! Use G.T. instead")
        print(f"Smoothing failed with status {sol.stats()['return_status']}! Use G.T. instead")
        return initial_guess[1:], out_control
    
    ego_smoothed: List[np.float32] = np.vstack(
        [
            sol.value(_optimizer.state),
        ]
    )
    ego_smoothed = ego_smoothed.T
    control: List[np.float32] = np.vstack(
        [
            sol.value(_optimizer.control),
        ]
    )
    out_control[:,:3] = control.T
    if _optimizer.wheel_constrain:
        wheel_vx: List[np.float32] = np.vstack(
            [
                sol.value(_optimizer.wheel_vx),
            ]
        )
        wheel_vx = wheel_vx.T
        wheel_vy: List[np.float32] = np.vstack(
            [
                sol.value(_optimizer.wheel_vy),
            ]
        )
        wheel_vy = wheel_vy.T
        ego_smoothed[:,6:10] = wheel_vx[:,:]
        ego_smoothed[:,10:14] = wheel_vy[:,:]
    if _optimizer.var_dt:
        var_dts: List[np.float32] = np.vstack(
            [
                sol.value(_optimizer.var_dts),
            ]
        )
    result_traj = ego_smoothed[1:,:]
    out_control[:,3:4] = _optimizer.dt
    if _optimizer.var_dt:
        var_dts = var_dts.T
        out_control[:,3:4] = var_dts
    return result_traj, out_control

def is_discontinuous(vxvyvyaw, last_vs, dt):
    l_dx, l_dy, l_dyaw = last_vs[0,0], last_vs[0,1], last_vs[0,2]
    l_wheels_v, l_wheels_s = relative_v_and_omega_control(l_dx, l_dy, l_dyaw)
    dx, dy, dyaw = vxvyvyaw[0,0], vxvyvyaw[0,1], vxvyvyaw[0,2]
    wheels_v, wheels_s = relative_v_and_omega_control(dx, dy, dyaw)
    ws = wheels_s - l_wheels_s
    wv = wheels_v - l_wheels_v
    return (np.abs(wv).max()/VehicleConst.MAX_WHEEL_V)>dt or (np.abs(ws).max()/VehicleConst.MAX_dSTEER)>dt

def point_generation_dxdydyaw():
    T = VehicleConst.T + 1
    dt = VehicleConst.dt
    vxvyvyaw_bounds = np.array([[VehicleConst.MAX_V,VehicleConst.MAX_V,VehicleConst.MAX_dyaw]])
    # vxvyvyaw_bounds = np.array([[0.01,0.02,VehicleConst.MAX_dyaw]])
    axayayaw_bounds = np.array([[VehicleConst.MAX_ACC,VehicleConst.MAX_ACC,VehicleConst.MAX_ddyaw]])
    vxvyvyaw = np.random.rand(1,3)*2*vxvyvyaw_bounds-vxvyvyaw_bounds
    # init_vel = deepcopy(vxvyvyaw)
    points = np.array([[0.,0.,0.]])
    vs = vxvyvyaw
    for i_t in range(T):
        # forword simulation
        last_vs = vxvyvyaw
        axayayaw = (np.random.rand(1,3)*2*axayayaw_bounds-axayayaw_bounds)
        vxvyvyaw = vxvyvyaw + axayayaw*dt
        vxvyvyaw = vxvyvyaw.clip(-vxvyvyaw_bounds,vxvyvyaw_bounds)
        axayayaw = (vxvyvyaw-last_vs)/dt
        if is_discontinuous(vxvyvyaw, last_vs, dt):
            vxvyvyaw = last_vs

        if i_t>0:
            points = np.concatenate([points, points[-1:,:]+last_vs*VehicleConst.dt+0.5*axayayaw*VehicleConst.dt**2], axis=0)
            vs = np.concatenate([vs, vxvyvyaw], axis=0)
    return points, vs


# if __name__=='__main__':
from traj_generator import min_jerk
import matplotlib.pyplot as plt
def test(points = None, vs = None):

    VISULIZE = True
    T = VehicleConst.T if points is None else points.shape[0]-1
    if T <= 0:
        print(f"T should be greater than 0, now it is {T}, optimization skiped")
        return
    dt = VehicleConst.dt
    wheel_constrain = True
    optimizer = AWSMassPointOptimizer(T, dt, wheel_constrain=wheel_constrain)
    # optimizer = AWSCurvatureOptimizer(T, dt, wheel_constrain=wheel_constrain)
    dim = optimizer.nx
    init_state = np.zeros([T+1, dim])

    if points is None or vs is None:
        points, vs = point_generation_dxdydyaw() if optimizer.name=="AWSMassPointOptimizer" else point_generation_ktv()
    if points.shape[0]!=T+1:
        raise ValueError(f"points shape[0] must be {T+1}")
    init_state[:, :3] = points
    for i in range(T+1):
        if optimizer.name=="AWSMassPointOptimizer":
            init_state[i,3:6] = vs[i]
            wheel_v, wheel_s = relative_v_and_omega_control(vs[i,0],vs[i,1],vs[i,2])
            init_state[i,6:10] = wheel_v
            init_state[i,10:dim] = wheel_s

        if optimizer.name=="AWSCurvatureOptimizer":
            k,theta,v = dxdydawy2ktv(vs[i,0],vs[i,1],vs[i,2])
            init_state[i,3:6] = np.array([k,theta,v])
            wheel_v, wheel_s = k_theta_v_to_control(k,theta,v)
            init_state[i,6:10] = wheel_v
            init_state[i,10:dim] = wheel_s


    if VISULIZE:
        plt.figure(figsize=(20, 10),dpi=300)
        plt.plot(points[:,0], points[:,1], 'ro', label='waypoints')
        for i in range(T+1):
            plt.arrow(init_state[i,0], init_state[i,1], 
                    0.5*np.cos(init_state[i,2]), 0.5*np.sin(init_state[i,2]), 
                    width=0.02, color='b')
        plt.axis('equal')
        plt.savefig(f'{optimizer.name} points.png')
        # plt.show()
    # optimizing
    result_traj, control = traj_smooth(init_state, optimizer)

    # visulization
    if VISULIZE:
        fig = plt.figure(figsize=(20, 10),dpi=300)
        fig.suptitle(f'{optimizer.name} smoother', fontsize=16)
        ax1 = plt.subplot(221)
        ax1.set_title('trajectory')
        ax2 = plt.subplot(222)
        ax2.set_title('control')
        ax3 = plt.subplot(223)
        ax3.set_title('optimized ICM')
        ax4 = plt.subplot(224)
        ax4.set_title('original ICM')
        ax1.plot(points[:,0], points[:,1], 'ro', label='waypoints')
        ax1.plot(init_state[:,0], init_state[:,1], 'b-',label='traj')
        ax1.plot(result_traj[:,0], result_traj[:,1], 'g-',label='smoothed')
        for i in range(T):
            ax1.arrow(init_state[i+1,0], init_state[i+1,1], 
                    0.5*np.cos(init_state[i+1,2]), 0.5*np.sin(init_state[i+1,2]), 
                    width=0.02, color='b')
            ax1.arrow(result_traj[i,0], result_traj[i,1], 
                    0.5*np.cos(result_traj[i,2]), 0.5*np.sin(result_traj[i,2]), 
                    width=0.02, color='g')
        ax1.axis('equal')
        ax1.legend()


        ts = np.arange(0, T*VehicleConst.dt, VehicleConst.dt)

        ax2.plot(ts, result_traj[:,3], 'r-', label='control 1')
        ax2.plot(ts, result_traj[:,4], 'b-', label='control 2')
        ax2.plot(ts, result_traj[:,5], 'g-', label='control 3')
        ax2.plot(ts, init_state[1:,3], 'r--', label='ori control 1')
        ax2.plot(ts, init_state[1:,4], 'b--', label='ori control 2')
        ax2.plot(ts, init_state[1:,5], 'g--', label='ori control 3')
        ax2.axis('equal')
        ax2.legend()
        # if wheel_constrain:
        
        if optimizer.name=="AWSMassPointOptimizer":
            radius, theta, omega = dxdydyaw2ICM(result_traj[:,3],result_traj[:,4], result_traj[:,5])
        if optimizer.name=="AWSCurvatureOptimizer":
            radius, theta, omega = ktv2ICM(result_traj[:,3],result_traj[:,4], result_traj[:,5])
        radius = radius.clip(-50,50)
        rect = patches.Rectangle((-VehicleConst.half_l, -VehicleConst.half_w), VehicleConst.WHEEL_BASE, VehicleConst.WHEEL_WHIDTH, linewidth=1, edgecolor='r', facecolor='none')
        ax3.add_patch(rect)
        ax3.plot(radius*np.cos(theta), radius*np.sin(theta), 'b-', label='r_c')
        # ax3.plot(ts, omega, 'g-', label='d_yaw')
        ax3.axis('equal')
        ax3.legend()

        if optimizer.name=="AWSMassPointOptimizer":
            radius, theta, omega = dxdydyaw2ICM(init_state[:,3],init_state[:,4], init_state[:,5])
        if optimizer.name=="AWSCurvatureOptimizer":
            radius, theta, omega = ktv2ICM(init_state[:,3],init_state[:,4], init_state[:,5])
        radius = radius.clip(-50,50)
        rect = patches.Rectangle((-VehicleConst.half_l, -VehicleConst.half_w), VehicleConst.WHEEL_BASE, VehicleConst.WHEEL_WHIDTH, linewidth=1, edgecolor='r', facecolor='none')
        ax4.add_patch(rect)
        ax4.plot(radius*np.cos(theta), radius*np.sin(theta), 'b-', label='r_c')
        # ax4.plot(ts, omega, 'g-', label='d_yaw')
        ax4.axis('equal')
        ax4.legend()
        plt.tight_layout()
        plt.savefig(f'{optimizer.name} smoother.png')
        # plt.show()

    return control


if __name__=='__main__':
    failed = 0
    test_num = 100
    for i in range(100):
        control = test()
        if np.all(control==0):
            failed += 1
        print("fail rate:", failed/(i+1))
