from re import X
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, TypeVar
import os
import sys
from matplotlib import patches
from sklearn.metrics import det_curve
print(os.path.dirname(__file__))
sys.path.append(os.path.dirname(__file__))

from pygame import init
from common import pi2pi, xyyaw2v

from constants import VehicleConst
from robot2actuator import dxdydyaw2ICM, k_theta_v_to_control, ktv2ICM, relative_v_and_omega_control
Pose = Tuple[float, float, float]  # (x, y, yaw)
from casadi import (DM, Opti, OptiSol, cos, diff, sin, 
                    sumsqr, vertcat, pi, fmod, norm_2, 
                    atan2, if_else, tan , horzcat, log, 
                    fabs, acos, sqrt)
import numpy as np
import logging
from mpc_controller import AWSMassPointOptimizer
from common import Casadi_pi2pi

def ktv2vxyvydyaw(k,theta,v):
    v_x = v*sin(theta)
    v_y = -v*cos(theta)
    d_yaw = k*v
    return v_x, v_y, d_yaw

def dxdydawy2ktv(dx,dy,dyaw):
    v = np.sqrt(dx**2+dy**2)
    k = np.divide(dyaw,v,out=np.zeros_like(dyaw),where=v!=0)+1e-8
    theta = np.arctan2(dy, dx)+np.pi/2
    return k, theta, v

def d_ktv2vxyvydyaw(dk,dtheta,dv,k,theta,v):
    d_v_x = dv*sin(theta)+v*cos(theta)*dtheta
    d_v_y = -dv*cos(theta)+v*sin(theta)*dtheta
    dd_yaw = dk*v+k*dv
    return d_v_x, d_v_y, dd_yaw

def d_ktv2dxdydyaw2wheelstate(dk,dtheta,dv,k,theta,v,L,W):
    half_l = L/2
    half_w = W/2
    wheel_r = VehicleConst.WHEEL_RADIUS
    # robot heading is x axis, wheel sequence FL, FR, RL, RR
    ddx,ddy,ddyaw = d_ktv2vxyvydyaw(dk,dtheta,dv,k,theta,v)
    wheel_positions = [[half_l,half_w],[half_l,-half_w],[-half_l,half_w],[-half_l,-half_w]]
    d_wheel_vx = [None, None, None, None]
    d_wheel_vy = [None, None, None, None]
    for i in range(4):
        d_wheel_vx[i] = ddx-wheel_positions[i][1]*ddyaw
        d_wheel_vy[i] = ddy+wheel_positions[i][0]*ddyaw

    return vertcat(d_wheel_vx[0], d_wheel_vx[1], d_wheel_vx[2], d_wheel_vx[3]), \
        vertcat(d_wheel_vy[0], d_wheel_vy[1], d_wheel_vy[2], d_wheel_vy[3])

def ktv2wheelstate(k,theta,v,L,W):
    # k,theta,v are the curvature, speed angle and speed of the robot
    # L is the wheel base
    # W is the wheel width
    # v is the robot's velocity in robot frame
    # v is each wheel speed
    # s is each steering angle
    half_l = L/2
    half_w = W/2
    wheel_r = VehicleConst.WHEEL_RADIUS
    # robot heading is x axis, wheel sequence FL, FR, RL, RR
    wheel_positions = [[half_l,half_w],[half_l,-half_w],[-half_l,half_w],[-half_l,-half_w]]
    rot_center = vertcat(-sin(theta)/k,cos(theta)/k)
    omega = v*k
    wheel_r_pos = [None, None, None, None]
    wheel_v = [None, None, None, None]
    wheel_s = [None, None, None, None]
    for i in range(4):
        wheel_r_pos[i] = vertcat(wheel_positions[i][0]-rot_center[0], wheel_positions[i][1]-rot_center[1])
        wheel_v[i] = norm_2(wheel_r_pos[i])*omega
        wheel_s[i] = atan2(wheel_r_pos[i][0],-wheel_r_pos[i][1])
    return vertcat(wheel_v[0], wheel_v[1], wheel_v[2], wheel_v[3]), vertcat(wheel_s[0], wheel_s[1], wheel_s[2], wheel_s[3])
    
    

def d_ktv2wheelstate(dk,dtheta,dv,k,theta,v):
    # k,theta,v are the curvature, speed angle and speed of the robot
    # L is the wheel base
    # W is the wheel width
    # v is the robot's velocity in robot frame
    # v is each wheel speed
    # s is each steering angle
    half_l = VehicleConst.WHEEL_BASE/2
    half_w = VehicleConst.WHEEL_WHIDTH/2
    # robot heading is x axis, wheel sequence FL, FR, RL, RR
    wheel_positions = [[half_l,half_w],[half_l,-half_w],[-half_l,half_w],[-half_l,-half_w]]
    rot_center = vertcat(-1/k*sin(theta),1/k*cos(theta))
    d_rot_center = vertcat(-1/k*cos(theta)*dtheta + dk/k**2*sin(theta),
                           -1/k*sin(theta)*dtheta - dk/k**2*cos(theta))
    omega = v*k
    d_omega = dv*k+v*dk
    wheel_r_pos = [None, None, None, None]
    wheel_v = [None, None, None, None]
    wheel_s = [None, None, None, None]
    d_wheel_v = [None, None, None, None]
    d_wheel_s = [None, None, None, None]
    for i in range(4):
        wheel_r_pos[i] = vertcat(wheel_positions[i][0]-rot_center[0], wheel_positions[i][1]-rot_center[1])
        d_wheel_r_pos = vertcat(-d_rot_center[0], -d_rot_center[1])
        wheel_v[i] = norm_2(wheel_r_pos[i])*omega
        d_wheel_v[i] = (wheel_r_pos[i][0]*d_wheel_r_pos[0]+wheel_r_pos[i][1]*d_wheel_r_pos[1])/norm_2(wheel_r_pos[i])*omega \
            + norm_2(wheel_r_pos[i])*d_omega
        wheel_s[i] = atan2(wheel_r_pos[i][0],-wheel_r_pos[i][1])
        d_wheel_s[i] = 1/(1+(wheel_r_pos[i][0]/-wheel_r_pos[i][1])**2) \
            *(d_wheel_r_pos[0]/-wheel_r_pos[i][1] \
              +wheel_r_pos[i][0]*d_wheel_r_pos[1]/wheel_r_pos[i][1]**2)
    return vertcat(d_wheel_v[0], d_wheel_v[1], d_wheel_v[2], d_wheel_v[3]), \
            vertcat(d_wheel_s[0], d_wheel_s[1], d_wheel_s[2], d_wheel_s[3])

def d_ktv2wheeldv(dk,dtheta,dv,k,theta,v):
    half_l = VehicleConst.WHEEL_BASE/2
    half_w = VehicleConst.WHEEL_WHIDTH/2
    wheel_r = VehicleConst.WHEEL_RADIUS
    # robot heading is x axis, wheel sequence FL, FR, RL, RR
    wheel_positions = [[half_l,half_w],[half_l,-half_w],[-half_l,half_w],[-half_l,-half_w]]
    rot_center = vertcat(-sin(theta)/k,cos(theta)/k)
    d_rot_center = vertcat(-1/k*cos(theta)*dtheta + dk/k**2*sin(theta),
                           -1/k*sin(theta)*dtheta - dk/k**2*cos(theta))
    omega = v*k
    d_omega = dv*k+v*dk
    r = 1/k
    d_r = -dk/k**2
    d_wheel_vx = [None, None, None, None]
    d_wheel_vy = [None, None, None, None]
    for i in range(4):
        wheel_r_pos = vertcat(wheel_positions[i][0]-rot_center[0], wheel_positions[i][1]-rot_center[1])
        d_wheel_r_pos = vertcat(-d_rot_center[0], -d_rot_center[1])
        d_wheel_vx[i] = d_omega*wheel_r_pos[0] + omega*d_wheel_r_pos[0]
        d_wheel_vy[i] = d_omega*-wheel_r_pos[1] + omega*d_wheel_r_pos[1]

    return vertcat(d_wheel_vx[0], d_wheel_vx[1], d_wheel_vx[2], d_wheel_vx[3]), \
            vertcat(d_wheel_vy[0], d_wheel_vy[1], d_wheel_vy[2], d_wheel_vy[3])



def d_ktv2wheelstate(dk,dtheta,dv,k,theta,v):
    # k,theta,v are the curvature, speed angle and speed of the robot
    # L is the wheel base
    # W is the wheel width
    # v is the robot's velocity in robot frame
    # v is each wheel speed
    # s is each steering angle
    half_l = VehicleConst.WHEEL_BASE/2
    half_w = VehicleConst.WHEEL_WHIDTH/2
    # robot heading is x axis, wheel sequence FL, FR, RL, RR
    wheel_positions = [[half_l,half_w],[half_l,-half_w],[-half_l,half_w],[-half_l,-half_w]]
    rot_center = vertcat(-1/k*sin(theta),1/k*cos(theta))
    d_rot_center = vertcat(-1/k*cos(theta)*dtheta + dk/k**2*sin(theta),
                           -1/k*sin(theta)*dtheta - dk/k**2*cos(theta))
    omega = v*k
    d_omega = dv*k+v*dk
    wheel_r_pos = [None, None, None, None]
    wheel_v = [None, None, None, None]
    wheel_s = [None, None, None, None]
    d_wheel_v = [None, None, None, None]
    d_wheel_s = [None, None, None, None]
    for i in range(4):
        wheel_r_pos[i] = vertcat(wheel_positions[i][0]-rot_center[0], wheel_positions[i][1]-rot_center[1])
        d_wheel_r_pos = vertcat(-d_rot_center[0], -d_rot_center[1])
        wheel_v[i] = norm_2(wheel_r_pos[i])*omega
        d_wheel_v[i] = (wheel_r_pos[i][0]*d_wheel_r_pos[0]+wheel_r_pos[i][1]*d_wheel_r_pos[1])/norm_2(wheel_r_pos[i])*omega \
            + norm_2(wheel_r_pos[i])*d_omega
        wheel_s[i] = atan2(wheel_r_pos[i][0],-wheel_r_pos[i][1])
        d_wheel_s[i] = 1/(1+(wheel_r_pos[i][0]/-wheel_r_pos[i][1])**2) \
            *(d_wheel_r_pos[0]/-wheel_r_pos[i][1] \
              +wheel_r_pos[i][0]*d_wheel_r_pos[1]/wheel_r_pos[i][1]**2)
    return vertcat(d_wheel_v[0], d_wheel_v[1], d_wheel_v[2], d_wheel_v[3]), \
            vertcat(d_wheel_s[0], d_wheel_s[1], d_wheel_s[2], d_wheel_s[3])

def d_ktv2wheeldv(dk,dtheta,dv,k,theta,v):
    half_l = VehicleConst.WHEEL_BASE/2
    half_w = VehicleConst.WHEEL_WHIDTH/2
    wheel_r = VehicleConst.WHEEL_RADIUS
    # robot heading is x axis, wheel sequence FL, FR, RL, RR
    wheel_positions = [[half_l,half_w],[half_l,-half_w],[-half_l,half_w],[-half_l,-half_w]]
    rot_center = vertcat(-sin(theta)/k,cos(theta)/k)
    d_rot_center = vertcat(-1/k*cos(theta)*dtheta + dk/k**2*sin(theta),
                           -1/k*sin(theta)*dtheta - dk/k**2*cos(theta))
    omega = v*k
    d_omega = dv*k+v*dk
    r = 1/k
    d_r = -dk/k**2
    d_wheel_vx = [None, None, None, None]
    d_wheel_vy = [None, None, None, None]
    for i in range(4):
        wheel_r_pos = vertcat(wheel_positions[i][0]-rot_center[0], wheel_positions[i][1]-rot_center[1])
        d_wheel_r_pos = vertcat(-d_rot_center[0], -d_rot_center[1])
        d_wheel_vx[i] = d_omega*wheel_r_pos[0] + omega*d_wheel_r_pos[0]
        d_wheel_vy[i] = d_omega*-wheel_r_pos[1] + omega*d_wheel_r_pos[1]

    return vertcat(d_wheel_vx[0], d_wheel_vx[1], d_wheel_vx[2], d_wheel_vx[3]), \
            vertcat(d_wheel_vy[0], d_wheel_vy[1], d_wheel_vy[2], d_wheel_vy[3])

class AWSCurvatureOptimizer(AWSMassPointOptimizer):
    def __init__(self, trajectory_len: int, dt: float, wheel_constrain=False, current_index=0):
        self.name = "AWSCurvatureOptimizer"
        self.wheel_constrain = wheel_constrain # add wheel state to state space and constrain them
        # super().__init__(trajectory_len, dt)
        self.dt = dt
        self.trajectory_len = trajectory_len
        self.current_index = current_index
        # Use a array of dts to make it compatible to situations with varying dts across different time steps.
        self._dts = np.asarray([[dt] * trajectory_len])
        self.nx = 6 if not self.wheel_constrain else 6+8 # state dim
        self.init_guess_dim = 6 if not self.wheel_constrain else 6+8
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
    
    def _create_decision_variables(self) -> None:
        """
        Define the decision variables for the trajectory optimization.
        """
        # State trajectory (x, y, yaw, speed)
        self.state = self._optimizer.variable(self.nx, self.trajectory_len+1)
        self.position_x = self.state[0, :]
        self.position_y = self.state[1, :]
        self.yaw = self.state[2, :]
        self.k = self.state[3, :] # curvature
        self.theta = self.state[4, :] # speed angle
        self.v = self.state[5, :] # speed
        # if self.wheel_constrain:
        #     self.wheel_vx = self.state[6:10,:]
        #     self.wheel_vy = self.state[10:,:]
        #     self.d_wheel_vx = diff(self.wheel_vx.T).T/vertcat(self._dts,self._dts,self._dts,self._dts)
        #     self.d_wheel_vy = diff(self.wheel_vy.T).T/vertcat(self._dts,self._dts,self._dts,self._dts)
        #     self.wheel_s = atan2(self.wheel_vy,fabs(self.wheel_vx)+1e-9)
        #     self.wheel_v = sqrt(self.wheel_vx**2+self.wheel_vy**2+1e-9)/VehicleConst.WHEEL_RADIUS
        #     self.d_wheel_s = diff(self.wheel_s.T).T/vertcat(self._dts,self._dts,self._dts,self._dts)
        #     self.d_wheel_v = diff(self.wheel_v.T).T/vertcat(self._dts,self._dts,self._dts,self._dts)

        # Control trajectory (curvature, accel)
        self.control = self._optimizer.variable(self.nu, self.trajectory_len)
        self.d_k = self.control[0, :]
        self.d_theta = self.control[1, :]
        self.d_v = self.control[2, :]

        # Derived control and state variables, dt[:, 1:] becuases state vector is one step longer than action.
        # minimize control jerk
        self.jerk_k = diff(self.d_k) / self._dts[:, :-1]
        self.jerk_theta = Casadi_pi2pi(diff(self.d_theta)) / self._dts[:, :-1]
        self.jerk_v = diff(self.d_v) / self._dts[:, :-1]
        # minimize dynamic jerk
        # self.jerk_x = diff(self.d_v*cos(self.theta[:])) / self._dts[:, :]
        # self.jerk_y = diff(self.d_v*sin(self.theta[:])) / self._dts[:, :]
        # self.jerk_yaw = Casadi_pi2pi(diff(self.d_yaw)) / self._dts

    def _process(self, x: Sequence[float], u: Sequence[float]) -> Any:
        """Process for state propagation."""
        yaw = x[2]
        curv = x[3]
        theta = x[4]
        v = x[5]
        d_yaw = curv*v
        d_position_x = cos(np.pi-yaw-theta)*v
        d_position_y = sin(np.pi-yaw-theta)*v
        
        # d_d_x, d_d_y, dd_yaw = d_ktv2vxyvydyaw(u[0],u[1],u[2],x[6],x[7],x[8])
        if self.wheel_constrain:
            d_wheel_vx, d_wheel_vy = d_ktv2dxdydyaw2wheelstate(u[0],u[1],u[2],x[6],x[7],x[8],VehicleConst.WHEEL_BASE,VehicleConst.WHEEL_WHIDTH)
            return vertcat(d_position_x,d_position_y,d_yaw,u,d_wheel_vx,d_wheel_vy)
        return vertcat(d_position_x,d_position_y,d_yaw,u)

    def _set_dynamic_constraints(self) -> None:
        state = self.state
        control = self.control
        dt = self.dt        
        for k in range(self.trajectory_len):  # loop over control intervals
            # Runge-Kutta 4 integration
            k1 = self._process(state[:, k], control[:, k])
            k2 = self._process(state[:, k] + dt / 2 * k1, control[:, k])
            k3 = self._process(state[:, k] + dt / 2 * k2, control[:, k])
            k4 = self._process(state[:, k] + dt * k3, control[:, k])
            next_state = state[:, k] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            self._optimizer.subject_to(state[:, k + 1] == next_state)  # close the gaps

    def _set_objective(self) -> None:
        # k can use log(k)
        # return super()._set_objective()
        alpha_xy = 1.0
        alpha_yaw = 0.5
        alpha_rate = 0.08
        alpha_abs = 0.08
        alpha_lat_accel_abs = 0.006
        alpha_lat_accel_rate = 0.006
        cost_stage = (
            alpha_xy * sumsqr(self.ref_traj[:2, :] - vertcat(self.position_x, self.position_y))
            + alpha_yaw * sumsqr(Casadi_pi2pi(self.ref_traj[2, :] - self.yaw))
            + alpha_rate * (sumsqr(self.jerk_theta) + sumsqr(self.jerk_v))
            + alpha_lat_accel_rate * sumsqr(self.jerk_k)
            # + alpha_abs * (sumsqr(self.jerk_x) + sumsqr(self.jerk_y))
            # + alpha_lat_accel_abs * sumsqr(self.jerk_yaw)
        )

        # Take special care with the final state
        alpha_terminal_xy = 20.0
        alpha_terminal_yaw = 40.0  # really care about final heading to help with lane changes
        cost_terminal = alpha_terminal_xy * sumsqr(
            self.ref_traj[:2, -1] - vertcat(self.position_x[-1], self.position_y[-1])
        ) + alpha_terminal_yaw * sumsqr(self.ref_traj[2, -1] - self.yaw[-1])

        self._optimizer.minimize(cost_stage + self.trajectory_len / 4.0 * cost_terminal)

    def _set_control_constraints(self) -> None:
        """Set the hard control constraints."""
        # if not self.wheel_constrain:
        self._optimizer.subject_to(self._optimizer.bounded(-VehicleConst.MAX_dACC, self.d_v, VehicleConst.MAX_ACC))
        self._optimizer.subject_to(self._optimizer.bounded(-VehicleConst.MAX_dSTEER, self.d_theta, VehicleConst.MAX_dSTEER))
            # self._optimizer.subject_to(self._optimizer.bounded(-VehicleConst.MAX_dCURV, 1/self.d_k, VehicleConst.MAX_dCURV))
        # else:
        #     pass

    def _set_state_constraints(self) -> None:
        """Set the hard state constraints."""
        # Constrain the current time -- NOT start of history
        self._optimizer.subject_to(self.state[:, self.current_index] == self.x_curr)  # initial boundary condition
        # self._optimizer.subject_to(fabs(self.k)>0)
        self._optimizer.subject_to(fabs(self.v)>0)
        # if not self.wheel_constrain:
        max_speed = VehicleConst.MAX_V # m/s
        self._optimizer.subject_to(self._optimizer.bounded(-max_speed, self.v, max_speed))
        max_yaw_rate = VehicleConst.MAX_dyaw # rad/s
        self._optimizer.subject_to(self._optimizer.bounded(-max_yaw_rate, self.k*self.v, max_yaw_rate))
        self._optimizer.subject_to(self.k>0)
        self._optimizer.subject_to((VehicleConst.half_l+1)<1/self.k)
        self._optimizer.subject_to(self._optimizer.bounded(0,self.theta,pi/2))
        
        # else:
            # self._optimizer.subject_to(self._optimizer.bounded(0, self.wheel_vx**2+self.wheel_vy**2, VehicleConst.MAX_WHEEL_V**2))
            # self._optimizer.subject_to(self._optimizer.bounded(-VehicleConst.MAX_WHEEL_ACC, 
            #                                                    self.d_wheel_vx/VehicleConst.WHEEL_RADIUS, 
            #                                                    VehicleConst.MAX_WHEEL_ACC))
            # self._optimizer.subject_to(self._optimizer.bounded(-VehicleConst.MAX_WHEEL_ACC, 
            #                                                    self.d_wheel_vy/VehicleConst.WHEEL_RADIUS, 
            #                                                    VehicleConst.MAX_WHEEL_ACC))
            # self._optimizer.subject_to(self._optimizer.bounded(0.0, self.wheel_vx, VehicleConst.MAX_WHEEL_V))
            # # self._optimizer.subject_to(self._optimizer.bounded(0.0, 
            # #                                                    self.d_wheel_vx**2+self.d_wheel_vy**2/(1e-9+self.wheel_vx[:,1:]**2+self.wheel_vy[:,1:]**2), 
            # #                                                    VehicleConst.MAX_WHEEL_ACC**2/VehicleConst.WHEEL_RADIUS**2))
            # d_steer = diff(atan2(self.wheel_vy,fabs(self.wheel_vx)+1e-9).T).T/self.dt
            # self._optimizer.subject_to(self._optimizer.bounded(-VehicleConst.MAX_dSTEER, d_steer, VehicleConst.MAX_dSTEER))
            # self._optimizer.subject_to(
            #     self._optimizer.bounded(0.,
            #                             self.wheel_vy**2/(0.02/60+self.wheel_vx**2), # inplace arbitrary rotation, speed threshold = 0.02 m/s
            #                             60**2)) # tan(s) < tan(89deg) = 57.28996, 
            # max_wheel_speed = VehicleConst.MAX_WHEEL_V
            # max_wheel_acc = VehicleConst.MAX_WHEEL_ACC
            # self._optimizer.subject_to(self._optimizer.bounded(-max_wheel_speed, self.wheel_v, max_wheel_speed))
            # self._optimizer.subject_to(self._optimizer.bounded(-max_wheel_acc, self.d_wheel_v, max_wheel_acc))

            # max_steering_angle = VehicleConst.MAX_STEER-1/180*pi
            # max_steering_rate = VehicleConst.MAX_dSTEER
            # self._optimizer.subject_to(self._optimizer.bounded(-max_steering_angle, self.wheel_s, max_steering_angle))
            # self._optimizer.subject_to(self._optimizer.bounded(-max_steering_rate, self.d_wheel_s, max_steering_rate))

# class AWSRotationMatrixOptimizer(AWSMassPointOptimizer):
#     def __init__(self, trajectory_len: int, dt: float, wheel_constrain=False, current_index=0):
#         self.name = "AWSRotationMatrixOptimizer"
#         self.wheel_constrain = wheel_constrain # add wheel state to state space and constrain them
#         # super().__init__(trajectory_len, dt)
#         self.dt = dt
#         self.trajectory_len = trajectory_len
#         self.current_index = current_index
#         # Use a array of dts to make it compatible to situations with varying dts across different time steps.
#         self._dts = np.asarray([[dt] * trajectory_len])
#         self.nx = 6 if not self.wheel_constrain else 6+8 # state dim
#         self.init_guess_dim = 6 if not self.wheel_constrain else 6+8
#         self.nu = 3  # control dim
#         self._init_optimization()


def point_generation_ktv():
    T = VehicleConst.T + 1
    dt = VehicleConst.dt
    min_r = VehicleConst.half_l+1
    vxvyvyaw_bounds = np.array([[VehicleConst.MAX_V,VehicleConst.MAX_V,VehicleConst.MAX_dyaw]])
    # vxvyvyaw_bounds = np.array([[0.01,0.02,VehicleConst.MAX_dyaw]])
    axayayaw_bounds = np.array([[VehicleConst.MAX_ACC,VehicleConst.MAX_ACC,VehicleConst.MAX_ddyaw]])
    vxvyvyaw = np.random.rand(1,3)*2*vxvyvyaw_bounds-vxvyvyaw_bounds
    # init_vel = deepcopy(vxvyvyaw)
    points = np.array([[0.,0.,0.]])
    ktvs = np.array([dxdydawy2ktv(vxvyvyaw[0,0],vxvyvyaw[0,1],vxvyvyaw[0,2])])
    ktvs = ktvs.clip(np.array([0.01,0,-VehicleConst.MAX_V]),np.array([1/min_r,np.pi,VehicleConst.MAX_V]))
    vxvyvyaw = np.array([ktv2vxyvydyaw(ktvs[0,0],ktvs[0,1],ktvs[0,2])])
    vs = vxvyvyaw
    for i_t in range(T):
        # forword simulation
        last_vs = vxvyvyaw
        axayayaw = (np.random.rand(1,3)*2*axayayaw_bounds-axayayaw_bounds)
        vxvyvyaw = vxvyvyaw + axayayaw*dt
        ktv = np.array([dxdydawy2ktv(vxvyvyaw[0,0],vxvyvyaw[0,1],vxvyvyaw[0,2])])
        ktv = ktv.clip(np.array([0.01,0,-VehicleConst.MAX_V]),np.array([1/min_r,np.pi,VehicleConst.MAX_V]))
        vxvyvyaw = np.array([ktv2vxyvydyaw(ktv[0,0],ktv[0,1],ktv[0,2])])
        # vxvyvyaw = vxvyvyaw.clip(-vxvyvyaw_bounds,vxvyvyaw_bounds)
        axayayaw = (vxvyvyaw-last_vs)/dt
        # if is_discontinuous(vxvyvyaw, last_vs, dt):
        #     vxvyvyaw = last_vs

        if i_t>0:
            points = np.concatenate([points, points[-1:,:]+last_vs*VehicleConst.dt+0.5*axayayaw*VehicleConst.dt**2], axis=0)
            vs = np.concatenate([vs, vxvyvyaw], axis=0)
            # ktvs = np.concatenate([ktvs,ktv], axis=0)
    return points, vs

