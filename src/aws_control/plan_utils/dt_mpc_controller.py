import math
import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, TypeVar
import os
import sys
print(os.path.dirname(__file__))
sys.path.append(os.path.dirname(__file__))
from matplotlib import patches
from constants import VehicleConst
from casadi import (DM, Opti, OptiSol, cos, diff, sin, 
                    sumsqr, vertcat, pi, fmod, norm_2, 
                    atan2, if_else, tan , horzcat, log, 
                    fabs, acos, sqrt, exp, cumsum, atan)
import logging
from mpc_controller import traj_smooth
from map_utils import MapUI
from plan_utils.common import (np_RotationMatrix, 
                               np_dRotationMatrix, 
                               np_ddRotationMatrix)
Pose = Tuple[float, float, float]  # (x, y, yaw)
# def np_RotationMatrix(theta):
#     return np.array([[np.cos(theta),-np.sin(theta)],
#                      [np.sin(theta),np.cos(theta)]])

# def np_dRotationMatrix(theta, dtheta):
#     return np.array([[-np.sin(theta),-np.cos(theta)],
#                      [np.cos(theta),-np.sin(theta)]])*(dtheta)

# def np_ddRotationMatrix(theta, dtheta, ddtheta):
#     return np.array([[-np.cos(theta),np.sin(theta)],
#                      [-np.sin(theta),-np.cos(theta)]])*dtheta**2 \
#         +np.array([[-np.sin(theta),-np.cos(theta)],
#                    [np.cos(theta),-np.sin(theta)]])*ddtheta

def Casadi_pi2pi(angle):
    return fmod((angle+pi),2*pi)-pi

def RotationMatrix(theta):
    return vertcat(horzcat(cos(theta),-sin(theta)),
                   horzcat(sin(theta),cos(theta)))

def dRotationMatrix(theta, dtheta):
    return vertcat(horzcat(-sin(theta),-cos(theta)),
                   horzcat(cos(theta),-sin(theta)))*dtheta

def ddRotationMatrix(theta, dtheta, ddtheta):
    return vertcat(horzcat(-cos(theta),sin(theta)),
                   horzcat(-sin(theta),-cos(theta)))*dtheta**2 \
        + vertcat(horzcat(-sin(theta),-cos(theta)),
                  horzcat(cos(theta),-sin(theta)))*ddtheta

def global_dxdydyaw2wheelstate(yaw,dx,dy,dyaw):
    wheel_positions = VehicleConst.wheel_positions
    wheel_num = len(VehicleConst.wheel_positions)
    wheel_v = [None]*wheel_num
    for i in range(wheel_num):
        
        wheel_v[i] = vertcat(dx,dy) \
            + dRotationMatrix(yaw, dyaw) \
            @vertcat(wheel_positions[i][0],wheel_positions[i][1])
    wheel_v = horzcat(*wheel_v)
    return wheel_v[0, :].T, wheel_v[1, :].T

def global_d_dxdydyaw2wheelstate(yaw, dyaw, ddx, ddy, ddyaw):
    wheel_positions = VehicleConst.wheel_positions
    wheel_num = len(VehicleConst.wheel_positions)
    d_wheel_v = [None]*wheel_num # horzcat(d_wheel_vx, d_wheel_vy).T
    for i in range(wheel_num):
        d_wheel_v[i] = vertcat(ddx,ddy) \
            + ddRotationMatrix(yaw, dyaw, ddyaw)@\
                vertcat(wheel_positions[i][0],wheel_positions[i][1])
    d_wheel_v = horzcat(*d_wheel_v)
    return d_wheel_v[0,:].T, d_wheel_v[1,:].T

def d_vdelta(wheel_vx, wheel_vy, d_wheel_vx, d_wheel_vy, d_yaw):
    wheel_v = sqrt(wheel_vx**2+wheel_vy**2)
    d_wheel_v = (wheel_vx*d_wheel_vx+wheel_vy*d_wheel_vy)/(wheel_v)
    d_wheel_s = (wheel_vx*d_wheel_vy-wheel_vy*d_wheel_vx)/(wheel_v**2) - d_yaw
    return d_wheel_v, d_wheel_s

def mode_constraint(r_r, mode_indicator, steering_lim):
    # r_r: 2xN
    # indicator: 1xN
    # steering_lim: 1x1
    wheel_positions = VehicleConst.wheel_positions
    wheel_num = len(VehicleConst.wheel_positions)
    ulim_vec = vertcat(cos(steering_lim+pi/2), sin(steering_lim+pi/2))
    llim_vec = vertcat(cos(-steering_lim+pi/2), sin(-steering_lim+pi/2))
    return_list = []
    for i in range(wheel_num):
        # positive value is the vector while negative is the relative position 
        wheel_pos = vertcat(wheel_positions[i][0],wheel_positions[i][1])
        indicator = vertcat(mode_indicator[i,:],mode_indicator[i,:])
        r_r_w = r_r + wheel_pos
        # return_list+=[(-r_r_w@(indicator*ulim_vec)) * (-r_r_w@(indicator*llim_vec))<=0]
        # cross multiply of (ax,ay).T, (bx,by).T here equals to (ax*by-ay*bx)
        ub_dir = -r_r_w[0,:]*(indicator*ulim_vec)[1,:] \
                 +r_r_w[1,:]*(indicator*ulim_vec)[0,:]
        lb_dir = -r_r_w[0,:]*(indicator*llim_vec)[1,:] \
                 +r_r_w[1,:]*(indicator*llim_vec)[0,:]
        return_list+=[ub_dir*lb_dir<=0]
    return return_list

def polyline(C1,C2,C3,C4,t):
    return C1*t**3 + C2*t**2 + C3*t + C4

def d_polyline(C1,C2,C3,C4,t):
    return 3*C1*t**2 + 2*C2*t + C3

def dd_polyline(C1,C2,C3,C4,t):
    return 6*C1*t + 2*C2

class AWSGlobalMassPointOptimizer:
    """
    Smoothing a set of xy observations with a vehicle dynamics model.
    Solved with direct multiple-shooting.

    :param trajectory_len: trajectory length
    :param dt: timestep (sec)
    """

    def __init__(self, trajectory_len: int, dt: float, 
                 wheel_constrain=True, 
                 current_index=0, 
                 free_end=True, 
                 var_dt=False,
                 following_mode=False,
                 simplify_model=False,
                 mapui: Optional[MapUI] = None,
                 constant_interval=False,
                 polyline = False,
                 mode_search = False):
        """
        :param trajectory_len: the length of trajectory to be optimized.
        :param dt: the time interval between trajectory points.
        """
        self.name = "AWSGlobalMassPointOptimizer"
        self.discrete = []
        self.dt = dt
        self.free_end = free_end
        self.var_dt = var_dt
        self.constant_interval = constant_interval
        self.trajectory_len = trajectory_len
        self.wheel_constrain = wheel_constrain
        self.current_index = current_index
        # Use a array of dts to make it compatible to 
        # situations with varying dts across different time steps.
        self.init_guess_dim = 6 if not self.wheel_constrain else 6+8
        self.nx = 6 if not self.wheel_constrain else 6+8 # state dim
        self.nu = 3  # control dim
        self.following_mode = following_mode
        self.simplify_model = simplify_model
        self.mapui = mapui
        self.polyline = polyline
        self.mode_search = mode_search
        # self.discrete = []

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
        self._optimizer.solver("ipopt", {
            # 'max_iter': 200,
            # "ipopt.print_level": 1, 
            # 'discrete':self.discrete
            }
            )
        # print('show index 18: \n',self._optimizer.debug.g_describe(18))
        # print('show index 33: \n',self._optimizer.debug.g_describe(33))
        print("Solver initialized!")


    def set_reference_trajectory(self, 
                                 initial_guess: Sequence[Pose], 
                                 dts: Sequence[float]) -> None:
        """
        Set the reference trajectory that the smoother is trying to loosely track.

        :param x_curr: current state of size nx (x, y, yaw, speed)
        :param reference_trajectory: N+1 x 3 reference, where the second dim is for (x, y, yaw)
        """
        self._check_inputs(initial_guess[0,:], initial_guess)

        self.trajectory_len = len(initial_guess)-1
        if self.mapui is not None:
            self._preprocess_map(initial_guess)
        self._init_optimization()

        self._optimizer.set_value(self.x_curr, 
                                  DM(initial_guess[self.current_index,:]))
        self._optimizer.set_value(self.ref_traj, 
                                  DM(initial_guess[:,:self.nx]).T)
        if not self.free_end:
            self._optimizer.set_value(self.x_final, DM(initial_guess[-1,:]))
        self._set_initial_guess(initial_guess, dts)

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
        if self.polyline:
            self.C1 = self._optimizer.variable(3, 1)
            self.C2 = self._optimizer.variable(3, 1)
            self.C3 = self._optimizer.variable(3, 1)
            self.C4 = self._optimizer.variable(3, 1)
            self.t_0 = self._optimizer.variable(1, 1)
            self._optimizer.subject_to(self.t_0==0)
        # State trajectory (x, y, yaw, speed)
        self.state = self._optimizer.variable(self.nx, self.trajectory_len+1)
        # self.discrete += [False]*(self.nx*(self.trajectory_len+1))
        self.position_x = self.state[0, :]
        self.position_y = self.state[1, :]
        self.yaw = self.state[2, :]
        self.d_x = self.state[3, :]
        self.d_y = self.state[4, :]
        self.d_yaw = self.state[5, :]
        self.v = sqrt(self.d_x**2+self.d_y**2)
        self.r_d_x = cos(-self.yaw)*self.d_x - sin(-self.yaw)*self.d_y
        self.r_d_y = sin(-self.yaw)*self.d_x + cos(-self.yaw)*self.d_y
        self.r_v = sqrt(self.d_x**2+self.d_y**2)
        # Control trajectory (curvature, accel)
        self.control = self._optimizer.variable(self.nu, self.trajectory_len)
        # self.discrete += [False]*(self.nu*self.trajectory_len)
        self.accel_x = self.control[0, :]
        self.accel_y = self.control[1, :]
        self.dd_yaw = self.control[2, :]
        self.r_accel_x = \
            cos(-self.yaw[:,:-1])*self.accel_x \
            - sin(-self.yaw[:,:-1])*self.accel_y
        self.r_accel_y = \
            sin(-self.yaw[:,:-1])*self.accel_x \
            + cos(-self.yaw[:,:-1])*self.accel_y
        if self.wheel_constrain:
            wheel_vxs = []
            wheel_vys = []
            wheel_axs = []
            wheel_ays = []
            if self.simplify_model:
                for i in range(self.trajectory_len+1):
                    wheel_vx, wheel_vy = \
                        global_dxdydyaw2wheelstate(self.yaw[:,i], 
                                                   self.d_x[:,i], 
                                                   self.d_y[:,i], 
                                                   self.d_yaw[:,i])
                    wheel_vxs.append(wheel_vx)
                    wheel_vys.append(wheel_vy)
                for i in range(self.trajectory_len):
                    wheel_ax, wheel_ay = \
                        global_d_dxdydyaw2wheelstate(self.yaw[:,i],
                                                        self.d_yaw[:,i],
                                                        self.accel_x[:,i],
                                                        self.accel_y[:,i],
                                                        self.dd_yaw[:,i])
                    wheel_axs.append(wheel_ax)
                    wheel_ays.append(wheel_ay)
                # self.wheel_s = atan2(self.wheel_vy, self.wheel_vx) - vertcat(self.yaw,self.yaw,self.yaw,self.yaw)
                self.wheel_vx = horzcat(*wheel_vxs)
                self.wheel_vy = horzcat(*wheel_vys)
                self.wheel_ax = horzcat(*wheel_axs)
                self.wheel_ay = horzcat(*wheel_ays)
                self.wheel_v = sqrt(self.wheel_vx**2+self.wheel_vy**2)
                self.wheel_a = sqrt(self.wheel_ax**2+self.wheel_ay**2)

            ## get ICM rx ry
            ## init wheel_dir
            r_r_list = []
            wheel_dir_list = []
            r_wheel_vx = []
            r_wheel_vy = []
            for i in range(self.trajectory_len+1):
                # positive value is the vector while negative is the relative position 
                r_r = dRotationMatrix(self.yaw[i], self.d_yaw[i]) \
                    @(vertcat(self.d_x[i], self.d_y[i])/(self.d_yaw[i]**2))

                rotat_m = RotationMatrix(-self.yaw[i])
                r_wheel_v = rotat_m@vertcat(self.wheel_vx[:,i].T,
                                            self.wheel_vy[:,i].T)
                # self.slack_wheel_dir[:,i] = (1-2/1+exp(-5*r_wheel_v[0,:])).T
                r_r_list.append(r_r)
                wheel_dir_list.append(r_wheel_v[0,:]/ \
                                      fabs(r_wheel_v[0,:]))
                r_wheel_vx.append(r_wheel_v[0,:])
                r_wheel_vy.append(r_wheel_v[1,:])
            self.r_r = horzcat(*r_r_list)
            self.wheel_dir = vertcat(*wheel_dir_list).T
            self.r_wheel_vx = vertcat(*r_wheel_vx).T
            self.r_wheel_vy = vertcat(*r_wheel_vy).T
            # self.wheel_dir = self._optimizer.variable(4, self.trajectory_len+1)
            # self.discrete += [True]*(4*(self.trajectory_len+1))
            # self.slack_wheel_dir = \
            #     self._optimizer.variable(4, self.trajectory_len+1)
            # self.slack_wheel_dir = self._optimizer.parameter(
            #     4, self.trajectory_len+1)


        # Derived control and state variables, dt[:, 1:] becuases state vector is one step longer than action.
        if self.var_dt:
            self.var_dts = self._optimizer.variable(1, self.trajectory_len)
            # self.discrete += [False]*(self.trajectory_len)
            # self.jerk_x = diff(self.accel_x) / self.var_dts[:,:-1]
            # self.jerk_y = diff(self.accel_y) / self.var_dts[:,:-1]
            # self.jerk_yaw = diff(self.dd_yaw) / self.var_dts[:,:-1]
        else:
            self._dts = np.asarray([[self.dt] * self.trajectory_len])
            # self.jerk_x = diff(self.accel_x) / self._dts[:, :-1]
            # self.jerk_y = diff(self.accel_y) / self._dts[:, :-1]
            # self.jerk_yaw = diff(self.dd_yaw) / self._dts[:, :-1]
        print("decision variables created!")



    def _create_parameters(self) -> None:
        """
        Define the expert trjactory and current position for the trajectory optimizaiton.
        """
        self.ref_traj = self._optimizer.parameter(self.nx, 
                                                  self.trajectory_len+1)
        self.x_curr = self._optimizer.parameter(self.nx, 1)
        if not self.free_end:
            self.x_final = self._optimizer.parameter(self.nx, 1)

    def _set_dynamic_constraints(self) -> None: # TODO
        r"""
        Set the system dynamics constraints as following:
          dx/dt = f(x,u)
          \dot{x} = speed * cos(yaw)
          \dot{y} = speed * sin(yaw)
          \dot{yaw} = speed * curvature
          \dot{speed} = accel
        """
        sd = self.nx if not self.simplify_model else 6
        state = self.state
        control = self.control
        if self.polyline:
            dts = self.var_dts if self.var_dt else self._dts
            dts = horzcat(self.t_0, dts)
            self._optimizer.subject_to(
                self.position_x[:, :] == polyline(self.C1[0],self.C2[0],
                                                  self.C3[0],self.C4[0],
                                                  cumsum(dts,-1)) )
            self._optimizer.subject_to(
                self.position_y[:, :] == polyline(self.C1[1],self.C2[1],
                                                  self.C3[1],self.C4[1],
                                                  cumsum(dts,-1)) )
            self._optimizer.subject_to(
                self.yaw[:, :] == acos(sin(polyline(self.C1[2],self.C2[2],
                                           self.C3[2],self.C4[2],
                                           cumsum(dts,-1)) ) ))
            self._optimizer.subject_to(
                self.d_x[:, :] == d_polyline(self.C1[0],self.C2[0],
                                             self.C3[0],self.C4[0],
                                             cumsum(dts,-1)) )
            self._optimizer.subject_to(
                self.d_y[:, :] == d_polyline(self.C1[1],self.C2[1],
                                             self.C3[1],self.C4[1],
                                             cumsum(dts,-1)) )
            self._optimizer.subject_to(
                self.d_yaw[:, :] == d_polyline(self.C1[2],self.C2[2],
                                               self.C3[2],self.C4[2],
                                               cumsum(dts,-1)) )
            self._optimizer.subject_to(
                self.accel_x[:, :] == dd_polyline(self.C1[0],self.C2[0],
                                                  self.C3[0],self.C4[0],
                                                  cumsum(dts[:,1:],-1)) )
            self._optimizer.subject_to(
                self.accel_y[:, :] == dd_polyline(self.C1[1],self.C2[1],
                                                  self.C3[1],self.C4[1],
                                                  cumsum(dts[:,1:],-1)) )
            self._optimizer.subject_to(
                self.dd_yaw[:, :] == dd_polyline(self.C1[2],self.C2[2],
                                                 self.C3[2],self.C4[2],
                                                 cumsum(dts[:,1:],-1)) )
            return

        def process(x: Sequence[float], u: Sequence[float]) -> Any:
            """Process for state propagation."""
            yaw = x[2]
            vx = x[3]
            vy = x[4]
            dyaw = x[5]
            if self.wheel_constrain and not self.simplify_model:
                wheel_ddx, wheel_ddy = global_d_dxdydyaw2wheelstate(
                    yaw, dyaw, u[0], u[1], u[2])
                return vertcat(vx,
                            vy,
                            dyaw,
                            u,
                            wheel_ddx,
                            wheel_ddy,
                            )
            return vertcat(vx,
                            vy,
                            dyaw,
                            u,
                            )

        for k in range(self.trajectory_len):  # loop over control intervals
            # Runge-Kutta 4 integration
            dt = self.dt if not self.var_dt else self.var_dts[:, k]
            k1 = process(state[:sd, k], control[:, k])
            k2 = process(state[:sd, k] + dt / 2 * k1, control[:, k])
            k3 = process(state[:sd, k] + dt / 2 * k2, control[:, k])
            k4 = process(state[:sd, k] + dt * k3, control[:, k])
            next_state = state[:sd, k] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            self._optimizer.subject_to(state[:2, k + 1] == next_state[:2])
            self._optimizer.subject_to(state[3:sd, k + 1] == next_state[3:sd])
            self._optimizer.subject_to(
                # state[2, k + 1] == acos(cos(next_state[2]))
                # state[2, k + 1] == Casadi_pi2pi(next_state[2])
                state[2, k + 1] == next_state[2]
                )

        # for k in range(self.trajectory_len):
        #     # Runge-Kutta 4 integration
        #     dt = self.dt if not self.var_dt else self.var_dts[:, k]
        #     rot_vec_k1 = self.d_rot_vecs[:, k]
        #     rot_vec_k2 = self.d_rot_vecs[:, k] + dt / 2 * rot_vec_k1
        #     rot_vec_k3 = self.d_rot_vecs[:, k] + dt / 2 * rot_vec_k2
        #     rot_vec_k4 = self.d_rot_vecs[:, k] + dt * rot_vec_k3
        #     rot_d_vec_k1 = self.dd_rot_vecs[:, k]
        #     rot_d_vec_k2 = self.dd_rot_vecs[:, k] + dt / 2 * rot_d_vec_k1
        #     rot_d_vec_k3 = self.dd_rot_vecs[:, k] + dt / 2 * rot_d_vec_k2
        #     rot_d_vec_k4 = self.dd_rot_vecs[:, k] + dt * rot_d_vec_k3

        #     next_rot_vec = self.rot_vecs[:, k] \
        #         + dt / 6 * (rot_vec_k1 + 
        #                     2 * rot_vec_k2 + 
        #                     2 * rot_vec_k3 + 
        #                     rot_vec_k4)
        #     next_d_rot_vec = self.d_rot_vecs[:, k] \
        #         + dt / 6 * (rot_d_vec_k1 +
        #                     2 * rot_d_vec_k2 +
        #                     2 * rot_d_vec_k3 +
        #                     rot_d_vec_k4)
        #     self._optimizer.subject_to(
        #         self.rot_vecs[:, k + 1] == next_rot_vec[:])
        #     self._optimizer.subject_to(
        #         self.d_rot_vecs[:, k + 1] == next_d_rot_vec[:])
            

    def _set_control_constraints(self) -> None:
        """Set the hard control constraints."""
        accel_limit = VehicleConst.MAX_ACC # m/s^2
        self._optimizer.subject_to(
            self._optimizer.bounded(0, 
                                    self.accel_x**2+self.accel_y**2, 
                                    accel_limit**2)
            )
        ddyaw_limit = VehicleConst.MAX_ddyaw # rad/s^2
        self._optimizer.subject_to(
            self._optimizer.bounded(-ddyaw_limit, 
                                    self.dd_yaw, 
                                    ddyaw_limit)
            )
        

    def _set_state_constraints(self) -> None:
        """Set the hard state constraints."""
        # Constrain the current time -- NOT start of history
        # initial boundary condition
        self._optimizer.subject_to(
            self.state[:, self.current_index] == self.x_curr)
        if self.var_dt:
            self._optimizer.subject_to(self.var_dts>=0.1*self.dt)
            self._optimizer.subject_to(self.var_dts<3.0*self.dt)
        if not self.free_end:
            self._optimizer.subject_to(
                self.state[:3, -1] == self.x_final[:3])
            self._optimizer.subject_to(
                self.state[3, -1] == 0)
            self._optimizer.subject_to(
                self.state[4:6, -1] == 0.)
        if not self.wheel_constrain:
            max_speed = VehicleConst.MAX_V # m/s
            self._optimizer.subject_to(
                self.v**2<=max_speed**2)
            max_yaw_rate = VehicleConst.MAX_dyaw # rad/s
            self._optimizer.subject_to(
                self.d_yaw**2 <= max_yaw_rate**2)

        if self.wheel_constrain:
            for i in range(VehicleConst.wheel_num):
                self._optimizer.subject_to(
                    self.wheel_v[i,:]<=VehicleConst.MAX_V)
                self._optimizer.subject_to(
                    self.wheel_a[i,:]<=VehicleConst.MAX_ACC)
            wheel_steer_limits = \
                    np.array(VehicleConst.wheel_steer_limits).T
            ub_x = cos(wheel_steer_limits[1,:])
            ub_y = sin(wheel_steer_limits[1,:])
            lb_x = cos(wheel_steer_limits[0,:])
            lb_y = sin(wheel_steer_limits[0,:])
            for t in range(self.trajectory_len):
                self._optimizer.subject_to(
                    (self.r_wheel_vx[:,t+1] * ub_y - \
                     self.r_wheel_vy[:,t+1] * ub_x)
                    *(self.r_wheel_vx[:,t+1] * lb_y - \
                      self.r_wheel_vy[:,t+1] * lb_x)
                    <0)
                ### steering speed constraints
                # r_wheel_v_t = RotationMatrix(-self.yaw[:,t]) \
                #               @vertcat(self.wheel_vx[:,t].T, 
                #                        self.wheel_vy[:,t].T)
                r_wheel_vx_t = self.r_wheel_vx[:,t].T
                r_wheel_vy_t = self.r_wheel_vy[:,t].T
                # r_wheel_v_tp1 = RotationMatrix(-self.yaw[:,t+1]) \
                #                 @vertcat(self.wheel_vx[:,t+1].T, 
                #                          self.wheel_vy[:,t+1].T)
                r_wheel_vx_tp1 = self.r_wheel_vx[:,t+1].T
                r_wheel_vy_tp1 = self.r_wheel_vy[:,t+1].T
                
                # steer_diff = atan(r_wheel_v_t[1,:]/r_wheel_v_t[0,:])\
                #         -atan(r_wheel_v_tp1[1,:]/r_wheel_v_tp1[0,:])
                # if not self.var_dt:
                #     self._optimizer.subject_to(
                #             steer_diff
                #             <VehicleConst.MAX_dSTEER*self.dt)
                #     continue
                # if self.var_dt:
                #     self._optimizer.subject_to(
                #             steer_diff
                #             <VehicleConst.MAX_dSTEER*self.var_dts[:,t])
                #     continue
                inner_product = self.wheel_dir[:, t].T\
                                *r_wheel_vx_t \
                                *self.wheel_dir[:, t+1].T\
                                *r_wheel_vx_tp1 \
                                +self.wheel_dir[:, t].T\
                                    *r_wheel_vy_t\
                                *self.wheel_dir[:, t+1].T\
                                    *r_wheel_vy_tp1
                if not self.var_dt:
                    self._optimizer.subject_to(
                        inner_product.T
                        >cos(VehicleConst.MAX_dSTEER*self.dt) \
                             *self.wheel_v[:,t]*self.wheel_v[:,t+1])
                if self.var_dt:
                    # steer = atan2(self.r_wheel_vy,self.r_wheel_vx)
                    # self._optimizer.subject_to(
                    #     cos(steer[:,t+1]*self.wheel_dir[:, t]
                    #         -steer[:,t]*self.wheel_dir[:, t+1])
                    #     >cos(VehicleConst.MAX_dSTEER*self.var_dts[:,t]))

                    self._optimizer.subject_to(
                        inner_product.T
                          >cos(VehicleConst.MAX_dSTEER*self.var_dts[:,t])\
                            *self.wheel_v[:,t]*self.wheel_v[:,t+1])
                    if t>=1 and self.constant_interval:
                        self._optimizer.subject_to(
                            self.var_dts[:,t]==self.var_dts[:,t-1])
                
                ### steering phase constrain, wheel_v around 0 when changing direction
                ### mode priors parameters needed, body v 0 can be more promising
                # if not self.following_mode:
                #     self._optimizer.subject_to(
                #         self.wheel_v[:,t]**2 <=\
                #             (self.wheel_dir[:,t]
                #             +self.wheel_dir[:,t+1])**2/4 \
                #                 * VehicleConst.MAX_V
                #     )
                #     self._optimizer.subject_to(
                #         self.wheel_v[:,t+1]**2 <=\
                #             (self.wheel_dir[:,t]
                #             +self.wheel_dir[:,t+1])**2/4 \
                #                 * VehicleConst.MAX_V
                #     )
            # TODO explore the functionality of this term
            # self._optimizer.subject_to(
            #     self.slack_wheel_dir==self.wheel_dir
            # )

    def _set_objective(self) -> None: # TODO
        """Set the objective function. Use care when modifying these weights."""
        # Follow reference, minimize control rates and absolute inputs
        alpha_xy = 0.5 if not self.following_mode else 0.5
        alpha_yaw = 2.5 if not self.following_mode else 2.5
        # speed_ratio = 0. if not self.following_mode else .0

        cost_stage = (0.
            + alpha_xy * sumsqr(
                self.ref_traj[:2, 1:] - self.state[:2, 1:])
            + alpha_yaw * sumsqr(
                -cos(self.state[2, 1:] - self.ref_traj[2, 1:])+1)
            # + alpha_yaw * sumsqr(
            #     -self.state[2, 1:] - self.ref_traj[2, 1:])
            # + alpha_jerk * (sumsqr(self.jerk_x) + sumsqr(self.jerk_y))
            # + alpha_jerk_yaw * sumsqr(self.jerk_yaw)
            +  .5 * sumsqr(self.r_accel_x)
            +  .8 * sumsqr(self.r_accel_y)
            + 1.  * sumsqr(self.dd_yaw)
            # - 1.2 * sumsqr(self.r_d_x)*speed_ratio
            # - 1.2 * sumsqr(self.r_d_y)*speed_ratio
            # +  .1 * sumsqr(self.d_yaw)*speed_ratio
        )
        # cost_stage += 3*sumsqr(self.r_d_y) \
        #     if VehicleConst._rear <0.1 and not self.wheel_constrain else 0.
        # cost_stage += .0005 * sumsqr(self.r_wheel_vy) \
        #     if self.wheel_constrain else 0.
        cost_stage += \
            +2.0*sumsqr(self.state[3:6,1:]-self.ref_traj[3:6,1:]) \
                if self.following_mode else 2.

        if self.var_dt:
            # alpha_var_dt = 0.1
            alpha_var_T = 2. if self.following_mode else 2.
            # cost_stage += alpha_var_dt * sumsqr(self.var_dts - self.dt)
            cost_stage += alpha_var_T * sumsqr(self.var_dts)
        # Take special care with the final state
        if self.free_end:
            alpha_terminal_xy = 10.0
            alpha_terminal_yaw = 10.0  
            # really care about final heading to help with lane changes
            alpha_terminal_v = 10.0 if self.following_mode else 15.
            cost_stage += (
                alpha_terminal_xy * sumsqr(self.ref_traj[:2, -1] 
                       - vertcat(self.position_x[:,-1], self.position_y[:,-1])) 
                + alpha_terminal_yaw * sumsqr(
                        -cos(self.yaw[:,-1] - self.ref_traj[2, -1])+1)
                # + alpha_terminal_yaw * sumsqr(
                #         self.yaw[:,-1] - self.ref_traj[2, -1])
                + alpha_terminal_v
                    * (sumsqr(self.d_x[:,-1]-self.ref_traj[3,-1])
                        +sumsqr(self.d_y[:,-1]-self.ref_traj[4,-1])
                        +sumsqr(self.d_yaw[:,-1]-self.ref_traj[5,-1]))
            )
        ## the weight should be carefully tuned, 1.0 is a good start
        # cost_stage += 0.*sumsqr(self.slack_wheel_dir-self.wheel_dir) \
        #     if self.wheel_constrain else 0.
        
        if self.mapui is not None:
            occ_list = self.occ_lists
            for i in range(1, self.trajectory_len+1):
                for c in VehicleConst.cccl:
                    dis_error = sqrt(
                        (occ_list[:,0]-(self.position_x[:,i]
                                        +cos(self.yaw[:,i])*c))**2 
                        +(occ_list[:,1]-(self.position_y[:,i]
                                         +sin(self.yaw[:,i])*c))**2
                                         ) \
                        -(VehicleConst.occ_r+VehicleConst.ccr)
                    cost_stage += 1.*sumsqr(exp(-dis_error**3))
                        

        self._optimizer.minimize(cost_stage) #+ self.trajectory_len / 4.0 *

    def _set_initial_guess(self, 
                           initial_guess: Sequence[Pose], 
                           initial_dts: Sequence[float]) -> None:
        """Set a warm-start for the solver based on the reference trajectory."""
        self._check_inputs(initial_guess[0,:], initial_guess)

        # Initialize state guess based on reference
        self._optimizer.set_initial(
            self.state[:self.init_guess_dim,:],
            DM(initial_guess[:,:self.init_guess_dim]).T)  # (x, y, yaw)
        if initial_dts is not None and self.var_dt:
            self._optimizer.set_initial(
                self.var_dts[0,:], DM(initial_dts))
        if initial_dts is None and self.var_dt:
            self._optimizer.set_initial(
                self.var_dts[0,:], DM([self.dt]*self.trajectory_len))
        # Initialize control guess based on reference TODO
        # if self.wheel_constrain:
        #     for i in range(self.trajectory_len+1):
        #         global_wheel_vx, global_wheel_vy = \
        #             global_dxdydyaw2wheelstate(
        #             initial_guess[i,2], 
        #             initial_guess[i,3], 
        #             initial_guess[i,4], 
        #             initial_guess[i,5])
        #         relative_wheel_v = RotationMatrix(-initial_guess[i,2])@\
        #             horzcat(global_wheel_vx,global_wheel_vy).T
        #         wheel_dir = \
        #             relative_wheel_v[0,:]\
        #                 /fabs(relative_wheel_v[0,:])
        #         self._optimizer.set_value(
        #             self.slack_wheel_dir[:,i], wheel_dir)
                # self._optimizer.set_initial(
                #     self.slack_wheel_dir[:,i], wheel_dir)

        # I think initializing the controls would be quite noisy, so using default zero init

    def _check_inputs(self, 
                      x_curr: Sequence[float], 
                      reference_trajectory: Sequence[Pose]) -> None:
        """Raise ValueError if inputs are not of proper size."""
        if len(x_curr) != self.nx:
            raise ValueError(f"x_curr length {len(x_curr)} must be equal to state dim {self.nx}")

        if len(reference_trajectory) != self.trajectory_len + 1:
            raise ValueError(
                f"reference traj length {len(reference_trajectory)} must be equal to {self.trajectory_len + 1}"
            )
        
    def _preprocess_map(self, initial_guess: Sequence[Pose]):
        """Preprocess the map for the optimization."""
        # self.occ_lists = self.mapui.get_occ_in_range(initial_guess[:,:2], 25)
        self.occ_lists = self.mapui.get_occ_in_range_traj(
                                        initial_guess[:,:2], 
                                        VehicleConst.OCC_RANGE)

# if __name__=='__main__':
import matplotlib.pyplot as plt

def test_dt(points = None, vs = None, dts = None, VISULIZE=False, 
            mapui=None, result_dict:dict=None, no_smooth=False):
    # VISULIZE = True
    wheel_constrain = True and not VehicleConst.no_wheel_constrain_mode
    dt = VehicleConst.dt
    wheel_positions = np.array(VehicleConst.wheel_positions).T
    T = VehicleConst.T if points is None else points.shape[0]-1
    if T <= 0:
        print(f"T should be greater than 0, now it is {T}, optimization skiped")
        return
    optimizer = AWSGlobalMassPointOptimizer(T, dt, 
                                            wheel_constrain=wheel_constrain,
                                            current_index=0,
                                            free_end=True,
                                            var_dt=True and dts is not None,
                                            following_mode=True,
                                            simplify_model=True,
                                            mapui=mapui,
                                            polyline=False)
                                            
    # optimizer = AWSCurvatureOptimizer(T, dt, wheel_constrain=wheel_constrain)
    dim = optimizer.nx
    init_state = np.zeros([T+1, dim])

    if points.shape[0]!=T+1:
        raise ValueError(f"points shape[0] must be {T+1}")
    init_state[:, :3] = points
    for i in range(T+1):
        if optimizer.name=="AWSGlobalMassPointOptimizer":
            global_vs = np_RotationMatrix(points[i,2])@vs[i,:2]
            init_state[i,3:5] = global_vs
            init_state[i,5] = vs[i,2]
            if not optimizer.wheel_constrain:
                continue
            wheel_vs = global_vs[:, None] \
                        +np_dRotationMatrix(points[i,2],vs[i,2]) \
                        @wheel_positions
            init_state[i,6:10] = wheel_vs[0,:]
            init_state[i,10:14] = wheel_vs[1,:]
    print("init_state has NaN: ",np.isnan(init_state).any())
    print("init_state[0]: ", init_state[0])

    ### optimizing
    if no_smooth:
        result_traj = init_state
        control = np.zeros([T, 4])
    else:
        result_traj, control = traj_smooth(init_state, optimizer, dts, result_dict)
    opt_dts = control[:,-1:]

    if VISULIZE:
        from traj_generator import min_jerk
        # prepare data
        cm_per_inch = 1/2.54
        # fig.suptitle(f'{optimizer.name} smoother', fontsize=16)
        opt_ts = np.arange(0, T)*VehicleConst.dt \
                if not optimizer.var_dt else np.cumsum(opt_dts)
        ori_ts = np.arange(0, T)*VehicleConst.dt \
                if not optimizer.var_dt else np.cumsum(dts)
        result_traj = np.concatenate([init_state[0:1,:], 
                                     result_traj], axis=0)
        opt_ts = np.concatenate([[0], opt_ts], axis=0)
        ori_ts = np.concatenate([[0], ori_ts], axis=0)

        def is_feasible(r, phi):
            r = np.array([[-r*np.cos(phi), -r*np.sin(phi)]]).T
            w = np.array(VehicleConst.wheel_positions).T
            rw = r+w 
            s_limits = np.array(VehicleConst.wheel_steer_limits)
            u_s_lim = s_limits[:,1]
            l_s_lim = s_limits[:,0]
            ulim = u_s_lim+np.pi/2
            llim = l_s_lim+np.pi/2
            ulim_v = np.array([np.cos(ulim),np.sin(ulim)])
            llim_v = np.array([np.cos(llim),np.sin(llim)])
            return ((rw[0,:]*ulim_v[1]-rw[1,:]*ulim_v[0]) \
                *(rw[0,:]*llim_v[1]-rw[1,:]*llim_v[0]) < 0).all()

        
        def get_feasible_uv(u,v):
            # u is phi
            # v is r
            f_u = []
            f_v = []
            r_v = []
            for ui in u:
                for vi in v:
                    phi = ui
                    r = np.tan(vi/2)
                    if is_feasible(r, phi):
                        f_u.append(ui)
                        f_v.append(vi)
                        r_v.append([r*cos(phi), r*sin(phi)])
            return np.array(f_u), np.array(f_v), np.array(r_v)
        
        u = np.linspace(0, 2 * np.pi, 150)
        v = np.linspace(0, np.pi, 150)
        

        # 球面坐标转笛卡尔坐标
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        fu, fv, rv = get_feasible_uv(u, v)
        region_enlarge_ratio = 1
        fx, fy, fz = region_enlarge_ratio*np.cos(fu)*np.sin(fv), \
                    region_enlarge_ratio*np.sin(fu)*np.sin(fv), \
                    -np.cos(fv)*region_enlarge_ratio
        

        fig = plt.figure(figsize=(8.5, 6),dpi=300)
        ax1 = plt.subplot(111)
        ax1.set_title('Trajectorys')
        # ax1.plot(points[:,0], points[:,1], 'ro', label='waypoints')
        # ax1.scatter(points[:,0], points[:,1], c=ori_ts, cmap='viridis', label='waypoints',s=3)
        # ax1.plot(init_state[:,0], init_state[:,1], 'cyan',label='initial guess')
        # ax1.plot(result_traj[:,0], result_traj[:,1], 'b-',label='smoothed trajectory')
        for i in range(T+1):
            label = 'initial trajectory' if i==0 else None
            # heading direction
            ax1.arrow(init_state[i,0], init_state[i,1], 
                    0.5*np.cos(init_state[i,2]), 
                    0.5*np.sin(init_state[i,2]), 
                    width=0.1, color='cyan',alpha=0.3)
            for w in range(len(VehicleConst.wheel_positions)):
                w_label = 'initial wheel velocity' if w==0 and i==0 else None
                # original ones
                ori_w_start = init_state[i,:2] \
                    +np_RotationMatrix(init_state[i,2])@wheel_positions[:,w]
                ori_w_end = np.array([init_state[i,6+w],init_state[i,10+w]])
                ori_w_end = ori_w_end/np.linalg.norm(ori_w_end)*0.5
                ax1.arrow(ori_w_start[0], ori_w_start[1],
                        ori_w_end[0], ori_w_end[1],
                        width=0.01, color='cyan',
                        label = w_label)
            rect = patches.Rectangle( 
                (init_state[i,0]-VehicleConst.half_l, 
                 init_state[i,1]-VehicleConst.half_w),
                VehicleConst.WHEEL_BASE, VehicleConst.WHEEL_WHIDTH,
                angle = init_state[i,2]/math.pi*180, 
                rotation_point='center',
                linewidth=0, 
                # edgecolor='k', 
                facecolor='cyan',
                alpha=0.2,
                label= label
                )
            ax1.add_patch(rect)


        for i in range(T+1):
            ax1.arrow(result_traj[i,0], result_traj[i,1], 
                    0.5*np.cos(result_traj[i,2]), 
                    0.5*np.sin(result_traj[i,2]), 
                    width=0.1, color='b',alpha=0.3)
            label = 'smoothed trajectory' if i==0 else None
            for w in range(len(VehicleConst.wheel_positions)):
                w_label = 'smoothed wheel velocity' if w==0 and i==0 else None
                w_start = result_traj[i,:2] \
                    +np_RotationMatrix(result_traj[i,2])@wheel_positions[:,w]
                w_end = np.array([result_traj[i,6+w],result_traj[i,10+w]])
                w_end = w_end/np.linalg.norm(w_end)*0.5
                ax1.arrow(w_start[0], w_start[1],
                        w_end[0], w_end[1],
                        width=0.02, color='b',
                        label=w_label)
            rect = patches.Rectangle( 
                (result_traj[i,0]-VehicleConst.half_l, 
                 result_traj[i,1]-VehicleConst.half_w),
                VehicleConst.WHEEL_BASE, VehicleConst.WHEEL_WHIDTH,
                angle = result_traj[i,2]/math.pi*180, 
                rotation_point='center',
                linewidth=0, 
                # edgecolor='k', 
                facecolor='b',
                alpha=0.2,
                label = label,
                )
            ax1.add_patch(rect)
            
        ax1.axis('equal')
        ax1.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(f'{optimizer.name} smoother_1.png')
        plt.savefig(f'{optimizer.name} smoother_1.pdf')
        


        plt.cla()
        plt.figure(figsize=(8.5, 6),dpi=300)
        ax2 = plt.subplot(111)
        ax2.set_title('Velocities')
        ax2.plot(ori_ts, init_state[:,3], 'r--', label='original v_x', 
                 alpha=0.5)
        ax2.plot(ori_ts, init_state[:,4], 'b--', label='original v_y', 
                 alpha=0.5)
        ax2.plot(ori_ts, init_state[:,5], 'g--', label='original v_yaw',
                 alpha=0.5)
        ax2.plot(opt_ts, result_traj[:,3], 'r-', label='smoothed v_x')
        ax2.plot(opt_ts, result_traj[:,4], 'b-', label='smoothed v_y')
        ax2.plot(opt_ts, result_traj[:,5], 'g-', label='smoothed v_yaw')

        if optimizer.var_dt:
            ax2.plot(opt_ts[1:], opt_dts, 'k-', label='var dt')
            ax2.plot(ori_ts[1:], dts, 'k--', label='var dt')
        ax2.axis('equal')
        ax2.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(f'{optimizer.name} smoother_2.png')
        plt.savefig(f'{optimizer.name} smoother_2.pdf')



        ###### AX3
        plt.cla()
        plt.figure(figsize=(4.5, 4.5),dpi=300)
        ax3 = plt.subplot(111)
        ax3.set_title('Smoothed ICM')
        # if wheel_constrain:
        # :                +------------------+
        # :                |                  |
        # :              height               |
        # :                |                  |
        # :               (xy)---- width -----+
        # original ones
        if optimizer.name=="AWSGlobalMassPointOptimizer":
            radius = []
            for i in range(T):
                radius.append(
                    np_dRotationMatrix(init_state[i,2], init_state[i,5]).T
                    @init_state[i,3:5] 
                    / (init_state[i,5]**2))
            radius = np.array(radius)
            radius = -radius
        original_radius_scale = np.linalg.norm(radius,2,axis=-1)
        original_radius_sphere = 2*np.arctan(original_radius_scale)
        original_psi_sphere = np.arctan2(radius[:,1], radius[:,0])
        # ax3.plot(radius[:,0], radius[:,1], 'b-', label='original ICM')
        ax3.scatter(radius[:,0], radius[:,1], 
                    c=ori_ts[1:], 
                    cmap='winter', 
                    # color = 'r', 
                    # s=5,
                    marker='+', 
                    label='initial ICM centers ',
                    )
        # smoothed ones
        rect = patches.Rectangle((-VehicleConst.half_l, -VehicleConst.half_w),
                                  VehicleConst.WHEEL_BASE, 
                                  VehicleConst.WHEEL_WHIDTH, 
                                  linewidth=0, 
                                  edgecolor='cyan', 
                                  facecolor='cyan',
                                  alpha = 0.5,
                                  label = 'vehicle footprint'
                                  )
        ax3.add_patch(rect)
        if optimizer.name=="AWSGlobalMassPointOptimizer":
            radius = []
            for i in range(T):
                radius.append(np_dRotationMatrix(
                    result_traj[i,2], result_traj[i,5]).T
                    @result_traj[i,3:5] 
                    / (result_traj[i,5]**2))
            radius = np.array(radius)
            radius = -radius
        smoothed_radius_scale = np.linalg.norm(radius,2,axis=-1)
        smoothed_radius_sphere = 2*np.arctan(smoothed_radius_scale)
        smoothed_psi_sphere = np.arctan2(radius[:,1], radius[:,0])
        # ax3.plot(radius[:,0], radius[:,1], 'g-', 
        #          label='smoothed ICM')
        im3s = ax3.scatter(radius[:,0], radius[:,1], 
                    c=opt_ts[1:], cmap='winter',
                    s = 5, 
                    label='smoothed ICM centers'
                    )


        ax3.axis('equal')
        ax3.set_xscale('symlog', linthresh=1)
        ax3.set_yscale('symlog', linthresh=1)
        ax3.set_xlim(-1e3,1e3)
        ax3.set_ylim(-1e3,1e3)
        ax3.legend(loc="upper left")

        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im3s, cax = cax, label='time (s)')
        


        ax3.scatter(rv[:, 0], 
                    rv[:, 1], 
                    c='r', s=1, alpha=.1,
                    zorder = 1,
                    label='feasible region',
                    marker=',',
                    )
        plt.tight_layout()
        plt.savefig(f'{optimizer.name} smoother_3.png')
        plt.savefig(f'{optimizer.name} smoother_3.pdf')

        #### AX4
        plt.cla()
        plt.figure(figsize=(4.5, 4.5),dpi=300)
        ax4 = plt.subplot(111,projection='3d')
        ax4.set_title('Smoothed ICM in Sphere coordinate')
        # 设置球体参数

        ax4.plot_wireframe(x, y, z, 
                           rcount=30, ccount=30, 
                           color='k', alpha=1., linewidth=0.05)


        ax4.scatter(fx, fy, fz,
                    c='r', 
                    s=1,
                    alpha=.2,
                    label='feasible region',
                    marker=',',
                    )
        
        smooth_enlarge_ratio = 1.03
        # ax4.plot_surface(fx, fy, fz, 
        #                  rstride=8, cstride=8, 
        #                  color='grey', alpha=0.5)
        ax4.scatter(smooth_enlarge_ratio*np.sin(smoothed_radius_sphere)
                    *np.cos(smoothed_psi_sphere),
                    smooth_enlarge_ratio*np.sin(smoothed_radius_sphere)
                    *np.sin(smoothed_psi_sphere), 
                    -np.cos(smoothed_radius_sphere)*smooth_enlarge_ratio, 
                    c=opt_ts[1:], 
                    cmap='winter', 
                    label='smoothed ICM centers',
                    s = 15,
                    alpha=1.)
        # ax4.plot(np.sin(smoothed_radius_sphere)
        #             *np.cos(smoothed_psi_sphere),
        #             np.sin(smoothed_radius_sphere)
        #             *np.sin(smoothed_psi_sphere), 
        #             -np.cos(smoothed_radius_sphere), 
        #             # c=opt_ts[1:], 
        #             # cmap='winter', 
        #             label='smoothed ICM track',
        #             # s = 5,
        #             color='y',
        #             alpha=1.
        #             )

        ax4.scatter(smooth_enlarge_ratio*np.sin(original_radius_sphere)
                    *np.cos(original_psi_sphere),
                    smooth_enlarge_ratio*np.sin(original_radius_sphere)
                    *np.sin(original_psi_sphere), 
                    -np.cos(original_radius_sphere)*smooth_enlarge_ratio, 
                    c=ori_ts[1:], 
                    cmap='winter', 
                    label='inital ICM centers',
                    alpha=1.,
                    marker = '+',
                    s=20
                    )
        ax4.view_init(elev=60, azim=60)
        ax4.set_xlim(-0.5,0.5)
        ax4.set_ylim(-0.5,0.5)
        ax4.set_zlim(0,1)
        ax4.axis('equal')

        ax4.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(f'{optimizer.name} smoother_4.png')
        plt.savefig(f'{optimizer.name} smoother_4.pdf')

        #### AX5
        ## use relative vs
        r_result_traj = result_traj.copy()
        for i in range(T+1):
            vehicle_vs = np_RotationMatrix(
                -result_traj[i,2])@result_traj[i,3:5].T
            r_result_traj[i,3:5] = vehicle_vs.T
            wheel_vs = np.array([result_traj[i,6:10],result_traj[i,10:14]])
            relative_wheel_vs = np_RotationMatrix(
                -result_traj[i,2])@wheel_vs
            r_result_traj[i,6:10] = relative_wheel_vs[0,:]
            r_result_traj[i,10:14] = relative_wheel_vs[1,:]
        result_traj = r_result_traj
        plt.cla()
        plt.figure(figsize=(6, 7),dpi=300)
        ax5_0 = plt.subplot(511)
        ax5_0.set_title('Optimized Relative Velocitys')
        ax5_0.hlines(0,0,opt_ts[-1],colors='k',linestyles='dashed')
        ax5_0.plot(opt_ts, result_traj[:,3], 'r-', label='vx')
        ax5_0.plot(opt_ts, result_traj[:,4], 'b-', label='vy')
        ax5_0.plot(opt_ts, result_traj[:,5], 'g-', label='vyaw')
        ax5_0.legend(loc="upper left")
        ax5_1 = plt.subplot(512)
        # ax5_1.set_title('Smoothed Wheel Velocitys')
        ax5_1.hlines(0,0,opt_ts[-1],colors='k',linestyles='dashed')
        ax5_1.plot(opt_ts, result_traj[:,6], 'r-', label='wheel 1 vx')
        ax5_1.plot(opt_ts, result_traj[:,10], 'b-', label='wheel 1 vy')
        ax5_1.plot(opt_ts, (result_traj[:,6]**2+result_traj[:,10]**2)**0.5, 
                   'g-', label='wheel 1 v')
        ax5_1.legend(loc="upper left")
        ax5_2 = plt.subplot(513)
        ax5_2.hlines(0,0,opt_ts[-1],colors='k',linestyles='dashed')
        ax5_2.plot(opt_ts, result_traj[:,7], 'r-', label='wheel 2 vx')
        ax5_2.plot(opt_ts, result_traj[:,11], 'b-', label='wheel 2 vy')
        ax5_2.plot(opt_ts, (result_traj[:,7]**2+result_traj[:,11]**2)**0.5, 
                   'g-', label='wheel 2 v')
        ax5_2.legend(loc="upper left")

        ax5_3 = plt.subplot(514)
        ax5_3.hlines(0,0,opt_ts[-1],colors='k',linestyles='dashed')
        ax5_3.plot(opt_ts, result_traj[:,8], 'r-', label='wheel 3 vx')
        ax5_3.plot(opt_ts, result_traj[:,12], 'b-', label='wheel 3 vy')
        ax5_3.plot(opt_ts, (result_traj[:,8]**2+result_traj[:,12]**2)**0.5, 
                   'g-', label='wheel 3 v')
        ax5_3.legend(loc="upper left")
        ax5_4 = plt.subplot(515)
        ax5_4.hlines(0,0,opt_ts[-1],colors='k',linestyles='dashed')
        ax5_4.plot(opt_ts, result_traj[:,9], 'r-', label='wheel 4 vx')
        ax5_4.plot(opt_ts, result_traj[:,13], 'b-', label='wheel 4 vy')
        ax5_4.plot(opt_ts, (result_traj[:,9]**2+result_traj[:,13]**2)**0.5, 
                   'g-', label='wheel 4 v')
        ax5_4.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(f'{optimizer.name} smoother_5.png')
        plt.savefig(f'{optimizer.name} smoother_5.pdf')
    del optimizer
    return result_traj, control

