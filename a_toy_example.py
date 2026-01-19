import math
import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, TypeVar
import os
import sys
print(os.path.dirname(__file__))
sys.path.append(os.path.dirname(__file__))
from matplotlib import patches
from casadi import (DM, Opti, OptiSol, cos, diff, sin, 
                    sumsqr, vertcat, pi, fmod, norm_2, 
                    atan2, if_else, tan , horzcat, log, 
                    fabs, acos, sqrt, exp, cumsum, atan)
Pose = Tuple[float, float, float]  # (x, y, yaw)
class VehicleConst:
    T = 20
    dt = .3

    WHEEL_BASE = 2.520
    WHEEL_WHIDTH = 1.100
    WHEEL_RADIUS = 0.25

    MAX_V = 2.5
    MAX_dyaw = math.pi/2

    MAX_ACC = 2.5
    MAX_dACC = 2.5
    MAX_ddyaw = math.pi/4

    MAX_STEER_DEG = 75
    MAX_STEER = math.pi/180*MAX_STEER_DEG
    MAX_dSTEER = math.pi/2

    half_l = WHEEL_BASE/2
    half_w = WHEEL_WHIDTH/2
    wheel_frames = ['FLaxis','FRaxis','RLaxis','RRaxis']
    _control_center_offset = -half_l
    # _control_center_offset = 0.
    wheel_positions = [[half_l, half_w],
                       [half_l, -half_w],
                       [_control_center_offset, half_w],
                       [_control_center_offset, -half_w]]
    # anticlockwise feasible steer limits
    _front = MAX_STEER
    _rear = MAX_STEER
    # _rear = 1.e-3
    wheel_steer_limits = [[-_front,_front],
                          [-_front,_front],
                          [-_rear,_rear],
                          [-_rear,_rear]]
    wheel_num = len(wheel_positions)
def np_RotationMatrix(theta):
    return np.array([[np.cos(theta),-np.sin(theta)],
                     [np.sin(theta),np.cos(theta)]])

def np_dRotationMatrix(theta, dtheta):
    return np.array([[-np.sin(theta),-np.cos(theta)],
                     [np.cos(theta),-np.sin(theta)]])*(dtheta)

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
# 旋转矩阵的时间导数
def dRotationMatrix(theta, dtheta):
    return vertcat(horzcat(-sin(theta),-cos(theta)),
                   horzcat(cos(theta),-sin(theta)))*dtheta

def ddRotationMatrix(theta, dtheta, ddtheta):
    return vertcat(horzcat(-cos(theta),sin(theta)), horzcat(-sin(theta),-cos(theta)))*dtheta**2 \
        + vertcat(horzcat(-sin(theta),-cos(theta)), horzcat(cos(theta),-sin(theta)))*ddtheta
# 用于将车辆质心运动（dx, dy, dyaw）转换为每个车轮的速度向量的函数
def global_dxdydyaw2wheelstate(yaw,dx,dy,dyaw):
    # 四个轮子
    wheel_positions = VehicleConst.wheel_positions
    wheel_num = len(VehicleConst.wheel_positions)
    wheel_v = [None]*wheel_num
    for i in range(wheel_num):
        wheel_v[i] = vertcat(dx,dy) + dRotationMatrix(yaw, dyaw) @vertcat(wheel_positions[i][0],wheel_positions[i][1])
    wheel_v = horzcat(*wheel_v)
    return wheel_v[0, :].T, wheel_v[1, :].T
# 用于将车辆质心运动（ddx, ddy, ddyaw）转换为每个车轮的加速度向量的函数
def global_d_dxdydyaw2wheelstate(yaw, dyaw, ddx, ddy, ddyaw):
    wheel_positions = VehicleConst.wheel_positions
    wheel_num = len(VehicleConst.wheel_positions)
    d_wheel_v = [None]*wheel_num # horzcat(d_wheel_vx, d_wheel_vy).T
    for i in range(wheel_num):
        d_wheel_v[i] = vertcat(ddx,ddy) + ddRotationMatrix(yaw, dyaw, ddyaw)@vertcat(wheel_positions[i][0],wheel_positions[i][1])
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
        ub_dir = -r_r_w[0,:]*(indicator*ulim_vec)[1,:] + r_r_w[1,:]*(indicator*ulim_vec)[0,:]
        lb_dir = -r_r_w[0,:]*(indicator*llim_vec)[1,:] + r_r_w[1,:]*(indicator*llim_vec)[0,:]
        return_list+=[ub_dir*lb_dir<=0]
    return return_list

class AWSGlobalMassPointOptimizer:
    def __init__(self, trajectory_len: int, 
                 dt: float, 
                 wheel_constrain=True, 
                 current_index=0, 
                 free_end=True, 
                 var_dt=False,
                 following_mode=False,
                 simplify_model=False,
                 constant_interval=False):
        self.name = "AWSGlobalMassPointOptimizer"
        self.discrete = []
        self.dt = dt
        self.free_end = free_end
        self.var_dt = var_dt
        self.constant_interval = constant_interval
        self.trajectory_len = trajectory_len
        self.wheel_constrain = wheel_constrain
        self.current_index = current_index
        # Use a array of dts to make it compatible to situations with varying dts across different time steps.
        self.init_guess_dim = 6 if not self.wheel_constrain else 6+8
        self.nx = 6 if not self.wheel_constrain else 6+8 # state dim
        self.nu = 3  # control dim
        self.following_mode = following_mode
        self.simplify_model = simplify_model
        # self.discrete = []
    # 初始化求解
    def _init_optimization(self) -> None:
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


    def set_reference_trajectory(self, initial_guess: Sequence[Pose], dts: Sequence[float]) -> None:
        """
        Set the reference trajectory that the smoother is trying to loosely track.

        :param x_curr: current state of size nx (x, y, yaw, speed)
        :param reference_trajectory: N+1 x 3 reference, where the second dim is for (x, y, yaw)
        """
        self._check_inputs(initial_guess[0,:], initial_guess)

        self.trajectory_len = len(initial_guess)-1
        self._init_optimization()

        self._optimizer.set_value(self.x_curr, DM(initial_guess[self.current_index,:]))
        self._optimizer.set_value(self.ref_traj, DM(initial_guess[:,:self.nx]).T)
        if not self.free_end:
            self._optimizer.set_value(self.x_final, DM(initial_guess[-1,:]))
        self._set_initial_guess(initial_guess, dts)

    def solve(self) -> OptiSol:
        return self._optimizer.solve()

    def _create_decision_variables(self) -> None:
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
        self.r_accel_x = cos(-self.yaw[:,:-1])*self.accel_x - sin(-self.yaw[:,:-1])*self.accel_y
        self.r_accel_y = sin(-self.yaw[:,:-1])*self.accel_x + cos(-self.yaw[:,:-1])*self.accel_y
        # 得到轮子的相关数据
        if self.wheel_constrain:
            wheel_vxs = []
            wheel_vys = []
            wheel_axs = []
            wheel_ays = []
            if self.simplify_model:
                for i in range(self.trajectory_len+1):
                    # 用于将车辆质心运动（dx, dy, dyaw）转换为每个车轮的速度向量的函数
                    wheel_vx, wheel_vy = global_dxdydyaw2wheelstate(self.yaw[:,i], self.d_x[:,i], self.d_y[:,i], self.d_yaw[:,i])
                    wheel_vxs.append(wheel_vx)
                    wheel_vys.append(wheel_vy)
                for i in range(self.trajectory_len):
                    # 用于将车辆质心运动（ddx, ddy, ddyaw）转换为每个车轮的加速度向量的函数
                    wheel_ax, wheel_ay = global_d_dxdydyaw2wheelstate(self.yaw[:,i], self.d_yaw[:,i], self.accel_x[:,i], self.accel_y[:,i], self.dd_yaw[:,i])
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
            r_r_list = [] # 世界坐标系下
            wheel_dir_list = []
            r_wheel_vx = []
            r_wheel_vy = []
            for i in range(self.trajectory_len+1):
                # positive value is the vector while negative is the relative position 
                r_r = dRotationMatrix(self.yaw[i], self.d_yaw[i]) @ (vertcat(self.d_x[i], self.d_y[i])/(self.d_yaw[i]**2))
                rotat_m = RotationMatrix(-self.yaw[i])
                r_wheel_v = rotat_m @ vertcat(self.wheel_vx[:,i].T, self.wheel_vy[:,i].T)
                # self.slack_wheel_dir[:,i] = (1-2/1+exp(-5*r_wheel_v[0,:])).T
                r_r_list.append(r_r)
                wheel_dir_list.append(r_wheel_v[0,:] / fabs(r_wheel_v[0,:]))
                r_wheel_vx.append(r_wheel_v[0,:])
                r_wheel_vy.append(r_wheel_v[1,:])
            self.r_r = horzcat(*r_r_list)
            self.wheel_dir = vertcat(*wheel_dir_list).T
            self.r_wheel_vx = vertcat(*r_wheel_vx).T
            self.r_wheel_vy = vertcat(*r_wheel_vy).T
            # self.wheel_dir = self._optimizer.variable(4, self.trajectory_len+1)
            # self.discrete += [True]*(4*(self.trajectory_len+1))
            # self.slack_wheel_dir =  self._optimizer.variable(4, self.trajectory_len+1)
            # self.slack_wheel_dir = self._optimizer.parameter(4, self.trajectory_len+1)


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
        self.ref_traj = self._optimizer.parameter(self.nx, self.trajectory_len+1)
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

        def process(x: Sequence[float], u: Sequence[float]) -> Any:
            """Process for state propagation."""
            yaw = x[2]
            vx = x[3]
            vy = x[4]
            dyaw = x[5]
            if self.wheel_constrain and not self.simplify_model:
                wheel_ddx, wheel_ddy = global_d_dxdydyaw2wheelstate(yaw, dyaw, u[0], u[1], u[2])
                return vertcat(vx, vy, dyaw, u, wheel_ddx, wheel_ddy)
            return vertcat(vx, vy, dyaw, u)

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

        #     next_rot_vec = self.rot_vecs[:, k] + dt / 6 * (rot_vec_k1 + 2 * rot_vec_k2 + 2 * rot_vec_k3 +  rot_vec_k4)
        #     next_d_rot_vec = self.d_rot_vecs[:, k] + dt / 6 * (rot_d_vec_k1 + 2 * rot_d_vec_k2 + 2 * rot_d_vec_k3 + rot_d_vec_k4)
        #     self._optimizer.subject_to(self.rot_vecs[:, k + 1] == next_rot_vec[:])
        #     self._optimizer.subject_to(self.d_rot_vecs[:, k + 1] == next_d_rot_vec[:])
            

    def _set_control_constraints(self) -> None:
        accel_limit = VehicleConst.MAX_ACC # m/s^2
        self._optimizer.subject_to(self._optimizer.bounded(0, self.accel_x**2+self.accel_y**2, accel_limit**2))
        ddyaw_limit = VehicleConst.MAX_ddyaw # rad/s^2
        self._optimizer.subject_to(self._optimizer.bounded(-ddyaw_limit, self.dd_yaw, ddyaw_limit))
        
    # 核心约束在这边
    def _set_state_constraints(self) -> None:
        # Constrain the current time -- NOT start of history
        # initial boundary condition
        self._optimizer.subject_to(self.state[:, self.current_index] == self.x_curr)
        if self.var_dt:
            self._optimizer.subject_to(self.var_dts>=0.1*self.dt)
            self._optimizer.subject_to(self.var_dts<3.0*self.dt)
        if not self.free_end:
            self._optimizer.subject_to(self.state[:3, -1] == self.x_final[:3])
            self._optimizer.subject_to(self.state[3, -1] == 0)
            self._optimizer.subject_to(self.state[4:6, -1] == 0.)
        if not self.wheel_constrain:
            max_speed = VehicleConst.MAX_V # m/s
            self._optimizer.subject_to(self.v**2<=max_speed**2)
            max_yaw_rate = VehicleConst.MAX_dyaw # rad/s
            self._optimizer.subject_to(self.d_yaw**2 <= max_yaw_rate**2)

        if self.wheel_constrain:
            for i in range(VehicleConst.wheel_num):
                self._optimizer.subject_to(self.wheel_v[i,:]<=VehicleConst.MAX_V)
                # self._optimizer.subject_to(self.wheel_a[i,:]<=VehicleConst.MAX_ACC)
            wheel_steer_limits = np.array(VehicleConst.wheel_steer_limits).T
            ub_x = cos(wheel_steer_limits[1,:])
            ub_y = sin(wheel_steer_limits[1,:])
            lb_x = cos(wheel_steer_limits[0,:])
            lb_y = sin(wheel_steer_limits[0,:])
            for t in range(self.trajectory_len):
                self._optimizer.subject_to((self.r_wheel_vx[:,t+1] * ub_y - self.r_wheel_vy[:,t+1] * ub_x)*(self.r_wheel_vx[:,t+1] * lb_y - self.r_wheel_vy[:,t+1] * lb_x)<0)
                ### steering speed constraints
                # r_wheel_v_t = RotationMatrix(-self.yaw[:,t]) @ vertcat(self.wheel_vx[:,t].T, self.wheel_vy[:,t].T)
                r_wheel_vx_t = self.r_wheel_vx[:,t].T
                r_wheel_vy_t = self.r_wheel_vy[:,t].T
                # r_wheel_v_tp1 = RotationMatrix(-self.yaw[:,t+1]) @ vertcat(self.wheel_vx[:,t+1].T, self.wheel_vy[:,t+1].T)
                r_wheel_vx_tp1 = self.r_wheel_vx[:,t+1].T
                r_wheel_vy_tp1 = self.r_wheel_vy[:,t+1].T
                
                # steer_diff = atan(r_wheel_v_t[1,:]/r_wheel_v_t[0,:]) -atan(r_wheel_v_tp1[1,:]/r_wheel_v_tp1[0,:])
                # if not self.var_dt:
                #     self._optimizer.subject_to(steer_diff < VehicleConst.MAX_dSTEER*self.dt)
                #     continue
                # if self.var_dt:
                #     self._optimizer.subject_to(steer_diff < VehicleConst.MAX_dSTEER*self.var_dts[:,t])
                #     continue
                inner_product = self.wheel_dir[:, t].T * r_wheel_vx_t * self.wheel_dir[:, t+1].T * r_wheel_vx_tp1+self.wheel_dir[:, t].T * r_wheel_vy_t * self.wheel_dir[:, t+1].T * r_wheel_vy_tp1
                if not self.var_dt:
                    self._optimizer.subject_to(inner_product.T > cos(VehicleConst.MAX_dSTEER*self.dt) * self.wheel_v[:,t]*self.wheel_v[:,t+1])
                if self.var_dt:
                    # steer = atan2(self.r_wheel_vy,self.r_wheel_vx)
                    # self._optimizer.subject_to(cos(steer[:,t+1]*self.wheel_dir[:, t]-steer[:,t]*self.wheel_dir[:, t+1]) > cos(VehicleConst.MAX_dSTEER*self.var_dts[:,t]))

                    self._optimizer.subject_to(inner_product.T>cos(VehicleConst.MAX_dSTEER*self.var_dts[:,t])*self.wheel_v[:,t]*self.wheel_v[:,t+1])
                    if t>=1 and self.constant_interval:
                        self._optimizer.subject_to(self.var_dts[:,t]==self.var_dts[:,t-1])
                
                ### steering phase constrain, wheel_v around 0 when changing direction
                ### mode priors parameters needed, body v 0 can be more promising
                # if not self.following_mode:
                #     self._optimizer.subject_to(self.wheel_v[:,t]**2 <= (self.wheel_dir[:,t]+self.wheel_dir[:,t+1])**2/4 * VehicleConst.MAX_V)
                #     self._optimizer.subject_to(self.wheel_v[:,t+1]**2 <=(self.wheel_dir[:,t]+self.wheel_dir[:,t+1])**2/4* VehicleConst.MAX_V)
            # TODO explore the functionality of this term
            # self._optimizer.subject_to(self.slack_wheel_dir==self.wheel_dir)

    def _set_objective(self) -> None: # TODO
        """Set the objective function. Use care when modifying these weights."""
        # Follow reference, minimize control rates and absolute inputs
        alpha_xy = 0.5 if not self.following_mode else 0.5
        alpha_yaw = 2.5 if not self.following_mode else 2.5
        # speed_ratio = 0. if not self.following_mode else .0

        cost_stage = (0.
            + alpha_xy * sumsqr(self.ref_traj[:2, 1:] - self.state[:2, 1:])
            + alpha_yaw * sumsqr(-cos(self.state[2, 1:] - self.ref_traj[2, 1:])+1)
            # + alpha_yaw * sumsqr(-self.state[2, 1:] - self.ref_traj[2, 1:])
            # + alpha_jerk * (sumsqr(self.jerk_x) + sumsqr(self.jerk_y))
            # + alpha_jerk_yaw * sumsqr(self.jerk_yaw)
            +  .5 * sumsqr(self.r_accel_x)
            +  .8 * sumsqr(self.r_accel_y)
            + 1.  * sumsqr(self.dd_yaw)
            # - 1.2 * sumsqr(self.r_d_x)*speed_ratio
            # - 1.2 * sumsqr(self.r_d_y)*speed_ratio
            # +  .1 * sumsqr(self.d_yaw)*speed_ratio
        )
        # cost_stage += 3*sumsqr(self.r_d_y)if VehicleConst._rear <0.1 and not self.wheel_constrain else 0.
        # cost_stage += .0005 * sumsqr(self.r_wheel_vy)if self.wheel_constrain else 0.
        # cost_stage += +2.0*sumsqr(self.state[3:6,1:]-self.ref_traj[3:6,1:]) if self.following_mode else 2.

        # if self.var_dt:
        #     # alpha_var_dt = 0.1
        #     alpha_var_T = 2. if self.following_mode else 2.
        #     # cost_stage += alpha_var_dt * sumsqr(self.var_dts - self.dt)
        #     cost_stage += alpha_var_T * sumsqr(self.var_dts)
        # Take special care with the final state
        if self.free_end:
            alpha_terminal_xy = 10.0
            alpha_terminal_yaw = 10.0  
            # really care about final heading to help with lane changes
            alpha_terminal_v = 10.0 if self.following_mode else 15.
            cost_stage += (
                alpha_terminal_xy * sumsqr(self.ref_traj[:2, -1] - vertcat(self.position_x[:,-1], self.position_y[:,-1]))
                + alpha_terminal_yaw * sumsqr(-cos(self.yaw[:,-1] - self.ref_traj[2, -1])+1)
                # + alpha_terminal_yaw * sumsqr(self.yaw[:,-1] - self.ref_traj[2, -1])
                + alpha_terminal_v * (sumsqr(self.d_x[:,-1]-self.ref_traj[3,-1])+sumsqr(self.d_y[:,-1]-self.ref_traj[4,-1])+sumsqr(self.d_yaw[:,-1]-self.ref_traj[5,-1]))
            )
        ## the weight should be carefully tuned, 1.0 is a good start
        # cost_stage += 0.*sumsqr(self.slack_wheel_dir-self.wheel_dir) if self.wheel_constrain else 0.
                        

        self._optimizer.minimize(cost_stage) #+ self.trajectory_len / 4.0 *

    def _set_initial_guess(self, initial_guess: Sequence[Pose], initial_dts: Sequence[float]) -> None:
        self._check_inputs(initial_guess[0,:], initial_guess)

        # Initialize state guess based on reference
        self._optimizer.set_initial(self.state[:self.init_guess_dim,:], DM(initial_guess[:,:self.init_guess_dim]).T)  # (x, y, yaw)
        if initial_dts is not None and self.var_dt:
            self._optimizer.set_initial(self.var_dts[0,:], DM(initial_dts))
        if initial_dts is None and self.var_dt:
            self._optimizer.set_initial(self.var_dts[0,:], DM([self.dt]*self.trajectory_len))
        # Initialize control guess based on reference TODO
        # if self.wheel_constrain:
        #     for i in range(self.trajectory_len+1):
        #         global_wheel_vx, global_wheel_vy = global_dxdydyaw2wheelstate(initial_guess[i,2], initial_guess[i,3], initial_guess[i,4], initial_guess[i,5])
        #         relative_wheel_v = RotationMatrix(-initial_guess[i,2]) @ horzcat(global_wheel_vx,global_wheel_vy).T
        #         wheel_dir = relative_wheel_v[0,:]/fabs(relative_wheel_v[0,:])
        #         self._optimizer.set_value(self.slack_wheel_dir[:,i], wheel_dir)
        #         self._optimizer.set_initial(self.slack_wheel_dir[:,i], wheel_dir)

        # I think initializing the controls would be quite noisy, so using default zero init

    def _check_inputs(self, x_curr: Sequence[float], reference_trajectory: Sequence[Pose]) -> None:
        """Raise ValueError if inputs are not of proper size."""
        if len(x_curr) != self.nx:
            raise ValueError(f"x_curr length {len(x_curr)} must be equal to state dim {self.nx}")

        if len(reference_trajectory) != self.trajectory_len + 1:
            raise ValueError(f"reference traj length {len(reference_trajectory)} must be equal to {self.trajectory_len + 1}")

# if __name__=='__main__':
import matplotlib.pyplot as plt
import time
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
        # === 5. 提取解 ===
        X_opt = sol.value(_optimizer.state).T       # 状态轨迹
        U_opt = sol.value(_optimizer.control).T     # 控制输入
        if _optimizer.var_dt:
            dts_opt = sol.value(_optimizer.var_dts).T  # 时间步长
            print("优化后的时间步：", dts_opt)

        # === 6. 可视化优化结果 ===
        plt.figure(figsize=(10, 6))
        plt.plot(X_opt[:, 0], X_opt[:, 1], label="trajectory", marker='o')
        plt.plot(initial_guess[:, 0], initial_guess[:, 1], label="initial_guess", linestyle='--', marker='+')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("trajectory opt")
        plt.legend()
        plt.grid()
        plt.axis("equal")
        plt.show()
    except RuntimeError:
        print("Smoothing crashed! Use G.T. instead")
        _optimizer._optimizer.debug.show_infeasibilities()
        return initial_guess[1:], out_control
    if not sol.stats()['success']:
        if result_dict is not None:
            result_dict['failed'] = True
        print(f"Smoothing failed with status {sol.stats()['return_status']}! Use G.T. instead")
        return initial_guess[1:], out_control

    ego_smoothed: List[np.float32] = np.vstack([sol.value(_optimizer.state),])
    ego_smoothed = ego_smoothed.T
    control: List[np.float32] = np.vstack([sol.value(_optimizer.control),])
    out_control[:,:3] = control.T
    if _optimizer.wheel_constrain:
        wheel_vx: List[np.float32] = np.vstack([sol.value(_optimizer.wheel_vx),])
        wheel_vx = wheel_vx.T
        wheel_vy: List[np.float32] = np.vstack([sol.value(_optimizer.wheel_vy),])
        wheel_vy = wheel_vy.T
        ego_smoothed[:,6:10] = wheel_vx[:,:]
        ego_smoothed[:,10:14] = wheel_vy[:,:]
    if _optimizer.var_dt:
        var_dts: List[np.float32] = np.vstack([sol.value(_optimizer.var_dts),])
    result_traj = ego_smoothed[1:,:]
    out_control[:,3:4] = _optimizer.dt
    if _optimizer.var_dt:
        var_dts = var_dts.T
        out_control[:,3:4] = var_dts
    return result_traj, out_control
def test_dt(points = None, vs = None, dts = None, result_dict:dict=None):
    wheel_constrain = True
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
                                            simplify_model=True)

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
            wheel_vs = global_vs[:, None] + np_dRotationMatrix(points[i,2],vs[i,2]) @ wheel_positions
            init_state[i,6:10] = wheel_vs[0,:]
            init_state[i,10:14] = wheel_vs[1,:]
    print("init_state has NaN: ",np.isnan(init_state).any())
    # print("init_state[0]: ", init_state[0])
    print("init_state:", init_state)
    print("dts:", dts)
    ### optimizing
    result_traj, control = traj_smooth(init_state, optimizer, dts, result_dict)
    print("result_traj:", result_traj)
    print("control:", control)
    opt_dts = control[:,-1:]

    del optimizer
    return result_traj, control

if __name__ == "__main__":
    T = 20
    dt = 0.1
    times = np.array([dt] * T)

    # 初始状态 (x, y, yaw, vx, vy, dyaw)
    points = np.zeros((T + 1, 3))
    vs = np.zeros((T + 1, 3))
    dts = np.zeros((1, T))
    for i in range(T + 1):
        points[i, 0] = i * dt * 1.0  # x
        points[i, 1] = i * dt * 1.0  # y
        points[i, 2] = 0.0           # yaw
        vs[i, 0] = 0.5           # vx
        vs[i, 1] = 0.0           # vy
        vs[i, 2] = 0.0           # dyaw
    for i in range(T):
        dts[0, i] = 0.1           # dyaw
    result_dict = {}
    test_dt(points, vs, dts, result_dict)

