from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, TypeVar

from pygame import init
from common import pi2pi, xyyaw2v

from constants import VehicleConst
from robot2actuator import k_theta_v_to_control, relative_v_and_omega_control
Pose = Tuple[float, float, float]  # (x, y, yaw)
from casadi import (DM, Opti, OptiSol, cos, diff, sin, 
                    sumsqr, vertcat, pi, fmod, norm_2, 
                    atan2, if_else, tan , horzcat, log, 
                    fabs, acos, sqrt)
import numpy as np
import logging
    # constrain.append(v[1]*cos(s[1])-v[0]*cos(s[0]) - omega*W)
    # constrain.append(v[3]*cos(s[3])-v[2]*cos(s[2]) - omega*W)
    # constrain.append(v[0]*sin(s[0])-v[2]*sin(s[2]) - omega*L)
    # constrain.append(v[1]*sin(s[1])-v[3]*sin(s[3]) - omega*L)
def wheel2robot(flx,fly,rly):
    # robot heading is x axis, wheel sequence FL, FR, RL, RR
    wheel_positions = np.array([VehicleConst.wheel_positions[0][0],VehicleConst.wheel_positions[0][2]])
    vy = (fly+rly)/2
    omega = (fly-rly)/VehicleConst.WHEEL_BASE
    vx = flx+wheel_positions[0][1]*omega
    return vx,vy,omega

def d_wheel2robot(dflx,dfly,drly):
    # robot heading is x axis, wheel sequence FL, FR, RL, RR
    wheel_positions = np.array([VehicleConst.wheel_positions[0][0],VehicleConst.wheel_positions[0][2]])
    dvy = (dfly+drly)/2
    domega = (dfly-drly)/VehicleConst.WHEEL_BASE
    dvx = dflx+wheel_positions[0][1]*domega
    return dvx,dvy,domega

def flvflsrlvrls2dxdydyaw(flv,fls,rlv,rls):
    # robot heading is x axis, wheel sequence FL, FR, RL, RR
    wheel_positions = np.array([VehicleConst.wheel_positions[0][0],VehicleConst.wheel_positions[0][2]])
    flx = flv*cos(fls)
    fly = flv*sin(fls)
    rlx = rlv*cos(rls)
    rly = rlv*sin(rls)
    vy = (fly+rly)/2
    omega = (fly-rly)/VehicleConst.WHEEL_BASE
    vx = flx+wheel_positions[0][1]*omega
    return vx,vy,omega

def dflvdflsdrlvdrls2dxdydyaw(dflv,dfls,drlv,drls,flv,fls,rlv,rls):
    # robot heading is x axis, wheel sequence FL, FR, RL, RR
    wheel_positions = np.array([VehicleConst.wheel_positions[0][0],VehicleConst.wheel_positions[0][2]])
    dflx = dflv*cos(fls)+flv*(-sin(fls))*dfls
    dfly = dflv*sin(fls)+flv*cos(fls)*dfls
    drlx = drlv*cos(rls)+rlv*(-sin(rls))*drls
    drly = drlv*sin(rls)+rlv*cos(rls)*drls
    dvx = dflx+wheel_positions[0][1]*dfls
    dvy = dfly+wheel_positions[0][0]*dfls
    domega = (dfly-drly)/VehicleConst.WHEEL_BASE
    return dvx,dvy,domega

# TODO: only proposed idea. not implemented