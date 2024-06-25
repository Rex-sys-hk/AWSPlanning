import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/../')
import math
from matplotlib.pyplot import sca
import numpy as np
from scipy.spatial.transform import Rotation
from plan_utils.constants import VehicleConst
from casadi import fmod
def pi2pi(angle):
    angle = (angle + math.pi)%(2*math.pi) - math.pi
    return angle

def deg2rad(angle):
    return angle/180*math.pi

pi = math.pi
def Casadi_pi2pi(angle):
    return fmod((angle+pi),2*pi)-pi

def xyyaw2v(traj:np.ndarray, dt = 0.1, prepend = np._NoValue, append = np._NoValue):
    if traj.ndim != 2:
        raise ValueError('traj must be 2D array')
    if traj.shape[1] != 3:
        raise ValueError('traj must be Nx3 array')    
    diff = np.diff(traj,axis=0,prepend=prepend,append=append)
    diff[...,2] = pi2pi(diff[...,2])
    return diff/dt

def np_RotationMatrix(theta):
    return np.array([[np.cos(theta),-np.sin(theta)],
                     [np.sin(theta),np.cos(theta)]])

def np_dRotationMatrix(theta, dtheta):
    return np.array([[-np.sin(theta),-np.cos(theta)],
                     [np.cos(theta),-np.sin(theta)]])*(dtheta)

def np_ddRotationMatrix(theta, dtheta, ddtheta):
    return np.array([[-np.cos(theta),np.sin(theta)],
                     [-np.sin(theta),-np.cos(theta)]])*dtheta**2 \
        +np.array([[-np.sin(theta),-np.cos(theta)],
                   [np.cos(theta),-np.sin(theta)]])*ddtheta


def np_dxdydyaw2wheelstate(dxdydyaw:np.ndarray, theta:np.ndarray):
    if dxdydyaw.ndim != 2:
        raise ValueError('dxdydyaw must be 2D array')
    if dxdydyaw.shape[1] != 3:
        raise ValueError('dxdydyaw must be Nx3 array')
    wheel_positions = np.array(VehicleConst.wheel_positions).T
    vx = dxdydyaw[...,0:1]\
        -dxdydyaw[...,2:3]*wheel_positions[0:1,:]*np.sin(theta)\
        -dxdydyaw[...,2:3]*wheel_positions[1:2,:]*np.cos(theta)
    vy = dxdydyaw[...,1:2]\
        +dxdydyaw[...,2:3]*wheel_positions[0:1,:]*np.cos(theta)\
        -dxdydyaw[...,2:3]*wheel_positions[1:2,:]*np.sin(theta)
    print('vx,vy: ',vx, vy)
    return vx,vy

def quart_to_rpy_wxyz(w, x, y, z):
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(2 * (w * y - x * z))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))
    return [roll, pitch, yaw]

def quart_to_rpy_xyzw(x, y, z, w):
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(2 * (w * y - x * z))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))
    return [roll, pitch, yaw]

def rpy2quaternion(roll, pitch, yaw):
    x=np.sin(pitch/2)*np.sin(yaw/2)*np.cos(roll/2)\
        +np.cos(pitch/2)*np.cos(yaw/2)*np.sin(roll/2)
    y=np.sin(pitch/2)*np.cos(yaw/2)*np.cos(roll/2)\
        +np.cos(pitch/2)*np.sin(yaw/2)*np.sin(roll/2)
    z=np.cos(pitch/2)*np.sin(yaw/2)*np.cos(roll/2)\
        -np.sin(pitch/2)*np.cos(yaw/2)*np.sin(roll/2)
    w=np.cos(pitch/2)*np.cos(yaw/2)*np.cos(roll/2)\
        -np.sin(pitch/2)*np.sin(yaw/2)*np.sin(roll/2)
    return x, y, z, w


import rospy
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray
from tf import TransformListener
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import Point
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import Transform
from geometry_msgs.msg import Pose
from visualization_msgs.msg import Marker, MarkerArray

class ROS_INTERFACE:
    def __init__(self, node_name = 'rosbase', anonymous = False, rate = 10):

        rospy.init_node(node_name, anonymous=anonymous)
        # rospy.spin()
        self.rate = rospy.Rate(rate)
        self.tf_listener = TransformListener(True, rospy.Duration(2.0))

        self.publisher = {}
        self.subscriber = {}


    def get_tf(self, source, target, time = rospy.Time(0), eular_angle = True):
        # try:
        (trans, rot) = self.tf_listener.lookupTransform(source, target, time)
        if eular_angle:
            erot = quart_to_rpy_xyzw(*rot)
        return trans, erot
        # except:
        #     print('tf not found')
        #     pass
            # return [0,0,0],[0,0,0,0]
    
    def add_subscriber(self, topic, msg_type, callback, args_dic:dict, queue_size = 10):
        self.subscriber[topic] = rospy.Subscriber(topic, msg_type, callback, args_dic, queue_size = queue_size)


    def add_publisher(self, topic, msg_type, queue_size = 10):
        self.publisher[topic] = rospy.Publisher(topic, msg_type, queue_size = queue_size)
    
    def publish(self, topic, msg):
        self.publisher[topic].publish(msg)

    def make_marker_array(self, seq, frame_id='world', ns='my_namespace', m_type=Marker.ARROW, action=Marker.ADD, scale=[0.05, 0.05, 0.], color=[1.0, 1.0, 1.0, 1.0]):
        """
        seq: Nx6 array, each row is [x, y, z, roll, pitch, yaw]
        color: [r, g, b, a]
        """
        ma = MarkerArray()
        for i,p in enumerate(seq):
            marker = Marker()
            marker.id = i
            marker.header.frame_id = frame_id
            marker.header.stamp = rospy.Time.now()
            marker.ns = ns
            marker.type = m_type
            marker.action = action
            marker.pose.position.x = p[0]
            marker.pose.position.y = p[1]
            marker.pose.position.z = p[2]
            # r = Rotation.from_euler('xyz', p[3:5], degrees=False)
            # q = r.as_quat()
            q = rpy2quaternion(p[3], p[4], p[5])
            marker.pose.orientation.x = q[0]
            marker.pose.orientation.y = q[1]
            marker.pose.orientation.z = q[2]
            marker.pose.orientation.w = q[3]
            marker.scale.x = scale[0]
            marker.scale.y = scale[1]
            marker.scale.z = scale[2]
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = color[3]
            ma.markers.append(marker)

        return ma

class TimeoutException(Exception): pass
from contextlib import contextmanager
import signal
@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def vel_world_to_robot(vel_world, yaw_world):
    v_x = vel_world[0]
    v_y = vel_world[1]
    dyaw = vel_world[2]
    dx_r = np.sin(yaw_world)*v_y+np.cos(yaw_world)*v_x
    dy_r = np.cos(yaw_world)*v_y-np.sin(yaw_world)*v_x
    return np.array([dx_r,dy_r,dyaw])

def coordinate_transform_to_robot(x, y, yaw, robot_theta):
    x_r = x*np.cos(robot_theta)+y*np.sin(robot_theta)
    y_r = -x*np.sin(robot_theta)+y*np.cos(robot_theta)
    yaw_r = yaw-robot_theta
    yaw_r = pi2pi(yaw_r)
    return x_r, y_r, yaw_r

def coordinate_transform_to_world(x_r, y_r, yaw_r, robot_theta):
    x = x_r*np.cos(-robot_theta)+y_r*np.sin(-robot_theta)
    y = -x_r*np.sin(-robot_theta)+y_r*np.cos(-robot_theta)
    yaw = yaw_r+robot_theta
    return x, y, yaw

def AWSConstructSamplingTable(r_samples, psi_samples, omega_sample_list):
    epsilon = 1e-9
    delta_x = VehicleConst.cellSize
    delta_y = VehicleConst.cellSize
    delta_cur = math.sqrt(delta_x**2 + delta_y**2)
    # delta_theta = VehicleConst.deltaHeadingRad
    steering_limit = np.array(VehicleConst.wheel_steer_limits).T
    wheel_positions = np.array(VehicleConst.wheel_positions).T

    delta_sample_r = math.pi / (2 * r_samples) - epsilon
    delta_psi = 2 * math.pi / psi_samples


    samplingTable = []
    r_table = []

    for i in range(r_samples + 1):
        for j in range(psi_samples + 1):
            r_scale = math.tan(delta_sample_r * i + epsilon)
            psi = -math.pi + delta_psi * j
            r_x = -r_scale * math.cos(psi)
            r_y = -r_scale * math.sin(psi)
            infeasible = False
            
            for k in range(4):
                steer_ulim_p_pi_d_2 = steering_limit[1,k] + math.pi / 2
                steer_llim_p_pi_d_2 = steering_limit[0,k] + math.pi / 2
                wx = wheel_positions[0,k]
                wy = wheel_positions[1,k]
                r_w_x = r_x + wx
                r_w_y = r_y + wy

                # deg = math.atan2(r_w_y, r_w_x)
                # deg = pi2pi(deg)
                # if not ((abs(deg)<=steer_ulim_p_pi_d_2 \
                #      and \
                #      abs(deg)>=steer_llim_p_pi_d_2)):
                #     infeasible = True
                #     break
                
                if (r_w_x * math.sin(steer_ulim_p_pi_d_2) 
                    - r_w_y * math.cos(steer_ulim_p_pi_d_2)) * \
                   (r_w_x * math.sin(steer_llim_p_pi_d_2) 
                    - r_w_y * math.cos(steer_llim_p_pi_d_2)) > 0:
                    infeasible = True
                    break

            if infeasible:
                continue

            for omega in omega_sample_list:
                max_omega = delta_cur / r_scale
                omega = np.sign(omega)*min(np.abs(omega), np.abs(max_omega))
                d_x = -omega * r_y
                d_y = omega * r_x
                r_table.append([r_x, r_y, omega])
                # r_table.append([r_x, r_y, -omega])
                # r_table.append([-r_x, -r_y, omega])
                # r_table.append([-r_x, -r_y, -omega])

                samplingTable.append([d_x, d_y, omega])
                # samplingTable.append(combo_neg.copy())
                infeasible = False
                for c in r_table:
                    if np.abs(c[0] - r_x) < 1e-2 \
                        and np.abs(c[1] - r_y) < 1e-2 \
                        and np.abs(c[2] - omega) < 1e-2:
                        infeasible = False
                        break
                if infeasible:
                    continue
                if omega == max_omega:
                    break

    return samplingTable, r_table