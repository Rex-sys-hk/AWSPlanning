from os import path
import re

from click import command

from plan_utils.constants import VehicleConst
from plan_utils.mpc_controller import AWSMassPointOptimizer, traj_smooth, dxdydyaw2wheelstate, test
from plan_utils.common import (ROS_INTERFACE, coordinate_transform_to_robot, coordinate_transform_to_world, pi2pi, quart_to_rpy_wxyz, 
                               quart_to_rpy_xyzw, xyyaw2v, 
                               vel_world_to_robot, rpy2quaternion,
                               time_limit)
from plan_utils.ros_mpc_follower_test import TimeoutException
import rospy
import numpy as np
from geometry_msgs.msg import Twist, PoseArray
from trajectory_msgs.msg import JointTrajectory
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
import matplotlib.pyplot as plt
from plan_utils.dt_mpc_controller import test_dt



if __name__ == '__main__':
    loop = 0
    wheel_constrain = True
    to_vehicle_coo = True
    ## initialize ros node
    ri = ROS_INTERFACE(node_name='follower')
    # subscribe path
    path_dict = {'path':None,'failed':0,'total':0}
    def path_callback(msg, path_dict):
        # global traj
        trans = np.array([
            [p.position.x, p.position.y, p.orientation.z] for p in msg.poses
            ])
        rot = np.array(
            [quart_to_rpy_xyzw(p.orientation.x, 
                               p.orientation.y, 
                               p.orientation.z, 
                               p.orientation.w) for p in msg.poses])
        path_dict['path'] = np.concatenate([trans[...,:2], 
                                            rot[...,-1:]], axis=-1)
    ri.add_subscriber('/path_planning', PoseArray, path_callback, path_dict)
    def path_v_callback(msg, path_dict):
        print("recevied new msg, point num: ", len(msg.points))
        # path_dict['path'] = np.zeros((T,6))
        path = np.zeros((len(msg.points),6))
        dts = []
        # if len(msg.points) < path_dict['path'].shape[0]:
        #     return
        for i, m in enumerate(msg.points):
            if i >= path.shape[0]:
                break
            x = m.positions[0]
            y = m.positions[1]
            rot = m.positions[2]
            v_x = m.velocities[0]
            v_y = m.velocities[1]
            v_rot = m.velocities[2]
            duration = m.time_from_start.to_sec()
            path[i] = np.array([x,y,rot,v_x,v_y,v_rot])
            dts.append(duration)
        # control = test(path[:,:3], path[:,3:])
        dts = np.array(dts)
        dts = np.diff(dts, axis=0)
        path[1:,2] = path[0,2] + np.cumsum(path[1:,5])
        vs = path[:,3:6]
        vs = np.concatenate([
                             vs[1:,:]/dts[:,None], 
                             vs[-1:,:]*1e-5], axis=0)
        control = test_dt(path[:,:3], vs, dts, VISULIZE=True)
        path_dict['total'] += 1
        if np.all(control==0):
            path_dict['failed'] += 1
        print("fail rate:", path_dict['failed']/path_dict['total'])
        return
        # rospy.spin()

    ri.add_subscriber('/path_sample_vel', JointTrajectory, path_v_callback, path_dict)

    while not rospy.is_shutdown():
        rospy.spin()


