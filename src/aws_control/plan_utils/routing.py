from ast import arg, main
from itertools import count
from matplotlib import axis, figure
import numpy as np
from constants import VehicleConst
import rospy
import yaml
import matplotlib.pyplot as plt
import argparse
from visualization_msgs.msg import Marker, MarkerArray

from geometry_msgs.msg import PoseStamped, PoseArray, Pose

from common import ROS_INTERFACE, pi2pi, quart_to_rpy_wxyz, quart_to_rpy_xyzw, rpy2quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

## record my poses and save them to a yaml file
def record_poses(ri: ROS_INTERFACE):
    poses = []
    while not rospy.is_shutdown():
        try:
            trans, rot = ri.get_tf('world', 'Imu_Sensor')
            poses.append(trans+rot)
            print('added a waypoint', trans, rot, len(poses))
            np.save('poses.npy', np.array(poses))
            plt.plot(np.array(poses)[:,0], np.array(poses)[:,1])
            plt.pause(0.01)
        except:
            print('waiting for tf')
        ri.rate.sleep()


## show recorded poses
def show_poses(ri,poses):
    # plt.cla()
    # plt.plot(poses[:,0], poses[:,1])
    # plt.show()
    ma = ri.make_marker_array(poses, 
                              frame_id='world', 
                              ns='my_namespace', 
                              m_type=Marker.ARROW, 
                              action=Marker.ADD, 
                              scale=[0.1, 0.1, 0.1], 
                              color=[0.0, 1.0, 0.0, 1.0])
    ri.publish('global_routing', ma)

## offer reference path regarding to current pose
def offer_path(ri, poses, robot_odom, robot_imu):
    path_errors = []
    imu_ts = []
    imus = []
    odom_ts = []
    odom_velocities = []
    # heading_offset = 0 #(np.pi/4)/100
    heading_offset = (np.pi/3)/100
    plotting = True
    last_plot_flag = False
    last_nearest = 0
    while rospy.is_shutdown() is False:
        # try:
            # (trans, rot) = ri.get_tf('world', 'Imu_Sensor')
            # (trans, rot) = 
            # curreent_pose = trans[:2]
        curreent_pose = robot_odom['robot_pose']
        if curreent_pose is None:
            ri.rate.sleep()
            print('waiting for odom')
            continue

        find_nearest = lambda x, poses: \
            np.argmin(np.linalg.norm(np.array(poses)[:, :2] \
                                     - np.array(x)[:2], axis=1))
        nearest = find_nearest(curreent_pose, 
                               poses[last_nearest:])\
                                +last_nearest
        last_nearest = nearest
        # print('error(m):', np.linalg.norm(curreent_pose[:2] - poses[nearest][:2],axis=-1))
        traj = poses[nearest+1:nearest+1+VehicleConst.T]
        if traj.shape[0] < VehicleConst.T:
            last_plot_flag = True
        traj = np.pad(traj,
                      ((0,VehicleConst.T-traj.shape[0]),(0,0)), 
                      mode='edge')

        headings = traj[:,5]
        off_headings = headings + heading_offset*nearest + \
            np.cumsum(np.ones_like(headings))*heading_offset
        off_headings = pi2pi(off_headings)
        ma = ri.make_marker_array(
            np.concatenate([traj[...,:5],
                            off_headings[...,None]],
                            axis=-1), 
                                  frame_id='world', 
                                  ns='my_namespace', 
                                  m_type=Marker.ARROW, 
                                  action=Marker.ADD, 
                                  scale=[0.2, 0.1, 0.1], 
                                  color=[1.0, 0.0, 1.0, 1.0])
        ri.publish('path_planning_show', ma)
        pa = PoseArray()
        pa.header.frame_id = 'world'
        pa.header.stamp = rospy.Time.now()
        for i, p in enumerate(traj):
            pose = Pose()
            pose.position.x = p[0]
            pose.position.y = p[1]
            pose.position.z = p[2]
            q = rpy2quaternion(p[3], p[4], off_headings[i])
            pose.orientation.x = q[0]
            pose.orientation.y = q[1]
            pose.orientation.z = q[2]
            pose.orientation.w = q[3]
            pa.poses.append(pose)
        ri.publish('path_planning', pa)


        if  plotting and robot_imu['time_stamp'] and robot_odom['time_stamp']:
            ## plot
            # curreent_poses.append(curreent_pose)
            # times.append(rospy.Time.now().to_sec())

            imu_ts.append(robot_imu['time_stamp'])
            imus.append(robot_imu['robot_imu'])
            odom_ts.append(robot_odom['time_stamp'])
            odom_velocities.append(robot_odom['robot_vel'])
            
            path_errors.append(np.linalg.norm(curreent_pose[:2] - poses[nearest][:2], axis=-1))
            print(f"data at time stamp: {robot_imu['time_stamp'] - imu_ts[0]:.1f} s, recorded")
        else:
            continue
        if last_plot_flag:
            ts = np.array(imu_ts)
            ts = ts-ts[0]
            ts = ts[:, None]

            plt.cla()
            plt.figure(figsize=(9,15), dpi=100)
            ax_error = plt.subplot(5,1,1)
            ax_dxdy = plt.subplot(5,1,2)
            ax_dyaw = plt.subplot(5,1,3)
            ax_acc = plt.subplot(5,1,4)
            ax_ddyaw = plt.subplot(5,1,5)
            ax_error.plot(ts, np.array(path_errors), '-k')
            ax_error.set_xlabel('time step')
            ax_error.set_ylabel('navigation error (m)')
            ax_error.set_ylim(0,1)
            ax_error.axhline(y=0.5, color = 'r', linestyle='--')
            
            # d
            # np_curreent_poses = np.array(curreent_poses)
            # dx = np.diff(np_curreent_poses, axis=0)/np.diff(ts, axis=0)
            imu_ddv_dyaw = np.array(imus)
            # velocity
            # ax_dxdy.plot(ts[0:-1], np.linalg.norm(dx[...,:2], axis=-1), '-k')
            ax_dxdy.plot(ts, np.linalg.norm(np.array(odom_velocities)[...,:2], axis=-1), '-k')
            ax_dxdy.set_xlabel('time step')
            ax_dxdy.set_ylabel('velocity (m/s)')
            ax_dxdy.set_ylim(0, 3.)
            # dyaw
            ax_dyaw.plot(ts, imu_ddv_dyaw[...,2], '-k')
            ax_dyaw.set_xlabel('time step')
            ax_dyaw.set_ylabel('dyaw (rad/s)')
            ax_dyaw.set_ylim(-2, 2)
            
            ## dd
            ## acc
            ax_acc.plot(ts, np.linalg.norm(imu_ddv_dyaw[...,:2], axis=-1), '-k')
            ax_acc.set_xlabel('time step')
            ax_acc.set_ylabel('acc (m/s^2)')
            ax_acc.set_ylim(0, 5)
            ## ddyaw
            imu_ts = np.array(imu_ts)
            imu_ts = imu_ts-imu_ts[0]
            d_imu_t = np.diff(imu_ts, axis=0)
            d_imu_t = np.where(d_imu_t==0, 1, d_imu_t)
            ddyaw = np.diff(imu_ddv_dyaw[...,2], axis=0)/d_imu_t
            ax_ddyaw.plot(ts[0:-1], np.abs(ddyaw), '-k')
            ax_ddyaw.set_xlabel('time step')
            ax_ddyaw.set_ylabel('ddyaw (rad/s^2)')
            ax_ddyaw.set_ylim(0, 5)

            plt.subplots_adjust(hspace=0.3)
            plt.savefig('path_errors.png')
            return
        else:
            continue

    
def main():
    ri = ROS_INTERFACE()
    ri.add_publisher('global_routing', MarkerArray)
    ri.add_publisher('path_planning_show', MarkerArray)
    ri.add_publisher('path_planning', PoseArray)
    robot_odom = {'robot_vel': None, 'robot_pose': None, 'time_stamp': None}
    def vel_callback(msg, robot_odom):
        rot = quart_to_rpy_xyzw(msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)
        # robot_odom['robot_pose'] = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, rot[-1]])
        tf_trans, tf_rtoation = ri.get_tf('world', 'body')
        robot_odom['robot_pose'] = np.concatenate([tf_trans[:2], tf_rtoation[-1:]], axis=-1)
        robot_odom['robot_vel'] = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.angular.z])
        robot_odom['time_stamp'] = msg.header.stamp.to_sec()
    ri.add_subscriber('odom', Odometry, vel_callback, robot_odom)
    robot_imu = {'robot_imu': None, 'time_stamp': None}
    def imu_callback(msg, robot_imu):
        rot = np.array([msg.angular_velocity.z])
        trans = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y])
        robot_imu['robot_imu'] = np.concatenate([trans, rot], axis=-1)
        robot_imu['time_stamp'] = msg.header.stamp.to_sec()
    ri.add_subscriber('imu', Imu, imu_callback, robot_imu)
    if args.record:
        print('recording')
        record_poses(ri)
    else:
        poses = np.load('poses.npy')
        show_poses(ri, poses)
        offer_path(ri, poses, robot_odom, robot_imu)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--record', action='store_true', default=False)
    args = argparser.parse_args()

    main()  


