import os,sys
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/../')
from plan_utils.common import ROS_INTERFACE, quart_to_rpy_xyzw
from plan_utils.constants import VehicleConst
import rospy
import numpy as np
from nav_msgs.msg import Odometry
from visualization_msgs.msg import MarkerArray
import argparse


folder_name = 'runs/motion_records/tmp'
# folder_name = 'ackermann'
# folder_name = 'aws'
os.makedirs(folder_name, exist_ok=True)

ri = ROS_INTERFACE('motion_analyser')

def odom_callback(msg, robot_odom):
    # robot states at that time
    robot_odom['vs'].append(np.array([msg.twist.twist.linear.x, 
                                        msg.twist.twist.linear.y, 
                                        msg.twist.twist.angular.z]))
    # robot_odom['received_vs'] = True
    t,r = ri.get_tf('world', 'body')
    robot_odom['body_tf'].append(np.array([t[0],t[1],r[2]]))
    wvs=[]
    for i in range(VehicleConst.wheel_num):
        t,r = ri.get_tf('body', VehicleConst.wheel_frames[i])
        wvs.append(r[2])
    robot_odom['wheel_tf'].append(np.array(wvs))

def path_callback(msg, robot_odom):
    p = []
    for m in msg.markers:
        x = m.pose.position.x
        y = m.pose.position.y
        yaw = quart_to_rpy_xyzw(m.pose.orientation.x,
                                m.pose.orientation.y,
                                m.pose.orientation.z,
                                m.pose.orientation.w)[2]
        p.append(np.array([x,y,yaw]))
    robot_odom['path'] = p if len(p) > 3 else robot_odom['path']


def recording():

    state_dict = {'body_tf':[],
                'wheel_tf':[],
                'vs':[],
                'received_vs':False,
                'path':[]
                }

    ri.add_subscriber('odom', Odometry, odom_callback, state_dict)
    ri.add_subscriber('/smoothpath', MarkerArray, path_callback, state_dict)


    while not rospy.is_shutdown():
        rospy.spin()
        ri.rate.sleep()

    np.array(state_dict['body_tf']).dump(f'{folder_name}/body_tf.npy')
    np.array(state_dict['wheel_tf']).dump(f'{folder_name}/wheel_tf.npy')
    np.array(state_dict['vs']).dump(f'{folder_name}/vs.npy')
    np.array(state_dict['path']).dump(f'{folder_name}/path.npy')
    print('\n')
    print('='*30,'finished','='*30)
    print("tf counts are: ",len(state_dict['body_tf']), len(state_dict['wheel_tf']), len(state_dict['vs']))
    print('path length is: ', len(state_dict['path']))
    print('files are saved in ', folder_name)

if __name__ == '__main__':
    arg = argparse.ArgumentParser()
    arg.add_argument('--folder', type=str, default=folder_name)
    args = arg.parse_args()
    folder_name = args.folder
    recording()