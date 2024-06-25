import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/../')
from plan_utils.common import ROS_INTERFACE, quart_to_rpy_xyzw
from geometry_msgs.msg import PoseStamped
import rospy 
import numpy as np
import os
folder = 'exp_utils/qe_records'

def record_start_point(msg, points):
    x = msg.pose.position.x
    y = msg.pose.position.y
    # yaw = msg.pose.orientation.z
    yaw = quart_to_rpy_xyzw(msg.pose.orientation.x,
                            msg.pose.orientation.y,
                            msg.pose.orientation.z,
                            msg.pose.orientation.w)[2]
    points['start'].append([x,y,yaw])
    print("new start: ", [x,y,yaw])
    print(f"start: {len(points['start'])}, end: {len(points['end'])}")

def record_end_point(msg, points):
    x = msg.pose.position.x
    y = msg.pose.position.y
    # yaw = msg.pose.orientation.z
    yaw = quart_to_rpy_xyzw(msg.pose.orientation.x,
                            msg.pose.orientation.y,
                            msg.pose.orientation.z,
                            msg.pose.orientation.w)[2]
    points['end'].append([x,y,yaw])
    print('new end: ', [x,y,yaw])
    print(f"start: {len(points['start'])}, end: {len(points['end'])}")

def main():
    ri = ROS_INTERFACE()
    # record start points
    points = {'start':[], 'end':[]}
    ri.add_subscriber('/move_base_simple/start', 
                      PoseStamped, 
                      record_start_point, 
                      points)
    # record end points
    ri.add_subscriber('/move_base_simple/goal', 
                      PoseStamped, 
                      record_end_point, 
                      points)
    while not rospy.is_shutdown():
        ri.rate.sleep()
    start = np.array(points['start'])
    end = np.array(points['end'])

    os.makedirs(folder, exist_ok=True)
    np.save(f'{folder}/start_points.npy', start)
    np.save(f'{folder}/end_points.npy', end)
    print(f'file saved in {folder}')
    

if __name__ == "__main__":
    main()