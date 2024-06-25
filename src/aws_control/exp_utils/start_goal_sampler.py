import os, sys
from tracemalloc import start
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/../')
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from plan_utils.common import ROS_INTERFACE, rpy2quaternion
from start_goal_recording import folder

def main():
    ri = ROS_INTERFACE()
    ri.add_publisher('/move_base_simple/start', PoseStamped, 1)
    ri.add_publisher('/move_base_simple/goal', PoseStamped, 1)
    recording = {'rec':False}
    def rec_callback(msg,recording):
        recording['rec'] = msg.data
    ri.add_subscriber('/need_recording',Bool,rec_callback,recording)
    start = np.load(f'{folder}/start_points.npy')
    end = np.load(f'{folder}/end_points.npy')
    for s in start:
        if rospy.is_shutdown():
            break
        if s[0]>71:
            s[0] = 71
        for e in end:
            if rospy.is_shutdown():
                break
            if e[0]>71:
                e[0] = 71
            print('start:', s)
            p = PoseStamped()
            p.header.frame_id = 'map'
            p.pose.position.x = s[0]
            p.pose.position.y = s[1]
            quad = rpy2quaternion(0,0,s[2])
            p.pose.orientation.x = quad[0]
            p.pose.orientation.y = quad[1]
            p.pose.orientation.z = quad[2]
            p.pose.orientation.w = quad[3]
            # this is actually set to goal
            ri.publish('/move_base_simple/goal', p)
            rospy.sleep(5.)
            while not rospy.is_shutdown() and recording['rec']:
                ri.rate.sleep()
            print('end:', e)
            p = PoseStamped()
            p.header.frame_id = 'map'
            p.pose.position.x = e[0]
            p.pose.position.y = e[1]
            quad = rpy2quaternion(0,0,e[2])
            p.pose.orientation.x = quad[0]
            p.pose.orientation.y = quad[1]
            p.pose.orientation.z = quad[2]
            p.pose.orientation.w = quad[3]
            ri.publish('/move_base_simple/goal', p)
            rospy.sleep(5.)
            while not rospy.is_shutdown() and recording['rec']:
                ri.rate.sleep()
            if rospy.is_shutdown():
                break
    print('all done')
if __name__ == "__main__":
    main()