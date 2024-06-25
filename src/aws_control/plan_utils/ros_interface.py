import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/../')
from plan_utils.common import pi2pi
from plan_utils.constants import VehicleConst
import rospy
# twist message
from geometry_msgs.msg import Twist
# JointState message
from sensor_msgs.msg import JointState
from robot2actuator import relative_v_and_omega_control

def callback(data):
    print('='*20)
    print('Motion command received:',f'vx = {data.linear.x}, vy = {data.linear.y}, wz = {data.angular.z}')
    v,s = relative_v_and_omega_control(data.linear.x, 
                                       data.linear.y, 
                                       data.angular.z, 
                        lim_angle=VehicleConst.wheel_steer_limits, 
                                       wheel_rotate_velocity=True)
    if VehicleConst._rear<1e-2 \
        and VehicleConst._control_center_offset<1e-2:
        print('[Warning]: using symetric steering mode')
        v = [v[0],v[1],v[0],v[1]]
        s = [s[0],s[1],-pi2pi(s[0]),-pi2pi(s[1])]
    velocitys = JointState(
        header=None,
        name=['FLWalk', 'FRWalk', 'RLWalk', 'RRWalk'],
        position=None,
        velocity=[v[0], v[1], v[2], v[3]],
        effort=None,
    )
    pub_v.publish(velocitys)
    steerings = JointState(
        header=None,
        name=['FLSteer', 'FRSteer', 'RLSteer', 'RRSteer'],
        position=[s[0], s[1], s[2], s[3]],
        velocity=None,
        effort=None,
    )
    pub_steering.publish(steerings)
    print('Walk Motor(rad/s):',v,
          '\n',
          'Steering Motor(angle(-pi~pi)):',s)

def main():

    # subscribe robot control command
    rospy.Subscriber('robot_control_command', Twist, callback)

    # commond publisher
    rospy.spin()

if __name__ == '__main__':
    try:
        # init node
        rospy.init_node('four_ws_drive', anonymous=True)
        pub_v = rospy.Publisher('v_command', JointState, queue_size=10)
        pub_steering = rospy.Publisher('steer_command', JointState, queue_size=10)
        main()
    except rospy.ROSInterruptException:
        pass
