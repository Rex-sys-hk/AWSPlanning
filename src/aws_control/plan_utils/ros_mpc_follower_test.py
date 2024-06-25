from mpc_controller import *
from traj_generator import min_jerk
from pid_controller import AWS_PID
from common import *
from scipy.spatial.transform import Rotation
import rospy
import tf
from geometry_msgs.msg import Twist, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray

import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# env = AGVEnv(headless=False, skip_frame=1, max_episode_length = 60*5)


class ROS_INTERFACE:
    def __init__(self) -> None:
        rospy.init_node('mpc_test', anonymous=True)
        self.listener = tf.TransformListener(True, rospy.Duration(2.0))
        self.goal_listener = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback)
        self.vel_pub = rospy.Publisher('/robot_control_command', Twist, queue_size=10)
        self.marker_pub = rospy.Publisher('/plantraj', MarkerArray, queue_size=1)
        self.goal = None
        print('finished init')

    def goal_callback(self, data):
        pos = [data.pose.position.x, data.pose.position.y, data.pose.position.z]
        _,_,yaw = quart_to_rpy_wxyz(data.pose.orientation.w, data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z)

        goal = [pos[0], pos[1], yaw]
        print('set new goal:', goal)
        self.goal = goal

    def publish_traj(self, traj):
        ma = MarkerArray()
        # marker = Marker()
        # marker.header.frame_id = "world"
        # marker.header.stamp = rospy.Time.now()
        # marker.ns = "my_namespace"
        # marker.id = 0
        # marker.type = Marker.ARROW
        # marker.action = Marker.DELETEALL
        # self.marker_pub.publish(marker)
        for i,tr in enumerate(traj):
            marker = Marker()
            marker.id = i
            marker.header.frame_id = "world"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "my_namespace"
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.pose.position.x = tr[0]
            marker.pose.position.y = tr[1]
            marker.pose.position.z = 0
            r = Rotation.from_euler('xyz', [0, 0, tr[2]], degrees=False)
            q = r.as_quat()
            marker.pose.orientation.x = q[0]
            marker.pose.orientation.y = q[1]
            marker.pose.orientation.z = q[2]
            marker.pose.orientation.w = q[3]
            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.scale.z = 0.
            marker.color.a = 1.0
            marker.color.r = 1.0
            ma.markers.append(marker)

        self.marker_pub.publish(ma)

        
# TODO calculation time too long, need to be optimized or rewrite by C++

if __name__ == '__main__':
    loop = 0
    T = VehicleConst.T + 1
    dt = VehicleConst.dt
    wheel_constrain = True
    optimizer = AWSMassPointOptimizer(VehicleConst.T, dt, wheel_constrain=wheel_constrain)
    # optimizer = AWS_PID()
    ri = ROS_INTERFACE()


    dim = optimizer.nx
    traj_dim = optimizer.init_guess_dim
    # for i in range(100):
    rate = rospy.Rate(60)
    last_robot_rpy_orien = None
    while last_robot_rpy_orien is None:
        print('last_robot_rpy_orien not found')
        try:
            (trans,rot) = ri.listener.lookupTransform('world', 'body', rospy.Time(0))
            last_robot_pose = np.array(trans)
            last_robot_rpy_orien = quart_to_rpy_xyzw(*rot)
            last_robot_pose[2] = last_robot_rpy_orien[2]
            time = rospy.Time.now()
        except:
            pass
    while not rospy.is_shutdown():
        print('='*20,loop,'='*20)
        loop += 1
        (trans,rot) = ri.listener.lookupTransform('world', 'body', rospy.Time(0))
        print('current state:', trans, rot)
        robot_pose = np.array(trans)
        robot_rpy_orien = quart_to_rpy_xyzw(*rot)
        robot_pose[2] = robot_rpy_orien[2]
        print('time:',(rospy.Time.now()-time).to_sec())
        robot_v = (robot_pose-last_robot_pose)/(rospy.Time.now()-time).to_sec()
        time = rospy.Time.now() # update last time
        last_robot_pose = robot_pose # update last robot pose
        if ri.goal is None:
            continue 
        points = np.array([robot_pose, (robot_pose+ri.goal)/2 ,ri.goal])
        points[...,:2] = points[...,:2] - robot_pose[...,:2]
        print(points)
        vel = np.array([
            # robot_v,
            [0,0,0],
            [0,0,0]])
        print('vel:', vel)

        traj, td = min_jerk(points,T,vel)
        target_traj = traj.copy()

        vs = xyyaw2v(target_traj, dt=VehicleConst.dt)
        traj_stats = np.zeros((T-1,dim))
        traj_stats[:,:3] = traj[1:]
        # traj_stats[:,3:6] = vs

        robot_vel = vel_world_to_robot(robot_v, robot_pose[2])
        init_state = np.array([[0.]*dim])
        init_state[0,:3] = traj[0]
        init_state[0,3:6] = robot_vel
        if wheel_constrain:
            vx,vy = dxdydyaw2wheelstate(robot_vel[0], robot_vel[1], robot_vel[2],vs=True)
            init_state[0,6:10] = vx.T
            init_state[0,10:] = vy.T
        try:
            print(init_state)
            print('try solving...')
            with time_limit(1):
                try:
                    result_traj, control = traj_smooth(traj_stats, init_state, optimizer)
                    print('solved!')
                except:
                    pass
            pub_traj = result_traj[...,:3].copy()
            pub_traj[...,:2] = pub_traj[...,:2]+robot_pose[...,:2]
            ri.publish_traj(pub_traj)

            print('control:', control[0])
            vel_command = Twist()
            real_dt = (rospy.Time.now()-time).to_sec()
            print('real_dt:',real_dt)
            vel_command.linear.x = control[0,0]*real_dt+robot_vel[0]
            vel_command.linear.y = control[0,1]*real_dt+robot_vel[1]
            vel_command.angular.z = control[0,2]*real_dt+robot_vel[2]
            ri.vel_pub.publish(vel_command)
        except TimeoutException as e:
            pass

            
