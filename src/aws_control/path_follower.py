from plan_utils.map_utils import MapUI
import pickle, os
from plan_utils.constants import VehicleConst
from plan_utils.mpc_controller import AWSMassPointOptimizer, traj_smooth, dxdydyaw2wheelstate
from plan_utils.dt_mpc_controller import AWSGlobalMassPointOptimizer, np_RotationMatrix, test_dt
from plan_utils.poly_mpc_controller import AWSPolynomialOptimizer, test_poly
from plan_utils.common import (ROS_INTERFACE, coordinate_transform_to_robot, coordinate_transform_to_world, np_dxdydyaw2wheelstate, pi2pi, quart_to_rpy_wxyz, 
                               quart_to_rpy_xyzw, xyyaw2v, 
                               vel_world_to_robot, rpy2quaternion,
                               time_limit)
from plan_utils.ros_mpc_follower_test import TimeoutException
import rospy
import numpy as np
from geometry_msgs.msg import Twist, PoseArray, PoseStamped
from trajectory_msgs.msg import JointTrajectory
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from exp_utils.start_goal_analys_logs import folder


casadi_occ = False

def path_callback(msg, path_dict):
    # global traj
    trans = np.array([[p.position.x, p.position.y, p.orientation.z] 
                        for p in msg.poses])
    rot = np.array([
        quart_to_rpy_xyzw(p.orientation.x, p.orientation.y,
                            p.orientation.z, p.orientation.w) 
                            for p in msg.poses])
    path_dict['path'] = np.concatenate([trans[...,:2], 
                                        rot[...,-1:]], axis=-1)
    path_dict['new_path'] = True
    path_dict['wait_new_path'] = False

def vel_callback(msg, robot_odom):
    # robot states at that time
    rot = quart_to_rpy_xyzw(msg.pose.pose.orientation.x, 
                            msg.pose.pose.orientation.y, 
                            msg.pose.pose.orientation.z, 
                            msg.pose.pose.orientation.w)
    
    robot_odom['robot_pose'] = np.array([msg.pose.pose.position.x, 
                                            msg.pose.pose.position.y, 
                                            rot[-1]])
    robot_odom['robot_vel'] = np.array([msg.twist.twist.linear.x, 
                                        msg.twist.twist.linear.y, 
                                        msg.twist.twist.angular.z])
    robot_odom['received_vel'] = True

def path_v_callback(msg, path_dict):
    print("recevied new msg, point num: ", len(msg.points))
    path = np.zeros((len(msg.points),6))
    dts = []
    path_dict['path_static']['init_path_received_times'].append(
        rospy.Time.now().to_sec())
    while casadi_occ:
        pass
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
    if path.shape[0]<=3:
        path = np.pad(path,((0,3-path.shape[0]),(0,0)), 
                mode='edge')
    dts = np.array(dts)
    dts = np.diff(dts, axis=0)
    path[1:,2] = path[0,2]+np.cumsum(path[0:-1,5])
    # path[:,2] = pi2pi(path[:,2])
    vs = path[:,3:6]
    vs = np.concatenate([
        # vs[1:2,:]*1e-6,
        vs[1:,:]/dts[:,None], 
        # vs[1:,:]*2, 
        vs[-1:,:]*1e-8#*1e-8
        ], 
        axis=0)
    path_dict['path_static']['opt_started_times'].append(
        rospy.Time.now().to_sec())   
    result_traj, control = test_dt(path[:,:3], vs, dts, 
                                   mapui=path_dict['map'],
                                   VISULIZE=False,
                                   result_dict=path_dict['result_dict'],
                                   no_smooth=VehicleConst.no_smooth_mode,
                                   )
    result_traj[-1,3:6] = result_traj[-1,3:6]*1e-6
    path_dict['path_static']['opt_finished_times'].append(
        rospy.Time.now().to_sec())
    # result_traj, control = test_poly(path[:,:3], vs, dts)
    # result_traj = path
    # control = np.zeros((result_traj.shape[0],4))
    path_dict['path'] = result_traj
    path_dict['control'] = control
    path_dict['new_path'] = True
    path_dict['wait_new_path'] = False
    path_dict['total'] += 1
    path_dict['path_static']['paths'].append(result_traj)
    path_dict['path_static']['controls'].append(control)
    path_dict['path_static']['goal_received_times'].append(
        path_dict['goal_received_time'])
    if path_dict['result_dict']['failed']:
        path_dict['failed'] += 1
    path_dict['path_static']['opt_success'].append(
        not path_dict['result_dict']['failed'])
    path_dict['path_static']['headless_opt_times'].append(
        path_dict['result_dict']['opt_time'])
    print("fail rate:", path_dict['failed']/path_dict['total'])
    return
    # rospy.spin()


def local_routing(curreent_pose,origin_traj,index,T=VehicleConst.T):
    if origin_traj['new_path']:
        origin_traj['new_path'] = False
        index = 0
    find_nearest = lambda x, poses: \
        np.argmin(np.linalg.norm(np.array(poses)[:, :2] \
                                    - np.array(x)[None,:2], axis=1))
    nearest = find_nearest(curreent_pose, 
                            origin_traj['path'][index:index+T])\
                            +index
    
    index = nearest
    traj_length = origin_traj['path'].shape[0]
    ind_start = min(nearest, traj_length-1)
    ind_end = min(nearest+T, traj_length)

    all_zero_index = np.argmax(
        np.sum(
            path_dict['path'][ind_start:ind_start+T, 3:6]**2,
            axis=-1)
            <=1e-5) 
    ind_end = min(all_zero_index+ind_start, ind_end) \
        if all_zero_index else ind_end

    traj = origin_traj['path'][ind_start:ind_end]
    traj = np.pad(traj,((0,T-traj.shape[0])), 
                    mode='edge')
    if ind_start >= traj_length-1:
        path_dict['wait_new_path'] = True
    return traj, index


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--path_topic', type=str, 
                           default='/path_sample_vel')
    argparser.add_argument('--folder', type=str,
                            default=folder)
    args = argparser.parse_args()
    folder = args.folder
    plot_show = False
    loop = 0
    # T = VehicleConst.T
    T = 3
    dt = VehicleConst.dt
    wheel_constrain = VehicleConst.follow_with_constrain \
        and not VehicleConst.no_wheel_constrain_mode
    opt_with_map = False
    optimizer = AWSGlobalMassPointOptimizer(T, 
                                            dt, 
                                            wheel_constrain=wheel_constrain,
                                            var_dt=True,
                                            free_end=True,
                                            following_mode=True,
                                            simplify_model=True,
                                            )
    dim = optimizer.nx
    traj_dim = optimizer.init_guess_dim
    ## initialize ros node
    ri = ROS_INTERFACE(node_name='follower')
    mapui = MapUI(ri, map_topic='/map') if opt_with_map else None
    # subscribe path
    # path_static = {'paths':[], 
    #                'controls':[],
    #                 'opt_success':[],
    #                 'goal_received_times':[],
    #                 'init_path_received_times':[],
    #                 'opt_started_times':[],
    #                 'opt_finished_times':[],
    #                 'headless_opt_times':[],
    #                 'done_following_times':[],
    #                 'tf_receive_times_list':[],
    #                 'robot_poses_list':[],
    #                 'robot_vels_list':[],
    #                 'robot_tfs_list':[],
    #                 'wheel_tfs_list':[],
    #                 }
    path_dict = {
                'initial_path':None,
                'path':None, 
                'control':None,
                'failed':0,
                'total':0,
                'new_path':False,
                'map':mapui,
                'goal_received_time':0, #'/path_sample_vel
                # 'opt_time':0,
                # 'opt_success':True,
                # 'opt_started_time':0,
                # 'opt_finished_time':0,
                'path_static':{'paths':[], 
                   'controls':[],
                    'opt_success':[],
                    'goal_received_times':[],
                    'init_path_received_times':[],
                    'opt_started_times':[],
                    'opt_finished_times':[],
                    'headless_opt_times':[],
                    'done_following_times':[],
                    'tf_receive_times_list':[],
                    'robot_poses_list':[],
                    'robot_vels_list':[],
                    'robot_tfs_list':[],
                    'wheel_tfs_list':[],
                    },
                'result_dict':{'opt_time':0.,
                               'failed':True},
                 }
    # read pickle
    if not os.path.exists(f'{folder}/path_static.pkl'):
        os.makedirs(folder, exist_ok=True)
        pickle.dump(path_dict['path_static'], 
                    open(f'{folder}/path_static.pkl', 'wb'))
    with open(f'{folder}/path_static.pkl', 'rb') as f:
        path_dict['path_static'] = pickle.load(f)
    robot_odom = {'robot_vel': np.zeros((3)), 
                  'robot_pose': np.zeros((3)),
                  'robot_tf': np.zeros((3)),
                  'wheel_tf': np.zeros((VehicleConst.wheel_num)),
                  'tf_receive_time': 0,
                  'received_vel': False}
    static_odom = {'robot_vels': [],
                     'robot_poses': [],
                     'robot_tfs': [],
                     'wheel_tfs': [],
                     'tf_receive_times':[]
                     }
    ri.add_publisher('/need_recording', Bool, queue_size=1)
    def goal_call_back(msg, path_dict):
        path_dict['goal_received_time'] = \
                    rospy.Time.now().to_sec()
        path_dict['new_path'] = True
        ri.publish('/need_recording', Bool(True))
    ri.add_subscriber('/move_base_simple/goal', 
                      PoseStamped, 
                      goal_call_back, 
                      path_dict)
    if not args.path_topic:
        ri.add_subscriber('path_planning', PoseArray, 
                          path_callback, path_dict)
    if args.path_topic:
        ri.add_subscriber(args.path_topic,  #'/path_sample_vel'
                          JointTrajectory, 
                          path_v_callback, 
                          path_dict)
    # subscribe robot odom
    ri.add_subscriber('odom', Odometry, vel_callback, robot_odom)
    # publish trajectory for visulization
    ri.add_publisher('plantraj', MarkerArray, queue_size=1)
    ri.add_publisher('planpath', MarkerArray, queue_size=1)
    ri.add_publisher('smoothpath', MarkerArray, queue_size=1)
    ri.add_publisher('current_pose', MarkerArray, queue_size=1)

    # publish control command
    ri.add_publisher('robot_control_command', Twist, queue_size=10)
    ##### after ros interface definition
    control_list = []
    commands = []
    times = []
    routing_index = 0
    initial_guess = np.zeros((T+1,dim))
    ##### looping when simulator not running
    FoundTFFlag = False
    while (not FoundTFFlag) and (not rospy.is_shutdown()):
        try:
            (trans,rot) = ri.get_tf('world', 'Imu_Sensor', rospy.Time(0))
            FoundTFFlag = True
            print('Found TF!', trans, rot)
        except:
            print('last_robot_rpy_orien not found')
            pass

    #### main loop
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        if not robot_odom['received_vel']:
            ri.publish('/need_recording',
                       Bool(True if path_dict['new_path'] else False))
            continue
        if  path_dict['path'] is None:
            ri.publish('/need_recording',
                       Bool(True if path_dict['new_path'] else False))
            continue
        current_robot_v = robot_odom['robot_vel']
        if path_dict['wait_new_path'] and (current_robot_v**2<=5e-3).all():
            vel_command = Twist()
            vel_command.linear.x = current_robot_v[0]*1e-5
            vel_command.linear.y = current_robot_v[1]*1e-5
            vel_command.angular.z = current_robot_v[2]*1e-5
            ri.publish('robot_control_command',vel_command)
            vaccant_ma = ri.make_marker_array(
                np.array([[0,0,0,0,0,0]]),
                frame_id='world',
                ns='my_namespace',
                m_type=Marker.ARROW,
                action=Marker.DELETEALL,
                scale=[0.2, 0.1, 0.1],
                color=[1.0, 0.0, 0.0, 1.0])
            ri.publish('smoothpath', vaccant_ma)
            ri.publish('planpath', vaccant_ma)
            ri.publish('plantraj', vaccant_ma)
            routing_index = 0
            if len(static_odom['robot_tfs'])>3:
                print('recorded new path')
                path_dict['path_static']['done_following_times'].append(
                    rospy.Time.now().to_sec())
                path_dict['path_static']['tf_receive_times_list'].append(
                    static_odom['tf_receive_times'].copy())
                path_dict['path_static']['robot_vels_list'].append(
                    static_odom['robot_vels'].copy())
                path_dict['path_static']['robot_poses_list'].append(
                    static_odom['robot_poses'].copy())
                path_dict['path_static']['robot_tfs_list'].append(
                    static_odom['robot_tfs'].copy())
                path_dict['path_static']['wheel_tfs_list'].append(
                    static_odom['wheel_tfs'].copy())
                static_odom['robot_vels'] = []
                static_odom['robot_poses'] = []
                static_odom['robot_tfs'] = []
                static_odom['wheel_tfs'] = []
                static_odom['tf_receive_times'] = []
                ri.publish('/need_recording', Bool(False))
                os.makedirs(folder, exist_ok=True)
                with open(f'{folder}/path_static.pkl', 'wb') as f:
                    pickle.dump(path_dict['path_static'], f)
                print(f'file saved in {folder}/path_static.pkl')
            continue
        
        time = rospy.Time.now() # update last time

        ## for recording
        ri.publish('/need_recording', Bool(True))
        robot_tf_trans, robot_tf_rot = \
            ri.get_tf('world', 'body', rospy.Time(0))
        static_odom['robot_tfs'].append(np.array([robot_tf_trans[0], 
                                           robot_tf_trans[1],
                                           robot_tf_rot[2]]))
        static_odom['wheel_tfs'].append(np.array([
            ri.get_tf('body', f, rospy.Time(0))[1][2] 
            for f in VehicleConst.wheel_frames]))
        static_odom['robot_poses'].append(robot_odom['robot_pose'])
        static_odom['robot_vels'].append(robot_odom['robot_vel'])
        static_odom['tf_receive_times'].append(time.to_sec())
        

        if (current_robot_v**2<=5e-3).all() \
            and (path_dict['path'][routing_index, 3:6]**2<=5e-3).all() \
            and routing_index < path_dict['path'].shape[0]-1:
            routing_index += 1
        print('='*20,loop,'='*20)
        loop += 1
        if (current_robot_v==0).all():
            current_robot_v = np.array([0.01, 0.0, 0.0])
        print('robot_v:', current_robot_v)
        robot_tf_trans, robot_tf_rot = \
            ri.get_tf('world', 'body', rospy.Time(0))
        initial_guess[0,:2] = robot_tf_trans[:2]
        initial_guess[0,2] = robot_tf_rot[2]
        ma = ri.make_marker_array(
            np.array([[initial_guess[0,0],
                        initial_guess[0,1],0,0,0,
                        initial_guess[0,2]]]),
            frame_id='world',
            ns='my_namespace',
            m_type=Marker.CUBE,
            action=Marker.ADD,
            scale=[2.52, 1.1, 0.1],
            color=[1.0, 1.0, 0.0, .5])
        ri.publish('current_pose', ma)


        initial_guess[0, 3:5] = \
                np_RotationMatrix(initial_guess[0,2])@current_robot_v[:2]
        initial_guess[0, 5] = current_robot_v[2]
        if args.path_topic:
            traj, routing_index = local_routing(initial_guess[0,:3],
                                                path_dict,
                                                routing_index,
                                                T=T)
            # initial_guess[1:,:dim] = traj[:,:dim]
            initial_guess[1:,:6] = traj[:,:6]
        else:
            initial_guess[1:,:3] = path_dict['path'].copy()
            initial_guess[1:,3:6] = np.diff(
                initial_guess[:, :3], axis=0)\
                /optimizer.dt

        if wheel_constrain:
            vx,vy = np_dxdydyaw2wheelstate(initial_guess[:,3:6], 
                                           0,
                                        # initial_guess[:,2:3]
                                        )
            ### TODO remap to global frame
            initial_guess[:,6:10] = vx
            initial_guess[:,10:] = vy
        try:
            print('try solving...')
            with time_limit(2):
                casadi_occ = True
                result_traj, control = traj_smooth(initial_guess, 
                                                    optimizer)
                print('feed back received!')
                print(control[0])
            casadi_occ = False
            ## publish planned trajectory for visulization
            pub_traj = np.zeros((result_traj.shape[0],6))
            pub_traj[:,:2]= result_traj[...,:2]
            pub_traj[:,5] = result_traj[...,2]
            ma = ri.make_marker_array(pub_traj,
                                        frame_id='world',
                                        ns='my_namespace',
                                        m_type=Marker.ARROW,
                                        action=Marker.ADD,
                                        scale=[0.2, 0.1, 0.1],
                                        color=[1.0, 0.0, 0.0, 1.0])
            ri.publish('plantraj',ma)
            pub_traj[:,:2] = initial_guess[1:,:2]
            pub_traj[:,5] = initial_guess[1:,2]
            ma = ri.make_marker_array(pub_traj,
                                frame_id='world',
                                ns='my_namespace',
                                m_type=Marker.ARROW,
                                action=Marker.ADD,
                                scale=[0.2, 0.1, 0.1],
                                color=[0.0, 1.0, 0.0, 1.0])
            ri.publish('planpath', ma)
            pub_s_traj = np.zeros((path_dict['path'].shape[0],6))
            pub_s_traj[:,:2] = path_dict['path'][...,:2]
            pub_s_traj[:,5] = path_dict['path'][...,2]
            ma = ri.make_marker_array(pub_s_traj,
                                frame_id='world',
                                ns='my_namespace',
                                m_type=Marker.ARROW,
                                action=Marker.ADD,
                                scale=[0.2, 0.1, 0.1],
                                color=[0.0, 0.0, 1.0, 1.0])
            ri.publish('smoothpath', ma)

            ## publish control command
            real_dt = (rospy.Time.now()-time).to_sec()
            control_dt = control[0,3] if control[0,3] else dt
            control_dt = max(0.1/control_dt, 0.1)
            print('\n real_dt:',real_dt, 
                  '\n var_dt: ', control[0,3],
                  '\n control_dt:', control_dt,
                  '\n control:', control[0,:3],
                  '\n vs: \n', result_traj[:,3:6],
                  '\n init vs: \n', initial_guess[:,3:6]
            )
            # control_dt = 0.1
            # control_dt = dt
            vel_command = Twist()
            control[0,:2] = np_RotationMatrix(
                -initial_guess[0,2])@control[0,:2]
            vel_command.linear.x = control[0,0]*control_dt+current_robot_v[0]
            vel_command.linear.y = control[0,1]*control_dt+current_robot_v[1]
            vel_command.angular.z = control[0,2]*control_dt+current_robot_v[2]
            ri.publish('robot_control_command',vel_command)
            control_list.append(control[0])
            commands.append([vel_command.linear.x, 
                             vel_command.linear.y, 
                             vel_command.angular.z])
            print('commands:', commands[-1])
            times.append(rospy.Time.now().to_sec())
            # rospy.spin()
            # rate.sleep()
            if plot_show:
                # continue
                plt.cla()
                # ax1 = plt.subplot(211)
                # ax2 = plt.subplot(212)
                plt.plot(np.array(times)-times[0], 
                        np.array(control_list)[...,0],
                        '-r',label='dvx')
                plt.plot(np.array(times)-times[0], 
                        np.array(control_list)[...,1],
                        '-g',label='dvy')
                plt.plot(np.array(times)-times[0], 
                        np.array(control_list)[...,2],
                        '-b',label='ddyaw')
                plt.xlabel('Time(s)')
                plt.ylabel('Control States')
                plt.legend()
                plt.savefig('control1.png')
                plt.cla()
                plt.plot(np.array(times)-times[0], 
                        np.array(commands)[...,0],'-r',label='vx')
                plt.plot(np.array(times)-times[0], 
                        np.array(commands)[...,1],'-g',label='vy')
                plt.plot(np.array(times)-times[0], 
                        np.array(commands)[...,2],'-b',label='dyaw')
                plt.xlabel('Time(s)')
                plt.ylabel('Velocity States')
                plt.legend()
                # plt.legend()
                # plt.pause(0.01)
                plt.savefig('control2.png')

        except TimeoutException as e:
            pass
    # os.makedirs(folder, exist_ok=True)
    # with open(f'{folder}/path_static.pkl', 'wb') as f:
    #     pickle.dump(path_dict['path_static'], f)
    # print(f'file saved in {folder}/path_static.pkl')
    # print('done!')