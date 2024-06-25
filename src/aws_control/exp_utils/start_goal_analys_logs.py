import argparse
import os, sys 
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/../')
import pickle
import numpy as np
from plan_utils.common import pi2pi
from exp_utils.motion_analyser import plot_motion_records
from plan_utils.traj_generator import min_jerk
folder = 'runs/statistic_experiments/tmp'

find_nearest = lambda x, poses: \
    np.argmin(np.linalg.norm(np.array(poses)[:, :2] \
                                - np.array(x)[None,:2], axis=1))
# nearest = find_nearest(curreent_pose, 
#                         origin_traj['path'][index:index+T])\
#                         +index

def load_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    print('loaded data from', file_path)
    return data

def save_data(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    print('saved data to', file_path)

def trim_data(data,start,end):
    for k,v in data.items():
        if type(v) is float:
            continue
        data[k] = v[start:end]

# def parse_data(data):

def main():
    start_iter = 0
    file_path = \
    f'{folder}/path_static.pkl'    
    data = load_data(file_path)
    for k, v in data.items():
        if type(v) in [np.ndarray, list]:
            print(k, len(v))
        elif type(v) == float:
            print(k, v)
        else:
            print(k, type(v))

    processed_list_dict = {'step20_time_to_search':[],
                            'step20_time_to_opt':[],
                            'step20_time_to_opt_headless':[],
                            'step20_time_to_follow':[],
                            'mean_dis_error':[],
                            'mean_yaw_error':[],
                            'mean_slide_ratio':[],
                            'std_slide_ratio':[],
                            'mean_d_steering':[],
                            'std_d_steering':[],
                            'mean_vs':[],
                            'mean_acc':[],
                            'mean_jerk':[],
                            'opt_succes_list':[],
                            'mean_control_rate':[],
                            }
    
    success_list = data['opt_success'][start_iter:]
    opt_succes_rate = np.sum(np.array(success_list))/len(success_list)
    processed_list_dict['opt_succes_list'] = success_list

    for i in range(len(data['paths'])):
        step_num = len(data['paths'][i])
        path = np.array(data['paths'][i])

        robot_poses = np.array(data['robot_poses_list'][i])
        body_tfs = np.array(data['robot_tfs_list'][i])
        wheel_tfs = np.array(data['wheel_tfs_list'][i])
        vs = np.array(data['robot_vels_list'][i])
        control_times = np.diff(
            np.array(data['tf_receive_times_list'][i]),axis=0)[:, None]
        mean_control_rate = 1./np.mean(control_times)

        ### times
        time_to_search = data['init_path_received_times'][i] \
                        -data['goal_received_times'][i]
        time_to_opt = data['opt_finished_times'][i] \
                        -data['opt_started_times'][i] 
        time_to_opt_headless = data['headless_opt_times'][i]
        time_to_follow = data['done_following_times'][i] \
                        - data['opt_finished_times'][i]
        step20_time_to_search = time_to_search/step_num*20
        step20_time_to_opt = time_to_opt/step_num*20
        step20_time_to_opt_headless = time_to_opt_headless/step_num*20
        step20_time_to_follow = time_to_follow/step_num*20


        ### tf erros
        errors = []
        for p in path:
            min_index = find_nearest(p, body_tfs)
            body_tf = body_tfs[min_index]
            errors.append(p[:3]-body_tf[:3])
        errors = np.array(errors)
        mean_dis_error = np.mean(np.linalg.norm(errors[:,:2],axis=-1))
        mean_yaw_error = np.mean(np.abs(pi2pi(errors[:,2])))

        ### dynamics
        slide_ratio = plot_motion_records(folder,
                            body_tf=body_tfs, 
                            wheel_tf=wheel_tfs,
                            vs=vs, 
                            path=path,
                            visualize=False)
        mean_slide_ratio = np.mean(slide_ratio)
        std_slide_ratio = np.std(slide_ratio)

        d_steering = pi2pi(np.diff(wheel_tfs[:,2],axis=0))/control_times
        mean_d_steering = np.mean(np.abs(d_steering))
        std_d_steering = np.std(np.abs(d_steering))

        # n_vs = np.linalg.norm(vs[:,:2],axis=-1)
        n_vs = np.abs(vs)
        mean_vs = np.mean(n_vs, axis=0)

        accs = np.diff(vs,axis=0)/control_times
        # n_accs = np.linalg.norm(accs[:,:2],axis=-1)
        n_accs = np.abs(accs)
        mean_acc = np.mean(n_accs, axis=0)

        jerk = np.diff(accs,axis=0)#/control_times[1:]
        # n_jerk = np.linalg.norm(jerk,axis=-1)
        n_jerk = np.abs(jerk)
        mean_jerk = np.mean(n_jerk, axis=0)

        processed_list_dict[
            'step20_time_to_search'
            ].append(step20_time_to_search)
        processed_list_dict[
            'step20_time_to_opt'
            ].append(step20_time_to_opt)
        processed_list_dict[
            'step20_time_to_opt_headless'
            ].append(step20_time_to_opt_headless)
        processed_list_dict[
            'step20_time_to_follow'
            ].append(step20_time_to_follow)
        processed_list_dict[
            'mean_dis_error'
            ].append(mean_dis_error)
        processed_list_dict[
            'mean_yaw_error'
            ].append(mean_yaw_error)
        processed_list_dict[
            'mean_slide_ratio'
            ].append(mean_slide_ratio)
        processed_list_dict[
            'std_slide_ratio'
            ].append(std_slide_ratio)
        processed_list_dict[
            'mean_d_steering'
            ].append(mean_d_steering)
        processed_list_dict[
            'std_d_steering'
            ].append(std_d_steering)
        processed_list_dict[
            'mean_vs'
            ].append(mean_vs)
        processed_list_dict[
            'mean_acc'
            ].append(mean_acc)
        processed_list_dict[
            'mean_jerk'
            ].append(mean_jerk)
        processed_list_dict[
            'mean_control_rate'
            ].append(mean_control_rate)
        
    print('opt_succes_rate:', opt_succes_rate)
    print(success_list)
    # print(processed_list_dict['step20_time_to_search'])
    validate_only = True
    for k ,v in processed_list_dict.items():
        v = np.array(v)
        success_list = np.array(success_list)
        v = v[success_list] if validate_only \
            and (success_list).any() \
                else v
        print(f'\n=={k}==')
        print('mean: ', np.sum(v,axis=0)/len(v))
        print('std:  ', np.std(v,axis=0))
        print('max:  ', np.max(v,axis=0))
        print('min:  ', np.min(v,axis=0))

    #### Please be very careful to use the following two commonds
    # trim_data(data=data, start=0, end=-1)
    # save_data(data=data, file_path=file_path)


    


if __name__ == "__main__":
    # parse folder name
    args = argparse.ArgumentParser()
    args.add_argument('--folder', type=str, default=folder)
    args = args.parse_args()
    folder = args.folder
    main()