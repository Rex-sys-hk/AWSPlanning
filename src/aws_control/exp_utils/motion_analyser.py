import argparse
import os, sys
from tkinter import font
from matplotlib import font_manager
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/../')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from plan_utils.dt_mpc_controller import np_RotationMatrix
from plan_utils.constants import VehicleConst
from plan_utils.common import np_dxdydyaw2wheelstate, pi2pi

def plot_motion_records(folder, body_tf=None, wheel_tf=None, 
                        vs=None, path=None, visualize=True,
                        start=0, end=-1):
    body_tf = np.load(f'{folder}/body_tf.npy',allow_pickle=True) \
        if body_tf is None else body_tf
    wheel_tf = np.load(f'{folder}/wheel_tf.npy',allow_pickle=True) \
        if wheel_tf is None else wheel_tf
    vs = np.load(f'{folder}/vs.npy',allow_pickle=True) \
        if vs is None else vs
    body_tf = body_tf[start:end]
    wheel_tf = wheel_tf[start:end]
    vs = vs[start:end]
    path = np.load(f'{folder}/path.npy',allow_pickle=True) \
        if path is None else path
    vx,vy = np_dxdydyaw2wheelstate(vs, 0)
    right_r = np.arctan2(vy,vx)
    wv = np.sqrt(vx**2+vy**2)
    slide_rate = (-np.cos(2*(right_r-wheel_tf))+1)*wv
    if not visualize:
        return np.max(np.abs(slide_rate[...,:2]),axis=-1)
    slide_rate = slide_rate[...,:2]

    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    plt.figure(figsize=(8.5,3.5),dpi=300)
    plt.plot(path[:,0], path[:,1], 
             label='reference path', 
             linewidth=0.8, color='black')
    plt.plot(body_tf[:,0], body_tf[:,1], 
             label='vehicle motion record', 
             linewidth=0.8, color='orange')
    slide_rate = slide_rate[::2]
    for b,w in zip(body_tf[::20],wheel_tf[::20]):
        patch = Rectangle(
            (b[0]-VehicleConst.half_l,b[1]-VehicleConst.half_w), 
            VehicleConst.WHEEL_BASE, VehicleConst.WHEEL_WHIDTH, 
            b[2]*180/np.pi, 
            rotation_point='center', 
            fill=False, 
            edgecolor='orange',
            linewidth=0.2)
        plt.gca().add_patch(patch)
    # for w in zip(wheel_tf[::20]:
        for i in range(VehicleConst.wheel_num):
            wheel_pos = np_RotationMatrix(b[2])@\
                np.array(VehicleConst.wheel_positions[i]).T\
                +np.array(b[:2])
            patch = Rectangle(
                (wheel_pos[0]-0.25,
                 wheel_pos[1]-0.05), 
                0.5, 0.1, 
                (w[i]+b[2])*180/np.pi, 
                rotation_point='center', 
                fill=False, 
                edgecolor='blue',
                linewidth=0.2)
            plt.gca().add_patch(patch)
    plt.axis('equal')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.tight_layout()
    plt.legend(loc='upper right', 
                fontsize=15,
               )
    plt.savefig(f'{folder}/path.png')
    plt.savefig(f'{folder}/path.pdf')


    plt.figure(figsize=(8.5,2.5),dpi=300)
    plt.plot(vs[:,0], label='vx')
    plt.plot(vs[:,1], label='vy')
    plt.plot(vs[:,2], label='w')
    plt.legend()
    plt.tight_layout()


    plt.figure(figsize=(8.5,2.5),dpi=300)
    plt.plot(wheel_tf[:,0], label='wheel_tf_0')
    plt.plot(wheel_tf[:,1], label='wheel_tf_1')
    plt.plot(wheel_tf[:,2], label='wheel_tf_2')
    plt.plot(wheel_tf[:,3], label='wheel_tf_3')
    plt.plot(right_r, label='right_r')
    plt.legend()
    plt.tight_layout()


    plt.figure(figsize=(8.5,2.),dpi=300)
    plt.plot(np.max(np.abs(slide_rate),axis=-1), label='max slide ratio')
    plt.xlim(0, len(slide_rate))
    plt.ylim(-0.3, 4.7)
    plt.xlabel('time step')
    plt.ylabel('slide ratio')
    plt.tight_layout()
    plt.legend(
               loc='upper left',
               fontsize=15,
            #    prop = font_manager.FontProperties(
            #        family='Times New Roman', size=15)
               )
    plt.savefig(f'{folder}/slide_ratio.png')
    plt.savefig(f'{folder}/slide_ratio.pdf')



    plt.show()


if __name__ == '__main__':
    # folder = 'motion_records/omni'
    from motion_recorder import folder_name
    arg = argparse.ArgumentParser()
    arg.add_argument('--folder', type=str, default=folder_name)
    arg.add_argument('--start', type=int, default=0)
    arg.add_argument('--end', type=int, default=-1)
    args = arg.parse_args()

    plot_motion_records(folder = args.folder, start=args.start, end=args.end)