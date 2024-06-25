import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/../')
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from plan_utils.common import AWSConstructSamplingTable
from plan_utils.constants import VehicleConst

folder = 'runs/rsample_results'

def R(theta):
    return np.array([[math.cos(theta), -math.sin(theta)],
                    [math.sin(theta), math.cos(theta)]])

def dR(theta,dtheta):
    return np.array([[-math.sin(theta), -math.cos(theta)*dtheta],
                    [math.cos(theta), -math.sin(theta)*dtheta]])*dtheta

def test_AWSConstructSamplingTable():
    degree_symbol = chr(176) 
    slim = VehicleConst.MAX_STEER_DEG
    # VehicleConst.MAX_STEER = math.pi*(slim/180)
    steer_lim = f"{slim}{degree_symbol}"
    tb,rb = AWSConstructSamplingTable(8, 8, 
                                   np.linspace(-math.pi/2, math.pi/2, 8))
    # ws = np.linspace(-math.pi/2, math.pi/2, 10)
    c_size = 10
    inch_per_cm = 0.393701
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    plt.figure(figsize=(8.3*inch_per_cm,8.3*inch_per_cm),dpi=300)
    plt.rcParams['font.size'] = c_size
    plt.rcParams['axes.labelsize'] = c_size
    plt.rcParams['axes.titlesize'] = c_size
    ax = plt.gca()
    plt.scatter(0,0, c='k', s = 5, label='rotating center')
    plt.scatter(0,0, c='r', s = 5, label='velocity vector end')
    plt.scatter(0,0,
                c = 'b', s = 5, 
                label = 'next location')
    for r,v in zip(rb, tb):
        rotate_r = R(r[2])@np.array(r[:2]) - np.array(r[:2])
        rotate_r = rotate_r.T
        plt.scatter(-r[0], -r[1], c='k', s = 5, )
        plt.scatter(v[0], v[1], c='r', s = 5, )
        plt.scatter(rotate_r[0], rotate_r[1],
                    c = 'b', s = 5, 
                    )
        center = (-r[0], -r[1])
        radius = np.linalg.norm(np.array([r[0], r[1]]),2,axis=-1)
        start_angle = np.arctan2(r[1], r[0])
        t1 = min(start_angle, start_angle+r[2])
        t2 = max(start_angle, start_angle+r[2])
        arc = patches.Arc(center, 
                    2 * radius, 
                    2 * radius, 
                    angle=0, 
                    theta1=np.degrees(t1), 
                    theta2=np.degrees(t2),
                    linewidth=0.2,
                    )
        ax.add_patch(arc)
    # rect = patches.Rectangle((-VehicleConst.half_l, -VehicleConst.half_w), 
    #                          VehicleConst.WHEEL_BASE, 
    #                          VehicleConst.WHEEL_WHIDTH, 
    #                          linewidth=1, 
    #                          edgecolor='r', 
    #                          facecolor='none')
    # ax.add_patch(rect)
    # plt.xscale('symlog', linthresh=3)
    # plt.yscale('symlog', linthresh=3)
    # plt.legend()
    plt.axis('equal')
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title(f'Forward Samples (steer lim = {steer_lim})')
    plt.tight_layout()
    # plt.show()
    suffix = VehicleConst._control_center_offset
    plt.savefig(f'{folder}/rsample_vis_{steer_lim}_offset_{suffix}.pdf')
    plt.savefig(f'{folder}/rsample_vis_{steer_lim}_offset_{suffix}.png')
    plt.savefig('rsample_vis.png')
    print('saved to', f'{folder}/rsample_vis_{steer_lim}_offset_{suffix}.png')
    # plt.show()

if __name__ == '__main__':
    test_AWSConstructSamplingTable()