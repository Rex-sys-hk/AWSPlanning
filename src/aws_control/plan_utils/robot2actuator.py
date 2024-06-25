import re
from matplotlib import patches
import numpy as np
import math
from constants import VehicleConst
import matplotlib.pyplot as plt
from common import deg2rad, np_dxdydyaw2wheelstate, pi2pi

def correct_oob_wheel(wheel_v,wheel_s,lim_angle=None):
    lim_angle = np.array(lim_angle)
    if (lim_angle>10).any():
        print('[Warning]: lim_angle is too large, please check the unit, it should be radian rather than degree')
    wheel_s = pi2pi(wheel_s)
    ulim = lim_angle[:,1]
    llim = lim_angle[:,0]
    oob_wheels_ub = np.logical_and(
        wheel_s>(ulim), 
        pi2pi(wheel_s+np.pi)>llim)
    oob_wheels_lb = np.logical_and(
        wheel_s<llim, 
        pi2pi(wheel_s-np.pi)<ulim)
    oob_wheels = np.logical_or(oob_wheels_ub, oob_wheels_lb)
    wheel_v[oob_wheels] *= -1
    wheel_s[oob_wheels] += math.pi
    wheel_s = pi2pi(wheel_s)
    wheel_s[wheel_s>ulim] = ulim[wheel_s>ulim]
    wheel_s[wheel_s<llim] = llim[wheel_s<llim]
    return wheel_v, wheel_s

def k_theta_v_to_control(k,theta,norm_v,lim_angle=90):
    half_l = VehicleConst.WHEEL_BASE/2
    half_w = VehicleConst.WHEEL_WHIDTH/2
    wheel_r = VehicleConst.WHEEL_RADIUS
    # robot heading is x axis, wheel sequence FL, FR, RL, RR
    wheel_positions = np.array([[half_l,half_w],[half_l,-half_w],[-half_l,half_w],[-half_l,-half_w]])
    if k == 0:
        wheel_v = np.ones(4)*norm_v
        wheel_s = np.ones(4)*theta
    if k != 0:
        rot_center = np.array([-math.sin(theta)/k,math.cos(theta)/k])
        omega = norm_v*k
        wheel_r_pos = wheel_positions-rot_center
        wheel_v = np.linalg.norm(wheel_r_pos, axis=-1)*omega
        wheel_s = np.arctan2(wheel_r_pos[:,0],-wheel_r_pos[:,1])

    return correct_oob_wheel(wheel_v,wheel_s,lim_angle)

def relative_v_and_omega_control(vx,vy,omega, lim_angle=90, wheel_rotate_velocity=False):
    # vx,vy,omega are the robot's velocity in robot frame
    # v is each wheel speed
    # s is each steering angle
    wheel_r = VehicleConst.WHEEL_RADIUS
    wheel_vx, wheel_vy = np_dxdydyaw2wheelstate(np.array([[vx,vy,omega]]),0)
    wheel_vx = wheel_vx[0]
    wheel_vy = wheel_vy[0]
    wheel_v = np.stack([wheel_vx,wheel_vy],axis=-1)
    wheel_v = np.linalg.norm(wheel_v,axis=-1)
    wheel_s = np.arctan2(wheel_vy, wheel_vx)
    wheel_v,wheel_s = correct_oob_wheel(wheel_v,wheel_s,lim_angle)
    if wheel_rotate_velocity:
        wheel_v = wheel_v/wheel_r
    return wheel_v,wheel_s

def ICM_control(r_theta, r_rad, r_omega, lim_angle=90, wheel_rotate_velocity=False):
    # wheel_r = VehicleConst.WHEEL_RADIUS
    # wheel_positions = np.array(VehicleConst.wheel_positions)

    omega = r_omega
    vx = r_rad*omega*math.sin(r_theta)
    vy = -r_rad*omega*math.cos(r_theta)
    return relative_v_and_omega_control(vx,vy,omega, lim_angle, wheel_rotate_velocity)

def dxdydyaw2ICM(dx, dy, dyaw):
    radius = np.divide(np.sqrt(dx**2+dy**2),dyaw,out=np.zeros_like(dyaw),where=dyaw!=0)
    theta = np.arctan2(dy,dx) + np.pi/2
    return radius, theta, dyaw

def ktv2ICM(k, theta, v):
    radius = 1/k
    theta = theta
    omega = k*v
    return radius, theta, omega

def d_dxdydyaw2wheelstate(ddx,ddy,ddyaw,wheel_vx=None,wheel_vy=None,vs=False):
    # robot heading is x axis, wheel sequence FL, FR, RL, RR
    wheel_positions = VehicleConst.wheel_positions
    wheel_positions = np.array(wheel_positions)

    d_wheel_vx = ddx-wheel_positions[...,1]*ddyaw
    d_wheel_vy = ddy+wheel_positions[...,0]*ddyaw
    if vs:
        d_wheel_v = 1/np.sqrt(wheel_vx**2+wheel_vy**2)*(wheel_vx*d_wheel_vx+wheel_vy*d_wheel_vy)
        d_wheel_s = 1/(1+(wheel_vy/wheel_vx)**2)*(d_wheel_vy/wheel_vx-wheel_vy*wheel_vx/d_wheel_vx**2)
        return d_wheel_vx,d_wheel_vy,d_wheel_v,d_wheel_s
    return d_wheel_vx,d_wheel_vy

def sample_dxdydayw():
    dim_sample_num = 20+1
    sample_range = 5
    v_coeff = 100
    s_coeff = 3 # pi * 10
    sample_ratio = sample_range/dim_sample_num
    v_range = np.linspace(-sample_range,sample_range,dim_sample_num*2) 
    v_intterval = sample_range/dim_sample_num/5

    v_theta_range = np.linspace(-math.pi,math.pi,dim_sample_num*2)
    v_theta_intterval = math.pi/dim_sample_num/5

    w_range = np.linspace(-sample_range,sample_range,dim_sample_num*2)
    w_intterval = sample_range/dim_sample_num/5

    dt = []
    continuous_range = []


    for v in v_range:
        for v_theta in v_theta_range:
            for w in w_range:
                w_v,w_s = relative_v_and_omega_control(v*math.cos(v_theta),v*math.sin(v_theta),w)
                dt.append(np.concatenate([[v*math.cos(v_theta)],[v*math.sin(v_theta)],[w],w_v,w_s]))
                wv,ws = [],[]
                for noise_v in [-v_intterval,v_intterval]:
                    for noise_v_theta in [-v_theta_intterval,v_theta_intterval]:
                        for noise_w in [-w_intterval,w_intterval]:
                            n_w_v,n_w_s = relative_v_and_omega_control((v+noise_v)*math.cos(v_theta+noise_v_theta),(v+noise_v)*math.sin(v_theta+noise_v_theta),w+noise_w)
                            wv.append(n_w_v-w_v)
                            ws.append(n_w_s-w_s)
                wv = np.array(wv)
                ws = np.array(ws)
                if np.abs(wv).max()>v_coeff*sample_ratio or np.abs(ws).max()>s_coeff*sample_ratio:
                    print(np.abs(wv).max(), np.abs(ws).max())
                    if np.abs(wv).max()>v_coeff*sample_ratio:
                        reason = 0.
                    if np.abs(ws).max()>s_coeff*sample_ratio:
                        reason = 1.
                    if np.abs(wv).max()>v_coeff*sample_ratio and np.abs(ws).max()>s_coeff*sample_ratio:
                        reason = 2.
                    reason_v = np.abs(wv).argmax(-1).argmax(-1) 
                    reason_s = np.abs(ws).argmax(-1).argmax(-1)                   
                    continuous_range.append(np.concatenate([[(v+noise_v)*math.cos(v_theta+noise_v_theta)],[(v+noise_v)*math.sin(v_theta+noise_v_theta)],[w+noise_w],[reason],[reason_v],[reason_s]]))
                    # continue
    return dt, continuous_range

lim_angle = 60
def sample_ICM():
    dim_sample_num = 50+1
    # sample_range = 
    # v_coeff = 100
    # s_coeff = 3 # pi * 10
    # sample_ratio = sample_range/dim_sample_num
    # v_coeff = 100
    # s_coeff = 3 # pi * 10
    # sample_ratio = math.pi/dim_sample_num
    max_dv = 1.5
    max_ds = math.pi/2
    max_v = 2.5
    delta_t = 1.
    r_range = np.linspace(0,10,dim_sample_num) 
    r_intterval = 1.02/dim_sample_num

    theta_range = np.linspace(-math.pi,math.pi,dim_sample_num)
    theta_intterval = math.pi/dim_sample_num

    w_range = np.linspace(-math.pi,math.pi,dim_sample_num)
    w_intterval = math.pi/dim_sample_num

    dt = []
    continuous_range = []


    for r in r_range:
        for theta in theta_range:
            for w in w_range:
                w_v,w_s = ICM_control(theta,r,w,lim_angle=lim_angle)
                # dt.append(np.concatenate([[theta],[r],[w],w_v,w_s]))
                dt.append(np.concatenate([[r*np.cos(theta)], [r*np.sin(theta)],[w],w_v,w_s]))
                wv,ws = [],[]
                for noise_r in [-r_intterval,r_intterval]:
                    for noise_theta in [-theta_intterval,theta_intterval]:
                        for noise_w in [-w_intterval,w_intterval]:
                            n_w_v,n_w_s = ICM_control(theta+noise_theta,r+noise_r,w+noise_w,lim_angle=lim_angle)
                            wv.append(n_w_v-w_v)
                            ws.append(pi2pi(n_w_s-w_s))
                wv = np.array(wv)
                ws = np.array(ws)
                if (np.abs(wv).max()/max_dv)>delta_t or (np.abs(ws).max()/max_ds)>delta_t:
                    print(np.abs(wv).max()/max_dv, np.abs(ws).max()/max_ds)
                    # reason = -1
                    # if np.abs(wv).max()/max_dv>delta_t:
                    #     reason = 0.
                    # if np.abs(ws).max()/max_ds>delta_t:
                    #     reason = 1.
                    # if np.abs(wv).max()/max_dv>delta_t and np.abs(ws).max()/max_ds>delta_t:
                    #     reason = 2.
                    reason = max(np.abs(wv).max(-1).max(-1)/max_dv, np.abs(ws).max(-1).max(-1)/max_ds)
                    # reason_v = np.abs(wv).argmax(-1).argmax(-1) 
                    reason_v = np.abs(wv).max(-1).max(-1)/max_dv
                    # reason_s = np.abs(ws).argmax(-1).argmax(-1)    
                    reason_s = np.abs(ws).max(-1).max(-1)/max_ds
                    # continuous_range.append(np.concatenate([[theta+noise_theta],[r+noise_r],[w+noise_w],[reason],[reason_v],[reason_s]]))
                    continuous_range.append(np.concatenate([[r*np.cos(theta)], [r*np.sin(theta)],[w],[reason],[reason_v],[reason_s]]))

    return dt, continuous_range

# show boundary conditions
if __name__ == "__main__":

    
    # dt, continuous_range = sample_dxdydayw()
    dt, continuous_range = sample_ICM()
    # plot states wrt v, v_theta, w
    # fig = plt.figure(figsize=(20,10),dpi=300)
    # ax1 = fig.add_subplot(241, projection='3d')
    # ax2 = fig.add_subplot(242, projection='3d')
    # ax3 = fig.add_subplot(243, projection='3d')
    # ax4 = fig.add_subplot(244, projection='3d')
    # ax5 = fig.add_subplot(245, projection='3d')
    # ax6 = fig.add_subplot(246, projection='3d')
    # ax7 = fig.add_subplot(247, projection='3d')
    # ax8 = fig.add_subplot(248, projection='3d')
    # axs = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]
    # dt = np.array(dt)
    # for i, ax in enumerate(axs):
    #     im = ax.scatter(dt[:,0],dt[:,1],dt[:,2],c=dt[:,i+3],alpha=0.2,marker='o',cmap='bwr')
    #     ax.axis('equal')
    #     plt.colorbar(im, ax=ax, shrink=0.6, aspect=10)
    # plt.tight_layout()
    # # plt.show()
    # plt.savefig('wheel_states.png')

    # plot boundarys
    fig = plt.figure(figsize=(20,20),dpi=300)
    continuous_range = np.array(continuous_range)
    ax1 = fig.add_subplot(111, projection='3d')
    # ax2 = fig.add_subplot(132, projection='3d')
    # ax3 = fig.add_subplot(133, projection='3d')
    # axs = [ax1, ax2, ax3]
    axs = [ax1]
    import mpl_toolkits.mplot3d.art3d as art3d
    for i, ax in enumerate(axs):
        im = ax.scatter(continuous_range[:,0],continuous_range[:,1],continuous_range[:,2],c = continuous_range[:,-3+i],alpha=0.2,marker='o')
        ax.set_xlabel('ICM_x')
        ax.set_ylabel('ICM_y')
        ax.set_zlabel('ICM_omega')
        rect = patches.Rectangle((-VehicleConst.half_l, -VehicleConst.half_w), VehicleConst.WHEEL_BASE, VehicleConst.WHEEL_WHIDTH, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        art3d.pathpatch_2d_to_3d(rect, z=0, zdir="z")
        ax.axis('equal')
        
        #add color bar
        plt.colorbar(im, ax=ax, shrink=0.6, aspect=10)
    plt.title(f'angle_lim {lim_angle} boundarys')
    plt.tight_layout()
    plt.savefig(f'angle_lim {lim_angle} boundarys.png')
    plt.show()