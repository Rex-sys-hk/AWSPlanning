import numpy as np

if __name__ == '__main__':
    slim = 180/180*np.pi
    phi = 1/180*np.pi
    r = 10
    r = -np.array([r*np.cos(phi), r*np.sin(phi)])

    ulim = np.array([np.cos(slim+np.pi/2), np.sin(slim+np.pi/2)])
    llim = np.array([np.cos(-slim+np.pi/2), np.sin(-slim+np.pi/2)])

    print(r[0]*ulim[1]-r[1]*ulim[0])
    print(r[0]*llim[1]-r[1]*llim[0])
    print((r[0]*ulim[1]-r[1]*ulim[0])* (r[0]*llim[1]-r[1]*llim[0]) <= 0)