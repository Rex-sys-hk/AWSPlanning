import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 创建一个立体三维坐标轴
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 设置球体参数
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)

# 球面坐标转笛卡尔坐标
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))

# 绘制球体表面
ax.plot_surface(x, y, z, rstride=4, cstride=4, color='grey', alpha=0.2)

# 在球面上随机分布一些点
theta = np.random.uniform(0, 2*np.pi, 100)
phi = np.arccos(2*np.random.uniform(0, 1, 100) - 1)
x_sphere = np.cos(phi)*np.sin(theta)
y_sphere = np.sin(phi)*np.sin(theta)
z_sphere = np.cos(theta)

ax.scatter(x_sphere, y_sphere, z_sphere, color='r')  # 绘制球面上的点

# 设置视角
ax.view_init(elev=30., azim=20)
ax.axis('equal')  # 设置坐标轴比例一致
plt.show()