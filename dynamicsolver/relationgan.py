import numpy as np
from matplotlib import animation
"""
移动方程：
t时刻的位置P(x,y,z)
steps：dt的大小
sets：相关参数
"""

T = 70
h = 0.1
lamb = 1
c =1
Regst1 = 0.2
Regst2 = 0.1



def step(x):
    return np.array(x>0,dtype=int)




def reg(t1,t2):
    Regst1 = t1
    Regst2 = t2

def move(P, steps, sets,v):
    varphi1, varphi2, theta1, theta2 = P
    delt1 = theta1 + Regst1
    delt2 = theta2 + Regst2
    print(Regst1, Regst2)

    norm2 = delt1 ** 2 + delt2 ** 2
    norm1 = theta1 ** 2 + theta2 ** 2

    dot1 = varphi1 * theta1 + varphi2 * theta2

    dot2 = varphi1 * delt1 + varphi2 * delt2
    s = step(-dot1 + 1000 * norm1)
    reg(varphi1, varphi2)

    s2 = step(dot2 - dot1 + 1000 * norm2)


    dv1 = theta1*s+(delt1-theta1)*s2
    dv2 = theta2*s+(delt2-theta2)*s2
    dt1 = -varphi1
    dt2 = -varphi2


    return  [varphi1+dv1 * (steps),varphi2+dv2 * steps, theta1+dt1 * steps*0.95**v,theta2+dt2*steps*0.95**v],[theta1,theta2]











# 设置sets参数
sets = [10., 28., 3.]
t = np.arange(0, 200, 0.01)

# 位置1：
P0 = [0.1, 0.1, 0.2,0.1]
P = P0
d = []
for v in t:
    P,the = move(P, 0.01, sets,v)
    Regst1 = 0.8*the[0]
    Regst2 = 0.8*the[1]
    d.append(P)
dnp = np.array(d)




"""
画图
"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure()
#ax = Axes3D(fig)
#ax.plot(dnp[:, 2], dnp[:, 3])

plt.figure(figsize=(6, 6))
plt.clf()
plt.plot(dnp[:,2],dnp[:,3] )
plt.scatter(0, 0, 100, color='red')
plt.xlabel('$\\theta_1$')
plt.ylabel('$\\theta_2$')
plt.savefig('test.pdf')
plt.show()

theta1=dnp[:,2]
theta2=dnp[:,3]

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(0, 0, 30, color='red')
ax.set_xlim(-0.3, 0.3)
ax.set_ylim(-0.3, 0.3)
ax.set_xlabel('$\\theta_1$')
ax.set_ylabel('$\\theta_2$')
line, = plt.plot(theta1[:2], theta2[:2], color='blue')

def animate(i):
    line.set_data(theta1[:i], theta2[:i])
    return line,

anim = animation.FuncAnimation(fig=fig,
                               func=animate,
                               frames=range(0, len(theta1), 100),
                               interval=100)

anim.save('relation2d.gif', writer='imagemagick', dpi=80)