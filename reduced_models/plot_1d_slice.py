import os
import numpy as np
from matplotlib import pyplot as plt
import dataclasses


'''
# folder = 'output_spherical_9'
folder = 'output/spherical_patient_1/dt=0.04166/ft=2400.0/dos=1.0/'
time_steps = [1, 148, 192, 506]
r_max_list = [0.012]*4
'''

# folder = 'output_spherical_new_1'
folder = 'output/spherical_patient_2/dt=0.04166/ft=1800.0/tt=1/'
folder = 'output/spherical_patient_2/dt=0.041666666666666664/ft=2400.0/tt=1/dos=1.0/'
time_steps = [1, 175, 560, 1108]
#time_steps = [1, 145, 560, 560]
time_steps = [1, 170, 600, 1175]
r_max_list = [0.02, 0.02, 0.04, 0.04]


@dataclasses.dataclass
class Result:
    t: np.array
    r: np.array
    data: np.array


def load(folder, name):
    filepath = os.path.join(folder, f'{name}.csv')

    data = np.loadtxt(filepath)
    r = data[0, 1:]
    d = data[1:,1:]
    t = data[1:,0]

    return Result(t=t, r=r, data=d)


def limit(r,r_max):
    return np.sum(r < r_max)


width_ratios = list(np.array(r_max_list) / np.sum(r_max_list))
fig, axes = plt.subplots(1, 4, sharey=True, gridspec_kw=dict(
    width_ratios=width_ratios
))
fig.set_size_inches(10., 4)

pro = load(folder, 'prohyp')
nec = load(folder, 'nec')
nut = load(folder, 'nut')
tum = load(folder, 'tumor')

ax = axes[0]
time_step = time_steps[0]
r_max = r_max_list[0]
ax.title.set_text(f'{time_step}')
ax.set_xlabel('r')
ax.plot(pro.r[:limit(pro.r, r_max)], pro.data[time_step,:limit(pro.r, r_max)], linewidth=2, color='blue', label='$\phi_P$')
ax.plot(nec.r[:limit(nec.r, r_max)], nec.data[time_step,:limit(nec.r, r_max)], linewidth=2, color='red', label='$\phi_N$')
ax.plot(tum.r[:limit(tum.r, r_max)], tum.data[time_step,:limit(tum.r, r_max)], ':', linewidth=2, color='black', label='$\phi_T$')
ax.plot(nut.r[:limit(nut.r, r_max)], nut.data[time_step,:limit(nut.r, r_max)], '--', linewidth=2, color='green', label='$\phi_{\sigma,i}$')
ax.grid(True)

ax = axes[1]
time_step = time_steps[1]
r_max = r_max_list[1]
ax.title.set_text(f'{time_step}')
ax.set_xlabel('r')
ax.plot(pro.r[:limit(pro.r, r_max)], pro.data[time_step,:limit(pro.r, r_max)], linewidth=2, color='blue', label='$\phi_P$')
ax.plot(nec.r[:limit(nec.r, r_max)], nec.data[time_step,:limit(nec.r, r_max)], linewidth=2, color='red', label='$\phi_N$')
ax.plot(tum.r[:limit(tum.r, r_max)], tum.data[time_step,:limit(tum.r, r_max)], ':', linewidth=2, color='black', label='$\phi_T$')
ax.plot(nut.r[:limit(nut.r, r_max)], nut.data[time_step,:limit(nut.r, r_max)], '--', linewidth=2, color='green', label='$\phi_{\sigma,i}$')
ax.grid(True)

ax = axes[2]
time_step = time_steps[2]
r_max = r_max_list[2]
ax.title.set_text(f'{time_step}')
ax.set_xlabel('r')
ax.plot(pro.r[:limit(pro.r, r_max)], pro.data[time_step,:limit(pro.r, r_max)], linewidth=2, color='blue', label='$\phi_P$')
ax.plot(nec.r[:limit(nec.r, r_max)], nec.data[time_step,:limit(nec.r, r_max)], linewidth=2, color='red', label='$\phi_N$')
ax.plot(tum.r[:limit(tum.r, r_max)], tum.data[time_step,:limit(tum.r, r_max)], ':', linewidth=2, color='black', label='$\phi_T$')
ax.plot(nut.r[:limit(nut.r, r_max)], nut.data[time_step,:limit(nut.r, r_max)], '--', linewidth=2, color='green', label='$\phi_{\sigma,i}$')
ax.grid(True)

ax = axes[3]
time_step = time_steps[3]
r_max = r_max_list[3]
ax.title.set_text(f'{time_step}')
ax.set_xlabel('r')
ax.plot(pro.r[:limit(pro.r, r_max)], pro.data[time_step,:limit(pro.r, r_max)], linewidth=2, color='blue', label='$\phi_P$')
ax.plot(nec.r[:limit(nec.r, r_max)], nec.data[time_step,:limit(nec.r, r_max)], linewidth=2, color='red', label='$\phi_N$')
ax.plot(tum.r[:limit(tum.r, r_max)], tum.data[time_step,:limit(tum.r, r_max)], ':', linewidth=2, color='black', label='$\phi_T$')
ax.plot(nut.r[:limit(nut.r, r_max)], nut.data[time_step,:limit(nut.r, r_max)], '--', linewidth=2, color='green', label='$\phi_{\sigma,i}$')
ax.grid(True)

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
