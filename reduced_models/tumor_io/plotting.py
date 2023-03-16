from matplotlib import pyplot as plt
import matplotlib.colors as mc
from tumor_io.experimental_data import load_experimental_data
import pathlib
import os


def plot(data_list, labels=None):
    if labels is None:
        labels = [f'{i+1}' for i,x in enumerate(data_list)]
    fig, axes = plt.subplots(3, 1, sharex=True)

    color_list = list(mc.BASE_COLORS.keys())

    for i,data in enumerate(data_list): 
        t = data['time']
        p = data['tumor'] 
        p_vis = data['tumor_visible'] 
        d = data['medicine'] 

        label = labels[i]
        mod = '' if len(t) > 100 else 'x'
        axes[0].plot(t, p_vis*1e6, f'{mod}:', label=label, color=color_list[i])
        dt = t[1]-t[0]
        axes[2].plot(0.5*(t[2:] + t[:-2]), (p[2:] - p[:-2])/p[0]/dt/2, color=color_list[i])
        axes[2].grid(True)
        axes[0].grid(True)
        axes[0].set_ylabel('V $[cm^3]$')
        axes[1].plot(t, d, label=label, color=color_list[i])
        axes[1].grid(True)
        axes[1].set_ylabel('immunotherapy')
        axes[1].set_xlabel('t [days]')
    axes[0].legend()
    plt.show()


def plot_comparative_patient1(data_list, with_error=False):
    num_subplots = 3 + (1 if with_error else 0)
    fig, axes = plt.subplots(num_subplots, 1, sharex=True)

    exp_data = load_experimental_data(t_start=0)

    axes[0].plot(exp_data.t, exp_data.volumes, '-x', label='real')
    axes[0].set_xlim(left=exp_data.t[0], right=exp_data.t[-1])
    axes[1].set_xlim(left=exp_data.t[0], right=exp_data.t[-1])
    axes[2].set_xlim(left=exp_data.t[0], right=exp_data.t[-1])

    time_offset=176

    color_list = list(mc.BASE_COLORS.keys())

    for i,data in enumerate(data_list): 
        t = data['time'] + time_offset 
        p = data['tumor'] 
        p_vis = data['tumor_visible'] 
        d = data['medicine'] 

        axes[0].plot(t, p_vis*1e6, ':', label='simulated', color=color_list[i])
        axes[0].grid(True)
        axes[0].set_ylabel('tumor volume [$cm^3$]')
        axes[1].plot(t, d, label=None, color=color_list[i])
        axes[1].grid(True)
        axes[1].set_ylabel('immunotherapy')

        axes[2].plot(t, p/p[0], label=None, color=color_list[i])
        axes[2].grid(True)
        axes[2].set_ylabel('tumor')

        if with_error:
            axes[3].set_ylabel('error')
    axes[num_subplots-1].set_xlabel('t [days]')
    axes[0].legend()
    for i, a in enumerate(axes):
        # plot E
        print(exp_data.dates[0])
        t_emb = 1135 
        if i == 0:
            a.text(t_emb+10,1.5,'E',fontweight='bold')
        a.axvline(x=t_emb, color='black', linestyle='dashed')

        # plot P
        t_pau = 1655 
        if i == 0:
            a.text(t_pau+10,1.5,'P',fontweight='bold')
        a.axvline(x=t_pau, color='black', linestyle='dashed')

        # plot D
        t_dexa = 2240 
        if i == 0:
            a.text(t_dexa+10,1.5,'D',fontweight='bold')
        a.axvline(x=t_dexa, color='black', linestyle='dashed')

    plt.tight_layout()
    plt.show()


def plot_comparative_patient2(data_list):
    current_dir = pathlib.Path(__file__).parent.resolve()
    path = os.path.join(current_dir, '..', '..', 'jupyter-plots/data/tumor/pirmin-tumor-pat2.csv')

    exp_data = load_experimental_data(path=path,t_start=0)

    fig, axes = plt.subplots(3, 1, sharex=True)

    axes[0].plot(exp_data.t, exp_data.volumes, '-x', label='real')

    color_list = list(mc.BASE_COLORS.keys())

    time_offset = exp_data.t[0]

    for i,data in enumerate(data_list): 
        t = data['time'] + time_offset 
        p = data['tumor'] 
        p_vis = data['tumor_visible'] 
        d = data['medicine'] 

        axes[0].plot(t, p_vis*1e6, ':', label='simulated', color=color_list[i])
        axes[0].grid(True)
        axes[0].set_ylabel('tumor volume [$cm^3$]')
        axes[1].plot(t, d, label=None, color=color_list[i])
        axes[1].grid(True)
        axes[1].set_ylabel('immunotherapy')
        #axes[1].set_xlabel('t [days]')

        axes[2].plot(t, p/p[0], label=None, color=color_list[i])
        axes[2].grid(True)
        axes[2].set_ylabel('tumor')
    axes[0].set_xlim(left=exp_data.t[0], right=max(exp_data.t[-1], t[-1]))
    axes[0].legend()
    for i, a in enumerate(axes):
        mid = 0.1 

        t_pau = 671 
        if i == 0:
            a.text(t_pau+10,mid,'Q6W',fontweight='bold')
        print(t_pau)
        a.axvline(x=t_pau, color='black', linestyle='dashed')
        t_dexa = 1218 
        print(t_dexa)
        if i == 0:
            a.text(t_dexa+10,mid,'PD',fontweight='bold')
        a.axvline(x=t_dexa, color='black', linestyle='dashed')
        print(t_dexa)
        t_el = 1431 
        a.text(t_el+10,mid,'EL',fontweight='bold')
        a.axvline(x=t_el, color='black', linestyle='dashed')

    plt.tight_layout()
    plt.show()