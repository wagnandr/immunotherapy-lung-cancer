import os
import pathlib
import argparse
import matplotlib.colors as mc
from matplotlib import pyplot as plt

from utils import QoILogger
from tumor_io.plotting import plot_comparative_patient2 
from tumor_io.experimental_data import load_experimental_data


def plot_comparative_patient2(data_list, patient: int, labels):
    current_dir = pathlib.Path(__file__).parent.resolve()
    if patient == 1:
        path = os.path.join(current_dir, '..', 'data/tumor/pirmin-tumor-pat1.csv')
    elif patient == 2:
        path = os.path.join(current_dir, '..', 'data/tumor/pirmin-tumor-pat2.csv')
    else:
        raise RuntimeError("unknown patient")

    exp_data = load_experimental_data(path=path,t_start=0)

    fig, axes = plt.subplots(3, 1, sharex=True)

    axes[0].plot(exp_data.t, exp_data.volumes, '-x', label='real', color='black')

    color_list = list(mc.BASE_COLORS.keys())

    if patient == 1:
        time_offset = 176 
    elif patient == 2:
        path = os.path.join(current_dir, '..', 'data/tumor/pirmin-tumor-pat2.csv')
        time_offset = exp_data.t[0]
    else:
        raise RuntimeError("unknown patient")

    #residual_functional = ResidualFunctional(time_offset=time_offset)
    for i,qoi_logger in enumerate(data_list): 
        t = qoi_logger.time + time_offset
        p_vis = qoi_logger.tumor_mass_visible
        d = qoi_logger.medicine_mass
        t_mass = qoi_logger.tumor_mass
        p_mass = qoi_logger.proliferative_mass
        n_mass = qoi_logger.necrotic_mass

        axes[0].plot(t, p_vis*1e6, ':', label=labels[i], color=color_list[i])
        axes[0].grid(True)
        axes[0].set_ylabel('tumor volume [$cm^3$]')
        if patient == 2:
            linewidth = len(data_list)-i
        else:
            linewidth = None
        axes[1].plot(t, d, linewidth=linewidth, label=None, color=color_list[i])
        axes[1].grid(True)
        axes[1].set_ylabel('immunotherapy')

        label='tumor'
        axes[2].plot(t, t_mass, '-', label=label, color=color_list[i])
        label='proliferative'
        axes[2].plot(t, p_mass, '--', label=label, color=color_list[i])
        label='necrotic'
        axes[2].plot(t, n_mass, ':', label=label, color=color_list[i])
        axes[2].grid(True)
        axes[2].set_ylabel('mass')
        axes[2].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        if i == 0:
            axes[2].legend()
            leg = axes[2].get_legend()
            if len(data_list) > 1:
                for i in range(3):
                    leg.legendHandles[i].set_color('black')
        #axes[3].plot(t, qoi_logger.nutrient_i_mass, color=color_list[i])
        #axes[3].grid(True)
        #axes[3].set_ylabel('nutrient mass')
    axes[0].set_xlim(left=exp_data.t[0], right=max(exp_data.t[-1], t[-1]))
    axes[0].legend(ncol=1+len(data_list))
    for i, a in enumerate(axes):
        dates = exp_data.dates 

        mid = 0.1 

        if patient == 1:
            # plot E
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

        elif patient == 2:
            t_pau = 557 
            if i == 0:
                a.text(t_pau+10,mid,'Q6W',fontweight='bold')
            print(t_pau)
            a.axvline(x=t_pau, color='black', linestyle='dashed')
            t_dexa = 1104 
            print(t_dexa)
            if i == 0:
                a.text(t_dexa+10,mid,'PD',fontweight='bold')
            a.axvline(x=t_dexa, color='black', linestyle='dashed')
            print(t_dexa)
            t_el = 1317 
            if i == 0:
                a.text(t_el+10,mid,'EL',fontweight='bold')
            a.axvline(x=t_el, color='black', linestyle='dashed')

    plt.tight_layout()
    plt.show()


parser = argparse.ArgumentParser()
parser.add_argument('-i','--input-directories', nargs='+', help='List of input directories', required=True)
parser.add_argument('-l','--labels', nargs='+', help='List of labels', required=False)
parser.add_argument('--patient', type=int, required=True)
parser.add_argument('--t-end', help='Stop plotting', type=float, required=False)
parser.add_argument('--refresh', help='Refresh the plot every 3s', action='store_true')
args = parser.parse_args()

if args.labels is None:
    labels = [f'simulation {i}' for i in range(len(args.input_directories))]
    if len(args.input_directories) == 1:
        labels = ['simulated']
else:
    labels = args.labels

loggers = []
for dir in args.input_directories:
    qoi_logger = QoILogger()
    filepath = os.path.join(dir, 'info.txt')
    qoi_logger.read(filepath)
    loggers.append(qoi_logger)

plot_comparative_patient2(loggers, args.patient, labels)