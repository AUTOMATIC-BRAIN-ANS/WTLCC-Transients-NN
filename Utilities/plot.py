from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

CB91_Blue = '#1C2385'
CB91_Green = '#4AA161'
CB91_Pink = '#F3A0F2'
CB91_Purple = '#9D2EC5'
CB91_Violet = '#661D98'
CB91_Amber = '#F5B14C'
CB91_Red = '#E00922'
color_list = [CB91_Blue, CB91_Red, CB91_Amber, CB91_Pink, CB91_Green, CB91_Purple, CB91_Violet]
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)
plt.rcParams['font.family'] = "serif"
plt.rcParams['xtick.labelsize']='small'
plt.rcParams['ytick.labelsize']='small'
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.titlesize']=10
plt.rcParams['axes.labelsize']=10
plt.rcParams['legend.fontsize']=15


def plot_waveform(*, datax=[], datay=[], zero_c = 0, locations=[], show=True, title="",  y_label = "ICP (mmHg)", x_label="Time (min)"):
    xs = np.linspace(0, int((len(datax)/100)), len(datax))
    fig, ax = plt.subplots(figsize=(20, 5))
    ax.plot(xs, datax, label='Ground Truth ICP')
 
    if(len(locations)!=0):


        locations = locations[np.where(locations<len(datax))]
   
        ax.vlines(xs[locations], -10, max(datax), linestyles='dashed', colors=['r'], label='Pulse Onset')
        zero_locs = np.array(locations)-zero_c

        print(locations)
        print(zero_c)

        if(zero_c!=0):
            
            for i in range(0, len(zero_locs)):
                if(i==0):
                    ax.axvspan(xs[zero_locs[i]], xs[locations[i]], color='g', alpha=0.5, label="Search Window")
                else:
                    ax.axvspan(xs[zero_locs[i]], xs[locations[i]], color='g', alpha=0.5)
        

    if(len(datay) > 0):
        ax.plot(xs, datay, label='Predicted ICP')
    ax.grid(color='b', linestyle=':', alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_xlabel(x_label, fontsize=20)
    ax.set_ylabel(y_label)
    ax.set_title(title)    
    ax.legend(loc='upper right')
    fig.tight_layout()   
    if(show==True):
        plt.show()
    
    return fig
 

def plot_waveform_double(datax, datay, test_list = [], title="Wave",  x_label="Time", y_label="ABP [mmHg]"):


    fig, axes = plt.subplots(2, figsize=(20, 5))
    axes[0].plot(datax, label='Ground Truth')
    axes[1].plot(datay, label='Prediction')
    
    for ax in axes: 
        ax.grid(color='b', linestyle=':', alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)    
        ax.legend(loc='upper right')

    fig.tight_layout()    
    plt.show()
    plt.close()


def plot_triple_pulse(pulse, icp,  pred_pulse = []):


    abp = pulse[:, [0]]
    fv = pulse[:, [2]]

    if(len(icp) == 0):

        icp = pulse[:, [1]]

    scaler = MinMaxScaler()


    abp = scaler.fit_transform(abp)
    fv = scaler.fit_transform(fv)
    icp = scaler.fit_transform(icp)


    xs = np.linspace(0, int(len(pulse)/50), len(pulse))
    fig, ax = plt.subplots(figsize=(20, 5))
    ax.plot(xs, abp, label='ABP')
    ax.plot(xs, fv, label='FV')
    ax.plot(xs, icp, label='ICP')
    if(len(pred_pulse) != 0):
        ax.plot(xs, pred_pulse, label='Predicted ICP')
    ax.grid(color='b', linestyle=':', alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_xlabel("Time (sec)")
    ax.set_title("Pulse waveform for ABP, FV, ICP signals.")    
    ax.legend(loc='upper right')
    fig.tight_layout()   
    plt.show()


def plot_correlation(datax, datay, stats=""):


    datax = (datax.flatten())
    datay = (datay.flatten())
    fig, ax = plt.subplots()
    ax.text(0.05, 0.8, stats, transform=ax.transAxes)
    ax.grid(color='b', linestyle=':', alpha=0.3)
    ax.scatter(datax, datay, s=10) 
    ax.set_xlabel("Mean Ground Truth ICP (mmHg)")
    ax.set_ylabel("Mean Predicted ICP (mmHg)")
    m, b = np.polyfit(datax, datay, 1)
    ax.plot(datax, b+m*datax, '-')
    ax.set_xlim(left=0, right=65)
    ax.set_ylim(bottom=-40, top=20)
    fig.tight_layout()
    #plt.show()
    return fig

def plot_histogram(data):

    fig, ax = plt.subplots()
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    ax.set_xticks(bins, minor=False)
    ax.set_title("Data Distribution")
    ax.set_xlabel("ICP [mmHg]")
    ax.set_ylabel("Fraction of total dataset")
    ax.hist(data, bins=bins, alpha=0.5, histtype='bar', ec='black', weights=np.ones(len(data)) / len(data))
    fig.tight_layout
    plt.show()

