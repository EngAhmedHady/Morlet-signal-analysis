# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 13:17:37 2024

@author: Ahmed H. Hanfy
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import morlet as wl
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm
plt.rcParams.update({'font.size': 30})
plt.rcParams["text.usetex"] =  False
plt.rcParams["font.family"] = "Times New Roman"

file_path = 'signal.txt'
signal = pd.read_csv(fr'{file_path}', header=None)
n_samples = len(signal)

# Plot parameters
n_color_levels = 50
n_levels = 5
maxvalue = 0.01

# Signal info.
fs = 1500
T = n_samples/fs
ts = np.linspace(0, T, n_samples)

# Wavelet prams.
transition_steps = 100
n_cycles = 64
fw = np.linspace(1,int(fs/2),transition_steps)

# Performing wavelet analysis
cwtm = wl.morlet(signal[1], fs, 
                 n_cycles = n_cycles, fw = fw, 
                 review = [8,9],
                 units = 'mm')

# Visualise morlet CWT
fig,ax = plt.subplots(figsize=(20,10))
levels = MaxNLocator(nbins=n_color_levels).tick_values(0, maxvalue)
cmap = plt.colormaps['jet']
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
im = ax.contourf(ts, fw, cwtm, levels=levels, cmap=cmap, norm=norm)
levels = MaxNLocator(nbins=n_levels).tick_values(0, maxvalue)
imk = ax.contour(im, colors='k', levels=levels, alpha = np.linspace(1,0.2,n_levels+1))
cbar = fig.colorbar(im, ax=ax, ticks=np.around(np.linspace(0, maxvalue, 5), 3))
ax.set_xlabel("Time (sec)")
ax.set_ylabel("Frequency (Hz)")
ax.set_ylim([1,750])
# Location of imposed signal
ax.hlines(60, 0, T, 'w', ls = '--', alpha = 0.5)
ax.text(T-0.45, 60+5, '60Hz', fontsize=22, alpha=0.5, color = 'w')