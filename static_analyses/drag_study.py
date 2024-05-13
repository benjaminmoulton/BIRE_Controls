#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 17:15:38 2022

@author: christian
"""

import numpy as np
from bire_aero import BIREAero
from f16_aero import F16Aero
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import machupX as mx


mpl.rcParams['axes.linewidth'] = 1.75 #set the value globally
mpl.rcParams["font.family"] = "serif"
plt.rc('font', weight='bold')

major_dict = {"width" : 1.25, "size" : 7., "labelsize" : 16.,
             "direction" : 'in', "which" : 'major'}
minor_dict = {"width" : 1.25, "size" : 4.,
             "direction" : 'in', "which" : 'minor'}

def pretty_plot(ax, xlims, ylims, dx, dy):
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.xaxis.set_major_locator(MultipleLocator(dx["major"]))
    ax.xaxis.set_minor_locator(MultipleLocator(dx["minor"]))
    ax.yaxis.set_major_locator(MultipleLocator(dy["major"]))
    ax.yaxis.set_minor_locator(MultipleLocator(dy["minor"]))
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(**major_dict)
    ax.tick_params(**minor_dict)
    return ax

def find_moment(deltas, C_moment, function):
    [delta_e, delta_B] = deltas
    results = function.aero_results(alpha, beta, pbar, qbar, rbar, da, delta_e, delta_B)
    est_moment = results[5]
    return (est_moment - C_moment)**2

def generate_data(params):
    alpha = params[0]
    beta = params[1]
    d_e = params[2]
    d_a = params[3]
    d_r = params[4]
    p = params[5]
    q = params[6]
    r = params[7]
    rates = [p, q, r]
    my_scene.set_aircraft_state(state={"alpha": alpha,
                                       "beta": beta,
                                       "angular_rates": rates,
                                       "velocity": 222.5211})
    my_scene.set_aircraft_control_state(control_state={"elevator": d_e,
                                                       "aileron": d_a,
                                                       "rudder": d_r})
    x = my_scene.solve_forces(**forces_options)["F16"]["total"]
    fm = [x['CD'], x['CS'], x['CL'], x['Cl'], x['Cm'], x['Cn']]
    return (*params, *fm)


b_aero = BIREAero()
f_aero = F16Aero()
my_scene = mx.Scene('./F16_input.json')

forces_options = {'body_frame': True,
                  'stab_frame': False,
                  'wind_frame': True,
                  'dimensional': False,
                  'verbose': False}

N = 50
max_dr = 30.*np.pi/180.
dr_range = np.linspace(-max_dr, max_dr, N)
Cn_base = np.zeros(N)
CD_base = np.zeros(N)
CD_twist = np.zeros(N)
CD_bire = np.zeros(N)

alpha = 0.
beta = 0.
pbar = 0.
qbar = 0.
rbar = 0.
da = 0.
de = 0.

for i in range(N):
    base_results = generate_data([alpha, beta, de, da, dr_range[i]*180/np.pi, pbar, qbar, rbar])
    CD_base[i] = base_results[8]
    Cn_base[i] = base_results[-1]
    res = optimize.minimize(find_moment, [0., 1.], args=(Cn_base[i], b_aero), method='Nelder-Mead').x
    bire_results = b_aero.aero_results(alpha, beta, pbar, qbar, rbar, da, res[0], res[1])
    CD_bire[i] = bire_results[2]
    Cn_bire = bire_results[5]
    print('de', res[0]*180/np.pi)
    print('dB', res[1]*180/np.pi)

plt.close('all')
fig, ax = plt.subplots()
ax.plot(Cn_base, CD_base, color='k', linestyle='-', label=r'Baseline')
ax.plot(Cn_base, 2.*np.abs(Cn_base), color='k', linestyle='--', label=r'Montgomery et al. [124]')
ax.plot(Cn_base, CD_bire, color='k', linestyle=':', label=r'BIRE')
ax.legend(fontsize=16)
ax.set_xlabel(r'\textbf{Yawing Moment Coefficient, }\boldmath$C_n$', fontsize=16)
ax.set_ylabel(r'\textbf{Drag Coefficient, }\boldmath$C_D$', fontsize=16)

xlims = (-0.035, 0.035)
dx = {'major': 0.01, 'minor': 0.01/4}
ylims = (-0.01, 0.11)
dy = {'major': 0.02, 'minor': 0.02/4}
ax = pretty_plot(ax, xlims, ylims, dx, dy)
ax.grid()
plt.tight_layout()
plt.savefig('./Drag Study/BIRE_Drag_Comparison.pdf', dpi=1000)
