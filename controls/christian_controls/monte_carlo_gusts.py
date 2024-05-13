#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 13:36:35 2022

@author: christian
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from state_control_simulator import simulate
import aero_trim as trim
from hunsaker_atm import stdatm_english
import scipy.optimize as optimize
import sinesummationpredictivefit as sspf
from scipy.signal import find_peaks
import time
import datetime as dt

def calcProcessTime(starttime, cur_iter, max_iter):

    telapsed = time.time() - starttime
    testimated = (telapsed/cur_iter)*(max_iter)

    finishtime = starttime + testimated
    finishtime = dt.datetime.fromtimestamp(finishtime).strftime("%H:%M:%S")  # in time

    lefttime = testimated-telapsed  # in seconds

    return (int(telapsed), int(lefttime), finishtime)

def find_dr(x, t_range, z):
    fit = x[0]*np.exp(-x[1]*(t_range - 1.))*np.sin(x[2]*(t_range - 1.))
    return np.linalg.norm(z - fit)

plt.close('all')
H = 15000.
a = stdatm_english(H)[-1]
M = 0.6
V = M*a
gamma = np.deg2rad(0.)
phi = np.deg2rad(0.)
Gamma = 0.5
cg_shift = [0., 0., 0.]
aero_dir = '/home/christian/Python Projects/AFRL BIRE/Static Analysis/main/'

with open('./BIRE_linearization.lin', 'rb') as f:
    BIRE_lin = pickle.load(f)
with open('./BIRE_solution.trim', 'rb') as f:
    BIRE_trim = pickle.load(f)
props = trim.AircraftProperties(V, H, Gamma, aero_dir, bire=True)

t_range = np.arange(0., 20., 0.1)
N = 11
MC_states = np.zeros((N, N, N, 8, len(t_range)))
s_range = np.linspace(-1., 1., N)
omega = 5.
model_gust = {"type": "gust", "params": {"A": 80.,
                                         "gamma": 1.,
                                         "w": omega,
                                         "s_x": -1.,
                                         "s_y": 1.,
                                         "s_z": 1.,
                                         "t_0": 1.}}

start = time.time()
cur_iter = 0
max_iter = N*N*N*8
for i in range(N):
    model_gust['params']['s_x'] = s_range[i]
    for j in range(N):
        model_gust['params']['s_y'] = s_range[j]
        for k in range(N):
            model_gust['params']['s_z'] = s_range[k]
            simulate(BIRE_trim, t_range, BIRE_lin, props, cg_shift, True, model=model_gust)
            save_dir = './Simulation Data/BIRE/'
            save_dir_controlled = save_dir + 'Controlled/'
            z_ctr = np.load(save_dir_controlled + 'shifted_states_CG_' + str(cg_shift[0]) + '.npy')
            MC_states[i, j, k, :] = z_ctr.T
            # for z in range(8):
            #     MC_states[i, j, k, z] = np.sqrt(np.sum(np.square(z_ctr[t_range > 15., z])))/np.max(np.abs(z_ctr[:, z]))
            cur_iter += 1
            prstime = calcProcessTime(start,cur_iter ,max_iter)
            print("time elapsed: %s(s), time left: %s(s), estimated finish time: %s"%prstime)
np.save('./MC_states_w_' + str(int(omega)) + '.npy', MC_states)
# harmonic_params = sspf.sinesummationfit(t_range, z_ctr[:, 4]*180./np.pi, num_sine=3)
# y_fit = sspf.sum_sines(t_range, harmonic_params, len(harmonic_params))
# damping_rate = optimize.minimize(find_dr, 0.5, args=(t_range, y_fit), method='Nelder-Mead').x
# plt.plot(t_range, z_ctr[:, 4]*180./np.pi, color='b')
# plt.plot(t_range, np.exp(-damping_rate*t_range)*y_fit, color='k')
