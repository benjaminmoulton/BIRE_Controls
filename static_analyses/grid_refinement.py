#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 16:53:46 2022

@author: christian
"""

import numpy as np
import matplotlib.pyplot as plt
import machupX as mx
import json
import Richardson_Extrapolation as rx
import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator, LogLocator,
                               AutoLocator)
import time

mpl.rcParams['axes.linewidth'] = 1.75 #set the value globally
mpl.rcParams["font.family"] = "serif"
plt.rc('font', weight='bold')

major_dict = {"width" : 1.25, "size" : 7., "labelsize" : 16.,
             "direction" : 'in', "which" : 'major'}
minor_dict = {"width" : 1.25, "size" : 4.,
             "direction" : 'in', "which" : 'minor'}

forces_options = {'body_frame': True,
                  'stab_frame': False,
                  'wind_frame': True,
                  'dimensional': False,
                  'verbose': False}
def pretty_plot(ax, xlims, ylims, dx, dy, **kwargs):
    log = kwargs.get('log', False)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.xaxis.set_major_locator(MultipleLocator(dx["major"]))
    ax.xaxis.set_minor_locator(MultipleLocator(dx["minor"]))
    if log:
        ax.yaxis.set_major_locator(LogLocator())
        ax.yaxis.set_minor_locator(LogLocator(subs=np.arange(0.2, 1., 0.2)))
    else:
        ax.yaxis.set_major_locator(MultipleLocator(dy["major"]))
        ax.yaxis.set_minor_locator(MultipleLocator(dy["minor"]))
    ax.xaxis.set_ticks_position('both')
    ax.tick_params(**major_dict)
    ax.tick_params(**minor_dict)
    return ax

def generate_data(params, scene):
    alpha = params[0]
    beta = params[1]
    d_e = params[2]
    d_a = params[3]
    d_r = params[4]
    p = params[5]
    q = params[6]
    r = params[7]
    rates = [p, q, r]
    scene.set_aircraft_state(state={"alpha": alpha,
                                       "beta": beta,
                                       "angular_rates": rates,
                                       "velocity": 222.5211})
    scene.set_aircraft_control_state(control_state={"elevator": d_e,
                                                       "aileron": d_a,
                                                       "rudder": d_r})
    x = scene.solve_forces(**forces_options)["F16"]["total"]
    fm = [x['CD'], x['CS'], x['CL'], x['Cl'], x['Cm'], x['Cn']]
    return (*params, *fm)

def create_inputs(inp_dir, d_B):
    rotation_angle = str(int(d_B))

    f_inp = open(inp_dir + 'BIRE_input.json',)
    inp_data = json.load(f_inp)

    f_air = open(inp_dir + 'BIRE_airplane.json',)
    air_data = json.load(f_air)

    bire_left = d_B
    bire_right = -d_B
    air_data["wings"]["BIRE_left"]["dihedral"] = bire_left
    air_data["wings"]["BIRE_right"]["dihedral"] = bire_right

    new_air_fn = inp_dir + 'BIRE_airplane_dB_' + rotation_angle + '.json'
    with open(new_air_fn, 'w') as fp:
        json.dump(air_data, fp, indent=5)

    inp_data["scene"]["aircraft"]["BIRE"]["file"] = new_air_fn
    new_inp_fn = inp_dir + 'BIRE_input_dB_' + rotation_angle + '.json'
    with open(new_inp_fn, 'w') as fp:
        json.dump(inp_data, fp, indent=5)
    return new_inp_fn

def bire_case(params, inp_dir, scene=None):
    [alpha, beta, d_e, d_a, d_B, p, q, r] = params
    rotation_angle = str(int(d_B))
    try:
        f = open(inp_dir + 'BIRE_input_dB_' + rotation_angle + '.json',)
    except FileNotFoundError:
        create_inputs(inp_dir, d_B)
    if scene is None:
        input_file = inp_dir + 'BIRE_input_dB_' + rotation_angle + '.json'
        BIRE_scene = mx.Scene(input_file)
    else:
        BIRE_scene = scene
    rates = [p, q, r]
    BIRE_scene.set_aircraft_state(state={"alpha": alpha,
                                         "beta": beta,
                                         "angular_rates": rates,
                                         "velocity": 222.5211})
    BIRE_scene.set_aircraft_control_state(control_state={"elevator": d_e,
                                                         "aileron": d_a})
    x = BIRE_scene.solve_forces(**forces_options)["BIRE"]["total"]
    fm = [x['CD'], x['CS'], x['CL'], x['Cl'], x['Cm'], x['Cn']]
    return (*params, *fm)

def baseline_gcs(save_data):
    plt.close('all')
    FM = np.zeros(((280 - 70)//10 + 1, 6))
    times = np.zeros((280 - 70)//10 + 1)
    if save_data:
        input_file = "./F16_input.json"
        grid = 70
        f16_airplane = json.load(open('./F16_airplane.json'))
        f16_airplane["wings"]["main_wing"]["grid"]["N"] = grid
        f16_airplane["wings"]["h_stab"]["grid"]["N"] = grid
        f16_airplane["wings"]["v_stab"]["grid"]["N"] = grid
        with open("F16_airplane.json", "w") as outfile:
            json.dump(f16_airplane, outfile, indent=4)
        params = np.zeros(8)
        params[0] = 5.
        i = 0
        while grid < 290:
            f16_airplane = json.load(open('./F16_airplane.json'))
            f16_airplane["wings"]["main_wing"]["grid"]["N"] = grid
            with open("F16_airplane.json", "w") as outfile:
                json.dump(f16_airplane, outfile, indent=4)
            t0 = time.time()
            scene = mx.Scene(input_file)
            FM[i, :] = generate_data(params, scene)[8:]
            t1 = time.time()
            times[i] = t1 - t0
            print(grid, FM[i, 2], times[i])
            grid += 10
            i += 1
        np.save('./Mesh Refinement/FM_wing.npy', FM)
        np.save('./Mesh Refinement/times_wing.npy', times)
        grid = 70
        i = 0
        f16_airplane = json.load(open('./F16_airplane.json'))
        f16_airplane["wings"]["main_wing"]["grid"]["N"] = grid
        with open("F16_airplane.json", "w") as outfile:
            json.dump(f16_airplane, outfile, indent=4)
        while grid < 290:
            f16_airplane = json.load(open('./F16_airplane.json'))
            f16_airplane["wings"]["h_stab"]["grid"]["N"] = grid
            with open("F16_airplane.json", "w") as outfile:
                json.dump(f16_airplane, outfile, indent=4)
            t0 = time.time()
            scene = mx.Scene(input_file)
            FM[i, :] = generate_data(params, scene)[8:]
            t1 = time.time()
            times[i] = t1 - t0
            print(grid, FM[i, 2], times[i])
            grid += 10
            i += 1
        np.save('./Mesh Refinement/FM_HT.npy', FM)
        np.save('./Mesh Refinement/times_HT.npy', times)
        i = 0
        grid = 70
        params[0] = 0.
        params[1] = 5.
        f16_airplane = json.load(open('./F16_airplane.json'))
        f16_airplane["wings"]["h_stab"]["grid"]["N"] = grid
        with open("F16_airplane.json", "w") as outfile:
            json.dump(f16_airplane, outfile, indent=4)
        while grid < 290:
            f16_airplane = json.load(open('./F16_airplane.json'))
            f16_airplane["wings"]["v_stab"]["grid"]["N"] = grid
            with open("F16_airplane.json", "w") as outfile:
                json.dump(f16_airplane, outfile, indent=4)
            t0 = time.time()
            scene = mx.Scene(input_file)
            FM[i, :] = generate_data(params, scene)[8:]
            t1 = time.time()
            times[i] = t1 - t0
            print(grid, FM[i, 2], times[i])
            grid += 10
            i += 1
        np.save('./Mesh Refinement/FM_VT.npy', FM)
        np.save('./Mesh Refinement/times_VT.npy', times)
    else:
        FM_w = np.load('./Mesh Refinement/FM_wing.npy')
        times_w = np.load('./Mesh Refinement/times_wing.npy')
        FM_h = np.load('./Mesh Refinement/FM_HT.npy')
        times_h = np.load('./Mesh Refinement/times_HT.npy')
        FM_v = np.load('./Mesh Refinement/FM_VT.npy')
        times_v = np.load('./Mesh Refinement/times_VT.npy')

    grids = np.arange(70, 290, 10)
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.set_yscale('log')
    ax.scatter(grids, np.abs((FM_w[:, 2] - FM_w[21, 2])), ec='k', fc='none')
    ax.axhline(FM_w[21, 2], 0, 290, color='k', linestyle='--')
    ax.axhline(FM_w[21, 2]*1.01, 0, 290, color='k', linestyle=':')
    ax.axhline(FM_w[21, 2]*0.99, 0, 290, color='k', linestyle=':')
    ax2.scatter(grids, times_w, ec='b', fc='none')
    xlims = (60, 290)
    ylims = (6e-9, 1.5e-4)
    ylims2 = (1.9, 4.1)
    dx = {"major": 40, "minor": 10}
    dy = {"major": 1e-2, "minor": 1e-2/4}
    dy2 = {"major": 0.5, "minor": 0.5/4}
    axs = pretty_plot(ax, xlims, ylims, dx, dy, log=True)
    axs2 = pretty_plot(ax2, xlims, ylims2, dx, dy2)
    ax.set_ylabel(r'\boldmath$C_L - \left(C_L\right)_{n=280}$', fontsize=16)
    ax.set_xlabel(r'\textbf{Grid Refinement, }\boldmath$n$', fontsize=16)
    ax2.set_ylabel(r'\textbf{Run Time, }\boldmath$t$\textbf{ [sec]}', fontsize=16, color='b')
    ax.grid(axis='y', which='both')
    ax.grid(axis='x')
    plt.tight_layout()
    plt.savefig('./Mesh Refinement/gcs_wing.pdf', dpi=1000)


    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.set_yscale('log')
    ax.scatter(grids, np.abs((FM_h[:, 2] - FM_h[21, 2])), ec='k', fc='none')
    ax.axhline(FM_h[21, 2], 0, 290, color='k', linestyle='--')
    ax.axhline(FM_h[21, 2]*1.01, 0, 290, color='k', linestyle=':')
    ax.axhline(FM_h[21, 2]*0.99, 0, 290, color='k', linestyle=':')
    ax2.scatter(grids, times_h, ec='b', fc='none')
    xlims = (60, 290)
    ylims = (6e-9, 1.5e-4)
    ylims2 = (1.9, 4.1)
    dx = {"major": 40, "minor": 10}
    dy = {"major": 0.005, "minor": 0.005/4}
    dy2 = {"major": 0.5, "minor": 0.5/4}
    axs = pretty_plot(ax, xlims, ylims, dx, dy, log=True)
    axs2 = pretty_plot(ax2, xlims, ylims2, dx, dy2)
    ax.set_ylabel(r'\boldmath$C_L - \left(C_L\right)_{n=280}$', fontsize=16)
    ax.set_xlabel(r'\textbf{Grid Refinement, }\boldmath$n$', fontsize=16)
    ax2.set_ylabel(r'\textbf{Run Time, }\boldmath$t$\textbf{ [sec]}', fontsize=16, color='b')
    ax.grid(axis='y', which='both')
    ax.grid(axis='x')
    plt.tight_layout()
    plt.savefig('./Mesh Refinement/gcs_HT.pdf', dpi=1000)

    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.set_yscale('log')
    ax.scatter(grids, np.abs((FM_v[:, 1] - FM_v[21, 1])), ec='k', fc='none')
    ax.axhline(FM_v[21, 1], 0, 290, color='k', linestyle='--')
    ax.axhline(FM_v[21, 1]*1.01, 0, 290, color='k', linestyle=':')
    ax.axhline(FM_v[21, 1]*0.99, 0, 290, color='k', linestyle=':')
    ax2.scatter(grids, times_v, ec='b', fc='none')
    xlims = (60, 290)
    ylims = (6e-9, 1.5e-4)
    ylims2 = (1.9, 4.1)
    dx = {"major": 40, "minor": 10}
    dy = {"major": 0.0005, "minor": 0.0005/4}
    dy2 = {"major": 0.5, "minor": 0.5/4}
    axs = pretty_plot(ax, xlims, ylims, dx, dy, log=True)
    axs2 = pretty_plot(ax2, xlims, ylims2, dx, dy2)
    ax.set_ylabel(r'\boldmath$C_S - \left(C_S\right)_{n=280}$', fontsize=16)
    ax.set_xlabel(r'\textbf{Grid Refinement, }\boldmath$n$', fontsize=16)
    ax2.set_ylabel(r'\textbf{Run Time, }\boldmath$t$\textbf{ [sec]}', fontsize=16, color='b')
    ax.grid(axis='y', which='both')
    ax.grid(axis='x')
    plt.tight_layout()
    plt.savefig('./Mesh Refinement/gcs_VT.pdf', dpi=1000)

def bire_gcs(N_L, N_U, dN, save_data):
    plt.close('all')
    FM = np.zeros(((N_U - N_L)//dN + 1, 6))
    times = np.zeros((N_U - N_L)//dN + 1)
    bire_dir = './BIRE Inputs/'
    if save_data:
        input_file = "./BIRE_input.json"
        grid = N_L
        bire_airplane = json.load(open('./BIRE_airplane.json'))
        bire_airplane["wings"]["main_wing"]["grid"]["N"] = grid
        bire_airplane["wings"]["BIRE_left"]["grid"]["N"] = grid//2
        bire_airplane["wings"]["BIRE_right"]["grid"]["N"] = grid//2
        with open("BIRE_airplane.json", "w") as outfile:
            json.dump(bire_airplane, outfile, indent=4)
        params = np.zeros(8)
        params[0] = 5.
        i = 0
        while grid < N_U + 1:
            bire_airplane = json.load(open('./BIRE_airplane.json'))
            bire_airplane["wings"]["main_wing"]["grid"]["N"] = grid
            with open("BIRE_airplane.json", "w") as outfile:
                json.dump(bire_airplane, outfile, indent=4)
            t0 = time.time()
            scene = mx.Scene(input_file)
            FM[i, :] = bire_case(params, bire_dir, scene=scene)[8:]
            t1 = time.time()
            times[i] = t1 - t0
            print(grid, FM[i, 2], times[i])
            grid += dN
            i += 1
        np.save('./Mesh Refinement/FM_wing_B.npy', FM)
        np.save('./Mesh Refinement/times_wing_B.npy', times)
        grid = N_L
        i = 0
        bire_airplane = json.load(open('./BIRE_airplane.json'))
        bire_airplane["wings"]["main_wing"]["grid"]["N"] = grid
        with open("BIRE_airplane.json", "w") as outfile:
            json.dump(bire_airplane, outfile, indent=4)
        while grid < N_U + 1:
            bire_airplane = json.load(open('./BIRE_airplane.json'))
            bire_airplane["wings"]["BIRE_left"]["grid"]["N"] = grid//2
            bire_airplane["wings"]["BIRE_right"]["grid"]["N"] = grid//2
            with open("BIRE_airplane.json", "w") as outfile:
                json.dump(bire_airplane, outfile, indent=4)
            t0 = time.time()
            scene = mx.Scene(input_file)
            FM[i, :] = bire_case(params, bire_dir, scene=scene)[8:]
            t1 = time.time()
            times[i] = t1 - t0
            print(grid, FM[i, 2], times[i])
            grid += dN
            i += 1
        np.save('./Mesh Refinement/FM_HT_B.npy', FM)
        np.save('./Mesh Refinement/times_HT_B.npy', times)
        grid = N_L
        i = 0
        params[4] = 80.
        params[1] = 5.
        bire_airplane = json.load(open('./BIRE_airplane.json'))
        bire_airplane["wings"]["BIRE_left"]["grid"]["N"] = grid//2
        bire_airplane["wings"]["BIRE_right"]["grid"]["N"] = grid//2
        with open("BIRE_airplane.json", "w") as outfile:
            json.dump(bire_airplane, outfile, indent=4)
        while grid < N_U + 1:
            bire_airplane = json.load(open('./BIRE_airplane.json'))
            bire_airplane["wings"]["BIRE_left"]["grid"]["N"] = grid//2
            bire_airplane["wings"]["BIRE_right"]["grid"]["N"] = grid//2
            with open("BIRE_airplane.json", "w") as outfile:
                json.dump(bire_airplane, outfile, indent=4)
            t0 = time.time()
            scene = mx.Scene(input_file)
            FM[i, :] = bire_case(params, bire_dir, scene=scene)[8:]
            t1 = time.time()
            times[i] = t1 - t0
            print(grid, FM[i, 1], times[i])
            grid += dN
            i += 1
        np.save('./Mesh Refinement/FM_HT_B_rot.npy', FM)
        np.save('./Mesh Refinement/times_HT_B_rot.npy', times)
        FM_w = np.load('./Mesh Refinement/FM_wing_B.npy')
        times_w = np.load('./Mesh Refinement/times_wing_B.npy')
        FM_h = np.load('./Mesh Refinement/FM_HT_B.npy')
        times_h = np.load('./Mesh Refinement/times_HT_B.npy')
        FM_v = np.load('./Mesh Refinement/FM_HT_B_rot.npy')
        times_v = np.load('./Mesh Refinement/times_HT_B_rot.npy')
    else:
        FM_w = np.load('./Mesh Refinement/FM_wing_B.npy')
        times_w = np.load('./Mesh Refinement/times_wing_B.npy')
        FM_h = np.load('./Mesh Refinement/FM_HT_B.npy')
        times_h = np.load('./Mesh Refinement/times_HT_B.npy')
        FM_v = np.load('./Mesh Refinement/FM_HT_B_rot.npy')
        times_v = np.load('./Mesh Refinement/times_HT_B_rot.npy')

    grids = np.arange(N_L, N_U + 1, dN)
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.set_yscale('log')
    ax.scatter(grids, np.abs((FM_w[:, 2] - FM_w[21, 2])), ec='k', fc='none')
    ax.axhline(FM_w[-1, 2], 0, N_U + dN, color='k', linestyle='--')
    ax.axhline(FM_w[-1, 2]*1.01, 0, N_U + dN, color='k', linestyle=':')
    ax.axhline(FM_w[-1, 2]*0.99, 0, N_U + dN, color='k', linestyle=':')
    ax2.scatter(grids, times_w, ec='b', fc='none')
    xlims = (N_L - dN, N_U + dN)
    ylims = (6e-9, 1.5e-4)
    ylims2 = (0.9, 3.1)
    dx = {"major": 40, "minor": 10}
    dy = {"major": 0.005, "minor": 0.005/4}
    dy2 = {"major": 0.5, "minor": 0.5/4}
    axs = pretty_plot(ax, xlims, ylims, dx, dy, log=True)
    axs2 = pretty_plot(ax2, xlims, ylims2, dx, dy2)
    ax.set_ylabel(r'\boldmath$C_L - \left(C_L\right)_{n=280}$', fontsize=16)
    ax.set_xlabel(r'\textbf{Grid Refinement, }\boldmath$n$', fontsize=16)
    ax2.set_ylabel(r'\textbf{Run Time, }\boldmath$t$\textbf{ [sec]}', fontsize=16, color='b')
    ax.grid(axis='y', which='both')
    ax.grid(axis='x')
    plt.tight_layout()
    plt.savefig('./Mesh Refinement/gcs_wing_B.pdf', dpi=1000)


    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.set_yscale('log')
    ax.scatter(grids, np.abs((FM_h[:, 2] - FM_h[21, 2])), ec='k', fc='none')
    ax.axhline(FM_h[-1, 2], 0, N_U + dN, color='k', linestyle='--')
    ax.axhline(FM_h[-1, 2]*1.01, 0, N_U + dN, color='k', linestyle=':')
    ax.axhline(FM_h[-1, 2]*0.99, 0, N_U + dN, color='k', linestyle=':')
    ax2.scatter(grids, times_h, ec='b', fc='none')
    xlims = (N_L - dN, N_U + dN)
    ylims = (6e-9, 1.5e-4)
    ylims2 = (0.9, 3.1)
    dx = {"major": 40, "minor": 10}
    dy = {"major": 0.005, "minor": 0.005/4}
    dy2 = {"major": 0.5, "minor": 0.5/4}
    axs = pretty_plot(ax, xlims, ylims, dx, dy, log=True)
    axs2 = pretty_plot(ax2, xlims, ylims2, dx, dy2)
    ax.set_ylabel(r'\boldmath$C_L - \left(C_L\right)_{n=280}$', fontsize=16)
    ax.set_xlabel(r'\textbf{Grid Refinement, }\boldmath$n$', fontsize=16)
    ax2.set_ylabel(r'\textbf{Run Time, }\boldmath$t$\textbf{ [sec]}', fontsize=16, color='b')
    ax.grid(axis='y', which='both')
    ax.grid(axis='x')
    plt.tight_layout()
    plt.savefig('./Mesh Refinement/gcs_HT_B.pdf', dpi=1000)

    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.set_yscale('log')
    ax.scatter(grids, np.abs((FM_v[:, 2] - FM_v[21, 2])), ec='k', fc='none')
    ax.axhline(FM_v[-1, 2], 0, N_U + dN, color='k', linestyle='--')
    ax.axhline(FM_v[-1, 2]*1.01, 0, N_U + dN, color='k', linestyle=':')
    ax.axhline(FM_v[-1, 2]*0.99, 0, N_U + dN, color='k', linestyle=':')
    ax2.scatter(grids, times_v, ec='b', fc='none')
    xlims = (N_L - dN, N_U + dN)
    ylims = (6e-9, 1.5e-4)
    ylims2 = (0.9, 3.1)
    dx = {"major": 40, "minor": 10}
    dy = {"major": 0.005, "minor": 0.005/4}
    dy2 = {"major": 0.5, "minor": 0.5/4}
    axs = pretty_plot(ax, xlims, ylims, dx, dy, log=True)
    axs2 = pretty_plot(ax2, xlims, ylims2, dx, dy2)
    ax.set_ylabel(r'\boldmath$C_L - \left(C_L\right)_{n=280}$', fontsize=16)
    ax.set_xlabel(r'\textbf{Grid Refinement, }\boldmath$n$', fontsize=16)
    ax2.set_ylabel(r'\textbf{Run Time, }\boldmath$t$\textbf{ [sec]}', fontsize=16, color='b')
    ax.grid(axis='y', which='both')
    ax.grid(axis='x')
    plt.tight_layout()
    plt.savefig('./Mesh Refinement/gcs_HT_B_rot.pdf', dpi=1000)

if __name__ == "__main__":
    save_data = False
    baseline_gcs(save_data)
    save_data = False
    bire_gcs(70, 280, 10, save_data)