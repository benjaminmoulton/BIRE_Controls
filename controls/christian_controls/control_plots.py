#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 12:53:41 2022

@author: christian
"""

import state_control_simulator
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

mpl.rcParams['axes.linewidth'] = 1.75 #set the value globally
mpl.rcParams["font.family"] = "serif"
plt.rc('font', weight='bold')

major_dict = {"width" : 1.25, "size" : 7., "labelsize" : 14.,
             "direction" : 'in', "which" : 'major'}
minor_dict = {"width" : 1.25, "size" : 4.,
             "direction" : 'in', "which" : 'minor'}

def pretty_plot(axs, xlims, ylims, dx, dy, annotations):
    i = 0
    for ax in axs:
        ax.set_xlim(xlims[i])
        ax.set_ylim(ylims[i])
        ax.xaxis.set_major_locator(MultipleLocator(dx[i]["major"]))
        ax.xaxis.set_minor_locator(MultipleLocator(dx[i]["minor"]))
        ax.yaxis.set_major_locator(MultipleLocator(dy[i]["major"]))
        ax.yaxis.set_minor_locator(MultipleLocator(dy[i]["minor"]))
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.annotate(annotations[i]["string"], annotations[i]["loc"], fontsize=14)
        ax.tick_params(**major_dict)
        ax.tick_params(**minor_dict)
        i += 1
    return axs

def simulation_results(bire, cg_shift, loop=False, **kwargs):
    fig_V, (ax_V1, ax_V2) = plt.subplots(2, 1, sharex=True)
    fig_R, (ax_R1, ax_R2, ax_R3) = plt.subplots(3, 1, sharex=True)
    fig_X, (ax_X, ax_Y, ax_Z) = plt.subplots(3, 1, sharex=True)
    fig_O, (ax_phi, ax_theta, ax_psi) = plt.subplots(3, 1, sharex=True)
    fig_Vz, (ax_uz, ax_vz, ax_wz) = plt.subplots(3, 1, sharex=True)
    fig_Rz, (ax_pz, ax_qz, ax_rz) = plt.subplots(3, 1, sharex=True)
    fig_Oz, (ax_phiz, ax_thetaz) = plt.subplots(2, 1, sharex=True)
    fig_u, (ax_da, ax_de, ax_dr) = plt.subplots(3, 1, sharex=True)
    fig_udot, (ax_dadot, ax_dedot, ax_drdot) = plt.subplots(3, 1, sharex=True)
    fig_w, ax_w = plt.subplots()
    if bire:
        save_dir = './Simulation Data/BIRE/'
    else:
        save_dir = './Simulation Data/Baseline/'
    save_dir_uncontrolled = save_dir + 'Uncontrolled/'
    save_dir_controlled = save_dir + 'Controlled/'
    t_range = np.load(save_dir_uncontrolled + 'time_range_CG_' + str(cg_shift[0]) + '.npy')
    x = np.load(save_dir_uncontrolled + 'states_CG_' + str(cg_shift[0]) + '.npy')
    u = np.load(save_dir_uncontrolled + 'inputs_CG_' + str(cg_shift[0]) + '.npy')
    x_hat = np.load(save_dir_uncontrolled + 'trim_states_CG_' + str(cg_shift[0]) + '.npy')
    u_hat = np.load(save_dir_uncontrolled + 'trim_inputs_CG_' + str(cg_shift[0]) + '.npy')
    z = np.load(save_dir_uncontrolled + 'shifted_states_CG_' + str(cg_shift[0]) + '.npy')
    x_ctr = np.load(save_dir_controlled + 'states_CG_' + str(cg_shift[0]) + '.npy')
    u_ctr = np.load(save_dir_controlled + 'inputs_CG_' + str(cg_shift[0]) + '.npy')
    x_hat_ctr = np.load(save_dir_controlled + 'trim_states_CG_' + str(cg_shift[0]) + '.npy')
    u_hat_ctr = np.load(save_dir_controlled + 'trim_inputs_CG_' + str(cg_shift[0]) + '.npy')
    z_ctr = np.load(save_dir_controlled + 'shifted_states_CG_' + str(cg_shift[0]) + '.npy')
    Vx, Vy, Vz = np.load(save_dir_controlled + 'wind_CG_' + str(cg_shift[0]) + '.npy')
    t_cutoff = kwargs.get("t_cut", t_range[-1])
    dt = kwargs.get("dt", 10.)

    """
    State Plots
    """

    ax_V1.plot(t_range, x_ctr[:, 0], color='k', linestyle='-')
    ax_V1.plot(t_range, x[:, 0], color='0.5', linestyle='-')
    ax_V1.plot(t_range, x_ctr[:, 1], color='k', linestyle='--')
    ax_V1.plot(t_range, x[:, 1], color='0.5', linestyle='--')
    ax_V1.plot(t_range, x_ctr[:, 2], color='k', linestyle='-.')
    ax_V1.plot(t_range, x[:, 2], color='0.5', linestyle='-.')
    ax_V2.plot(t_range, x_ctr[:, 0], color='k', linestyle='-')
    ax_V2.plot(t_range, x[:, 0], color='0.5', linestyle='-')
    ax_V2.plot(t_range, x_ctr[:, 1], color='k', linestyle='--')
    ax_V2.plot(t_range, x[:, 1], color='0.5', linestyle='--')
    ax_V2.plot(t_range, x_ctr[:, 2], color='k', linestyle='-.')
    ax_V2.plot(t_range, x[:, 2], color='0.5', linestyle='-.')
    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax_V1.transAxes, color='k', clip_on=False)
    ax_V1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    ax_V1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
    kwargs.update(transform=ax_V2.transAxes)  # switch to the bottom axes
    ax_V2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax_V2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    ax_R1.plot(t_range, x_ctr[:, 3]*180./np.pi, color='k', linestyle='-')
    ax_R1.plot(t_range, x[:, 3]*180./np.pi, color='0.5', linestyle='-')
    ax_R2.plot(t_range, x_ctr[:, 4]*180./np.pi, color='k', linestyle='--')
    ax_R2.plot(t_range, x[:, 4]*180./np.pi, color='0.5', linestyle='--')
    ax_R3.plot(t_range, x_ctr[:, 5]*180./np.pi, color='k', linestyle='-.')
    ax_R3.plot(t_range, x[:, 5]*180./np.pi, color='0.5', linestyle='-.')

    ax_X.plot(t_range, x_ctr[:, 6], color='k', linestyle='-')
    ax_X.plot(t_range, x[:, 6], color='0.5', linestyle='-')
    ax_Y.plot(t_range, x_ctr[:, 7], color='k', linestyle='--')
    ax_Y.plot(t_range, x[:, 7], color='0.5', linestyle='--')
    ax_Z.plot(t_range, x_ctr[:, 8], color='k', linestyle='-.')
    ax_Z.plot(t_range, x[:, 8], color='0.5', linestyle='-.')

    ax_phi.plot(t_range, x_ctr[:, 9]*180./np.pi, color='k', linestyle='-')
    ax_phi.plot(t_range, x[:, 9]*180./np.pi, color='0.5', linestyle='-')
    ax_theta.plot(t_range, x_ctr[:, 10]*180./np.pi, color='k', linestyle='--')
    ax_theta.plot(t_range, x[:, 10]*180./np.pi, color='0.5', linestyle='--')
    ax_psi.plot(t_range, x_ctr[:, 11]*180./np.pi, color='k', linestyle='-.')
    ax_psi.plot(t_range, x[:, 11]*180./np.pi, color='0.5', linestyle='-.')

    """
    Z-Plots
    """

    ax_uz.plot(t_range, z_ctr[:, 0], color='k', linestyle='-')
    ax_uz.plot(t_range, z[:, 0], color='0.5', linestyle='-')
    ax_vz.plot(t_range, z_ctr[:, 1], color='k', linestyle='--')
    ax_vz.plot(t_range, z[:, 1], color='0.5', linestyle='--')
    ax_wz.plot(t_range, z_ctr[:, 2], color='k', linestyle='-.')
    ax_wz.plot(t_range, z[:, 2], color='0.5', linestyle='-.')

    ax_pz.plot(t_range, z_ctr[:, 3]*180./np.pi, color='k', linestyle='-')
    ax_pz.plot(t_range, z[:, 3]*180./np.pi, color='0.5', linestyle='-')
    ax_qz.plot(t_range, z_ctr[:, 4]*180./np.pi, color='k', linestyle='--')
    ax_qz.plot(t_range, z[:, 4]*180./np.pi, color='0.5', linestyle='--')
    ax_rz.plot(t_range, z_ctr[:, 5]*180./np.pi, color='k', linestyle='-.')
    ax_rz.plot(t_range, z[:, 5]*180./np.pi, color='0.5', linestyle='-.')

    ax_phiz.plot(t_range, z_ctr[:, 6]*180./np.pi, color='k', linestyle='-')
    ax_phiz.plot(t_range, z[:, 6]*180./np.pi, color='0.5', linestyle='-')
    ax_thetaz.plot(t_range, z_ctr[:, 7]*180./np.pi, color='k', linestyle='--')
    ax_thetaz.plot(t_range, z[:, 7]*180./np.pi, color='0.5', linestyle='--')

    """
    Controls
    """

    ax_da.plot(t_range, u_ctr[:, 0]*180./np.pi, color='k', linestyle='-')
    ax_da.plot(t_range, u[:, 0]*180./np.pi, color='0.5', linestyle='-')
    ax_de.plot(t_range, u_ctr[:, 1]*180./np.pi, color='k', linestyle='-')
    ax_de.plot(t_range, u[:, 1]*180./np.pi, color='0.5', linestyle='-')
    ax_dr.plot(t_range, u_ctr[:, 2]*180./np.pi, color='k', linestyle='-')
    ax_dr.plot(t_range, u[:, 2]*180./np.pi, color='0.5', linestyle='-')


    """
    Control Rates
    """
    ax_dadot.plot(t_range, np.gradient(u_ctr[:, 0]*180./np.pi), color='k', linestyle='-')
    ax_dadot.plot(t_range, np.gradient(u[:, 0]*180./np.pi), color='0.5', linestyle='-')
    ax_dedot.plot(t_range, np.gradient(u_ctr[:, 1]*180./np.pi), color='k', linestyle='-')
    ax_dedot.plot(t_range, np.gradient(u[:, 1]*180./np.pi), color='0.5', linestyle='-')
    ax_drdot.plot(t_range, np.gradient(u_ctr[:, 2]*180./np.pi), color='k', linestyle='-')
    ax_drdot.plot(t_range, np.gradient(u[:, 2]*180./np.pi), color='0.5', linestyle='-')


    linestyles = ['-', '-.', ':', '--']

    ax_w.plot(t_range, Vx, label='$V_{g_x}$', color='k', linestyle=linestyles[0])
    ax_w.plot(t_range, Vy, label='$V_{g_y}$', color='k', linestyle=linestyles[1])
    ax_w.plot(t_range, Vz, label='$V_{g_z}$', color='k', linestyle=linestyles[2])
    figs = [fig_V, fig_Vz, fig_R, fig_Rz, fig_O, fig_Oz, fig_X, fig_u, fig_udot, fig_w]
    axs = [ax_V1, ax_V2, ax_uz, ax_vz, ax_wz,
           ax_R1, ax_R2, ax_R3, ax_pz, ax_qz, ax_rz,
           ax_phi, ax_theta, ax_psi, ax_phiz, ax_thetaz,
           ax_X, ax_Y, ax_Z, ax_da, ax_de, ax_dr,
           ax_dadot, ax_dedot, ax_drdot,
           ax_w]
    return figs, axs


if __name__ == "__main__":
    plt.close('all')
    bire = True
    t_range = np.load('./Simulation Data/BIRE/Uncontrolled/' + 'time_range_CG_' + str(0.0) + '.npy')
    t_cutoff = t_range[-1]
    dt = t_range[1] - t_range[0]
    figs, axs = simulation_results(bire, [0., 0., 0.], dt=dt)