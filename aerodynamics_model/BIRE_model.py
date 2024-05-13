#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 11:38:58 2021

@author: christian
"""

import numpy as np
import f16_model
import matplotlib.pyplot as plt
import machupX as mx
import pandas as pd
import imageio
from os.path import exists, isdir
import scipy.optimize as optimize
import json
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from os import mkdir, remove
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

def remove_outliers(data, m=2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return s < m

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
    forces_options = {'body_frame': True,
                      'stab_frame': False,
                      'wind_frame': True,
                      'dimensional': False,
                      'verbose': False}
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

def _plot_data_fit(mean, coeff_data, coeff_delta, model, params, range_1p, ylabel, baseline_coeff, scale=1., **kwargs):
    fig, ax = plt.subplots()
    dB_plot = np.arange(-180., 185., 1.)*np.pi/180.
    model_plot = scale*(params[0]*np.sin(params[1]*dB_plot + params[2]) + params[3] + coeff_delta)
    ax.scatter(dB_rad*180./np.pi, scale*(coeff_data + coeff_delta), facecolor='none', edgecolor='k', label='BIRE Coefficient')
    ax.plot(dB_plot*180/np.pi, model_plot, label='BIRE Fit', color='k')
    ax.axhline((baseline_coeff + coeff_delta)*scale, label='Baseline Coefficient', color='0.5', linestyle='--')
    ax.set_xlabel(r'\textbf{BIRE Rotation, }\boldmath$\delta_B$\textbf{ [deg]}', fontsize=14)
    ax.set_ylabel(r'\boldmath$' + ylabel[5:], fontsize=14)
    loc = kwargs.get('loc', 'upper right')
    handles, labels = ax.get_legend_handles_labels()
    # sort both labels and handles by labels
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    order = [0, 1, 2]
    ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc=loc, fontsize=14)
    xlims = (-190, 190)
    dx = {"major": 45., "minor": 45./4.}
    ylims = kwargs.get('y_lim', (model_plot.min()*0.7, model_plot.max()*1.3))
    dy = kwargs.get('dy', {'major': (ylims[1] - ylims[0])/5, 'minor': (ylims[1] - ylims[0])/20})
    ax = pretty_plot(ax, xlims, ylims, dx, dy)
    ax.grid()
    plt.tight_layout()
    return fig

def _CL_beta(CLbeta_data):
    betas = CLbeta_data[:, 1]*np.pi/180.
    CL = CLbeta_data[:, 10]
    mask = remove_outliers(CL)
    [CL_beta, CL0] = np.polyfit(betas[mask], CL[mask], 1)
    return CL_beta

def _CL_pbar(CLpbar_data):
    CL1 = np.array([x[10] for x in CLpbar_data if x[5] == 0.])
    CLp_p = np.array([x[10] for x in CLpbar_data if x[5] == 90.*np.pi/180.])
    CLp_m = np.array([x[10] for x in CLpbar_data if x[5] == -90.*np.pi/180.])
    DCLpbar_p = (CLp_p - CL1)/(np.deg2rad(90.)*b_w/(2.*V))
    DCLpbar_m = (CLp_m - CL1)/(np.deg2rad(-90.)*b_w/(2.*V))
    mask = remove_outliers(DCLpbar_p)*remove_outliers(DCLpbar_m)
    CL_pbar = np.average(np.vstack((DCLpbar_p[mask], DCLpbar_m[mask])))
    return CL_pbar

def _CL_rbar(CLrbar_data):
    CL1 = np.array([x[10] for x in CLrbar_data if x[7] == 0.])
    CLr_p = np.array([x[10] for x in CLrbar_data if x[7] == 30.*np.pi/180.])
    CLr_m = np.array([x[10] for x in CLrbar_data if x[7] == -30.*np.pi/180.])
    DCLrbar_p = (CLr_p - CL1)/(np.deg2rad(30.)*b_w/(2.*V))
    DCLrbar_m = (CLr_m - CL1)/(np.deg2rad(-30.)*b_w/(2.*V))
    mask = remove_outliers(DCLrbar_m)*remove_outliers(DCLrbar_p)
    CL_rbar = np.average(np.vstack((DCLrbar_p[mask], DCLrbar_m[mask])))
    return CL_rbar

def _CL_da(CLda_data):
    CL1 = np.array([x[10] for x in CLda_data if x[3] == 0.])
    CLda_p = np.array([x[10] for x in CLda_data if x[3] == 20.])
    DCLda_p = (CLda_p - CL1)/np.deg2rad(20.)
    mask = remove_outliers(DCLda_p)
    CL_da = np.average(DCLda_p[mask])
    return CL_da

def CL_models(baseline_coeffs, plot=True):
    weight_CL0 = (abs(dB_range) > 135)*(abs(dB_range) < 45)
    modelCL0 = lambda x : x[0]*np.sin(2.*dB_rad + np.pi/2.) + np.average(CL0_dB)
    errorCL0 = lambda x : modelCL0(x) - CL0_dB
    params_CL0 = np.append(optimize.leastsq(errorCL0, [0.2])[0], [2., np.pi/2., np.average(CL0_dB)])

    weight_CLalpha = [True]*N_dB
    weight_CLalpha[13] = False
    weight_CLalpha[59] = False
    weight_CLalpha[23] = False
    weight_CLalpha[49] = False
    modelCLalpha = lambda x : x[0]*np.sin(2.*dB_rad + np.pi/2.) + np.average(CLalpha_dB[weight_CLalpha])
    errorCLalpha = lambda x : (x[0]*np.sin(2.*dB_rad[weight_CLalpha] + np.pi/2.) + np.average(CLalpha_dB[weight_CLalpha]) - CLalpha_dB[weight_CLalpha])
    params_CLalpha = np.append(optimize.leastsq(errorCLalpha, [0.2])[0], [2., np.pi/2., np.average(CLalpha_dB[weight_CLalpha])])

    modelCLbeta = lambda x : x[0]*np.sin(2.*dB_rad)
    errorCLbeta = lambda x : (x[0]*np.sin(2.*dB_rad) - CLbeta_dB)
    params_CLbeta = np.append(optimize.leastsq(errorCLbeta, [0.6])[0], [2., 0., 0.])

    modelCLpbar = lambda x: 0*dB_rad
    # modelCLpbar = lambda x : x[0]*np.sin(2.*dB_rad)
    # errorCLpbar = lambda x : modelCLpbar(x) - CLpbar_dB
    # params_CLpbar = np.append(optimize.leastsq(errorCLpbar, [-0.2])[0], [2., 0., 0.])
    params_CLpbar = [0.]*4

    weight_CLqbar = [True]*N_dB
    weight_CLqbar[13] = False
    weight_CLqbar[59] = False
    weight_CLqbar[23] = False
    weight_CLqbar[49] = False
    modelCLqbar = lambda x : x[0]*np.sin(2.*dB_rad + np.pi/2.) + np.average(CLqbar_dB[weight_CLqbar])
    errorCLqbar = lambda x : (x[0]*np.sin(2.*dB_rad[weight_CLqbar] + np.pi/2.) + np.average(CLqbar_dB[weight_CLqbar]) - CLqbar_dB[weight_CLqbar])
    params_CLqbar = np.append(optimize.leastsq(errorCLqbar, [2.])[0], [2., np.pi/2., np.average(CLqbar_dB[weight_CLqbar])])

    modelCLrbar = lambda x : x[0]*np.sin(2.*dB_rad)
    errorCLrbar = lambda x : (x[0]*np.sin(2.*dB_rad) - CLrbar_dB)
    params_CLrbar = np.append(optimize.leastsq(errorCLrbar, [1.0])[0], [2., 0., 0.])

    modelCLda = lambda x : 0.*dB_rad + np.average(CLda_dB)
    params_CLda = [0.]*3 + [np.average(CLda_dB)]

    modelCLde = lambda x : x[0]*np.sin(1.*dB_rad + np.pi/2.) + 0
    errorCLde = lambda x : (x[0]*np.sin(1.*dB_rad + np.pi/2.) + 0. - CLde_dB)
    params_CLde = np.append(optimize.leastsq(errorCLde, [2.])[0], [1., np.pi/2., 0.])

    if plot:
        figs = []
        figs += [_plot_data_fit(np.average(CL0_dB), CL0_dB, CL0_delta, modelCL0, params_CL0, meanCL_1p, r'$\hat{C}_{L_0}$', baseline_coeffs["CL_0"], y_lim=(0.025, 0.095), dy={"major": 0.01, "minor": 0.01/4})]
        print("CL0", (np.max(modelCL0(params_CL0)) - np.min(modelCL0(params_CL0))))
        figs += [_plot_data_fit(np.average(CLalpha_dB), CLalpha_dB, CLalpha_delta, modelCLalpha, params_CLalpha, meanCL_1p/max_alpha, r'$\hat{C}_{L,\alpha}$', baseline_coeffs["CL_alpha"], y_lim=(3.05, 4.15), dy={"major": 0.15, "minor": 0.15/4})]
        print("CLa", (np.max(modelCLalpha(params_CLalpha)) - np.min(modelCLalpha(params_CLalpha)))*max_alpha)
        figs += [_plot_data_fit(np.average(CLbeta_dB), CLbeta_dB, CLbeta_delta, modelCLbeta, params_CLbeta, meanCL_1p/max_beta, r'$\hat{C}_{L,\beta}$', 0.0, y_lim=(-1, 1), dy={"major": 0.3, "minor": 0.3/4})]
        print("CLb", (np.max(modelCLbeta(params_CLbeta)) - np.min(modelCLbeta(params_CLbeta)))*max_beta)
        figs += [_plot_data_fit(np.average(CLpbar_dB), CLpbar_dB, CLpbar_delta, modelCLpbar, params_CLpbar, meanCL_1p/max_pbar, r'$\hat{C}_{L,\bar{p}}$', 0.0, y_lim=(-0.055, 0.055), dy={"major": 0.015, "minor": 0.015/4})]
        print("CLp", (np.max(modelCLpbar(params_CLpbar)) - np.min(modelCLpbar(params_CLpbar)))*max_pbar)
        figs += [_plot_data_fit(np.average(CLqbar_dB), CLqbar_dB, CLqbar_delta, modelCLqbar, params_CLqbar, meanCL_1p/max_qbar, r'$\hat{C}_{L,\bar{q}}$', baseline_coeffs["CL_qbar"], y_lim=(-1.5, 5.5), dy={"major": 1.0, "minor": 1.0/4})]
        print("CLq", (np.max(modelCLqbar(params_CLqbar)) - np.min(modelCLqbar(params_CLqbar)))*max_qbar)
        figs += [_plot_data_fit(np.average(CLrbar_dB), CLrbar_dB, CLrbar_delta, modelCLrbar, params_CLrbar, meanCL_1p/max_rbar, r'$\hat{C}_{L,\bar{r}}$', 0.0, y_lim=(-1, 1), dy={"major": 0.3, "minor": 0.3/4})]
        print("CLr", (np.max(modelCLrbar(params_CLrbar)) - np.min(modelCLrbar(params_CLrbar)))*max_rbar)
        figs += [_plot_data_fit(np.average(CLde_dB), CLde_dB, CLde_delta, modelCLde, params_CLde, meanCL_1p/max_de, r'$\hat{C}_{L,\delta_e}$', baseline_coeffs["CL_de"], y_lim=(-1., 1.), dy={"major": 0.3, "minor": 0.3/4})]
        print("CLde", (np.max(modelCLde(params_CLde)) - np.min(modelCLde(params_CLde)))*max_de)
        figs += [_plot_data_fit(np.average(CLda_dB), CLda_dB, CLda_delta, modelCLda, params_CLda, meanCL_1p/max_da, r'$\hat{C}_{L,\delta_a}$', 0., y_lim=(-0.0035, 0.0035), dy={"major": 0.001, "minor": 0.001/4})]
        print("CLda", (np.max(modelCLda(params_CLda)) - np.min(modelCLda(params_CLda)))*max_da)
        print("Average 1%", meanCL_1p)
        plot_dir = "./Fit Figures/"
        if not isdir(plot_dir):
            mkdir(plot_dir)
        pdf = matplotlib.backends.backend_pdf.PdfPages(plot_dir + "/Lift_BIRE_fits.pdf")
        fig_names = ["CL0", "CLa", "CLb", "CLp", "CLq", "CLr", "CLde", "CLda"]
        i = 0
        for fig in figs:
            fig.savefig(plot_dir + fig_names[i] + "_BIRE.pdf")
            pdf.savefig( fig )
            i += 1
        pdf.close()

    models_dict["CL"]["CL_0"] = {key : coeff for key,coeff in zip(model_coeff_keys, params_CL0)}
    models_dict["CL"]["CL_alpha"] = {key : coeff for key,coeff in zip(model_coeff_keys, params_CLalpha)}
    models_dict["CL"]["CL_beta"] = {key : coeff for key,coeff in zip(model_coeff_keys, params_CLbeta)}
    models_dict["CL"]["CL_pbar"] = {key : coeff for key,coeff in zip(model_coeff_keys, params_CLpbar)}
    models_dict["CL"]["CL_qbar"] = {key : coeff for key,coeff in zip(model_coeff_keys, params_CLqbar)}
    models_dict["CL"]["CL_rbar"] = {key : coeff for key,coeff in zip(model_coeff_keys, params_CLrbar)}
    models_dict["CL"]["CL_da"] = {key : coeff for key,coeff in zip(model_coeff_keys, params_CLda)}
    models_dict["CL"]["CL_de"] = {key : coeff for key,coeff in zip(model_coeff_keys, params_CLde)}

def _CS_alpha(CSalpha_data):
    alphas = CSalpha_data[:, 0]*np.pi/180.
    CS = CSalpha_data[:, 9]
    [CS_alpha, CS0] = np.polyfit(alphas, CS, 1)
    return CS_alpha

def _CS_qbar(CSqbar_data):
    CS1 = np.array([x[9] for x in CSqbar_data if x[6] == 0.])
    CSq_p = np.array([x[9] for x in CSqbar_data if x[6] == 30.*np.pi/180.])
    CSq_m = np.array([x[9] for x in CSqbar_data if x[6] == -30.*np.pi/180.])
    DCSqbar_p = (CSq_p - CS1)/(np.deg2rad(30.)*c_w/(2.*V))
    DCSqbar_m = (CSq_m - CS1)/(np.deg2rad(-30.)*c_w/(2.*V))
    CS_qbar = np.average(np.vstack((DCSqbar_p, DCSqbar_m)))
    return CS_qbar

def _CS_de(CSde_data):
    CS1 = np.array([x[9] for x in CSde_data if x[2] == 0.])
    CSde_p = np.array([x[9] for x in CSde_data if x[2] == 10.])
    CSde_m = np.array([x[9] for x in CSde_data if x[2] == -10.])
    DCSde_p = (CSde_p - CS1)/np.deg2rad(10.)
    DCSde_m = (CSde_m - CS1)/np.deg2rad(-10.)
    CS_de = np.average(np.vstack((DCSde_p, DCSde_m)))
    return CS_de

def CS_models(baseline_coeffs, plot=True):
    modelCS0 = lambda x : x[0]*np.sin(2.*dB_rad)
    errorCS0 = lambda x : (x[0]*np.sin(2.*dB_rad) - CS0_dB)
    params_CS0 = np.append(optimize.leastsq(errorCS0, [-0.01])[0], [2., 0., 0.])


    modelCSalpha = lambda x : x[0]*np.sin(2.*dB_rad)
    errorCSalpha = lambda x : (x[0]*np.sin(2.*dB_rad) - CSalpha_dB)
    params_CSalpha = np.append(optimize.leastsq(errorCSalpha, [0.2])[0], [2., 0., 0.])

    modelCSbeta = lambda x : x[0]*np.sin(2.*dB_rad + np.pi/2.) + np.average(CSbeta_dB)
    errorCSbeta = lambda x : (x[0]*np.sin(2.*dB_rad + np.pi/2.) + np.average(CSbeta_dB) - CSbeta_dB)
    params_CSbeta = np.append(optimize.leastsq(errorCSbeta, [0.6])[0], [2., np.pi/2., np.average(CSbeta_dB)])

    # modelCSpbar = lambda x : x[0]*np.sin(2.*dB_rad + np.pi/2.) + np.average(CSpbar_dB)
    # errorCSpbar = lambda x : modelCSpbar(x) - CSpbar_dB
    # params_CSpbar = np.append(optimize.leastsq(errorCSpbar, [0.005])[0], [2., np.pi/2., np.average(CSpbar_dB)])
    modelCSpbar = lambda x : 0.*dB_rad + np.average(CSpbar_dB)
    params_CSpbar = [0.]*3 + [np.average(CSpbar_dB)]

    modelCSLpbar = lambda x : x[0]*np.sin(2.*dB_rad + np.pi/2.) + np.average(CSLpbar_dB)
    errorCSLpbar = lambda x : modelCSLpbar(x) - CSLpbar_dB
    params_CSLpbar = np.append(optimize.leastsq(errorCSLpbar, [0.05])[0], [2., np.pi/2., np.average(CSLpbar_dB)])
    # modelCSLpbar = lambda x : 0.*dB_rad + np.average(CSLpbar_dB)
    # params_CSLpbar = [0.]*3 + [np.average(CSLpbar_dB)]

    modelCSqbar = lambda x : x[0]*np.sin(2.*dB_rad)
    errorCSqbar = lambda x : (x[0]*np.sin(2.*dB_rad) - CSqbar_dB)
    params_CSqbar = np.append(optimize.leastsq(errorCSqbar, [1.6])[0], [2., 0., 0.])

    weight_CSrbar = [True]*N_dB
    weight_CSrbar[6:9] = [False]*3
    weight_CSrbar[10:13] = [False]*3
    weight_CSrbar[24:27] = [False]*3
    weight_CSrbar[28:31] = [False]*3
    modelCSrbar = lambda x : x[0]*np.sin(2.*dB_rad + np.pi/2.) + np.average(CSrbar_dB)
    errorCSrbar = lambda x : (x[0]*np.sin(2.*dB_rad[weight_CSrbar] + np.pi/2.) + np.average(CSrbar_dB) - CSrbar_dB[weight_CSrbar])
    params_CSrbar = np.append(optimize.leastsq(errorCSrbar, [-2.])[0], [2., np.pi/2., np.average(CSrbar_dB)])

    weight_CSda = abs(CSda_dB) < 0.01
    modelCSda = lambda x : x[0]*np.sin(2.*dB_rad[weight_CSda] + np.pi/2.) + np.average(CSda_dB[weight_CSda])
    errorCSda = lambda x : (x[0]*np.sin(2.*dB_rad[weight_CSda] + np.pi/2.) + np.average(CSda_dB[weight_CSda]) - CSda_dB[weight_CSda])
    params_CSda = np.append(optimize.leastsq(errorCSda, [0.6])[0], [2., np.pi/2., np.average(CSda_dB[weight_CSda])])
    # modelCSda = lambda x : 0.*dB_rad + np.average(CSda_dB)
    # params_CSda = [0.]*3 + [np.average(CSda_dB)]

    modelCSde = lambda x : x[0]*np.sin(1.*dB_rad)
    errorCSde = lambda x : (x[0]*np.sin(1.*dB_rad) - CSde_dB)
    params_CSde = np.append(optimize.leastsq(errorCSde, [2.])[0], [1., 0., 0.])

    if plot:
        figs = []
        figs += [_plot_data_fit(np.average(CS0_dB), CS0_dB, CS0_delta, modelCS0, params_CS0, meanCS_1p, r'$\hat{C}_{S_0}$', 0., y_lim=(-0.014, 0.014), dy={"major": 0.004, "minor": 0.004/4})]
        print("CS0", (np.max(modelCS0(params_CS0)) - np.min(modelCS0(params_CS0))))
        figs += [_plot_data_fit(np.average(CSalpha_dB), CSalpha_dB, CSalpha_delta, modelCSalpha, params_CSalpha, meanCS_1p/max_alpha, r'$\hat{C}_{S,\alpha}$', 0., y_lim=(-0.35, 0.35), dy={"major": 0.1, "minor": 0.1/4})]
        print("CSa", (np.max(modelCSalpha(params_CSalpha)) - np.min(modelCSalpha(params_CSalpha)))*max_alpha)
        figs += [_plot_data_fit(np.average(CSbeta_dB), CSbeta_dB, CSbeta_delta, modelCSbeta, params_CSbeta, meanCS_1p/max_beta, r'$\hat{C}_{S,\beta}$', baseline_coeffs["CS_beta"], y_lim=(-2.75, 0.75), dy={"major": 0.5, "minor": 0.5/4})]
        print("CSb", (np.max(modelCSbeta(params_CSbeta)) - np.min(modelCSbeta(params_CSbeta)))*max_beta)
        figs += [_plot_data_fit(CSpbar_dB[N_dB//2], CSpbar_dB, CSpbar_delta, modelCSpbar, params_CSpbar, meanCS_1p/max_pbar, r'$\hat{C}_{S,\bar{p}}$', baseline_coeffs["CS_pbar"], y_lim=(-0.045, 0.025), dy={"major": 0.01, "minor": 0.01/4})]
        print("CSp", (np.max(modelCSpbar(params_CSpbar)) - np.min(modelCSpbar(params_CSpbar)))*max_pbar)
        figs += [_plot_data_fit(CSLpbar_dB[N_dB//2], CSLpbar_dB, CSLpbar_delta, modelCSLpbar, params_CSLpbar, meanCS_1p/max_CL1/max_pbar, r'$\hat{C}_{S,L\bar{p}}$', baseline_coeffs["CS_Lpbar"], y_lim=(0.13, 0.47), dy={"major": 0.05, "minor": 0.05/4})]
        print("CSLp", (np.max(modelCSLpbar(params_CSLpbar)) - np.min(modelCSLpbar(params_CSLpbar)))*max_CL1*max_pbar)
        figs += [_plot_data_fit(np.average(CSqbar_dB), CSqbar_dB, CSqbar_delta, modelCSqbar, params_CSqbar, meanCS_1p/max_qbar, r'$\hat{C}_{S,\bar{q}}$', 0., y_lim=(-3.5, 3.5), dy={"major": 1, "minor": 1/4})]
        print("CSq", (np.max(modelCSqbar(params_CSqbar)) - np.min(modelCSqbar(params_CSqbar)))*max_qbar)
        figs += [_plot_data_fit(np.average(CSrbar_dB), CSrbar_dB, CSrbar_delta, modelCSrbar, params_CSrbar, meanCS_1p/max_rbar, r'$\hat{C}_{S,\bar{r}}$', baseline_coeffs["CS_rbar"], y_lim=(-0.1, 1.9), dy={"major": 0.3, "minor": 0.3/4})]
        print("CSr", (np.max(modelCSrbar(params_CSrbar)) - np.min(modelCSrbar(params_CSrbar)))*max_rbar)
        figs += [_plot_data_fit(np.average(CSde_dB), CSde_dB, CSde_delta, modelCSde, params_CSde, meanCS_1p/max_de, r'$\hat{C}_{S,\delta_e}$', 0., y_lim=(-1., 1.), dy={"major": 0.3, "minor": 0.3/4})]
        print("CSde", (np.max(modelCSde(params_CSde)) - np.min(modelCSde(params_CSde)))*max_de)
        figs += [_plot_data_fit(np.average(CSda_dB), CSda_dB, CSda_delta, modelCSda, params_CSda, meanCS_1p/max_da, r'$\hat{C}_{S,\delta_a}$', baseline_coeffs["CS_da"], y_lim=(-0.07, 0.07), dy={"major": 0.02, "minor": 0.02/4})]
        print("CSda", (np.max(modelCSda(params_CSda)) - np.min(modelCSda(params_CSda)))*max_da)
        print("Average 1%", meanCS_1p)
        plot_dir = "./Fit Figures/"
        if not isdir(plot_dir):
            mkdir(plot_dir)
        pdf = matplotlib.backends.backend_pdf.PdfPages(plot_dir + "/Side_BIRE_fits.pdf")
        fig_names = ["CS0", "CSa", "CSb", "CSp", "CSLp", "CSq", "CSr", "CSde", "CSda"]
        i = 0
        for fig in figs: ## will open an empty extra figure :(
            fig.savefig(plot_dir + fig_names[i] + "_BIRE.pdf")
            pdf.savefig( fig )
            i += 1
        pdf.close()

    models_dict["CS"]["CS_0"] = {key : coeff for key,coeff in zip(model_coeff_keys, params_CS0)}
    models_dict["CS"]["CS_alpha"] = {key : coeff for key,coeff in zip(model_coeff_keys, params_CSalpha)}
    models_dict["CS"]["CS_beta"] = {key : coeff for key,coeff in zip(model_coeff_keys, params_CSbeta)}
    models_dict["CS"]["CS_pbar"] = {key : coeff for key,coeff in zip(model_coeff_keys, params_CSpbar)}
    models_dict["CS"]["CS_Lpbar"] = {key : coeff for key,coeff in zip(model_coeff_keys, params_CSLpbar)}
    models_dict["CS"]["CS_qbar"] = {key : coeff for key,coeff in zip(model_coeff_keys, params_CSqbar)}
    models_dict["CS"]["CS_rbar"] = {key : coeff for key,coeff in zip(model_coeff_keys, params_CSrbar)}
    models_dict["CS"]["CS_da"] = {key : coeff for key,coeff in zip(model_coeff_keys, params_CSda)}
    models_dict["CS"]["CS_de"] = {key : coeff for key,coeff in zip(model_coeff_keys, params_CSde)}

def _CD_pbar(CDpbar_data):
    CD1 = np.array([x[8] for x in CDpbar_data if x[5] == 0.])
    CDp_p = np.array([x[8] for x in CDpbar_data if x[5] == 90.*np.pi/180.])
    CDp_m = np.array([x[8] for x in CDpbar_data if x[5] == -90.*np.pi/180.])
    DCDpbar_p = (CDp_p - CD1)/(np.deg2rad(90.)*b_w/(2.*V))
    DCDpbar_m = (CDp_m - CD1)/(np.deg2rad(-90.)*b_w/(2.*V))
    CD_pbar = np.average(np.vstack((DCDpbar_p, DCDpbar_m)))
    return CD_pbar

def _CD_rbar(CDrbar_data):
    CD1 = np.array([x[8] for x in CDrbar_data if x[7] == 0.])
    CDr_p = np.array([x[8] for x in CDrbar_data if x[7] == 30.*np.pi/180.])
    CDr_m = np.array([x[8] for x in CDrbar_data if x[7] == -30.*np.pi/180.])
    DCDrbar_p = (CDr_p - CD1)/(np.deg2rad(30.)*b_w/(2.*V))
    DCDrbar_m = (CDr_m - CD1)/(np.deg2rad(-30.)*b_w/(2.*V))
    CD_rbar = np.average(np.vstack((DCDrbar_p, DCDrbar_m)))
    return CD_rbar

def _CD_da(CDda_data):
    CD1 = np.array([x[8] for x in CDda_data if x[3] == 0.])
    CDda_p = np.array([x[8] for x in CDda_data if x[3] == 20.])
    DCDda_p = (CDda_p - CD1)/np.deg2rad(20.)
    CD_da = np.average(DCDda_p)
    return CD_da

def CD_models(baseline_coeffs, plot=True):
    weight_CD0 = [True]*N_dB
    # weight_CD0[7:12] = [False]*5
    # weight_CD0[-12:-7] = [False]*5
    modelCD0 = lambda x : 0.*dB_rad + np.average(CD0_dB[weight_CD0])
    errorCD0 = lambda x : x[0]*np.sin(2.*dB_rad[weight_CD0] + np.pi/2.) + np.average(CD0_dB[weight_CD0]) - CD0_dB[weight_CD0]
    params_CD0 = [0.]*3 + [np.average(CD0_dB[weight_CD0])]

    weight_CDL = [True]*N_dB
    weight_CDL[13] = False
    weight_CDL[59] = False
    weight_CDL[23] = False
    weight_CDL[49] = False
    modelCDL = lambda x : x[0]*np.sin(1.*dB_rad + np.pi/2.) + np.average(CDL_dB[weight_CDL])
    errorCDL = lambda x : (x[0]*np.sin(1.*dB_rad[weight_CDL] + np.pi/2.) + np.average(CDL_dB[weight_CDL]) - CDL_dB[weight_CDL])
    params_CDL = [0.]*3 + [np.average(CDL_dB[weight_CDL])]
    # params_CDL = np.append(optimize.leastsq(errorCDL, [0.005])[0], [1., 0., np.average(CDL_dB[weight_CDL])])

    weight_CDL2 = [True]*N_dB
    weight_CDL2[11:26] = [False]*15
    weight_CDL2[47:62] = [False]*15
    modelCDL2 = lambda x : x[0]*np.sin(4.*dB_rad[weight_CDL2] + np.pi/2.) + np.average(CDL2_dB[weight_CDL2])
    errorCDL2 = lambda x : modelCDL2(x) - CDL2_dB[weight_CDL2]
    # modelCDL2 = lambda x : 0*dB_rad[weight_CDL2] + np.average(CDL2_dB[weight_CDL2])
    # errorCDL2 = lambda x : (0*dB_rad[weight_CDL2] + np.average(CDL2_dB[weight_CDL2]) - CDL2_dB[weight_CDL2])
    # params_CDL2 = [0.]*3 + [np.average(CDL2_dB[weight_CDL2])]
    params_CDL2 = np.append(optimize.leastsq(errorCDL2, [0.02])[0], [4., np.pi/2., np.average(CDL2_dB[weight_CDL2])])

    modelCDS = lambda x : x[0]*np.sin(2.*dB_rad) + np.average(CDS_dB)
    errorCDS = lambda x : modelCDS(x) - CDS_dB
    params_CDS = np.append(optimize.leastsq(errorCDS, [0.005])[0], [2., 0., np.average(CDS_dB)])

    weight_CDS2 = [True]*N_dB
    weight_CDS2[:8] = [False]*8
    weight_CDS2[-8:] = [False]*8
    weight_CDS2[31:41] = [False]*10
    modelCDS2 = lambda x : x[0]*np.sin(2.*dB_rad + np.pi/2.) + np.average(CDS2_dB[weight_CDS2])
    errorCDS2 = lambda x : (x[0]*np.sin(2.*dB_rad[weight_CDS2] + np.pi/2.) + np.average(CDS2_dB[weight_CDS2]) - CDS2_dB[weight_CDS2])
    params_CDS2 = np.append(optimize.leastsq(errorCDS2, [1.])[0], [2., np.pi/2., np.average(CDS2_dB[weight_CDS2])])

    modelCDpbar = lambda x : 0.*dB_rad
    params_CDpbar = [0.]*4

    weight_CDSpbar = [True]*N_dB
    weight_CDSpbar[:8] = [False]*8
    weight_CDSpbar[-8:] = [False]*8
    weight_CDSpbar[31:41] = [False]*10
    modelCDSpbar = lambda x : 0.*np.sin(2.*dB_rad + np.pi/2.) + np.average(CDSpbar_dB[weight_CDSpbar])
    errorCDSpbar = lambda x : (x[0]*np.sin(2.*dB_rad[weight_CDSpbar] + np.pi/2.) + np.average(CDSpbar_dB[weight_CDSpbar]) - CDSpbar_dB[weight_CDSpbar])
    params_CDSpbar = [0.]*3 + [np.average(CDSpbar_dB[weight_CDSpbar])]

    weight_CDqbar = [True]*N_dB
    modelCDqbar = lambda x : 0.*np.sin(2.*dB_rad + np.pi/2.) + np.average(CDqbar_dB[weight_CDqbar])
    errorCDqbar = lambda x : (modelCDqbar(x) - CDqbar_dB)[weight_CDqbar]
    params_CDqbar = [0.]*3 + [np.average(CDqbar_dB)]

    weight_CDLqbar = [True]*N_dB
    modelCDLqbar = lambda x : x[0]*np.sin(2.*dB_rad + np.pi/2.) + np.average(CDLqbar_dB[weight_CDLqbar])
    errorCDLqbar = lambda x : (x[0]*np.sin(2.*dB_rad[weight_CDLqbar] + np.pi/2.) + np.average(CDLqbar_dB[weight_CDLqbar]) - CDLqbar_dB[weight_CDLqbar])
    params_CDLqbar = np.append(optimize.leastsq(errorCDLqbar, [0.5])[0], [2., np.pi/2., np.average(CDLqbar_dB[weight_CDLqbar])])

    weight_CDL2qbar = [True]*N_dB
    modelCDL2qbar = lambda x : 0.*dB_rad + np.average(CDL2qbar_dB[weight_CDL2qbar])
    errorCDL2qbar = lambda x : modelCDL2qbar(x)[weight_CDL2qbar]
    params_CDL2qbar = [0.]*3 + [np.average(CDL2qbar_dB[weight_CDL2qbar])]

    modelCDrbar = lambda x : 0.*dB_rad
    params_CDrbar = [0.]*4

    weight_CDSrbar = [True]*N_dB
    weight_CDSrbar[:8] = [False]*8
    weight_CDSrbar[-8:] = [False]*8
    weight_CDSrbar[31:41] = [False]*10
    modelCDSrbar = lambda x : 0.*dB_rad + np.average(CDSrbar_dB[weight_CDSrbar])
    errorCDSrbar = lambda x : (x[0]*np.sin(2.*dB_rad[weight_CDSrbar] + np.pi/2.) + np.average(CDSrbar_dB[weight_CDSrbar]) - CDSrbar_dB[weight_CDSrbar])
    params_CDSrbar = [0.]*3 + [np.average(CDSrbar_dB[weight_CDSrbar])]

    modelCDda = lambda x : x[0]*np.sin(2.*dB_rad) + np.average(CDda_dB)
    errorCDda = lambda x : (x[0]*np.sin(2.*dB_rad) + np.average(CDda_dB) - CDda_dB)
    params_CDda = np.append(optimize.leastsq(errorCDda, [0.015])[0], [2., 0., np.average(CDda_dB)])

    weight_CDSda = abs(CDSda_dB) < 0.1
    # weight_CDSda[3:17] = [True]*14
    # weight_CDSda[39:53] = [True]*14
    # weight_CDSda[::18] = [False]*5
    modelCDSda = lambda x : x[0]*np.sin(2.*dB_rad + np.pi/2.) + np.average(CDSda_dB[weight_CDSda])
    errorCDSda = lambda x : (modelCDSda(x) - CDSda_dB)[weight_CDSda]
    params_CDSda = np.append(optimize.leastsq(errorCDSda, 0.03)[0], [2., np.pi/2., np.average(CDSda_dB[weight_CDSda])])

    modelCDde = lambda x : x[0]*np.sin(1.*dB_rad + np.pi/2.) + np.average(CDde_dB)
    errorCDde = lambda x : modelCDde(x) - CDde_dB
    params_CDde = np.append(optimize.leastsq(errorCDde, [0.02])[0], [1., np.pi/2., np.average(CDde_dB)])

    modelCDLde = lambda x : x[0]*np.sin(1.*dB_rad + np.pi/2.)
    errorCDLde = lambda x : (x[0]*np.sin(1.*dB_rad + np.pi/2.) - CDLde_dB)
    params_CDLde = np.append(optimize.leastsq(errorCDLde, [0.2])[0], [1., np.pi/2., 0.])

    modelCDde2 = lambda x : x[0]*np.sin(1.*dB_rad + np.pi/2.) + np.average(CDde2_dB)
    errorCDde2 = lambda x : (x[0]*np.sin(1.*dB_rad + np.pi/2.) + np.average(CDde2_dB) - CDde2_dB)
    params_CDde2 = np.append(optimize.leastsq(errorCDde2, [0.3])[0], [1., np.pi/2., np.average(CDde2_dB)])

    if plot:
        figs = []
        figs += [_plot_data_fit(params_CD0[-1], CD0_dB, CD0_delta, modelCD0, params_CD0, meanCD_1p, r'$\hat{C}_{D_0}$', baseline_coeffs["CD_0"], y_lim=(0.013, 0.047), dy={"major": 0.005, "minor": 0.005/4})]
        print("CD0", (np.max(modelCD0(params_CD0)) - np.min(modelCD0(params_CD0))))
        figs += [_plot_data_fit(params_CDL[-1], CDL_dB, CDL_delta, modelCDL, params_CDL, meanCD_1p/max_CL1, r'$\hat{C}_{D,L}$', baseline_coeffs["CD_L"], y_lim=(-0.0435, -0.0225), dy={"major": 0.003, "minor": 0.003/4})]
        print("CDL", (np.max(modelCDL(params_CDL)) - np.min(modelCDL(params_CDL)))*max_CL1)
        figs += [_plot_data_fit(params_CDL2[-1], CDL2_dB, CDL2_delta, modelCDL2, params_CDL2, meanCD_1p/max_CL1**2, r'$\hat{C}_{D,L^2}$', baseline_coeffs["CD_L2"], y_lim=(0.15, 0.29), dy={"major": 0.02, "minor": 0.02/4})]
        print("CDL2", (np.max(modelCDL2(params_CDL2)) - np.min(modelCDL2(params_CDL2)))*max_CL1*max_CL1)
        figs += [_plot_data_fit(params_CDS[-1], CDS_dB, CDS_delta, modelCDS, params_CDS, meanCD_1p/max_CS1, r'$\hat{C}_{D,S}$', 0., y_lim=(-0.225, 0.225), dy={"major": 0.06, "minor": 0.06/4})]
        print("CDS", (np.max(modelCDS(params_CDS)) - np.min(modelCDS(params_CDS)))*max_CS1)
        figs += [_plot_data_fit(params_CDS2[-1], CDS2_dB, CDS2_delta, modelCDS2, params_CDS2, meanCD_1p/max_CS1**2, r'$\hat{C}_{D,S^2}$', baseline_coeffs["CD_S2"], y_lim=(-0.1, 1.3), dy={"major": 0.2, "minor": 0.2/4})]
        print("CDS2", (np.max(modelCDS2(params_CDS2)) - np.min(modelCDS2(params_CDS2)))*max_CS1*max_CS1)
        figs += [_plot_data_fit(params_CDpbar[-1], CDpbar_dB, CDpbar_delta, modelCDpbar, params_CDpbar, meanCD_1p/max_pbar, r'$\hat{C}_{D,\bar{p}}$', 0., y_lim=(-0.001, 0.001), dy={"major": 0.0003, "minor": 0.0003/4})]
        print("CDp", (np.max(modelCDpbar(params_CDpbar)) - np.min(modelCDpbar(params_CDpbar)))*max_pbar)
        figs += [_plot_data_fit(params_CDSpbar[-1], CDSpbar_dB, CDSpbar_delta, modelCDSpbar, params_CDSpbar, meanCD_1p/max_CS1/max_pbar, r'$\hat{C}_{D,S\bar{p}}$', baseline_coeffs["CD_Spbar"], y_lim=(-0.03, 0.11), dy={"major": 0.02, "minor": 0.02/4})]
        print("CDSp", (np.max(modelCDSpbar(params_CDSpbar)) - np.min(modelCDSpbar(params_CDSpbar)))*max_CS1*max_pbar)
        figs += [_plot_data_fit(params_CDqbar[-1], CDqbar_dB, CDqbar_delta, modelCDqbar, params_CDqbar, meanCD_1p/max_qbar, r'$\hat{C}_{D,\bar{q}}$', baseline_coeffs["CD_qbar"], y_lim=(-0.125, 0.225), dy={"major": 0.05, "minor": 0.05/4})]
        print("CDq", (np.max(modelCDqbar(params_CDqbar)) - np.min(modelCDqbar(params_CDqbar)))*max_qbar)
        figs += [_plot_data_fit(params_CDLqbar[-1], CDLqbar_dB, CDLqbar_delta, modelCDLqbar, params_CDLqbar, meanCD_1p/max_CL1/max_qbar, r'$\hat{C}_{D,L\bar{q}}$', baseline_coeffs["CD_Lqbar"], y_lim=(-0.45, 1.65), dy={"major": 0.3, "minor": 0.3/4})]
        print("CDLq", (np.max(modelCDLqbar(params_CDLqbar)) - np.min(modelCDLqbar(params_CDLqbar)))*max_CL1*max_qbar)
        figs += [_plot_data_fit(params_CDL2qbar[-1], CDL2qbar_dB, CDL2qbar_delta, modelCDL2qbar, params_CDL2qbar, meanCD_1p/max_CL1/max_CL1/max_qbar, r'$\hat{C}_{D,L^2\bar{q}}$', baseline_coeffs["CD_L2qbar"], y_lim=(-0.7, 0.7), dy={"major": 0.2, "minor": 0.2/4})]
        print("CDL2q", (np.max(modelCDL2qbar(params_CDL2qbar)) - np.min(modelCDL2qbar(params_CDL2qbar)))*max_CL1*max_CL1*max_qbar)
        figs += [_plot_data_fit(params_CDrbar[-1], CDrbar_dB, CDrbar_delta, modelCDrbar, params_CDrbar, meanCD_1p/max_rbar, r'$\hat{C}_{D,\bar{r}}$', 0., y_lim=(-0.014, 0.014), dy={"major": 0.004, "minor": 0.004/4})]
        print("CDr", (np.max(modelCDrbar(params_CDrbar)) - np.min(modelCDrbar(params_CDrbar)))*max_rbar)
        figs += [_plot_data_fit(params_CDSrbar[-1], CDSrbar_dB, CDSrbar_delta, modelCDSrbar, params_CDSrbar, meanCD_1p/max_CS1/max_rbar, r'$\hat{C}_{D,S\bar{r}}$', baseline_coeffs["CD_Srbar"], y_lim=(-0.9, 0.5), dy={"major": 0.2, "minor": 0.2/4})]
        print("CDSr", (np.max(modelCDSrbar(params_CDSrbar)) - np.min(modelCDSrbar(params_CDSrbar)))*max_CS1*max_rbar)
        figs += [_plot_data_fit(params_CDde[-1], CDde_dB, CDde_delta, modelCDde, params_CDde, meanCD_1p/max_de, r'$\hat{C}_{D,\delta_e}$', baseline_coeffs["CD_de"], y_lim=(-0.017, 0.017), dy={"major": 0.005, "minor": 0.005/4})]
        print("CDde", (np.max(modelCDde(params_CDde)) - np.min(modelCDde(params_CDde)))*max_de)
        figs += [_plot_data_fit(params_CDLde[-1], CDLde_dB, CDLde_delta, modelCDLde, params_CDLde, meanCD_1p/max_de/max_CL1, r'$\hat{C}_{D,L\delta_e}$', baseline_coeffs["CD_Lde"], y_lim=(-0.32, 0.32), dy={"major": 0.1, "minor": 0.1/4})]
        print("CDLde", (np.max(modelCDLde(params_CDLde)) - np.min(modelCDLde(params_CDLde)))*max_CL1*max_de)
        figs += [_plot_data_fit(params_CDde2[-1], CDde2_dB, CDde2_delta, modelCDde2, params_CDde2, meanCD_1p/max_de**2, r'$\hat{C}_{D,\delta_e^2}$', baseline_coeffs["CD_de2"], y_lim=(0.05, 0.75), dy={"major": 0.1, "minor": 0.1/4})]
        print("CDde2", (np.max(modelCDde2(params_CDde2)) - np.min(modelCDde2(params_CDde2)))*max_de**2)
        figs += [_plot_data_fit(params_CDda[-1], CDda_dB, CDda_delta, modelCDda, params_CDda, meanCD_1p/max_da, r'$\hat{C}_{D,\delta_a}$', 0., y_lim=(-0.017, 0.017), dy={"major": 0.005, "minor": 0.005/4})]
        print("CDda", (np.max(modelCDda(params_CDda)) - np.min(modelCDda(params_CDda)))*max_da)
        figs += [_plot_data_fit(params_CDSda[-1], CDSda_dB, CDSda_delta, modelCDSda, params_CDSda, meanCD_1p/max_CS1/max_da, r'$\hat{C}_{D,S\delta_a}$', baseline_coeffs["CD_Sda"], y_lim=(-0.175, 0.175), dy={"major": 0.05, "minor": 0.05/4})]
        print("CDSda", (np.max(modelCDSda(params_CDSda)) - np.min(modelCDSda(params_CDSda)))*max_CS1*max_da)
        print("Average 1%", meanCD_1p)
        plot_dir = "./Fit Figures/"
        if not isdir(plot_dir):
            mkdir(plot_dir)
        pdf = matplotlib.backends.backend_pdf.PdfPages(plot_dir + "/Drag_BIRE_fits.pdf")
        fig_names = ["CD0", "CDL", "CDL2", "CDS", "CDS2", "CDp", "CDSp", "CDq", "CDLq", "CDL2q", "CDr", "CDSr", "CDde", "CDLde", "CDde2", "CDda", "CDSda"]
        i = 0
        for fig in figs: ## will open an empty extra figure :(
            fig.savefig(plot_dir + fig_names[i] + "_BIRE.pdf")
            pdf.savefig( fig )
            i += 1
        pdf.close()

    models_dict["CD"]["CD_0"] = {key : coeff for key,coeff in zip(model_coeff_keys, params_CD0)}
    models_dict["CD"]["CD_L"] = {key : coeff for key,coeff in zip(model_coeff_keys, params_CDL)}
    models_dict["CD"]["CD_L2"] = {key : coeff for key,coeff in zip(model_coeff_keys, params_CDL2)}
    models_dict["CD"]["CD_S"] = {key : coeff for key,coeff in zip(model_coeff_keys, params_CDS)}
    models_dict["CD"]["CD_S2"] = {key : coeff for key,coeff in zip(model_coeff_keys, params_CDS2)}
    models_dict["CD"]["CD_pbar"] = {key : coeff for key,coeff in zip(model_coeff_keys, params_CDpbar)}
    models_dict["CD"]["CD_Spbar"] = {key : coeff for key,coeff in zip(model_coeff_keys, params_CDSpbar)}
    models_dict["CD"]["CD_qbar"] = {key : coeff for key,coeff in zip(model_coeff_keys, params_CDqbar)}
    models_dict["CD"]["CD_Lqbar"] = {key : coeff for key,coeff in zip(model_coeff_keys, params_CDLqbar)}
    models_dict["CD"]["CD_L2qbar"] = {key : coeff for key,coeff in zip(model_coeff_keys, params_CDL2qbar)}
    models_dict["CD"]["CD_rbar"] = {key : coeff for key,coeff in zip(model_coeff_keys, params_CDrbar)}
    models_dict["CD"]["CD_Srbar"] = {key : coeff for key,coeff in zip(model_coeff_keys, params_CDSrbar)}
    models_dict["CD"]["CD_da"] = {key : coeff for key,coeff in zip(model_coeff_keys, params_CDda)}
    models_dict["CD"]["CD_Sda"] = {key : coeff for key,coeff in zip(model_coeff_keys, params_CDSda)}
    models_dict["CD"]["CD_de"] = {key : coeff for key,coeff in zip(model_coeff_keys, params_CDde)}
    models_dict["CD"]["CD_Lde"] = {key : coeff for key,coeff in zip(model_coeff_keys, params_CDLde)}
    models_dict["CD"]["CD_de2"] = {key : coeff for key,coeff in zip(model_coeff_keys, params_CDde2)}

def _Cl_alpha(Clalpha_data):
    alphas = Clalpha_data[:, 0]*np.pi/180.
    Cl = Clalpha_data[:, 11]
    [Cl_alpha, Cl0] = np.polyfit(alphas, Cl, 1)
    return Cl_alpha

def _Cl_qbar(Clqbar_data):
    Cl1 = np.array([x[11] for x in Clqbar_data if x[6] == 0.])
    Clq_p = np.array([x[11] for x in Clqbar_data if x[6] == 30.*np.pi/180.])
    Clq_m = np.array([x[11] for x in Clqbar_data if x[6] == -30.*np.pi/180.])
    DClqbar_p = (Clq_p - Cl1)/(np.deg2rad(30.)*c_w/(2.*V))
    DClqbar_m = (Clq_m - Cl1)/(np.deg2rad(-30.)*c_w/(2.*V))
    Cl_qbar = np.average(np.vstack((DClqbar_p, DClqbar_m)))
    return Cl_qbar

def _Cl_de(Clde_data):
    Cl1 = np.array([x[11] for x in Clde_data if x[2] == 0.])
    Clde_p = np.array([x[11] for x in Clde_data if x[2] == 10.])
    Clde_m = np.array([x[11] for x in Clde_data if x[2] == -10.])
    DClde_p = (Clde_p - Cl1)/np.deg2rad(10.)
    DClde_m = (Clde_m - Cl1)/np.deg2rad(-10.)
    Cl_de = np.average(np.vstack((DClde_p, DClde_m)))
    return Cl_de

def Cl_models(baseline_coeffs, plot=True):
    modelCl0 = lambda x : x[0]*np.sin(2.*dB_rad)
    errorCl0 = lambda x : (x[0]*np.sin(2.*dB_rad) - Cl0_dB)
    # params_Cl0 = [0]*4
    params_Cl0 = np.append(optimize.leastsq(errorCl0, [0.01])[0], [2., 0., 0.])

    weight_Clalpha = [True]*N_dB
    # weight_Clalpha = (abs(dB_range) < 20.)
    modelClalpha = lambda x : x[0]*np.sin(4.*dB_rad)
    errorClalpha = lambda x : x[0]*np.sin(4.*dB_rad[weight_Clalpha]) - Clalpha_dB[weight_Clalpha]
    params_Clalpha = np.append(optimize.leastsq(errorClalpha, [-0.04])[0], [4., 0., 0.])
    # params_Clalpha = [0.]*4

    modelClbeta = lambda x : x[0]*np.sin(2.*dB_rad + np.pi/2.) + np.average(Clbeta_dB)
    errorClbeta = lambda x : (x[0]*np.sin(2.*dB_rad + np.pi/2.) + np.average(Clbeta_dB) - Clbeta_dB)
    params_Clbeta = np.append(optimize.leastsq(errorClbeta, [0.04])[0], [2., np.pi/2., np.average(Clbeta_dB)])
    # params_Clbeta = [0.]*3 + [np.average(Clbeta_dB)]

    weight_Clpbar = [True]*N_dB
    weight_Clpbar[1:3] = [False]*2
    weight_Clpbar[8:11] = [False]*3
    weight_Clpbar[16:18] = [False]*2
    weight_Clpbar[-3:-1] = [False]*2
    weight_Clpbar[-11:-8] = [False]*3
    weight_Clpbar[-18:-16] = [False]*2
    modelClpbar = lambda x : x[0]*np.sin(2.*dB_rad + np.pi/2.) + np.average(Clpbar_dB[weight_Clpbar])
    errorClpbar = lambda x : (x[0]*np.sin(2.*dB_rad[weight_Clpbar] + np.pi/2.) + np.average(Clpbar_dB[weight_Clpbar]) - Clpbar_dB[weight_Clpbar])
    params_Clpbar = np.append(optimize.leastsq(errorClpbar, [0.02])[0], [2., np.pi/2., np.average(Clpbar_dB[weight_Clpbar])])
    # params_Clpbar = [0.]*3 + [np.average(Clpbar_dB[weight_Clpbar])]

    weight_Clqbar = abs(dB_rad) < np.pi/4
    modelClqbar = lambda x : x[0]*np.sin(4.*dB_rad[weight_Clqbar])
    errorClqbar = lambda x : modelClqbar(x) - Clqbar_dB[weight_Clqbar]
    # params_Clqbar = np.append(optimize.leastsq(errorClqbar, [0.02])[0], [4., 0., 0.])
    params_Clqbar = [0.]*4

    modelClrbar = lambda x : x[0]*np.sin(2.*dB_rad + np.pi/2.) + np.average(Clrbar_dB)
    errorClrbar = lambda x : modelClrbar(x) - Clrbar_dB
    # params_Clrbar = np.append(optimize.leastsq(errorClrbar, [0.001])[0], [2., np.pi/2., np.average(Clrbar_dB)])
    params_Clrbar = [0.]*3 + [np.average(Clrbar_dB)]

    modelClLrbar = lambda x : x[0]*np.sin(3.*dB_rad + np.pi/2.) + np.average(ClLrbar_dB)
    errorClLrbar = lambda x : modelClLrbar(x) - ClLrbar_dB
    params_ClLrbar = [0.]*3 + [np.average(ClLrbar_dB)]

    weight_Clda = np.abs(Clda_dB) < 0.2
    modelClda = lambda x : x[0]*np.sin(2.*dB_rad[weight_Clda] + np.pi/2.) + np.average(Clda_dB[weight_Clda])
    errorClda = lambda x : (x[0]*np.sin(2.*dB_rad[weight_Clda] + np.pi/2.) + np.average(Clda_dB[weight_Clda]) - Clda_dB[weight_Clda])
    params_Clda = np.append(optimize.leastsq(errorClda, [0.03])[0], [2., np.pi/2., np.average(Clda_dB[weight_Clda])])

    weight_Clde = [True]*N_dB
    weight_Clde[7:31] = [False]*24
    weight_Clde[42:66] = [False]*24
    modelClde = lambda x : x[0]*np.sin(dB_rad)
    errorClde = lambda x : (x[0]*np.sin(dB_rad[weight_Clde]) - Clde_dB[weight_Clde])
    params_Clde = np.append(optimize.leastsq(errorClde, [0.0005])[0], [1., 0., 0.])
    # params_Clde = [0.]*3 + [np.average(Clde_dB)]

    if plot:
        figs = []
        figs += [_plot_data_fit(params_Cl0[-1], Cl0_dB, Cl0_delta, modelCl0, params_Cl0, meanCl_1p, r'$\hat{C}_{\ell_0}$', 0., y_lim=(-0.0008125, 0.0008125), dy={"major": 0.00025, "minor": 0.00025/4})]
        print("Cl0", (np.max(modelCl0(params_Cl0)) - np.min(modelCl0(params_Cl0))))
        figs += [_plot_data_fit(params_Clalpha[-1], Clalpha_dB, Clalpha_delta, modelClalpha, params_Clalpha, meanCl_1p/max_alpha, r'$\hat{C}_{\ell,\alpha}$', 0., y_lim=(-0.014, 0.014), dy={"major": 0.004, "minor": 0.004/4})]
        print("Cla", (np.max(modelClalpha(params_Clalpha)) - np.min(modelClalpha(params_Clalpha)))*max_alpha)
        figs += [_plot_data_fit(params_Clbeta[-1], Clbeta_dB, Clbeta_delta, modelClbeta, params_Clbeta, meanCl_1p/max_beta, r'$\hat{C}_{\ell,\beta}$', baseline_coeffs["Cl_beta"], y_lim=(-0.0975, 0.0075), dy={"major": 0.015, "minor": 0.015/4})]
        print("Clb", (np.max(modelClbeta(params_Clbeta)) - np.min(modelClbeta(params_Clbeta)))*max_beta)
        figs += [_plot_data_fit(params_Clpbar[-1], Clpbar_dB, Clpbar_delta, modelClpbar, params_Clpbar, meanCl_1p/max_pbar, r'$\hat{C}_{\ell,\bar{p}}$', baseline_coeffs["Cl_pbar"], y_lim=(-0.3875, -0.2125), dy={"major": 0.025, "minor": 0.025/4})]
        print("Clp", (np.max(modelClpbar(params_Clpbar)) - np.min(modelClpbar(params_Clpbar)))*max_pbar)
        figs += [_plot_data_fit(params_Clqbar[-1], Clqbar_dB, Clqbar_delta, modelClqbar, params_Clqbar, meanCl_1p/max_qbar, r'$\hat{C}_{\ell,\bar{q}}$', 0., y_lim=(-0.035, 0.035), dy={"major": 0.01, "minor": 0.01/4})]
        print("Clq", (np.max(modelClqbar(params_Clqbar)) - np.min(modelClqbar(params_Clqbar)))*max_qbar)
        figs += [_plot_data_fit(params_Clrbar[-1], Clrbar_dB, Clrbar_delta, modelClrbar, params_Clrbar, meanCl_1p/max_rbar, r'$\hat{C}_{\ell,\bar{r}}$', baseline_coeffs["Cl_rbar"], y_lim=(-0.03, 0.09), dy={"major": 0.02, "minor": 0.02/4})]
        print("Clr", (np.max(modelClrbar(params_Clrbar)) - np.min(modelClrbar(params_Clrbar)))*max_rbar)
        figs += [_plot_data_fit(params_ClLrbar[-1], ClLrbar_dB, ClLrbar_delta, modelClLrbar, params_ClLrbar, meanCl_1p/max_CL1/max_rbar, r'$\hat{C}_{\ell,L\bar{r}}$', baseline_coeffs["Cl_Lrbar"], y_lim=(-0.025, 0.325), dy={"major": 0.05, "minor": 0.05/4})]
        print("ClLr", (np.max(modelClLrbar(params_ClLrbar)) - np.min(modelClLrbar(params_ClLrbar)))*max_CL1*max_rbar)
        figs += [_plot_data_fit(params_Clde[-1], Clde_dB, Clde_delta, modelClde, params_Clde, meanCl_1p/max_de, r'$\hat{C}_{\ell,\delta_e}$', 0., y_lim=(-0.007, 0.007), dy={"major": 0.002, "minor": 0.002/4})]
        print("Clde", (np.max(modelClde(params_Clde)) - np.min(modelClde(params_Clde)))*max_de)
        figs += [_plot_data_fit(params_Clda[-1], Clda_dB, Clda_delta, modelClda, params_Clda, meanCl_1p/max_da, r'$\hat{C}_{\ell,\delta_a}$', baseline_coeffs["Cl_da"], y_lim=(-0.22, 0.06), dy={"major": 0.04, "minor": 0.04/4})]
        print("Clda", (np.max(modelClda(params_Clda)) - np.min(modelClda(params_Clda)))*max_da)
        print("Mean 1\%", meanCl_1p)
        plot_dir = "./Fit Figures/"
        if not isdir(plot_dir):
            mkdir(plot_dir)
        pdf = matplotlib.backends.backend_pdf.PdfPages(plot_dir + "/Roll_BIRE_fits.pdf")
        fig_names = ["Cl0", "Cla", "Clb", "Clp", "Clq", "Clr", "ClLr", "Clde", "Clda"]
        i = 0
        for fig in figs: ## will open an empty extra figure :(
            fig.savefig(plot_dir + fig_names[i] + "_BIRE.pdf")
            pdf.savefig( fig )
            i += 1
        pdf.close()

    models_dict["Cell"]["Cl_0"] = {key : coeff for key,coeff in zip(model_coeff_keys, params_Cl0)}
    models_dict["Cell"]["Cl_alpha"] = {key : coeff for key,coeff in zip(model_coeff_keys, params_Clalpha)}
    models_dict["Cell"]["Cl_beta"] = {key : coeff for key,coeff in zip(model_coeff_keys, params_Clbeta)}
    models_dict["Cell"]["Cl_pbar"] = {key : coeff for key,coeff in zip(model_coeff_keys, params_Clpbar)}
    models_dict["Cell"]["Cl_qbar"] = {key : coeff for key,coeff in zip(model_coeff_keys, params_Clqbar)}
    models_dict["Cell"]["Cl_rbar"] = {key : coeff for key,coeff in zip(model_coeff_keys, params_Clrbar)}
    models_dict["Cell"]["Cl_Lrbar"] = {key : coeff for key,coeff in zip(model_coeff_keys, params_ClLrbar)}
    models_dict["Cell"]["Cl_da"] = {key : coeff for key,coeff in zip(model_coeff_keys, params_Clda)}
    models_dict["Cell"]["Cl_de"] = {key : coeff for key,coeff in zip(model_coeff_keys, params_Clde)}

def _Cm_beta(Cmbeta_data, plot=False, yminmax=(None, None), fn='', dB=0.):
    betas = Cmbeta_data[:, 1]*np.pi/180.
    Cm = Cmbeta_data[:, 12]
    [Cm_beta, Cm0] = np.polyfit(betas, Cm, 1)
    if plot:
        plt.figure()
        plt.scatter(betas*180/np.pi, Cm, edgecolors='k', facecolor='None', s=60, label='Data')
        plt.plot(betas*180/np.pi, Cm0 + Cm_beta*betas, color='r', label='Fit')
        plt.annotate(f'$\delta_B =$ {dB:3.2f}', (-4., 0.13), fontsize=18)
        plt.annotate(r'$C_{{m,\beta}} = {0:3.2f}$'.format(Cm_beta), (-4., 0.1), fontsize=18)
        plt.xlim(-6.1, 6.1)
        plt.ylim(yminmax[0], yminmax[1])
        plt.xlabel(r'$\beta$, deg')
        plt.ylabel(r'$C_m$')
        plt.legend()
        plt.tight_layout()
        plt.savefig(fn)
        plt.close()
    return Cm_beta

def _Cm_pbar(Cmpbar_data):
    Cm1 = np.array([x[12] for x in Cmpbar_data if x[5] == 0.])
    Cmp_p = np.array([x[12] for x in Cmpbar_data if x[5] == 90.*np.pi/180.])
    Cmp_m = np.array([x[12] for x in Cmpbar_data if x[5] == -90.*np.pi/180.])
    DCmpbar_p = (Cmp_p - Cm1)/(np.deg2rad(90.)*b_w/(2.*V))
    DCmpbar_m = (Cmp_m - Cm1)/(np.deg2rad(-90.)*b_w/(2.*V))
    Cm_pbar = np.average(np.vstack((DCmpbar_p, DCmpbar_m)))
    return Cm_pbar

def _Cm_rbar(Cmrbar_data):
    Cm1 = np.array([x[12] for x in Cmrbar_data if x[7] == 0.])
    Cmr_p = np.array([x[12] for x in Cmrbar_data if x[7] == 30.*np.pi/180.])
    Cmr_m = np.array([x[12] for x in Cmrbar_data if x[7] == -30.*np.pi/180.])
    DCmrbar_p = (Cmr_p - Cm1)/(np.deg2rad(30.)*b_w/(2.*V))
    DCmrbar_m = (Cmr_m - Cm1)/(np.deg2rad(-30.)*b_w/(2.*V))
    Cm_rbar = np.average(np.vstack((DCmrbar_p, DCmrbar_m)))
    return Cm_rbar

def _Cm_da(Cmda_data):
    Cm1 = np.array([x[12] for x in Cmda_data if x[3] == 0.])
    Cmda_p = np.array([x[12] for x in Cmda_data if x[3] == 20.])
    DCmda_p = (Cmda_p - Cm1)/np.deg2rad(20.)
    Cm_da = np.average(DCmda_p)
    return Cm_da

def Cm_models(baseline_coeffs, plot=True):
    weight_Cm0 = [True]*N_dB
    # weight_Cm0 = (abs(dB_range) < 140.)*(abs(dB_range) > 20.)
    modelCm0 = lambda x : x[0]*np.sin(2.*dB_rad + np.pi/2.) + np.average(Cm0_dB[weight_Cm0])
    errorCm0 = lambda x : (x[0]*np.sin(2.*dB_rad[weight_Cm0] + np.pi/2.) + np.average(Cm0_dB[weight_Cm0]) - Cm0_dB[weight_Cm0])
    params_Cm0 = np.append(optimize.leastsq(errorCm0, [0.02])[0], [2., np.pi/2., np.average(Cm0_dB)])

    weight_Cmalpha = (abs(Cmalpha_dB) < 0.3)
    modelCmalpha = lambda x : x[0]*np.sin(2.*dB_rad + np.pi/2.) + np.average(Cmalpha_dB[weight_Cmalpha])
    errorCmalpha = lambda x : (x[0]*np.sin(2.*dB_rad + np.pi/2.) + np.average(Cmalpha_dB[weight_Cmalpha]) - Cmalpha_dB)[weight_Cmalpha]
    params_Cmalpha = np.append(optimize.leastsq(errorCmalpha, [0.4])[0], [2., np.pi/2., np.average(Cmalpha_dB[weight_Cmalpha])])

    modelCmbeta = lambda x : x[0]*np.sin(2.*dB_rad)
    errorCmbeta = lambda x : (x[0]*np.sin(2.*dB_rad) - Cmbeta_dB)
    params_Cmbeta = np.append(optimize.leastsq(errorCmbeta, [0.6])[0], [2., 0., 0.])

    weight_Cmpbar = np.abs(Cmpbar_dB) <= 0.02
    modelCmpbar = lambda x : x[0]*np.sin(2.*dB_rad)
    errorCmpbar = lambda x : (x[0]*np.sin(2.*dB_rad) - Cmpbar_dB)[weight_Cmpbar]
    # params_Cmpbar = [0.]*4
    params_Cmpbar = np.append(optimize.leastsq(errorCmpbar, [0.02])[0], [2., 0., 0.])

    weight_Cmqbar = Cmqbar_dB < 0.
    modelCmqbar = lambda x : x[0]*np.sin(2.*dB_rad + np.pi/2.) + np.average(Cmqbar_dB[weight_Cmqbar])
    errorCmqbar = lambda x : (x[0]*np.sin(2.*dB_rad + np.pi/2.) + np.average(Cmqbar_dB[weight_Cmqbar]) - Cmqbar_dB)[weight_Cmqbar]
    params_Cmqbar = np.append(optimize.leastsq(errorCmqbar, [2.])[0], [2., np.pi/2., np.average(Cmqbar_dB[weight_Cmqbar])])

    weight_Cmrbar = [True]*N_dB
    weight_Cmrbar[8] = False
    weight_Cmrbar[10] = False
    weight_Cmrbar[-9] = False
    weight_Cmrbar[-11] = False
    modelCmrbar = lambda x : x[0]*np.sin(2.*dB_rad)
    errorCmrbar = lambda x : (x[0]*np.sin(2.*dB_rad[weight_Cmrbar]) - Cmrbar_dB[weight_Cmrbar])
    params_Cmrbar = np.append(optimize.leastsq(errorCmrbar, [2.])[0], [2., 0., 0.])

    modelCmda = lambda x :x[0]*np.sin(2.*dB_rad) + np.average(Cmda_dB)
    errorCmda = lambda x : (modelCmda(x) - Cmda_dB)
    params_Cmda = np.append(optimize.leastsq(errorCmda, [0.01])[0], [2., 0., np.average(Cmda_dB)])

    modelCmde = lambda x : x[0]*np.sin(1.*dB_rad + np.pi/2.)
    errorCmde = lambda x : (x[0]*np.sin(1.*dB_rad + np.pi/2.) - Cmde_dB)
    params_Cmde = np.append(optimize.leastsq(errorCmde, [-2.])[0], [1., np.pi/2., 0.])


    if plot:
        figs = []
        figs += [_plot_data_fit(params_Cm0[-1], Cm0_dB, Cm0_delta, modelCm0, params_Cm0, meanCm_1p, r'$\hat{C}_{m_0}$', baseline_coeffs["Cm_0"], y_lim=(-0.055, 0.015), dy={"major": 0.01, "minor": 0.01/4})]
        print("Cm0", (np.max(modelCm0(params_Cm0)) - np.min(modelCm0(params_Cm0))))
        figs += [_plot_data_fit(params_Cmalpha[-1], Cmalpha_dB, Cma_delta, modelCmalpha, params_Cmalpha, meanCm_1p/max_alpha, r'$\hat{C}_{m,\alpha}$', baseline_coeffs["Cm_alpha"], y_lim=(-0.05, 0.65), dy={"major": 0.1, "minor": 0.1/4})]
        print("Cma", (np.max(modelCmalpha(params_Cmalpha)) - np.min(modelCmalpha(params_Cmalpha)))*max_alpha)
        figs += [_plot_data_fit(params_Cmbeta[-1], Cmbeta_dB, Cmbeta_delta, modelCmbeta, params_Cmbeta, meanCm_1p/max_beta, r'$\hat{C}_{m,\beta}$', 0., y_lim=(-1., 1.), dy={"major": 0.3, "minor": 0.3/4})]
        print("Cmb", (np.max(modelCmbeta(params_Cmbeta)) - np.min(modelCmbeta(params_Cmbeta)))*max_beta)
        figs += [_plot_data_fit(params_Cmpbar[-1], Cmpbar_dB, Cmpbar_delta, modelCmpbar, params_Cmpbar, meanCm_1p/max_pbar, r'$\hat{C}_{m,\bar{p}}$', 0., y_lim=(-0.035, 0.035), dy={"major": 0.01, "minor": 0.01/4})]
        print("Cmp", (np.max(modelCmpbar(params_Cmpbar)) - np.min(modelCmpbar(params_Cmpbar)))*max_pbar)
        figs += [_plot_data_fit(params_Cmqbar[-1], Cmqbar_dB, Cmqbar_delta, modelCmqbar, params_Cmqbar, meanCm_1p/max_qbar, r'$\hat{C}_{m,\bar{q}}$', baseline_coeffs["Cm_qbar"], y_lim=(-5.5, 1.5), dy={"major": 1.0, "minor": 1.0/4})]
        print("Cmq", (np.max(modelCmqbar(params_Cmqbar)) - np.min(modelCmqbar(params_Cmqbar)))*max_qbar)
        figs += [_plot_data_fit(params_Cmrbar[-1], Cmrbar_dB, Cmrbar_delta, modelCmrbar, params_Cmrbar, meanCm_1p/max_rbar, r'$\hat{C}_{m,\bar{r}}$', 0., y_lim=(-1.7, 1.7), dy={"major": 0.5, "minor": 0.5/4})]
        print("Cmr", (np.max(modelCmrbar(params_Cmrbar)) - np.min(modelCmrbar(params_Cmrbar)))*max_rbar)
        figs += [_plot_data_fit(params_Cmde[-1], Cmde_dB, Cmde_delta, modelCmde, params_Cmde, meanCm_1p/max_de, r'$\hat{C}_{m,\delta_e}$', baseline_coeffs["Cm_de"], y_lim=(-0.8, 1.4), dy={"major": 0.3, "minor": 0.3/4})]
        print("Cmde", (np.max(modelCmde(params_Cmde)) - np.min(modelCmde(params_Cmde)))*max_de)
        figs += [_plot_data_fit(params_Cmda[-1], Cmda_dB, Cmda_delta, modelCmda, params_Cmda, meanCm_1p/max_da, r'$\hat{C}_{m,\delta_a}$', 0., y_lim=(-0.0035, 0.0035), dy={"major": 0.001, "minor": 0.001/4})]
        print("Cmda", (np.max(modelCmda(params_Cmda)) - np.min(modelCmda(params_Cmda)))*max_da)
        plot_dir = "./Fit Figures/"
        print("Average 1%", meanCm_1p)
        if not isdir(plot_dir):
            mkdir(plot_dir)
        pdf = matplotlib.backends.backend_pdf.PdfPages(plot_dir + "/Pitch_BIRE_fits.pdf")
        fig_names = ["Cm0", "Cma", "Cmb", "Cmp", "Cmq", "Cmr", "Cmde", "Cmda"]
        i = 0
        for fig in figs: ## will open an empty extra figure :(
            fig.savefig(plot_dir + fig_names[i] + "_BIRE.pdf")
            pdf.savefig( fig )
            i += 1
        pdf.close()

    models_dict["Cm"]["Cm_0"] = {key : coeff for key,coeff in zip(model_coeff_keys, params_Cm0)}
    models_dict["Cm"]["Cm_alpha"] = {key : coeff for key,coeff in zip(model_coeff_keys, params_Cmalpha)}
    models_dict["Cm"]["Cm_beta"] = {key : coeff for key,coeff in zip(model_coeff_keys, params_Cmbeta)}
    models_dict["Cm"]["Cm_pbar"] = {key : coeff for key,coeff in zip(model_coeff_keys, params_Cmpbar)}
    models_dict["Cm"]["Cm_qbar"] = {key : coeff for key,coeff in zip(model_coeff_keys, params_Cmqbar)}
    models_dict["Cm"]["Cm_rbar"] = {key : coeff for key,coeff in zip(model_coeff_keys, params_Cmrbar)}
    models_dict["Cm"]["Cm_da"] = {key : coeff for key,coeff in zip(model_coeff_keys, params_Cmda)}
    models_dict["Cm"]["Cm_de"] = {key : coeff for key,coeff in zip(model_coeff_keys, params_Cmde)}

def _Cn_alpha(Cnalpha_data):
    alphas = Cnalpha_data[:, 0]*np.pi/180.
    Cn = Cnalpha_data[:, 13]
    [Cn_alpha, Cn0] = np.polyfit(alphas, Cn, 1)
    return Cn_alpha

def _Cn_qbar(Cnqbar_data):
    Cn1 = np.array([x[13] for x in Cnqbar_data if x[6] == 0.])
    Cnq_p = np.array([x[13] for x in Cnqbar_data if x[6] == 30.*np.pi/180.])
    Cnq_m = np.array([x[13] for x in Cnqbar_data if x[6] == -30.*np.pi/180.])
    DCnqbar_p = (Cnq_p - Cn1)/(np.deg2rad(30.)*c_w/(2.*V))
    DCnqbar_m = (Cnq_m - Cn1)/(np.deg2rad(-30.)*c_w/(2.*V))
    Cn_qbar = np.average(np.vstack((DCnqbar_p, DCnqbar_m)))
    return Cn_qbar

def _Cn_de(Cnde_data):
    Cn1 = np.array([x[13] for x in Cnde_data if x[2] == 0.])
    Cnde_p = np.array([x[13] for x in Cnde_data if x[2] == 10.])
    Cnde_m = np.array([x[13] for x in Cnde_data if x[2] == -10.])
    DCnde_p = (Cnde_p - Cn1)/np.deg2rad(10.)
    DCnde_m = (Cnde_m - Cn1)/np.deg2rad(-10.)
    Cn_de = np.average(np.vstack((DCnde_p, DCnde_m)))
    return Cn_de

def Cn_models(baseline_coeffs, plot=True):
    modelCn0 = lambda x : x[0]*np.sin(2.*dB_rad)
    errorCn0 = lambda x : (x[0]*np.sin(2.*dB_rad) - Cn0_dB)
    params_Cn0 = np.append(optimize.leastsq(errorCn0, [-0.01])[0], [2., 0., 0.])

    modelCnalpha = lambda x : x[0]*np.sin(2.*dB_rad)
    errorCnalpha = lambda x : (x[0]*np.sin(2.*dB_rad) - Cnalpha_dB)
    params_Cnalpha = np.append(optimize.leastsq(errorCnalpha, [-0.2])[0], [2., 0., 0.])

    modelCnbeta = lambda x : x[0]*np.sin(2.*dB_rad + np.pi/2.) + np.average(Cnbeta_dB)
    errorCnbeta = lambda x : (x[0]*np.sin(2.*dB_rad + np.pi/2.) + np.average(Cnbeta_dB) - Cnbeta_dB)
    params_Cnbeta = np.append(optimize.leastsq(errorCnbeta, [1.])[0], [2., np.pi/2., np.average(Cnbeta_dB)])

    modelCnpbar = lambda x : 0.*dB_rad + Cnpbar_dB[N_dB//2]
    params_Cnpbar = [0.]*3 + [Cnpbar_dB[N_dB//2]]

    weight_CnLpbar = (CnLpbar_dB < -0.1)*(CnLpbar_dB > -0.14)
    modelCnLpbar = lambda x : x[0]*np.sin(2.*dB_rad[weight_CnLpbar] + np.pi/2.) + np.average(CnLpbar_dB[weight_CnLpbar])
    errorCnLpbar = lambda x : modelCnLpbar(x) - CnLpbar_dB[weight_CnLpbar]
    params_CnLpbar = np.append(optimize.leastsq(errorCnLpbar, [0.001])[0], [2., np.pi/2., np.average(CnLpbar_dB[weight_CnLpbar])])
    # params_CnLpbar = [0.]*3 + [np.average(CnLpbar_dB[weight_CnLpbar])]

    modelCnqbar = lambda x : x[0]*np.sin(2.*dB_rad)
    errorCnqbar = lambda x : (x[0]*np.sin(2.*dB_rad) - Cnqbar_dB)
    params_Cnqbar = np.append(optimize.leastsq(errorCnqbar, [1.6])[0], [2., 0., 0.])

    weight_Cnrbar = [True]*N_dB
    weight_Cnrbar[8] = False
    weight_Cnrbar[10] = False
    weight_Cnrbar[-9] = False
    weight_Cnrbar[-11] = False
    modelCnrbar = lambda x : x[0]*np.sin(2.*dB_rad + np.pi/2.) + np.average(Cnrbar_dB[weight_Cnrbar])
    errorCnrbar = lambda x : (x[0]*np.sin(2.*dB_rad[weight_Cnrbar] + np.pi/2.) + np.average(Cnrbar_dB[weight_Cnrbar]) - Cnrbar_dB[weight_Cnrbar])
    params_Cnrbar = np.append(optimize.leastsq(errorCnrbar, [1.])[0], [2., np.pi/2., np.average(Cnrbar_dB[weight_Cnrbar])])

    modelCnda = lambda x : 0.*dB_rad + np.average(Cnda_dB)
    params_Cnda = [0.]*3 + [np.average(Cnda_dB)]

    modelCnLda = lambda x : x[0]*np.sin(2.*dB_rad + np.pi/2.) + np.average(CnLda_dB)
    errorCnLda = lambda x : (x[0]*np.sin(2.*dB_rad + np.pi/2.) + np.average(CnLda_dB) - CnLda_dB)
    params_CnLda = [0.]*3 + [np.average(CnLda_dB)]
    params_CnLda = np.append(optimize.leastsq(errorCnLda, [1.])[0], [2., np.pi/2., np.average(CnLda_dB)])

    modelCnde = lambda x : x[0]*np.sin(dB_rad)
    errorCnde = lambda x : (x[0]*np.sin(dB_rad) - Cnde_dB)
    params_Cnde = np.append(optimize.leastsq(errorCnde, [2.])[0], [1., 0., 0.])

    if plot:
        figs = []
        figs += [_plot_data_fit(params_Cn0[-1], Cn0_dB, Cn0_delta, modelCn0, params_Cn0, meanCn_1p, r'$\hat{C}_{n_0}$', 0., y_lim=(-0.007, 0.007), dy={"major": 0.002, "minor": 0.002/4})]
        print("Cn0", (np.max(modelCn0(params_Cn0)) - np.min(modelCn0(params_Cn0))))
        figs += [_plot_data_fit(params_Cnalpha[-1], Cnalpha_dB, Cnalpha_delta, modelCnalpha, params_Cnalpha, meanCn_1p/max_alpha, r'$\hat{C}_{n,\alpha}$', 0., y_lim=(-0.16, 0.16), dy={"major": 0.05, "minor": 0.05/4})]
        print("Cna", (np.max(modelCnalpha(params_Cnalpha)) - np.min(modelCnalpha(params_Cnalpha)))*max_alpha)
        figs += [_plot_data_fit(params_Cnbeta[-1], Cnbeta_dB, Cnbeta_delta, modelCnbeta, params_Cnbeta, meanCn_1p/max_beta, r'$\hat{C}_{n,\beta}$', baseline_coeffs["Cn_beta"], y_lim=(-0.3, 1.1), dy={"major": 0.2, "minor": 0.2/4})]
        print("Cnb", (np.max(modelCnbeta(params_Cnbeta)) - np.min(modelCnbeta(params_Cnbeta)))*max_beta)
        figs += [_plot_data_fit(params_Cnpbar[-1], Cnpbar_dB, Cnpbar_delta, modelCnpbar, params_Cnpbar, meanCn_1p/max_pbar, r'$\hat{C}_{n,\bar{p}}$', baseline_coeffs["Cn_pbar"], y_lim=(-0.025, 0.045), dy={"major": 0.01, "minor": 0.01/4})]
        print("Cnp", (np.max(modelCnpbar(params_Cnpbar)) - np.min(modelCnpbar(params_Cnpbar)))*max_pbar)
        figs += [_plot_data_fit(params_CnLpbar[-1], CnLpbar_dB, CnLpbar_delta, modelCnLpbar, params_CnLpbar, meanCn_1p/max_CL1/max_pbar, r'$\hat{C}_{n,L\bar{p}}$', baseline_coeffs["Cn_Lpbar"], y_lim=(-0.13, 0.01), dy={"major": 0.02, "minor": 0.02/4})]
        print("CnLp", (np.max(modelCnLpbar(params_CnLpbar)) - np.min(modelCnLpbar(params_CnLpbar)))*max_CL1*max_pbar)
        figs += [_plot_data_fit(params_Cnqbar[-1], Cnqbar_dB, Cnqbar_delta, modelCnqbar, params_Cnqbar, meanCn_1p/max_qbar, r'$\hat{C}_{n,\bar{q}}$', 0., y_lim=(-1.4, 1.4), dy={"major": 0.4, "minor": 0.4/4})]
        print("Cnq", (np.max(modelCnqbar(params_Cnqbar)) - np.min(modelCnqbar(params_Cnqbar)))*max_qbar)
        figs += [_plot_data_fit(params_Cnrbar[-1], Cnrbar_dB, Cnrbar_delta, modelCnrbar, params_Cnrbar, meanCn_1p/max_rbar, r'$\hat{C}_{n,\bar{r}}$', baseline_coeffs["Cn_rbar"], y_lim=(-0.65, 0.05), dy={"major": 0.1, "minor": 0.1/4})]
        print("Cnr", (np.max(modelCnrbar(params_Cnrbar)) - np.min(modelCnrbar(params_Cnrbar)))*max_rbar)
        figs += [_plot_data_fit(params_Cnde[-1], Cnde_dB, Cnde_delta, modelCnde, params_Cnde, meanCn_1p/max_de, r'$\hat{C}_{n,\delta_e}$', 0., y_lim=(-0.65, 0.65), dy={"major": 0.2, "minor": 0.2/4})]
        print("Cnde", (np.max(modelCnde(params_Cnde)) - np.min(modelCnde(params_Cnde)))*max_de)
        figs += [_plot_data_fit(params_Cnda[-1], Cnda_dB, Cnda_delta, modelCnda, params_Cnda, meanCn_1p/max_da, r'$\hat{C}_{n,\delta_a}$', baseline_coeffs["Cn_da"], y_lim=(-0.045, 0.025), dy={"major": 0.01, "minor": 0.01/4})]
        print("Cnda", (np.max(modelCnda(params_Cnda)) - np.min(modelCnda(params_Cnda)))*max_da)
        figs += [_plot_data_fit(params_CnLda[-1], CnLda_dB, CnLda_delta, modelCnLda, params_CnLda, meanCn_1p/max_CL1/max_da, r'$\hat{C}_{n,L\delta_a}$', baseline_coeffs["Cn_Lda"], y_lim=(-0.005, 0.065), dy={"major": 0.01, "minor": 0.01/4})]
        print("CnLda", (np.max(modelCnLda(params_CnLda)) - np.min(modelCnLda(params_CnLda)))*max_CL1*max_da)
        print("Average 1%", meanCn_1p)
        plot_dir = "./Fit Figures/"
        if not isdir(plot_dir):
            mkdir(plot_dir)
        pdf = matplotlib.backends.backend_pdf.PdfPages(plot_dir + "/Yaw_BIRE_fits.pdf")
        fig_names = ["Cn0", "Cna", "Cnb", "Cnp", "CnLp", "Cnq", "Cnr", "Cnde", "Cnda", "CnLda"]
        i = 0
        for fig in figs: ## will open an empty extra figure :(
            fig.savefig(plot_dir + fig_names[i] + "_BIRE.pdf")
            pdf.savefig( fig )
            i += 1
        pdf.close()

    models_dict["Cn"]["Cn_0"] = {key : coeff for key,coeff in zip(model_coeff_keys, params_Cn0)}
    models_dict["Cn"]["Cn_alpha"] = {key : coeff for key,coeff in zip(model_coeff_keys, params_Cnalpha)}
    models_dict["Cn"]["Cn_beta"] = {key : coeff for key,coeff in zip(model_coeff_keys, params_Cnbeta)}
    models_dict["Cn"]["Cn_pbar"] = {key : coeff for key,coeff in zip(model_coeff_keys, params_Cnpbar)}
    models_dict["Cn"]["Cn_Lpbar"] = {key : coeff for key,coeff in zip(model_coeff_keys, params_CnLpbar)}
    models_dict["Cn"]["Cn_qbar"] = {key : coeff for key,coeff in zip(model_coeff_keys, params_Cnqbar)}
    models_dict["Cn"]["Cn_rbar"] = {key : coeff for key,coeff in zip(model_coeff_keys, params_Cnrbar)}
    models_dict["Cn"]["Cn_da"] = {key : coeff for key,coeff in zip(model_coeff_keys, params_Cnda)}
    models_dict["Cn"]["Cn_Lda"] = {key : coeff for key,coeff in zip(model_coeff_keys, params_CnLda)}
    models_dict["Cn"]["Cn_de"] = {key : coeff for key,coeff in zip(model_coeff_keys, params_Cnde)}

def create_database(inp_dir):
    data = np.zeros((N_dB*(N_alpha*N_other_a + N_beta*N_other_b), 14))
    params = np.zeros(8)
    zz = 0
    k = 0
    for dB in dB_range:
        params[4] = dB
        print("BIRE Angle : ", dB)
        for a in alpha_range:
            params[0] = a
            data[zz, :] = bire_case(params, inp_dir, scenes[k])
            zz += 1
        params[0] = 0.
        for b in beta_range:
            params[1] = b
            data[zz, :] = bire_case(params, inp_dir, scenes[k])
            zz += 1
        params[1] = 0.
        for e in de_range:
            params[2] = e
            for a in alpha_range:
                params[0] = a
                data[zz, :] = bire_case(params, inp_dir, scenes[k])
                zz += 1
        params[2] = 0.
        params[0] = 0.
        for da in da_range:
            params[3] = da
            for b in beta_range:
                params[1] = b
                data[zz, :] = bire_case(params, inp_dir, scenes[k])
                zz += 1
            params[1] = 0.
            for a in alpha_range:
                params[0] = a
                data[zz, :] = bire_case(params, inp_dir, scenes[k])
                zz += 1
        params[0] = 0.
        params[3] = 0.
        for p in p_range:
            params[5] = p
            for a in alpha_range:
                params[0] = a
                data[zz, :] = bire_case(params, inp_dir, scenes[k])
                zz += 1
            params[0] = 0.
            for b in beta_range:
                params[1] = b
                data[zz, :] = bire_case(params, inp_dir, scenes[k])
                zz += 1
            params[1] = 0.
        params[5] = 0.
        for q in q_range:
            params[6] = q
            for a in alpha_range:
                params[0] = a
                data[zz, :] = bire_case(params, inp_dir, scenes[k])
                zz += 1
            params[0] = 0.
            for b in beta_range:
                params[1] = b
                data[zz, :] = bire_case(params, inp_dir, scenes[k])
                zz += 1
            params[1] = 0.
        params[6] = 0.
        for r in r_range:
            params[7] = r
            for a in alpha_range:
                params[0] = a
                data[zz, :] = bire_case(params, inp_dir, scenes[k])
                zz += 1
            params[0] = 0.
            for b in beta_range:
                params[1] = b
                data[zz, :] = bire_case(params, inp_dir, scenes[k])
                zz += 1
            params[1] = 0.
        params[7] = 0.
        k += 1
    return data

if __name__ == "__main__":
    plt.close('all')
    path_to_db_file = './BIRE_database.csv'
    
    file_exists = exists(path_to_db_file)
    c_w = 11.32
    b_w = 30.
    V = 222.5211
    
    if not file_exists:
        alpha_range = np.arange(-10., 11., 5.)
        N_alpha = len(alpha_range)
        beta_range = np.arange(-6., 7., 2.)
        N_beta = len(beta_range)
        da_range = np.array([-20., 20.])
        dB_range = np.arange(-180., 185., 5.)
        N_dB = len(dB_range)
        de_range = np.array([-10., 10.])
        p_range = np.array([-90., 90.])*np.pi/180.
        q_range = np.array([-30., 30.])*np.pi/180.
        r_range = np.array([-30., 30.])*np.pi/180.
        N_other_a = 1 + len(de_range) + len(p_range) + len(q_range) + len(r_range) + len(da_range)
        N_other_b = 1 + len(p_range) + len(q_range) + len(r_range) + len(da_range)
        scenes = []
        print("Making Inputs")
        for d_B in dB_range:
            print(d_B)
            input_file = "./BIRE Inputs/BIRE_input_dB_" + str(d_B) + ".json"
            input_exists = exists(input_file)
            if not input_exists:
                input_file = create_inputs("./BIRE Inputs/", d_B)
            scenes.append(mx.Scene(input_file))
        forces_options = {'body_frame': True,
                          'stab_frame': False,
                          'wind_frame': True,
                          'dimensional': False,
                          'verbose': False}
        print("Creating Database")
        database = np.unique(create_database('./BIRE Inputs/'), axis=0)
        np.savetxt(path_to_db_file, database, delimiter=',')
    else:
        dB_range = np.arange(-180., 185., 5.)
        N_dB = len(dB_range)
        database = np.genfromtxt(path_to_db_file, delimiter=',')

    df = pd.DataFrame(database, columns = ['Alpha','Beta','d_e', 'd_a', 'd_B', 'p', 'q', 'r', 'CD', 'CS', 'CL', 'Cl', 'Cm', 'Cn'])

    dB_rad = np.deg2rad(dB_range)

    CL0_dB = np.zeros(N_dB)
    CLalpha_dB = np.zeros(N_dB)
    CLbeta_dB = np.zeros(N_dB)
    CLpbar_dB = np.zeros(N_dB)
    CLqbar_dB = np.zeros(N_dB)
    CLrbar_dB = np.zeros(N_dB)
    CLda_dB = np.zeros(N_dB)
    CLde_dB = np.zeros(N_dB)

    CL0_delta = 0.
    CLalpha_delta = 0.
    CLbeta_delta = 0.
    CLpbar_delta = 0.
    CLqbar_delta = 0.
    CLrbar_delta = 0.
    CLda_delta = 0.
    CLde_delta = -0.1822

    CS0_dB = np.zeros(N_dB)
    CSalpha_dB = np.zeros(N_dB)
    CSbeta_dB = np.zeros(N_dB)
    CSpbar_dB = np.zeros(N_dB)
    CSLpbar_dB = np.zeros(N_dB)
    CSqbar_dB = np.zeros(N_dB)
    CSrbar_dB = np.zeros(N_dB)
    CSLrbar_dB = np.zeros(N_dB)
    CSda_dB = np.zeros(N_dB)
    CSde_dB = np.zeros(N_dB)

    CS0_delta = 0.
    CSalpha_delta = 0.
    CSbeta_delta = -0.1785
    CSpbar_delta = 0.
    CSLpbar_delta = 0.
    CSqbar_delta = 0.
    CSrbar_delta = 0.
    CSda_delta = -0.0448
    CSde_delta = 0.

    CD0_dB = np.zeros(N_dB)
    CDL_dB = np.zeros(N_dB)
    CDL2_dB = np.zeros(N_dB)
    CDS_dB = np.zeros(N_dB)
    CDS2_dB = np.zeros(N_dB)
    CDpbar_dB = np.zeros(N_dB)
    CDSpbar_dB = np.zeros(N_dB)
    CDLqbar_dB = np.zeros(N_dB)
    CDL2qbar_dB = np.zeros(N_dB)
    CDqbar_dB = np.zeros(N_dB)
    CDrbar_dB = np.zeros(N_dB)
    CDSrbar_dB = np.zeros(N_dB)
    CDda_dB = np.zeros(N_dB)
    CDSda_dB = np.zeros(N_dB)
    CDde_dB = np.zeros(N_dB)
    CDLde_dB = np.zeros(N_dB)
    CDde2_dB = np.zeros(N_dB)

    CD0_delta = 0.0154
    CDL_delta = -0.0304
    CDL2_delta = 0.0714
    CDS_delta = 0.
    CDS2_delta = 0.1118
    CDpbar_delta = 0.
    CDSpbar_delta = 0.
    CDqbar_delta = 0.
    CDLqbar_delta = 0.
    CDL2qbar_delta = 0.
    CDrbar_delta = 0.
    CDSrbar_delta = 0.
    CDda_delta = 0.
    CDSda_delta = 0.
    CDde_delta = 0.
    CDLde_delta = 0.
    CDde2_delta = 0.

    Cl0_dB = np.zeros(N_dB)
    Clalpha_dB = np.zeros(N_dB)
    Clbeta_dB = np.zeros(N_dB)
    Clpbar_dB = np.zeros(N_dB)
    Clqbar_dB = np.zeros(N_dB)
    Clrbar_dB = np.zeros(N_dB)
    ClLrbar_dB = np.zeros(N_dB)
    Clda_dB = np.zeros(N_dB)
    Clde_dB = np.zeros(N_dB)

    Cl0_delta = 0.
    Clalpha_delta = 0.
    Clbeta_delta = -0.0101
    Clpbar_delta = 0.
    Clqbar_delta = 0.
    Clrbar_delta = 0.
    ClLrbar_delta = 0.
    Clda_delta = 0.
    Clde_delta = 0.

    Cm0_dB = np.zeros(N_dB)
    Cmalpha_dB = np.zeros(N_dB)
    Cmbeta_dB = np.zeros(N_dB)
    Cmpbar_dB = np.zeros(N_dB)
    Cmqbar_dB = np.zeros(N_dB)
    Cmrbar_dB = np.zeros(N_dB)
    Cmda_dB = np.zeros(N_dB)
    Cmde_dB = np.zeros(N_dB)

    Cm0_delta = -0.0196
    Cma_delta = 0.2865
    Cmbeta_delta = 0.
    Cmpbar_delta = 0.
    Cmqbar_delta = 0.
    Cmrbar_delta = 0.
    Cmda_delta = 0.
    Cmde_delta = 0.2914

    Cn0_dB = np.zeros(N_dB)
    Cnalpha_dB = np.zeros(N_dB)
    Cnbeta_dB = np.zeros(N_dB)
    Cnpbar_dB = np.zeros(N_dB)
    CnLpbar_dB = np.zeros(N_dB)
    Cnqbar_dB = np.zeros(N_dB)
    Cnrbar_dB = np.zeros(N_dB)
    Cnda_dB = np.zeros(N_dB)
    CnLda_dB = np.zeros(N_dB)
    Cnde_dB = np.zeros(N_dB)

    Cn0_delta = 0.
    Cnalpha_delta = 0.
    Cnbeta_delta = -0.0326
    Cnpbar_delta = 0.
    CnLpbar_delta = 0.0602
    Cnqbar_delta = 0.
    Cnrbar_delta = 0.
    Cnda_delta = 0.0122
    CnLda_delta = 0.0254
    Cnde_delta = 0.0

    for i in range(N_dB):
        print(dB_range[i])
        CLalpha_data = df.loc[(df['Beta'] + df['d_e'] + df['d_a'] + df['p'] + df['q'] + df['r'] == 0) & (df['d_B'] == dB_range[i])].to_numpy()
        CL0_dB[i], CLalpha_dB[i] = f16_model._CL0_CLalpha(CLalpha_data, False)


        CLbeta_data = df.loc[(df['Alpha'] + df['d_e'] + df['d_a'] + df['p'] + df['q'] + df['r'] == 0) & (df['d_B'] == dB_range[i]) & (df['Alpha'] == 0)].to_numpy()
        CLbeta_dB[i] = _CL_beta(CLbeta_data)

        CLpbar_data = df.loc[(df['Beta'] + df['d_e'] + df['d_a'] + df['q'] + df['r'] == 0) & (df['d_B'] == dB_range[i])].to_numpy()
        CLpbar_dB[i] = _CL_pbar(CLpbar_data)

        CLqbar_data = df.loc[(df['Beta'] + df['d_e'] + df['d_a'] + df['p'] + df['r'] == 0) & (df['d_B'] == dB_range[i])].to_numpy()
        CLqbar_dB[i] = f16_model._CL_qbar(CLqbar_data, False)

        CLrbar_data = df.loc[(df['Beta'] + df['d_e'] + df['d_a'] + df['p'] + df['q'] == 0) & (df['d_B'] == dB_range[i])].to_numpy()
        CLrbar_dB[i] = _CL_rbar(CLrbar_data)

        CLda_data = df.loc[(df['Beta'] + df['d_e'] + df['p'] + df['q'] + df['r'] == 0) & (df['d_B'] == dB_range[i])].to_numpy()
        CLda_dB[i] = _CL_da(CLda_data)

        CLde_data = df.loc[(df['Beta'] + df['d_a'] + df['p'] + df['q'] + df['r'] == 0) & (df['d_B'] == dB_range[i])].to_numpy()
        CDp_data = df.loc[((df['Alpha'] + df['d_e'] + df['d_a'] + df['q'] + df['r'] == 0) & (df['Alpha'] == 0.) & (df['d_B'] == dB_range[i]))].to_numpy()
        CDr_data = df.loc[((df['Alpha'] + df['d_e'] + df['d_a'] + df['q'] + df['p'] == 0) & (df['Alpha'] == 0.) & (df['d_B'] == dB_range[i]))].to_numpy()

        CLde_dB[i] = f16_model._CL_de(CLde_data, False)

        CS0_dB[i], CSbeta_dB[i] = f16_model._CS_beta(CLbeta_data, False)

        CSalpha_dB[i] = _CS_alpha(CLalpha_data)

        CSpbar_dB[i], CSLpbar_dB[i] = f16_model._CS_pbar(CLpbar_data, False)

        CSqbar_dB[i] = _CS_qbar(CLqbar_data)

        CSrbar_dB[i] = f16_model._CS_rbar(CLrbar_data, False)

        CSda_data = df.loc[((df['Alpha'] + df['d_e'] + df['r'] + df['q'] + df['p'] == 0) & (df['Alpha'] == 0.) & (df['d_B'] == dB_range[i]))].to_numpy()
        CSda_dB[i] = f16_model._CS_da(CSda_data, False, skip_mask=True)

        CSde_dB[i] = _CS_de(CLde_data)

        CD0_dB[i], CDL_dB[i], CDL2_dB[i] = f16_model._CD_polar(CLalpha_data, False)

        CDS_dB[i], CDS2_dB[i] = f16_model._CD_Spolar(CLbeta_data, False)[1:]

        CDpbar_dB[i], CDSpbar_dB[i] = f16_model._CD_pbar(CDp_data, False)

        CDqbar_dB[i], CDLqbar_dB[i], CDL2qbar_dB[i] = f16_model._CD_qbar(CLqbar_data, False)

        CDrbar_dB[i], CDSrbar_dB[i] = f16_model._CD_rbar(CDr_data, False)

        CDda_dB[i], CDSda_dB[i] = f16_model._CD_da(CLda_data, False)[:2]

        CDde_dB[i], CDLde_dB[i], CDde2_dB[i] = f16_model._CD_de(CLde_data, False)

        Cl0_dB[i], Clbeta_dB[i] = f16_model._Cl_beta(CLbeta_data, False)

        Clalpha_dB[i] = _Cl_alpha(CLalpha_data)

        Clpbar_dB[i] = f16_model._Cl_pbar(CLpbar_data, False)

        Clqbar_dB[i] = _Cl_qbar(CLqbar_data)

        Clrbar_dB[i], ClLrbar_dB[i] = f16_model._Cl_rbar(CLrbar_data, False)

        Clda_dB[i] = f16_model._Cl_da(CSda_data, False)

        Clde_dB[i] = _Cl_de(CLde_data)

        Cm0_dB[i], Cmalpha_dB[i] = f16_model._Cm0_Cmalpha(CLalpha_data, False, skip_mask=False)

        Cmbeta_dB[i] = _Cm_beta(CLbeta_data)

        Cmpbar_dB[i] = _Cm_pbar(CLpbar_data)

        Cmqbar_dB[i] = f16_model._Cm_qbar(CLqbar_data, False)

        Cmrbar_dB[i] = _Cm_rbar(CLrbar_data)

        Cmda_dB[i] = _Cm_da(CLda_data)

        Cmde_dB[i] = f16_model._Cm_de(CLde_data, False)

        Cn0_dB[i], Cnbeta_dB[i] = f16_model._Cn_beta(CLbeta_data, False)

        Cnalpha_dB[i] = _Cn_alpha(CLalpha_data)

        Cnpbar_dB[i], CnLpbar_dB[i] = f16_model._Cn_pbar(CLpbar_data, False)

        Cnqbar_dB[i] = _Cn_qbar(CLqbar_data)

        Cnrbar_dB[i] = f16_model._Cn_rbar(CLrbar_data, False)

        Cnda_dB[i], CnLda_dB[i] = f16_model._Cn_da(CLda_data, False)

        Cnde_dB[i] = _Cn_de(CLde_data)
        

    max_alpha = 20.*np.pi/180.
    max_beta = 10.*np.pi/180.
    max_pbar = 90.*b_w/(2.*V)*np.pi/180.
    max_qbar = 30.*c_w/(2.*V)*np.pi/180.
    max_rbar = 30.*b_w/(2.*V)*np.pi/180.
    max_da = 21.5*np.pi/180.
    max_de = 25.*np.pi/180.
    CL1_data = df.loc[(df['Beta'] + df['d_e'] + df['d_a'] + df['p'] + df['q'] + df['r'] == 0)].to_numpy()
    CS1_data = df.loc[(df['Alpha'] + df['d_e'] + df['d_a'] + df['p'] + df['q'] + df['r'] == 0)].to_numpy()
    max_CL1 = np.max(np.abs(CL1_data[:, 10]))
    max_CS1 = np.max(np.abs(CS1_data[:, 9]))

    meanCL_1p = np.average(np.abs(database[:, 10]))*0.01
    meanCS_1p = np.average(np.abs(database[:, 9]))*0.01
    meanCD_1p = np.average(np.abs(database[:, 8]))*0.01
    meanCl_1p = np.average(np.abs(database[:, 11]))*0.01
    meanCm_1p = np.average(np.abs(database[:, 12]))*0.01
    meanCn_1p = np.average(np.abs(database[:, 13]))*0.01

    model_coeff_keys = ["A", "w", "phi", "z"]
    model_coeff_dict = {key: 0. for key in model_coeff_keys}

    models_dict = {"CL": {
                          "CL_0" : model_coeff_dict,
                          "CL_alpha" : model_coeff_dict,
                          "CL_beta" : model_coeff_dict,
                          "CL_pbar" : model_coeff_dict,
                          "CL_qbar" : model_coeff_dict,
                          "CL_rbar" : model_coeff_dict,
                          "CL_da" : model_coeff_dict,
                          "CL_de" : model_coeff_dict
                          },
                    "CS": {
                          "CS_0" : model_coeff_dict,
                          "CS_alpha" : model_coeff_dict,
                          "CS_beta" : model_coeff_dict,
                          "CS_pbar" : model_coeff_dict,
                          "CS_Lpbar" : model_coeff_dict,
                          "CS_qbar" : model_coeff_dict,
                          "CS_rbar" : model_coeff_dict,
                          "CS_da" : model_coeff_dict,
                          "CS_de" : model_coeff_dict
                          },
                    "CD": {
                          "CD_0" : model_coeff_dict,
                          "CD_L" : model_coeff_dict,
                          "CD_L2" : model_coeff_dict,
                          "CD_S" : model_coeff_dict,
                          "CD_S2" : model_coeff_dict,
                          "CD_pbar" : model_coeff_dict,
                          "CD_Spbar" : model_coeff_dict,
                          "CD_qbar" : model_coeff_dict,
                          "CD_Lqbar" : model_coeff_dict,
                          "CD_L2qbar" : model_coeff_dict,
                          "CD_rbar" : model_coeff_dict,
                          "CD_Srbar" : model_coeff_dict,
                          "CD_da" : model_coeff_dict,
                          "CD_Sda" : model_coeff_dict,
                          "CD_de" : model_coeff_dict,
                          "CD_Lde" : model_coeff_dict,
                          "CD_de2" : model_coeff_dict
                          },
                    "Cell": {
                          "Cl_0" : model_coeff_dict,
                          "Cl_alpha" : model_coeff_dict,
                          "Cl_beta" : model_coeff_dict,
                          "Cl_pbar" : model_coeff_dict,
                          "Cl_qbar" : model_coeff_dict,
                          "Cl_rbar" : model_coeff_dict,
                          "Cl_Lrbar" : model_coeff_dict,
                          "Cl_da" : model_coeff_dict,
                          "Cl_de" : model_coeff_dict
                          },
                    "Cm": {
                          "Cm_0" : model_coeff_dict,
                          "Cm_alpha" : model_coeff_dict,
                          "Cm_beta" : model_coeff_dict,
                          "Cm_pbar" : model_coeff_dict,
                          "Cm_qbar" : model_coeff_dict,
                          "Cm_rbar" : model_coeff_dict,
                          "Cm_da" : model_coeff_dict,
                          "Cm_de" : model_coeff_dict
                          },
                    "Cn": {
                          "Cn_0" : model_coeff_dict,
                          "Cn_alpha" : model_coeff_dict,
                          "Cn_beta" : model_coeff_dict,
                          "Cn_pbar" : model_coeff_dict,
                          "Cn_Lpbar" : model_coeff_dict,
                          "Cn_qbar" : model_coeff_dict,
                          "Cn_rbar" : model_coeff_dict,
                          "Cn_da" : model_coeff_dict,
                          "Cn_Lda" : model_coeff_dict,
                          "Cn_de" : model_coeff_dict
                          }
                    }

    base_coeffs_dict = json.load(open('./f16_model.json'))


    CL_models(base_coeffs_dict["CL"], plot=False)
    CS_models(base_coeffs_dict["CS"], plot=False)
    CD_models(base_coeffs_dict["CD"], plot=False)
    Cl_models(base_coeffs_dict["Cell"], plot=False)
    Cm_models(base_coeffs_dict["Cm"], plot=False)
    Cn_models(base_coeffs_dict["Cn"], plot=True)
    with open("bire_model.json", "w") as outfile:
        json.dump(models_dict, outfile, indent=4)

    """ Create GIF for process example"""
    # plots = []
    # filenames = []
    # for i in range(N_dB):
    #     # create file name and append it to a list
    #     filename = f'./Fit Figures/Procedure/{i}.png'
    #     filenames.append(filename)

    #     # save frame
    #     plt.savefig(filename)
    #     plt.close()
    #     CLbeta_data = df.loc[(df['Alpha'] + df['d_e'] + df['d_a'] + df['p'] + df['q'] + df['r'] == 0) & (df['d_B'] == dB_range[i]) & (df['Alpha'] == 0)].to_numpy()
    #     _Cm_beta(CLbeta_data, plot=True, yminmax=(np.min(database[abs(database[:, 1])==6., 12]), np.max(database[abs(database[:, 1])==6., 12])), fn=filename, dB=dB_range[i])

    # # build gif
    # with imageio.get_writer('./Fit Figures/Procedure/fit_process.gif', mode='I') as writer:
    #     for filename in filenames:
    #         image = imageio.imread(filename)
    #         writer.append_data(image)

    # # Remove files
    # for filename in set(filenames):
    #     remove(filename)

