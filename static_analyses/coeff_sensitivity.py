#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 16:51:55 2022

@author: christian
"""

import numpy as np
import json
import matplotlib.pyplot as plt
import aero_trim
from f16_aero import F16Aero
from bire_aero import BIREAero
from hunsaker_atm import gravity_english, stdatm_english

def sensitivity_study(V, H, phi, shss, cg_shift, model_fn, max_errors, trim_0):
    solution = aero_trim.trim(V, H, gamma, phi, Gamma, shss=shss, bire=False, cg_shift=cg_shift, fixed_point=True, verbose=False, model_filename=model_fn)
    [tau, alpha, beta, da, de, dr] = solution.x
    [p, q, r] = solution.rates
    [alpha_0, beta_0, da_0, de_0, dr_0, p_0, q_0, r_0] = trim_0
    error_alpha = max(abs((alpha - alpha_0)*180./np.pi), max_errors[0])
    error_beta = max(abs((beta - beta_0)*180./np.pi), max_errors[1])
    error_da = max(abs((da - da_0)*180./np.pi), max_errors[2])
    error_de = max(abs((de - de_0)*180./np.pi), max_errors[3])
    error_dr = max(abs((dr - dr_0)*180./np.pi), max_errors[4])
    error_p = max(abs((p - p_0)*180./np.pi), max_errors[5])
    error_q = max(abs((q - q_0)*180./np.pi), max_errors[6])
    error_r = max(abs((r - r_0)*180./np.pi), max_errors[7])
    max_errors = [error_alpha, error_beta, error_da, error_de, error_dr,
                  error_p, error_q, error_r]
    return max_errors


if __name__ == "__main__":
    M = 0.2
    H = 1000.
    gamma = np.deg2rad(0.)
    Gamma = 0.3
    cg_shift = [0., 0., 0.]
    errors = {"a_error": 0., "b_error": 0., "da_error": 0., "de_error": 0., "dr_error": 0., "p_error": 0., "q_error": 0., "r_error": 0.}
    sensitivity_data = {"CL_0": {key:value for (key, value) in errors.items()},
                        "CL_alpha": {key:value for (key, value) in errors.items()},
                        "CL_qbar": {key:value for (key, value) in errors.items()},
                        "CL_de": {key:value for (key, value) in errors.items()},
                        "CS_beta": {key:value for (key, value) in errors.items()},
                        "CS_pbar": {key:value for (key, value) in errors.items()},
                        "CS_Lpbar": {key:value for (key, value) in errors.items()},
                        "CS_rbar": {key:value for (key, value) in errors.items()},
                        "CS_da": {key:value for (key, value) in errors.items()},
                        "CS_dr": {key:value for (key, value) in errors.items()},
                        "CD_0": {key:value for (key, value) in errors.items()},
                        "CD_L": {key:value for (key, value) in errors.items()},
                        "CD_L2": {key:value for (key, value) in errors.items()},
                        "CD_S2": {key:value for (key, value) in errors.items()},
                        "CD_Spbar": {key:value for (key, value) in errors.items()},
                        "CD_qbar": {key:value for (key, value) in errors.items()},
                        "CD_Lqbar": {key:value for (key, value) in errors.items()},
                        "CD_L2qbar": {key:value for (key, value) in errors.items()},
                        "CD_Srbar": {key:value for (key, value) in errors.items()},
                        "CD_de": {key:value for (key, value) in errors.items()},
                        "CD_Lde": {key:value for (key, value) in errors.items()},
                        "CD_de2": {key:value for (key, value) in errors.items()},
                        "CD_Sda": {key:value for (key, value) in errors.items()},
                        "CD_Sdr": {key:value for (key, value) in errors.items()},
                        "Cl_beta": {key:value for (key, value) in errors.items()},
                        "Cl_pbar": {key:value for (key, value) in errors.items()},
                        "Cl_rbar": {key:value for (key, value) in errors.items()},
                        "Cl_Lrbar": {key:value for (key, value) in errors.items()},
                        "Cl_da": {key:value for (key, value) in errors.items()},
                        "Cl_dr": {key:value for (key, value) in errors.items()},
                        "Cm_0": {key:value for (key, value) in errors.items()},
                        "Cm_alpha": {key:value for (key, value) in errors.items()},
                        "Cm_qbar": {key:value for (key, value) in errors.items()},
                        "Cm_de": {key:value for (key, value) in errors.items()},
                        "Cn_beta": {key:value for (key, value) in errors.items()},
                        "Cn_pbar": {key:value for (key, value) in errors.items()},
                        "Cn_Lpbar": {key:value for (key, value) in errors.items()},
                        "Cn_rbar": {key:value for (key, value) in errors.items()},
                        "Cn_da": {key:value for (key, value) in errors.items()},
                        "Cn_Lda": {key:value for (key, value) in errors.items()},
                        "Cn_dr": {key:value for (key, value) in errors.items()}}
    mux_model = json.load(open('./f16_model.json'))
    nasa_model = json.load(open('./nasa_model.json'))
    a = stdatm_english(H)[-1]
    V = M*a
    for k in mux_model.keys():
        for c in mux_model[k].keys():
            print(c)
            max_error = [0.]*8
            i = 0
            mux_model_adj = json.load(open('./f16_model.json'))
            mux_model_adj[k][c] = nasa_model[k][c]
            with open('./mux_sensitivity.json', 'w') as outfile:
                json.dump(mux_model_adj, outfile, indent=4)
            shss = True
            phi = np.deg2rad(5.)
            solution_shss = aero_trim.trim(V, H, gamma, phi, Gamma, shss=shss, bire=False, cg_shift=cg_shift, fixed_point=True, verbose=False, model_filename='f16_model.json')
            [tau_shss, alpha_shss, beta_shss, da_shss, de_shss, dr_shss] = solution_shss.x
            [p_shss, q_shss, r_shss] = solution_shss.rates
            sens_params_shss = [alpha_shss, beta_shss, da_shss, de_shss, dr_shss, p_shss, q_shss, r_shss]
            max_error = sensitivity_study(V, H, phi, shss, cg_shift, 'mux_sensitivity.json', max_error, sens_params_shss)
            shss = False
            solution_sct = aero_trim.trim(V, H, gamma, phi, Gamma, shss=shss, bire=False, cg_shift=cg_shift, fixed_point=True, verbose=False, model_filename='f16_model.json')
            [tau_sct, alpha_sct, beta_sct, da_sct, de_sct, dr_sct] = solution_sct.x
            [p_sct, q_sct, r_sct] = solution_sct.rates
            sens_params_sct = [alpha_sct, beta_sct, da_sct, de_sct, dr_sct, p_sct, q_sct, r_sct]
            max_error = sensitivity_study(V, H, phi, shss, cg_shift, 'mux_sensitivity.json', max_error, sens_params_sct)
            sensitivity_data[c] = {key: error for key, error in zip(errors.keys(), max_error)}
    with open("./model_sensitivity.json", "w") as outfile:
            json.dump(sensitivity_data, outfile, indent=4)

