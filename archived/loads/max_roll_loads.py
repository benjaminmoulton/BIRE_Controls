#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 15:54:38 2022

@author: christian
"""

import numpy as np
import aero_trim as trim
import machupX as mx
import json
import f16_aero
import bire_aero
import scipy.optimize as optimize
from hunsaker_atm import gravity_english, stdatm_english


def create_inputs(inp_dir, d_B):
    rotation_angle = str(int(d_B))

    f_inp = open(inp_dir + 'BIRE_input_for_loads.json',)
    inp_data = json.load(f_inp)

    f_air = open(inp_dir + 'BIRE_airplane_for_loads.json',)
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

def bire_case(params, V, inp_dir, scene=None):
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
                                         "velocity": V})
    BIRE_scene.set_aircraft_control_state(control_state={"elevator": d_e,
                                                         "aileron": d_a})
    x = BIRE_scene.solve_forces(**forces_options)["BIRE"]
    return x

def find_loadfactor(phi, n, V, bire):
    solution = trim.trim(V, H, gamma, phi[0], Gamma, shss=True, bire=bire, cg_shift=cg_shift, verbose=False, fixed_point=True, compressible=True)
    n_a = solution.load
    return (n - n_a)**2

H = 25600.
a = stdatm_english(H)[-1]
M = 2.
V = M*a
gamma = np.deg2rad(0.)
Gamma = 0.1
cg_shift = [1.825, 0., 0.]
n = 0.16
phi_base = optimize.minimize(find_loadfactor, 0., args=(n, V, False), method='Nelder-Mead').x[0]
phi_bire = optimize.minimize(find_loadfactor, 0., args=(n, V, True), method='Nelder-Mead').x[0]
solution_base = trim.trim(V, H, gamma, phi_base, Gamma, shss=True, bire=False, cg_shift=cg_shift, fixed_point=False, compressible=False, props_filename='F16_props_for_loads.json')
[tau_base, alpha_base, beta_base, da_base, de_base, dr_base] = solution_base.x
base_rates = solution_base.rates
solution_BIRE = trim.trim(V, H, gamma, phi_bire, Gamma, shss=True, bire=True, cg_shift=cg_shift, fixed_point=False, compressible=False, props_filename='BIRE_props_for_loads.json')
[tau_bire, alpha_bire, beta_bire, da_bire, de_bire, dB_bire] = solution_BIRE.x
bire_rates = solution_BIRE.rates

base_input = json.load(open('./F16_input_for_loads.json'))
base_input["scene"]["atmosphere"]["rho"] = stdatm_english(H)[3]

bire_input = json.load(open('./BIRE_input_for_loads.json'))
bire_input["scene"]["atmosphere"]["rho"] = stdatm_english(H)[3]

base_scene = mx.Scene(base_input)
BIRE_scene = mx.Scene(bire_input)
forces_options = {"body_frame": True, "report_by_segment": True}
da_max = 21.5*np.pi/180.
coeffs_base = f16_aero.F16Aero()
coeffs_BIRE = bire_aero.BIREAero()
b_w = trim.AircraftProperties(V, H, Gamma).b_w
dummy = coeffs_base._Cl(0., 0., 0., 0., 0., 0., 0., 0.)
CL_base = solution_base.FM[2]
Clda_base = coeffs_base.Clda
Clp_base = coeffs_base.Clp
Cnda_base = coeffs_base.Cnda
CnLda_base = coeffs_base.CnLda
Cnp_base = coeffs_base.Cnp
CnLp_base = coeffs_base.CnLp
Cndr_base = coeffs_base.Cndr

CL_bire = solution_BIRE.FM[2]
Clda_bire = coeffs_BIRE._Cl_da(dB_bire)
Clp_bire = coeffs_BIRE._Cl_pbar(dB_bire)
Cnda_bire = coeffs_BIRE._Cn_da(dB_bire)
CnLda_bire = coeffs_BIRE._Cn_Lda(dB_bire)
Cnp_bire = coeffs_BIRE._Cn_pbar(dB_bire)
CnLp_bire = coeffs_BIRE._Cn_Lpbar(dB_bire)

roll_rate = np.deg2rad(30.)
pbar = roll_rate*b_w/(2.*V)
da_req_base = -Clp_base*pbar/Clda_base
da_req_bire = -Clp_bire*pbar/Clda_bire
res_Cn_base = (CnLp_base*CL_base + Cnp_base)*pbar + (CnLda_base*CL_base + Cnda_base)*da_req_base
res_Cn_bire = (CnLp_bire*CL_bire + Cnp_bire)*pbar + (CnLda_bire*CL_bire + Cnda_bire)*da_req_bire
CndB_bire = coeffs_BIRE._dCn_dB(alpha_bire, beta_bire, pbar, 0., 0., da_req_bire, de_bire, dB_bire)
dr_req_base = -res_Cn_base/Cndr_base
dB_req_bire = -res_Cn_bire/CndB_bire

base_scene.set_aircraft_state({"alpha": alpha_base*180./np.pi,
                               "beta": beta_base*180./np.pi,
                               "rates": [roll_rate, base_rates[1:]],
                               "velocity": V})
base_scene.set_aircraft_control_state({"elevator": de_base*180./np.pi,
                                       "aileron": da_req_base*180./np.pi,
                                       "rudder": dr_base*180./np.pi})
FM_base = base_scene.solve_forces(**forces_options)["F16"]

FM_BIRE = bire_case([alpha_bire*180./np.pi, beta_bire*180./np.pi,
                     de_bire*180./np.pi, da_req_bire*180./np.pi,
                     dB_bire*180./np.pi + dB_req_bire*180./np.pi, roll_rate, bire_rates[1], bire_rates[2]],
                    V, './', scene=BIRE_scene)

baseline_left_stab_Fz = (FM_base["inviscid"]["Fz"]["h_stab_left"] + FM_base["viscous"]["Fz"]["h_stab_left"])/np.sqrt(M**2 - 1.)
baseline_right_stab_Fz = (FM_base["inviscid"]["Fz"]["h_stab_right"] + FM_base["viscous"]["Fz"]["h_stab_right"])/np.sqrt(M**2 - 1.)
baseline_left_main_Fz = (FM_base["inviscid"]["Fz"]["main_wing_left"] + FM_base["viscous"]["Fz"]["main_wing_left"])/np.sqrt(M**2 - 1.)
baseline_right_main_Fz = (FM_base["inviscid"]["Fz"]["main_wing_right"] + FM_base["viscous"]["Fz"]["main_wing_right"])/np.sqrt(M**2 - 1.)
baseline_v_stab_Fz = (FM_base["inviscid"]["Fz"]["v_stab_left"] + FM_base["viscous"]["Fz"]["v_stab_left"])/np.sqrt(M**2 - 1.)
BIRE_left_stab_Fz = (FM_BIRE["inviscid"]["Fz"]["BIRE_left_left"] + FM_BIRE["viscous"]["Fz"]["BIRE_left_left"])/np.sqrt(M**2 - 1.)
BIRE_right_stab_Fz = (FM_BIRE["inviscid"]["Fz"]["BIRE_right_right"] + FM_BIRE["viscous"]["Fz"]["BIRE_right_right"])/np.sqrt(M**2 - 1.)
BIRE_left_main_Fz = (FM_BIRE["inviscid"]["Fz"]["main_wing_left"] + FM_BIRE["viscous"]["Fz"]["main_wing_left"])/np.sqrt(M**2 - 1.)
BIRE_right_main_Fz = (FM_BIRE["inviscid"]["Fz"]["main_wing_right"] + FM_BIRE["viscous"]["Fz"]["main_wing_right"])/np.sqrt(M**2 - 1.)

print("Main Wing Load: ", baseline_left_main_Fz + baseline_right_main_Fz)
print("Horizontal Stabilizer Load: ", baseline_left_stab_Fz + baseline_right_stab_Fz)
print("Vertical Stabilizer Load : ", baseline_v_stab_Fz)
