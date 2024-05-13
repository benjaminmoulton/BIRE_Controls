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
    solution = trim.trim(V, H, gamma, phi[0], Gamma, shss=False, bire=bire, cg_shift=cg_shift, verbose=False, fixed_point=True, compressible=True)
    n_a = solution.load
    return (n - n_a)**2

H = 15000. #2500.
V = 634.0 # 845.8844204683032 #1051.
a = stdatm_english(H)[-1]
M = V/a
W = 20500.0 # lbf
# print(0.8*a)
gamma = np.deg2rad(0.)
Gamma = 0.1
# nominal cg is 328.084872 inches aft of nose (aft is more negative)
#cg_shift = [2, 0., 0.] 
cg_shift = [0.0, 0., 0.] 
#cg_shift = [0.09707266666, 0., 0.] 
n = 1.0
#phi_base = optimize.minimize(find_loadfactor, 80.*np.pi/180., args=(n, V, False), method='Nelder-Mead').x[0]
#phi_bire = optimize.minimize(find_loadfactor, 80.*np.pi/180., args=(n, V, True), method='Nelder-Mead').x[0]
phi_base = 0.0
#phi_bire = 0
solution_base = trim.trim(V, H, gamma, phi_base, Gamma, shss=True, bire=False, verbose = False, cg_shift=cg_shift, fixed_point=False, compressible=True, props_filename='F16_props_for_loads.json')
[tau_base, alpha_base, beta_base, da_base, de_base, dr_base] = solution_base.x
base_rates = solution_base.rates

print("\n--Baseline Trim--")
# print("Bank angle, phi: ", phi_base*180./np.pi)
# print("??, tau: ", tau_base)
# print("Angle of attack, alpha: ", alpha_base*180./np.pi)
# print("Sideslip angle, beta: ", beta_base*180./np.pi)
# print("Aileron deflection, da: ", da_base*180./np.pi)
# print("Elevator deflection, de: ", de_base*180./np.pi)
# print("Rudder deflection, dr: ", dr_base*180./np.pi)
print("  u  ", solution_base.states[0])
print("  v  ", solution_base.states[1])
print("  w  ", solution_base.states[2])
print("  p  ", solution_base.states[3])
print("  q  ", solution_base.states[4])
print("  r  ", solution_base.states[5])
print(" phi ", solution_base.states[6])
print("theta", solution_base.states[7])
print("_-_-_-_-_-_-_-_-_-_")
print("  da ", solution_base.x[3])
print("  de ", solution_base.x[4])
print("  dr ", solution_base.x[5])
print(" tau ", solution_base.x[0])
print()

quit()


#solution_BIRE = trim.trim(V, H, gamma, phi_bire, Gamma, shss=False, bire=True, cg_shift=cg_shift, fixed_point=False, compressible=True, props_filename='BIRE_props_for_loads.json')
#[tau_bire, alpha_bire, beta_bire, da_bire, de_bire, dB_bire] = solution_BIRE.x
#bire_rates = solution_BIRE.rates

base_input = json.load(open('./F16_input_for_loads.json'))
base_input["scene"]["atmosphere"]["rho"] = stdatm_english(H)[3]

bire_input = json.load(open('./BIRE_input_for_loads.json'))
bire_input["scene"]["atmosphere"]["rho"] = stdatm_english(H)[3]

base_scene = mx.Scene(base_input)
BIRE_scene = mx.Scene(bire_input)

forces_options = {"body_frame": True, "report_by_segment": True}

base_scene.set_aircraft_state({"alpha": alpha_base*180./np.pi,
                               "beta": beta_base*180./np.pi,
                               "rates": base_rates,
                               "velocity": V})
base_scene.set_aircraft_control_state({"elevator": de_base*180./np.pi,
                                       "aileron": da_base*180./np.pi,
                                       "rudder": dr_base*180/np.pi})
FM_base = base_scene.solve_forces(**forces_options)["F16"]

#FM_BIRE = bire_case([alpha_bire*180./np.pi, beta_bire*180./np.pi,
#                     de_bire*180./np.pi, da_bire*180./np.pi,
#                     dB_bire*180./np.pi, bire_rates[0], bire_rates[1], bire_rates[2]],
#                    V, './', scene=BIRE_scene)

baseline_left_stab_Fx = (FM_base["inviscid"]["Fx"]["h_stab_left"] + FM_base["viscous"]["Fx"]["h_stab_left"])/np.sqrt(1. - M**2)
baseline_left_stab_Fy = (FM_base["inviscid"]["Fy"]["h_stab_left"] + FM_base["viscous"]["Fy"]["h_stab_left"])/np.sqrt(1. - M**2)
baseline_left_stab_Fz = (FM_base["inviscid"]["Fz"]["h_stab_left"] + FM_base["viscous"]["Fz"]["h_stab_left"])/np.sqrt(1. - M**2)
baseline_left_stab_Mx = (FM_base["inviscid"]["Mx"]["h_stab_left"] + FM_base["viscous"]["Mx"]["h_stab_left"])/np.sqrt(1. - M**2)
baseline_left_stab_My = (FM_base["inviscid"]["My"]["h_stab_left"] + FM_base["viscous"]["My"]["h_stab_left"])/np.sqrt(1. - M**2)
baseline_left_stab_Mz = (FM_base["inviscid"]["Mz"]["h_stab_left"] + FM_base["viscous"]["Mz"]["h_stab_left"])/np.sqrt(1. - M**2)
x_left_stab = baseline_left_stab_My/baseline_left_stab_Fz

baseline_right_stab_Fx = (FM_base["inviscid"]["Fx"]["h_stab_right"] + FM_base["viscous"]["Fx"]["h_stab_right"])/np.sqrt(1. - M**2)
baseline_right_stab_Fy = (FM_base["inviscid"]["Fy"]["h_stab_right"] + FM_base["viscous"]["Fy"]["h_stab_right"])/np.sqrt(1. - M**2)
baseline_right_stab_Fz = (FM_base["inviscid"]["Fz"]["h_stab_right"] + FM_base["viscous"]["Fz"]["h_stab_right"])/np.sqrt(1. - M**2)
baseline_right_stab_Mx = (FM_base["inviscid"]["Mx"]["h_stab_right"] + FM_base["viscous"]["Mx"]["h_stab_right"])/np.sqrt(1. - M**2)
baseline_right_stab_My = (FM_base["inviscid"]["My"]["h_stab_right"] + FM_base["viscous"]["My"]["h_stab_right"])/np.sqrt(1. - M**2)
baseline_right_stab_Mz = (FM_base["inviscid"]["Mz"]["h_stab_right"] + FM_base["viscous"]["Mz"]["h_stab_right"])/np.sqrt(1. - M**2)


baseline_left_main_Fx = (FM_base["inviscid"]["Fx"]["main_wing_left"] + FM_base["viscous"]["Fx"]["main_wing_left"])/np.sqrt(1. - M**2)
baseline_left_main_Fy = (FM_base["inviscid"]["Fy"]["main_wing_left"] + FM_base["viscous"]["Fy"]["main_wing_left"])/np.sqrt(1. - M**2)
baseline_left_main_Fz = (FM_base["inviscid"]["Fz"]["main_wing_left"] + FM_base["viscous"]["Fz"]["main_wing_left"])/np.sqrt(1. - M**2)
baseline_left_main_Mx = (FM_base["inviscid"]["Mx"]["main_wing_left"] + FM_base["viscous"]["Mx"]["main_wing_left"])/np.sqrt(1. - M**2)
baseline_left_main_My = (FM_base["inviscid"]["My"]["main_wing_left"] + FM_base["viscous"]["My"]["main_wing_left"])/np.sqrt(1. - M**2)
baseline_left_main_Mz = (FM_base["inviscid"]["Mz"]["main_wing_left"] + FM_base["viscous"]["Mz"]["main_wing_left"])/np.sqrt(1. - M**2)
x_left_main = baseline_left_main_My/baseline_left_main_Fz

baseline_right_main_Fx = (FM_base["inviscid"]["Fx"]["main_wing_right"] + FM_base["viscous"]["Fx"]["main_wing_right"])/np.sqrt(1. - M**2)
baseline_right_main_Fy = (FM_base["inviscid"]["Fy"]["main_wing_right"] + FM_base["viscous"]["Fy"]["main_wing_right"])/np.sqrt(1. - M**2)
baseline_right_main_Fz = (FM_base["inviscid"]["Fz"]["main_wing_right"] + FM_base["viscous"]["Fz"]["main_wing_right"])/np.sqrt(1. - M**2)
baseline_right_main_Mx = (FM_base["inviscid"]["Mx"]["main_wing_right"] + FM_base["viscous"]["Mx"]["main_wing_right"])/np.sqrt(1. - M**2)
baseline_right_main_My = (FM_base["inviscid"]["My"]["main_wing_right"] + FM_base["viscous"]["My"]["main_wing_right"])/np.sqrt(1. - M**2)
baseline_right_main_Mz = (FM_base["inviscid"]["Mz"]["main_wing_right"] + FM_base["viscous"]["Mz"]["main_wing_right"])/np.sqrt(1. - M**2)

baseline_v_stab_Fx = (FM_base["inviscid"]["Fx"]["v_stab_left"] + FM_base["viscous"]["Fx"]["v_stab_left"])/np.sqrt(1. - M**2)
baseline_v_stab_Fy = (FM_base["inviscid"]["Fy"]["v_stab_left"] + FM_base["viscous"]["Fy"]["v_stab_left"])/np.sqrt(1. - M**2)
baseline_v_stab_Fz = (FM_base["inviscid"]["Fz"]["v_stab_left"] + FM_base["viscous"]["Fz"]["v_stab_left"])/np.sqrt(1. - M**2)
baseline_v_stab_Mx = (FM_base["inviscid"]["Mx"]["v_stab_left"] + FM_base["viscous"]["Mx"]["v_stab_left"])/np.sqrt(1. - M**2)
baseline_v_stab_My = (FM_base["inviscid"]["My"]["v_stab_left"] + FM_base["viscous"]["My"]["v_stab_left"])/np.sqrt(1. - M**2)
baseline_v_stab_Mz = (FM_base["inviscid"]["Mz"]["v_stab_left"] + FM_base["viscous"]["Mz"]["v_stab_left"])/np.sqrt(1. - M**2)

Total_Lift = baseline_left_main_Fz+baseline_right_main_Fz+baseline_left_stab_Fz+baseline_right_stab_Fz+baseline_v_stab_Fz

Total_Lift = Total_Lift * np.sqrt(1. - M**2)

Total_Lift = Total_Lift * np.cos(alpha_base) / 9.0

#BIRE_left_stab_Fz = (FM_BIRE["inviscid"]["Fz"]["BIRE_left_left"] + FM_BIRE["viscous"]["Fz"]["BIRE_left_left"])/np.sqrt(1. - M**2)
#BIRE_right_stab_Fz = (FM_BIRE["inviscid"]["Fz"]["BIRE_right_right"] + FM_BIRE["viscous"]["Fz"]["BIRE_right_right"])/np.sqrt(1. - M**2)
#BIRE_left_main_Fz = (FM_BIRE["inviscid"]["Fz"]["main_wing_left"] + FM_BIRE["viscous"]["Fz"]["main_wing_left"])/np.sqrt(1. - M**2)
#BIRE_right_main_Fz = (FM_BIRE["inviscid"]["Fz"]["main_wing_right"] + FM_BIRE["viscous"]["Fz"]["main_wing_right"])/np.sqrt(1. - M**2)

#print("\n\n--Results--")
#print("Main Wing Load: ", baseline_left_main_Fz + baseline_right_main_Fz)
#print("Horizontal Stabilizer Load: ", baseline_left_stab_Fz + baseline_right_stab_Fz)
#print("Vertical Stabilizer Load : ", baseline_v_stab_Fz)

print("\n--Baseline Forces--")
print("Main Wing Left X-Load: ", baseline_left_main_Fx)
print("Main Wing Left Y-Load: ", baseline_left_main_Fy)
print("Main Wing Left Z-Load: ", baseline_left_main_Fz)
print("Main Wing Left X-Moment: ", baseline_left_main_Mx)
print("Main Wing Left Y-Moment: ", baseline_left_main_My)
print("Main Wing Left Z-Moment: ", baseline_left_main_Mz)
print("Main Wing Left Load Location: ", x_left_main)
print("\n")
print("Main Wing Right X-Load: ", baseline_right_main_Fx)
print("Main Wing Right Y-Load: ", baseline_right_main_Fy)
print("Main Wing Right Z-Load: ", baseline_right_main_Fz)
print("Main Wing Right X-Moment: ", baseline_right_main_Mx)
print("Main Wing Right Y-Moment: ", baseline_right_main_My)
print("Main Wing Right Z-Moment: ", baseline_right_main_Mz)
print("\n")
print("Horizontal Stabilizer Left X-Load: ", baseline_left_stab_Fx)
print("Horizontal Stabilizer Left Y-Load: ", baseline_left_stab_Fy)
print("Horizontal Stabilizer Left Z-Load: ", baseline_left_stab_Fz)
print("Horizontal Stabilizer Left X-Moment: ", baseline_left_stab_Mx)
print("Horizontal Stabilizer Left Y-Moment: ", baseline_left_stab_My)
print("Horizontal Stabilizer Left Z-Moment: ", baseline_left_stab_Mz)
print("Horizontal Stabilizer Left Load Location: ", x_left_stab)

print("Horizontal Stabilizer Right X-Load: ", baseline_right_stab_Fx)
print("Horizontal Stabilizer Right Y-Load: ", baseline_right_stab_Fy)
print("Horizontal Stabilizer Right Z-Load: ", baseline_right_stab_Fz)
print("Horizontal Stabilizer Right X-Moment: ", baseline_right_stab_Mx)
print("Horizontal Stabilizer Right Y-Moment: ", baseline_right_stab_My)
print("Horizontal Stabilizer Right Z-Moment: ", baseline_right_stab_Mz)

print("Vertical Stabilizer X-Load : ", baseline_v_stab_Fx)
print("Vertical Stabilizer Y-Load : ", baseline_v_stab_Fy)
print("Vertical Stabilizer Z-Load : ", baseline_v_stab_Fz)
print("Vertical Stabilizer X-Moment: ", baseline_v_stab_Mx)
print("Vertical Stabilizer Y-Moment: ", baseline_v_stab_My)
print("Vertical Stabilizer Z-Moment: ", baseline_v_stab_Mz)

print("Total Lift : ", Total_Lift)


print("\n--Results Listing--")
print(cg_shift[0])
print(phi_base*180./np.pi)
print(alpha_base*180./np.pi)
print(beta_base*180./np.pi)
print(da_base*180./np.pi)
print(de_base*180./np.pi)
print(dr_base*180./np.pi)
print("\n")
print(baseline_left_main_Fx)
print(baseline_left_main_Fy)
print(baseline_left_main_Fz)
print(baseline_left_main_Mx)
print(baseline_left_main_My)
print(baseline_left_main_Mz)
print("\n")
print(baseline_right_main_Fx)
print(baseline_right_main_Fy)
print(baseline_right_main_Fz)
print(baseline_right_main_Mx)
print(baseline_right_main_My)
print(baseline_right_main_Mz)
print("\n")
print(baseline_left_stab_Fx)
print(baseline_left_stab_Fy)
print(baseline_left_stab_Fz)
print(baseline_left_stab_Mx)
print(baseline_left_stab_My)
print(baseline_left_stab_Mz)
print("\n")
print(baseline_right_stab_Fx)
print(baseline_right_stab_Fy)
print(baseline_right_stab_Fz)
print(baseline_right_stab_Mx)
print(baseline_right_stab_My)
print(baseline_right_stab_Mz)