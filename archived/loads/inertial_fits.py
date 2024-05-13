#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 16:15:40 2022

@author: christian
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import json

def model(coeff, dB, sin=True, freq=2., square=False):
    if sin:
        phi = 0.
    else:
        phi = np.pi/2.
    if not square:
        m = lambda x : x[0]*np.sin(freq*dB + phi) + x[1]
        e = lambda x : m(x) - coeff
        res = optimize.leastsq(e, [300, np.average(coeff)])
        A = res[0][0]
        z = res[0][1]
        return A, freq, phi, z
    else:
        m = lambda x : x[0]*np.sin(freq*dB + phi)
        e = lambda x : m(x) - coeff
        A = optimize.leastsq(e, [300])[0][0]
        return A, freq, phi, -A

Ixx = np.array([6235.]*13)
Iyy = np.array([58449., 58427., 58368., 58288., 58207., 58149., 58127., 58149., 58207., 58288., 58368., 58427., 58449.])
Izz = np.array([65445., 65466., 65525., 65606., 65686., 65745., 65766., 65745., 65686., 65606., 65525., 65466., 65445.])
Ixy = np.zeros(13)
Ixz = np.array([-5.]*13)
Iyz = np.array([0., -80., -139., -161., -139., -80., 0., -80., -139., -161., -139., -80., 0.])

dB = np.arange(-90., 95., 15.)
dB_rad = np.deg2rad(dB)

plt.scatter(dB, Ixx)

# A, freq, phi, z = model(Iyy, dB_rad, sin=False, freq=2.)
# plt.plot(dB, A*abs(np.sin(freq*dB_rad + phi) + z))
# plt.plot(dB, A*np.sin(freq*dB_rad + phi) + z)

model_coeff_keys = ["A", "w", "phi", "z"]
model_coeff_dict = {key: 0. for key in model_coeff_keys}

models_dict = {"Ixx": model_coeff_dict,
               "Iyy": model_coeff_dict,
               "Izz": model_coeff_dict,
               "Ixy": model_coeff_dict,
               "Ixz": model_coeff_dict,
               "Iyz": model_coeff_dict}

models_dict["Ixx"] = {key: x for key, x in zip(model_coeff_keys, [0., 0., 0., np.average(Ixx)])}
A, freq, phi, z = model(Iyy, dB_rad, sin=False, freq=2.)
models_dict["Iyy"] = {key: x for key, x in zip(model_coeff_keys, [A, freq, phi, z])}
A, freq, phi, z = model(Izz, dB_rad, sin=False, freq=2.)
models_dict["Izz"] = {key: x for key, x in zip(model_coeff_keys, [A, freq, phi, z])}
A, freq, phi, z = model(Iyz, dB_rad, sin=False, freq=4., square=True)
models_dict["Iyz"] = {key: x for key, x in zip(model_coeff_keys, [A, freq, phi, z])}
models_dict["Ixz"] = {key: x for key, x in zip(model_coeff_keys, [0., 0., 0., np.average(Ixz)])}


with open("bire_inertia_model.json", "w") as outfile:
    json.dump(models_dict, outfile, indent=4)
