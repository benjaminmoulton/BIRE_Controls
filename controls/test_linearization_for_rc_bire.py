import numpy as np
import json
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from cycler import cycler
import mpl_toolkits.mplot3d.axes3d as ax3
from matplotlib.animation import FuncAnimation
from numpy import sign, matmul as mm
from datetime import datetime, timedelta
from time import sleep
import control as co
from scipy.linalg import block_diag
from scipy.integrate import ode, odeint
from scipy.interpolate import interp1d, interpn
from scipy.optimize import curve_fit,minimize,minimize_scalar,newton
from scipy.io import savemat, loadmat
from scipy.signal import tf2zpk as scipy_tf2zpk
# from math import pi, sin, cos, tan, exp, asin, acos, atan, atan2
from numpy import pi, sin, cos, tan, exp, arcsin as asin, arccos as acos, arctan as atan, arctan2 as atan2
from std_atm import stdatm_english
from quat import quat_mult, euler_2_quat, quat_2_euler, quat_norm, body_2_fixed, fixed_2_body, eulerdot_2_quatdot, quatdot_2_eulerdot
from linearization import linearization as lin,Anderson_correction_der_coeff,Anderson_correction_der_M

from controller_simulation import Aircraft,run_single_simulation, \
    monte_carlo_perturbations, report_latex, report_eigprops, rep2D,BIREAero



if __name__ == "__main__":

    # filenames 
    # base_fs_file = "base_fs_in.json"
    # bire_fs_file = "bire_fs_in.json"
    # base_rc_file = "base_rc_in.json"
    bire_rc_file = "bire_rc_in.json"

    # read in json to ensure no file changes while running
    # base_fs_dict = json.loads( open(base_fs_file).read() )
    # bire_fs_dict = json.loads( open(bire_fs_file).read() )
    # base_rc_dict = json.loads( open(base_rc_file).read() )
    bire_rc_dict = json.loads( open(bire_rc_file).read() )

    
    # trim for BIRE, determine LQR for controller code example
    V = 110.0
    H = 4600.0
    compr = False
    stall = False
    phi_trim = 0.0
    #
    # bire_rc_dict["initial"].pop("mach")
    bire_rc_dict["initial"]["airspeed[ft/s]"] = V
    bire_rc_dict["initial"]["altitude[ft]"] = H
    bire_rc_dict["initial"]["trim"]["bank_angle[deg]"] = phi_trim
    bire_rc_dict["simulation"]["include_compressibility"] = compr
    bire_rc_dict["simulation"]["include_stall"] = stall
    bire_rc_dict["simulation"]["use_fitted_thrust_model"] = False
    bire_rc_dict["initial"]["trim"]["type"] = "sct"
    bire_rc_dict["initial"]["type"] = "trim"
    bire = Aircraft(bire_rc_dict)
    # print(bire.inertia_model.W)
    # print(bire.cgshift)
    bire._report_trim_solution()
    # # build linearized system
    bire._build_controller(save_matrices=False,mrrr=[0,1,2,6,7,8,9,10,11],
        mrrc=[3],drop_actrs=True,run_freq=False)
    print(bire.Lin_Model.A)
    print(bire.Lin_Model.B)
    print()

    rows = [3,4,5]
    cols = [0,1,2]
    print((bire.Lin_Model.A[rows])[:,rows])
    print((bire.Lin_Model.B[rows])[:,cols])
    print()

    A,B = bire.Lin_Model.build_jacobians(bire.x_trim_euler,bire.u_trim,bire.cgshift,True,bire._nonlinear_euler_dynamics)
    print("numerical")
    print((A[rows])[:,rows])
    print((B[rows])[:,cols])
    print()