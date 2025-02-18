import numpy as np
import json
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import mpl_toolkits.mplot3d.axes3d as ax3
from matplotlib.animation import FuncAnimation
from numpy import sign, matmul as mm
from datetime import datetime
import control as co
from scipy.linalg import block_diag
from scipy.integrate import ode, odeint
from scipy.interpolate import interp1d, interpn
from scipy.optimize import curve_fit,minimize,minimize_scalar,newton
from scipy.io import savemat, loadmat
from scipy.signal import tf2zpk as scipy_tf2zpk
# from math import pi, sin, cos, tan, exp, asin, acos, atan, atan2
from numpy import pi, sin, cos, tan, exp, arcsin as asin, arccos as acos, arctan as atan, arctan2 as atan2
from std_atm import stdatm_english, stdatm_si
from quat import quat_mult, euler_2_quat, quat_2_euler, quat_norm, body_2_fixed, fixed_2_body, eulerdot_2_quatdot, quatdot_2_eulerdot
from linearization import linearization as lin,Anderson_correction_der_coeff,Anderson_correction_der_M

from controller_simulation import Aircraft,run_single_simulation, \
    monte_carlo_perturbations, report_latex, report_eigprops, rep2D,BIREAero, \
    GainSchedulingAircraft



if __name__ == "__main__":

    # filenames 
    base_file = "base_fs_in.json"
    bire_file = "bire_fs_in.json"

    # read in json to ensure no file changes while running
    base_dict = json.loads( open(base_file).read() )
    bire_dict = json.loads( open(bire_file).read() )

    plot_vars = {
        "show" : False,
        "plot_full" : True,
        "plot_delta" : True,
        "zoom_deltas" : True,
        # "zoom_fraction" : 0.05,
        "zoom_fraction" : 2./15.,
        "transparent" : False,
        "format" : "pdf"
    }

    # bire aero err dict
    bire_errs = { # 3-sig bounds written at end of line
        "CL" : {
            "0"  : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.1600}, #z+-0.4?
            "a"  : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.0500}, #z(+0.2,-0.15)
            "b"  : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "p"  : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "q"  : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "r"  : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "da" : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "de" : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  }
        },
        "CS" : {
            "0"  : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.0230},#z(+0.069,-0.097)
            "a"  : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "b"  : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "p"  : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "Lp" : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "q"  : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "r"  : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "da" : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "de" : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  }
        },
        "CD" : {
            "0"   : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "L"   : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "L2"  : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "S"   : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "S2"  : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "p"   : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "Sp"  : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "q"   : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "Lq"  : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "L2q" : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "r"   : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "Sr"  : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "da"  : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "Sda" : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "de"  : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "Lde" : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "de2" : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  }
        },
        "Cl" : {
            "0"  : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.0240}, #z(+0.073,-0.097)
            "a"  : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "b"  : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "p"  : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "q"  : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "r"  : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "Lr" : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "da" : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "de" : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  }
        },
        "Cm" : {
            "0"  : {"A":0.0600,"w":0.25  ,"phi":0.1500,"z":0.25  },#A(+0.2,-0.2),p(+0.5,-0.5)
            "a"  : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "b"  : {"A":0.25  ,"w":0.25  ,"phi":0.1000,"z":0.25  },
            "p"  : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "q"  : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "r"  : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "da" : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "de" : {"A":0.0333,"w":0.25  ,"phi":0.0667,"z":0.25  }
        },
        "Cn" : {
            "0"   : {"A":0.15  ,"w":0.15  ,"phi":0.0067,"z":0.0002},
            #(z<=-0.0067*p+0.00033)(z>=-0.04*p-0.002)
            "a"   : {"A":0.0333,"w":0.15  ,"phi":0.0033,"z":0.0025},
            #(z<=0.5*p+0.025)(z>=0.5*p-0.025)
            "b"   : {"A":0.0067,"w":0.15  ,"phi":0.0600,"z":0.0067}, #(z<=-0.8A),p(+0.2,-0.2)
            "p"   : {"A":0.15  ,"w":0.15  ,"phi":0.15  ,"z":0.15  },
            "Lp"  : {"A":0.15  ,"w":0.15  ,"phi":0.15  ,"z":0.15  },
            "q"   : {"A":0.15  ,"w":0.15  ,"phi":0.15  ,"z":0.15  },
            "r"   : {"A":0.15  ,"w":0.15  ,"phi":0.15  ,"z":0.15  },
            "da"  : {"A":0.15  ,"w":0.15  ,"phi":0.15  ,"z":0.15  },
            "Lda" : {"A":0.15  ,"w":0.15  ,"phi":0.15  ,"z":0.15  },
            "de"  : {"A":0.0800,"w":0.15  ,"phi":0.0300,"z":0.0333} #z(+0.18,-0.2)
            # linear relationship between errors in Cn,bA and Cn,bz.
            # Cn,bz ~= -1.0 * Cn,bA + 0.15
            # linear relationship bounds between errors in Cn,dep and Cn,dez
            # Cn,dep <= 0.6 * Cn,dez + 0.3
            # Cn,dep >= 0.5 * Cn,dez - 0.2
        }
    }
    # bire inertia
    bire_iner = {
        "Ixx" : {"A":0.25  ,"w":0.25  ,"p":0.25  ,"z":0.25  },
        "Iyy" : {"A":0.25  ,"w":0.25  ,"p":0.25  ,"z":0.25  },
        "Izz" : {"A":0.25  ,"w":0.25  ,"p":0.25  ,"z":0.25  },
        "Ixy" : {"A":0.25  ,"w":0.25  ,"p":0.25  ,"z":0.25  },
        "Ixz" : {"A":0.25  ,"w":0.25  ,"p":0.25  ,"z":0.25  },
        "Iyz" : {"A":0.25  ,"w":0.25  ,"p":0.25  ,"z":0.25  },
        "hx" : 0.25  ,
        "hy" : 0.25  ,
        "hz" : 0.25  ,
        "W" : 0.0667
    }
    # base make f16 aero err dict
    base_errs = {
        "CL" : {
            "0"  : 0.25 ,"a"  : 0.1  ,"q"  : 0.25 ,"de" : 0.25 # a -0.1,+?(all good)
        },
        "CS" : {
            "b"  : 0.25 ,"p"  : 0.25 ,"Lp" : 0.25 ,"r" : 0.25 ,
            "da" : 0.25 ,"dr" : 0.25 
        },
        "CD" : {
            "0"   : 0.25 ,"L"   : 0.25 ,"L2"  : 0.25 ,"S2"  : 0.25 ,
            "Sp"  : 0.25 ,"q"   : 0.25 ,"Lq"  : 0.25 ,"L2q" : 0.25 ,
            "Sr"  : 0.25 ,
            "Sda" : 0.25 ,"de"  : 0.25 ,"Lde" : 0.25 ,"de2" : 0.25 ,"Sdr" : 0.25
        },
        "Cl" : {
            "b" : 0.25 ,
            "p"  : 0.25 ,"r" : 0.25 ,"Lr" : 0.25 ,
            "da" : 0.25 ,"dr" : 0.25 
        },
        "Cm" : {
            "0"  : 0.25 ,"a"  : 0.25 ,"q"  : 0.25 ,"de" : 0.25 
        },
        "Cn" : {
            "b" : 0.25 ,
            "p"  : 0.25 ,"Lp"  : 0.25 ,"r"  : 0.25 ,
            "da" : 0.25 ,"Lda" : 0.25 ,"dr" : 0.25 
        }
    }
    # base inertia
    base_iner = {
        "Ixx" : 0.25 ,
        "Iyy" : 0.25 ,
        "Izz" : 0.25 ,
        "Ixy" : 0.25 ,
        "Ixz" : 0.25 ,
        "Iyz" : 0.25 ,
        "hx" : 0.25 ,
        "hy" : 0.25 ,
        "hz" : 0.25 ,
        "W" : 0.125 # +-0.125
    }
    # bire FM
    bire_FM_errs = [
        0.0700, # CL +0.50,-0.24 ## SCT
        0.25  , # CS
        0.1200, # CD +-0.4
        0.25  , # Cl
        0.25  , # Cm
        0.25   # Cn
    ]
    # base FM
    base_FM_errs = [
        0.0800, # CL +0.6,-0.25
        0.25  , # CS
        0.1200, # CD +-0.4
        0.25  , # Cl
        0.25  , # Cm
        0.25   # Cn
    ]
    
    flight_conditions = {
        "T1" : { "m" : 0.2 , "h" :  1000., "V" : 222., "Re" : 15641000. },
        "T2" : { "m" : 0.19, "h" : 15000., "V" : 201., "Re" :  9919000. },
        "C1" : { "m" : 0.8 , "h" :  1000., "V" : 890., "Re" : 62563000. },
        "C2" : { "m" : 0.6 , "h" : 15000., "V" : 634., "Re" : 31324000. },
        "C3" : { "m" : 0.8 , "h" : 30000., "V" : 796., "Re" : 25828000. }
    }
    f1 = "C2" # 
    f2 = "C3"
    state_threshold = [
        10., 15., 15.,
        20., 10., 10., # 
        1., 1., 50., # 
        25., 10., 1., # 
        5., 5., 5., 0.05
    ]

    run_base = {
        "actr_warm_start" : False,
        "num" : 1000,
        "final_time" : 15., # 120., # 
        "time_step" : 0.01,
        "initial_mach" : flight_conditions[f1]["m"]*1.,
        "initial_altitude" : flight_conditions[f1]["h"]*1.,
        "trim_bank" : 0.0, # 75.5224878, # 78.463041, # 80.4059318, # 60.0, # 
        "trim_climb" : 0.0,
        "start_climbing" : False,
        "end_gs_climbing" : False,
        "final_mach" : flight_conditions[f1]["m"]*1., # f2]["m"]*1., # 
        "final_altitude" : flight_conditions[f1]["h"]*1., # f2]["h"]*1., # 
        "t_gain_schedule" : 0.1, # 90., # 
        "gain_steps" : 30,
        "cut_mine" : True,
        "save_data" : True,
        "statistical" : True,
        "has_turbulence" : False, # True, # 
        "turbulence_setting" : "light", # "moderate", # "severe", # 
        "has_model_error" : False, # True, # 
        "aero_model_errors" : base_errs,
        "inertia_model_errors" : base_iner,
        "FM_errors" : base_FM_errs,
        "state_threshold" : state_threshold, # 64.0, # 
        "random_seed" : 13,
        "turbulence_random_seed" : 14, # 13, # 
        "error_random_seed" : 15, # 13, # 
        "rerandomize_turbulence" : True,
        "mrrr" : [6,7,11], # 0,1,2,8,9,10,
        # "mrrc" : [2,3], # [3], # [2], # 
        "get_aero_FM" : True,
        "include_stall_derivatives" : False, # True, # 
        "skip_simulation" : False, # True, # 
        "skip_video" : True, # False, # 
        "name_end" : "_" + f1 + "_BK_3"#4_wSd" # _1e1pqr" #+ "_" + name
        # 4 -- incr wt on tau, decr wt on da,de
        # 5 -- decr wt on da
    }
    run_bire = {**run_base}
    run_bire["aero_model_errors"] = bire_errs
    run_bire["inertia_model_errors"] = bire_iner
    run_bire["FM_errors"] = bire_FM_errs

    # run GS case
    # # # 
    plot_vars["plot_full"] = True # False # 
    plot_vars["plot_delta"] = False # True # 
    plot_vars["zoom_deltas"] = False
    plot_vars["plot_norm"] = True
    plot_vars["format"] = "pdf" # "png" # 
    # plot_vars["zoom_fraction"] = 1./15.
    # plot_vars["plot_input_limits_zoomed"] = False
    # # #
    di = [0.,0.,0.]
    t_gain = 90.0 # 5.0 # 
    adt = 0. # 
    scale = 1. # 
    offset = 15.0 # 
    # run_bire["aircraft_class"] = GainSchedulingAircraft
    run_bire["num"] = 1 # 1000 # 
    run_bire["final_time"] = (t_gain + adt)*scale + offset # 15. # 
    run_bire["trim_bank"] = 0.0
    run_bire["trim_climb"] = 0.0 # 0. # 
    run_bire["start_climbing"] = False # False # 
    run_bire["end_gs_climbing"] = False # True # 
    # run_bire["initial_mach"] = 1.5 
    run_bire["final_mach"] = flight_conditions[f2]["m"]*1. # flight_conditions[f1]["m"]*1. # 1.5 # 
    run_bire["final_altitude"] = flight_conditions[f2]["h"]*1. # flight_conditions[f1]["h"]*1. # 20000.0 # 
    run_bire["initial_bank"] = 0.0
    run_bire["final_bank"] = 0.0 # 75.0 # 60.0 # 30.0 # 
    run_bire["t_gain_schedule"] = t_gain + adt # 0. # 
    run_bire["gain_steps"] = 40 # 10 # 
    run_bire["trim_steps"] = 40 # 10 # 
    run_bire["interpolation_type"] = "linear" # "next" # "nearest-up" # 
    run_bire["has_turbulence"] = True # False # 
    run_bire["turbulence_setting"] = "light" # "moderate" # "severe" # 
    # run_bire["turbulence_random_seed"] = 25
    run_bire["has_model_error"] = True # False # 
    run_bire["fixed_FM_errors"] = [0.1,0.1,0.1,0.1,0.1,0.1]
    run_bire["skip_simulation"] = False # True # 
    run_bire["save_data"] = True # False # 
    # # # bire_dict["aircraft"]["CG_shift[ft]"] = [+1.0,+0.0,0.0]
    run_bire["name_end"] = "_" + f1 + "_GS" + "_dHdM" # "_dP" # 
    # run_bire["mrrr"] = [1,3,5,6,7,9,11] # [6,7,11] # 
    # run_bire["mrrc"] = [2] # None # 
    bire_dict["controller"]["LQR"] = {
        "note" : "_almost_current",
        "Q" : [1.0e-3, 1.0e-6, 2.0e-4, # 
            1.0e0, 1.0e0, 1.0e0,
            0.0, 0.0, 5.0e-6,
            1.0e0, 1.0e0, 0.0],
        "Q1a" : [0.0e0, 0.0e0, 0.0e0, 0.0e0],
        "Q2a" : [0.0, 0.0, 0.0, 0.0],
        "R" : [5.0e0, 5.0e0, 5.0e0, 5.0e-2]
    }
    run_bire["num"] = run_base["num"] = 1
    bire_dict["simulation"]["integrator"] = "rk4"
    #
    run_bire["include_stall_derivatives"] = True # False # 
    run_bire["include_altitude_derivatives"] = True # False # 
    #
    run_single_simulation(bire_dict,rtdst_1sg=di,**run_bire,**plot_vars)
    # run_single_simulation(base_dict,rtdst_1sg=di,**run_base,**plot_vars)
    quit()

    

