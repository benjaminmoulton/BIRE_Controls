import numpy as np
import json
from matplotlib import pyplot as plt
from controller_simulation import Aircraft,monte_carlo_perturbations,run_single_simulation

from scipy.optimize import minimize

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
    bire_errs = {"nill":0.0}
    # bire inertia
    bire_iner = {"nill":0.0}
    # base make f16 aero err dict
    base_errs = {"nill":0.0}
    # base inertia
    base_iner = {"nill":0.0}
    # bire FM
    bire_FM_errs = [0.0]*6
    # base FM
    base_FM_errs = [0.0]*6

    # acceptable threshold values based on intensity

    flight_conditions = {
        "T1" : { "m" : 0.2 , "h" :  1000., "V" : 222., "Re" : 15641000. },
        "T2" : { "m" : 0.19, "h" : 15000., "V" : 201., "Re" :  9919000. },
        "C1" : { "m" : 0.8 , "h" :  1000., "V" : 890., "Re" : 62563000. },
        "C2" : { "m" : 0.6 , "h" : 15000., "V" : 634., "Re" : 31324000. },
        "C3" : { "m" : 0.8 , "h" : 30000., "V" : 796., "Re" : 25828000. }
    }
    f1 = "C2"
    f2 = "C2"
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
        # "time_step" : 0.01,
        "initial_mach" : flight_conditions[f1]["m"]*1.,
        "initial_altitude" : flight_conditions[f1]["h"]*1.,
        "trim_bank" : 30.0, # 75.5224878, # 78.463041, # 80.4059318, # 60.0, # 
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
        "rerandomize_turbulence" : True,
        "mrrr" : [6,7,11], # 0,1,2,8,9,10,
        # "mrrc" : [2,3], # [3], # [2], # 
        "get_aero_FM" : True,
        "include_stall_derivatives" : False, # True, # 
        "skip_simulation" : False, # True, # 
        "name_end" : "_" + f1 + "_BK_3"#4_wSd" # _1e1pqr" #+ "_" + name
        # 4 -- incr wt on tau, decr wt on da,de
        # 5 -- decr wt on da
    }
    run_bire = {**run_base}
    run_bire["aero_model_errors"] = bire_errs
    run_bire["inertia_model_errors"] = bire_iner
    run_bire["FM_errors"] = bire_FM_errs


    bire_dict["controller"]["LQR"] = {
        # "note" : "_current",
        # "Q" : [1.0e-6, 1.0e-6, 1.0e-6, # ### BK_3
        #     1.0e0, 1.0e0, 1.0e0,
        #     0.0, 0.0, 1.0e-6, 
        #     1.0e0, 1.0e0, 0.0],
        # "Q1a" : [0.0e0, 0.0e0, 0.0e0, 0.0e0],
        # "Q2a" : [0.0, 0.0, 0.0, 0.0],
        # "R" : [5.0e0, 5.0e0, 5.0e0, 5.0e-2]
        # # #
        "note" : "_current_sct",
        "Q" : [1.0e-5, 1.0e-6, 5.0e-6, # ### hs
               1.5e-2, 1.0e+1, 2.0e+0, # 
               0.0e+0, 0.0e+0, 1.0e-6, # 
               2.0e-3, 5.0e-3, 0.0e+0], # 
        "Q1a" : [0.0e0, 0.0e0, 0.0e0, 0.0e0],
        "Q2a" : [0.0e0, 0.0e0, 0.0e0, 0.0e0],
        "R" : [1.0e+0, 1.0e+0, 1.0e+0, 1.0e+0] # 
        # # #
        # "note" : "_current_scta",
        # "Q" : [2.0e-5, 5.0e-7, 5.0e-6, # ### acv
        #        55.0e-3, 1.0e+1, 1.0e+0, # 
        #        0.0e+0, 0.0e+0, 2.0e-7, # 
        #        1.0e-3, 5.0e-3, 0.0e+0], # 
        # "Q1a" : [0.0e0, 0.0e0, 0.0e0, 0.0e0],
        # "Q2a" : [0.0e0, 0.0e0, 0.0e0, 0.0e0],
        # "R" : [1.0e+0, 5.0e+0, 2.0e+0, 1.0e+0] # 
        # # #
    }
    base_dict["controller"]["LQR"] = {**bire_dict["controller"]["LQR"]}
    base_dict["initial"]["mach"] = \
        bire_dict["initial"]["mach"] = flight_conditions[f1]["m"]*1.
    base_dict["initial"]["altitude[ft]"] = \
        bire_dict["initial"]["altitude[ft]"] = flight_conditions[f1]["h"]*1.
    # run_bire["FM_errors"][0] = 0.03
    # run_bire["FM_errors"][2] = 0.1
    run_base["name_end"] = run_bire["name_end"] = "_" + f1 + "_BK_adh"
    run_base["has_turbulence"] = run_bire["has_turbulence"] = False # True # 
    run_base["has_model_error"] = run_bire["has_model_error"] = False # True #

    # # test trim with cg forward
    # base_dict["aircraft"]["CG_shift[ft]"] = \
    #     bire_dict["aircraft"]["CG_shift[ft]"] = [1.0,0.0,0.0]
    base_dict["initial"]["trim_guess"] = \
        bire_dict["initial"]["trim_guess"] = {}
    base_dict["initial"]["trim_guess"]["rudder[deg]"] = \
        bire_dict["initial"]["trim_guess"]["BIRE[deg]"] = 0.0
    banks = [0.0, 15.0, 30.0, 45.0, 60.0]
    run_banks = True
    if run_banks:
        base_dict["initial"]["trim"]["bank_angle[deg]"] = \
            bire_dict["initial"]["trim"]["bank_angle[deg]"] = 0.0
    else:
        base_dict["initial"]["trim"].pop("bank_angle[deg]")
        bire_dict["initial"]["trim"].pop("bank_angle[deg]")
        base_dict["initial"]["trim"]["sideslip_angle[deg]"] = \
            bire_dict["initial"]["trim"]["sideslip_angle[deg]"] = 15.0
        # MAX crosswind is about 12 deg beta at V = 222 ft/s (max crosswind is 45.5709 ft/s)
    base_dict["initial"]["trim"]["type"] = \
        bire_dict["initial"]["trim"]["type"] = "spu" # "shss" # "sct" # 
    bire_dict["initial"]["trim"]["pitch_rate[deg/s]"] = 10.0
    bire = Aircraft(bire_dict) # base_dict) # 
    # bire.run_trim()

    # numerical trim
    bire.use_quaternions = False
    x0 = np.zeros((16,))
    V = flight_conditions[f1]["V"]*1.
    x0[0] = V
    qrad = np.deg2rad(bire_dict["initial"]["trim"]["pitch_rate[deg/s]"])
    x0[4] = qrad
    zf = -flight_conditions[f1]["h"]*1.
    x0[8] = zf
    x0[x0 == 0.0] = 0.1
    dyn = lambda x : np.linalg.norm(bire._nonlinear_euler_dynamics(0.0,x,
        is_controlled=True,given_control=True,u=x[12:17],
        force_control_to_inputs=False))
    bounds = ((V-100.,V+100.),(-100.0,+100.0),(-200.0,+200.0),
              (-0.0,+0.0),(+qrad,+qrad),(-0.0,+0.0),
              (-0.0,+0.0),(-0.0,+0.0),(+zf,+zf),
              (-0.0,+0.0),(+qrad,+qrad),(-0.0,+0.0),
              (bire.min_da,bire.max_da),(bire.min_de,bire.max_de),
              (bire.min_dr,bire.max_dr),(bire.min_tau,bire.max_tau))
    res = minimize(dyn,x0,bounds=bounds)
    print(res.x)
    # quit()

    print()
    bire.use_quaternions = True
    u_guess = np.zeros((4,))
    u_guess[2] = np.deg2rad(+0.0)
    bire.verbose_trim = True # False # 
    bire.trim_iter = 0
    bire._initialize_state(u_guess=u_guess)#,no_report=False)
    bire._report_trim_solution()


