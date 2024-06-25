import numpy as np
import json
from matplotlib import pyplot as plt
from controller_simulation import Aircraft#,monte_carlo_perturbations,run_single_simulation


if __name__ == "__main__":

    # filenames 
    base_file = "base_fs_in.json"
    bire_file = "bire_fs_in.json"

    # read in json to ensure no file changes while running
    base_dict = json.loads( open(base_file).read() )
    bire_dict = json.loads( open(bire_file).read() )
    
    # flight conditions
    flight_conditions = {
        "T1" : { "m" : 0.2 , "h" :  1000., "V" : 222., "Re" : 15641000. },
        "T2" : { "m" : 0.19, "h" : 15000., "V" : 201., "Re" :  9919000. },
        "C1" : { "m" : 0.8 , "h" :  1000., "V" : 890., "Re" : 62563000. },
        "C2" : { "m" : 0.6 , "h" : 15000., "V" : 634., "Re" : 31324000. },
        "C3" : { "m" : 0.8 , "h" : 30000., "V" : 796., "Re" : 25828000. }
    }

    # settings
    run_bire = True # False # 
    run_sct = True # False # 
    run_fs = True
    skip_run = False # True # 
    if run_sct: trim_bank_deg = 80.0
    else: trim_beta_deg = 0.0
    fc = "C2" # "T1"
    mfc = flight_conditions[fc]["m"] # 0.2 # 
    hfc = flight_conditions[fc]["h"] # 1000.0 # 
    cgshift = [0.0,0.0,0.0] # [1.0,0.0,0.0] # 
    #
    run_num = 1000
    a_scale = 20.0
    if run_sct: p_scale = 180.0
    else: b_scale = 20.0
    u_scale = np.array([20.0,20.0,70.0]) # 30.0]) # 

    # set up run
    craftdict = bire_dict if run_bire else base_dict
    trim_type = "sct" if run_sct  else "shss"
    # name
    run_name  = "bire" if run_bire else "base"
    run_name += "_fs" if run_fs else "_rc"
    run_name += "_" + trim_type
    run_name += "_" + fc
    run_name += "_M{:2.1f}".format(mfc).replace(".","") + \
        "_H{:04.1f}".format(hfc/1000.).replace(".","") 
    run_name += "_CG{:+03d}".format(int(cgshift[0]*10.)).replace("+","p").replace("-","m")
    run_name += "{:+03d}".format(int(cgshift[1]*10.)).replace("+","p").replace("-","m")
    run_name += "{:+03d}".format(int(cgshift[2]*10.)).replace("+","p").replace("-","m")
    run_name += "_P{:02d}".format(int(trim_bank_deg)) if run_sct else \
        "_B{:02d}".format(int(trim_beta_deg))
    run_file = run_name + ".json"
    print("\n\nrunning", run_name,"...")
    
    # open this file if it exists
    folder = "trim_files/"
    try: 
        run_dict = json.loads( open(folder + run_file).read() )
        print("success")
    except: run_dict = {}
    
    # pull in saved trim states
    x_trims=[]; u_trims=[]; CFM_trims=[]; guess_trims=[]; final_i_trims=[]
    for case in run_dict:
        x_trims.append(run_dict[case]["x_trim_euler"])
        u_trims.append(run_dict[case]["u_trim"])
        CFM_trims.append(run_dict[case]["CFM_trim"])
        guess_trims.append(run_dict[case]["guess_trim"])
        final_i_trims.append(run_dict[case]["final_i_trim"])
    
    # initialize aircraft
    # set to initial params
    craftdict["initial"] = craftdict.get("initial",{})
    craftdict["initial"]["mach"] = mfc
    craftdict["initial"]["altitude"] = hfc
    craftdict["initial"].pop("airspeed[ft/s]",None)
    craftdict["initial"]["type"] = "trim"
    craftdict["initial"]["trim"] = craftdict["initial"].get("trim",{})
    craftdict["initial"]["trim"]["type"] = trim_type
    craftdict["initial"]["trim"].pop("elevation_angle[deg]",None)
    craftdict["initial"]["trim"]["climb_angle[deg]"] = 0.0
    if run_sct:
        craftdict["initial"]["trim"]["bank_angle[deg]"] = trim_bank_deg
        craftdict["initial"]["trim"].pop("sideslip_angle[deg]",None)
    else:
        craftdict["initial"]["trim"]["sideslip_angle[deg]"] = trim_beta_deg
        craftdict["initial"]["trim"].pop("bank_angle[deg]",None)
    # initialize
    craft = Aircraft(craftdict)

    # run trims if desired
    if not(skip_run):
        print("\nseeking trims...")
    
        # randomize initial guesses
        if run_sct:
            statement="randomizing, Dxcg = {:> 4.1f} ft, phi = {:> 5.1f} deg,".\
                format(cgshift[0],trim_bank_deg)+\
                " is BIRE = "+str(run_bire)+" ..."
        else:
            statement="randomizing, Dxcg = {:> 4.1f} ft, beta = {:> 5.1f} deg,".\
                format(cgshift[0],trim_beta_deg)+\
                " is BIRE = "+str(run_bire)+" ..."
        print(statement)
        ags = np.deg2rad((np.random.random(size=(run_num,))*2. - 1.)*a_scale)
        if run_sct:
            pgs = np.deg2rad((np.random.random(size=(run_num,))*2. - 1.)*p_scale)
        else:
            bgs = np.deg2rad((np.random.random(size=(run_num,))*2. - 1.)*b_scale)
        ugs = np.random.random(size=(run_num,4))#*0.
        ugs[:,0:3] = np.deg2rad((ugs[:,0:3]*2. - 1.)*u_scale)
        #
        tol = craft.NR_tol#*1.0e+1
        # run
        for i in range(run_num):
            # run trim
            if run_sct:
                guess_dict = dict(a_guess=ags[i],phi_guess=pgs[i],
                    u_guess=ugs[i].tolist())
            else:
                guess_dict = dict(a_guess=ags[i],b_guess=bgs[i],
                    u_guess=ugs[i].tolist())
            craft._initialize_state(no_report=True,**guess_dict)
            
            # check if we have found this before
            have_found_before = False
            for j in range(len(x_trims)):
                if np.linalg.norm(x_trims[j] - craft.x_trim_euler[0:12]) < tol\
                    and np.linalg.norm(u_trims[j] - craft.u_trim[0:4]) < tol:
                    have_found_before = True
            
            if not(craft.trim_failed) and not(have_found_before):
                x_trims.append(craft.x_trim_euler[0:12].tolist())
                u_trims.append(craft.u_trim[0:4].tolist())
                # CFM # #
                x = craft.x_trim_euler*1.0
                u = craft.u_trim*1.0
                # aero angles
                a = np.arctan2(x[2],x[0])
                V = (x[0] * x[0] + x[1] * x[1] + x[2] * x[2])**0.5
                b = np.arcsin(x[1]/V)
                sos = craft.stdatm(-x[8])[5]
                M = V/sos
                # nondimensionalize rates
                pbar = (x[3])*craft.bw/2./V
                qbar = (x[4])*craft.cw/2./V
                rbar = (x[5])*craft.bw/2./V
                # pass in controls state
                ail = u[0]; ele = u[1]; rud = u[2]; thr = u[3]
                # use aircraft model
                CFM_trims.append(craft.aero_model.aero_results(*[
                    a,b,pbar,qbar,rbar,ail,ele,rud,
                    craft.is_compressible,M,craft.use_anderson,craft.has_stall
                ]))
                # # # # #
                guess_trims.append(guess_dict)
                final_i_trims.append(craft.trim_iter+0)
                if run_sct: Vgs = pgs
                else: Vgs = bgs
                print("run {:> 5d}, {:> 7.2f}, {:> 5d}".\
                    format(i+1,Vgs[i],len(x_trims))+" new, *** found new trim")
            else:
                print("run {:> 5d}, {:> 5d} new".format(i+1,len(x_trims)))

        # save to file
        print("\nsaving to file",run_name,"...")
        # create datadict
        data_dict = {}
        for j in range(len(x_trims)):
            case = str(j)
            data_dict[case] = {}
            data_dict[case]["x_trim_euler"] = x_trims[j]
            data_dict[case]["u_trim"] = u_trims[j]
            data_dict[case]["CFM_trim"] = CFM_trims[j]
            data_dict[case]["guess_trim"] = guess_trims[j]
            data_dict[case]["final_i_trim"] = final_i_trims[j]
        # save
        with open(folder+run_file, "w") as f:
            json.dump(data_dict, f, indent=4)

    quit()

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
    f1 = "T1"
    f2 = "T1"
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
    run_banks = False
    if run_banks:
        base_dict["initial"]["trim"]["bank_angle[deg]"] = \
            bire_dict["initial"]["trim"]["bank_angle[deg]"] = 30.0
    else:
        base_dict["initial"]["trim"].pop("bank_angle[deg]")
        bire_dict["initial"]["trim"].pop("bank_angle[deg]")
        base_dict["initial"]["trim"]["sideslip_angle[deg]"] = \
            bire_dict["initial"]["trim"]["sideslip_angle[deg]"] = 15.0
        # MAX crosswind is about 12 deg beta at V = 222 ft/s (max crosswind is 45.5709 ft/s)
    base_dict["initial"]["trim"]["type"] = \
        bire_dict["initial"]["trim"]["type"] = "shss" # "sct" # 
    bire = Aircraft(bire_dict) # base_dict) # 
    # bire.run_trim()
    x0 = bire.x_trim*1.
    u0 = bire.u_trim*1.
    i0 = bire.trim_iter + 0
    # bire._report_trim_solution(x0,u0,i0)
    print()
    # print("running with initial guess of +45 deg")
    # u_guess = np.zeros((4,))
    # u_guess[2] = np.deg2rad(+45.0)
    # # bire.verbose_trim = True # False # 
    # bire._initialize_state(u_guess=u_guess)#,no_report=False)
    # # bire._report_trim_solution(bire.x_trim,bire.u_trim,bire.trim_iter)
    # # bire._report_trim_solution(x0,u0,i0)

    # randomize initial guesses
    if run_banks:
        statement = "randomizing, Dxcg = {:> 4.1f} ft, phi = {:> 5.1f} deg,".\
            format(bire.cgshift[0], np.rad2deg(bire.phi_trim))\
            +" is BIRE = "+str(bire.is_BIRE)+" ..."
    else:
        statement = "randomizing, Dxcg = {:> 4.1f} ft, beta = {:> 5.1f} deg,".\
            format(bire.cgshift[0], np.rad2deg(bire.beta_trim)\
            )+" is BIRE = "+str(bire.is_BIRE)+" ..."
    print(statement)
    num = 200
    a_scale = 20.0
    ags = np.deg2rad((np.random.random(size=(num,))*2. - 1.)*a_scale)
    if run_banks:
        b_scale = 20.0
        bgs = np.deg2rad((np.random.random(size=(num,))*2. - 1.)*b_scale)
    else:
        phi_scale = 180.0
        pgs = np.deg2rad((np.random.random(size=(num,))*2. - 1.)*phi_scale)
    u_scale = np.array([20.0,20.0,70.0]) # 30.0]) # 
    ugs = np.random.random(size=(num,4))#*0.
    ugs[:,0:3] = np.deg2rad((ugs[:,0:3]*2. - 1.)*u_scale)
    # #
    # num = 4
    # ags = np.zeros((num,))
    # bgs = np.zeros((num,))
    # ugs = np.zeros((num,4))
    # de = 25.0; dB = 45.0
    # ugs[:,1] = np.deg2rad([ de,-de, de,-de])
    # ugs[:,2] = np.deg2rad([ dB, dB,-dB,-dB])
    # # # for cg = 1, phi = 75
    # # de = 25.0; dB = 45.0
    # # ugs[:,1] = np.deg2rad([ 5.0,-de, 5.0,-de])
    # # ugs[:,2] = np.deg2rad([ 5.0, dB,-dB,-dB])
    # # # 
    x_trims = []; u_trims=[]; i_trims = []; final_i_trims = []
    tol = bire.NR_tol#*1.0e+1
    # run
    for i in range(num):
        # run trim
        if run_banks:
            bire._initialize_state(a_guess=ags[i],b_guess=bgs[i],
                u_guess=ugs[i],no_report=True)
        else:
            bire._initialize_state(a_guess=ags[i],phi_guess=pgs[i],
                u_guess=ugs[i],no_report=True)
        # check if we have found this before
        have_found_before = False
        for j in range(len(x_trims)):
            # print(x_trims[j])
            # print(bire.x_trim)
            # print(u_trims[j])
            # print(bire.u_trim)
            # print(x_trims[j] - bire.x_trim)
            # print(u_trims[j] - bire.u_trim[0:4])
            # print(np.linalg.norm(x_trims[j] - bire.x_trim))
            # print(np.linalg.norm(u_trims[j] - bire.u_trim[0:4]))
            # print()
            # if np.max(np.abs(x_trims[j] - bire.x_trim)) < tol and \
            #     np.max(np.abs(u_trims[j] - bire.u_trim[0:4])) < tol:
            if np.linalg.norm(x_trims[j] - bire.x_trim) < tol and \
                np.linalg.norm(u_trims[j] - bire.u_trim[0:4]) < tol:
                have_found_before = True
        
        if not(bire.trim_failed) and not(have_found_before):
            x_trims.append(bire.x_trim*1.)
            u_trims.append(bire.u_trim[0:4]*1.)
            i_trims.append(i+0)
            final_i_trims.append(bire.trim_iter+0)
            if run_banks:
                Vgs = bgs
            else:
                Vgs = pgs
            print("run {:> 5d}, {:> 7.2f}, {:> 5d} new, *** found new trim".format(i+1,Vgs[i],len(x_trims)))

            # # # run simulation
            # try:
            #     del bire.Lin_Model
            # except:
            #     True
            # bire.run_simulation(report_trim=False,save_matrices=False,
            #    report_simulation_deltas=False)#True)
        else:
            print("run {:> 5d}, {:> 5d} new".format(i+1,len(x_trims)))
    
    print()
    print()
    print(statement)
    print()
    for i in range(len(i_trims)):
        print("run {:> 5d}, found new trim".format(i_trims[i]+1))
        if run_banks:
            print(" a_guess = {:> 10.6f} deg,  b_guess = {:> 10.6f} deg".format(\
                np.rad2deg(ags[i_trims[i]]),np.rad2deg(bgs[i_trims[i]])))
        else:
            print(" a_guess = {:> 10.6f} deg,  p_guess = {:> 10.6f} deg".format(\
                np.rad2deg(ags[i_trims[i]]),np.rad2deg(pgs[i_trims[i]])))
        print("da_guess = {:> 10.6f} deg, de_guess = {:> 10.6f} deg".format(\
            np.rad2deg(ugs[i_trims[i],0]),np.rad2deg(ugs[i_trims[i],1])))
        print("dB_guess = {:> 10.6f} deg,tau_guess = {:> 10.6f}".format(\
            np.rad2deg(ugs[i_trims[i],2]),ugs[i_trims[i],3]))
        bire._report_trim_solution(x_trims[i],u_trims[i],final_i_trims[i])
        print()
    print()
    print()

    quit()