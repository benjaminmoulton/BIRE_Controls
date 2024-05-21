import numpy as np
import json
from matplotlib import pyplot as plt
from controller_simulation import Aircraft,monte_carlo_perturbations,run_single_simulation


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

    # #####
    # # # for running in unreal sim
    # di = [90.0,10.0,2.5]
    # # di = [90.0,5.0,2.5]
    # # plot_vars["format"] = "png"
    # bire_dict["initial"]["trim_guess"] = {}
    # bire_dict["initial"]["trim_guess"]["BIRE[deg]"] = 0.0
    # # bire_dict["aircraft"]["CG_shift[ft]"] = [1.0,0.0,0.0]
    # bire_dict["simulation"]["include_compressibility"] = True
    # bire_dict["simulation"]["use_Anderson_corrections"] = True
    # bire_dict["simulation"]["use_fitted_thrust_model"] = False
    # # run_bire["skip_simulation"] = True
    # # run_bire["save_data"] = False
    # run_bire["mrrc"] = [2]
    # run_bire["trim_bank"] = 30.0
    # if run_bire["trim_bank"] == 30.0:
    #     bire_dict["controller"]["LQR"] = {
    #         "note" : "_current_sct",
    #         "Q" : [1.0e-5, 1.0e-6, 5.0e-6, # ### hs
    #             1.5e-2, 1.0e+1, 2.0e+0, # 
    #             0.0e+0, 0.0e+0, 1.0e-6, # 
    #             2.0e-3, 5.0e-3, 0.0e+0], # 
    #         "Q1a" : [0.0e0, 0.0e0, 0.0e0, 0.0e0],
    #         "Q2a" : [0.0e0, 0.0e0, 0.0e0, 0.0e0],
    #         "R" : [1.0e+0, 1.0e+0, 1.0e+0, 1.0e+0] # 
    #     }
    #     run_base["name_end"] = run_bire["name_end"] = "_" + f1 + "_BK_hs"
    # run_bire["num"] = 1
    # run_bire.pop("initial_mach")
    # run_bire.pop("final_mach")
    # run_bire["initial_velocity"] = 634.0
    # run_single_simulation(bire_dict,rtdst_1sg=di,**run_bire,**plot_vars)
    # quit()
    # #####

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
    dB_max = 60.
    num = 2*int(dB_max)+1 # 5 # 
    bire.verbose_trim = False # True # 
    dB_guesses_deg = np.linspace(-dB_max,dB_max,num=num)
    dB_guesses = np.deg2rad(dB_guesses_deg)
    fig,ax1 = plt.subplots()
    ax2 = ax1.twinx()
    dBs = np.zeros((num,))
    des = np.zeros((num,))
    for i in range(num):
        print("running {:>03d}, dB = {:> 6.1f} deg".format(i+1,
            dB_guesses_deg[i]))
        u_guess[2] = dB_guesses[i]
        bire._initialize_state(u_guess=u_guess,no_report=True)
        des[i],dBs[i] = np.rad2deg(bire.u_trim[1:3])
    lns2 = ax2.plot(dB_guesses_deg,des,"b",label="de")
    lns1 = ax1.plot( dB_guesses_deg,dBs,"r",label="dB")
    lns1 = ax1.lines; lns2 = ax2.lines
    lns = lns1 + lns2
    labs = [lns1[0].get_label(),lns2[0].get_label()]
    ax1.legend(lns,labs)#loc=0)
    
    ax1.set_xlabel( "BIRE initial guess [deg]")
    ax1.set_ylabel( "BIRE trim solution [deg]")
    ax2.set_ylabel("elevator trim solution [deg]")
    plt.show()
    quit()

    ## code output, cg nominal
    ##
    # run     1, found new trim
    #  a_guess =   0.000000 deg,  b_guess =   0.000000 deg
    # da_guess =   0.000000 deg, de_guess =  25.000000 deg
    # dB_guess =  60.000000 deg,tau_guess =   0.000000
    # ================================ Trim Settings ================================
    #     "elevation[deg,rad]"    :      2.7415356075546513 :      0.0478488229124918
    #     "bank_angle[deg,rad]"   :     29.9999999999999964 :      0.5235987755982988
    #     "climb_angle[deg,rad]"  :     -0.4266463421095789 :     -0.0074463834114023
    #     "alpha[deg,rad]"        :      3.1681819496642305 :      0.0552952063238941
    #     "beta[deg,rad]"         :     -0.0057832719998144 :     -0.0001009371379352
    #     "M"                     :      0.6000000000000000
    #     "V[ft/s]"               :    634.4133153512273111
    #     "u[ft/s]"               :    633.4436808826867491
    #     "v[ft/s]"               :     -0.0640358642107716
    #     "w[ft/s]"               :     35.0621411801497231
    #     "p[deg/s,rad/s]"        :     -0.0800043056586731 :     -0.0013963385495158
    #     "q[deg/s,rad/s]"        :      0.8353731041767737 :      0.0145800111504903
    #     "r[deg/s,rad/s]"        :      1.4469086597107013 :      0.0252533200875700
    #     "H[ft]"                 :  15000.0000000000000000
    #     "aileron[deg,rad]"      :      0.0188033521590918 :      0.0003281804055881
    #     "elevator[deg,rad]"     :      0.0840930140173057 :      0.0014676999725277
    #     "BIRE[deg,rad]"         :      0.7374829932309876 :      0.0128715064093438
    #     "throttle"              :      0.2855005081171955
    #     thrust[lbf]             :   2170.2069562713363666                          
    #     "load factor"           :      1.1524963510499875
    #     "iterations"            :        0             
    # ===============================================================================

    # run     2, found new trim
    #  a_guess =   0.000000 deg,  b_guess =   0.000000 deg
    # da_guess =   0.000000 deg, de_guess = -25.000000 deg
    # dB_guess =  60.000000 deg,tau_guess =   0.000000
    # ================================ Trim Settings ================================
    #     "elevation[deg,rad]"    :      2.6830529048244629 :      0.0468281071943849
    #     "bank_angle[deg,rad]"   :     29.9999999999999964 :      0.5235987755982988
    #     "climb_angle[deg,rad]"  :     -0.4863054530294384 :     -0.0084876313257663
    #     "alpha[deg,rad]"        :      3.1693583578539011 :      0.0553157385201513
    #     "beta[deg,rad]"         :     -0.1248724015269748 :     -0.0021794345515181
    #     "M"                     :      0.6000000000000000
    #     "V[ft/s]"               :    634.4133153512273111
    #     "u[ft/s]"               :    633.4414596674114364
    #     "v[ft/s]"               :     -1.3826612048272098
    #     "w[ft/s]"               :     35.0750640391087387
    #     "p[deg/s,rad/s]"        :     -0.0783041992237089 :     -0.0013666660945913
    #     "q[deg/s,rad/s]"        :      0.8354699615635683 :      0.0145817016307947
    #     "r[deg/s,rad/s]"        :      1.4470764216257181 :      0.0252562480853465
    #     "H[ft]"                 :  15000.0000000000000000
    #     "aileron[deg,rad]"      :      0.0445593456907176 :      0.0007777072948374
    #     "elevator[deg,rad]"     :     -0.1527697276989047 :     -0.0026663347457211
    #     "BIRE[deg,rad]"         :     16.0139693175897726 :      0.2794964909052910
    #     "throttle"              :      0.2834787196199255
    #     thrust[lbf]             :   2142.9440590852886999                          
    #     "load factor"           :      1.1525144699031176
    #     "iterations"            :        0             
    # ===============================================================================

    # run     4, found new trim
    #  a_guess =   0.000000 deg,  b_guess =   0.000000 deg
    # da_guess =   0.000000 deg, de_guess = -25.000000 deg
    # dB_guess = -60.000000 deg,tau_guess =   0.000000
    # ================================ Trim Settings ================================
    #     "elevation[deg,rad]"    :      2.8132121502435199 :      0.0490998145788588
    #     "bank_angle[deg,rad]"   :     29.9999999999999964 :      0.5235987755982988
    #     "climb_angle[deg,rad]"  :     -0.3557227137013812 :     -0.0062085325782182
    #     "alpha[deg,rad]"        :      3.1689348639449011 :      0.0553083471570771
    #     "beta[deg,rad]"         :      0.1363811510137799 :      0.0023803001228501
    #     "M"                     :      0.5999999999999999
    #     "V[ft/s]"               :    634.4133153512271974
    #     "u[ft/s]"               :    633.4414288195611107
    #     "v[ft/s]"               :      1.5100926664795764
    #     "w[ft/s]"               :     35.0703659819406468
    #     "p[deg/s,rad/s]"        :     -0.0820880039056296 :     -0.0014327059445432
    #     "q[deg/s,rad/s]"        :      0.8352580178704383 :      0.0145780025155208
    #     "r[deg/s,rad/s]"        :      1.4467093243808731 :      0.0252498410297489
    #     "H[ft]"                 :  15000.0000000000000000
    #     "aileron[deg,rad]"      :     -0.0141230153291487 :     -0.0002464931178033
    #     "elevator[deg,rad]"     :     -0.1463000958880727 :     -0.0025534183692303
    #     "BIRE[deg,rad]"         :    -17.5224909740564101 :     -0.3058251606482726
    #     "throttle"              :      0.2833560810358977
    #     thrust[lbf]             :   2141.2903336278982351                          
    #     "load factor"           :      1.1524733542717376
    #     "iterations"            :        0             
    # ===============================================================================
