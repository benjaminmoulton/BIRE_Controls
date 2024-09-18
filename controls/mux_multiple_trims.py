import machupX as mux
import numpy as np
from controller_simulation import Aircraft
from math import atan2,asin,sin,cos
from os import mkdir, rmdir, walk, remove, listdir
from scipy.optimize import minimize as minmiz
import json

if __name__ == "__main__":
    # filenames 
    base_file = "base_fs_in.json"
    bire_file = "bire_fs_in.json"

    # read in json to ensure no file changes while running
    base_dict = json.loads( open(base_file).read() )
    bire_dict = json.loads( open(bire_file).read() )
    
    # flight conditions
    u1M = 0.331014489952403
    flight_conditions = {
        "A1" : { "m" : 0.2 , "h" :     0., "V" : 222., "Re" :        0. },
        "T1" : { "m" : 0.2 , "h" :  1000., "V" : 222., "Re" : 15641000. },
        "T2" : { "m" : 0.19, "h" : 15000., "V" : 201., "Re" :  9919000. },
        "C1" : { "m" : 0.8 , "h" :  1000., "V" : 890., "Re" : 62563000. },
        "C2" : { "m" : 0.6 , "h" : 15000., "V" : 634., "Re" : 31324000. },
        "C3" : { "m" : 0.8 , "h" : 30000., "V" : 796., "Re" : 25828000. },
        "U1" : { "m" : u1M , "h" : 15000., "V" : 350., "Re" : "unkn"    }, # no compr no stall
        "F1" : { "m" : 0.6 , "h" : 15000., "V" : 634., "Re" : 31324000. }, # no compr no stall
    }

    # settings 
    ## Continue from bire_fs_shss_T1_M02_H010_CGp10p00p00_B14
    run_bire = True # False # 
    run_sct  = True # False # 
    run_fs = True
    skip_run = False # True # 
    skip_DOC = True # False # 
    if run_sct: trim_bank_degs = np.linspace(0.0,60.0,num=13).tolist() # [0.0] # np.linspace(0.0,75.0,num=16).tolist() # [10.0] # [60.0] # np.linspace(0.0,75.0,num=16).tolist() # 
    else: trim_beta_degs = [6.0] # np.linspace(0.0,16.0,num=9).tolist() # np.linspace(0.0,16.0,num=9).tolist() # [14.0,16.0] # [0.0] # 
    fc = "F1" # "U1" # "C2" # "T1" # "A1" # 
    cgshift = [0.0,0.0,0.0] # [1.0,0.0,0.0] # [0.5,0.0,0.0] # [0.5,0.0,0.0] # 
    include_compressibility =  False # True # 
    use_Anderson_corrections =  True # False # 
    include_stall =  False # True # 
    plotting_xcgs = [0.0,0.5,1.0]
    plot_inverted_trims = False # True # 
    plot_alternate_trims = True # False # 
    show_plots = False # True # 
    plot_format = "pdf" # "png" # 
    plot_transparent = True if plot_format == "pdf" else False # False # True # 
    plot_dark = True # False # 
    #
    # other settings
    run_num = 30 # 1000 # 
    trim_iter = 1000 # 1000
    mfc = flight_conditions[fc]["m"] # 0.2 # 
    hfc = flight_conditions[fc]["h"] # 1000.0 # 
    a_scale = 20.0 # 0.02 # 0.0 # 
    b_scale = 20.0 # 0.2 # 0.0 # 
    p_scale = 180.0 # 0.0 # 
    u_scale = np.array([20.0,20.0,70.0,1.0]) # np.array([0.1,0.2,20.0,0.02]) # np.array([0.0]*4) # 
    a_shift = 0.0 # 3.17 # 
    b_shift = 0.0 # 0.0 # 
    p_shift = 0.0 # 0.0 # 
    u_shift = np.array([0.0,0.0,0.0,0.0]) # np.array([0.0,-0.05,0.0,0.276]) # 
    # set up run
    # craftdict = bire_dict if run_bire else base_dict
    trim_type = "sct" if run_sct else "shss"
    scale_type = "fs" if run_fs  else "rc"
    folder = "trim_files/" + fc + "_" + trim_type + "/"

    # initialize MUX model -- read in JSON
    input_file = "../aerodynamics_model/BIRE Inputs/BIRE_input.json"
    input_dict = json.loads( open(input_file).read() )
    craft_file = "../aerodynamics_model/BIRE Inputs/BIRE_airplane.json"
    craft_dict = json.loads( open(craft_file).read() )

    # make BIRE sim
    bire_dict["actuators"]["order"] = 0
    bire_dict["simulation"]["use_quaternions"] = False
    bire_dict["simulation"]["include_compressibility"] = False
    bire_dict["simulation"]["include_stall"] = False
    bire = Aircraft(bire_dict)
    bire.aero_model._CL = lambda alpha, beta, pbar, qbar, rbar, da, de, dB : bire.aero_model.CFM[0]
    bire.aero_model._CS = lambda alpha, beta, pbar, qbar, rbar, da, de, dB : bire.aero_model.CFM[1]
    bire.aero_model._CD = lambda alpha, beta, pbar, qbar, rbar, da, de, dB : bire.aero_model.CFM[2]
    bire.aero_model._Cl = lambda alpha, beta, pbar, qbar, rbar, da, de, dB : bire.aero_model.CFM[3]
    bire.aero_model._Cm = lambda alpha, beta, pbar, qbar, rbar, da, de, dB : bire.aero_model.CFM[4]
    bire.aero_model._Cn = lambda alpha, beta, pbar, qbar, rbar, da, de, dB : bire.aero_model.CFM[5]

    # optimizer runs
    for filename in listdir(folder):
        file_split = filename.replace(".json","").split("_")
        if file_split[1:4] == [scale_type, trim_type, fc]:
            # open file
            cases_dict = json.loads( open(folder + filename).read() )
            for case in cases_dict:
                # pull out state
                x_trim = np.array(cases_dict[case]["x_trim_euler"])
                u_trim = np.array(cases_dict[case]["u_trim"])
                # determine fixed states
                V_trim = (x_trim[0]**2. + x_trim[1]**2. + x_trim[2]**2.)**0.5
                a = atan2(x_trim[2],x_trim[0])
                b = asin(x_trim[1]/V_trim)
                p = x_trim[3]
                q = x_trim[4]
                r = x_trim[5]
                zf_trim = x_trim[8]
                phi_trim = x_trim[9]
                theta = x_trim[10]

                x0 = np.concatenate(([a,b,p,q,r,theta],u_trim))

                def CFM_from_mux(a,b,pbar,qbar,rbar,da,de,dB):
                    # print("running case...")
                    # pull out rates
                    p = pbar/bire.bw*2.*V_trim
                    q = qbar/bire.cw*2.*V_trim
                    r = rbar/bire.bw*2.*V_trim

                    # setup mux file
                    craft_dict["wings"]["BIRE_left" ]["dihedral"] = np.rad2deg( dB)
                    craft_dict["wings"]["BIRE_right"]["dihedral"] = np.rad2deg(-dB)
                    craft_dict["airfoils"]["NACA_64A204"]["geometry"]["outline_points"] = \
                        "../aerodynamics_model/BIRE Inputs/64A204.txt"
                    input_dict["scene"]["aircraft"]["BIRE"]["file"] = craft_dict
                    bire_mux = mux.Scene(input_dict)
                    # bire_mux._airplanes["BIRE"]\
                    #     .wing_segments["BIRE_right_right"].get_dihedral = \
                    #     lambda span : -dB*(span*0.0 + 1.) \
                    #     if isinstance(span,np.ndarray) else -dB
                    # bire_mux._airplanes["BIRE"]\
                    #     .wing_segments["BIRE_left_left"].get_dihedral = \
                    #     lambda span :  dB*(span*0.0 + 1.) \
                    #     if isinstance(span,np.ndarray) else  dB
                    # hstab_right = bire_mux._airplanes["BIRE"]\
                    #     .wing_segments["BIRE_right_right"]
                    # hstab_right._setup_cp_data()
                    # hstab_right._setup_node_data()
                    # hstab__left = bire_mux._airplanes["BIRE"]\
                    #     .wing_segments["BIRE_left_left"]
                    # hstab__left._setup_cp_data()
                    # hstab__left._setup_node_data()
                    # bire_mux._airplanes["BIRE"]._calculate_geometry()

                    # update state and solve for forces
                    bire_mux.set_aircraft_state(state={
                        "velocity":V_trim,
                        "alpha":np.rad2deg(a),
                        "beta":np.rad2deg(b),
                        "orientation":[np.rad2deg(phi_trim),np.rad2deg(theta),0.0],
                        "angular_rates":[p,q,r]
                    })
                    bire_mux.set_aircraft_control_state(control_state={
                        "aileron":np.rad2deg(da),
                        "elevator":np.rad2deg(de)
                    })
                    CFM_dict = bire_mux.solve_forces(non_dimensional=True,
                        dimensional=False,report_by_segment=False,body_frame=True,
                        stab_frame=False,wind_frame=True)["BIRE"]["total"]
                    CFM = np.array([CFM_dict["CL"],CFM_dict["CS"],CFM_dict["CD"],
                                    CFM_dict["Cl"],CFM_dict["Cm"],CFM_dict["Cn"]])
                    
                    return CFM

                def fun(x1):
                    # split apart x1
                    a,b,p,q,r,theta,da,de,dB,tau = x1
                    #
                    Vxb = V_trim*cos(a)*cos(b)
                    Vyb = V_trim*sin(b)
                    Vzb = V_trim*sin(a)*cos(b)
                    x_trim = np.array([Vxb,Vyb,Vzb,p,q,r,0.0,0.0,zf_trim,phi_trim,theta,0.0])
                    u_trim = np.array([da,de,dB,tau])
                    # get dimensionless rates
                    pbar = p*bire.bw/2./V_trim
                    qbar = q*bire.cw/2./V_trim
                    rbar = r*bire.bw/2./V_trim

                    # get CFM from mux
                    CFM = CFM_from_mux(a,b,pbar,qbar,rbar,da,de,dB)

                    # determine aerodynamics
                    bire.aero_model.CFM = CFM
                    xdot = bire._nonlinear_euler_dynamics(0.0,x_trim,True,True,u_trim,True)
                    xdot = xdot.tolist()
                    xdot = np.array(xdot[0:6] + xdot[8:11])
                    J = np.linalg.norm(xdot)
                    print(J)

                    return J

                # # run optimizer
                # print("starting optimizer...")
                # sol = minmiz(fun,x0)
                # print(sol)

                # CFM from mux for this condition
                # get dimensionless rates
                pbar = p*bire.bw/2./V_trim
                qbar = q*bire.cw/2./V_trim
                rbar = r*bire.bw/2./V_trim
                da,de,dB,tau = u_trim
                # CFM = CFM_from_mux(a,b,pbar,qbar,rbar,da,de,dB)
                # print(CFM)
                print("x_trim =",x_trim)
                print("u_trim =",u_trim)
                print()

                # run trim solver with MUX aero
                bire.aero_model._inc_aero_results = CFM_from_mux
                bire.phi_trim = phi_trim
                bire.verbose_trim = True
                bire._initialize_state(a,b,phi_trim,u_trim)

                # print success or no...
                quit()

