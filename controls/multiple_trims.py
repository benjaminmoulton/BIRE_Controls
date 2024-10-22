import numpy as np
from numpy import matmul as mm
import json
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from controller_simulation import Aircraft#,monte_carlo_perturbations,run_single_simulation
from os import mkdir, rmdir, walk, remove, listdir
from linearization import linearization
import control as co
from scipy.linalg import block_diag
from math import asin,atan2,sin,tan

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
        "E2" : { "m" : 0.6 , "h" : 15000., "V" : 634., "Re" : 31324000. }, # negative bank angles
    }

    # settings 
    run_bire = True # False # 
    run_sct  = False # True # 
    run_fs = True
    skip_run = True # False # 
    skip_DOC = False # True # 
    if run_sct: trim_bank_degs = np.linspace(0.0,75.0,num=16).tolist() # [30.0] # np.linspace(0.0,60.0,num=13).tolist() # [0.0] # [10.0] # [60.0] # np.linspace(0.0,75.0,num=16).tolist() # 
    else: trim_beta_degs = np.linspace(0.0,16.0,num=9).tolist() # [6.0] # np.linspace(0.0,16.0,num=9).tolist() # [14.0,16.0] # [0.0] # 
    trim_climb_deg = 0.0 # 10.0 # 
    fc = "T1" # "C2" # "E2" # "F1" # "U1" # "A1" # 
    cgshift = [0.0,0.0,0.0] # [1.0,0.0,0.0] # [0.5,0.0,0.0] # 
    include_compressibility =  True # False # 
    use_Anderson_corrections =  True # False # 
    include_stall =  True # False # 
    bire_plotting_xcgs = [0.0,0.5,1.0] # [-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5] # 
    cg_v_bb_xcgs_bire = [-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5]
    base_plotting_xcgs = [0.0,1.0]
    plotting_gammas = [0.0]
    plot_inverted_trims = False # True # 
    plot_alternate_trims = True # False # 
    plot_base_DOC = True # False # 
    show_plots = False # True # 
    plot_format = "pdf" # "png" # 
    plot_transparent = True if plot_format == "pdf" else False # False # True # 
    plot_dark = True # False # 
    skip_save_if_not_new = False # True # 
    add_to_guesses_list = False # True # 
    reset_guesses_list = False # True # 
    # the above only affects whether previously found trim solution guesses are saved
    # all new trim solution guesses are saved automatically
    #
    # other settings
    run_num = 1000 # 10 # 30 # 
    trim_iter = 1000 # 1000
    mfc = flight_conditions[fc]["m"] # 0.2 # 
    hfc = flight_conditions[fc]["h"] # 1000.0 # 
    a_scale = 20.0 # 0.02 # 0.0 # 
    b_scale = 20.0 # 0.2 # 0.0 # 
    p_scale = 180.0 # 0.0 # 
    u_scale = np.array([21.5,25.0,90.0,1.0]) # np.array([20.0,20.0,70.0,1.0]) # np.array([0.1,0.2,20.0,0.02]) # np.array([0.0]*4) # 
    a_shift = 0.0 # 3.17 # 
    b_shift = 0.0 # 0.0 # 
    p_shift = 0.0 # 0.0 # 
    u_shift = np.array([0.0,0.0,0.0,0.0]) # np.array([0.0,-0.05,0.0,0.276]) # 
    # set up run
    craftdict = bire_dict if run_bire else base_dict
    trim_type = "sct" if run_sct else "shss"
    scale_type = "fs" if run_fs  else "rc"
    folder = "trim_files/" + fc + "_" + trim_type + "/"
    d3 = "B" if run_bire else "r"
    ind = [0,1,2,3,4,5,8,9,10]
    pdBm = "+" # "$/$" # 
    ndBm = "_" # "$-$" # "x" # 
    odBm = "o"
    odrm = "d"
    xcg_shade = lambda xcg : xcg*0.5 + (abs(xcg)>0.0)*0.25
    other_shade = ["r","b","g","m","c","y"]
    shades = {}; shades_counter = 0
    # xcg_shade = lambda xcg : "k" if xcg == 0.0 else ("r" if xcg == 0.5 else ("b" if xcg == 1.0 else "y"))
    bire_prob_bins = (6,10)
    base_prob_bins = (6,6)
    bire_bins = (np.linspace(-u_scale[1],u_scale[1],num=bire_prob_bins[0]+1),
        np.linspace(-u_scale[2],u_scale[2],num=bire_prob_bins[1]+1))
    base_bins = (np.linspace(-u_scale[1],u_scale[1],num=base_prob_bins[0]+1),
        np.linspace(-u_scale[2],u_scale[2],num=base_prob_bins[1]+1))
    color_bar_segs = 16
    #
    max_open_figs_warn = 60

    # # rename files to include climb angle in file name
    # trim_files_folder = "./trim_files"
    # for folder in listdir(trim_files_folder):
    #     if folder != "trim_output.txt":
    #         files_folder = trim_files_folder+"/"+folder
    #         for old_filename in listdir(files_folder):
    #             if old_filename.split(".")[-1] == "json":
    #                 # pull in file
    #                 with open(files_folder+"/"+old_filename,"r") as f:
    #                     file_str = f.read()
    #                     file_dict =json.loads(file_str)

    #                 # remove file
    #                 remove(files_folder+"/"+old_filename)

    #                 # # new filename
    #                 # file_split = old_filename.replace(".json","").split("_")
    #                 # file_split = file_split + ["Gp00"] #+ file_split[6:]
    #                 # print(old_filename)
    #                 # new_filename = "_".join(file_split) + ".json"
    #                 # print(new_filename)
    #                 # print()

    #                 # save file
    #                 with open(files_folder+"/"+old_filename,"w") as f:
    #                     json.dump(file_dict, f, indent=4)
    #         print()
    # quit()

    
    # initialize aircraft
    # set to initial params
    craftdict["simulation"] = craftdict.get("simulation",{})
    craftdict["simulation"]["include_compressibility"] = include_compressibility
    craftdict["simulation"]["use_Anderson_corrections"] = use_Anderson_corrections
    craftdict["simulation"]["include_stall"] = include_stall
    craftdict["aircraft"] = craftdict.get("aircraft",{})
    craftdict["aircraft"]["CG_shift[ft]"] = cgshift
    craftdict["initial"] = craftdict.get("initial",{})
    craftdict["initial"]["mach"] = mfc
    craftdict["initial"]["altitude[ft]"] = hfc
    craftdict["initial"].pop("airspeed[ft/s]",None)
    craftdict["initial"]["type"] = "trim"
    craftdict["initial"]["trim"] = craftdict["initial"].get("trim",{})
    craftdict["initial"]["trim"]["type"] = trim_type
    craftdict["initial"]["trim"].pop("elevation_angle[deg]",None)
    craftdict["initial"]["trim"]["climb_angle[deg]"] = trim_climb_deg
    craftdict["initial"]["trim"]["solver"] = \
        craftdict["initial"]["trim"].get("solver",{})
    craftdict["initial"]["trim"]["solver"]["max_iterations"] = trim_iter
    if run_sct:
        craftdict["initial"]["trim"]["bank_angle[deg]"] = 0.0
        craftdict["initial"]["trim"].pop("sideslip_angle[deg]",None)
    else:
        craftdict["initial"]["trim"]["sideslip_angle[deg]"] = 0.0
        craftdict["initial"]["trim"].pop("bank_angle[deg]",None)

    if not(skip_run):
        # initialize
        craft = Aircraft(craftdict)
        # determine vars of interest
        if run_sct: loopvars = trim_bank_degs
        else: loopvars = trim_beta_degs
        for i in range(len(loopvars)):
            if run_sct: trim_bank_deg = loopvars[i]
            else: trim_beta_deg = loopvars[i]
            # name
            run_name  = "bire" if run_bire else "base"
            run_name += "_" + scale_type
            run_name += "_" + trim_type
            run_name += "_" + fc
            run_name += "_M{:2.1f}".format(mfc).replace(".","") + \
                "_H{:04.1f}".format(hfc/1000.).replace(".","") 
            run_name += "_CG{:+03d}".format(int(cgshift[0]*10.)).replace("+","p").replace("-","m")
            run_name += "{:+03d}".format(int(cgshift[1]*10.)).replace("+","p").replace("-","m")
            run_name += "{:+03d}".format(int(cgshift[2]*10.)).replace("+","p").replace("-","m")
            run_name += "_P{:02d}".format(int(trim_bank_deg)) if run_sct else \
                "_B{:02d}".format(int(trim_beta_deg))
            run_name += "_G{:+03d}".format(int(trim_climb_deg)).replace("+","p").replace("-","m")
            run_file = run_name + ".json"
            print("\n\nrunning", run_name,"...")
            
            # open this file if it exists
            try: 
                run_dict = json.loads( open(folder + run_file).read() )
            except: run_dict = {}
            
            # pull in saved trim states
            x_trims=[]; u_trims=[]; CFM_trims=[]; guess_bounds = []
            guess_trims=[]; final_i_trims=[]; Lin_trims = []
            for case in run_dict:
                x_trims.append(run_dict[case]["x_trim_euler"])
                u_trims.append(run_dict[case]["u_trim"])
                CFM_trims.append(run_dict[case]["CFM_trim"])
                guess_trims.append(run_dict[case]["guess_trim"])
                if "guess_bounds_aVu[deg/pu]" in run_dict[case]:
                    guess_bounds.append(run_dict[case]["guess_bounds_aVu[deg/pu]"])
                final_i_trims.append(run_dict[case]["final_i_trim"])
                Lin_trims.append(run_dict[case]["Linearized_system_trim"])
            
            num_b4 = len(x_trims)
            old_guess_trims = guess_trims*1

            # initialize aircraft -- change settings!!
            if run_sct:
                craft.phi_trim = np.deg2rad(trim_bank_deg)
                # craftdict["initial"]["trim"].pop("sideslip_angle[deg]",None)
            else:
                craft.beta_trim = np.deg2rad(trim_beta_deg)
                # craftdict["initial"]["trim"].pop("bank_angle[deg]",None)
            

            # run trims
            print("\nseeking trims...")
        
            # randomize initial guesses
            statement = "running " + run_name + " ..."
            print(statement)
            ags = np.deg2rad((np.random.random(size=(run_num,))*2. - 1.)*a_scale + a_shift)
            if run_sct:
                bgs = np.deg2rad((np.random.random(size=(run_num,))*2. - 1.)*b_scale + b_shift)
            else:
                pgs = np.deg2rad((np.random.random(size=(run_num,))*2. - 1.)*p_scale + p_shift)
            ugs = np.random.random(size=(run_num,4))#*0.
            ugs[:,0:3] = np.deg2rad((ugs[:,0:3]*2. - 1.)*u_scale[0:3] + u_shift[0:3])
            ugs[:,3] = ugs[:,3]*u_scale[3] + u_shift[3]
            # save trim sols found
            trim_sol_nums = np.array([None]*len(ags))
            #
            tol = craft.NR_tol#*1.0e+1
            # run
            found = ""
            for i in range(run_num):
                # run trim
                if run_sct:
                    guess_dict = dict(a_guess=ags[i],b_guess=bgs[i],
                        u_guess=ugs[i].tolist())
                else:
                    guess_dict = dict(a_guess=ags[i],phi_guess=pgs[i],
                        u_guess=ugs[i].tolist())
                craft._initialize_state(no_report=True,**guess_dict)
                
                # check if we have found this before
                have_found_before = False
                for j in range(len(x_trims)):
                    if np.linalg.norm(x_trims[j] - craft.x_trim_euler[0:12]) < tol\
                        and np.linalg.norm(u_trims[j] - craft.u_trim[0:4]) < tol:
                        have_found_before = True
                        trim_sol_nums[i] = j
                
                #
                if run_sct:
                    Vgs = bgs
                    vsm = "bet"
                else:
                    Vgs = pgs
                    vsm = "phi"
                report = "run {:> 5d}/{:> 5d}".format(i+1,run_num)
                report +=", guesses:a={:> 6.2f}".format(np.rad2deg(ags[i]))
                report +=",{}={:> 7.2f}".format(vsm,np.rad2deg(Vgs[i]  ))
                report +=",da={:> 6.2f}".format(    np.rad2deg(ugs[i,0]))
                report +=",de={:> 6.2f}".format(    np.rad2deg(ugs[i,1]))
                report +=",d{}={:> 6.2f}".format(d3,np.rad2deg(ugs[i,2]))
                report +=",tau={:> 5.2f}".format(              ugs[i,3] )
                #
                if not(craft.trim_failed) and not(have_found_before):
                    x_trims.append(craft.x_trim_euler[0:12].tolist())
                    u_trims.append(craft.u_trim[0:4].tolist())
                    #
                    trim_sol_nums[i] = len(x_trims) - 1
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
                    found = ", *** found new trim"
                else:
                    found = ""
                report +=", sol#= {:>2d}".format(
                    trim_sol_nums[i] if trim_sol_nums[i]!=None else -1)
                report +=", was {:>2d} now {:>2d}".format(num_b4,len(x_trims))
                print(report+found)
                
                # occasional reminder
                if (i+1) % 25 == 0:
                    print(statement)

            # Build linearized systems
            for i in range(num_b4,len(x_trims)):
                print("running linearization for trim {}...".format(i+1))

                # build linearized system
                Lin_Model = craft._build_controller(
                    x_tr=np.array(x_trims[i]+u_trims[i]),
                    u_tr=np.array(u_trims[i]),
                    report=False,save_matrices=False,
                    mrrr=[6,7,11],mrrc=None,drop_actrs=True,run_freq=False,
                    turn_off_warnings=True,skip_reporting=True)[1]
                
                # print(Lin_Model.A_min)
                Lin_trims.append(dict(A=Lin_Model.A_min.tolist(),
                    B=Lin_Model.B_min.tolist()))
            
            # guess bounds for the solution
            for j in range(len(x_trims)):
                # determine bounds
                trimsj = trim_sol_nums == j
                #
                if any(trimsj):
                    agsj = ags[trimsj]
                    if run_sct: bgsj = bgs[trimsj]
                    else:       pgsj = pgs[trimsj]
                    ugsj = ugs[trimsj]
                    # append values to list
                    if run_sct: new_guesses = np.vstack((agsj,bgsj,ugsj.T)).T
                    else:       new_guesses = np.vstack((agsj,pgsj,ugsj.T)).T
                    # rad2deg
                    new_guesses[:,:-1] = np.rad2deg(new_guesses[:,:-1])
                    print(len(guess_bounds))

                if len(guess_bounds) >= j + 1:
                    guess_bounds[j] = np.concatenate((guess_bounds[j],new_guesses),axis=0).tolist()
                else:
                    guess_bounds.append(new_guesses.tolist())
            
            # save to file (if we found more)
            if len(x_trims) > num_b4 or not(skip_save_if_not_new):
                print("\nsaving to file",run_name,"...")
                # create datadict
                data_dict = {}
                for j in range(len(x_trims)):
                    case = str(j)
                    data_dict[case] = {}
                    data_dict[case]["x_trim_euler"] = x_trims[j]
                    data_dict[case]["u_trim"] = u_trims[j]
                    data_dict[case]["CFM_trim"] = CFM_trims[j]
                    if reset_guesses_list:
                        data_dict[case]["guess_trim"] = \
                            guess_trims[j][len(old_guess_trims[j]):]
                    elif (add_to_guesses_list or len(old_guess_trims) == 0 
                        or len(old_guess_trims[j]) == 0):
                        data_dict[case]["guess_trim"] = guess_trims[j]
                    else:
                        data_dict[case]["guess_trim"] = old_guess_trims[j]
                    data_dict[case]["guess_bounds_aVu[deg/pu]"] = guess_bounds[j]
                    data_dict[case]["final_i_trim"] = final_i_trims[j]
                    data_dict[case]["Linearized_system_trim"] = Lin_trims[j]
                # save
                with open(folder+run_file, "w") as f:
                    json.dump(data_dict, f, indent=4)
            else:
                print("\nnot saving to file",run_name,
                    ", no new trim sols found")
            
            # report trim solutions
            for j in range(len(x_trims)):
                header = "trim solution {:> 3d}: ".format(j)
                header += "phi={:> 5.2f}, ".format(np.rad2deg(x_trims[j][9]))
                header += "{}".format("dB" if run_bire else "dr")
                header += "={:> 5.2f}".format(np.rad2deg(u_trims[j][2]))
                print(header)
                keys = ["alpha[deg]_[min,max]"]
                if run_sct: keys += ["beta[deg]_[min,max]"]
                else      : keys += [ "phi[deg]_[min,max]"]
                keys += ["da[deg]_[min,max]","de[deg]_[min,max]",
                         "u3[deg]_[min,max]","tau[pu]_[min,max]"]
                for k in range(6):
                    gls = np.array(guess_bounds[j]).T
                    print("    {:>20s} : {:> 5.2f},{:> 5.2f}".format(
                        keys[k],np.min(gls[k]),np.max(gls[k])))
                print("    ")
    
    # plot trim values
    if skip_run:
        # report
        print("plotting {} {} {} ...\n".format(scale_type,trim_type,fc))
        # initialize values dict
        trims = {}
        # read in all values
        for filename in listdir(folder):
            file_split = filename.replace(".json","").split("_")
            if file_split[1:4] == [scale_type, trim_type, fc]:
                # name of aircraft is bire/base + CG info
                name = file_split[0] + "_" + file_split[6] + "_" + file_split[8]
                if name not in trims:
                    trims[name] = {}
                    trims[name]["ind_var"] = []
                    trims[name]["dicts"] = []
                    trims[name]["filenames"] = []
                
                # read in info
                path = folder + filename
                trims[name]["dicts"].append(json.loads( open(path).read() ))
                trims[name]["ind_var"].append(float(file_split[7][1:]))
                trims[name]["filenames"].append(filename)
                
                # print(file_split, name)
                # calculate alpha and beta and save to dicts, as well as psi_dot
                sols = trims[name]["dicts"][-1]
                for sol in sols:
                    x = sols[sol]["x_trim_euler"]
                    a = np.arctan2(x[2],x[0])
                    V = (x[0] * x[0] + x[1] * x[1] + x[2] * x[2])**0.5
                    b = np.arcsin(x[1]/V)
                    trims[name]["dicts"][-1][sol]["angles"] = [a,b,x[9]]
                    psi_dot = (np.sin(x[9])*x[4] 
                        + np.cos(x[9])*x[5])/np.cos(x[10])
                    trims[name]["dicts"][-1][sol]["psi_dot"] = psi_dot
        
        # sort by ind_var
        for craft in trims:
            trims[craft]["min_ind"] = []
            # numpify
            trims[craft]["ind_var"] = np.array(trims[craft]["ind_var"])
            trims[craft]["dicts"] = np.array(trims[craft]["dicts"])
            trims[craft]["filenames"] = np.array(trims[craft]["filenames"])
            # get sort indices
            sorter = np.argsort(trims[craft]["ind_var"])
            # sort
            trims[craft]["ind_var"] = trims[craft]["ind_var"][sorter]
            trims[craft]["dicts"] = trims[craft]["dicts"][sorter]
            trims[craft]["filenames"] = trims[craft]["filenames"][sorter]

            # if BIRE, determine smallest tail angle case
            if craft[:4] == "bire":
                for j in range(len(trims[craft]["ind_var"])):
                    # determine list of tail angles
                    trimsols = trims[craft]["dicts"][j]
                    tails = [trimsols[sol]["u_trim"][2] for sol in trimsols]
                    noninvs=[abs(trimsols[sol]["x_trim_euler"][9]) < np.pi/2. 
                        for sol in trimsols]
                    #
                    indsort = np.argsort(np.abs(tails))
                    noninvs = np.array(noninvs)[indsort].tolist()
                    if any(noninvs): minind = (indsort[noninvs])[0]
                    else: minind = None
                    trims[craft]["min_ind"].append(minind)
                    # print(craft,trims[craft]["ind_var"][j],tails,noninvs,any(noninvs),indsort,minind)
            
            # Determine eigendecomposed matrix
            if not(skip_DOC):
                for j,trim_set in enumerate(trims[craft]["dicts"]):
                    for trim_sol in trim_set:
                        solution = trim_set[trim_sol]
                        A = solution["Linearized_system_trim"]["A"]
                        B = solution["Linearized_system_trim"]["B"]
                        # rows = [0,2,4,8]
                        # cols = [0,1,2]
                        # A = (np.array(A)[rows])[:,rows].tolist()
                        # B = (np.array(B)[rows])[:,cols].tolist()
                        # f = 2.0
                        # A = np.array([ [-1.,-f-1.,0.], [0.,0.,1.], [1.,2.,0.] ])
                        # B = np.array([ [1.,0.], [0.,0.], [-1.,-1.] ])
                        eigs,Q = np.linalg.eig(A)
                        i_eigs = np.argsort(eigs)
                        i_s = list(range(len(eigs[eigs < 0.0])))
                        i_a = np.delete(range(len(eigs)),i_s).tolist()
                        eigs = eigs[i_eigs]; Q = Q[:,i_eigs]
                        Qinv = np.linalg.solve(Q,np.eye(Q.shape[0]))
                        # A
                        AT = mm(Qinv,mm(A,Q))
                        As = (AT[i_s])[:,i_s]
                        Aa = (AT[i_a])[:,i_a]
                        # C
                        Cs = np.eye(len(A))[i_s]
                        Ca = np.eye(len(A))[i_a]
                        Wss = []
                        Was = []
                        for col in range(len(B[0])):
                            # B
                            BT = mm(Qinv,B)[:,col:col+1] # [:,0:1] #  # 
                            # slice off stable and unstable
                            Bs = BT[i_s]
                            Ba = BT[i_a]
                            # solve for grammians co lyap is A X + X A^T + Q = 0
                            Wsinv = np.linalg.solve(co.lyap( \
                                As,mm(Bs,Bs.conj().T),method="scipy"),np.eye(len(i_s)))
                            if len(i_a):
                                Wainv = np.linalg.solve(co.lyap( \
                                    -Aa,mm(Ba,Ba.conj().T),method="scipy"),np.eye(len(i_a)))
                            else:
                                Wainv = []

                            Wsinvf = mm(mm(mm(mm(\
                                Qinv.conj().T,Cs.T),Wsinv),Cs),Qinv)
                            if len(i_a):
                                Wainvf = mm(mm(mm(mm(\
                                    Qinv.conj().T,Ca.T),Wainv),Ca),Qinv)
                            else:
                                Wainvf = np.zeros((len(A),len(A)))
                            # print(Cs)
                            # print(Wsinvf)
                            # print(Wsinvf.shape)
                            # print(Ca)
                            # print(Wainvf)
                            # print(Wainvf.shape)
                            Wss.append(Wsinvf)
                            Was.append(Wainvf)
                        trims[craft]["dicts"][j][trim_sol]["Linearized_system_trim"]["Wss"] = Wss
                        trims[craft]["dicts"][j][trim_sol]["Linearized_system_trim"]["Was"] = Was
                        # x0 = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0])
                        # xf = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
                        # #
                        # rhobars = [np.abs(mm(xf,mm(Wss[k],xf))) for k in range(len(Wss))]
                        # rhobara = [np.abs(mm(x0,mm(Was[k],x0))) for k in range(len(Was))]
                        # rhobarT = [rhobars[k] + rhobara[k] for k in range(len(rhobars))]
                        # print(rhobarT)
                        # # newWss = np.block([Wss[0],Wss[1],Wss[2],Wss[3]])
                        # # newWss = mm(newWss,np.block([[np.eye(9)],[np.eye(9)],[np.eye(9)],[np.eye(9)]]))
                        # # rhobarT = 
                        # quit()
                        # #
                        # #
                        # xT0 = mm(Qinv,x0)
                        # xTf = mm(Qinv,xf)
                        # xa0 = xT0[i_a]
                        # xsf = xTf[i_s]
                        # xc = np.concatenate((xsf,xa0))
                        # rhobar = mm(xc[:,np.newaxis].conj().T,mm(Winv,xc))[0]
                        # print("i_s\n",i_s)
                        # print("i_a\n",i_a)
                        # print("eigs\n",eigs)
                        # print()
                        # print("Q\n",Q)
                        # print()
                        # print("Qinv\n",Qinv)
                        # print()
                        # print("AT\n",AT)
                        # print()
                        # print("BT\n",BT)
                        # print()
                        # print("Wsinv\n",Wsinv)
                        # print()
                        # print("Wainv\n",Wainv)
                        # print()
                        # print("xT0\n",xT0)
                        # print()
                        # print("xTf\n",xTf)
                        # print()
                        # print("rhobar\n",rhobar)
                        # print()
                        # quit()
        # quit()
        
        # width in inches
        width = 3.25 # 4.0 # 
        scale_font_size = 3.25/width

        if plot_dark:
            plots_to_run = ["default","dark_background"]
        else:
            plots_to_run = ["default"]
        
        for plot_type in plots_to_run:
            plt.style.use([plot_type])
            if plot_type == "dark_background":
                xcg_shade_inverter = lambda xcg : xcg_shade(xcg)*-1. + 1.
            else:
                xcg_shade_inverter = lambda xcg : xcg_shade(xcg)

            # change plot text parameters
            plt.rcParams["font.family"] = "Serif"
            plt.rcParams["font.size"] = 10.0#*scale_font_size
            plt.rcParams["axes.labelsize"] = 10.0#*scale_font_size
            # plt.rcParams['axes.xmargin'] = 0 # uncomment to have xaxis fit data
            plt.rcParams['lines.linewidth'] = 0.75 # 1.0
            plt.rcParams["xtick.minor.visible"] = True
            plt.rcParams["ytick.minor.visible"] = True
            plt.rcParams["xtick.direction"] = plt.rcParams["ytick.direction"] = "in"
            plt.rcParams["xtick.bottom"] = plt.rcParams["xtick.top"] = True
            plt.rcParams["ytick.left"] = plt.rcParams["ytick.right"] = True
            plt.rcParams["xtick.major.width"] = plt.rcParams["ytick.major.width"] = 0.75
            plt.rcParams["xtick.minor.width"] = plt.rcParams["ytick.minor.width"] = 0.75
            plt.rcParams["xtick.major.size"] = plt.rcParams["ytick.major.size"] = 5.0
            plt.rcParams["xtick.minor.size"] = plt.rcParams["ytick.minor.size"] = 2.5
            plt.rcParams["mathtext.fontset"] = "dejavuserif"
            plt.rcParams['figure.dpi'] = 300.0
            plt.rcParams['figure.max_open_warning'] = max_open_figs_warn

            # initialize plot saving params
            save_dict = dict(transparent=plot_transparent,dpi=300.0)
            sv_fldr = "plots" if plot_type == "default" else "plots_inv"
            sv_fldr = folder + sv_fldr + "/"

            # initialize plots
            plot_dict = dict(figsize=(width,3.5),dpi=300.0, # sharex=True,
                constrained_layout=True)
            prob_dict = dict(figsize=(3.25,3.5),dpi=300.0, # sharex=True,
                constrained_layout=True)
            fig_da,axs_da = plt.subplots(1,1,**plot_dict)
            fig_de,axs_de = plt.subplots(1,1,**plot_dict)
            fig_dB,axs_dB = plt.subplots(1,1,**plot_dict)
            fig_ta,axs_ta = plt.subplots(1,1,**plot_dict)
            fig_t2,axs_t2 = plt.subplots(1,1,**plot_dict)
            fig_vr,axs_vr = plt.subplots(1,1,**plot_dict)
            fig_af,axs_af = plt.subplots(1,1,**plot_dict)
            fig_th,axs_th = plt.subplots(1,1,**plot_dict)
            fig_wp,axs_wp = plt.subplots(1,1,**plot_dict)
            fig_wq,axs_wq = plt.subplots(1,1,**plot_dict)
            fig_wr,axs_wr = plt.subplots(1,1,**plot_dict)
            fig_ps,axs_ps = plt.subplots(1,1,**plot_dict)
            fig_CL,axs_CL = plt.subplots(1,1,**plot_dict)
            fig_CS,axs_CS = plt.subplots(1,1,**plot_dict)
            fig_CD,axs_CD = plt.subplots(1,1,**plot_dict)
            fig_Cl,axs_Cl = plt.subplots(1,1,**plot_dict)
            fig_Cm,axs_Cm = plt.subplots(1,1,**plot_dict)
            fig_Cn,axs_Cn = plt.subplots(1,1,**plot_dict)
            fig_eg,axs_eg = plt.subplots(1,1,**plot_dict)
            fig_bb,axs_bb = plt.subplots(1,1,**plot_dict)
            fig_bp,axs_bp = plt.subplots(1,1,**plot_dict)
            fig_gs,axs_gs = plt.subplots(1,1,**plot_dict)
            fig_lg,axs_lg = plt.subplots(1,1,**plot_dict)
            rows_DC = 9; cols_DC = 4
            fig_DC = [[None for _ in range(cols_DC)] for _ in range(rows_DC)]
            axs_DC = [[None for _ in range(cols_DC)] for _ in range(rows_DC)]
            for i in range(rows_DC):
                for j in range(cols_DC):
                    if j == 0: shares ={}
                    else: shares ={"sharex":axs_DC[i][0],"sharey":axs_DC[i][0]}
                    fig_DC[i][j],axs_DC[i][j] = plt.subplots(1,1,**shares,**plot_dict)
            axs = [axs_da,axs_de,axs_dB,axs_ta,axs_vr,axs_af]
            # # ctrb plots
            # ctrb_fig = {}
            # for craft in trims:
            #     if craft not in ctrb_fig:
            #         ctrb_fig[craft] = {}
            #         ctrb_fig[craft]["fig"],ctrb_fig[craft]["ax"] = \
            #             plt.subplots(1,1,**plot_dict)
            #         print(ctrb_fig.keys())
            # #
            if run_sct: k = 1
            else:       k = 2
            # add grids
            if plot_type == "dark_background": grid_color = "0.15"
            else                             : grid_color = "0.85"
            grid_lw = 0.6
            axs_da.grid(which="major",lw=grid_lw,ls="-",c=grid_color)
            axs_de.grid(which="major",lw=grid_lw,ls="-",c=grid_color)
            axs_dB.grid(which="major",lw=grid_lw,ls="-",c=grid_color)
            axs_ta.grid(which="major",lw=grid_lw,ls="-",c=grid_color)
            axs_t2.grid(which="major",lw=grid_lw,ls="-",c=grid_color)
            axs_vr.grid(which="major",lw=grid_lw,ls="-",c=grid_color)
            axs_af.grid(which="major",lw=grid_lw,ls="-",c=grid_color)
            axs_CL.grid(which="major",lw=grid_lw,ls="-",c=grid_color)
            axs_CS.grid(which="major",lw=grid_lw,ls="-",c=grid_color)
            axs_CD.grid(which="major",lw=grid_lw,ls="-",c=grid_color)
            axs_Cl.grid(which="major",lw=grid_lw,ls="-",c=grid_color)
            axs_Cm.grid(which="major",lw=grid_lw,ls="-",c=grid_color)
            axs_Cn.grid(which="major",lw=grid_lw,ls="-",c=grid_color)
            axs_eg.grid(which="major",lw=grid_lw,ls="-",c=grid_color)
            axs_bb.grid(which="major",lw=grid_lw,ls="-",c=grid_color)
            axs_bp.grid(which="major",lw=grid_lw,ls="-",c=grid_color)
            axs_th.grid(which="major",lw=grid_lw,ls="-",c=grid_color)
            axs_wp.grid(which="major",lw=grid_lw,ls="-",c=grid_color)
            axs_wq.grid(which="major",lw=grid_lw,ls="-",c=grid_color)
            axs_wr.grid(which="major",lw=grid_lw,ls="-",c=grid_color)
            axs_ps.grid(which="major",lw=grid_lw,ls="-",c=grid_color)
            for i in range(rows_DC):
                for j in range(cols_DC):
                    axs_DC[i][j].grid(which="major",lw=grid_lw,ls="-",
                        c=grid_color)

            # plot points
            rest_dict = dict(ms=3.5,mew=0.75,fillstyle="none",ls="none")
            for craft in trims:
                cname,cgs,clm = craft.split("_")
                xcg = float(cgs.replace("p","_+").replace("m","_-").split("_")[1])/10.
                for i in range(len(trims[craft]["dicts"])):
                    # pull out sol
                    
                    # pull out trim guess matrices
                    M = []; X = []; Y = []
                    bound_counter = 0
                    for j,trim_sol in enumerate(trims[craft]["dicts"][i]):
                        sol = trims[craft]["dicts"][i][trim_sol]
                        if "guess_bounds_aVu[deg/pu]" in sol:
                            bound_counter += 1
                            guesses = np.array(sol["guess_bounds_aVu[deg/pu]"])
                            de_guesses = guesses[:,3]
                            dB_guesses = guesses[:,4]
                            Mj, xedges, yedges = \
                                np.histogram2d(de_guesses,dB_guesses,
                                bins=bire_bins if cname == "bire" 
                                else base_bins)
                            M.append(Mj.T)
                            Xj, Yj = np.meshgrid(xedges, yedges)
                            X.append(Xj); Y.append(Yj)
                    if bound_counter > 0:
                        # determine sum
                        Msum = np.sum(M,axis=0)
                        # make so each is a percentage
                        for iM in range(len(M)):
                            M[iM][Msum==0.0] = -1.0
                            M[iM][Msum!=0.0] /= Msum[Msum!=0.0]
                    
                    # move on
                    for j,trim_sol in enumerate(trims[craft]["dicts"][i]):
                        sol = trims[craft]["dicts"][i][trim_sol]
                        ivr = trims[craft]["ind_var"][i]
                        # setup marker style
                        kdict = {}
                        # craft type
                        if cname == "bire":
                            # print(i,j,trims[craft]["min_ind"][i],ivr)
                            if j == trims[craft]["min_ind"][i]:
                                kdict = {**kdict, **dict(marker = odBm)}
                            elif sol["u_trim"][2] > 0.0:
                                kdict = {**kdict, **dict(marker = pdBm)}
                            else: # if sol["u_trim"][2] < 0.0:
                                kdict = {**kdict, **dict(marker = ndBm)}
                        else:   kdict = {**kdict, **dict(marker = odrm)}
                        # cg loc
                        kdict = {**kdict, **dict(color = str(xcg_shade_inverter(xcg)))}
                        # neg xcgs
                        if xcg < 0.0 or xcg > 1.0:
                            if xcg not in shades:
                                shades[xcg] = shades_counter % len(other_shade)
                                shades_counter += 1
                            kdict["color"] = other_shade[shades[xcg]]
                        #
                        if np.rad2deg(abs(sol["angles"][k])) > 90.0:
                            kdict["color"] = "#FFA500"
                        # others
                        kdict = {**kdict, **rest_dict}
                        # #
                        # Amat = sol["Linearized_system_trim"]["A"]
                        # Bmat = sol["Linearized_system_trim"]["B"]
                        # Gmat = co.ctrb(Amat,Bmat)
                        # print(craft,np.linalg.matrix_rank(Gmat))
                        # #
                        # PLOTS! quit if not meeting criterion
                        if cname == "bire" and j != trims[craft]["min_ind"][i] \
                            and not(plot_alternate_trims):
                            continue
                        if np.rad2deg(abs(sol["angles"][k])) > 90.0 and not(plot_inverted_trims):
                            continue
                        # cg vs beta / phi
                        if cname == "bire" and xcg in cg_v_bb_xcgs_bire:
                            temp_c = kdict["color"]*1
                            kdict["color"] = str(xcg_shade_inverter(0.0))
                            axs_bb.plot(xcg,np.rad2deg(sol["angles"][1]),**kdict)
                            axs_bp.plot(xcg,np.rad2deg(sol["angles"][2]),**kdict)
                            kdict["color"] = temp_c
                        # #
                        if cname == "bire" and xcg not in bire_plotting_xcgs:
                            continue
                        if cname == "base" and xcg not in base_plotting_xcgs:
                            continue
                        # aileron
                        axs[0].plot(ivr,np.rad2deg(sol["u_trim"][0]),**kdict)
                        # elevator
                        axs[1].plot(ivr,np.rad2deg(sol["u_trim"][1]),**kdict)
                        # rudder
                        axs[2].plot(ivr,np.rad2deg(sol["u_trim"][2]),**kdict)
                        # throttle
                        axs[3].plot(ivr,           sol["u_trim"][3] ,**kdict)
                        # throttle diff
                        if cname == "bire" and j != trims[craft]["min_ind"][i]:
                            name = str(trims[craft]["min_ind"][i])
                            tau_zro = trims[craft]["dicts"][i][name]["u_trim"][3]
                            axs_t2.plot(ivr,sol["u_trim"][3] - tau_zro,**kdict)
                        # sideslip / bank
                        axs[4].plot(ivr,np.rad2deg(sol["angles"][k]),**kdict)
                        # aoa
                        axs[5].plot(ivr,np.rad2deg(sol["angles"][0]),**kdict)
                        # theta
                        axs_th.plot(ivr,np.rad2deg(sol["x_trim_euler"][10]),
                            **kdict)
                        # p
                        axs_wp.plot(ivr,np.rad2deg(sol["x_trim_euler"][3]),
                            **kdict)
                        # q
                        axs_wq.plot(ivr,np.rad2deg(sol["x_trim_euler"][4]),
                            **kdict)
                        # r
                        axs_wr.plot(ivr,np.rad2deg(sol["x_trim_euler"][5]),
                            **kdict)
                        # psi_dot
                        axs_ps.plot(ivr,np.rad2deg(sol["psi_dot"]),**kdict)
                        #
                        # aero
                        axs_CL.plot(ivr,sol["CFM_trim"][0],**kdict)
                        axs_CS.plot(ivr,sol["CFM_trim"][1],**kdict)
                        axs_CD.plot(ivr,sol["CFM_trim"][2],**kdict)
                        axs_Cl.plot(ivr,sol["CFM_trim"][3],**kdict)
                        axs_Cm.plot(ivr,sol["CFM_trim"][4],**kdict)
                        axs_Cn.plot(ivr,sol["CFM_trim"][5],**kdict)
                        #
                        # most unstable eigenvalue
                        LinSys = sol["Linearized_system_trim"]
                        A_min = np.array(LinSys["A"])
                        # A_min = (A_min[ind,:])[:,ind]
                        evals,_ = np.linalg.eig(A_min)
                        axs_eg.plot(np.max(np.real(evals)),ivr,**kdict)
                        #
                        # DOC
                        # # # if skip for base, then skip
                        if cname == "base" and not(plot_base_DOC): continue
                        DOCval = 10.0
                        if not(skip_DOC):
                            for g in range(rows_DC):
                                # to zero
                                x0 = np.zeros((9,)); x0[g] = DOCval; xf = x0*0.0
                                # info
                                Vxb,Vyb,Vzb = sol["x_trim_euler"][0:3]
                                V = (Vxb**2. + Vyb**2. + Vzb**2.)**0.5
                                a = atan2(Vzb,Vxb)
                                b = asin(Vyb/V)
                                if g == 0: # do V
                                    V_new = V + DOCval
                                    Vyb_new = sin(b)*V_new
                                    Vxb_new = ((V_new**2. - Vyb_new**2.)/
                                        (1. + tan(a)**2.))**0.5
                                    Vzb_new = Vxb_new*tan(a)
                                    x0[0] = Vxb_new - Vxb
                                    x0[1] = Vyb_new - Vyb
                                    x0[2] = Vzb_new - Vzb
                                elif g == 1: # do alpha
                                    a_new = a + np.deg2rad(DOCval)
                                    Vyb_new = Vyb
                                    Vxb_new = ((Vxb**2. + Vzb**2.)/
                                        (1. + tan(a_new)**2.))**0.5
                                    Vzb_new = Vxb_new*tan(a_new)
                                    x0[0] = Vxb_new - Vxb
                                    x0[1] = Vyb_new - Vyb
                                    x0[2] = Vzb_new - Vzb
                                elif g == 2: # do beta
                                    b_new = b + np.deg2rad(DOCval)
                                    Vyb_new = sin(b_new)*V
                                    Vxb_new = Vxb
                                    Vzb_new = Vzb
                                    x0[0] = Vxb_new - Vxb
                                    x0[1] = Vyb_new - Vyb
                                    x0[2] = Vzb_new - Vzb
                                elif g in [3,4,5,7,8]: # put in radians
                                    x0[g] = np.deg2rad(DOCval)
                                Wss = LinSys["Wss"]; Was = LinSys["Was"]
                                for h in range(cols_DC):
                                    rhos = np.abs(mm(xf,mm(Wss[h],xf)))
                                    rhoa = np.abs(mm(x0,mm(Was[h],x0)))
                                    rho = rhos + rhoa
                                    axs_DC[g][h].plot(ivr,rho,**kdict)
                                # # from zero
                                # xf = x0*1.0; x0 *= 0.0
                                # rhoss = [np.abs(mm(xf,mm(Wss[m],xf))) for m in range(len(Wss))]
                                # rhoas = [np.abs(mm(x0,mm(Was[m],x0))) for m in range(len(Was))]
                                # rhos = [rhoss[m] + rhoas[m] for m in range(len(rhoss))]
                                # axs_DC[g][1].plot(ivr,np.min(rhos),**kdict)
                        # print(cname,ivr,xcg)
                        # if cname == "bire" and ivr == 30.0 and xcg == 0.0: # len(trims[craft]["dicts"][i]) > 1:
                        #     print(craft,ivr,xcg,np.rad2deg(sol["u_trim"][1]),
                        #         np.rad2deg(sol["u_trim"][2]),len(trims[craft]["dicts"][i])) #trims[craft]["dicts"][i])
                        if len(trims[craft]["dicts"][i]) > 1:
                            # print(craft,ivr,xcg,np.rad2deg(sol["u_trim"][1]),
                            #     np.rad2deg(sol["u_trim"][2]),len(trims[craft]["dicts"][i])) #trims[craft]["dicts"][i])
                            # plot contingency table
                            if plot_type == "dark_background": 
                                cmap = "gray_r"
                                mec = "w"
                                mfc = "k"
                            else:
                                cmap = "gray"
                                mec = "k"
                                mfc = "w"
                            colmap = plt.get_cmap(cmap,color_bar_segs)
                            pc = axs_gs.pcolormesh(X[j], Y[j], M[j],
                                cmap=colmap,vmin=-1.0,vmax=1.0)
                            # cb = fig_gs.colorbar(pc)
                            axs_gs.plot(np.rad2deg(sol["u_trim"][1]),
                                        np.rad2deg(sol["u_trim"][2]),".",
                                        mec=mec,mfc=mfc)
                            if cname == "bire":
                                axs_gs.set_xlabel(r"$\delta_e^B$ guess")
                                axs_gs.set_ylabel(r"$\delta_B$ guess")
                            else:
                                axs_gs.set_xlabel(r"$\delta_e$ guess")
                                axs_gs.set_ylabel(r"$\delta_r$ guess")
                            # save fig / show
                            filename = trims[craft]["filenames"][i].replace(".json","")
                            file_desc = filename.split("_")
                            gs_fn = "_".join(file_desc[0:2] + file_desc[6:9])
                            # print(craft,j,gs_fn)
                            fig_gs.savefig(sv_fldr+"11_"+gs_fn+"_"+str(j)+"."+plot_format,**save_dict)
                            # cb.remove()
                            axs_gs.cla()
            plt.close(fig_gs)
            # quit()
            
            # other plot params
            if trim_type == "sct": 
                xlabel = r"Bank angle ($\phi$), deg"
                vrylbl = r"Sideslip angle ($\beta$), deg"
            else: 
                xlabel = r"Sideslip angle ($\beta$), deg"
                vrylbl = r"Bank angle ($\phi$), deg"
            [ax.set_xlabel(xlabel) for ax in axs]
            axs[0].set_ylabel(r"Aileron ($\delta_a$), deg")
            axs[1].set_ylabel(r"Stabilator ($\delta_e^B$/$\delta_e$), deg")
            axs[2].set_ylabel(r"Tail rotation/rudder ($\delta_B$/$\delta_r$), deg")
            axs[3].set_ylabel(r"Throttle setting ($\tau$), per-unit")
            axs[4].set_ylabel(vrylbl)
            axs[5].set_ylabel(r"Angle of attack ($\alpha$), deg")
            #
            axs_t2.set_xlabel(xlabel)
            axs_t2.set_ylabel(r"Throttle difference ($\tau - \tau_{n0}$), per-unit")
            #
            axs_th.set_xlabel(xlabel)
            axs_th.set_ylabel(r"Elevation angle ($\theta$), deg")
            axs_wp.set_xlabel(xlabel)
            axs_wp.set_ylabel( r"Roll rate ($p$), deg/s")
            axs_wq.set_xlabel(xlabel)
            axs_wq.set_ylabel(r"Pitch rate ($q$), deg/s")
            axs_wr.set_xlabel(xlabel)
            axs_wr.set_ylabel(  r"Yaw rate ($r$), deg/s")
            axs_ps.set_xlabel(xlabel)
            axs_ps.set_ylabel(r"Change in heading ($\dot{\psi}$), deg/s")
            #
            axs_CL.set_xlabel(xlabel)
            axs_CL.set_ylabel(      r"Lift coefficient ($C_L$)")
            axs_CS.set_xlabel(xlabel)
            axs_CS.set_ylabel(r"Side-force coefficient ($C_S$)")
            axs_CD.set_xlabel(xlabel)
            axs_CD.set_ylabel(      r"Drag coefficient ($C_D$)")
            axs_Cl.set_xlabel(xlabel)
            axs_Cl.set_ylabel( r"Rolling moment coefficient ($C_\ell$)")
            axs_Cm.set_xlabel(xlabel)
            axs_Cm.set_ylabel(r"Pitching moment coefficient ($C_m$)")
            axs_Cn.set_xlabel(xlabel)
            axs_Cn.set_ylabel(  r"Yawing moment coefficient ($C_n$)")
            #
            axs_eg.set_ylabel(xlabel)
            axs_eg.set_xlabel(r"$\max \left( \operatorname{real} \left( " + \
                r"\lambda \right) \right)$, 1/sec")
            #
            axs_bb.set_xlabel(r"Center of gravity shift ($\Delta x_{cg}$), ft")
            axs_bb.set_ylabel(r"Sideslip angle ($\beta$), deg")
            axs_bp.set_xlabel(r"Center of gravity shift ($\Delta x_{cg}$), ft")
            axs_bp.set_ylabel(r"Bank angle ($\phi$), deg")
            # DOC
            state_names = ["V","a","b","p","q","r","zf","phi","theta"]
            state_vars = ["$V$",r"$\alpha$",r"$\beta$","$p$","$q$","$r$",
                "$z_f$",r"$\phi$",r"$\theta$"]
            ctrl_names = ["da","de","dB","ta"]
            ctrl_vars = [r"$\delta_a$",r"$\delta_e^B$/$\delta_e$",
                r"$\delta_B$/$\delta_r$",r"$\tau$"]
            for g in range(rows_DC):
                for h in range(cols_DC):
                    axs_DC[g][h].set_xlabel(xlabel)
                    axs_DC[g][h].set_ylabel(
                        "DOC of "+state_vars[g]+" using "+ctrl_vars[h])# + state_names[g])
                    axs_DC[g][h].set_yscale("log")
                    if g == 6:
                        axs_DC[g][h].set_ylim((1.0e-26,1.0e-17))
                    elif g in [7,8]:
                        axs_DC[g][h].set_ylim((1.0e-6,1.0e+2))
                    else:
                        axs_DC[g][h].set_ylim((1.0e-4,1.0e+2))
            # legend
            sp = [
                Line2D([0], [0],color=str(xcg_shade_inverter(0.0)),marker=odBm,**rest_dict),
                Line2D([0], [0],color=str(xcg_shade_inverter(0.0)),marker=odrm,**rest_dict),
                Line2D([0], [0],color=str(xcg_shade_inverter(0.5)),marker=odBm,**rest_dict),
                Line2D([0], [0],color=str(xcg_shade_inverter(1.0)),marker=odBm,**rest_dict),
                Line2D([0], [0],color=str(xcg_shade_inverter(0.0)),marker=pdBm,**rest_dict),
                Line2D([0], [0],color=str(xcg_shade_inverter(0.0)),marker=ndBm,**rest_dict)
            ]
            lbls = [
                "BIRE",
                "base",
                r"$\Delta x_{cg} = 0.5$",
                r"$\Delta x_{cg} = 1.0$",
                r"$\delta_B$ > $\min(|\delta_B|)$",
                r"$\delta_B$ < $\min(|\delta_B|)$"
            ]
            legdict = dict(handles=sp,labels=lbls,#loc=(1.0,0.0),
                borderpad=0.1,handletextpad=0.0)
            # axs[1].legend(**legdict)
            # axs[2].legend(**legdict)
            # axs[3].legend(**legdict)
            # axs[4].legend(**legdict)
            # axs[5].legend(**legdict)
            # axs_CD.legend(**legdict)
            # axs_eg.legend(**legdict)
            # if plot_base_DOC:
            #     legDOCdict = legdict
            # else:
            #     legDOCdict = dict(handles=sp[2:],labels=lbls[2:],loc=(1.0,0.0),
            #         borderpad=0.1,handletextpad=0.0)
            # for g in range(rows_DC):
            #     for h in range(cols_DC):
            #         axs_DC[g][h].legend(**legDOCdict)
            LEGdict = dict(handles=sp,labels=lbls,loc="lower center", #(1.0,0.0),
                borderpad=0.1,handletextpad=0.0)
            fr2dict = dict(handles=sp[2:],labels=lbls[2:],#loc="lower center", #(1.0,0.0),
                borderpad=0.1,handletextpad=0.0)
            cmpdict = dict(handles=sp[4:],labels=lbls[4:],#loc="lower center", #(1.0,0.0),
                borderpad=0.1,handletextpad=0.0)
            axs_lg.legend(**LEGdict)
            axs_lg.axis("off")

            # save plots
            # if plot type is different than previous, remove them
            for filename in listdir(sv_fldr[:-1]):
                if filename.split(".")[-1] != plot_format:
                    remove(sv_fldr+filename)
            nm = "beta" if trim_type == "sct" else "phi"
            fig_da   .savefig(sv_fldr+"00_da."    +plot_format,**save_dict)
            axs[0].legend(fontsize=8.0,**legdict) # aileron
            fig_da   .savefig(sv_fldr+"00_da_wlg."+plot_format,**save_dict)
            fig_de   .savefig(sv_fldr+"01_de."    +plot_format,**save_dict)
            fig_dB   .savefig(sv_fldr+"02_dB."    +plot_format,**save_dict)
            fig_ta   .savefig(sv_fldr+"03_tau."   +plot_format,**save_dict)
            axs_t2.legend(fontsize=8.0,**fr2dict)
            fig_t2   .savefig(sv_fldr+"03_tau_diff."+plot_format,**save_dict)
            #
            fig_vr   .savefig(sv_fldr+"04_"+nm+"."+plot_format,**save_dict)
            axs_vr.legend(fontsize=8.0,**legdict)
            fig_vr   .savefig(sv_fldr+"04_"+nm+"_wlg."+plot_format,**save_dict)
            fig_th   .savefig(sv_fldr+"04_theta."+plot_format,**save_dict)
            axs_th.legend(fontsize=8.0,**legdict)
            fig_th   .savefig(sv_fldr+"04_theta_wlg."+plot_format,**save_dict)
            fig_af   .savefig(sv_fldr+"05_alpha." +plot_format,**save_dict)
            axs_af.legend(fontsize=8.0,**legdict)
            fig_af   .savefig(sv_fldr+"05_alpha_wlg." +plot_format,**save_dict)
            #
            fig_CL   .savefig(sv_fldr+"06_CL."    +plot_format,**save_dict)
            axs_CL.legend(fontsize=8.0,**legdict)
            fig_CL   .savefig(sv_fldr+"06_CL_wlg."+plot_format,**save_dict)
            fig_CS   .savefig(sv_fldr+"06_CS."    +plot_format,**save_dict)
            axs_CS.legend(fontsize=8.0,**legdict)
            fig_CS   .savefig(sv_fldr+"06_CS_wlg."+plot_format,**save_dict)
            fig_CD   .savefig(sv_fldr+"06_CD."    +plot_format,**save_dict)
            axs_CD.legend(fontsize=8.0,**legdict)
            fig_CD   .savefig(sv_fldr+"06_CD_wlg."+plot_format,**save_dict)
            fig_Cl   .savefig(sv_fldr+"06_Cl."    +plot_format,**save_dict)
            axs_Cl.legend(fontsize=8.0,**legdict)
            fig_Cl   .savefig(sv_fldr+"06_Cl_wlg."+plot_format,**save_dict)
            fig_Cm   .savefig(sv_fldr+"06_Cm."    +plot_format,**save_dict)
            axs_Cm.legend(fontsize=8.0,**legdict)
            fig_Cm   .savefig(sv_fldr+"06_Cm_wlg."+plot_format,**save_dict)
            fig_Cn   .savefig(sv_fldr+"06_Cn."    +plot_format,**save_dict)
            axs_Cn.legend(fontsize=8.0,**legdict)
            fig_Cn   .savefig(sv_fldr+"06_Cn_wlg."+plot_format,**save_dict)
            #
            fig_eg   .savefig(sv_fldr+"07_maxeig."+plot_format,**save_dict)
            axs_eg.legend(fontsize=8.0,**legdict)
            fig_eg   .savefig(sv_fldr+"07_maxeig_wlg."+plot_format,**save_dict)
            #
            fig_lg   .savefig(sv_fldr+"08_legend."+plot_format,**save_dict)
            #
            axs_bb.legend(fontsize=8.0,**cmpdict)
            fig_bb   .savefig(sv_fldr+"08_cg_v_beta."+plot_format,**save_dict)
            axs_bp.legend(fontsize=8.0,**cmpdict)
            fig_bp   .savefig(sv_fldr+"08_cg_v_phi."+plot_format,**save_dict)
            #
            fig_ps   .savefig(sv_fldr+"09_psidot."+plot_format,**save_dict)
            axs_ps.legend(fontsize=8.0,**legdict)
            fig_ps   .savefig(sv_fldr+"09_psidot_wlg."+plot_format,**save_dict)
            fig_wp   .savefig(sv_fldr+"10_p."+plot_format,**save_dict)
            fig_wq   .savefig(sv_fldr+"10_q."+plot_format,**save_dict)
            fig_wr   .savefig(sv_fldr+"10_r."+plot_format,**save_dict)
            if not(skip_DOC):
                for g in range(rows_DC):
                    for h in range(cols_DC):
                        F = "13_"+state_names[g]+"_"+ctrl_names[h]+"_DOC."
                        fig_DC[g][h].savefig(sv_fldr+F+plot_format,**save_dict)
                        if h == 0:
                            axs_DC[g][h].legend(fontsize=8.0,**legdict) # aileron
                            F = F[:-1] + "_wlg."
                            fig_DC[g][h].savefig(sv_fldr+F+plot_format,**save_dict)
            
            if show_plots:
                plt.close(fig_da)
                plt.close(fig_de)
                plt.close(fig_dB)
                plt.close(fig_ta)
                plt.close(fig_t2)
                plt.close(fig_vr)
                plt.close(fig_af)
                plt.close(fig_CL)
                plt.close(fig_CS)
                plt.close(fig_CD)
                plt.close(fig_Cl)
                plt.close(fig_Cm)
                plt.close(fig_Cn)
                plt.close(fig_eg)
                plt.close(fig_bb)
                plt.close(fig_bp)
                plt.close(fig_lg)
                plt.close(fig_th)
                plt.close(fig_wp)
                plt.close(fig_wq)
                plt.close(fig_wr)
                plt.close(fig_ps)
                for g in range(rows_DC):
                    for h in range(cols_DC):
                        if g != 6:
                            plt.close(fig_DC[g][h])
                plt.show()
            else:
                plt.close("all")
