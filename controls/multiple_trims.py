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
    }

    # settings 
    ## Continue from bire_fs_shss_T1_M02_H010_CGp10p00p00_B14
    run_bire = True # False # 
    run_sct  = True # False # 
    run_fs = True
    skip_run = False # True # 
    skip_DOC = False # True # 
    if run_sct: trim_bank_degs = [30.0] # np.linspace(0.0,75.0,num=16).tolist() # np.linspace(0.0,60.0,num=13).tolist() # [0.0] # [10.0] # [60.0] # np.linspace(0.0,75.0,num=16).tolist() # 
    else: trim_beta_degs = [6.0] # np.linspace(0.0,16.0,num=9).tolist() # np.linspace(0.0,16.0,num=9).tolist() # [14.0,16.0] # [0.0] # 
    trim_climb_deg = 0.0 # 10.0 # 
    fc = "C2" # "T1" # "F1" # "U1" # "A1" # 
    cgshift = [0.0,0.0,0.0] # [1.0,0.0,0.0] # [0.5,0.0,0.0] # [0.5,0.0,0.0] # 
    include_compressibility =  True # False # 
    use_Anderson_corrections =  True # False # 
    include_stall =  True # False # 
    plotting_xcgs = [0.0,0.5,1.0]
    plotting_gammas = [0.0]
    plot_inverted_trims = False # True # 
    plot_alternate_trims = True # False # 
    plot_base_DOC = True # False # 
    show_plots = False # True # 
    plot_format = "pdf" # "png" # 
    plot_transparent = True if plot_format == "pdf" else False # False # True # 
    plot_dark = True # False # 
    skip_save_if_not_new = False # True # 
    #
    # other settings
    run_num = 10 # 1000 # 30 # 
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
    # xcg_shade = lambda xcg : "k" if xcg == 0.0 else ("r" if xcg == 0.5 else ("b" if xcg == 1.0 else "y"))

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
                if "guess_bounds" in run_dict[case]:
                    guess_bounds.append(run_dict[case]["guess_bounds"])
                final_i_trims.append(run_dict[case]["final_i_trim"])
                Lin_trims.append(run_dict[case]["Linearized_system_trim"])
            
            num_b4 = len(x_trims)

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
                    if run_sct:
                        bgsj = bgs[trimsj]
                    else:
                        pgsj = pgs[trimsj]
                    ugsj = ugs[trimsj]
                    #
                    a_max = np.max(agsj); a_max = np.rad2deg(a_max)
                    a_min = np.min(agsj); a_min = np.rad2deg(a_min)
                    if run_sct:
                        b_max = np.max(bgsj); b_max = np.rad2deg(b_max)
                        b_min = np.min(bgsj); b_min = np.rad2deg(b_min)
                    else:
                        p_max = np.max(pgsj); p_max = np.rad2deg(p_max)
                        p_min = np.min(pgsj); p_min = np.rad2deg(p_min)
                    u_max = np.max(ugsj,axis=0); u_max[0:3] = np.rad2deg(u_max[0:3])
                    u_min = np.min(ugsj,axis=0); u_min[0:3] = np.rad2deg(u_min[0:3])
                else:
                    a_max = -1.3e+10; a_min = +1.3e+10
                    if run_sct:
                        b_max = -1.3e+10; b_min = +1.3e+10
                    else:
                        p_max = -1.3e+10; p_min = +1.3e+10
                    u_max = np.array([-1.3e+10]*4)*1.0
                    u_min = np.array([+1.3e+10]*4)*1.0

                if len(guess_bounds) >= j + 1:
                    a_min = min(a_min,guess_bounds[j]["alpha[deg]_[min,max]"][0])
                    a_max = max(a_max,guess_bounds[j]["alpha[deg]_[min,max]"][1])
                    if run_sct:
                        b_min = min(b_min,guess_bounds[j]["beta[deg]_[min,max]"][0])
                        b_max = max(b_max,guess_bounds[j]["beta[deg]_[min,max]"][1])
                    else:
                        p_min = min(p_min,guess_bounds[j]["phi[deg]_[min,max]"][0])
                        p_max = max(p_max,guess_bounds[j]["phi[deg]_[min,max]"][1])
                    u_min[0] = min(u_min[0],guess_bounds[j]["da[deg]_[min,max]"][0])
                    u_max[0] = max(u_max[0],guess_bounds[j]["da[deg]_[min,max]"][1])
                    u_min[1] = min(u_min[1],guess_bounds[j]["de[deg]_[min,max]"][0])
                    u_max[1] = max(u_max[1],guess_bounds[j]["de[deg]_[min,max]"][1])
                    u_min[2] = min(u_min[2],guess_bounds[j]["u3[deg]_[min,max]"][0])
                    u_max[2] = max(u_max[2],guess_bounds[j]["u3[deg]_[min,max]"][1])
                    u_min[3] = min(u_min[3],guess_bounds[j]["tau[pu]_[min,max]"][0])
                    u_max[3] = max(u_max[3],guess_bounds[j]["tau[pu]_[min,max]"][1])
                else:
                    guess_bounds.append({})
                
                # add to guess bounds dict
                guess_bounds[j]["alpha[deg]_[min,max]"] = [a_min,a_max]
                if run_sct:
                    guess_bounds[j]["beta[deg]_[min,max]"] = [b_min,b_max]
                else:
                    guess_bounds[j]["phi[deg]_[min,max]"] = [p_min,p_max]
                guess_bounds[j]["da[deg]_[min,max]"] = [u_min[0],u_max[0]]
                guess_bounds[j]["de[deg]_[min,max]"] = [u_min[1],u_max[1]]
                guess_bounds[j]["u3[deg]_[min,max]"] = [u_min[2],u_max[2]]
                guess_bounds[j]["tau[pu]_[min,max]"] = [u_min[3],u_max[3]]

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
                    data_dict[case]["guess_trim"] = guess_trims[j]
                    data_dict[case]["guess_bounds"] = guess_bounds[j]
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
                for key in keys:
                    gls = guess_bounds[j][key]
                    print("    {:>20s} : {:> 5.2f},{:> 5.2f}".format(
                        key,gls[0],gls[1]))
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
                name = file_split[0] + "_" + file_split[6]
                if name not in trims:
                    trims[name] = {}
                    trims[name]["ind_var"] = []
                    trims[name]["dicts"] = []
                
                # read in info
                path = folder + filename
                trims[name]["dicts"].append(json.loads( open(path).read() ))
                trims[name]["ind_var"].append(float(file_split[7][1:]))
                
                # print(file_split, name)
                # calculate alpha and beta and save to dicts
                sols = trims[name]["dicts"][-1]
                for sol in sols:
                    x = sols[sol]["x_trim_euler"]
                    a = np.arctan2(x[2],x[0])
                    V = (x[0] * x[0] + x[1] * x[1] + x[2] * x[2])**0.5
                    b = np.arcsin(x[1]/V)
                    trims[name]["dicts"][-1][sol]["angles"] = [a,b,x[9]]
        
        # sort by ind_var
        for craft in trims:
            trims[craft]["min_ind"] = []
            # numpify
            trims[craft]["ind_var"] = np.array(trims[craft]["ind_var"])
            trims[craft]["dicts"] = np.array(trims[craft]["dicts"])
            # get sort indices
            sorter = np.argsort(trims[craft]["ind_var"])
            # sort
            trims[craft]["ind_var"] = trims[craft]["ind_var"][sorter]
            trims[craft]["dicts"] = trims[craft]["dicts"][sorter]

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
                        # if craft[:4] == "bire":
                        #     continue
                        # if craft[4:] != "_CGp10p00p00":
                        #     continue
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
        width = 4.0
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
            plt.rcParams["font.size"] = 8.0#*scale_font_size
            plt.rcParams["axes.labelsize"] = 8.0#*scale_font_size
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
            plt.rcParams['figure.max_open_warning'] = 50

            # initialize plots
            plot_dict = dict(figsize=(width,3.0),dpi=300.0, # sharex=True,
                constrained_layout=True)
            fig_da,axs_da = plt.subplots(1,1,**plot_dict)
            fig_de,axs_de = plt.subplots(1,1,**plot_dict)
            fig_dB,axs_dB = plt.subplots(1,1,**plot_dict)
            fig_ta,axs_ta = plt.subplots(1,1,**plot_dict)
            fig_vr,axs_vr = plt.subplots(1,1,**plot_dict)
            fig_af,axs_af = plt.subplots(1,1,**plot_dict)
            fig_CD,axs_CD = plt.subplots(1,1,**plot_dict)
            fig_eg,axs_eg = plt.subplots(1,1,**plot_dict)
            rows_DC = 9; cols_DC = 4
            fig_DC = [[None for _ in range(cols_DC)] for _ in range(rows_DC)]
            axs_DC = [[None for _ in range(cols_DC)] for _ in range(rows_DC)]
            for i in range(rows_DC):
                for j in range(cols_DC):
                    if j == 0: shares ={}
                    else: shares ={"sharex":axs_DC[i][0],"sharey":axs_DC[i][0]}
                    fig_DC[i][j],axs_DC[i][j] = plt.subplots(1,1,**shares,**plot_dict)
            axs = [axs_da,axs_de,axs_dB,axs_ta,axs_vr,axs_af]
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
            axs_vr.grid(which="major",lw=grid_lw,ls="-",c=grid_color)
            axs_af.grid(which="major",lw=grid_lw,ls="-",c=grid_color)
            axs_CD.grid(which="major",lw=grid_lw,ls="-",c=grid_color)
            axs_eg.grid(which="major",lw=grid_lw,ls="-",c=grid_color)
            for i in range(rows_DC):
                for j in range(cols_DC):
                    axs_DC[i][j].grid(which="major",lw=grid_lw,ls="-",
                        c=grid_color)

            # plot points
            rest_dict = dict(ms=3.5,mew=0.75,fillstyle="none",ls="none")
            for craft in trims:
                cname,cgs = craft.split("_")
                xcg = float(cgs.replace("p","_+").replace("m","_-").split("_")[1])/10.
                for i in range(len(trims[craft]["dicts"])):
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
                        if xcg == -1.0:
                            kdict["color"] = "b"
                        elif xcg == -0.5:
                            kdict["color"] = "r"
                        elif xcg == -0.2:
                            kdict["color"] = "y"
                        elif xcg == -0.1:
                            kdict["color"] = "g"
                        if np.rad2deg(abs(sol["angles"][k])) > 90.0:
                            kdict["color"] = "m"
                        # others
                        kdict = {**kdict, **rest_dict}
                        # PLOTS!
                        if cname == "bire" and j != trims[craft]["min_ind"][i] \
                            and not(plot_alternate_trims):
                            continue
                        if xcg not in plotting_xcgs:
                            continue
                        if np.rad2deg(abs(sol["angles"][k])) > 90.0 and not(plot_inverted_trims):
                            continue
                        # aileron
                        axs[0].plot(ivr,np.rad2deg(sol["u_trim"][0]),**kdict)
                        # elevator
                        axs[1].plot(ivr,np.rad2deg(sol["u_trim"][1]),**kdict)
                        # rudder
                        axs[2].plot(ivr,np.rad2deg(sol["u_trim"][2]),**kdict)
                        # throttle
                        axs[3].plot(ivr,           sol["u_trim"][3] ,**kdict)
                        # sideslip / bank
                        axs[4].plot(ivr,np.rad2deg(sol["angles"][k]),**kdict)
                        # aoa
                        axs[5].plot(ivr,np.rad2deg(sol["angles"][0]),**kdict)
                        #
                        # CD
                        axs_CD.plot(ivr,sol["CFM_trim"][2],**kdict)
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
            axs[2].set_ylabel(r"Tail Rotation/Rudder ($\delta_B$/$\delta_r$), deg")
            axs[3].set_ylabel(r"Throttle ($\tau$), per-unit")
            axs[4].set_ylabel(vrylbl)
            axs[5].set_ylabel(r"Angle of attack ($\alpha$), deg")
            axs_CD.set_xlabel(xlabel)
            axs_CD.set_ylabel(r"Drag coefficient ($C_D$)")
            axs_eg.set_ylabel(xlabel)
            axs_eg.set_xlabel(r"$\max \left( \operatorname{real} \left( " + \
                r"\lambda \right) \right)$, 1/sec")
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
            legdict = dict(handles=sp,labels=lbls,loc=(1.0,0.0),
                borderpad=0.1,handletextpad=0.0)
            axs[0].legend(**legdict)
            axs[1].legend(**legdict)
            axs[2].legend(**legdict)
            axs[3].legend(**legdict)
            axs[4].legend(**legdict)
            axs[5].legend(**legdict)
            axs_CD.legend(**legdict)
            axs_eg.legend(**legdict)
            if plot_base_DOC:
                legDOCdict = legdict
            else:
                legDOCdict = dict(handles=sp[2:],labels=lbls[2:],loc=(1.0,0.0),
                    borderpad=0.1,handletextpad=0.0)
            for g in range(rows_DC):
                for h in range(cols_DC):
                    axs_DC[g][h].legend(**legDOCdict)

            # save plots
            save_dict = dict(transparent=plot_transparent,dpi=300.0)
            sv_fldr = "plots" if plot_type == "default" else "plots_inv"
            sv_fldr = folder + sv_fldr + "/"
            # if plot type is different than previous, remove them
            for filename in listdir(sv_fldr[:-1]):
                if filename.split(".")[-1] != plot_format:
                    remove(sv_fldr+filename)
            nm = "beta" if trim_type == "sct" else "phi"
            fig_da   .savefig(sv_fldr+"00_da."    +plot_format,**save_dict)
            fig_de   .savefig(sv_fldr+"01_de."    +plot_format,**save_dict)
            fig_dB   .savefig(sv_fldr+"02_dB."    +plot_format,**save_dict)
            fig_ta   .savefig(sv_fldr+"03_tau."   +plot_format,**save_dict)
            fig_vr   .savefig(sv_fldr+"04_"+nm+"."+plot_format,**save_dict)
            fig_af   .savefig(sv_fldr+"05_alpha." +plot_format,**save_dict)
            fig_CD   .savefig(sv_fldr+"06_CD."    +plot_format,**save_dict)
            fig_eg   .savefig(sv_fldr+"07_maxeig."+plot_format,**save_dict)
            if not(skip_DOC):
                for g in range(rows_DC):
                    for h in range(cols_DC):
                        F = "13_"+state_names[g]+"_"+ctrl_names[h]+"_DOC."
                        fig_DC[g][h].savefig(sv_fldr+F+plot_format,**save_dict)
            
            if show_plots:
                plt.close(fig_da)
                plt.close(fig_de)
                plt.close(fig_dB)
                plt.close(fig_ta)
                plt.close(fig_vr)
                plt.close(fig_af)
                plt.close(fig_CD)
                plt.close(fig_eg)
                for g in range(rows_DC):
                    for h in range(cols_DC):
                        if g != 6:
                            plt.close(fig_DC[g][h])
                plt.show()
            else:
                plt.close("all")
