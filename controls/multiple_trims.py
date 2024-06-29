import numpy as np
import json
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from controller_simulation import Aircraft#,monte_carlo_perturbations,run_single_simulation
from os import mkdir, rmdir, walk, remove, listdir
from linearization import linearization

if __name__ == "__main__":

    # filenames 
    base_file = "base_fs_in.json"
    bire_file = "bire_fs_in.json"

    # read in json to ensure no file changes while running
    base_dict = json.loads( open(base_file).read() )
    bire_dict = json.loads( open(bire_file).read() )
    
    # flight conditions
    flight_conditions = {
        "A1" : { "m" : 0.2 , "h" :     0., "V" : 222., "Re" :        0. },
        "T1" : { "m" : 0.2 , "h" :  1000., "V" : 222., "Re" : 15641000. },
        "T2" : { "m" : 0.19, "h" : 15000., "V" : 201., "Re" :  9919000. },
        "C1" : { "m" : 0.8 , "h" :  1000., "V" : 890., "Re" : 62563000. },
        "C2" : { "m" : 0.6 , "h" : 15000., "V" : 634., "Re" : 31324000. },
        "C3" : { "m" : 0.8 , "h" : 30000., "V" : 796., "Re" : 25828000. }
    }

    # settings 
    ## Continue from bire_fs_shss_T1_M02_H010_CGp10p00p00_B14
    run_bire = True # False # 
    run_sct  = False # True # 
    run_fs = True
    skip_run = True # False # 
    if run_sct: trim_bank_degs = [0.0] # np.linspace(0.0,75.0,num=16).tolist() # [10.0] # [60.0] # np.linspace(0.0,75.0,num=16).tolist() # 
    else: trim_beta_degs = [0.0] # np.linspace(0.0,16.0,num=9).tolist() # np.linspace(0.0,16.0,num=9).tolist() # [14.0,16.0] # [0.0] # 
    fc = "T1" # "C2" # "A1" # 
    cgshift = [0.0,0.0,0.0] # [0.5,0.0,0.0] # [1.0,0.0,0.0] # [0.5,0.0,0.0] # 
    include_compressibility =  True # False # 
    use_Anderson_corrections =  True # False # 
    include_stall =  True # False # 
    plot_negative_xcg = False # True # 
    plot_inverted_trims = False # True # 
    plot_alternate_trims = True # False # 
    #
    # other settings
    run_num = 1000 # 20 # 
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
    pdBm = "+" # "$/$" # 
    ndBm = "_" # "$-$" # "x" # 
    odBm = "o"
    odrm = "d"
    xcg_shade = lambda xcg : xcg*0.5 + (abs(xcg)>0.0)*0.25

    
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
    craftdict["initial"]["trim"]["climb_angle[deg]"] = 0.0
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
            run_file = run_name + ".json"
            print("\n\nrunning", run_name,"...")
            
            # open this file if it exists
            try: 
                run_dict = json.loads( open(folder + run_file).read() )
            except: run_dict = {}
            
            # pull in saved trim states
            x_trims=[]; u_trims=[]; CFM_trims=[]
            guess_trims=[]; final_i_trims=[]; Lin_trims = [];
            for case in run_dict:
                x_trims.append(run_dict[case]["x_trim_euler"])
                u_trims.append(run_dict[case]["u_trim"])
                CFM_trims.append(run_dict[case]["CFM_trim"])
                guess_trims.append(run_dict[case]["guess_trim"])
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

            # save to file (if we found more)
            if len(x_trims) > num_b4:
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
                    data_dict[case]["Linearized_system_trim"] = Lin_trims[j]
                # save
                with open(folder+run_file, "w") as f:
                    json.dump(data_dict, f, indent=4)
            else:
                print("\nnot saving to file",run_name,
                    ", no new trim sols found")
    
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
            # numpify
            trims[craft]["ind_var"] = np.array(trims[craft]["ind_var"])
            trims[craft]["dicts"] = np.array(trims[craft]["dicts"])
            # get sort indices
            sorter = np.argsort(trims[craft]["ind_var"])
            # sort
            trims[craft]["ind_var"] = trims[craft]["ind_var"][sorter]
            trims[craft]["dicts"] = trims[craft]["dicts"][sorter]

        # width in inches
        width = 4.0
        scale_font_size = 3.25/width

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

        # initialize plots
        plot_dict = dict(figsize=(width,3.0),dpi=300.0, # sharex=True,
            constrained_layout=True)
        fig_da,axs_da = plt.subplots(1,1,**plot_dict)
        fig_de,axs_de = plt.subplots(1,1,**plot_dict)
        fig_dB,axs_dB = plt.subplots(1,1,**plot_dict)
        fig_ta,axs_ta = plt.subplots(1,1,**plot_dict)
        fig_vr,axs_vr = plt.subplots(1,1,**plot_dict)
        fig_af,axs_af = plt.subplots(1,1,**plot_dict)
        # fig_FM,axs_FM = plt.subplots(3,2,**plot_dict)
        axs = [axs_da,axs_de,axs_dB,axs_ta,axs_vr,axs_af]
        if run_sct: k = 1
        else:       k = 2

        # plot points
        rest_dict = dict(ms=3.5,mew=0.75,fillstyle="none",ls="none")
        for craft in trims:
            cname,cgs = craft.split("_")
            xcg = float(cgs.replace("p","_+").replace("m","_-").split("_")[1])/10.
            for i in range(len(trims[craft]["dicts"])):
                for trim_sol in trims[craft]["dicts"][i]:
                    sol = trims[craft]["dicts"][i][trim_sol]
                    ivr = trims[craft]["ind_var"][i]
                    # setup marker style
                    kdict = {}
                    # craft type
                    if cname == "bire":
                        if np.rad2deg(sol["u_trim"][2]) > 5.0:
                            kdict |= dict(marker = pdBm)
                        elif np.rad2deg(sol["u_trim"][2]) < -5.0:
                            kdict |= dict(marker = ndBm) # 
                        else:
                            kdict |= dict(marker = odBm)
                    else              : kdict |= dict(marker = odrm)
                    # cg loc
                    kdict |= dict(color = str(xcg_shade(xcg)))
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
                    kdict |= rest_dict
                    # PLOTS!
                    if abs(np.rad2deg(sol["u_trim"][2])) > 5.0 and not(plot_alternate_trims):
                        continue
                    if xcg < 0.0 and not(plot_negative_xcg):
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
                    # # CFM
                    # axs_FM[0,0].plot(ivr,sol["CFM_trim"][0],**kdict)
                    # axs_FM[1,0].plot(ivr,sol["CFM_trim"][1],**kdict)
                    # axs_FM[2,0].plot(ivr,sol["CFM_trim"][2],**kdict)
                    # axs_FM[0,1].plot(ivr,sol["CFM_trim"][3],**kdict)
                    # axs_FM[1,1].plot(ivr,sol["CFM_trim"][4],**kdict)
                    # axs_FM[2,1].plot(ivr,sol["CFM_trim"][5],**kdict)
        
        # other plot params
        if trim_type == "sct": 
            xlabel = r"Bank angle, $\phi$ [$^\circ$]"
            vrylbl = r"Sideslip angle, $\beta$ [$^\circ$]"
        else: 
            xlabel = r"Sideslip angle, $\beta$ [$^\circ$]"
            vrylbl = r"Bank angle, $\phi$ [$^\circ$]"
        [ax.set_xlabel(xlabel) for ax in axs]
        axs[0].set_ylabel(r"Aileron, $\delta_a$ [$^\circ$]")
        axs[1].set_ylabel(r"Stabilator, $\delta_e^B$ [$^\circ$]")
        axs[2].set_ylabel(r"$\delta_B$/$\delta_r$ [$^\circ$]")
        axs[3].set_ylabel(r"Throttle, $\tau$ [per-unit]")
        axs[4].set_ylabel(vrylbl)
        axs[5].set_ylabel(r"Angle of attack, $\alpha$ [$^\circ$]")
        # legend
        sp = [
            Line2D([0], [0],color=str(xcg_shade(0.0)),marker=odBm,**rest_dict),
            Line2D([0], [0],color=str(xcg_shade(0.0)),marker=odrm,**rest_dict),
            Line2D([0], [0],color=str(xcg_shade(0.5)),marker=odBm,**rest_dict),
            Line2D([0], [0],color=str(xcg_shade(1.0)),marker=odBm,**rest_dict),
            Line2D([0], [0],color=str(xcg_shade(0.0)),marker=pdBm,**rest_dict),
            Line2D([0], [0],color=str(xcg_shade(0.0)),marker=ndBm,**rest_dict)
        ]
        lbls = [
            "BIRE",
            "base",
            "$\Delta x_{cg} = 0.5$",
            "$\Delta x_{cg} = 1.0$",
            "$\delta_B$ >> 0",
            "$\delta_B$ << 0"
        ]
        legdict = dict(handles=sp,labels=lbls,loc=(1.0,0.0),
            borderpad=0.1,handletextpad=0.0)
        axs[0].legend(**legdict)
        axs[1].legend(**legdict)
        axs[2].legend(**legdict)
        axs[3].legend(**legdict)
        axs[4].legend(**legdict)
        axs[5].legend(**legdict)
        
        plt.show()
