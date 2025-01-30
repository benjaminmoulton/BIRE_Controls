import numpy as np
import json
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm, SymLogNorm, ListedColormap
import mpl_toolkits.mplot3d.axes3d as ax3
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from numpy import sign, matmul as mm
from datetime import datetime
import control as co
from scipy.linalg import block_diag
from scipy.integrate import ode, odeint
from scipy.interpolate import interp1d, interpn
from scipy.optimize import curve_fit,minimize,minimize_scalar,newton
from scipy.io import savemat, loadmat
from scipy.signal import tf2zpk as scipy_tf2zpk
from math import pi, sin, cos, tan, exp, asin, atan, atan2
from std_atm import stdatm_english
from quat import quat_mult, euler_2_quat, quat_2_euler, quat_norm, body_2_fixed, fixed_2_body, eulerdot_2_quatdot, quatdot_2_eulerdot
from linearization import linearization as lin,Anderson_correction_der_coeff,Anderson_correction_der_M

from controller_simulation import Aircraft,run_single_simulation, \
    monte_carlo_perturbations, report_latex, report_eigprops, rep2D,BIREAero

from os.path import isfile


if __name__ == "__main__":

    # filenames 
    bire_fs_file = "bire_fs_in.json"

    # read in json to ensure no file changes while running
    bire_fs_dict = json.loads( open(bire_fs_file).read() )

    # initialize
    # trim 
    compr = True # False # 
    stall = True # False # 
    fitthrust = True # False # 
    phi_trim = 0.0 # 30.0 # 10.0 # 
    cgshift = [0.0, 0.0, 0.0] # [1.0, 0.0, 0.0] # [0.5, 0.0, 0.0] # 
    subfolder_end = "" # "_m" # "_p" # 

    # settings
    # run_cases = False # True # 
    folder_base = "determinant_data/"
    sgn = "p" if cgshift[0] >= 0.0 else "m"
    folder = folder_base + "SCT_cgX{:s}{:>02d}".format(sgn,int(cgshift[0]*10.0))
    sgn = "p" if phi_trim >= 0.0 else "m"
    folder = folder + "_P{:s}{:>02d}".format(sgn,int(phi_trim))
    folder = folder + subfolder_end + "/"
    overwrite_data = True # False # 
    print_data_after_run = False # True # 
    show_plots = False # True # 
    save_plots = True # False # 
    plot_type = "png" # "pdf" # 
    plot_transparent = True if plot_type == "pdf" else False
    plot_after_run = False # True # 

    #
    bire_fs_dict["simulation"]["include_compressibility"] = compr
    bire_fs_dict["simulation"]["include_stall"] = stall
    bire_fs_dict["simulation"]["use_fitted_thrust_model"] = fitthrust
    bire_fs_dict["aircraft"]["CG_shift[ft]"] = cgshift
    bire_fs_dict["initial"]["mach"] = 0.6
    bire_fs_dict["initial"]["altitude[ft]"] = 15000.0
    bire_fs_dict["initial"]["trim"]["bank_angle[deg]"] = phi_trim
    bire_fs_dict["initial"]["trim"]["type"] = "sct"
    bire_fs_dict["initial"]["type"] = "trim"
    bire_fs_dict["initial"]["trim_guess"] = {}
    if   subfolder_end == "_m":
        bire_fs_dict["initial"]["trim_guess"]["elevator[deg]"] = -25.0
        bire_fs_dict["initial"]["trim_guess"]["BIRE[deg]"] = -70.0
    elif subfolder_end == "_p":
        bire_fs_dict["initial"]["trim_guess"]["elevator[deg]"] = -25.0
        bire_fs_dict["initial"]["trim_guess"]["BIRE[deg]"] = 70.0
    else: # ""
        bire_fs_dict["initial"]["trim_guess"]["elevator[deg]"] = 20.0
        bire_fs_dict["initial"]["trim_guess"]["BIRE[deg]"] = 0.0
    bire = Aircraft(bire_fs_dict)
    x0 = bire.x_trim_euler
    u0 = bire.u_trim
    print("cg x, ft  =",cgshift[0])
    print("Bank, deg =",phi_trim)
    print("x0 =",x0)
    print("u0 =",u0)
    # print(bire.inertia_model.W)
    # print(bire.cgshift)
    # bire._report_trim_solution()
    # # build linearized system
    _,Lin_Model = bire._build_controller(x0,u0,save_matrices=False,mrrr=[0,1,2,6,7,8,9,10,11],
        mrrc=[3],drop_actrs=True,
        include_stall_derivatives=True,
        include_altitude_derivatives=True,
        run_freq=False,report=False)
    rep2D(Lin_Model.B_min,"B")
    S = np.diag([bire.s_da,bire.s_de,bire.s_dr])
    Sdet = np.linalg.det(S)
    print("S   det trim =", Sdet)
    print("B*S det trim =", np.linalg.det(Lin_Model.B_min)*Sdet)

    # run settings
    dVshift = 0.00; dVlim =  50.0 # dVlim = 400.0 # 
    dashift = 0.00; dalim =  15.0
    dbshift = 0.00; dblim =  15.0 # dblim =  60.0 # 
    dpshift = 0.00; dplim =  90.0 # dplim = 360.0 # 
    dqshift = 0.00; dqlim =  40.0
    drshift = 0.00; drlim =  20.0 # drlim =  80.0 # 
    dzshift = 0.00; dzlim = 500.0
    dPshift = 0.00; dPlim =  90.0
    dTshift = 0.00; dTlim =  45.0
    dAshift = 0.00; dAlim =  21.5 # dAlim =  20.0 # 
    dEshift = 0.00; dElim =  25.0 # dElim =  20.0 # 
    dBshift = 0.00; dBlim =  90.0 # dBlim =  60.0 # 
    #
    #
    report_every = 500 # 2500 # 
    hnum = 50 # 100 # 200 # 150 # 5 # 

    # cases to run
    lineup = ["V","a","b","p","q","r","z","P","T","A","E","B"]
    dictup    = {lineup[i]:i for i in range(len(lineup))}
    dictupinv = {i:lineup[i] for i in range(len(lineup))}
    run_cases = [
        # ("V","a"),("V","b"),("V","p"),("V","q"),("V","r"),("V","z"),
        # ("a","b"),("a","p"),("a","q"),("a","r"),("a","z"),
        # ("b","p"),("b","q"),("b","r"),("b","z"),
        # ("p","q"),("p","r"),("p","z"),
        # ("q","r"),("q","z"),
        # ("r","z"),
        # #
        # ("V","A"),("V","E"),("V","B"),
        # ("a","A"),("a","E"),("a","B"),
        # ("b","A"),("b","E"),("b","B"),
        # ("p","A"),("p","E"),("p","B"),
        # ("q","A"),("q","E"),("q","B"),
        # ("r","A"),("r","E"),("r","B"),
        # ("z","A"),("z","E"),("z","B"),
        # ("A","E"),("A","B"),
        # ("E","B"),
        # #
        # ("P","T"),
        # ("a","q"),
        # ("V","b"),
        # ("p","r"),
        # ("A","B"),
    ]
    if len(run_cases) > 0 and not(plot_after_run):
        plot_cases = []
    elif cgshift[0] == 0.0 and phi_trim ==  0.0:
        plot_cases = [
            # zero plots (interact a,q,E,B, and others)
            ("a","q"),("a","E"),("a","B"),
            ("b","B"),
            ("q","E"),("q","B"),
            ("r","B"),
            ("A","B"),
        ]
        # plot_cases = [
        #     # ("V","b"),
        #     # ("p","r"),
        #     ("A","B"),
        # ]
    elif cgshift[0] == 0.0 and phi_trim == 10.0:
        plot_cases = [
            # zero plots (interact a,q,E,B, and others)
            ("a","q"),("a","E"),("a","B"),
            ("b","q"),("b","B"),
            ("q","E"),("q","B"),
            ("r","B"),
            ("A","B"),
        ]
    elif cgshift[0] == 0.0 and phi_trim == 30.0 and subfolder_end == "":
        plot_cases = [
            # zero plots (interact a,q,E,B, and others)
            ("a","q"),("a","E"),("a","B"),
            ("b","q"),("b","B"),
            ("q","E"),("q","B"),
            ("r","B"),
            ("A","B"),
        ]
    elif cgshift[0] == 0.0 and phi_trim == 30.0 and subfolder_end == "_m":
        plot_cases = [
            # zero plots (interact a,b,q,E,B, and others)
            ("a","b"),("a","q"),("a","E"),("a","B"),
            ("b","q"),("b","E"),("b","B"),
            ("q","r"),("q","E"),("q","B"),
            ("r","B"),
            ("A","B"),
        ]
    elif cgshift[0] == 0.5 and phi_trim ==  0.0:
        plot_cases = [
            # zero plots (interact a,q,E,B, and others)
            ("a","q"),("a","E"),("a","B"),
            ("b","B"),
            ("q","E"),("q","B"),
            ("r","B"),
            ("A","B"), # keep this plot for cg comparison
        ]
    elif cgshift[0] == 0.5 and phi_trim == 10.0:
        plot_cases = [
            # zero plots (interact a,q,E,B, and others)
            ("V","q"),
            ("a","q"),("a","E"),("a","B"),
            ("b","B"),
            ("q","E"),("q","B"),
            ("r","B"),
        ]
    elif cgshift[0] == 0.5 and phi_trim == 30.0 and subfolder_end == "":
        plot_cases = [
            # zero plots (interact a,q,E,B, and others)
            ("V","q"),
            ("a","q"),("a","E"),("a","B"),
            ("b","B"),
            ("q","E"),("q","B"),
            ("r","B"),
        ]
    elif cgshift[0] == 1.0 and phi_trim ==  0.0:
        plot_cases = [
            # zero plots (interact a,q,E,B, and others)
            ("V","q"),
            ("a","q"),("a","E"),("a","B"),
            ("b","B"),
            ("q","E"),("q","B"),
            ("r","B"), # keep this plot for cg comparison
            ("A","B"), # keep this plot for cg comparison
        ]
    elif cgshift[0] == 1.0 and phi_trim == 10.0:
        plot_cases = [
            # zero plots (interact a,q,E,B, and others)
            # ("V","q"),
            # ("a","q"),("a","E"),("a","B"),
            # ("b","B"),
            # ("q","E"),("q","B"),
            ("V","q"),
            ("a","q"),("a","E"),("a","B"),
            ("b","B"),
            ("q","E"),("q","B"),
        ]
    elif cgshift[0] == 1.0 and phi_trim == 30.0 and subfolder_end == "":
        plot_cases = [
            # zero plots (interact a,q,E,B, and others)
            # ("V","q"),
            # ("a","q"),
            # ("b","B"),
            # ("q","E"),("q","B"),
            ("V","q"),
            ("a","q"),("a","E"),("a","B"),
            ("b","B"),
            ("q","E"),("q","B"),
        ]
    else:
        plot_cases = [
            # ("V","a"),("V","b"),("V","p"),("V","q"),("V","r"),("V","z"),
            # ("a","b"),("a","p"),("a","q"),("a","r"),("a","z"),
            # ("b","p"),("b","q"),("b","r"),("b","z"),
            # ("p","q"),("p","r"),("p","z"),
            # ("q","r"),("q","z"),
            # ("r","z"),
            # #
            # ("V","A"),("V","E"),("V","B"),
            # ("a","A"),("a","E"),("a","B"),
            # ("b","A"),("b","E"),("b","B"),
            # ("p","A"),("p","E"),("p","B"),
            # ("q","A"),("q","E"),("q","B"),
            # ("r","A"),("r","E"),("r","B"),
            # ("z","A"),("z","E"),("z","B"),
            # ("A","E"),("A","B"),
            # ("E","B"),
            #
            # ones that oughtn't have a line
            ("V","b"),("V","p"),("V","r"),("V","z"),("V","A"),("V","B"),
            ("b","p"),("b","r"),("b","z"),("b","A"),
            ("p","r"),("p","z"),("p","A"),("p","B"),
            ("r","z"),("r","A"),
            ("z","A"),("z","B"),
            ("A","B"),
            # ones that oughtn't have more than one straight line
            ("V","a"),("V","q"),("V","E"),
            ("a","b"),("a","p"),("a","r"),("a","z"),("a","A"),
            ("b","q"),("b","E"),
            ("p","q"),("p","E"),
            ("q","r"),("q","z"),("q","A"),
            ("r","E"),
            ("z","E"),
            ("A","E"),
            # ones that prolly have some weird stuff
            ("a","q"),
            ("b","B"),
            ("q","E"),("q","B"),
            ("r","B"),
            ("E","B"),
            # # 
            # #
            # # testing plots
            # ("V","b"),
            # # 
        ]
    # Notes
    # xcg 0.0, varying bank angles
    # ("r","B") plot changes after 10 deg bank. Like inverse and Yaxis mirror
    #     multi trim? (trend follow neg defl (_m)?)
    # ("A","B") plot changes after 10 deg bank. Like inverse and Yaxis mirror
    # ("b","q") comparing 30 deg m and 30 deg n0 plots. 90 deg rotation
    # ("r","B") comparing 30 deg m and 30 deg n0 plots. Like inverse and Yaxis mirror
    # ("A","B") comparing 30 deg m and 30 deg n0 plots. very different
    
    for case in run_cases:
        # report
        print("running case",case)
        # determine numbers
        dVnum = hnum*2+1 if "V" in case else 1
        danum = hnum*2+1 if "a" in case else 1
        dbnum = hnum*2+1 if "b" in case else 1
        dpnum = hnum*2+1 if "p" in case else 1
        dqnum = hnum*2+1 if "q" in case else 1
        drnum = hnum*2+1 if "r" in case else 1
        dznum = hnum*2+1 if "z" in case else 1
        dPnum = hnum*2+1 if "P" in case else 1
        dTnum = hnum*2+1 if "T" in case else 1
        dAnum = hnum*2+1 if "A" in case else 1
        dEnum = hnum*2+1 if "E" in case else 1
        dBnum = hnum*2+1 if "B" in case else 1

        # values to run
        dVs =            np.linspace(-dVlim,dVlim,num=dVnum) + dVshift
        das = np.deg2rad(np.linspace(-dalim,dalim,num=danum) + dashift)
        dbs = np.deg2rad(np.linspace(-dblim,dblim,num=dbnum) + dbshift)
        dps = np.deg2rad(np.linspace(-dplim,dplim,num=dpnum) + dpshift)
        dqs = np.deg2rad(np.linspace(-dqlim,dqlim,num=dqnum) + dqshift)
        drs = np.deg2rad(np.linspace(-drlim,drlim,num=drnum) + drshift)
        dzs =            np.linspace(-dzlim,dzlim,num=dznum) + dzshift
        dPs = np.deg2rad(np.linspace(-dPlim,dPlim,num=dPnum) + dPshift)
        dTs = np.deg2rad(np.linspace(-dTlim,dTlim,num=dTnum) + dTshift)
        dAs = np.deg2rad(np.linspace(-dAlim,dAlim,num=dAnum) + dAshift)
        dEs = np.deg2rad(np.linspace(-dElim,dElim,num=dEnum) + dEshift)
        dBs = np.deg2rad(np.linspace(-dBlim,dBlim,num=dBnum) + dBshift)

        # print(dVs,das,dbs,dps,dqs,drs,dzs,dPs,dTs)

        if dVnum == 1: dVs = np.array([0.0])
        if danum == 1: das = np.array([0.0])
        if dbnum == 1: dbs = np.array([0.0])
        if dpnum == 1: dps = np.array([0.0])
        if dqnum == 1: dqs = np.array([0.0])
        if drnum == 1: drs = np.array([0.0])
        if dznum == 1: dzs = np.array([0.0])
        if dPnum == 1: dPs = np.array([0.0])
        if dTnum == 1: dTs = np.array([0.0])
        if dAnum == 1: dAs = np.array([0.0])
        if dEnum == 1: dEs = np.array([0.0])
        if dBnum == 1: dBs = np.array([0.0])

        dets = np.zeros((dVnum,danum,dbnum,dpnum,dqnum,drnum,dznum,dPnum,dTnum,
             dAnum,dEnum,dBnum))

        print("running...")
        print("# cases =",("{:>6s}, "*12).format("dVnum","danum","dbnum",
            "dpnum","dqnum","drnum",
            "dznum","dPnum","dTnum",
            "dAnum","dEnum","dBnum"))
        print("         ",("{:>6d}, "*12).format(dVnum,danum,dbnum,dpnum,
            dqnum,drnum,dznum,dPnum,dTnum,dAnum,dEnum,dBnum))

        counter = 1
        totalnum = dVnum*danum*dbnum*dpnum*dqnum*drnum*dznum*dPnum*dTnum*dAnum*dEnum*dBnum
        x = x0*1.0
        u = u0*1.0
        vx_trim = x0[0]*1.0; vy_trim = x0[1]*1.0; vz_trim = x0[2]*1.0
        V_trim = (vx_trim**2.0 + vy_trim**2.0 + vz_trim**2.0)**0.5
        a_trim = atan2(vz_trim,vx_trim)
        b_trim = asin(vy_trim/V_trim)
        for Ri in range(dVnum):
            for Rj in range(danum):
                for Rk in range(dbnum):
                    Vnew = V_trim + dVs[Ri]
                    anew = a_trim + das[Rj]
                    bnew = b_trim + dbs[Rk]
                    # print(Vnew,np.rad2deg(anew),np.rad2deg(bnew))
                    x[0] = Vnew*cos(anew)*cos(bnew)
                    x[1] = Vnew*sin(bnew)
                    x[2] = Vnew*sin(anew)*cos(bnew)
                    for Rl in range(dpnum):
                        x[3] = x0[3] + dps[Rl]
                        for Rm in range(dqnum):
                            x[4] = x0[4] + dqs[Rm]
                            for Rn in range(drnum):
                                x[5] = x0[5] + drs[Rn]
                                for Ro in range(dznum):
                                    x[8] = x0[8] + dzs[Ro]
                                    for Rp in range(dPnum):
                                        x[9] = x0[9] + dPs[Rp]
                                        for Rq in range(dTnum):
                                            x[10] = x0[10] + dTs[Rq]
                                            for Rr in range(dAnum):
                                                u[0] = u0[0] + dAs[Rr]
                                                for Rs in range(dEnum):
                                                    u[1] = u0[1] + dEs[Rs]
                                                    for Rt in range(dBnum):
                                                        u[2] = u0[2] + dBs[Rt]
                                                        #
                                                        B = Lin_Model.\
                                                            _build_input_jacobian(
                                                            x, u, 
                                                            cg_shift=cgshift)[3:6,0:3]
                                                        dets[Ri,Rj,Rk,Rl,Rm,Rn,Ro,Rp,Rq,Rr,Rs,Rt] =\
                                                            np.linalg.det(B) # *Sdet
                                                        if counter % report_every == 0:
                                                            print(("i = {:>9d} / {:>9d}" +
                                                                ", det = {:>+10.3e}")
                                                                .format(counter,totalnum,
                                                                dets[Ri,Rj,Rk,Rl,Rm,Rn,Ro,Rp,Rq,Rr,Rs,Rt]))
                                                        counter += 1
        
        # reshape determinants
        ifs,jfs,kfs,lfs,mfs,nfs,ofs,pfs,qfs,rfs,sfs,tfs = np.indices(dets.shape).reshape(12,-1)
        dets_flat = dets.reshape(-1)

        # Create the 2D array with the companion values and 3D array values
        dets_2D = np.column_stack(
            (dVs[ifs],das[jfs],dbs[kfs],
            dps[lfs],dqs[mfs],drs[nfs],
            dzs[ofs],dPs[pfs],dTs[qfs],
            dAs[rfs],dEs[sfs],dBs[tfs], dets_flat)
        )

        # print(dets_2D)
        # rep2D(dets_2D,"current_data")

        # save data
        headers = ["V[ft/s]","a[rad]","b[rad]",
                "p[rad/s]","q[rad/s]","r[rad/s]",
                "zf[ft]","phi[rad]","theta[rad]",
                "da[rad]","de[rad]","dB[rad]","determinant"]
        header = ",".join(headers) + "\n"
        header += "x_tr = " + str(x0) + "\n"
        header += "u_tr = " + str(u0)

        # filename
        casenums = np.sort([dictup[j] for j in case])
        casevars = "_".join([dictupinv[casenum] for casenum in casenums])
        casenums = "_" .join([str(casenum) for casenum in casenums])
        datafilename = casenums + "_" + casevars + "_data.csv"
        datafilepath = folder + datafilename

        # read in old data
        if not(overwrite_data) and isfile(datafilepath):
            old_data = np.loadtxt(datafilepath,delimiter=",",skiprows=3)
            # rep2D(dets_2D,"old_data")
            dets_2D = np.unique( np.concatenate((old_data, dets_2D), axis=0), axis=0)
        
        # save data
        np.savetxt(datafilepath,dets_2D,delimiter=",",header=header,fmt="%+23.16e")
        if print_data_after_run:
            rep2D(dets_2D,"saving_data")

    if len(plot_cases) > 0:
        # turn into 9D array
        print("plotting...")
        
            # plots
        # change plot text parameters
        plt.rcParams["font.family"] = "Serif"
        plt.rcParams["font.size"] = 8.0
        plt.rcParams["axes.labelsize"] = 8.0
        plt.rcParams['lines.linewidth'] = 1.0
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

        # plot contours
        cmap = "seismic" # "gray" # newcmap # "PuOr" # 

    for case in plot_cases: # if len(run_cases) == 0: # 
        # determine filename for case
        casenums = np.sort([dictup[j] for j in case])
        casevars = "_".join([dictupinv[casenum] for casenum in casenums])
        casenums = "_" .join([str(casenum) for casenum in casenums])
        datafilename = casenums + "_" + casevars + "_data.csv"
        datafilepath = folder + datafilename
        # read in data from file
        print("reading in data...")
        dets_2D = np.loadtxt(datafilepath,delimiter=",",skiprows=1)

        r2ds = [0,1,1,1,1,1,0,1,1,1,1,1,]
        lbls = [r"Velocity ($\Delta V$), ft/s",
                r"Angle of attack ($\Delta \alpha$), deg",
                r"Sideslip angle ($\Delta \beta$), deg",
                r"Roll rate ($\Delta p$), deg/s",
                r"Pitch rate ($\Delta q$), deg/s",
                r"Yaw rate ($\Delta r$), deg/s",
                r"Altitude ($\Delta z_f$), ft",
                r"Bank angle ($\Delta \phi$), deg",
                r"Elevation angle ($\Delta \theta$), deg",
                r"Aileron deflection ($\Delta \delta_a$), deg",
                r"Stabilator deflection ($\Delta \delta_e^B$), deg",
                r"Tail rotation ($\Delta \delta_B$), deg",]
        levels = 300
        vals = np.ones((levels,4))
        newcmap = ListedColormap(vals)

        # for plot in plots:
        #
        xi,yi = (dictup[j] for j in case) # plot # 
        #
        lblx = lbls[xi].split("(")[0][:-1]
        lbly = lbls[yi].split("(")[0][:-1]
        print("    plotting",case,":",lblx,"vs",lbly,"...")
        fig,ax = plt.subplots(figsize=(3.5,3.25),constrained_layout=True)
        # ax.grid(which="major",lw=0.6,ls="-",c="k")
        #
        cols = np.arange(dets_2D.shape[1]-1)
        cols = cols[np.logical_and(cols != xi, cols != yi)]
        row_inds = tuple(np.arange(dets_2D.shape[0])
            [np.all(dets_2D[:,cols] == 0.0,axis=1)])
        #
        # determine x and y values
        if r2ds[xi]: xs = np.rad2deg(dets_2D[row_inds,xi])
        else:        xs = dets_2D[row_inds,xi]
        if r2ds[yi]: ys = np.rad2deg(dets_2D[row_inds,yi])
        else:        ys = dets_2D[row_inds,yi]
        
        # determine z data
        zs = dets_2D[row_inds,-1]

        side = int(xs.shape[0]**0.5)
        xs = xs.reshape(side,side)
        ys = ys.reshape(side,side)
        zs = zs.reshape(side,side)
        # quit()
        
        # determine bounds for colorbar
        maxval = max(abs(np.max(zs)),abs(np.min(zs)))
        cf = ax.contourf( # ax.tricontourf( # 
            xs,ys,zs,
            cmap=cmap,
            levels=levels, # 300, # 100, # 
            vmin = -maxval,vmax = maxval,
        )
        fig.colorbar(cf,) # format="%+9.2e") # "{:+9.2e}") # 
        cs = ax.contour(
            xs,ys,zs,
            # cmap="seismic", # newcmap, # "PuOr", # "gray", # 
            levels=[0.0], # 300, # 100, # 
            colors="k",
            linewidths=0.6,
            vmin = -maxval,vmax = maxval,
        )
        ax.clabel(cs, inline=1, fontsize=6,fmt="% 4.1f")

        # axis labels
        ax.set_xlabel(lbls[xi])
        ax.set_ylabel(lbls[yi])

        if save_plots:
            plot_file = "det_" + casenums + "_" + casevars + "." + plot_type
            plot_name = folder + plot_file
            fig.savefig(plot_name,transparent=plot_transparent,dpi=300.0)

        if show_plots:
            plt.show()
