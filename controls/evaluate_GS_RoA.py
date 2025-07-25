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
from os import listdir,remove


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
    folder = "GS_RoA_plots/"
    overwrite_data = True # False # 
    print_data_after_run = False # True # 
    show_plots = False # True # 
    save_plots = True # False # 
    plot_type = "pdf" # "png" # 
    plot_transparent = True if plot_type == "pdf" else False

    #
    bire_fs_dict["simulation"]["include_compressibility"] = compr
    bire_fs_dict["simulation"]["include_stall"] = stall
    bire_fs_dict["simulation"]["use_fitted_thrust_model"] = fitthrust
    bire_fs_dict["aircraft"]["CG_shift[ft]"] = cgshift
    bire_fs_dict["initial"]["mach"] = 0.6 # 0.8 # 
    bire_fs_dict["initial"]["altitude[ft]"] = 15000.0 # 30000.0 # 
    bire_fs_dict["initial"]["trim"]["bank_angle[deg]"] = phi_trim
    bire_fs_dict["initial"]["trim"]["type"] = "sct"
    bire_fs_dict["initial"]["type"] = "trim"
    # bire_fs_dict["initial"]["trim_guess"] = {}
    # if   subfolder_end == "_m":
    #     bire_fs_dict["initial"]["trim_guess"]["elevator[deg]"] = -25.0
    #     bire_fs_dict["initial"]["trim_guess"]["BIRE[deg]"] = -70.0
    # elif subfolder_end == "_p":
    #     bire_fs_dict["initial"]["trim_guess"]["elevator[deg]"] = -25.0
    #     bire_fs_dict["initial"]["trim_guess"]["BIRE[deg]"] = 70.0
    # else: # ""
    #     bire_fs_dict["initial"]["trim_guess"]["elevator[deg]"] = 20.0
    #     bire_fs_dict["initial"]["trim_guess"]["BIRE[deg]"] = 0.0
    bire_fs_dict["controller"]["LQR"] = {
        "note" : "_almost_current",
        "Q" : [1.0e-3, 1.0e-6, 2.0e-4, # 
            1.0e0, 1.0e0, 1.0e0,
            0.0, 0.0, 5.0e-6,
            1.0e0, 1.0e0, 0.0],
        "Q1a" : [0.0e0, 0.0e0, 0.0e0, 0.0e0],
        "Q2a" : [0.0, 0.0, 0.0, 0.0],
        "R" : [5.0e0, 5.0e0, 5.0e0, 5.0e-2]
    }
    bire = Aircraft(bire_fs_dict)
    x0 = bire.x_trim_euler*1.0
    u0 = bire.u_trim*1.0
    print("cg x, ft  =",cgshift[0])
    print("Bank, deg =",phi_trim)
    print("x0 =",x0)
    print("u0 =",u0)
    # print(bire.inertia_model.W)
    # print(bire.cgshift)
    # bire._report_trim_solution()
    # # build linearized system
    _,Lin_Model = bire._build_controller(x0,u0,save_matrices=False,mrrr=[6,7,11],
        # mrrc=[3],
        drop_actrs=True,
        include_stall_derivatives=True,
        include_altitude_derivatives=True,
        run_freq=False,report=False)
    report_latex(x0[:,np.newaxis].T,"x_{tr}")
    report_latex(u0[:,np.newaxis].T,"u_{tr}")
    report_latex(Lin_Model.A_min,"A")
    report_latex(Lin_Model.B_min,"B")
    report_latex(Lin_Model.K,"K")
    report_latex(Lin_Model.P,"P")
    #
    K = Lin_Model.K*1.0
    P = Lin_Model.P*1.0
    A = Lin_Model.A_min*1.0
    B = Lin_Model.B_min*1.0
    # #
    # G = np.matmul(P,A-np.matmul(B,K))
    # Geval,Gevec = np.linalg.eig(G)
    # print(G)
    # print(Geval)
    # quit()

    # run settings
    dVshift = 0.00; dVlim =   20.0 # dVlim =    0.1 # dVlim =   10.0 # dVlim =  400.0 # 
    dzshift = 0.00; dzlim =  500.0 # dzlim = 1000.0 # dzlim =    1.0 # 
    #
    #
    report_every = 500 # 2500 # 
    hnum = 50 # 100 # 200 # 150 # 5 # 

    # cases to run
    lineup = ["V","z"]
    dictup    = {lineup[i]:i for i in range(len(lineup))}
    dictupinv = {i:lineup[i] for i in range(len(lineup))}
    run_cases = [
        ("V","z"),
        # #
    ]
    plot_cases = [
        ("V","z"),
        # #
    ]
    bire.use_quaternions = False
    
    for case in run_cases:
        # report
        print("running case",case)
        # determine numbers
        dVnum = hnum*2+1 if "V" in case else 1
        dznum = hnum*2+1 if "z" in case else 1

        # values to run
        dVs =            np.linspace(-dVlim,dVlim,num=dVnum) + dVshift
        dzs =            np.linspace(-dzlim,dzlim,num=dznum) + dzshift

        # print(dVs,dzs)

        if dVnum == 1: dVs = np.array([0.0])
        if dznum == 1: dzs = np.array([0.0])

        Vdots = np.zeros((dVnum,dznum))

        print("running...")
        print("# cases =",("{:>6s}, "*2).format("dVnum","dznum"))
        print("         ",("{:>6d}, "*2).format(dVnum,dznum))

        counter = 1
        totalnum = dVnum*dznum
        x = x0*1.0
        u = u0*1.0
        vx_trim = x0[0]*1.0; vy_trim = x0[1]*1.0; vz_trim = x0[2]*1.0
        V_trim = (vx_trim**2.0 + vy_trim**2.0 + vz_trim**2.0)**0.5
        a_trim = atan2(vz_trim,vx_trim)
        b_trim = asin(vy_trim/V_trim)
        for Ri in range(dVnum):
            Vnew = V_trim + dVs[Ri]
            # print(Vnew,np.rad2deg(anew),np.rad2deg(bnew))
            x[0] = Vnew*cos(a_trim)*cos(b_trim) # vx_trim + dVs[Ri] # 
            x[1] = Vnew*sin(b_trim) # vy_trim # 
            x[2] = Vnew*sin(a_trim)*cos(b_trim) # vz_trim # 
            for Ro in range(dznum):
                x[8] = x0[8] + dzs[Ro]
                #
                # calculate Vdot
                xm    = np.delete(x-x0,[6,7,11,12,13,14,15])
                #
                u = u0 - np.matmul(K,xm)
                xdot = bire._nonlinear_euler_dynamics(0.0,x,True,True,u,True)#False)#
                xdotm = np.delete(xdot,[6,7,11,12,13,14,15])
                # #
                # u = u0 - np.matmul(K,xm)
                # xdotm = np.matmul(A,xm) + np.matmul(B,u)
                #
                Vdot  = np.matmul(xdotm.T,np.matmul(P,xm   ))
                Vdot += np.matmul(xm   .T,np.matmul(P,xdotm))
                Vdots[Ri,Ro] = Vdot*1.0
                #
                if counter % report_every == 0:
                    # print(x-x0)
                    # print(xm[:,0])
                    print(("i = {:>9d} / {:>9d}, Vdot = {:>+10.3e}")
                        .format(counter,totalnum,Vdots[Ri,Ro]))
                counter += 1
    
    bire.use_quaternions = True
        

    if len(plot_cases) > 0:
        
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
        cmap = "gray" # "seismic" # newcmap # "PuOr" # 

        # # remove plots that are already there if they are the wrong file type
        # print("removing wrong plot types...")
        # for subfile in listdir(folder):
        #     subfile_format = subfile.split(".")[-1]
        #     if subfile_format in ["pdf","png"] and subfile_format != plot_type:
        #         remove(folder + subfile)
        
        print("plotting...")

    # j = 25
    # for i in range(Vdots.shape[1]):
    #     print("dzs =",dzs[i],"dVs =",dVs[j],"Vdots =",Vdots[j,i])

    for case in plot_cases: # if len(run_cases) == 0: # 

        lbls = [r"Velocity ($\Delta V$), ft/s",
                r"Altitude ($\Delta z_f$), ft",
                ]
        levels = 10
        vals = np.ones((levels,4))
        newcmap = ListedColormap(vals)

        xi,yi = (dictup[j] for j in case) # plot # 
        #
        lblx = lbls[xi].split("(")[0][:-1]
        lbly = lbls[yi].split("(")[0][:-1]
        print("    plotting",case,":",lblx,"vs",lbly,"...")
        fig,ax = plt.subplots(figsize=(3.25,3.5),constrained_layout=True)
        #
        # determine x and y values
        xs = dVs*1.0 # Vdots[row_inds,xi]
        ys = dzs*1.0 # Vdots[row_inds,yi]
        
        # determine z data
        zs = Vdots*1.0 # Vdots[row_inds,-1]
        xS,yS = np.meshgrid(xs,ys)

        for i in range(xs.shape[0]):
            for j in range(ys.shape[0]):
                if zs[i][j] < 0.0:
                    ax.plot(xs[i],ys[j],marker="o",
                        ms=1.0, # ms=1.25, # 
                        c="k")
        
        # # determine bounds for colorbar
        # cf = ax.contourf(xS,yS,zs)
        # maxval = max(abs(np.max(zs)),abs(np.min(zs)))
        # cf = ax.contourf( # ax.tricontourf( # 
        #     xS,yS,
        #     zs,
        #     # cmap = "viridis",
        #     cmap=cmap,
        #     levels=levels, # 300, # 100, # 
        #     vmin = -maxval,vmax = maxval,
        # )
        # fig.colorbar(cf,label=r"$\dot{\mathcal{V}} = \Delta \dot{x}^\top P \Delta x + \Delta x^\top P \Delta \dot{x}$",format="%+5.2f") # "{:+9.2e}") # 
        # cs = ax.contour(
        #     xS,yS,
        #     zs,
        #     # cmap="seismic", # newcmap, # "PuOr", # "gray", # 
        #     levels=[0.0], # 300, # 100, # 
        #     colors="k",
        #     linewidths=0.6,
        #     vmin = -maxval,vmax = maxval,
        # )
        # ax.clabel(cs, inline=1, fontsize=6,fmt="% 4.1f")
        # center dot
        ax.plot([0.0],[0.0],marker="o",
                ms=3.0, # ms=1.25, # 
                c="k",mfc="w",mew=1.0)

        # axis labels
        ax.set_xlabel(lbls[xi])
        ax.set_ylabel(lbls[yi])
        #
        ax.set_xlim(-dVlim,dVlim)
        ax.set_ylim(-dzlim,dzlim)

        if save_plots:
            plot_file = "RoA_V_z" + "." + plot_type
            plot_name = folder + plot_file
            fig.savefig(plot_name,transparent=plot_transparent,dpi=300.0)

        if show_plots:
            plt.show()
