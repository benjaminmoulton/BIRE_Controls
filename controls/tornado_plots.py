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
import math as m
from std_atm import stdatm_english
from quat import quat_mult, euler_2_quat, quat_2_euler, quat_norm, body_2_fixed, fixed_2_body, eulerdot_2_quatdot, quatdot_2_eulerdot
from linearization import linearization as lin,Anderson_correction_der_coeff,Anderson_correction_der_M

from controller_simulation import Aircraft,run_single_simulation, \
    monte_carlo_perturbations, report_latex, report_eigprops, rep2D,BIREAero

from os.path import isfile
from os import mkdir, rmdir, walk, remove, listdir


if __name__ == "__main__":

    # report
    print("running tornado plots!!!!")

    # file folder
    file_folder = "track_plots/"
    ff_ss = file_folder + "saved_single_simulation/"
    ff_mc = file_folder + "saved_monte_carlo/"

    # list of controller types
    ctrl_types = ["PI","DI","LQT","LQRDI","ITPI","NDI","CAMA"]
    mod_types  = ["CG05","CG10","BR100","BR200","SE11","SE12"]
    plot_types = ["CG","BR","SE"]
    pqr_cases = ["p","q","r"]
    CFM_cases = ["CL","CS","CD","Cell","Cm","Cn"]

    ctrl_props = {}
    for key in ctrl_types:
        ctrl_props[key] = {}
        for subkey in [key] + (key=="CAMA")*["UW"] + mod_types:
            ctrl_props[key][subkey] = {}

    for ss in listdir(ff_ss):
        ss_split = ss.split("_")
        ss_f1   = ss_split[12]
        key = ss_split[13]
        subkey  = ss_split[-1]
        #
        right_f1   = ss_f1 == "C2"
        right_ctrl = key in ctrl_types
        right_mod  = subkey in ctrl_types + (key=="CAMA")*["UW"] + mod_types

        if right_f1 and right_ctrl and right_mod:
            # open folder
            in_ss = listdir(ff_ss + ss + "/")
            for file in in_ss:
                if file[:10] == "final_bank":
                    final_bank = int((file.split("_")[-1].split(".")[0])[1:])
                    final_bank = int(round(final_bank/10.0)*10.0)
                    if key == "LQRDI" and subkey in ["SE11","SE12"]:
                        final_bank -= 10
                    ctrl_props[key][subkey]["final_bank"] = final_bank

    for mc in listdir(ff_mc):
        mc_split = mc.split("_")
        mc_f1   = mc_split[14]
        key = mc_split[15]
        subkey  = mc_split[-2]
        runcase = mc_split[-1]
        #
        right_f1   = mc_f1 == "C2"
        right_ctrl = key in ctrl_types
        right_mod  = subkey in ctrl_types + (key=="CAMA")*["UW"] + mod_types
        right_case = runcase in pqr_cases + CFM_cases

        if right_f1 and right_ctrl and right_mod and right_case:
            
            if   runcase in pqr_cases + CFM_cases:
                # open folder
                in_mc = listdir(ff_mc + mc + "/")
                for file in in_mc:
                    if file == "roa_est.txt":
                        with open(ff_mc + mc + "/" + file) as f:
                            if key != "PI":
                                f.readline()
                                f.readline()
                            if   runcase == "p"   : skip_lines = 0
                            elif runcase == "q"   : skip_lines = 1
                            elif runcase == "r"   : skip_lines = 2
                            elif runcase == "CL"  : skip_lines = 4
                            elif runcase == "CS"  : skip_lines = 5
                            elif runcase == "CD"  : skip_lines = 6
                            elif runcase == "Cell": skip_lines = 7
                            elif runcase == "Cm"  : skip_lines = 8
                            elif runcase == "Cn"  : skip_lines = 9
                            for i in range(skip_lines):
                                f.readline()
                            
                            # get important line
                            line_str = f.readline().split()
                            # if line_str[2] not in pqr_cases + CFM_cases + ["Cl"]:
                            #     print(mc,line_str)
                            keycase = line_str[2]
                            keyval  = float(line_str[5])
                            ctrl_props[key][subkey][keycase] = keyval
                            f.close()
            else:
                raise ValueError("This case shouldn't come!: " + mc)
    
    # from pprint import pprint
    # pprint(ctrl_props,indent=4)

    # create plots
    # initialize rc params
    plt.rcParams["font.family"] = "Serif"
    plt.rcParams["font.size"] = 8.0
    plt.rcParams["axes.labelsize"] = 8.0
    plt.rcParams['axes.xmargin'] = 0
    plt.rcParams['lines.linewidth'] = 0.75 # 1.0
    plt.rcParams["xtick.minor.visible"] = True
    # plt.rcParams["ytick.major.visible"] = True
    plt.rcParams["xtick.direction"] = plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.bottom"] = plt.rcParams["xtick.top"] = True
    # plt.rcParams["ytick.left"] = plt.rcParams["ytick.right"] = True
    plt.rcParams["xtick.major.width"] = 0.75 # plt.rcParams["ytick.major.width"] = 0.75
    plt.rcParams["xtick.minor.width"] = 0.75 # plt.rcParams["ytick.minor.width"] = 0.75
    plt.rcParams["xtick.major.size"] =  5.0 # plt.rcParams["ytick.major.size"] = 5.0
    plt.rcParams["xtick.minor.size"] =  2.5 # plt.rcParams["ytick.minor.size"] = 2.5
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams['figure.dpi'] = 300.0
    plt.rcParams['figure.max_open_warning'] = 30
    
    # initialize plots
    subdict = {
        "figsize" : (3.25,3.5),
        "constrained_layout" : True
    }
    plots = {}
    def major_formatter(x, pos):
        label = str(-x) if x < 0 else str(x)
        return label
    # initialize
    for key in ctrl_types:
        plots[key] = {}
        for subkey in plot_types:
            plots[key][subkey] = {}
            plots[key][subkey]["fig"], plots[key][subkey]["axs"] = \
                plt.subplots(3,1,height_ratios=[1,3,6],**subdict)
    
    # plot prep
    CL_mlt = {
        "PI"   :{"CG":1.0,"BR":1.0,"SE":1.0},
        "DI"   :{"CG":1.0,"BR":1.0,"SE":1.0},
        "LQT"  :{"CG":1.0,"BR":1.0,"SE":1.0},
        "LQRDI":{"CG":1.0,"BR":1.0,"SE":1.0},
        "ITPI" :{"CG":1.0,"BR":1.0,"SE":1.0},
        "NDI"  :{"CG":1.0,"BR":1.0,"SE":1.0},
        "CAMA" :{"CG":1.0,"BR":1.0,"SE":1.0},
    }
    p_mlt = {
        "PI"   :{"CG":.01,"BR":1.0,"SE":1.0},
        "DI"   :{"CG":0.1,"BR":0.1,"SE":0.1},
        "LQT"  :{"CG":0.1,"BR":0.1,"SE":0.1},
        "LQRDI":{"CG":0.1,"BR":0.1,"SE":0.1},
        "ITPI" :{"CG":1.0,"BR":1.0,"SE":1.0},
        "NDI"  :{"CG":0.1,"BR":1.0,"SE":1.0},
        "CAMA" :{"CG":1.0,"BR":1.0,"SE":1.0},
    }
    q_mlt = {
        "PI"   :{"CG":1.0,"BR":1.0,"SE":1.0},
        "DI"   :{"CG":1.0,"BR":1.0,"SE":1.0},
        "LQT"  :{"CG":1.0,"BR":1.0,"SE":1.0},
        "LQRDI":{"CG":0.1,"BR":0.1,"SE":0.1},
        "ITPI" :{"CG":0.1,"BR":1.0,"SE":1.0},
        "NDI"  :{"CG":0.1,"BR":0.1,"SE":1.0},
        "CAMA" :{"CG":1.0,"BR":1.0,"SE":1.0},
    }
    r_mlt = {
        "PI"   :{"CG":1.0,"BR":10.0,"SE":10.0},
        "DI"   :{"CG":1.0,"BR":1.0,"SE":1.0},
        "LQT"  :{"CG":1.0,"BR":1.0,"SE":1.0},
        "LQRDI":{"CG":1.0,"BR":1.0,"SE":1.0},
        "ITPI" :{"CG":1.0,"BR":1.0,"SE":1.0},
        "NDI"  :{"CG":1.0,"BR":10.0,"SE":10.0},
        "CAMA" :{"CG":1.0,"BR":1.0,"SE":1.0},
    }
    pqr_lim = {
        "PI"   :{"CG":(20,5),"BR":(20,5),"SE":(20,5)},
        "DI"   :{"CG":(15,7),"BR":(10,5),"SE":(10,5)},
        "LQT"  :{"CG":(5,11),"BR":(5,11),"SE":(5,11)},
        "LQRDI":{"CG":(30,7),"BR":(10,5),"SE":(10,5)},
        "ITPI" :{"CG":(40,5),"BR":(30,7),"SE":(30,7)},
        "NDI"  :{"CG":(15,7),"BR":(40,5),"SE":(50,5)},
        "CAMA" :{"CG":(30,7),"BR":(30,7),"SE":(30,7)},
    }
    #
    mod_w_type = {
        "CG" : ["CG05","CG10"],
        "BR" : ["BR100","BR200"],
        "SE" : ["SE11","SE12"],
    }
    # #######################################################################
    # ctrl_types = ["PI","DI","LQT","LQRDI","ITPI","NDI","CAMA"]
    # ctrl_types = ctrl_types[0:1] # [6:7] # 
    # #######################################################################
    
    # plot!
    for key in ctrl_types:
        for subkey in plot_types:
            fig,axs = plots[key][subkey]["fig"], plots[key][subkey]["axs"]

            # skip some
            if key == "ITPI" and subkey != "CG":
                continue
            if key == "CAMA":
                continue

            # modify plot limits
            i_pqrlim = pqr_lim[key][subkey]
            axs[0].set_xlim(-60.0,60.0)
            tick_lbl = np.linspace(-60.0,60.0,num=7)
            axs[0].set_xticks(tick_lbl.astype(int))
            axs[0].xaxis.set_major_formatter(major_formatter)
            #
            axs[1].set_xlim(-i_pqrlim[0],i_pqrlim[0])
            tick_lbl = np.linspace(-i_pqrlim[0],i_pqrlim[0],num=i_pqrlim[1])
            axs[1].set_xticks(tick_lbl.astype(int))
            axs[1].xaxis.set_major_formatter(major_formatter)
            #
            axs[2].set_xlim(-1,1)
            axs[2].xaxis.set_major_formatter(major_formatter)
            
            # plot bank angle
            for mod in [key] + mod_w_type[subkey]:
                # determine colors
                gray_ind = 0 # 1 # 
                ec = "k" if mod in ctrl_types \
                    else "0.5" if mod == mod_w_type[subkey][gray_ind] else "k"
                c = "k" if mod in ctrl_types \
                    else "k" if mod == mod_w_type[subkey][gray_ind] else "None"
                h = None if mod in ctrl_types \
                    else None if mod == mod_w_type[subkey][gray_ind] else "/"
                a = 1.0 if mod in ctrl_types \
                    else 0.5 if mod == mod_w_type[subkey][gray_ind] else 1.0
                bardict = dict(color=c,edgecolor=ec,hatch=h,alpha=a)
                mlt = -1.0 if mod in ctrl_types else 1.0

                # bank angle
                wid = ctrl_props[key][mod]["final_bank"]
                axs[0].barh("Bank angle, deg",width=mlt*wid,**bardict)

                # plot p,q,r
                wid = ctrl_props[key][mod]["r"]
                r_ord = m.floor(m.log(1.0/r_mlt[key][subkey],10))
                rep_r = r_ord != 0
                axs[1].barh(r"$\Delta r$," \
                    + rep_r*" 10$^{:s}{:d}{:s}$".format("{",r_ord,"}")+" deg/s",\
                    width=mlt*wid*r_mlt[key][subkey],**bardict)
                #
                wid = ctrl_props[key][mod]["q"]
                q_ord = m.floor(m.log(1.0/q_mlt[key][subkey],10))
                rep_q = q_ord != 0
                axs[1].barh(r"$\Delta q$," \
                    + rep_q*" 10$^{:s}{:d}{:s}$".format("{",q_ord,"}")+" deg/s",\
                    width=mlt*wid*q_mlt[key][subkey],**bardict)
                #
                wid = ctrl_props[key][mod]["p"]
                p_ord = m.floor(m.log(1.0/p_mlt[key][subkey],10))
                rep_p = p_ord != 0
                axs[1].barh(r"$\Delta p$," \
                    + rep_p*" 10$^{:s}{:d}{:s}$".format("{",p_ord,"}")+" deg/s",\
                    width=mlt*wid*p_mlt[key][subkey],**bardict)

                # plot CFM
                wid = ctrl_props[key][mod]["Cn"]
                axs[2].barh(r"$\Delta C_n$",width=mlt*wid,**bardict)
                #
                wid = ctrl_props[key][mod]["Cm"]
                axs[2].barh(r"$\Delta C_m$",width=mlt*wid,**bardict)
                #
                wid = ctrl_props[key][mod]["Cl"]
                axs[2].barh(r"$\Delta C_\ell$",width=mlt*wid,**bardict)
                #
                wid = ctrl_props[key][mod]["CD"]
                axs[2].barh(r"$\Delta C_D$",width=mlt*wid,**bardict)
                #
                wid = ctrl_props[key][mod]["CS"]
                axs[2].barh(r"$\Delta C_S$",width=mlt*wid,**bardict)
                #
                wid = ctrl_props[key][mod]["CL"]
                CL_ord = m.floor(m.log(1.0/CL_mlt[key][subkey],10))
                rep_CL = CL_ord != 0
                axs[2].barh(r"$\Delta C_L$" \
                    + rep_CL*", 10$^{:s}{:d}{:s}$".format("{",CL_ord,"}"),\
                    width=mlt*wid*CL_mlt[key][subkey],**bardict)
    
    # save plots
    show_plots = False # True # 
    save_folder = "tornado_dissertation_plots/"
    savedict = dict(
        format = "pdf", # "png", # 
        # transparent = True, # False, # 
        dpi = 300.0,
    )
    savedict["transparent"] = False # True if savedict["format"] == "pdf" else False
    for key in ctrl_types:
        for subkey in plot_types:
            fig,axs = plots[key][subkey]["fig"], plots[key][subkey]["axs"]
            
            # save figure
            figlst = ["tornado",key,subkey]
            name = save_folder + "_".join(figlst) + "." + savedict["format"]
            fig.savefig(name,**savedict)
    
    if show_plots:
        plt.show()
    else:
        plt.close("all")
