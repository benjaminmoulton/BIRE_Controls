import numpy as np
import json
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm, SymLogNorm
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

    # settings
    run_cases = False # True # 
    datafilename = "determinant_data.csv" # "V_a_data.csv" # 
    overwrite_data = True # False # 

    # run settings
    dVlim =  50.0; dVhnum =    0; dVshift = 0.00; dVnum =  dVhnum*2+1
    dalim =   5.0; dahnum =  100; dashift = 0.00; danum =  dahnum*2+1
    dblim =   1.0; dbhnum =  100; dbshift = 0.00; dbnum =  dbhnum*2+1
    dplim =  90.0; dphnum =    0; dpshift = 0.00; dpnum =  dphnum*2+1
    dqlim =  20.0; dqhnum =    0; dqshift = 0.00; dqnum =  dqhnum*2+1
    drlim =  10.0; drhnum =    0; drshift = 0.00; drnum =  drhnum*2+1
    dzlim = 500.0; dzhnum =    0; dzshift = 0.00; dznum =  dzhnum*2+1
    dPlim =  90.0; dPhnum =    0; dPshift = 0.00; dPnum =  dPhnum*2+1
    dTlim =  45.0; dThnum =    0; dTshift = 0.00; dTnum =  dThnum*2+1
    #
    #
    report_every = 2500

    
    if run_cases:
        # trim for BIRE, determine LQR for controller code example
        compr = True # False # 
        stall = True # False # 
        fitthrust = True # False # 
        phi_trim = 0.0
        #
        bire_fs_dict["initial"]["mach"] = 0.6
        bire_fs_dict["initial"]["altitude[ft]"] = 15000.0
        bire_fs_dict["initial"]["trim"]["bank_angle[deg]"] = phi_trim
        bire_fs_dict["simulation"]["include_compressibility"] = compr
        bire_fs_dict["simulation"]["include_stall"] = stall
        bire_fs_dict["simulation"]["use_fitted_thrust_model"] = fitthrust
        bire_fs_dict["initial"]["trim"]["type"] = "sct"
        bire_fs_dict["initial"]["type"] = "trim"
        bire = Aircraft(bire_fs_dict)
        x0 = bire.x_trim_euler
        u0 = bire.u_trim
        # print(bire.inertia_model.W)
        # print(bire.cgshift)
        # bire._report_trim_solution()
        # # build linearized system
        _,Lin_Model = bire._build_controller(x0,u0,save_matrices=False,mrrr=[0,1,2,6,7,8,9,10,11],
            mrrc=[3],drop_actrs=True,run_freq=False,report=False)
        rep2D(Lin_Model.B_min,"B")
        S = np.diag([bire.s_da,bire.s_de,bire.s_dr])
        Sdet = np.linalg.det(S)
        print("B*S det trim =", np.linalg.det(Lin_Model.B_min)*Sdet)

        dVs =            np.linspace(-dVlim,dVlim,num=dVnum) + dVshift
        das = np.deg2rad(np.linspace(-dalim,dalim,num=danum) + dashift)
        dbs = np.deg2rad(np.linspace(-dblim,dblim,num=dbnum) + dbshift)
        dps = np.deg2rad(np.linspace(-dplim,dplim,num=dpnum) + dpshift)
        dqs = np.deg2rad(np.linspace(-dqlim,dqlim,num=dqnum) + dqshift)
        drs = np.deg2rad(np.linspace(-drlim,drlim,num=drnum) + drshift)
        dzs =            np.linspace(-dzlim,dzlim,num=dznum) + dzshift
        dPs = np.deg2rad(np.linspace(-dPlim,dPlim,num=dPnum) + dPshift)
        dTs = np.deg2rad(np.linspace(-dTlim,dTlim,num=dTnum) + dTshift)

        if dVnum == 1: dVs = np.array([0.0])
        if danum == 1: das = np.array([0.0])
        if dbnum == 1: dbs = np.array([0.0])
        if dpnum == 1: dps = np.array([0.0])
        if dqnum == 1: dqs = np.array([0.0])
        if drnum == 1: drs = np.array([0.0])
        if dznum == 1: dzs = np.array([0.0])
        if dPnum == 1: dPs = np.array([0.0])
        if dTnum == 1: dTs = np.array([0.0])

        dets = np.zeros((dVnum,danum,dbnum,dpnum,dqnum,drnum,dznum,dPnum,dTnum))

        print("running...")
        print("# cases =",("{:>6s}, "*9).format("dVnum","danum","dbnum",
            "dpnum","dqnum","drnum","dznum","dPnum","dTnum"))
        print("         ",("{:>6d}, "*9).format(dVnum,danum,dbnum,dpnum,
            dqnum,drnum,dznum,dPnum,dTnum))

        counter = 1
        totalnum = dVnum*danum*dbnum*dpnum*dqnum*drnum*dznum*dPnum*dTnum
        x = x0
        u = u0
        vx_trim = x0[0]; vy_trim = x0[1]; vz_trim = x0[2]
        V_trim = (vx_trim**2.0 + vy_trim**2.0 + vz_trim**2.0)**0.5
        a_trim = atan2(vz_trim,vx_trim)
        b_trim = asin(vy_trim/V_trim)
        for i in range(dVnum):
            for j in range(danum):
                for k in range(dbnum):
                    Vnew = V_trim + dVs[i]
                    anew = a_trim + das[j]
                    bnew = b_trim + dbs[k]
                    # print(Vnew,np.rad2deg(anew),np.rad2deg(bnew))
                    x[0] = Vnew*cos(anew)*cos(bnew)
                    x[1] = Vnew*sin(bnew)
                    x[2] = Vnew*sin(anew)*cos(bnew)
                    for l in range(dpnum):
                        x[3] = x0[3] + dps[l]
                        for m in range(dqnum):
                            x[4] = x0[4] + dqs[m]
                            for n in range(drnum):
                                x[5] = x0[5] + drs[n]
                                for o in range(dznum):
                                    x[8] = x0[8] + dzs[o]
                                    for p in range(dPnum):
                                        x[9] = x0[9] + dPs[p]
                                        for q in range(dTnum):
                                            x[10] = x0[10] + dTs[q]
                                            #
                                            B = Lin_Model.\
                                                _build_input_jacobian(
                                                x, u, 
                                                cg_shift = [0.,0.,0.])[3:6,0:3]
                                            dets[i,j,k,l,m,n,o,p,q] = \
                                                np.linalg.det(B) # *Sdet
                                            if counter % report_every == 0:
                                                print(("i = {:>9d} / {:>9d}" +
                                                    ", det = {:>+10.3e}")
                                                    .format(counter,totalnum,
                                                    dets[i,j,k,l,m,n,o,p,q]))
                                            counter += 1
        
        # reshape determinants
        ifs,jfs,kfs,lfs,mfs,nfs,ofs,pfs,qfs = np.indices(dets.shape).reshape(9,-1)
        dets_flat = dets.reshape(-1)

        # Create the 2D array with the companion values and 3D array values
        dets_2D = np.column_stack(
            (dVs[ifs],das[jfs],dbs[kfs],
            dps[lfs],dqs[mfs],drs[nfs],
            dzs[ofs],dPs[pfs],dTs[qfs], dets_flat)
        )

        # print(dets_2D)
        # rep2D(dets_2D,"current_data")

        # save data
        headers = ["V[ft/s]","a[rad]","b[rad]",
                "p[rad/s]","q[rad/s]","r[rad/s]",
                "zf[ft]","phi[rad]","theta[rad]",]
        header = ",".join(headers) + "\n"
        header += "x_tr = " + str(x0) + "\n"
        header += "u_tr = " + str(u0)

        # read in old data
        if not(overwrite_data) and isfile(datafilename):
            old_data = np.loadtxt(datafilename,delimiter=",",skiprows=3)
            # rep2D(dets_2D,"old_data")
            dets_2D = np.unique( np.concatenate((old_data, dets_2D), axis=0), axis=0)
        
        # save data
        np.savetxt(datafilename,dets_2D,delimiter=",",header=header,fmt="%+23.16e")
        rep2D(dets_2D,"saving_data")
        
    
    else:
        # read in data from file
        print("reading in data...")
        dets_2D = np.loadtxt(datafilename,delimiter=",",skiprows=1)

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
        show_plots = True

        plots = [(1,2),(0,2),(0,1),(3,4),(4,5),(3,5),] # ,(1,2),(2,1),(3,5),(3,4)] # 

        r2ds = [0,1,1,1,1,1,0,1,1,]
        lbls = [r"Velocity ($\Delta V$), ft/s",
                r"Angle of attack ($\Delta \alpha$), deg",
                r"Sideslip angle ($\Delta \beta$), deg",
                r"Roll rate ($\Delta p$), deg/s",
                r"Pitch rate ($\Delta q$), deg/s",
                r"Yaw rate ($\Delta r$), deg/s",
                r"Altitude ($\Delta z_f$), ft",
                r"Bank angle ($\Delta \phi$), deg",
                r"Elevation angle ($\Delta \theta$), deg",]
        levels = 300
        vals = np.ones((levels,4))
        newcmap = ListedColormap(vals)

        for plot in plots:
            #
            xi,yi = plot
            #
            lblx = lbls[xi].split("(")[0][:-1]
            lbly = lbls[yi].split("(")[0][:-1]
            print("    plotting",plot,":",lblx,"vs",lbly,"...")
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
                cmap="seismic", # newcmap, # "PuOr", # "gray", # 
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
            # fig.savefig("p01_contour.pdf",dpi=300.0)
            if show_plots:
                plt.show()

    quit()

    # vars = [dVs,das,dbs,dps,dqs,drs]
    # hnms = [dVhnum,dahnum,dbhnum,dphnum,dqhnum,drhnum]
    # for plot in plots:
    #     fig,ax = plt.subplots(figsize=(3.5,3.25),constrained_layout=True)
    #     # ax.grid(which="major",lw=0.6,ls="-",c="k")
    #     #
    #     xi,yi = plot
    #     #
    #     cols = np.arange(dets_2D.shape[1]-1)
    #     cols = cols[np.logical_and(cols != xi, cols != yi)]
    #     row_inds = np.arange(dets_2D.shape[0])\
    #         [np.all(dets_2D[:,cols] == 0.0,axis=1)]
    #     #
    #     if r2ds[xi]: xs = np.rad2deg(vars[xi])
    #     else:        xs = vars[xi]
    #     if r2ds[yi]: ys = np.rad2deg(vars[yi])
    #     else:        ys = vars[yi]
    #     xm,ym = np.meshgrid(xs,ys)
    #     # [dVhnum,dahnum,dbhnum,dphnum,dqhnum,drhnum,dzhnum,dPhnum,dThnum]
    #     slices = hnms*1
    #     slices[xi] = slice(None)
    #     slices[yi] = slice(None)
    #     mdets = dets[tuple(slices)].squeeze()
        
    #     # determine bounds for colorbar
    #     maxval = max(abs(np.max(mdets)),abs(np.min(mdets)))
    #     cf = ax.contourf(xm,ym,mdets,
    #         cmap="seismic", # newcmap, # "PuOr", # "gray", # 
    #         levels=levels, # 300, # 100, # 
    #         vmin = -maxval,vmax = maxval,
    #         # norm=SymLogNorm(linthresh=0.1,),
    #         )
    #     fig.colorbar(cf,) # format="%+9.2e") # "{:+9.2e}") # 

    #     # axis labels
    #     ax.set_xlabel(lbls[xi])
    #     ax.set_ylabel(lbls[yi])
    #     # fig.savefig("p01_contour.pdf",dpi=300.0)
    #     if show_plots:
    #         plt.show()

    # quit()