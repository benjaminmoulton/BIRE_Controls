import numpy as np
import json
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm, SymLogNorm, ListedColormap
import mpl_toolkits.mplot3d.axes3d as ax3
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap, BoundaryNorm
# from matplotlib.cm import get_cmap
from numpy import sign, matmul as mm
from datetime import datetime
import control as co
from scipy.linalg import block_diag
from scipy.integrate import ode, odeint
from scipy.interpolate import interp1d, interpn
from scipy.optimize import curve_fit,minimize,minimize_scalar,newton
from scipy.io import savemat, loadmat
from scipy.signal import tf2zpk as scipy_tf2zpk
# from math import pi, sin, cos, tan, exp, asin, atan, atan2
from numpy import pi, sin, cos, tan, exp, arcsin as asin, arccos as acos, arctan as atan, arctan2 as atan2
import math as m
from std_atm import stdatm_english
from quat import quat_mult, euler_2_quat, quat_2_euler, quat_norm, body_2_fixed, fixed_2_body, eulerdot_2_quatdot, quatdot_2_eulerdot
from linearization import linearization as lin,Anderson_correction_der_coeff,Anderson_correction_der_M

from controller_simulation import Aircraft,run_single_simulation, \
    monte_carlo_perturbations, report_latex, report_eigprops, rep2D,BIREAero

from bire_controllers import ControlAllocationMomentAssignmentAircraft, \
    NonlinearDynamicInversionAircraft, ITPIAircraft

from os.path import isfile
from os import mkdir, rmdir, walk, remove, listdir

import shapely as sh


if __name__ == "__main__":

    # report
    print("running de vn contour...")

    # settings
    run_cases = False # True # 
    skip_60_and_sim = True # False # 
    show_plots = False # True # 
    plot_format = "pdf" # "png"
    transparent = False # True # 
    save_folder = "de_vn_contour_plots/"
    save_file = save_folder + "de_vn_contour_trims.mat"
    alt_low = 0.0; alt_high = 50000.0; num_alt = 41 # 21 # 3 # 5 # 
    M_low   = 0.0; M_high   =     2.0; num_M   = 41 # 21 # 3 # 5 # 
    #
    Mskip = [0.9,1.1]

    # run_vars array
    run_alt = np.linspace(alt_low,alt_high,num_alt)
    run_M   = np.linspace(  M_low,  M_high,num_M)

    # flight envelope outline
    x_start = 0.13
    left = [
        [x_start, 0.0,], # [0.132357005232832, 6.505830436501128,], # 
        [0.1340528575139226, 986.9894251012447,],
        [0.13440448106377534, 1675.6823123154536,],
        [0.1379522890401167, 2425.4689731901162,],
        [0.14148083245372423, 3559.904736655917,],
        [0.14488996439059876, 4529.683083946045,],
        [0.14848948751822744, 5675.077651503147,],
        [0.15344007129744375, 7070.552277082395,],
        [0.15724736223127878, 8146.074290909972,],
        [0.16106153160193637, 9084.2568495175,],
        [0.16499428755017764, 10000.638681570235,],
        [0.1686845355412635, 11067.140179757786,],
        [0.17346198244494138, 12045.489077241975,],
        [0.17851069744352932, 13114.263417269147,],
        [0.18205331880682696, 13967.609451918346,],
        [0.1852302153445441, 14780.934409445974,],
        [0.19160475487215356, 15993.346829403068,],
        [0.19567675360626768, 16755.287843426835,],
        [0.20116886932492062, 17765.939070994486,],
        [0.2069052633513634, 18870.918165459152,],
        [0.2130218245881098, 20094.668393465476,],
        [0.22194833470378006, 21381.905728906797,],
        [0.22700641442036362, 22263.6978662855,],
        [0.23179437862271368, 23032.051366949097,],
        [0.2375392441171279, 23967.88348424868,],
        [0.24246186678971748, 24800.928731250882,],
        [0.24903969410624693, 25625.525013047132,],
        [0.25396109833640723, 26482.898493888435,],
        [0.26149198843971244, 27551.615547911755,],
        [0.26847505283116513, 28545.095840103284,],
        [0.2712976357539254, 28989.09156718692,],
        [0.2772147532720548, 29875.455452776157,],
        [0.287104708266965, 31031.640460492134,],
        [0.29340479657135854, 31895.01394702601,],
        [0.3012868533086055, 32835.53521531674,],
        [0.3076543161280738, 33730.35048524887,],
        [0.31488537501265446, 34825.55598644374,],
        [0.32353520252959755, 35804.13607083021,],
        [0.3326563102112394, 36739.01909167839,],
        [0.340141716599602, 37614.54530653704,],
        [0.34782078535611693, 38478.00112026208,],
        [0.35453726401165775, 39287.564868586385,],
        [0.3612576189057828, 40019.73305317707,],
        [0.36797380255555523, 40835.18708334305,],
        [0.37620520443763916, 41683.88163169106,],
        [0.3836926106665548, 42601.571551336856,],
        [0.3901156023249266, 43330.81792828403,],
        [0.3973677769873388, 44024.27887062075,],
        [0.404989412805956, 44886.08745101491,],
        [0.41297980107742804, 45602.94109794785,],
        [0.42367082301673886, 46620.59460304967,],
        [0.43360222651171876, 47484.1848845203,],
        [0.4417357386192602, 48268.27207190065,],
        [0.4496145496069951, 48998.262885234166,],
        [0.46, 50000.0,], # [0.45986879575185413, 49840.29750684637,], # 
    ]
    top = left[-1:] + [[2.0,50000.0]]

    right_side = top[-1:] + [[2.0,32500.0]]

    right_upper = [
        [2.0, 32500.0,], # [2.0002952907638636, 32461.014089902983,], # 
        [1.9804432847044802, 32119.46631046173,],
        [1.9592291467248155, 31675.71751104466,],
        [1.9380162661058895, 31206.863408894373,],
        [1.9168043913755533, 30717.92506455751,],
        [1.8955948382720078, 30182.631571959653,],
        [1.8743802961406777, 29746.952334135407,],
        [1.8531697730731347, 29231.02578936033,],
        [1.8319592500055921, 28715.099244585253,],
        [1.8107493556184182, 28186.620048443565,],
        [1.7895389897209673, 27667.555340826828,],
        [1.7683286238235167, 27148.490633210098,],
        [1.7471182579260658, 26629.425925593365,],
        [1.7179062675936612, 25900.0,], # 25876.129102571173,], # 
    ]
    
    right_lower = [
        [1.49045046913155, 25900.0,], # 25844.459542105982,], # 
        [1.458206713447848, 24869.217940403396,],
        [1.4287079967447909, 23930.417277661378,],
        [1.4067016472810452, 23243.15132025124,],
        [1.3863025104488362, 22587.886066787603,],
        [1.3651036179681189, 21839.735471730226,],
        [1.3439026822762026, 21132.380993614333,],
        [1.3227048226275198, 20363.60818559752,],
        [1.302470136480225, 19637.078881385893,],
        [1.283200370667629, 18917.91464253936,],
        [1.263930604855033, 18198.75040369283,],
        [1.244660839042437, 17479.58616484629,],
        [1.2263565324334276, 16757.027575907763,],
        [1.2070903500989343, 15966.313224271551,],
        [1.1887908107332459, 15148.568677013747,],
        [1.1704867209191736, 14421.681415838073,],
        [1.152185257342499, 13642.356947941655,],
        [1.1338857531529745, 12823.910049952625,],
        [1.1165494030056835, 12048.095190215841,],
        [1.0992157422133042, 11218.582877410772,],
        [1.0814147164724903, 10390.685778359148,],
        [1.067876319376042, 9779.179926781107,],
        [1.0529947598315323, 8964.359187281712,],
        [1.0366241042063828, 8180.451450309447,],
        [1.021219503284836, 7381.259212922923,],
        [1.0105237918328864, 6863.09690164691,],
        [0.9990772553287178, 6195.165195176829,],
        [0.9854229705534262, 5405.804455258025,],
        [0.9729353214950253, 4740.32946601251,],
        [0.962500502733682, 4113.897578399032,],
        [0.9509485323883279, 3484.945829094322,],
        [0.9374745349724475, 2685.252424931554,],
        [0.9249650417647358, 1901.232745847963,],
        [0.9094234947836944, 930.8278224403475,],
        [0.9, 0.0,], # [0.8982808131728891, 7.064759666267491,], # 
    ]

    right_plateau = right_upper[-1:] + right_lower[:1]
    right_plateau_alt = right_plateau[0][1]
    
    bottom = [[0.9,0.0],[x_start, 0.0,],]
    
    # combine lines
    fltenv = left[:-1] + top[:-1] + right_side[:-1] + \
        right_upper[:-1] + right_plateau[:-1] + right_lower[:-1] + bottom
    fltenv = np.array(fltenv)
    # create shapely object
    fltenv_sh = sh.geometry.LinearRing(fltenv)
    fltenv_poly = sh.geometry.Polygon(fltenv_sh)

    # make numpy arrays for convenience
    left = np.array(left)
    right_lower = np.flip(np.array(right_lower),axis=0)
    right_upper = np.flip(np.array(right_upper),axis=0)
    right_plateau = np.array(right_plateau)
    # add in envelope sides
    left_alt = run_alt*1.0; left_mach = np.interp(left_alt,left[:,1],left[:,0])
    right_lower_alt = run_alt[np.argwhere(run_alt < right_plateau_alt)[:,0]]
    right_lower_mach = np.interp(right_lower_alt,right_lower[:,1],right_lower[:,0])
    right_upper_alt = run_alt[np.argwhere(run_alt > right_plateau_alt)[:,0]]
    right_upper_mach = np.interp(right_upper_alt,right_upper[:,1],right_upper[:,0])
    right_plateau_alt = right_plateau[:,1]*1.0
    right_plateau_mach = right_plateau[:,0]*1.0
    # combine
    add_alt  = np.concatenate((left_alt ,right_lower_alt ,right_upper_alt ,right_plateau_alt ))
    add_mach = np.concatenate((left_mach,right_lower_mach,right_upper_mach,right_plateau_mach))

    if run_cases:
        # add in to mask, run arrays, and integers
        for i in range(len(add_alt)):
            addH = add_alt[i]
            addM = add_mach[i]
            
            # check in current array
            ileqH = np.argwhere(run_alt  <= addH)[-1,0]
            ileqM = np.argwhere(run_M    <= addM)[-1,0]

            # if in current array, just flip in mask
            if addH == run_alt[ileqH]:
                iHtoadd = ileqH*1
            else: # else, add to mask and run_alt and num_alt
                iHtoadd = ileqH+1
                run_alt = np.insert(run_alt,ileqH,addH,axis=0)
                num_alt += 1

            # if in current array, just flip in mask
            if addM == run_M[ileqM]:
                iMtoadd = ileqM*1
            else: # else, add to mask and run_alt and num_alt
                iMtoadd = ileqM+1
                run_M = np.insert(run_M,ileqM,addM,axis=0)
                num_M += 1
        
        run_alt = np.sort(run_alt)
        run_M   = np.sort(run_M  )

        # create mask
        run_mask = np.ones((num_M,num_alt),dtype=bool)
        tol_dist = 1e-6
        for iH in range(run_alt.shape[0]):
            for iM in range(run_M.shape[0]):
                # check if transonic
                if Mskip[0] < run_M[iM] < Mskip[1]:
                    continue

                if run_M[iM] == 0.0:
                    continue
                
                # check in flight env
                ipt = sh.Point(run_M[iM],run_alt[iH])
                is_contained = fltenv_poly.contains(ipt)
                intersects = fltenv_poly.intersects(ipt)
                closeto = fltenv_poly.distance(ipt) < tol_dist
                if is_contained or intersects or closeto:
                    run_mask[iM,iH] = False
        
        total_runs = int(np.abs(np.sum(run_mask - 1)))

        # filenames 
        bire_fs_file = "bire_fs_in.json"

        # read in json to ensure no file changes while running
        bire_fs_dict = json.loads( open(bire_fs_file).read() )

        # initialize BIRE
        compr = True # False # 
        stall = True # False # 
        fitthrust = True # False # 
        phi_trim = 0.0 # 30.0 # 10.0 # 
        cgshift = [0.0, 0.0, 0.0] # [1.0, 0.0, 0.0] # [0.5, 0.0, 0.0] # 
        subfolder_end = "" # "_m" # "_p" # 
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
        # bire = Aircraft(bire_fs_dict)
        bire_fs_dict["controller"] = {
        "enforce_update_frequency" : False,
        "update_frequency[hz]" : 100.0,
        "type" : "gains",
        "name" : "gains",
        "integral_states" : [0,3,4,5],
        "gains" : {
            "K" : [ [ -10.0,  0.0,  12.0],
                    [  0.0, -5.0, -4.0],
                    [  0.0,  4.0, 30.0]],
            "KI" :[ [ -1.0,  0.0,  0.0],
                    [  0.0, -5.0,  0.0],
                    [  0.0,  0.0,  5.0]]
        }
    }
        bire = ControlAllocationMomentAssignmentAircraft(bire_fs_dict)
        # bire = NonlinearDynamicInversionAircraft(bire_fs_dict)
        # bire = ITPIAircraft(bire_fs_dict)
        x0 = bire.x_trim_euler
        u0 = bire.u_trim
        # error threshold
        state_threshold = [
        10., 15., 15.,
        0.5, 0.5, 0.5, # 20., 10., 10., # 
        1., 1., 50.,
        25., 10., 1.,
        5., 5., 5., 0.05
    ]
        controller_dict = bire_fs_dict.get("controller",{})
        l_i = len(controller_dict.get("integral_states",[]))
        E = np.diag(1./(np.array(state_threshold + [1.0]*l_i)**2.))
        bire._build_controller(report=False,save_matrices=False,
            drop_actrs=True,
            mrrr=[0,1,2,6,7,8,9,10,11],
            mrrc=[3],
            run_freq=False,
            include_stall_derivatives=True,
            include_altitude_derivatives=True,
            skip_reporting=True,
            save_name_end="",save_folder="")
        CTC = np.matmul(bire.Lin_Model.C.T,bire.Lin_Model.C)
        CEC = np.matmul(CTC,np.matmul(E,CTC))
        bire.is_monte_carlo = True

        # reference signal
        phi__deg = 60.0
        phi__rad = np.deg2rad(phi__deg)
        t_start = 0.0 # 1.0 # 
        transition_time = 2.0 # 1.3 # 2.8 # 1.0 # 
        signal_type = "1-cosine_cont" # "triangle_cont" # "step" # "triangle" # "1-cosine" # "quartic_bump" # 
        # calculations
        p_wind = phi__deg/transition_time
        t__end = t_start + transition_time
        ref_dict = {
            "deg2rad_states" : [1,2,3,4,5],
            "sct_on_5" : False
        }
        n_points = 101
        t_tran = np.linspace(t_start,t__end,n_points)
        onemcos = 1.0 - cos(2.0*pi/transition_time*(t_tran-t_start))
        #
        t__mid = t_start + transition_time/2.0
        n_pointsq = int((n_points-1)/2)
        t_tranq = np.linspace(t__mid,t__end,n_pointsq)
        onemcosq = (1.0 - cos(2.0*pi/transition_time*(t_tranq-t__mid)))/2.0
        #

        # run trim cases
        bire.verbose_trim = False
        # initialize
        x_trims = np.zeros((num_M,num_alt,x0.shape[0]))
        u_trims = np.zeros((num_M,num_alt,u0.shape[0]))
        CFM_trims = np.zeros((num_M,num_alt,6))
        trim_slf_success = np.zeros((num_M,num_alt),dtype=bool)
        trim_sct_success = np.zeros((num_M,num_alt),dtype=bool)
        CAMA_run_Dx = np.full((num_M,num_alt,x0.shape[0]),2.0)
        counter = 1
        #
        for iH in range(run_alt.shape[0]):
            for iM in range(run_M.shape[0]):
                # run case
                if not(run_mask[iM,iH]):
                    # modify trim values
                    bire.H0 = run_alt[iH]
                    bire.V0 = run_M[iM]*bire.stdatm(bire.H0)[5]

                    # run trim
                    bire._initialize_state(no_report=True,no_print_fail=True)

                    slf_trim_failed = bire.trim_failed
                    bire_60_bank_failed = False
                    Dx = 2.0
                    if slf_trim_failed:
                        run_mask[iM,iH] = True
                        # plt.plot(run_M[iM],run_alt[iH],"or")
                    else:
                        u_trims[iM,iH] = bire.u_trim*1.0
                        x_trims[iM,iH] = bire.x_trim_euler*1.0
                        # aero info
                        a = atan2(bire.x_trim_euler[2],bire.x_trim_euler[0])
                        b = asin(bire.x_trim_euler[1]/bire.V0)
                        sos = bire.stdatm(-bire.x_trim_euler[8])[5]
                        M = bire.V0/sos
                        # nondimensionalize rates
                        pbar = (bire.x_trim_euler[3])*bire.bw/2./bire.V0
                        qbar = (bire.x_trim_euler[4])*bire.cw/2./bire.V0
                        rbar = (bire.x_trim_euler[5])*bire.bw/2./bire.V0
                        # pass in controls state
                        ail = bire.u_trim[0]
                        ele = bire.u_trim[1]
                        rud = bire.u_trim[2]
                        thr = bire.u_trim[3]
                        # use aircraft model
                        CFM_trims[iM,iH] = bire.aero_model.aero_results(*[
                            a,b,pbar,qbar,rbar,ail,ele,rud,
                            bire.is_compressible,M,
                            bire.use_anderson,bire.has_stall
                        ])
                        # plt.plot(run_M[iM],run_alt[iH],"ok")

                        if not(skip_60_and_sim):
                            # get 60 deg bank info
                            bire.phi_trim = phi__rad*1.0
                            u60trim,x60trim = bire.run_trim(verbose=False,
                                no_report=True,no_print_fail=True,
                                imax=bire.trim_iter_max)
                            bire.phi_trim = 0.0
                            if bire.trim_failed:
                                bire_60_bank_failed = True
                            else:
                                # build reference signal
                                a_SLF_rad = atan2(x_trims[iM,iH,2],
                                    x_trims[iM,iH,0])
                                a_SLF_deg = np.rad2deg(a_SLF_rad)
                                a_tr_deg = np.rad2deg(atan2(x60trim[2],
                                    x60trim[0]))
                                b_tr_deg = np.rad2deg(asin(x60trim[1]/bire.V0))
                                p_tr_deg = np.rad2deg(x60trim[3])
                                q_tr_deg = np.rad2deg(x60trim[4])
                                r_tr_deg = np.rad2deg(x60trim[5])
                                #
                                r_roll = p_wind*np.sin(a_SLF_rad) # 
                                p_roll = p_wind*np.cos(a_SLF_rad) # 
                                #
                                p_signal = p_roll*onemcos
                                p_signal[n_points-n_pointsq:] += \
                                    p_tr_deg*onemcosq
                                r_signal = r_roll*onemcos
                                r_signal[n_points-n_pointsq:] += \
                                    r_tr_deg*onemcosq
                                #
                                a_sig = np.vstack(
                                    (t_tranq,(a_tr_deg - a_SLF_deg)\
                                    *onemcosq + a_SLF_deg)).T.tolist()
                                b_sig = np.vstack(
                                    (t_tranq,b_tr_deg*onemcosq)).T.tolist()
                                p_sig = np.vstack((t_tran,p_signal)).T.tolist()
                                q_sig = np.vstack(
                                    (t_tranq,q_tr_deg*onemcosq)).T.tolist()
                                r_sig = np.vstack((t_tran,r_signal)).T.tolist()
                                # create signal
                                ref_dict["0"] = [
                                    [ 0.0, bire.V0],[ 2.0, bire.V0],]
                                ref_dict["1"] = a_sig
                                ref_dict["2"] = b_sig
                                ref_dict["3"] = p_sig
                                ref_dict["4"] = q_sig
                                ref_dict["5"] = r_sig
                                bire._build_reference_signal(ref_dict)
                            
                                # run simulation, roll to 60 deg bank
                                bire.first_step = True

                                # call run sim
                                try:
                                    xr,ur = bire.run_simulation(
                                        report_simulation=False,
                                        report_controller=False,
                                        report_trim=False,
                                        save_matrices=False,
                                        # mrrr=[3,4,5],mrrc=[0,1,2],
                                        delta_x0=None,
                                        include_stall_derivatives=True,
                                        include_altitude_derivatives=True,
                                        actr_warm_start=False,
                                        report_simulation_deltas=False)
                                    #
                                    r_track = bire._get_reference(bire.tf)
                                    rows = [3,4,5,9,10,11]
                                    r_track[rows] = np.rad2deg(r_track[rows])
                                    
                                    # pull out last state, check if zeros
                                    x_zero = xr[:,-1]*1.
                                    dx = x_zero - r_track
                                    Dx_norm = np.matmul(dx.T,np.matmul(CEC,dx))
                                except ValueError:
                                    Dx = 13.0
                                except:
                                    Dx = 11.0
                    
                    # report
                    caserep ="{:> 4d}/{:> 4d} : H = {:> 7.0f}, M = {:> 6.3f}".\
                        format(counter,total_runs,run_alt[iH],run_M[iM])
                    trimsimrep = ", Trim {:>4s}, 60 Trim {:>4s}, Dx = {:>6.3f}".\
                        format("Fail" if slf_trim_failed else " ",
                               "Fail" if bire_60_bank_failed else " ",
                               Dx)
                    print(caserep + trimsimrep)
                    trim_slf_success[iM,iH] = False if slf_trim_failed else True
                    trim_sct_success[iM,iH] = False if \
                        (slf_trim_failed or bire_60_bank_failed) else True
                    CAMA_run_Dx[iM,iH] = 2.0 if \
                        (slf_trim_failed or bire_60_bank_failed) else Dx
                    counter += 1
        # save cases
        cases_dict = {}
        cases_dict["run_alt"] = run_alt
        cases_dict["run_M"] = run_M
        cases_dict["run_mask"] = run_mask
        cases_dict["x_trims"] = x_trims
        cases_dict["u_trims"] = u_trims
        cases_dict["CFM_trims"] = CFM_trims
        cases_dict["trim_slf_success"] = trim_slf_success
        cases_dict["trim_sct_success"] = trim_sct_success
        cases_dict["CAMA_run_Dx"] = CAMA_run_Dx
        savemat(save_file,cases_dict)
    else:
        cases_dict = loadmat(save_file)
        run_alt = cases_dict["run_alt"][0]; num_alt = len(run_alt)
        run_M = cases_dict["run_M"][0]; num_M = len(run_M)
        run_mask = cases_dict["run_mask"]
        x_trims = cases_dict["x_trims"]
        u_trims = cases_dict["u_trims"]
        CFM_trims = cases_dict["CFM_trims"]
        trim_slf_success = cases_dict.get("trim_slf_success",x_trims[:,:,0]*0.0)
        trim_sct_success = cases_dict.get("trim_sct_success",x_trims[:,:,0]*0.0)
        CAMA_run_Dx = cases_dict.get("CAMA_run_Dx",x_trims[:,:,0]*0.0 + 13.0)

        
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
        MH_fig,MH_axs = plt.subplots(figsize=(3.5,3.25),constrained_layout=True)

        # plot flight envelope
        MH_axs.plot(fltenv[:,0],(fltenv[:,1])/1.0e3,"k",lw=1.0)
        
        # plot flight conditions
        lbl_params = dict(ha="left",va="bottom",size=8.0)
        bbox_dict = dict(facecolor="w",linewidth=0,alpha=0.8,
            boxstyle="Square, pad=0.0")
        mdict = dict(c="k",marker="o",ms=2.0,mew=1.0,mfc="w")
        FC_H = [1000.0,15000.0,1000.0,15000.0,30000.0,]
        FC_M = [0.2,0.19,0.8,0.6,0.8,]
        FC_N = ["T1","T2","C1","C2","C3",]
        kerf_M = 0.01
        kerf_H = 200.0
        for i in range(len(FC_H)):
            MH_axs.plot(FC_M[i],(FC_H[i])/1.0e3,**mdict)
            MH_axs.text(FC_M[i]+kerf_M,(FC_H[i]+kerf_H)/1.0e3,FC_N[i],bbox=bbox_dict,**lbl_params)

        # # plot all the points
        # for i in range(num_alt):
        #     for j in range(num_M):
        #         MH_axs.plot(run_M[j],(run_alt[i])/1.0e3,c="k",marker="o",ms=1.0)

        # test contourf
        de_trims = np.rad2deg(abs(u_trims[:,:,1])) # np.rad2deg(u_trims[:,:,1]) # 
        de_mask = np.ma.masked_array(de_trims,mask=run_mask).T
        # levels = [-25.0,-1.0,-0.5,-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3,0.5,1.0,25.0]
        levels = [0.0,0.1,0.2,0.3,0.4,0.5,1.0,24.0,25.0] # ,25.0] # 
        # cmap = plt.get_cmap('viridis', len(levels) - 1)
        cmap = plt.get_cmap('gray', len(levels) - 1)
        norm = BoundaryNorm(levels, cmap.N)
        cb = MH_axs.contourf(run_M,(run_alt)/1.0e3,de_mask,corner_mask=True,
            levels=levels,
            # cmap="viridis", # cmap="gray", # 
            cmap = cmap,
            norm=norm,
            ) # 
        fcb = MH_fig.colorbar(cb,)
        fcb.ax.minorticks_off()
        # MH_axs.clabel(cb, inline=True, fontsize=8)

        # bounds
        MH_axs.set_xlim((0.0,2.0))
        MH_axs.set_ylim((0.0,50.0)) # 50000.0

        # axes titles
        MH_axs.set_xlabel("Mach number")
        MH_axs.set_ylabel("Altitude, kft")

        # save plots
        plot_dict = dict(transparent=transparent,dpi=300.0)
        MH_fig.savefig(save_folder + "M_H_plot." + plot_format,**plot_dict)

        # show plots
        if show_plots:
            plt.show()
        else:
            plt.close("all")
    quit()