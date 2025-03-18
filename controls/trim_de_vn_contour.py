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

import shapely as sh


if __name__ == "__main__":

    # report
    print("running de vn contour...")

    # settings
    run_cases = True # False
    show_plots = True # False
    alt_low = 0.0; alt_high = 50000.0; num_alt = 11
    M_low   = 0.2; M_high   =     2.0; num_M   = 10
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
    
    bottom = [[0.9,0.0],[x_start, 0.0,],]
    
    # combine lines
    fltenv = left[:-1] + top[:-1] + right_side[:-1] + \
        right_upper[:-1] + right_plateau[:-1] + right_lower[:-1] + bottom
    fltenv = np.array(fltenv)
    # create shapely object
    fltenv_sh = sh.geometry.LinearRing(fltenv)
    fltenv_poly = sh.geometry.Polygon(fltenv_sh)
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
    bire = Aircraft(bire_fs_dict)
    x0 = bire.x_trim_euler
    u0 = bire.u_trim

    # run trim cases
    bire.verbose_trim = False
    if run_cases:
        for iH in range(run_alt.shape[0]):
            for iM in range(run_M.shape[0]):
                # check if transonic
                if Mskip[0] < run_M[iM] < Mskip[1]:
                    continue
                
                # check in flight env
                ipt = sh.Point(run_M[iM],run_alt[iH])
                if fltenv_poly.contains(ipt) or fltenv_poly.intersects(ipt): # sh.Point(run_M[iM],run_alt[iH]).within(fltenv_poly): # 
                    print("H = {:> 7.0f}, M = {:> 6.3f}".format(run_alt[iH],run_M[iM]))

                    # modify trim values
                    bire.H0 = run_alt[iH]
                    bire.V0 = run_M[iM]*bire.stdatm(bire.H0)[5]

                    # run trim
                    bire._initialize_state(no_report=True)

                    if bire.trim_failed:
                        plt.plot(run_M[iM],run_alt[iH],"or")
                    else:
                        plt.plot(run_M[iM],run_alt[iH],"ok")

    plt.plot(fltenv[:,0],fltenv[:,1],"b")
    plt.show()
