import numpy as np
import json
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from cycler import cycler
import mpl_toolkits.mplot3d.axes3d as ax3
from matplotlib.animation import FuncAnimation
from numpy import sign, matmul as mm
from datetime import datetime, timedelta
from time import sleep
import control as co
from scipy.linalg import block_diag
from scipy.integrate import ode, odeint
from scipy.interpolate import interp1d, interpn
from scipy.optimize import curve_fit,minimize,minimize_scalar,newton
from scipy.io import savemat, loadmat
from scipy.signal import tf2zpk as scipy_tf2zpk
# from math import pi, sin, cos, tan, exp, asin, acos, atan, atan2
from numpy import pi, sin, cos, tan, exp, arcsin as asin, arccos as acos, arctan as atan, arctan2 as atan2
from std_atm import stdatm_english
from quat import quat_mult, euler_2_quat, quat_2_euler, quat_norm, body_2_fixed, fixed_2_body, eulerdot_2_quatdot, quatdot_2_eulerdot
from linearization import linearization as lin,Anderson_correction_der_coeff,Anderson_correction_der_M

from controller_simulation import Aircraft,run_single_simulation, \
    monte_carlo_perturbations, report_latex, report_eigprops, rep2D,BIREAero


class DirectShootingAircraft(Aircraft):
    """A default class for calculating and containing the mass properties of a
    Cuboid.

    Parameters
    ----------
    input_vars : dict , optional
        Must be a python dictionary
    """
    def __init__(self,input_dict={}):

        # invoke init of parent
        Aircraft.__init__(self,input_dict,folder_prefix = "track")
        self.tracking = True
        #
        self.first_step = True

    def _build_tracking_gains(self):

        # build controller
        if self.first_step:
            # build system, solve problem
            A_tr,B_tr = self._build_linear_slf_model([0,1,2,3,4,5],[0,1,2])

            # flip bool
            self.first_step = False

    def _get_control(self,t,x,is_controlled=True,given_control=False,u="o",
        force_control_to_inputs=False):
        # build control or pass through
        if not given_control:
            if is_controlled and (not(self.enforce_update_frequency) or 
                (self.enforce_update_frequency and self.can_update) ):
                if self.use_quaternions:
                    x_euler = self.quat2euler_state(x)
                else:
                    x_euler = x*1.
                    # reset angles
                    x_euler[9:12] = quat_2_euler(euler_2_quat(x_euler[9:12]))
                #
                ref = self._get_reference(t)[self.Lin_Model.Cslice]
                refdot = self._get_reference_derivative(t)[self.Lin_Model.Cslice]
                # per dave, full stick should be 270 deg/s in aileron
                # 120 deg/s in elevator
                # 60 deg/s in rudder
                #

                if self.first_step:
                    self._build_tracking_gains()
                
                #-------------------#
                # STATE DEFINITIONS #
                #-------------------#
                u = np.array([
                    np.interp(t,self.t__points,self.u_box[0]),
                    np.interp(t,self.t__points,self.u_box[1]),
                    np.interp(t,self.t__points,self.u_box[2]),
                    np.interp(t,self.t__points,self.u_box[3]),
                ])


                if self.order > 0:
                    q = 1*self.use_quaternions
                    inputs = x[12+q:16+q]*1.
                else:
                    inputs = u*1.
                # #
                self.u_til_next_update = u*1.
                self.can_update = False
            elif is_controlled and self.enforce_update_frequency and \
                not(self.can_update):
                u = self.u_til_next_update*1.
                if self.order > 0:
                    q = 1*self.use_quaternions
                    ## INTSTATE
                    inputs = x[12+q:16+q]*1.
                else:
                    inputs = u*1.
            else:
                inputs = u = self.Lin_Model.uhat_eq*1.
        elif given_control:
            if u[0] == "o":
                raise TypeError("Control input required.")
            else:
                if self.order > 0 and not force_control_to_inputs:
                    q = 1*self.use_quaternions
                    inputs = x[12+q:16+q]*1.
                else:
                    inputs = u*1.
        
        # limit actuators
        # #vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        if self.integrator == "odeint":
            u = self._limit_input(u)
        # #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        inputs = self._limit_input(inputs)
        if self.order > 0:
            q = 1*self.use_quaternions
            x[12+q:16+q] = np.array(inputs)*1.
        # quantize actuators
        inputs = self._quantize_input(inputs)

        return u,inputs



if __name__ == "__main__":

    # filenames 
    # base_fs_file = "base_fs_in.json"
    bire_fs_file = "bire_fs_in.json"
    # base_rc_file = "base_rc_in.json"
    # bire_rc_file = "bire_rc_in.json"

    # read in json to ensure no file changes while running
    # base_fs_dict = json.loads( open(base_fs_file).read() )
    bire_fs_dict = json.loads( open(bire_fs_file).read() )
    # base_rc_dict = json.loads( open(base_rc_file).read() )
    # bire_rc_dict = json.loads( open(bire_rc_file).read() )

    # changes
    bire_fs_dict["simulation"]["use_quaternions"] = False # True # 
    run_opt = True # False # 
    bank_angle_deg = 10.0 # 60.0 # 

    # build aircraft
    bire = DirectShootingAircraft(bire_fs_dict)
    bire_fs_dict["initial"]["trim"]["bank_angle[deg]"] = bank_angle_deg
    if 25.0 <= bank_angle_deg <= 55.0:
        bire_fs_dict["initial"]["trim_guess"] = {}
        bire_fs_dict["initial"]["trim_guess"]["elevator[deg]"] = 25.0
        bire_fs_dict["initial"]["trim_guess"]["BIRE[deg]"] = 0.0
    biresct = DirectShootingAircraft(bire_fs_dict)
    print("bire sct bank  [deg]   =",bank_angle_deg)
    print("bire sct rates [deg/s] =",np.rad2deg(biresct.x_trim[3:6]))

    # build control inputs
    num_nodes = 11 # 21 # 
    da_points = np.zeros((num_nodes,)) + bire.u_trim[0]
    de_points = np.zeros((num_nodes,)) + bire.u_trim[1]
    dB_points = np.zeros((num_nodes,)) + bire.u_trim[2]
    ta_points = np.zeros((num_nodes,)) + bire.u_trim[3]
    t__points = np.linspace(0.0,2.0,num_nodes)

    u__points = np.vstack((da_points,de_points,dB_points,ta_points))

    use_other = "10_deg"
    if   use_other == "the_great_before":
        u_pts = np.array([
            [ 1.77970058e-02, -1.72070668e-01, -1.98095527e-01,
            -1.64128993e-01, -1.88862752e-01,  2.75933662e-01,
             1.11108972e-01,  3.16955278e-02, -4.34863858e-03,
            -7.22010141e-02, -1.77070909e-02, -8.99376112e-02,
            -8.36623576e-02, -9.39961647e-02, -1.05965290e-01,
            -5.24094450e-02, -3.52006494e-02,  9.30767603e-03,
             1.11683345e-01,  4.34904384e-02,  4.94245702e-02],
           [-1.83060528e-02,  8.64445344e-02,  1.52573768e-03,
             1.51301452e-02, -2.35842663e-02,  1.27098481e-01,
             2.09670111e-02,  6.14413829e-02,  3.29695529e-02,
             2.68259246e-02,  3.13248958e-02, -5.39788120e-03,
            -7.25361756e-02, -9.28777872e-02, -8.78471452e-02,
            -9.03708595e-02, -7.22409320e-02, -8.95545602e-02,
            -6.93832092e-02, -1.22089585e-01, -3.47135213e-02],
           [-4.00425455e-07,  1.22669912e-05,  1.13034594e-05,
            -4.51569413e-07,  1.17581016e-05,  3.08231401e-06,
             1.25159075e-05,  8.60120056e-06,  4.79389002e-07,
            -1.03109892e-05, -4.29688877e-06, -4.72971310e-06,
            -1.11744846e-05,  3.23616451e-06,  7.88243887e-06,
             9.53568999e-06,  6.91431794e-06,  1.00498903e-05,
             4.25665385e-06,  3.87119671e-06,  1.03008451e-06],
           [ 3.00004390e-01,  2.95915320e-01,  3.06642907e-01,
             3.46433764e-01,  3.43659789e-01,  3.33993858e-01,
             3.13887291e-01,  3.55008266e-01,  3.23245432e-01,
             3.66321573e-01,  3.90780349e-01,  2.76004900e-01,
             2.70965979e-01,  2.75026896e-01,  2.74291264e-01,
             2.72823386e-01,  2.77967284e-01,  2.67845668e-01,
             2.72534279e-01,  2.73405166e-01,  2.73215001e-01]
        ]); t_pts = np.linspace(0.0,2.0,u_pts.shape[1])
        # split out constraints
        u_pts = np.array([
            3.02713853e-01,  3.38965336e-01, -2.30916378e-01, -3.31805917e-01,
            -3.07194981e-01, -1.67761517e-01,  1.27323537e-01,  5.75078975e-02,
            5.19542949e-03, -7.32828072e-02, -8.17843457e-02, -8.30059293e-02,
            -1.43247427e-01, -1.06598014e-01, -1.03181104e-01, -6.10361486e-02,
            -4.84778156e-02, -9.30267210e-03,  7.62673854e-02,  1.18030015e-01,
            3.35862369e-01,  4.47237017e-02,  1.09113256e-01,  4.04499162e-02,
            6.20070353e-02,  2.98718310e-02, -4.26142221e-02, -1.26878263e-01,
            -9.72694710e-02, -1.94532557e-02,  3.08636584e-02,  4.27818694e-02,
            -1.16173914e-02, -3.37405932e-02, -2.02832766e-02, -3.97778688e-02,
            -4.37387745e-02, -6.46562747e-02, -5.44138354e-02, -6.73204083e-02,
            2.28532369e-01,  1.20443660e-01,  3.16250189e-05,  6.17222597e-05,
            3.68161728e-05, -5.83800339e-05, -1.23933951e-04, -4.21409441e-05,
            7.74324087e-05,  7.66581220e-05, -3.61923543e-05,  1.38290912e-05,
            -1.33993996e-04,  5.12047036e-05, -6.29454139e-06, -1.01747341e-05,
            -1.65289810e-04, -4.14908772e-05, -1.01009630e-04,  4.83333420e-05,
            2.39873165e-05,  2.77848295e-05, -4.17022013e-05,  2.87174269e-01,
            2.49245341e-01,  2.86346696e-01,  3.11513602e-01,  2.91095000e-01,
            2.71049009e-01,  2.41710942e-01,  2.70828517e-01,  2.57234831e-01,
            3.03418398e-01,  2.83307234e-01,  2.82175095e-01,  2.87012681e-01,
            2.83032492e-01,  2.75683970e-01,  2.38470700e-01,  2.62271584e-01,
            2.83857594e-01,  2.86384123e-01,  2.75482590e-01,  2.73268857e-01,
        ]).reshape((4,21)); t_pts = np.linspace(0.0,2.0,u_pts.shape[1])
        # reinterp to current setup
        u__points[0] = np.interp(t__points,t_pts,u_pts[0])
        u__points[1] = np.interp(t__points,t_pts,u_pts[1])
        u__points[2] = np.interp(t__points,t_pts,u_pts[2])
        u__points[3] = np.interp(t__points,t_pts,u_pts[3])
    elif use_other == "10_deg":
        # 10 deg bank
        u_pts = np.array([
            [-3.00748757e-04,  3.82031497e-02, -9.87801241e-03,
            -1.50195093e-02,  1.37053408e-02, -2.01954494e-02,
            -8.35420930e-02, -4.82043714e-02,  7.25924095e-02,
            9.42454303e-05,  2.93741565e-04],
        [ 3.03097836e-02, -6.59673914e-03, -2.39562085e-02,
            1.39285195e-02, -1.12832550e-02,  2.44570976e-02,
            -1.43605428e-02,  1.59999906e-02, -1.16576276e-02,
            3.65396778e-03,  1.00881388e-03],
        [ 3.82552913e-02,  2.55362212e-02,  5.02916003e-02,
            1.90867177e-02,  5.92290542e-02,  3.36807018e-02,
            -5.85986950e-02, -2.78525410e-02,  4.55377617e-02,
            3.79994878e-03,  4.44138202e-03],
        [ 1.64307205e-01,  1.66953773e-01,  2.08837592e-01,
            1.61486872e-01,  2.05945379e-01,  1.78308395e-01,
            2.31292334e-01,  2.58535322e-01,  2.90139789e-01,
            4.36677943e-01,  2.75312903e-01]
        ]); t_pts = np.linspace(0.0,2.0,u_pts.shape[1])
        print(u_pts.shape,t_pts.shape)
        # reinterp to current setup
        u__points[0] = np.interp(t__points,t_pts,u_pts[0])
        u__points[1] = np.interp(t__points,t_pts,u_pts[1])
        u__points[2] = np.interp(t__points,t_pts,u_pts[2])
        u__points[3] = np.interp(t__points,t_pts,u_pts[3])





    u_flatin = u__points.reshape((num_nodes*4,))

    # build bounds
    inbounds  = (( bire.min_da, bire.max_da),)*num_nodes
    inbounds += (( bire.min_de, bire.max_de),)*num_nodes
    inbounds += (( bire.min_dr, bire.max_dr),)*num_nodes
    inbounds += ((bire.min_tau,bire.max_tau),)*num_nodes
    
    # build constraint
    def constraints(u_flat,ts,x0,bire,biresct):
        u_box = u_flat.reshape((4,num_nodes))
        xs = odeint(ode,x0,ts,args=(bire,u_box),tfirst=True,
            atol=1e-10,rtol=1e-10
            ).T
        
        xdot = ode(ts[-1],xs[:,-1],bire,u_box)
        xdot = np.delete(xdot,[6,7,11])
        # xdot[6] = xdot[7] = xdot[11] = 0.0
        
        print("p,q,r    [deg/s] =",np.rad2deg(xs[3:6,-1]))
        print("p,q,r tr [deg/s] =",np.rad2deg(biresct.x_trim[3:6]))
        print("phi      [deg]   =",np.rad2deg(xs[9,-1]))
        print("phi   tr [deg]   =",np.rad2deg(biresct.x_trim[9]))
        print("dynamics squared =",np.linalg.norm(xdot)**2.0)
        print("-"*30)
        dax = np.rad2deg(np.max(u_box[0])); dan = np.rad2deg(np.min(u_box[0]))
        dex = np.rad2deg(np.max(u_box[1])); den = np.rad2deg(np.min(u_box[1]))
        dBx = np.rad2deg(np.max(u_box[2])); dBn = np.rad2deg(np.min(u_box[2]))
        tax =            np.max(u_box[3]) ; tan =            np.min(u_box[3])
        print("min da  [deg] = {:>7.3f}, max da  [deg] = {:>7.3f}".format(dan,dax))
        print("min de  [deg] = {:>7.3f}, max de  [deg] = {:>7.3f}".format(den,dex))
        print("min dB  [deg] = {:>7.3f}, max dB  [deg] = {:>7.3f}".format(dBn,dBx))
        print("min tau [p-u] = {:>7.3f}, max tau [p-u] = {:>7.3f}".format(tan,tax))
        print()
        print()

        cnstr = np.concatenate((
                np.array([(xs[9,-1]-biresct.x_trim_euler[9]),
                         (xs[3,-1]-biresct.x_trim[3]),
                         (xs[4,-1]-biresct.x_trim[4]),
                         (xs[5,-1]-biresct.x_trim[5]),
                         ]),
                xdot
                ))
        print(cnstr)
        
        return cnstr
    
    # ode
    def ode(t,x,bire,u_box):

        uinterp = np.array([
            np.interp(t,t__points,u_box[0]),
            np.interp(t,t__points,u_box[1]),
            np.interp(t,t__points,u_box[2]),
            np.interp(t,t__points,u_box[3]),
        ])

        xdot = bire._nonlinear_euler_dynamics(t,x,
            is_controlled=True,given_control=True,u=uinterp,
            force_control_to_inputs=False)
        return xdot
    
    # odeint
    def optfun(u_flat,ts,x0,bire):
        u_box = u_flat.reshape((4,num_nodes))
        xs = odeint(ode,x0,ts,args=(bire,u_box),tfirst=True,
            atol=1e-10,rtol=1e-10
            ).T
        
        return np.linalg.norm(u_box)**2.0
    
    # optimization
    if run_opt:
        res = minimize(optfun,u_flatin,args=(t__points,bire.x_trim,bire),
            method="SLSQP", # "COBYLA", # 
            bounds=inbounds,
            constraints=[
                dict(type="eq",fun=constraints,args=(t__points,bire.x_trim,bire,biresct)),
            ],
            options=dict(disp=True),
        )
        print(res)
        soln = res.x.reshape(4,num_nodes)
    else:
        print("not running optimization...")
        soln = u_flatin.reshape(4,num_nodes)
    print("result =")
    print(repr(soln))
    

    # run sim with this control
    def modify_fun(craft):
        craft.t__points = t__points*1.0
        craft.u_box = soln*1.0
        return craft

    plot_vars = {
        "show" : False,
        "plot_full" : True,
        "plot_delta" : True,
        "zoom_deltas" : True,
        # "zoom_fraction" : 0.05,
        "zoom_fraction" : 0.13333333333333333333333,
        "transparent" : False, # True, # 
        "format" : "pdf"
    }

    # bire FM
    bire_fs_FM_errs = [
        0.25  , # CL
        0.25  , # CS
        0.25  , # CD
        0.25  , # Cl
        0.25  , # Cm
        0.25   # Cn
    ]
    # base FM
    base_fs_FM_errs = [
        0.25  , # CL
        0.25  , # CS
        0.25  , # CD
        0.25  , # Cl
        0.25  , # Cm
        0.25   # Cn
    ]

    flight_conditions = {
        "T1" : { "m" : 0.2 , "h" :  1000., "V" : 222., "Re" : 15641000. },
        "T2" : { "m" : 0.19, "h" : 15000., "V" : 201., "Re" :  9919000. },
        "C1" : { "m" : 0.8 , "h" :  1000., "V" : 890., "Re" : 62563000. },
        "C2" : { "m" : 0.6 , "h" : 15000., "V" : 634., "Re" : 31324000. },
        "C3" : { "m" : 0.8 , "h" : 30000., "V" : 796., "Re" : 25828000. },
        "B1" : { "m" : 0.8 , "h" : 15000., "V" :   0., "Re" :        0. },
        "B2" : { "m" : 0.2 , "h" :     0., "V" :   0., "Re" :        0. },
        "B3" : { "m" : 0.4 , "h" :     0., "V" :   0., "Re" :        0. },
        # "B4" : { "m" : 0.6 , "h" :     0., "V" :   0., "Re" :        0. },
        # "B5" : { "m" : 0.8 , "h" :     0., "V" :   0., "Re" :        0. },
    }
    f1 = "C2" # "B4" # "B5" # "C1" # "B3" # "B2" # "B1" # 
    f2 = "C3"
    state_threshold = [
        10., 15., 15.,
        0.5, 0.5, 0.5, # 20., 10., 10., # 
        1., 1., 50.,
        25., 10., 1.,
        5., 5., 5., 0.05
    ]

    run_base_fs = {
        "aircraft_class" : DirectShootingAircraft,
        "actr_warm_start" : False,
        "num" : 1000,
        "final_time" : 5., # 120., # 
        "track_check_time" : 1.,
        # "time_step" : 0.01,
        # "initial_velocity" : 100.,
        "initial_mach" : flight_conditions[f1]["m"],
        "initial_altitude" : flight_conditions[f1]["h"], # 4500., # 
        "trim_bank" :  0.0,
        "trim_climb" : 0.0,
        # "start_climbing" : False,
        # "end_gs_climbing" : False,
        # "final_mach" : flight_conditions[f1]["m"]*1., # f2]["m"]*1., # 
        # "final_altitude" : flight_conditions[f1]["h"]*1., # f2]["h"]*1., # 
        "t_gain_schedule" : 0.1, # 90., # 
        "gain_steps" : 2,
        "cut_mine" : True,
        "save_data" : True,
        "statistical" : True,
        "has_turbulence" : False,
        "turbulence_setting" : "light", # "moderate", # "severe", # 
        "has_model_error" : False,
        "FM_errors" : base_fs_FM_errs,
        "state_threshold" : state_threshold, # 64.0, # 
        "random_seed" : 13,
        "turbulence_random_seed" : 15, # 13, # 
        "error_random_seed" : 14, # 13, # 
        "rerandomize_turbulence" : True,
        "mrrr" : [0,1,2,6,7,8,9,10,11],
        "mrrc" : [3],
        "get_aero_FM" : True,
        "include_stall_derivatives" : True, # False, # 
        "include_altitude_derivatives" : True, # False, # 
        "skip_simulation" : False, # True, # 
        "skip_video" : True, # False, # 
        "plot_ul_bounds" : False,
        "name_end" : "_" + f1 + "_AA_1" # "_DI_1" # 
        # 4 -- incr wt on tau, decr wt on da,de
        # 5 -- decr wt on da
    }
    run_bire_fs = {**run_base_fs}
    run_bire_fs["FM_errors"] = bire_fs_FM_errs
    # run_base_rc = {**run_base_fs}
    # run_base_rc.pop("initial_mach")
    # run_base_rc["initial_velocity"] = 100.
    # run_base_rc["initial_altitude"] = 4500.
    # run_base_rc["FM_errors"] = base_rc_FM_errs
    # run_base_rc["name_end"] = "_" + "LGN" + run_base_fs["name_end"][3:]
    # run_bire_rc = {**run_base_rc}
    # run_bire_rc["FM_errors"] = bire_rc_FM_errs
    # run_bire_rc["mrrc"] = [3]

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
    

    # per dave, max throws would be p=270deg/s,q=120deg/s,r=60deg/s
    # from 2nd to last flight test:
    # about 1/6 throw was max commanded in flight

    # run single case
    # # 
    plot_vars["plot_full"] = True # False # 
    plot_vars["plot_delta"] = False # True # 
    plot_vars["zoom_deltas"] = False
    plot_vars["format"] = "png" # "pdf" # 
    # plot_vars["format"] = "pdf" # "png" # 
    plot_vars["output_states"] = True # False # 
    plot_vars["plot_norm"] = False # True # 
    #
    di = [0.,0.,0.]
    # di = [5.,10.,7.] # see below
    run_base_fs["num"] = run_bire_fs["num"] = 1
    # # #
    run_bire_fs["name_end"] = "_" + f1 + "_DS" # 2" # 
    #
    #
    # #
    # #
    # plot_vars["format"] = "png"
    # cnb_scale = abs(-0.31764903243224546/(0.31301066324220633 - 0.0326))
    # # # #
    # # #
    # # # 
    bire_fs_dict["aircraft"]["CG_shift[ft]"] = cg = [0.0, 0.0, 0.0] # [1.0, 0.0, 0.0] # [0.5, 0.0, 0.0] # 
    blm = 50.0 # 200.0 # 100.0 # 
    bire_fs_dict["actuators"]["BIRE"]["rate_limits[deg/s]"] = [-blm,blm]
    bire_fs_dict["aircraft"]["surface_effectiveness_scaling"] = ses = 1.0 # 1.2 # 1.1 # 
    bire_fs_dict["aircraft"]["yaw_stability_offset_scaling"] = yss = 1.0 # 1.4 # 2.0 # cnb_scale # -1.0 # 0.0 # 
    bire_fs_dict["controller"]["CAMA_coupled_weighting"] = cw = True # False # 
    bire_fs_dict["controller"]["CAMA_SAS_on"] = sason = False # True # 
    run_bire_fs[ "has_turbulence"] = False # True # 
    representative_response = False # True # 

    elm = 60.0 # 120.0 # 
    bire_fs_dict["actuators"]["elevator"]["rate_limits[deg/s]"] = [-elm,elm]
    run_bire_fs["name_end"] += "_SR" + str(int(elm)) if (elm != 60.0) else ""

    ctrl_type = run_bire_fs["name_end"][1:].split("_")[1:]
    
    phi__deg = 60.0

    # ensure tail near zero soln
    bire_fs_dict["initial"]["trim_guess"] = {
        # # tail negative
        # "elevator[deg]" : -25.0,
        # "BIRE[deg]" : -90.0,
        # tail near zero
        "elevator[deg]" : 25.0,
        "BIRE[deg]" : 0.0,
        # # tail positive
        # "elevator[deg]" : -25.0,
        # "BIRE[deg]" : +90.0,
    }


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # phi__deg = 30.0 # 50.0 # 60.0 # 40.0 # 10.0 # 0.0 # 20.0 # 15.0 # 25.0 # 35.0 # 90.0 # 180.0 # 360.0 # 
    tail = "0" # "+" # "-" # 
    # left_roll = True # False # 
    ###########################################################################
    # trim properties
    if   f1 == "C1":
        V_trim = 890.0843173837532731 # B1, SLF
        a_SLF_deg =  0.2218653843049645 # B1, SLF
    elif f1 == "C2":
        V_trim = 634.4133153512273111
        a_SLF_deg =  2.6447774345356545
    elif f1 == "B1":
        V_trim = 845.8844204683031194 # B1, SLF
        a_SLF_deg =  0.8428726926131993 # B1, SLF
    elif f1 == "B2":
        V_trim = 223.2899911643457926 # B2, SLF
        a_SLF_deg = 19.9761233434501335 # B2, SLF
    elif f1 == "B3":
        V_trim = 446.5799823286915853 # B3, SLF
        a_SLF_deg =  4.0402710922527971 # B3, SLF
    else: raise ValueError("f1 must be in ['C2','B1','B2'], not " + str(f1))
    a_SLF_rad = np.deg2rad(a_SLF_deg)
    #
    if   f1 == "C1":
        if   phi__deg ==   0.0: # #   0 deg bank fullscale BIRE
            a_tr_deg =  0.2218653843049645
            b_tr_deg =  0.0
            p_tr_deg =  0.0
            q_tr_deg =  0.0
            r_tr_deg =  0.0
        elif phi__deg ==  60.0: # #  60 deg bank fullscale BIRE
            a_tr_deg =  1.1061067542177971
            b_tr_deg =  0.0009142276046984
            p_tr_deg = -0.0346684315030015
            q_tr_deg =  3.1055972046896820
            r_tr_deg =  1.7930173821221378
    elif f1 == "C2":
        if   phi__deg ==   0.0: # #   0 deg bank fullscale BIRE
            a_tr_deg =  2.6447774345356545
            b_tr_deg =  0.0
            p_tr_deg =  0.0
            q_tr_deg =  0.0
            r_tr_deg =  0.0
        elif phi__deg ==  10.0: # #  10 deg bank fullscale BIRE
            a_tr_deg =  2.6968631334184625
            b_tr_deg =  0.0020460560691901
            p_tr_deg = -0.0236847366216922
            q_tr_deg =  0.0886486340380570
            r_tr_deg =  0.5027513865539764
            # # # # # # xcg 1 ft forward
            # a_tr_deg =  2.9223663445550123
            # b_tr_deg =  0.0001856786905957
            # p_tr_deg = -0.0260490036020453
            # q_tr_deg =  0.0886079085570975
            # r_tr_deg =  0.5025204208742158
        elif phi__deg ==  15.0: # #  15 deg bank fullscale BIRE
            a_tr_deg =  2.7639052703244644
            b_tr_deg =  0.0045440564259333
            p_tr_deg = -0.0361891562749016
            q_tr_deg =  0.2007714630167870
            r_tr_deg =  0.7492893006885849
        elif phi__deg ==  20.0: # #  20 deg bank fullscale BIRE
            a_tr_deg =  2.8615961861283603
            b_tr_deg =  0.0164208410323567
            p_tr_deg = -0.0495920266927013
            q_tr_deg =  0.3603497293338741
            r_tr_deg =  0.9900527444514043
        elif phi__deg ==  25.0: # #  25 deg bank fullscale BIRE # # (0) tail
            a_tr_deg =  2.9942923664414067
            b_tr_deg = -0.0166552038854683
            p_tr_deg = -0.0638179984370310
            q_tr_deg =  0.5703966435396264
            r_tr_deg =  1.2232195495061524
        elif phi__deg ==  30.0: # #  30 deg bank fullscale BIRE
            if   tail == "-": # # (-) tail
                a_tr_deg =  3.1689348639448318
                b_tr_deg =  0.1363811510137339
                p_tr_deg = -0.0820880039056245
                q_tr_deg =  0.8352580178704386
                r_tr_deg =  1.4467093243808735
            elif tail == "0": # # (0) tail
                a_tr_deg =  3.1681819496642230
                b_tr_deg = -0.0057832719998301
                p_tr_deg = -0.0800043056586719
                q_tr_deg =  0.8353731041767737
                r_tr_deg =  1.4469086597107013
            elif tail == "+": # # (+) tail
                a_tr_deg =  3.1693583578538327
                b_tr_deg = -0.1248724015269359
                p_tr_deg = -0.0783041992237063
                q_tr_deg =  0.8354699615635688
                r_tr_deg =  1.4470764216257186
        elif phi__deg ==  35.0: # #  35 deg bank fullscale BIRE # # (0) tail
            a_tr_deg =  3.3925013835398783
            b_tr_deg = -0.0034327708776077
            p_tr_deg = -0.0982988942950006
            q_tr_deg =  1.1619249236475548
            r_tr_deg =  1.6594007636912391
        elif phi__deg ==  40.0: # #  40 deg bank fullscale BIRE # # (0) tail
            a_tr_deg =  3.6804967969783347
            b_tr_deg = -0.0021978760255690
            p_tr_deg = -0.1195200902391827
            q_tr_deg =  1.5598776114662467
            r_tr_deg =  1.8589897474721753
        elif phi__deg ==  50.0: # #  50 deg bank fullscale BIRE # # (0) tail
            a_tr_deg =  4.5364040558706025
            b_tr_deg = -0.0003511907142316
            p_tr_deg = -0.1755564353175651
            q_tr_deg =  2.6372142861590873
            r_tr_deg =  2.2128855348515439
        elif phi__deg ==  60.0: # #  60 deg bank fullscale BIRE
            a_tr_deg =  6.0682014265285318
            b_tr_deg =  0.0021390527127303
            p_tr_deg = -0.2654216358834438
            q_tr_deg =  4.3218126454667702
            r_tr_deg =  2.4951996942473698
        elif phi__deg ==  90.0: # #  90 deg bank fullscale BIRE
            a_tr_deg =  0.0
            b_tr_deg =  2.6447774345356545
            p_tr_deg =  0.0
            q_tr_deg =  0.0
            r_tr_deg =  0.0
        elif phi__deg == 180.0: # # 180 deg bank fullscale BIRE
            a_tr_deg = -3.7617756892695953
            b_tr_deg =  0.0
            p_tr_deg =  0.0
            q_tr_deg =  0.0
            r_tr_deg =  0.0
        elif phi__deg == 360.0: # # 360 deg bank fullscale BIRE
            a_tr_deg =  2.6447774345356545
            b_tr_deg =  0.0
            p_tr_deg =  0.0
            q_tr_deg =  0.0
            r_tr_deg =  0.0
    elif f1 == "B1":
        if   phi__deg ==   0.0: # #   0 deg bank fullscale BIRE
            a_tr_deg =  0.8428726926131993
            b_tr_deg =  0.0
            p_tr_deg =  0.0
            q_tr_deg =  0.0
            r_tr_deg =  0.0
        elif phi__deg ==  60.0: # #  60 deg bank fullscale BIRE
            a_tr_deg =  2.3751832188859936
            b_tr_deg =  0.0027158494337641
            p_tr_deg = -0.0782415781128383
            q_tr_deg =  3.2607339316364152
            r_tr_deg =  1.8825856131860317
    elif f1 == "B2":
        if   phi__deg ==   0.0: # #   0 deg bank fullscale BIRE
            a_tr_deg = 19.9761233434501335
            b_tr_deg =  0.0
            p_tr_deg =  0.0
            q_tr_deg =  0.0
            r_tr_deg =  0.0
        elif phi__deg ==  60.0: # #  60 deg bank fullscale BIRE
            a_tr_deg = 35.2415227086718374
            b_tr_deg =  0.1538466652245341
            p_tr_deg = -3.9073530967961800
            q_tr_deg =  9.5025388975674421
            r_tr_deg =  5.4862933904954536
    elif f1 == "B3":
        if   phi__deg ==   0.0: # #   0 deg bank fullscale BIRE
            a_tr_deg =  4.0402710922527971
            b_tr_deg =  0.0
            p_tr_deg =  0.0
            q_tr_deg =  0.0
            r_tr_deg =  0.0
        elif phi__deg ==  60.0: # #  60 deg bank fullscale BIRE
            a_tr_deg =  8.8938589320100387
            b_tr_deg =  0.0101262361323840
            p_tr_deg = -0.5520641780910568
            q_tr_deg =  6.0983959762207327
            r_tr_deg =  3.5209105584959723
    else: ValueError("Maneuver not planned, f1 = "+str(f1)+", phi = "+str(phi__deg))
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    if "left_roll" in locals():
        phi__deg = - phi__deg
        p_tr_deg = - p_tr_deg
        r_tr_deg = - r_tr_deg
    ###########################################################################
    t_start = 0.0 # 1.0 # 
    transition_time = 2.0 # 1.3 # 2.8 # 1.0 # 
    signal_type = "1-cosine_cont" # "triangle_cont" # "step" # "triangle" # "1-cosine" # "quartic_bump" # 
    # calculations
    p_wind = phi__deg/transition_time
    r_roll = p_wind*np.sin(a_SLF_rad) # 
    p_roll = p_wind*np.cos(a_SLF_rad) # 
    t__end = t_start + transition_time
    bire_fs_dict["reference"] = {
        "deg2rad_states" : [1,2,3,4,5],
        "0" : [[ 0.0,   V_trim],[ 2.0,   V_trim],],
        "sct_on_5" : False
    }
    #
    if   signal_type == "step":
        a_sig = [ [t__end, a_SLF_deg], [t__end, a_tr_deg], ]
        b_sig = [ [t__end, 0.0], [t__end, b_tr_deg], ]
        p_sig = [ [t_start, 0.0], [t_start, p_roll], [t__end, p_roll], [t__end, p_tr_deg], ]
        q_sig = [ [t__end, 0.0], [t__end, q_tr_deg], ]
        r_sig = [ [t_start, 0.0], [t_start, r_roll], [t__end, r_roll], [t__end, r_tr_deg], ]
    elif signal_type == "triangle":
        t__mid = t_start + transition_time/2.0
        a_sig = [ [t_start, a_SLF_deg], [t__end, a_tr_deg], ]
        b_sig = [ [t_start, 0.0], [t__end, b_tr_deg], ]
        p_sig = [ [t_start, 0.0], [t__mid, p_roll*2.0], [t__end, 0.0], [t__end, p_tr_deg], ]
        q_sig = [ [t__mid, 0.0], [t__end, q_tr_deg], ]
        r_sig = [ [t_start, 0.0], [t__mid, r_roll*2.0], [t__end, 0.0], [t__end, r_tr_deg], ]
    elif signal_type == "triangle_cont": # continuous
        t__mid = t_start + transition_time/2.0
        a_sig = [ [t_start, a_SLF_deg], [t__end, a_tr_deg], ]
        b_sig = [ [t_start, 0.0], [t__end, b_tr_deg], ]
        p_sig = [ [t_start, 0.0], [t__mid, p_roll*2.0], [t__end, p_tr_deg], ]
        q_sig = [ [t__mid, 0.0], [t__end, q_tr_deg], ]
        r_sig = [ [t_start, 0.0], [t__mid, r_roll*2.0], [t__end, r_tr_deg], ]
    elif signal_type == "1-cosine":
        n_points = 101
        t_tran = np.linspace(t_start,t__end,n_points)
        onemcos = 1.0 - cos(2.0*pi/transition_time*(t_tran-t_start))
        #
        t__mid = t_start + transition_time/2.0
        n_points = int((n_points-1)/2)
        t_tranq = np.linspace(t__mid,t__end,n_points)
        onemcosq = (1.0 - cos(2.0*pi/transition_time*(t_tranq-t__mid)))/2.0
        #
        a_sig = np.vstack((t_tranq,(a_tr_deg - a_SLF_deg)*onemcosq + a_SLF_deg)).T.tolist()
        b_sig = np.vstack((t_tranq,b_tr_deg*onemcosq)).T.tolist()
        p_sig = np.vstack((t_tran,p_roll*onemcos)).T.tolist() + [[t__end, p_tr_deg], ]
        q_sig = np.vstack((t_tranq,q_tr_deg*onemcosq)).T.tolist()
        r_sig = np.vstack((t_tran,r_roll*onemcos)).T.tolist() + [[t__end, r_tr_deg], ]
    elif signal_type == "1-cosine_cont": # continuous
        n_points = 101
        t_tran = np.linspace(t_start,t__end,n_points)
        onemcos = 1.0 - cos(2.0*pi/transition_time*(t_tran-t_start))
        #
        t__mid = t_start + transition_time/2.0
        n_pointsq = int((n_points-1)/2)
        t_tranq = np.linspace(t__mid,t__end,n_pointsq)
        onemcosq = (1.0 - cos(2.0*pi/transition_time*(t_tranq-t__mid)))/2.0
        #
        p_signal = p_roll*onemcos
        p_signal[n_points-n_pointsq:] += p_tr_deg*onemcosq
        r_signal = r_roll*onemcos
        r_signal[n_points-n_pointsq:] += r_tr_deg*onemcosq
        #
        a_sig = np.vstack((t_tranq,(a_tr_deg - a_SLF_deg)*onemcosq + a_SLF_deg)).T.tolist()
        b_sig = np.vstack((t_tranq,b_tr_deg*onemcosq)).T.tolist()
        p_sig = np.vstack((t_tran,p_signal)).T.tolist()
        q_sig = np.vstack((t_tranq,q_tr_deg*onemcosq)).T.tolist()
        r_sig = np.vstack((t_tran,r_signal)).T.tolist()
    elif signal_type == "quartic_bump":
        phi_0 = 0.0
        w_0   = 0.0
        w_f   = 0.0 # p_wind
        dw_0  = 0.0
        dw_f  = 0.0
        T = transition_time
        # calcs
        M = np.array([
            [T**2.,T**3.,T**4.],
            [2.*T,3.*T**2.,4.*T**3.],
            [T**3./3.,T**4./4.,T**5./5.]
        ])
        E = np.array([w_f-w_0-dw_0*T,dw_f-dw_0,phi__deg-phi_0-w_0*T-dw_0*T**2./2.])
        X = np.matmul(np.linalg.inv(M),E)
        # signal
        n_points = 101
        ts = np.linspace(0.0,transition_time,n_points)
        p_tran = w_0 + dw_0*ts + X[0]*ts**2. + X[1]*ts**3. + X[2]*ts**4.
        # q calcs
        t__mid = t_start + transition_time/2.0
        n_pointsq = int((n_points-1)/2)
        t_tranq = np.linspace(t__mid,t__end,n_pointsq)
        onemcosq = (1.0 - cos(2.0*pi/transition_time*(t_tranq-t__mid)))/2.0
        #
        p_signal = p_tran*np.cos(a_SLF_rad)
        p_signal[n_points-n_pointsq:] += p_tr_deg*onemcosq
        r_signal = p_tran*np.sin(a_SLF_rad)
        r_signal[n_points-n_pointsq:] += r_tr_deg*onemcosq
        #
        ts += t_start
        #
        a_sig = np.vstack((t_tranq,(a_tr_deg - a_SLF_deg)*onemcosq + a_SLF_deg)).T.tolist()
        b_sig = np.vstack((t_tranq,b_tr_deg*onemcosq)).T.tolist()
        p_sig = np.vstack((ts,p_signal)).T.tolist()
        q_sig = np.vstack((t_tranq,q_tr_deg*onemcosq)).T.tolist()
        r_sig = np.vstack((ts,r_signal)).T.tolist()
    # create signal
    bire_fs_dict["reference"]["1"] = a_sig
    bire_fs_dict["reference"]["2"] = b_sig
    bire_fs_dict["reference"]["3"] = p_sig
    bire_fs_dict["reference"]["4"] = q_sig
    bire_fs_dict["reference"]["5"] = r_sig
    #
    tf = 2.0 # 10.0 # 60.0 # 30.0 # 600.0 # 
    #########################################################################
    run_bire_fs["track_check_time"] = run_bire_fs["final_time"] = tf
    # bire_fs_dict["simulation"]["include_stall"] = False
    # bire_fs_dict["simulation"]["include_compressibility"] = False
    bire_fs_dict["simulation"]["integrator"] = "rk4"
    # run_bire_fs["time_step"] = 0.001 # 0.0001 # 
    # #
    # bire_fs_dict["actuators"]["order"] = 0
    # run_bire_fs["state_threshold"] = run_bire_fs["state_threshold"][:-4]
    # run_bire_fs["name_end"] += "_noact"
    if bire_fs_dict["actuators"]["order"] > 1:
        state_threshold += [5., 5., 5., 0.05]
    # #
    # blm = 200.0 # 500.0 # 300.0 # 250.0 # 50.0 # 150.0 # 125.0 # 100.0 # 500.0 # 600.0 # 750.0 # 1000.0 # 1500.0 # 
    # bire_fs_dict["actuators"][    "BIRE"]["rate_limits[deg/s]"] = [-blm,blm]
    # alm = 5000.0 # 1000.0 # 870.0 # 50.0 # 150.0 # 125.0 # 100.0 # 250.0 # 500.0 # 600.0 # 750.0 # 1000.0 # 1500.0 # 
    # bire_fs_dict["actuators"][    "BIRE"]["acceleration_limits[deg/s]"] = [-alm,alm]
    # # # # # # # 
    # elm = 50.0
    # bire_fs_dict["actuators"]["elevator"]["rate_limits[deg/s]"] = [-elm,elm]
    # # # # # # #

    # bire_fs_dict["simulation"][      "limit_input"] = False # True # 
    # bire_fs_dict["simulation"]["limit_input_rates"] = False # True # 
    # run_bire_fs["name_end"] += "_nolim"
    # # # # # #
    # bire_fs_dict["simulation"]["constant_density"] = True # False # 
    # # # # # 
    # run_bire_fs["has_turbulence"] = True # False # 
    # run_bire_fs["has_model_error"] = False # True # 
    # # #######################################################################

    run_bire_fs["modify_base_craft_object_function"] = modify_fun
        
    # # # # # # zeros
    # bire_fs_dict["reference"] = {
    #     "deg2rad_states" : [1,2,3,4,5],
    #     "0" : [[ 0.0,   V_trim],[ 2.0,   V_trim],],
    #     "1" : [[ 0.0, a_SLF_deg],[ 2.0, a_SLF_deg],],
    #     "2" : [[ 0.0, 0.0],[ 2.0, 0.0],],
    #     "3" : [[0.0]*2]*2, "4" : [[0.0]*2]*2, "5" : [[0.0]*2]*2, "sct_on_5" : False
    # }
    # run_bire_fs["track_check_time"] = run_bire_fs["final_time"] = 1.0
    #
    # # # # starting in bank
    # bire_fs_dict["reference"] = {
    #     "deg2rad_states" : [1,2,3,4,5],
    #     "0" : [[ 0.0,   V_trim],[ 2.0,   V_trim],],
    #     "1" : [[ 0.0, a_tr_deg],[ 2.0, a_tr_deg],],
    #     "2" : [[ 0.0, b_tr_deg],[ 2.0, b_tr_deg],],
    #     "3" : [[ 0.0, p_tr_deg],[ 2.0, p_tr_deg]],
    #     "4" : [[ 0.0, q_tr_deg],[ 2.0, q_tr_deg]],
    #     "5" : [[ 0.0, r_tr_deg],[ 2.0, r_tr_deg]],
    #     "sct_on_5" : False
    # }
    # run_bire_fs["trim_bank"] = phi__deg
    # # di = [0.7,0.0,0.0] # [0.7772,0.0,0.0] # [1.0,1.0,1.0] # [0.0,0.0,0.0] # [0.1,0.1,0.1] # [10.0,10.0,10.0] # 
    # # di = [0.0, 35.0994612584134487, 0.0] # [0.0, 28.1437298697430229, 0.0] # 
    # # run_bire_fs[ "has_turbulence"] = True # False # 
    # # # run_bire_fs["has_model_error"] = False # True # 
    # # run_bire_fs["name_end"] += "_rt"
    run_single_simulation(bire_fs_dict,rtdst_1sg=di,**run_bire_fs,**plot_vars)
    # # run_single_simulation(base_fs_dict,rtdst_1sg=di,**run_base_fs,**plot_vars)
    # # run_single_simulation(bire_rc_dict,rtdst_1sg=di,**run_bire_rc,**plot_vars)
    # # run_single_simulation(base_rc_dict,rtdst_1sg=di,**run_base_rc,**plot_vars)
    quit()


