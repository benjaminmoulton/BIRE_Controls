import numpy as np
import control as co
import json
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import mpl_toolkits.mplot3d.axes3d as ax3
from matplotlib.animation import FuncAnimation
from numpy import sign
from scipy.linalg import block_diag
from scipy.integrate import ode, odeint
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.io import savemat, loadmat
from math import pi, sin, cos, tan, exp, acos, asin, atan2, fmod
from std_atm import stdatm_english
from SAL_std_atm import SAL_stdatm_english
from quat import quat_mult, euler_2_quat, quat_2_euler, quat_norm, \
    body_2_fixed, fixed_2_body, eulerdot_2_quatdot, quatdot_2_eulerdot
from linearization import linearization as lin, Anderson_correction_der_coeff,\
    Anderson_correction_der_M

import sys
aero_directory = '../aerodynamics_model/'
# loads_directory = '../trim/'
mass_directory = '../mass_properties/'
turb_directory = '../turbulence_models/'

sys.path.insert(1, aero_directory)
# sys.path.insert(1, loads_directory)
sys.path.insert(1, mass_directory)
sys.path.insert(1, turb_directory)

import os as os
from os import mkdir, rmdir, walk, remove, listdir
from os.path import exists as path_exists
from platform import system as sysplat

from f16_aero import F16Aero
from SAL_f16_aero import SALF16Aero
from bire_aero import BIREAero
from thrust import Propulsion
from SAL_thrust import TGEAR,PDOT
from inertia_model import InertiaModel
from turbulence import ZeroTurbulence, DampedSinusoidGust, VonKarmanTurbulence
from hunsaker_atm import stdatm_english as stdatm_hunsaker, gravity_english

# def rwfn(file_name,sep="/"):
#     if sysplat() == "Windows":
#         file_name = (file_name).replace("/",sep)
#     return file_name

class Aircraft:
    """ A class which simulates the flight of an aircraft.

    Parameters
    ----------
    scene_input, string or dict, optional
        Dictionary or path to the JSON object specifying the simulation 
        parameters. If not specified, all default values are chosen.
    
    Raises
    ------
    IOError
        If input filepath or filename is invalid.
    """

    def __init__(self, input_dictionary={}, folder_prefix = "stblz"):

        # report
        if isinstance(input_dictionary,(str)):
            print("\ninitializing " + input_dictionary + "...")
        else:
            print("\ninitializing aircraft...")
        
        self.fldr_prfx = folder_prefix

        # get input variables
        self._get_input_vars(input_dictionary)

        # initialize state
        self._initialize_state(self.a_guess,self.b_guess,self.phi_guess,
            self.u_guess)

        # initialize
        self.x,self.t = self.initialize_sim(self.x0)
        self.tracking = False
        self.additional_states = 0


    def _reinitialize(self):
        return


    def _stdatm_hunsaker(self,H):
        g = gravity_english(H)
        Z,T,p,rho,a = stdatm_hunsaker(H)
        return Z,g,T,p,rho,a
    

    def _get_input_vars(self,input_vars):
        # get info or raise error
        # determine if the input_vars is a file or a dictionary
        input_vars_type = type(input_vars)

        # dictionary
        if input_vars_type == dict:
            input_dict = input_vars
        
        # json file
        elif input_vars_type == str and input_vars.split(".")[-1] == "json":
            # import json file from file path
            json_string = open(input_vars).read()

            # save to vals dictionary
            input_dict = json.loads(json_string)

        # raise error
        else:
            raise IOError("input_vars must be json file path, or " + \
                "dictionary, not {0}".format(input_vars_type))
        
        # store simulation variables globally
        simulation = input_dict.get("simulation",{})
        self.constant_density = simulation.get("constant_density", False)
        self.dt = simulation.get("time_step[sec]", 0.01)
        self.tf = simulation.get("total_time[sec]", 10.0)
        self.integrator = simulation.get("integrator", "rk4")
        self._set_integration_method(self.integrator)
        self.use_nonlinear = simulation.get("nonlinear_dynamics",True)
        self.quat_linearization_built = False
        self.eulr_linearization_built = False
        self.use_quaternions = simulation.get("use_quaternions",True)
        self.is_stevens_and_lewis = simulation.get("stevens_and_lewis",False)
        self._set_dynamics_function(self.use_nonlinear,self.use_quaternions)
        self.bool_limit_inputs = simulation.get("limit_input",True)
        if self.bool_limit_inputs:
            self._limit_input = self._hit_limit_input
        else:
            self._limit_input = self._skip_limit_input
        if simulation.get("limit_input_rates",True):
            self._limit_input_rates = self._hit_limit_input_rates
        else:
            self._limit_input_rates = self._skip_limit_input_rates
        if simulation.get("limit_input_accelerations",True):
            self._limit_input_accelerations = \
                self._hit_limit_input_accelerations
        else:
            self._limit_input_accelerations = \
                self._skip_limit_input_accelerations
        self.atmosphere = simulation.get("atmosphere", "moulton")
        if self.is_stevens_and_lewis:
            self.atmosphere = "stevens_and_lewis"
        #
        if self.atmosphere == "moulton":
            self.stdatm = stdatm_english
        elif self.atmosphere == "hunsaker":
            self.stdatm = self._stdatm_hunsaker
        elif self.atmosphere == "stevens_and_lewis":
            self.stdatm = SAL_stdatm_english
        else:
            raise TypeError("Incorrect atmosphere type specified.")
        self.states_file = simulation.get("states_output","states_output.txt")
        self.is_compressible = simulation.get("include_compressibility",True)
        self.use_anderson = simulation.get("use_Anderson_corrections",True)
        self.has_stall = simulation.get("include_stall",True)
        self.run_unctrl = simulation.get("simulate_uncontrolled",False)
        self.use_fitted_thrust = simulation.get("use_fitted_thrust_model",True)
        self.is_BIRE = simulation.get("BIRE",True)
        self.is_rc = not(simulation.get("full_scale",True))
        self.random_seed = simulation.get("random_seed",None)
        self.rng = np.random.default_rng(seed=self.random_seed)
        turb_random_seed = simulation.get("turbulence_random_seed",\
            self.random_seed)
        self.error_random_seed = simulation.get("error_random_seed",\
            self.random_seed)
        self.err_rng = np.random.default_rng(seed=self.random_seed)

        # thrust model
        if self.is_stevens_and_lewis:
            self._get_thrust_model = self._get_SAL_thrust_model
        else:
            self._get_thrust_model = self._get_BOL_thrust_model

        # initialize aircraft model
        aero_dict = {
            "inp_dir" : aero_directory,
            "thrust_dir" : aero_directory,
            "use_fitted_thrust_model" : self.use_fitted_thrust,
            "use_rc_thrust_model" : self.is_rc,
            "atmosphere_model" : self.stdatm,
            "rho_index_in_model" : 4
        }
        if self.is_BIRE:
            self.aero_model = BIREAero(**aero_dict)
        else:
            if self.is_stevens_and_lewis:
                self.aero_model = SALF16Aero(**aero_dict)
            else:
                self.aero_model = F16Aero(**aero_dict)
        
        # initialize inertia model
        self.inertia_model = InertiaModel(inp_dir=mass_directory, \
            is_bire=self.is_BIRE,is_rc=self.is_rc,
            is_SAL=self.is_stevens_and_lewis)

        # store constants variables globally
        constants = input_dict.get("constants",{})
        self.dtor = np.pi / 180.0
        self.rtod = 180.0 / np.pi
        self.g = constants.get("g[ft/s^2]", 3.2174032152230971E+01)
        self.RE = 6378.1363 * 1000.0 / .3048 # in ft
        self.e2 = 0.0066943850

        # store aircraft variables globally
        aircraft = input_dict.get("aircraft",{})
        self.Sw  = aircraft.get("wing_area[ft^2]", 300.0)
        self.bw  = aircraft.get("wing_span[ft]", 30.0)
        self.cw  = aircraft.get("wing_aerodynamic_mean_chord[ft]", 11.32)
        self.cgshift  = np.array(aircraft.get("CG_shift[ft]", [0.0, 0.0, 0.0]))
        # store aircraft thrust values
        thrust = aircraft.get("thrust",{})
        self.T_loc  = np.array(thrust.get("location[ft]", [0.0, 0.0, 0.0]))
        self.T_dir  = np.array(thrust.get("direction", [1.0, 0.0, 0.0]))

        # store controller dictionary for Linear Model use
        controller_dict = input_dict.get("controller")
        self.enforce_update_frequency = \
            controller_dict.get("enforce_update_frequency",True)
        update_freq = controller_dict.get("update_frequency[hz]",None)
        if update_freq is None:
            self.dt_u_update = 0.0
            self.enforce_update_frequency = False
        else:
            self.dt_u_update = 1./update_freq
        self.can_update = True
        self.controller_type = controller_dict.get("type")
        control_name = controller_dict.get("name")
        integral_states = controller_dict.get("integral_states",[])
        self.control_dict = controller_dict.get(control_name)

        # store actuation variables globally
        self.actuators_dict = actuators = input_dict.get("actuators")
        self.order = actuators.get("order",2)
        if self.order == 0:
            self._actuation_dynamics = self._0th_order
        if self.order == 1:
            if self.is_stevens_and_lewis:
                self._actuation_dynamics = self._1st_order_SAL
            else:
                self._actuation_dynamics = self._1st_order_MOUL
        elif self.order == 2:
            self._actuation_dynamics = self._2nd_order
        self.is_quantized = actuators.get("quantized_actuators",True)
        if self.integrator in ["ode","odeint"] and \
            not(self.enforce_update_frequency):
            self.is_quantized = False
        #
        if self.is_quantized:
            self._quantize_input = self._hit_quantize_input
        else:
            self._quantize_input = self._skip_quantize_input
        ##
        aileron = actuators.get("aileron")
        self.s_da = 1. / aileron.get("lag[s]")
        self.z_da = aileron.get("damping_ratio")
        self.w_da = aileron.get("bandwidth[rad/s]")
        self.min_da, self.max_da = np.deg2rad(np.array(\
            aileron.get("limits[deg]")))
        self.min_dadot, self.max_dadot = np.deg2rad(np.array(\
            aileron.get("rate_limits[deg/s]")))
        aileron_steps = aileron.get("quantization_steps",1201)
        aileron_steps += aileron_steps % 2 - 1
        self.da_quants = np.linspace(self.min_da,self.max_da,num=aileron_steps)
        self.da_qstep = (self.max_da - self.min_da)/(aileron_steps - 1)
        ##
        elevator = actuators.get("elevator")
        self.s_de = 1. / elevator.get("lag[s]")
        self.z_de = elevator.get("damping_ratio")
        self.w_de = elevator.get("bandwidth[rad/s]")
        self.min_de, self.max_de = np.deg2rad(np.array(\
            elevator.get("limits[deg]")))
        self.min_dedot, self.max_dedot = np.deg2rad(np.array(\
            elevator.get("rate_limits[deg/s]")))
        elevator_steps = elevator.get("quantization_steps",1200)
        elevator_steps += elevator_steps % 2 - 1
        self.de_quants = np.linspace(self.min_de,self.max_de,\
            num=elevator_steps)
        self.de_qstep = (self.max_de - self.min_de)/(elevator_steps - 1)
        ##
        if self.is_BIRE:
            yaw_surface = actuators.get("BIRE")
        else:
            yaw_surface = actuators.get("rudder")
        self.s_dr = 1. / yaw_surface.get("lag[s]")
        self.z_dr = yaw_surface.get("damping_ratio")
        self.w_dr = yaw_surface.get("bandwidth[rad/s]")
        self.min_dr, self.max_dr = np.deg2rad(np.array(\
            yaw_surface.get("limits[deg]")))
        self.min_drdot, self.max_drdot = np.deg2rad(np.array(\
            yaw_surface.get("rate_limits[deg/s]")))
        self.min_drddot, self.max_drddot = np.deg2rad(np.array(\
            yaw_surface.get("acceleration_limits[deg/s^2]")))
        yaw_surface_steps = yaw_surface.get("quantization_steps",1200)
        yaw_surface_steps += yaw_surface_steps % 2 - 1
        self.dr_quants = np.linspace(self.min_da,self.max_dr,\
            num=yaw_surface_steps)
        self.dr_qstep = (self.max_dr - self.min_dr)/(yaw_surface_steps - 1)
        ##
        throttle = actuators.get("throttle")
        self.z_tau = throttle.get("damping_ratio")
        self.w_tau = throttle.get("bandwidth[rad/s]")
        self.min_tau, self.max_tau = throttle.get("limits[perc]")
        self.min_taudot, self.max_taudot = throttle.get("rate_limits[perc/s]")
        throttle_steps = throttle.get("quantization_steps",1200)
        throttle_steps += throttle_steps % 2 - 1
        self.tau_quants = np.linspace(self.min_tau,self.max_tau,\
            num=throttle_steps)
        self.tau_qstep = (self.max_tau - self.min_tau)/(throttle_steps - 1)

        # store initial variables globally
        initial = input_dict.get("initial",{})
        self.H0     = initial.get("altitude[ft]", 0.0)
        if "mach" in initial:
            self.M0 = initial.get("mach", 0.0)
            _,_,_,_,_,sos = self.stdatm(self.H0)
            self.V0 = self.M0*sos
        else:
            self.V0 = initial.get("airspeed[ft/s]", 0.0)
            _,_,_,_,_,sos = self.stdatm(self.H0)
            self.M0 = self.V0/sos
        self.psi0   = initial.get("heading[deg]", 0.0) * self.dtor
        self.state_type = initial.get("type","state")
        if self.state_type == "state":
            state = initial.get("state",{})
            self.theta0 = state.get("elevation_angle[deg]", 0.0) * self.dtor
            self.phi0   = state.get("bank_angle[deg]", 0.0) * self.dtor
            self.alpha0 = state.get("alpha[deg]", 0.0) * self.dtor
            self.beta0  = state.get("beta[deg]", 0.0) * self.dtor
            self.p0     = state.get("p[deg/s]", 0.0) * self.dtor
            self.q0     = state.get("q[deg/s]", 0.0) * self.dtor
            self.r0     = state.get("r[deg/s]", 0.0) * self.dtor
            self.ail0   = state.get("aileron[deg]", 0.0) * self.dtor
            self.ele0   = state.get("elevator[deg]", 0.0) * self.dtor
            if ("BIRE[deg]" in state) and self.is_BIRE:
                u_key = "BIRE[deg]"
            else:
                u_key = "rudder[deg]"
            self.rud0   = state.get(u_key, 0.0) * self.dtor
            self.thr0   = state.get("throttle", 0.0)
        # trim variables
        trim = initial.get("trim",{})
        self.trim_type = trim.get("type","shss")
        self.given_elevation = "elevation_angle[deg]" in trim
        if self.given_elevation:
            self.theta_trim = trim.get("elevation_angle[deg]",0.)*self.dtor
        else:
            self.climb_trim = trim.get("climb_angle[deg]", 0.0) * self.dtor
        if self.trim_type == "sct":
            self.given_bank = "bank_angle[deg]" in trim
            self.phi_trim = trim.get("bank_angle[deg]",0.0) * self.dtor
        elif self.trim_type == "shss":
            self.given_bank = "bank_angle[deg]" in trim
            if self.given_bank:
                self.phi_trim = trim.get("bank_angle[deg]",0.0) * self.dtor
            else:
                self.beta_trim=trim.get("sideslip_angle[deg]",0.)*self.dtor
        elif self.trim_type == "spu":
            self.given_bank = True
            self.phi_trim = 0.0 * self.dtor
            self.q_trim = trim.get("pitch_rate[deg/s]",0.0) * self.dtor
        solver = trim.get("solver",{})
        self.NR_dx  = solver.get("finite_difference_step_size",0.01)
        self.NR_G   = solver.get("relaxation_factor",0.5)
        self.NR_tol = solver.get("tolerance",1.0e-10)
        self.trim_iter_max = solver.get("max_iterations",1000)
        self.verbose_trim = trim.get("verbose_trim",True)
        # trim guess variables
        trim_guess = initial.get("trim_guess",{})
        # state
        self.a_guess   = trim_guess.get(     "alpha[deg]",None)
        if self.a_guess is not None: self.a_guess = self.a_guess*self.dtor
        self.b_guess   = trim_guess.get(      "beta[deg]",None)
        if self.b_guess is not None: self.b_guess = self.b_guess*self.dtor
        self.phi_guess = trim_guess.get("bank_angle[deg]",None)
        if self.phi_guess is not None: self.phi_guess = self.phi_guess*self.dtor
        # control
        da_guess = np.deg2rad(trim_guess.get("aileron[deg]",0.0))
        de_guess = np.deg2rad(trim_guess.get("elevator[deg]",0.0))
        if ("BIRE[deg]" in trim_guess) and self.is_BIRE:
            u_key = "BIRE[deg]"
        else:
            u_key = "rudder[deg]"
        u2_guess = np.deg2rad(trim_guess.get(u_key,0.0))
        tu_guess = np.deg2rad(trim_guess.get("throttle",0.0))
        self.u_guess = np.array([da_guess,de_guess,u2_guess,tu_guess])
        if np.linalg.norm(self.u_guess) == 0.0:
            self.u_guess = None

        # determine state indices
        n_states = 12 + 1*self.use_quaternions + 4*self.order
        self.xIi = []; self.xIi_eul = []
        self.xPi = []; self.xPi_eul = []
        for i in range(len(integral_states)):
            self.xPi.append(integral_states[i])
            self.xIi.append(n_states+i)
            if integral_states[i] >= 8:
                self.xPi_eul.append(integral_states[i]-1)
            else:
                self.xPi_eul.append(integral_states[i])
            if n_states+i >= 8:
                self.xIi_eul.append(n_states+i-1)
            else:
                self.xIi_eul.append(n_states+i)
        
        # save for alternate design
        self.prev_integral = np.zeros((len(self.xIi),))
        self.prev_error    = np.zeros((len(self.xIi),))

        # store reference signals
        reference = input_dict.get("reference",{})
        deg2rad_ref_inds = reference.get("deg2rad_states",[])
        self.sct_on_5 = reference.get("sct_on_5",False)
        self.r_ints = []
        xp = []
        fp = []
        for i in range(n_states - 1*self.use_quaternions + len(self.xIi)):
            data = np.array(reference.get(str(i),[[0.0,0.0],[1.0,0.0]]))
            # define reference interpolation
            xp.append(data[:,0]*1.)
            if i in deg2rad_ref_inds:
                data[:,1] = np.deg2rad(data[:,1])
            fp.append(data[:,1]*1.)
            ref = lambda j,t_i : np.interp(t_i,xp[j],fp[j])
            self.r_ints.append(ref)
        
        self.ref_data_xp = xp
        self.ref_data_fp = fp
        
        # gust parameters
        gust = input_dict.get("gust",{})

        Aw_max = gust.get("amplitude[ft/s]", 80.0)
        self.Aw = Aw_max * np.array(gust.get("directions(body-fixed)",
            [1.0,1.0,0.5]))
        self.zetaw = gust.get("damping_rate[1/s]", 1.0)
        self.ww = gust.get("frequency[rad/s]", 5.0)
        self.t_gust = gust.get("init_time[s]", 1.0)

        # atmospheric disturbance model
        disturbance = input_dict.get("disturbance",{})
        disturbance["random_seed"] = turb_random_seed
        dist_type = disturbance.get("type","von_Karman") # "none") # 
        # disturbance["turbulence_intensity"] = "2"
        poss_models = ["none","damped_sinusoid","von_Karman"]
        if dist_type == "none":
            self.has_turbulence = False
            model = ZeroTurbulence
        elif dist_type == "damped_sinusoid":
            self.has_turbulence = True
            model = DampedSinusoidGust
        elif dist_type == "von_Karman":
            self.has_turbulence = True
            disturbance["intensity_folder"] = turb_directory
            if self.is_rc:
                disturbance["initial_altitude[ft]"] = 200.0 
                # !!! flying altitude different than turbulence altitude
            else:
                disturbance["initial_altitude[ft]"] = self.H0
            model = VonKarmanTurbulence
        else:
            raise TypeError("Disturbance model " + \
                "{} invalid, must be of {}".format(dist_type,poss_models))
        #
        # tf = max(self.tf,200.)
        tf = self.tf
        self.disturbance_model = model(disturbance,self.bw,self.V0,self.dt,tf)
        #
        if False:
            self.get_disturbance = self.disturbance_model.get_disturbance
        else:
            self.get_disturbance = \
                self.disturbance_model.get_precomputed_disturbance


        # # Inertia parameter
        # Tx_max = 45800.0
        # Ixx = 15682.0
        # self.dBmax = Tx_max / Ixx

        # FM model error
        self.FM_errors = np.zeros((6,))

        return


    def _set_integration_method(self,integrator):

        if integrator == "rk4":
            self.int_method = self._rk4
        elif integrator == "odeint":
            self.int_method = self._odeint
        elif integrator == "ode":
            self.ode_integrator = ode(self._dynamics)
            self.int_method = self._ode
        else:
            raise TypeError("Incorrect integration method specified.")


    def _set_dynamics_function(self,use_nonlinear=True,use_quaternions=True):

        if use_nonlinear:
            if use_quaternions:
                self._get_dynamics = self._nonlinear_quaternion_dynamics
            else:
                self._get_dynamics = self._nonlinear_euler_dynamics
        else:
            if use_quaternions:
                self._get_dynamics = self._linear_quaternion_dynamics
            else:
                self._get_dynamics = self._linear_euler_dynamics

  
    def _given_state(self):
        x = np.zeros((12 + 1*self.use_quaternions + 4*self.order,))
        ## INTSTATE
        x[0] = self.V0 * cos(self.alpha0) * cos(self.beta0)
        x[1] = self.V0 * sin(self.beta0)
        x[2] = self.V0 * sin(self.alpha0) * cos(self.beta0)
        x[3] = self.p0
        x[4] = self.q0
        x[5] = self.r0
        x[8] = -self.H0
        x[9:13] = quat_norm(euler_2_quat([self.phi0,self.theta0,self.psi0]))
        if self.order >= 1:
            x[12 + 1*self.use_quaternions] = self.ail0
            x[13 + 1*self.use_quaternions] = self.ele0
            x[14 + 1*self.use_quaternions] = self.rud0
            x[15 + 1*self.use_quaternions] = self.thr0
            if self.is_stevens_and_lewis:
                x[15 + 1*self.use_quaternions] = \
                    TGEAR(x[15 + 1*self.use_quaternions])
        
        u = np.zeros((6,))
        u[0] = self.ail0
        u[1] = self.ele0
        u[2] = self.rud0
        u[3] = self.thr0

        self.trim_iter = 0
        
        return u,x


    def _get_elevation_angle(self,gamma,u,v,w,V,bank_angle):
        # create components of quadratic equation
        # angles
        sg = sin(gamma)
        sp = sin(bank_angle); cp = cos(bank_angle)
        # larger pieces
        bank_sum = v * sp + w * cp
        usq_bank_sumsq = u*u + bank_sum*bank_sum
        Vsg = V * sg
        # left and right numerators for quadratic equation
        lnum = u*Vsg
        rnum = bank_sum * ( usq_bank_sumsq - Vsg*Vsg ) ** 0.5
        
        # calculate quadratic equation
        plus = asin((lnum + rnum) / usq_bank_sumsq)
        minu = asin((lnum - rnum) / usq_bank_sumsq)
        
        # calculate what should be zero
        pzero = abs(Vsg - u*sin(plus) + bank_sum*cos(plus))
        mzero = abs(Vsg - u*sin(minu) + bank_sum*cos(minu))

        # return based on condition
        if pzero < mzero:
            return plus
        else:
            return minu


    def _update_trim_state(self,a,b,phi,g,x):
        # determine bank angle
        if self.trim_type == "sct" or \
            (self.trim_type == "shss" and self.given_bank):
            phi = self.phi_trim
        elif self.trim_type == "vc":
            phi = 0.0; b = 0.0
        elif self.trim_type == "spu":
            phi = 0.0; b = 0.0
        else:
            b = self.beta_trim
        
        # determine velocities
        ## INTSTATE
        x[0] = self.V0 * cos(a) * cos(b)
        x[1] = self.V0          * sin(b)
        x[2] = self.V0 * sin(a) * cos(b)

        # calculate elevation angle
        if self.trim_type in ["vc","spu"]:
            theta = self.climb_trim + a
        elif not self.given_elevation:
            ## INTSTATE
            theta = self._get_elevation_angle(self.climb_trim,x[0],x[1],x[2],\
                self.V0,phi)
        else:
            theta = self.theta_trim
        
        # calculate rates -- first need components
        if self.trim_type == "sct":
            st = sin(theta); ct = cos(theta)
            sp = sin(phi);   cp = cos(phi)
            numerator = g * sp * ct
            p_num = numerator * -st
            q_num = numerator * sp * ct
            r_num = numerator * cp * ct
            u_den = cp * ct
            w_den = st
            # determine p,q,r
            ## INTSTATE
            denominator = x[0] * u_den + x[2] * w_den
            x[3] = p_num / denominator
            x[4] = q_num / denominator
            x[5] = r_num / denominator
        elif self.trim_type == "spu":
            x[3] = 0.0
            x[4] = self.q_trim
            x[5] = 0.0

        return b,phi,theta,x


    def _report_trim_solution(self,x="o",u="o",iter="o",
        load_factors_axis="stab",report_coord_frame_rates=False):

        # if nothing given, use save trim state
        if isinstance(x,str):
            x = self.x_trim
        if isinstance(u,str):
            u = self.u_trim
        if isinstance(iter,str):
            iter = self.trim_iter

        # get values
        ## INTSTATE
        a = atan2(x[2],x[0])
        V = (x[0] * x[0] + x[1] * x[1] + x[2] * x[2])**0.5
        b = asin(x[1]/V)
        sos = self.stdatm(-x[8])[5]
        M = V/sos
        # T = self.aero_model.get_thrust(u[3],-x[8],V)
        T = self._get_thrust_model(u[3],u[3],-x[8],V,M,is_trim=True)
        if self.use_quaternions:
            ph,th,ps = self._euler_angles(x)
        else:
            ph, th, ps = x[ 9], x[10], x[11]
        
        # climb angle
        clm = asin((x[0]*sin(th) - (x[1]*sin(ph) + x[2]*cos(ph))*cos(th))/V)

        # load factors
        if load_factors_axis != "none":
            nxyz = self._load_factors(x,u,axis=load_factors_axis)

        # stability rates
        w = x[3:6]
        ca = cos(a); sa = sin(a); cb = cos(b); sb = sin(b)
        ps,qs,rs = np.matmul([[ca,0.,sa],[ 0.,1.,0.],[-sa,0.,ca]],w)
        pw,qw,rw = np.matmul([[cb,sb,0.],[-sb,cb,0.],[ 0.,0.,1.]],[ps,qs,rs])

        thrust_string = "    {:<23s} : {:>23.16f}   {:>23.16}".format(\
            "thrust[lbf]",T,"")
        title = " Trim Settings "
        num_eq_sn = int((len(thrust_string) - len(title))/2)
        print("=" * num_eq_sn + title + "=" * num_eq_sn)
        print("    {:<23s} : {:> 23.16f} : {:> 23.16f}".format(\
            "\"elevation[deg,rad]\"",th*self.rtod,th))
        print("    {:<23s} : {:> 23.16f} : {:> 23.16f}".format(\
            "\"bank_angle[deg,rad]\"",ph*self.rtod,ph))
        print("    {:<23s} : {:> 23.16f} : {:> 23.16f}".format(\
            "\"climb_angle[deg,rad]\"",clm*self.rtod,clm))
        print("    {:<23s} : {:> 23.16f} : {:> 23.16f}".format(\
            "\"alpha[deg,rad]\"",a*self.rtod,a))
        print("    {:<23s} : {:> 23.16f} : {:> 23.16f}".format(\
            "\"beta[deg,rad]\"",b*self.rtod,b))
        print("    {:<23s} : {:> 23.16f}".format("\"M\"",M))
        print("    {:<23s} : {:> 23.16f}".format("\"V[ft/s]\"",V))
        ## INTSTATE
        print("    {:<23s} : {:> 23.16f}".format("\"u[ft/s]\"",x[0]))
        print("    {:<23s} : {:> 23.16f}".format("\"v[ft/s]\"",x[1]))
        print("    {:<23s} : {:> 23.16f}".format("\"w[ft/s]\"",x[2]))
        print("    {:<23s} : {:> 23.16f} : {:> 23.16f}".format(\
            "\"p[deg/s,rad/s]\"",x[3]*self.rtod,x[3]))
        print("    {:<23s} : {:> 23.16f} : {:> 23.16f}".format(\
            "\"q[deg/s,rad/s]\"",x[4]*self.rtod,x[4]))
        print("    {:<23s} : {:> 23.16f} : {:> 23.16f}".format(\
            "\"r[deg/s,rad/s]\"",x[5]*self.rtod,x[5]))
        print("    {:<23s} : {:> 23.16f}".format("\"H[ft]\"",-x[8]))
        print("    {:<23s} : {:> 23.16f} : {:> 23.16f}".format(\
            "\"aileron[deg,rad]\"",u[0]*self.rtod,u[0]))
        print("    {:<23s} : {:> 23.16f} : {:> 23.16f}".format(\
            "\"elevator[deg,rad]\"",u[1]*self.rtod,u[1]))
        if self.is_BIRE:
            print("    {:<23s} : {:> 23.16f} : {:> 23.16f}".format(\
                "\"BIRE[deg,rad]\"",u[2]*self.rtod,u[2]))
        else:
            print("    {:<23s} : {:> 23.16f} : {:> 23.16f}".format(\
                "\"rudder[deg,rad]\"",u[2]*self.rtod,u[2]))
        print("    {:<23s} : {:> 23.16f}".format("\"throttle\"",u[3]))
        print(thrust_string)
        if load_factors_axis != "none":
            print("    {:<23s} : {:> 23.16f}".format(\
                "\""+load_factors_axis+" fwrd load factor\"",nxyz[0]))
            print("    {:<23s} : {:> 23.16f}".format(\
                "\""+load_factors_axis+" side load factor\"",nxyz[1]))
            print("    {:<23s} : {:> 23.16f}".format(\
                "\""+load_factors_axis+" norm load factor\"",nxyz[2]))
        if report_coord_frame_rates:
            print("    {:<23s} : {:> 23.16f} : {:> 23.16f}".format(\
                "\"ps[deg/s,rad/s]\"",ps*self.rtod,ps))
            print("    {:<23s} : {:> 23.16f} : {:> 23.16f}".format(\
                "\"qs[deg/s,rad/s]\"",qs*self.rtod,qs))
            print("    {:<23s} : {:> 23.16f} : {:> 23.16f}".format(\
                "\"rs[deg/s,rad/s]\"",rs*self.rtod,rs))
            print("    {:<23s} : {:> 23.16f} : {:> 23.16f}".format(\
                "\"pw[deg/s,rad/s]\"",pw*self.rtod,pw))
            print("    {:<23s} : {:> 23.16f} : {:> 23.16f}".format(\
                "\"qw[deg/s,rad/s]\"",qw*self.rtod,qw))
            print("    {:<23s} : {:> 23.16f} : {:> 23.16f}".format(\
                "\"rw[deg/s,rad/s]\"",rw*self.rtod,rw))
        print("    {:<23s} : {:> 8}{}".format("\"iterations\"",iter," "*13))
        print("=" * len(thrust_string))

        # trim aero coeffs
        # nondimensionalize rates
        pbar = (x[3])*self.bw/2./V
        qbar = (x[4])*self.cw/2./V
        rbar = (x[5])*self.bw/2./V
        # pass in controls state
        ail = u[0]
        ele = u[1]
        rud = u[2]
        thr = u[3]
        # use aircraft model
        [CL, CS, CD, Cl, Cm, Cn] = self.aero_model.aero_results(*[
            a,b,pbar,qbar,rbar,ail,ele,rud,
            self.is_compressible,M,self.use_anderson,self.has_stall
        ])
        # report
        title = " Trim Aerodynamic Coefficients "
        num_eq_sn = int((len(thrust_string) - len(title))/2)
        print("|" * num_eq_sn + title + "|" * num_eq_sn)
        print("    {:<23s} : {:> 23.16f}".format("\"CL\"",CL))
        print("    {:<23s} : {:> 23.16f}".format("\"CS\"",CS))
        print("    {:<23s} : {:> 23.16f}".format("\"CD\"",CD))
        print("    {:<23s} : {:> 23.16f}".format("\"Cl\"",Cl))
        print("    {:<23s} : {:> 23.16f}".format("\"Cm\"",Cm))
        print("    {:<23s} : {:> 23.16f}".format("\"Cn\"",Cn))
        print("|" * len(thrust_string))


    def run_trim(self,a_guess=None,b_guess=None,phi_guess=None,u_guess=None,
        verbose=True,no_report=False,imax="o"):
        # report
        if not no_report:
            print("running trim algorithm...")
        
        if imax == "o":
            imax = self.trim_iter_max

        # initialize state and controls
        self.trim_failed = False
        x = np.zeros((13 + 4*self.order + len(self.xIi),))
        if a_guess is None:
            a_guess = 0.
        if b_guess is None:
            b_guess = 0.
        if phi_guess is None:
            phi_guess = 0.
        if u_guess is None:
            da_guess = 0.
            de_guess = 0.
            dr_guess = 0.
            tu_guess = 0.
        else:
            da_guess = u_guess[0]
            de_guess = u_guess[1]
            dr_guess = u_guess[2]
            tu_guess = u_guess[3]
        if self.given_bank:
            u = np.array([da_guess,de_guess,dr_guess,tu_guess,a_guess,b_guess])
            G = u*1.
        else:
            u = np.array(
                [da_guess,de_guess,dr_guess,tu_guess,a_guess,phi_guess])
            G = u*1.

        # initialize arrays and matrices for use
        if self.trim_type == "vc":
            n = 5
        else:
            n = 6
        DG = np.zeros((6,))
        J  = np.zeros((6,6))
        forces_im1 = np.zeros((6,))
        forces_ip1 = np.zeros((6,))

        # determine gravitational acceleration at trim altitude
        atmosphere = self.stdatm(self.H0)
        g,rho = atmosphere[1], atmosphere[4]
        self.rho0 = rho*1.

        # determine state
        if self.trim_type == "vc":
            b_,ph,th,x = self._update_trim_state(G[4],0.0,0.0,g,x)
        elif self.given_bank:
            b_,ph,th,x = self._update_trim_state(G[4],G[5],0.0,g,x)
        else:
            b_,ph,th,x = self._update_trim_state(G[4],0.0,G[5],g,x)

        # determine altitude
        ## INTSTATE
        x[8] = -self.H0
        
        # initialize residual
        if self.trim_type == "vc" or self.given_bank:
            R = self._trim_forces(u[4],u[5],ph,th,g,x,u)
        else:
            R = self._trim_forces(u[4],b_,u[5],th,g,x,u)
        Rmag = np.max(np.abs(R))

        if verbose:
            # report names
            names = ["Iter","Throttle","Alpha[deg]","Beta[deg]",
                "Aileron[deg]","Elevator[deg]","Rudder[deg]","p[deg/s]",
                "q[deg/s]","r[deg/s]"]
            
            headings = "{:<10s}".format(names[0])
            for i in range(1,len(names)):
                headings += "{:<19s}".format(names[i])
            print(headings)

        # initialize iteration counter
        iter = 0
        if verbose:
            output = "{:>5d}{:>19.12f}".format(iter,G[3])
            output += "{:>19.12f}{:>19.12f}".format(G[4]*self.rtod,b_*self.rtod)
            output += "{:>19.12f}{:>19.12f}".format(G[0]*self.rtod,G[1]*self.rtod)
            ## INTSTATE
            output += "{:>19.12f}{:>19.12f}".format(G[2]*self.rtod,x[3]*self.rtod)
            output += "{:>19.12f}{:>19.12f}".format(x[4]*self.rtod,x[5]*self.rtod)
            print(output)

        var_names = ["da","de","dr","tr","a ","b "]
        
        # run trim solver
        while Rmag > self.NR_tol and iter < imax:
            iter += 1
            # update state
            if self.trim_type == "vc":
                b_,ph,th,x = self._update_trim_state(G[4],0.0,0.0,g,x)
            elif self.given_bank:
                b_,ph,th,x = self._update_trim_state(G[4],G[5],ph,g,x)
            else:
                b_,ph,th,x = self._update_trim_state(G[4],b_,G[5],g,x)

            # develop Jacobian
            for i in range(6):
                # determine forces with each step change
                u = G * 1.0
                u[i] = G[i] * 1.0 + self.NR_dx
                if self.trim_type == "vc" or self.given_bank:
                    forces_ip1 = self._trim_forces(u[4],u[5],ph,th,\
                        g,x,u)
                else:
                    forces_ip1 = self._trim_forces(u[4],b_,u[5],th,\
                        g,x,u)
                u[i] = G[i] * 1.0 - self.NR_dx
                if self.trim_type == "vc" or self.given_bank:
                    forces_im1 = self._trim_forces(u[4],u[5],ph,th,\
                        g,x,u)
                else:
                    forces_im1 = self._trim_forces(u[4],b_,u[5],th,\
                        g,x,u)

                # assign to jacobian
                J[:,i] = (forces_ip1 - forces_im1) / 2. / self.NR_dx
            
            # calculate change in control state
            try:
                DG[:n] = np.matmul(- np.linalg.solve(J[:n,:n],np.eye(n)),R[:n])
            except:
                print("Trim Failed!!! phi = ",self.phi_trim*self.rtod, "deg",\
                    "theta =",th*self.rtod,"deg")
                u = G*0.0
                self.trim_failed = True
                return G,x
            G[:n] = G[:n] + self.NR_G * DG[:n]
            # limit inputs
            G[0:4] = self._hit_limit_input(G[0:4])
            u = G * 1.0


            # calculate residual
            if self.trim_type == "vc" or self.given_bank:
                R = self._trim_forces(u[4],u[5],ph,th,g,x,u)
            else:
                R = self._trim_forces(u[4],b_,u[5],th,g,x,u)
            Rmag = np.max(np.abs(R))

            # print headings periodically to ease table reading
            if verbose and iter % 20 == 1 and iter != 1:
                print(headings)

            # report iteration
            if verbose:
                ## INTSTATE
                output = "{:>5d}{:>19.12f}".format(iter,G[3])
                output += "{:>19.12f}{:>19.12f}".format(G[4]*self.rtod,\
                    b_*self.rtod)
                output += "{:>19.12f}{:>19.12f}".format(G[0]*self.rtod,\
                    G[1]*self.rtod)
                output += "{:>19.12f}{:>19.12f}".format(G[2]*self.rtod,\
                    x[3]*self.rtod)
                output += "{:>19.12f}{:>19.12f}".format(x[4]*self.rtod,\
                    x[5]*self.rtod)
                print(output)
        
        # fail if the trim solution is not converged
        self.trim_iter = iter
        if Rmag > self.NR_tol and iter >= imax:
            print("Trim Failed, hit iteration limit")
            u = G*0.0
            self.trim_failed = True
            return G,x


        # finish off table
        if verbose:
            print(headings)

        # update state
        if self.trim_type == "vc":
            b_,ph,th,x = self._update_trim_state(G[4],0.0,0.0,g,x)
        elif self.given_bank:
            b_,ph,th,x = self._update_trim_state(G[4],G[5],ph,g,x)
        else:
            b_,ph,th,x = self._update_trim_state(G[4],b_,G[5],g,x)

        # determine orientation
        ## INTSTATE
        x[9:13] = quat_norm(euler_2_quat([ph,th,self.psi0]))

        # print out trim settings
        if verbose:
            self._report_trim_solution(x,G,iter)

        # initialize state with input
        if self.order >= 1:
            ## INTSTATE
            x[13:17] = G[0:4]
            if self.is_stevens_and_lewis:
                x[16] = TGEAR(x[16])

        return G[0:4],x


    def _initialize_state(self,a_guess=None,b_guess=None,phi_guess=None,
        u_guess=None,run2=False,no_report=False):
        # run trim at condition
        u_trim,x_trim = self.run_trim(a_guess,b_guess,phi_guess,u_guess,
            verbose=self.verbose_trim,no_report=no_report)
        ## INTSTATE
        x_trim_euler = np.delete(x_trim,9)
        x_trim_euler[9:12] = self._euler_angles(x_trim)
        x_trim_euler[12:] = x_trim[13:]*1.
        deg_ind = [3,4,5,9,10,11] + (self.order>=1)*[12,13,14] \
            + (self.order>1)*[15,16,17]
        x_trim_euler_deg = x_trim_euler*1.
        x_trim_euler_deg[deg_ind] = np.rad2deg(x_trim_euler[deg_ind])
        u_trim_deg = u_trim*1.
        u_trim_deg[0:3] = np.rad2deg(u_trim_deg[0:3])
        if not self.use_quaternions:
            x_trim = x_trim_euler*1.
        if not(run2):
            self.u_trim = u_trim
            self.x_trim = x_trim
            self.x_trim_euler = x_trim_euler
            self.x_trim_euler_deg = x_trim_euler_deg
            self.u_trim_deg = u_trim_deg
        else:
            self.u_trim2 = u_trim
            self.x_trim2 = x_trim
            self.x_trim2_euler = x_trim_euler
            self.x_trim2_euler_deg = x_trim_euler_deg
            self.u_trim2_deg = u_trim_deg

        # if state not given, determine
        if self.state_type == "state":
            u0,x0 = self._given_state()
        elif self.state_type == "trim":
            u0,x0 = u_trim*1.,x_trim*1.
        
        # save initial state and controls globally
        self.x0 = x0
        self.u = u0
        self.t_u_next_update = 0.0
        self.can_update = True


    def _throttle_gain(self,tau):
        if tau <= 0.3:
            return 1.0
        elif 0.3 <= tau <= 0.5:
            return (2.35 - 4.5 * tau)
        else: # tau >= 0.5
            return 0.1


    def _get_BOL_thrust_model(self,thr,POW,H,V,M,is_trim=False):
        return self.aero_model.get_thrust(thr,H,V)
    

    def _get_SAL_thrust_model(self,thr,POW,H,V,M,is_trim=False):
        if is_trim:
            return self.aero_model.get_thrust(TGEAR(thr),H,M)
        else:
            return self.aero_model.get_thrust(POW,H,M)


    def _aerodynamics(self,x,u,Vg=[0.0,0.0,0.0],Wg=[0.0,0.0,0.0],
        is_trim=False,is_VAB_format=False):
        # aero conditions
        ## INTSTATE
        if is_VAB_format:
            Vu = x[0]*cos(x[1])*cos(x[2]) + Vg[0]
            Vv = x[0]*sin(x[2])           + Vg[1]
            Vw = x[0]*sin(x[1])*cos(x[2]) + Vg[2]
        else:
            Vu,Vv,Vw = x[0]+Vg[0], x[1]+Vg[1], x[2]+Vg[2]
        a = atan2(Vw,Vu)
        V = (Vu * Vu + Vv * Vv + Vw * Vw)**0.5
        b = asin(Vv/V)
        _,g,_,_,rho,sos = self.stdatm(-x[8])
        # ##############################
        # g = 32.12780074195162
        # ##############################
        M = V / sos

        # nondimensionalize rates
        ## INTSTATE
        pbar = (x[3]+Wg[0])*self.bw/2./V
        qbar = (x[4]+Wg[1])*self.cw/2./V
        rbar = (x[5]+Wg[2])*self.bw/2./V

        # pass in controls state
        ail = u[0]
        ele = u[1]
        rud = u[2]
        thr = u[3]

        # use aircraft model
        aero_results = self.aero_model.aero_results(*[
            a,b,pbar,qbar,rbar,ail,ele,rud,
            self.is_compressible,M,self.use_anderson,self.has_stall
        ])
        # add in errors
        [CL, CS, CD, Cl, Cm, Cn] = [aero_results[i]*(1. + self.FM_errors[i]) \
            for i in range(len(aero_results))]

        # thrust forces
        ## INTSTATE
        T = self._get_thrust_model(thr,thr,-x[8],V,M,is_trim)
        FP = T  * self.T_dir
        MP = [
            FP[2] * self.T_loc[1] - FP[1] * self.T_loc[2],
            FP[0] * self.T_loc[2] - FP[2] * self.T_loc[0],
            FP[1] * self.T_loc[0] - FP[0] * self.T_loc[1]
        ]

        # aero forces
        ca = cos(a); sa = sin(a)
        cb = cos(b); sb = sin(b)
        dynF = 0.5 * rho * V*V * self.Sw
        Fx = FP[0] + dynF * (  CL*sa - CS*ca*sb - CD*ca*cb)
        Fy = FP[1] + dynF * (  CS*cb - CD*sb)
        Fz = FP[2] + dynF * (- CL*ca - CS*sa*sb - CD*sa*cb)
        Mx = MP[0] + Cl * dynF * self.bw
        My = MP[1] + Cm * dynF * self.cw
        Mz = MP[2] + Cn * dynF * self.bw
        # SAL ay
        self._SAL_ay = Fy/self.inertia_model.W

        # add in CG effects
        cg = self.cgshift
        Mx += Fy * cg[2] - Fz * cg[1]
        My += Fz * cg[0] - Fx * cg[2]
        Mz += Fx * cg[1] - Fy * cg[0]

        return Fx,Fy,Fz,Mx,My,Mz,g


    def _trim_forces(self,a,b,ph,th,g,x,u,var="",dir=""):
        # update state
        b,ph,th,x = self._update_trim_state(a,b,ph,g,x)
        
        # calculate aerodynamic forces
        Fx,Fy,Fz,Mx,My,Mz,g = self._aerodynamics(x,u,is_trim=True)

        # initialize residual
        R = np.zeros((6,))

        # read in mass properties
        W = self.inertia_model.W
        # W = W*self.stdatm(0.0)[1]/self.stdatm(-x[8])[1] # Hunsaker
        Ixx,Iyy,Izz,Ixy,Ixz,Iyz = self.inertia_model.inertia_results(u[3])
        hx,hy,hz = self.inertia_model.angular_momentum_results()

        # calculate residual
        ## INTSTATE
        pq = x[3] * x[4]; pr = x[3] * x[5]; qr = x[4] * x[5]
        R[0] = Fx - W * sin(th) + (x[5] * x[1] - x[4] * x[2]) * W / g
        R[1] = Fy + W * sin(ph) * cos(th) + (x[3] * x[2] - x[5] * x[0]) * W / g
        R[2] = Fz + W * cos(ph) * cos(th) + (x[4] * x[0] - x[3] * x[1]) * W / g
        R[3] = Mx + x[5]*hy - x[4]*hz + (Iyy - Izz)*qr + \
            Iyz * (x[4]**2. - x[5]**2.) + Ixz * pq - Ixy * pr
        R[4] = My + x[3]*hz - x[5]*hx + (Izz - Ixx)*pr + \
            Ixz * (x[5]**2. - x[3]**2.) + Ixy * qr - Iyz * pq
        R[5] = Mz + x[4]*hx - x[3]*hy + (Ixx - Iyy)*pq + \
            Ixy * (x[3]**2. - x[4]**2.) + Iyz * pr - Ixz * qr
        
        return R


    def _0th_order(self,x,u):
        # 0th order
        dda = 0.0
        dde = 0.0
        ddr = 0.0
        ddt = 0.0

        ddda = 0.0
        ddde = 0.0
        dddr = 0.0
        dddt = 0.0

        return dda,dde,ddr,ddt,ddda,ddde,dddr,dddt


    def _1st_order_MOUL(self,x,u):
        q = 1*self.use_quaternions
        # 1st order
        ## INTSTATE
        dda = self.s_da * (u[0] - x[12+q])
        dde = self.s_de * (u[1] - x[13+q])
        ddr = self.s_dr * (u[2] - x[14+q])
        s_tau = self._throttle_gain(x[15+q])
        ddt =     s_tau * (u[3] - x[15+q])

        # limit rates
        dda,dde,ddr,ddt = self._limit_input_rates([dda,dde,ddr,ddt])

        return dda,dde,ddr,ddt


    def _1st_order_SAL(self,x,u):
        q = 1*self.use_quaternions
        # 1st order
        ## INTSTATE
        dda = self.s_da * (u[0] - x[12+q])
        dde = self.s_de * (u[1] - x[13+q])
        ddr = self.s_dr * (u[2] - x[14+q])
        ddt =     PDOT(x[15+q], TGEAR(u[3])) # state is already POW

        # limit rates
        dda,dde,ddr,ddt = self._limit_input_rates([dda,dde,ddr,ddt])

        return dda,dde,ddr,ddt


    def _2nd_order(self,x,u):
        q = 1*self.use_quaternions
        # set 1st order
        ## INTSTATE
        dda = x[16+q]
        dde = x[17+q]
        ddr = x[18+q]
        ddt = x[19+q]

        # limit rates
        dda,dde,ddr,ddt = self._limit_input_rates([dda,dde,ddr,ddt])

        # 2nd order
        ## INTSTATE
        ddda = -2.*self.z_da *self.w_da *x[16+q] +self.w_da**2.*(u[0]-x[12+q])
        ddde = -2.*self.z_de *self.w_de *x[17+q] +self.w_de**2.*(u[1]-x[13+q])
        dddr = -2.*self.z_dr *self.w_dr *x[18+q] +self.w_dr**2.*(u[2]-x[14+q])
        dddt = -2.*self.z_tau*self.w_tau*x[19+q]+self.w_tau**2.*(u[3]-x[15+q])

        # limit accelerations
        ddda,ddde,dddr,dddt = self._limit_input_accelerations(
            [ddda,ddde,dddr,dddt])

        return dda,dde,ddr,ddt,ddda,ddde,dddr,dddt


    def _OLD__hit_quantize_input(self,u):
        da  = self. da_quants[ np.abs(self. da_quants - u[0]).argmin() ]
        de  = self. de_quants[ np.abs(self. de_quants - u[1]).argmin() ]
        dr  = self. dr_quants[ np.abs(self. dr_quants - u[2]).argmin() ]
        tau = self.tau_quants[ np.abs(self.tau_quants - u[3]).argmin() ]
        return da,de,dr,tau


    def _hit_quantize_input(self,u):
        da_b = (fmod(u[0], self.da_qstep) // (self.da_qstep/2.))
        da = ( u[0] // self.da_qstep + da_b*fmod(da_b,2) )*self.da_qstep
        #
        de_b = (fmod(u[1], self.de_qstep) // (self.de_qstep/2.))
        de = ( u[1] // self.de_qstep + de_b*fmod(de_b,2) )*self.de_qstep
        #
        dr_b = (fmod(u[2], self.dr_qstep) // (self.dr_qstep/2.))
        dr = ( u[2] // self.dr_qstep + dr_b*fmod(dr_b,2) )*self.dr_qstep
        #
        tau_b = (fmod(u[3], self.tau_qstep) // (self.tau_qstep/2.))
        tau = ( u[3] // self.tau_qstep + tau_b*fmod(tau_b,2) )*self.tau_qstep
        return da,de,dr,tau

    
    def _skip_quantize_input(self,u):
        return u


    def _hit_limit_input(self,u):
        da  = max(min(u[0],self.max_da ),self.min_da )
        # # limit elevator due to aero model being built da/4 passed to de 
        # # though referenced as da command
        # da_to_de = 0.25*abs(da)
        # de  = max(min(u[1],self.max_de - da_to_de ),self.min_de + da_to_de )
        de  = max(min(u[1],self.max_de ),self.min_de )
        dr  = max(min(u[2],self.max_dr ),self.min_dr )
        tau = max(min(u[3],self.max_tau),self.min_tau)
        return da,de,dr,tau


    def _skip_limit_input(self,u):
        print("these things ought not so to be!!")
        quit()
        return u


    def _hit_limit_input_rates(self,du):
        dda  = max(min(du[0],self.max_dadot ),self.min_dadot )
        dde  = max(min(du[1],self.max_dedot ),self.min_dedot )
        ddr  = max(min(du[2],self.max_drdot ),self.min_drdot )
        dtau = max(min(du[3],self.max_taudot),self.min_taudot)
        return dda,dde,ddr,dtau


    def _skip_limit_input_rates(self,du):
        return du


    def _hit_limit_input_accelerations(self,ddu):
        ddda  = ddda
        ddde  = ddde
        dddr  = max(min(ddu[2],self.max_drddot ),self.min_drddot )
        ddtau = ddtau
        return ddda,ddde,dddr,ddtau


    def _skip_limit_input_accelerations(self,ddu):
        return ddu


    def _get_reference(self,t):
        r = np.array([self.r_ints[i](i,t) for i in range(len(self.r_ints))])
        return r


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
                    ## INTSTATE
                    x_euler[9:12] = quat_2_euler(euler_2_quat(x_euler[9:12]))
                ##################################
                x_tr = self.Lin_Model.xhat_eq*1.
                u_tr = self.Lin_Model.uhat_eq*1.
                K_tr = self.Lin_Model.K
                #
                u = self.u_trim*1.
                ###################################
                Dx = x_euler[self.Lin_Model.Cslice] - x_tr
                u[self.Lin_Model.Cuslice] = u_tr - np.matmul(K_tr,Dx)#*0.
                if self.order > 0:
                    q = 1*self.use_quaternions
                    ## INTSTATE
                    inputs = x[12+q:16+q]*1.
                else:
                    inputs = u*1.
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
                    ## INTSTATE
                    inputs = x[12+q:16+q]*1.
                else:
                    inputs = u*1.
        
        # limit actuators
        # #vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        # This ensures that the actuator state is limited in odeint runs
        # While this is not totally accurate (the controller commands should 
        # not be limited), it is conservative. This is because when the 
        # actuator rate limit is not constraining in that it will command the 
        # actuators to move slower than otherwise near the saturation limit.
        if self.integrator == "odeint":
            u = self._limit_input(u)
        # #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # quantize actuators
        inputs = self._quantize_input(self._limit_input(inputs))
        if inputs[0] > self.max_da:
            print(t,np.rad2deg(inputs[0]))
        if inputs[0] < self.min_da:
            print(t,np.rad2deg(inputs[0]))

        return u,inputs


    def _linear_quaternion_dynamics(self,t,x,
        is_controlled=True,given_control=False,u="o",
        force_control_to_inputs=False):

        # get control
        u,inputs = self._get_control(t,x,is_controlled,given_control,u,
            force_control_to_inputs = force_control_to_inputs)

        # linear dynamics
        Dx = x - self.x_trim
        Du = inputs - self.u_trim
        dx = np.matmul(self.Lin_Model.A,Dx) + np.matmul(self.Lin_Model.B,Du)

        # x posn
        ## INTSTATE
        ud = x[0]
        vd = x[1]
        wd = x[2]
        EFvels = body_2_fixed([ud,vd,wd],[x[ 9],x[10],x[11],x[12]])
        dx[6] = EFvels[0]

        # limit rates
        if self.order >= 1:
            ## INTSTATE
            dx[13:17] = self._limit_input_rates(dx[13:17])

        # limit accelerations
        if self.order >= 2:
            ## INTSTATE
            dx[17:21] = self._limit_input_accelerations(dx[17:21])
        
        # integral states
        r = self._get_reference(t)[self.xPi]
        dx[self.xIi] = x[self.xPi]*1. - r

        return dx


    def _linear_euler_dynamics(self,t,x,
        is_controlled=True,given_control=False,u="o",
        force_control_to_inputs=False):

        # get control
        u,inputs = self._get_control(t,x,is_controlled,given_control,u,
            force_control_to_inputs = force_control_to_inputs)

        # linear dynamics
        xhat = x - self.x_trim
        uhat = inputs - self.u_trim
        dx = np.matmul(self.Lin_Model.A,xhat) + np.matmul(self.Lin_Model.B,uhat)

        # limit rates
        if self.order >= 1:
            ## INTSTATE
            dx[12:16] = self._limit_input_rates(dx[12:16])

        # limit accelerations
        if self.order >= 2:
            ## INTSTATE
            dx[16:20] = self._limit_input_accelerations(dx[16:20])
        
        # integral states
        r = self._get_reference(t)[self.xPi_eul]
        dx[self.xIi_eul] = x[self.xPi_eul]*1. - r

        return dx


    def _nonlinear_quaternion_dynamics(self,t,x,
        is_controlled=True,given_control=False,u="o",
        force_control_to_inputs=False):

        # get control
        u,inputs = self._get_control(t,x,is_controlled,given_control,u,
            force_control_to_inputs = force_control_to_inputs)

        # disturbance model
        ## INTSTATE
        V = (x[0]**2. + x[1]**2. + x[2]**2.)**0.5
        Du,Dv,Dw,Dp,Dq,Dr = self.get_disturbance(t,V)
        Vg = [Du,Dv,Dw]
        Wg = [Dp,Dq,Dr]

        # get aero forces
        Fx,Fy,Fz,Mx,My,Mz,g = self._aerodynamics(x,inputs,Vg=Vg,Wg=Wg)

        # read in mass properties
        W = self.inertia_model.W
        Ixx,Iyy,Izz,Ixy,Ixz,Iyz = self.inertia_model.inertia_results(inputs[3])
        Im1 = self.inertia_model.inverse_tensor(inputs[3])
        hx,hy,hz = self.inertia_model.angular_momentum_results()

        ## INTSTATE
        Vu = x[0]
        Vv = x[1]
        Vw = x[2]
        p = x[3]
        q = x[4]
        r = x[5]
        
        dx = x * 0.

        # u,v,w
        ## INTSTATE
        dx[0] = g/W*Fx + 2.*g*(x[10]*x[12] - x[11]*x[ 9]) + r*Vv - q*Vw
        dx[1] = g/W*Fy + 2.*g*(x[11]*x[12] + x[10]*x[ 9]) + p*Vw - r*Vu
        dx[2] = g/W*Fz + \
            g*(x[12]*x[12] + x[ 9]*x[ 9] - x[10]*x[10] - x[11]*x[11]) + \
            q*Vu - p*Vv

        # rhs for p,q,r
        pq = p*q; pr = p*r; qr = q*r
        p2, q2, r2 = p**2., q**2., r**2.
        rhs0 = r*hy - q*hz + Mx + (Iyy-Izz)*qr + Iyz*(q2-r2) + Ixz*pq - Ixy*pr
        rhs1 = p*hz - r*hx + My + (Izz-Ixx)*pr + Ixz*(r2-p2) + Ixy*qr - Iyz*pq
        rhs2 = q*hx - p*hy + Mz + (Ixx-Iyy)*pq + Ixy*(p2-q2) + Iyz*pr - Ixz*qr
        # p,q,r
        ## INTSTATE
        dx[3] = Im1[0][0]*rhs0 + Im1[0][1]*rhs1 + Im1[0][2]*rhs2
        dx[4] = Im1[1][0]*rhs0 + Im1[1][1]*rhs1 + Im1[1][2]*rhs2
        dx[5] = Im1[2][0]*rhs0 + Im1[2][1]*rhs1 + Im1[2][2]*rhs2
        
        ud = Vu
        vd = Vv
        wd = Vw
        ## INTSTATE
        EFvels = body_2_fixed([ud,vd,wd],[x[ 9],x[10],x[11],x[12]])
        dx[6] = EFvels[0]
        dx[7] = EFvels[1]
        dx[8] = EFvels[2]

        
        # e0,ex,ey,ez
        ## INTSTATE
        dx[ 9] = -0.5 * ( x[10]*x[3] + x[11]*x[4] + x[12]*x[5])
        dx[10] =  0.5 * ( x[ 9]*x[3] - x[12]*x[4] + x[11]*x[5])
        dx[11] =  0.5 * ( x[12]*x[3] + x[ 9]*x[4] - x[10]*x[5])
        dx[12] =  0.5 * (-x[11]*x[3] + x[10]*x[4] + x[ 9]*x[5])

        # actuator dynamics
        if self.order == 1:
            dx[13:17] = self._actuation_dynamics(x,u)
            # # relimit
            # if self.integrator == "odeint" and self.bool_limit_inputs:
            #     if   x[13] >= self.max_da  and dx[13] > 0.0: dx[13] = 0.0
            #     elif x[13] <= self.min_da  and dx[13] < 0.0: dx[13] = 0.0
            #     if   x[14] >= self.max_de  and dx[14] > 0.0: dx[14] = 0.0
            #     elif x[14] <= self.min_de  and dx[14] < 0.0: dx[14] = 0.0
            #     if   x[15] >= self.max_dr  and dx[15] > 0.0: dx[15] = 0.0
            #     elif x[15] <= self.min_dr  and dx[15] < 0.0: dx[15] = 0.0
            #     if   x[16] >= self.max_tau and dx[16] > 0.0: dx[16] = 0.0
            #     elif x[16] <= self.min_tau and dx[16] < 0.0: dx[16] = 0.0
        elif self.order == 2:
            dx[13:21] = self._actuation_dynamics(x,u)
        
        # integral states
        r = self._get_reference(t)[self.xPi]
        dx[self.xIi] = x[self.xPi]*1. - r

        return dx


    def _nonlinear_euler_dynamics(self,t,x,
        is_controlled=True,given_control=False,u="o",
        force_control_to_inputs=False):

        # get control
        u,inputs = self._get_control(t,x,is_controlled,given_control,u,
            force_control_to_inputs = force_control_to_inputs)

        # disturbance model
        ## INTSTATE
        V = (x[0]**2. + x[1]**2. + x[2]**2.)**0.5
        Du,Dv,Dw,Dp,Dq,Dr = self.get_disturbance(t,V)
        Vg = [Du,Dv,Dw]
        Wg = [Dp,Dq,Dr]

        # get aero forces
        Fx,Fy,Fz,Mx,My,Mz,g = self._aerodynamics(x,inputs,Vg=Vg,Wg=Wg)

        # read in mass properties
        W = self.inertia_model.W
        Ixx,Iyy,Izz,Ixy,Ixz,Iyz = self.inertia_model.inertia_results(u[3])
        Im1 = self.inertia_model.inverse_tensor(u[3])
        hx,hy,hz = self.inertia_model.angular_momentum_results()

        ## INTSTATE
        Vu = x[0]
        Vv = x[1]
        Vw = x[2]
        p = x[3]
        q = x[4]
        r = x[5]
        
        dx = x * 0.
        
        ## INTSTATE
        ph,th,ps = x[9],x[10],x[11] # self._euler_angles(x) # 
        cp = cos(ph); sp = sin(ph)
        ct = cos(th); st = sin(th)
        cs = cos(ps); ss = sin(ps)

        # u,v,w
        ## INTSTATE
        dx[0] = g/W*Fx - g*st    + r*Vv - q*Vw
        dx[1] = g/W*Fy + g*sp*ct + p*Vw - r*Vu
        dx[2] = g/W*Fz + g*cp*ct + q*Vu - p*Vv

        # rhs for p,q,r
        pq = p*q; pr = p*r; qr = q*r
        p2, q2, r2 = p**2., q**2., r**2.
        rhs0 = r*hy - q*hz + Mx + (Iyy-Izz)*qr + Iyz*(q2-r2) + Ixz*pq - Ixy*pr
        rhs1 = p*hz - r*hx + My + (Izz-Ixx)*pr + Ixz*(r2-p2) + Ixy*qr - Iyz*pq
        rhs2 = q*hx - p*hy + Mz + (Ixx-Iyy)*pq + Ixy*(p2-q2) + Iyz*pr - Ixz*qr
        # p,q,r
        ## INTSTATE
        dx[3] = Im1[0][0]*rhs0 + Im1[0][1]*rhs1 + Im1[0][2]*rhs2
        dx[4] = Im1[1][0]*rhs0 + Im1[1][1]*rhs1 + Im1[1][2]*rhs2
        dx[5] = Im1[2][0]*rhs0 + Im1[2][1]*rhs1 + Im1[2][2]*rhs2
        
        # x,y,z
        mat = [
            [ct*cs, sp*st*cs - cp*ss, cp*st*cs + sp*ss],
            [ct*ss, sp*st*ss + cp*cs, cp*st*ss - sp*cs],
            [-st, sp*ct, cp*ct]
        ]
        ## INTSTATE
        dx[6] = mat[0][0]*Vu + mat[0][1]*Vv + mat[0][2]*Vw
        dx[7] = mat[1][0]*Vu + mat[1][1]*Vv + mat[1][2]*Vw
        dx[8] = mat[2][0]*Vu + mat[2][1]*Vv + mat[2][2]*Vw

        
        # euler angles
        mat = [
            [1., sp*st/ct, cp*st/ct],
            [0., cp, -sp],
            [0., sp/ct, cp/ct]
        ]
        ## INTSTATE
        dx[ 9] = mat[0][0]*p + mat[0][1]*q + mat[0][2]*r
        dx[10] = mat[1][0]*p + mat[1][1]*q + mat[1][2]*r
        dx[11] = mat[2][0]*p + mat[2][1]*q + mat[2][2]*r

        # actuator dynamics
        if self.order == 1:
            dx[12:16] = self._actuation_dynamics(x,u)
        elif self.order == 2:
            dx[12:20] = self._actuation_dynamics(x,u)
        
        # integral states
        r = self._get_reference(t)[self.xPi_eul]
        dx[self.xIi_eul] = x[self.xPi_eul]*1. - r

        return dx


    def _nonlinear_euler_dynamics_VAB(self,t,x,
        is_controlled=True,given_control=False,u="o",
        force_control_to_inputs=False):
        # given state is in VAB format, get dynamics for VAB

        # # get control
        # u,inputs = self._get_control(t,x,is_controlled,given_control,u,
        #     force_control_to_inputs = force_control_to_inputs)

        # disturbance model
        ## INTSTATE
        V = x[0]
        Du,Dv,Dw,Dp,Dq,Dr = self.get_disturbance(t,V)
        Vg = [Du,Dv,Dw]
        Wg = [Dp,Dq,Dr]

        # get aero forces
        Fx,Fy,Fz,Mx,My,Mz,g = self._aerodynamics(x,u,Vg=Vg,Wg=Wg,
            is_VAB_format=True)

        # read in mass properties
        W = self.inertia_model.W
        Ixx,Iyy,Izz,Ixy,Ixz,Iyz = self.inertia_model.inertia_results(u[3])
        Im1 = self.inertia_model.inverse_tensor(u[3])
        hx,hy,hz = self.inertia_model.angular_momentum_results()

        ## INTSTATE
        Vu = x[0]*cos(x[1])*cos(x[2])
        Vv = x[0]*sin(x[2])
        Vw = x[0]*sin(x[1])*cos(x[2])
        p = x[3]
        q = x[4]
        r = x[5]
        
        dx = x * 0.
        
        ## INTSTATE
        ph,th,ps = x[9],x[10],x[11] # self._euler_angles(x) # 
        cp = cos(ph); sp = sin(ph)
        ct = cos(th); st = sin(th)
        cs = cos(ps); ss = sin(ps)

        # u,v,w
        ## INTSTATE
        dVu = g/W*Fx - g*st    + r*Vv - q*Vw
        dVv = g/W*Fy + g*sp*ct + p*Vw - r*Vu
        dVw = g/W*Fz + g*cp*ct + q*Vu - p*Vv
        # VAB
        dx[0] = (Vu*dVu + Vv*dVv + Vw*dVw)/V
        den = Vu**2. + Vw**2.
        dx[1] = (Vu*dVw - dVu*Vw)/den
        dx[2] = (dVv*V - Vv*dx[0])/V/den**0.5

        # rhs for p,q,r
        pq = p*q; pr = p*r; qr = q*r
        p2, q2, r2 = p**2., q**2., r**2.
        rhs0 = r*hy - q*hz + Mx + (Iyy-Izz)*qr + Iyz*(q2-r2) + Ixz*pq - Ixy*pr
        rhs1 = p*hz - r*hx + My + (Izz-Ixx)*pr + Ixz*(r2-p2) + Ixy*qr - Iyz*pq
        rhs2 = q*hx - p*hy + Mz + (Ixx-Iyy)*pq + Ixy*(p2-q2) + Iyz*pr - Ixz*qr
        # p,q,r
        ## INTSTATE
        dx[3] = Im1[0][0]*rhs0 + Im1[0][1]*rhs1 + Im1[0][2]*rhs2
        dx[4] = Im1[1][0]*rhs0 + Im1[1][1]*rhs1 + Im1[1][2]*rhs2
        dx[5] = Im1[2][0]*rhs0 + Im1[2][1]*rhs1 + Im1[2][2]*rhs2
        
        # x,y,z
        mat = [
            [ct*cs, sp*st*cs - cp*ss, cp*st*cs + sp*ss],
            [ct*ss, sp*st*ss + cp*cs, cp*st*ss - sp*cs],
            [-st, sp*ct, cp*ct]
        ]
        ## INTSTATE
        dx[6] = mat[0][0]*Vu + mat[0][1]*Vv + mat[0][2]*Vw
        dx[7] = mat[1][0]*Vu + mat[1][1]*Vv + mat[1][2]*Vw
        dx[8] = mat[2][0]*Vu + mat[2][1]*Vv + mat[2][2]*Vw

        
        # euler angles
        mat = [
            [1., sp*st/ct, cp*st/ct],
            [0., cp, -sp],
            [0., sp/ct, cp/ct]
        ]
        ## INTSTATE
        dx[ 9] = mat[0][0]*p + mat[0][1]*q + mat[0][2]*r
        dx[10] = mat[1][0]*p + mat[1][1]*q + mat[1][2]*r
        dx[11] = mat[2][0]*p + mat[2][1]*q + mat[2][2]*r

        # actuator dynamics
        if self.order == 1:
            dx[12:16] = self._actuation_dynamics(x,u)
        elif self.order == 2:
            dx[12:20] = self._actuation_dynamics(x,u)
        
        # integral states
        r = self._get_reference(t)[self.xPi]
        dx[self.xIi] = x[self.xPi]*1. - r

        return dx


    def _dynamics(self,t,x,is_controlled=True,given_control=False,u="o",
        force_control_to_inputs=False):

        # check if we can update the controls or not
        if t >= self.t_u_next_update:
            self.t_u_next_update = t + self.dt_u_update
            self.can_update = True

        # check that we have not hit gimbal lock
        if self.use_quaternions:
            euler = self._euler_angles(x)
            if euler[1] > 1.3962634015954636: # 80dg # gimbal lock, raise error
                self.t_gimbal = t
                raise ValueError("Hit gimbal lock t = {:> 8.4f}".format(t))
        else:
            ## INTSTATE
            if x[10] > 1.3962634015954636: # 80dg # gimbal lock, raise error
                self.t_gimbal = t
                raise ValueError("Hit gimbal lock t = {:> 8.4f}".format(t))

        # run dynamics
        dx = self._get_dynamics(t,x,\
            is_controlled=is_controlled,given_control=given_control,u=u,
            force_control_to_inputs=force_control_to_inputs)

        return dx


    def _rk4(self,t0,x0,dt):

        # calculate k values
        ht = 0.5 * dt
        k1 = self._dynamics(t0   ,x0        )
        k2 = self._dynamics(t0+ht,x0 + ht*k1)
        k3 = self._dynamics(t0+ht,x0 + ht*k2)

        # calculate derivatives
        ks = (k1 + 2.*(k2 + k3) + self._dynamics(t0+dt,x0 + dt*k3)) / 6.

        # update x1
        x1 = x0 + dt*ks

        return x1


    def _odeint(self,t0,x0,dt):

        # update x1
        x1 = odeint(self._dynamics,x0,[t0,t0+dt],tfirst=True)[1]

        return x1


    def _ode(self,t0,x0,dt):

        # update x1
        x1 = self.ode_integrator.integrate(self.ode_integrator.t+dt)

        return x1


    def _save_controller(self,Lin_Model,save_folder="plots/",filename=""):
        folder = self.fldr_prfx + "_" + save_folder
        system_notes = "bire_"*self.is_BIRE + "f_16_"*(not self.is_BIRE)
        system_notes += "quats_"*self.use_quaternions \
            + "euler_"*(not self.use_quaternions)
        system_notes += str(self.order) + "_ord_act_"
        system_notes += "system"
        if filename == "":
            filename = system_notes
        filename += ".mat"
        if self.use_quaternions:
            note = "x,y,z (7,8,9) (1 based indexing)" \
                + " states removed for state feedback design"
        else:
            note = "x,y,z and psi (7,8,9,12) (1 based indexing)" \
                + " states removed for state feedback design"

        file_dict = {
            "type" : Lin_Model.controller_type,
            "x_tr" : self.x_trim,
            "u_tr" : self.u_trim,
            "A" : Lin_Model.A,
            "B" : Lin_Model.B,
            "K" : Lin_Model.K,
            # "A_mdn" : self.Lin_Model.A_mdn,
            # "B_mdn" : self.Lin_Model.B_mdn,
            # "Q_mdn" : self.Lin_Model.Q_mdn,
            # "R_mdn" : self.Lin_Model.R_mdn,
            # "K_mdn" : self.Lin_Model.K_mdn,
            "note" : note,
            "system notes" : system_notes
        }
        if Lin_Model.controller_type == "LQR":
            file_dict["Q"] = Lin_Model.Q
            file_dict["R"] = Lin_Model.R
        savemat(folder+filename,file_dict,oned_as="column")


    def _build_controller(self,x_tr="o",u_tr="o",report=True,
        save_matrices=True,mrrr=None,mrrc=None,drop_actrs=True,run_freq=True,
        include_stall_derivatives=False,
        use_numerical_linearization=False,numerical_dynamics=None,
        use_VAB_format=False, turn_off_warnings=False,
        run2=False,
        save_folder="plots/",filename="",skip_reporting=False,save_name_end=""):
        # report
        if report:
            print("building controller...")

        if self.is_BIRE:
            name = "bire"
        else:
            name = "base"
        if self.is_rc:
            name += "_rc"
        else:
            name += "_fs"
        name += save_name_end

        # perform linearization, create feedback
        if isinstance(x_tr,str) or isinstance(u_tr,str):
            if run2:
                x_trim_euler = self.x_trim2_euler*1.
                u_trim = self.u_trim2*1.
            else:
                x_trim_euler = self.x_trim_euler*1.
                u_trim = self.u_trim*1.
            
        else:
            x_trim_euler = x_tr*1.
            u_trim = u_tr*1.
        Lin_Model = lin(
            # self.x_trim,
            x_trim_euler, # force euler linearization
            u_trim,self.cgshift,
            use_quaternion = self.use_quaternions,
            is_bire = self.is_BIRE,
            is_rc = self.is_rc,
            is_stevens_and_lewis = self.is_stevens_and_lewis,
            use_VAB_format = use_VAB_format,
            turn_off_warnings = turn_off_warnings,
            additional_states = self.additional_states,
            compressible = self.is_compressible,
            use_Anderson = self.use_anderson,
            enforce_stall = self.has_stall,
            include_stall = include_stall_derivatives,
            controller_type = self.controller_type,
            integral_states = self.xIi_eul,
            principal_states = self.xPi_eul,
            controller_properties = self.control_dict,
            actuators_properties = self.actuators_dict,
            aero_model = self.aero_model,
            use_simple_thrust_model = not self.use_fitted_thrust,
            use_numerical_linearization = use_numerical_linearization,
            numerical_dynamics = numerical_dynamics,
            min_realization_removal_rows = mrrr,
            min_realization_removal_cols = mrrc,
            drop_actuators = drop_actrs,
            run_frequency_analysis = run_freq,
            report = report,
            freq_folder = self.fldr_prfx + "_" + save_folder,
            controller_name = name
        )
        self.quat_linearization_built = True
        self.eulr_linearization_built = True     

        # store matrices
        if save_matrices:
            fold = self.fldr_prfx + "_" + save_folder
            self._save_controller(Lin_Model,save_folder=save_folder,
                filename=filename)
        
        if self.is_stevens_and_lewis:
            CL_a = 5.0 # fixed garbage value
        elif self.is_BIRE:
            CL_a = self.aero_model._CL_alpha(self.u_trim[2])*1.
        else:
            CL_a = self.aero_model.CLa*1.
        W = self.inertia_model.W*1.
        CW = W/0.5/self.rho0/self.V0**2./self.aero_model.S_w
        n_a = CL_a/CW

        # report trim condition, and linearized matrices
        repstr = ""
        if not(skip_reporting):
            ## INTSTATE
            if False: n = 13#self.use_quaternions: n = 13
            else: n = 12
            # trim condition
            repstr += report_latex(x_trim_euler[:n,np.newaxis].T,
                "x_{tr}",endln=True,transpose=True,print_report=report)
            repstr += report_latex(x_trim_euler[n:,np.newaxis].T,
                "\delta_{tr}",comquad=True,transpose=True,print_report=report)
            repstr += report_latex(u_trim[:,np.newaxis].T,"u_{tr}",
                transpose=True,print_report=report)
            # # dynamical matrices
            # repstr += report_latex(Lin_Model.A[0:n][:,0:6],"A_{dyn \, 1}",
            #     predecimals=5,align=True,endln=True,print_report=report)
            # repstr += report_latex(Lin_Model.A[0:n][:,6:n],"A_{dyn \, 2}",
            #     predecimals=5,align=True,endln=True,print_report=report)
            # if self.order == 1:
            #     repstr += report_latex(-Lin_Model.A[n:][:,n:],r"\Upsilon",
            #         predecimals=5,align=True,endln=True,print_report=report)
            # # 2nd order here
            # if drop_actrs or self.order == 0:   Bdyn = Lin_Model.B[0:n]
            # elif self.order == 1: Bdyn = Lin_Model.A[:n,n:]
            # else: pass
            # repstr += report_latex(Bdyn,"B_{dyn}",align=True,
            #     print_report=report)
            repstr += report_latex(Lin_Model.A_full,"A_{full}",
                predecimals=5,align=True,endln=True,print_report=report)
            repstr += report_latex(Lin_Model.B_full,"B_{full}",
                predecimals=5,align=True,print_report=report)
            # print(Lin_Model.B_full[5,1])
            reorganize = False
            if reorganize:
                # this doesn't include integrator states
                rows = [0,2,4,6,8,1,3,5,7]
                cols = [1,3,0,2]
            else:
                rows = list(range(Lin_Model.A_min.shape[0]))
                cols = list(range(Lin_Model.B_min.shape[1]))
            repstr += report_latex((Lin_Model.A_min[rows,:])[:,rows],"A",
                predecimals=5,align=True,endln=True,print_report=report)
            repstr += report_latex((Lin_Model.B_min[rows,:])[:,cols],"B",
                align=True,print_report=report)
            # open-loop eigenvalues
            # report_latex(Lin_Model.A_eigs,"\lambda_{ol}")#,decimals=16)
            repstr += report_latex(Lin_Model.A_min_eigs,"\lambda_{ol}",
                print_report=report)#,decimals=16)
            repstr += report_latex(Lin_Model.A_min_evecs,"\chi_{ol}",
                predecimals=3,decimals=4,print_report=report,eigvecs=True)
            if not self.is_stevens_and_lewis:
                repstr += report_eigprops(Lin_Model.A_min_eigs,n_a=n_a,
                    print_report=report)
            # sensitivity matrices
            if Lin_Model.controller_type == "LQR":
                repstr += report_latex(np.diag(Lin_Model.Q_min),"Q",
                    diag=True,comquad=True,sci=True,print_report=report)
                repstr += report_latex(np.diag(Lin_Model.R_min),"R",diag=True,
                    sci=True,print_report=report)
            # state feedback and closed-loop eigenvalues
            ctrb_str = "controllability rank = " + \
                str(Lin_Model.Gamma_rank_min) + "\n\n"
            if report:
                print(ctrb_str,end="")
            repstr += ctrb_str
            if len(self.xIi):
                ps = "_P"
            else:
                ps = ""
            repstr += report_latex(Lin_Model.K,"K"+ps,print_report=report)#,sci=True)
            repstr += rep2D(Lin_Model.K,"K"+ps,decimals=16)
            if len(self.xIi):
                repstr += report_latex(Lin_Model.KI,"K_I",print_report=report)
                repstr += rep2D(Lin_Model.KI,"K_I",decimals=16)
            # repstr += report_latex(Lin_Model.K.T,"K",transpose=True,
            #     print_report=report)#,sci=True)
            # report_latex(Lin_Model.K.T,"K",transpose=True,sci=True)
            if Lin_Model.controller_type == "LQR":
                repstr += report_latex(Lin_Model.P,"P",print_report=report)
            repstr += report_latex(Lin_Model.A_BK_eigs,"\lambda_{cl}",
                print_report=report)#,decimals=16)
            repstr += report_latex(Lin_Model.A_BK_evecs,"\chi_{cl}",
                predecimals=3,decimals=4,print_report=report,eigvecs=True)
            if not self.is_stevens_and_lewis:
                repstr += report_eigprops(Lin_Model.A_BK_eigs,n_a=n_a,
                    print_report=report)
        
        if isinstance(x_tr,str) or isinstance(u_tr,str):
            if run2:
                self.Lin_Model2 = Lin_Model
            else:
                self.Lin_Model = Lin_Model
            
            return repstr
        else:
            return repstr,Lin_Model


    def initialize_sim(self,x,atol=1e-12,rtol=1e-6,nonlinear=True,quat=True):
        """Method which initializes the state and time of the simulator.
        """
        # atol defaults to 1e-12
        # rtol defaults to 1e-6

        # # report
        # print("Running Simulator...")

        # initialize variables to store
        t = 0.0

        # initialize integration
        if self.integrator == "ode":
            self.ode_integrator.set_integrator("dopri5",atol=atol,rtol=rtol)
            self.ode_integrator.set_initial_value(x * 1., 0.0)
        
        return x,t


    def sim_step(self,dt,x):
        """ Method which steps once through the simulation integration
        sequence.

        Parameters
        ----------
        dt : float
            The time step of the integration step.
        
        u : numpy array
            The controls state array, size (6,). Contains aileron [0], 
            elevator [1], rudder [2], and throttle[3]. Has two empty spots used 
            in the trim algorithm.
        
        Returns
        -------
        x : numpy array
            The state array, size (13,). Contains u [0], v [1], w [2], p [3], 
            q [4], r [5], xf [6], yf [7], zf [8], e0 [9], e1 [10], e2 [11],
            e3 [12].
        """
        
        # propogate
        self.t = self.t + dt
        xm1 = x * 1.
        x = self.int_method(self.t,x,dt)

        # limit actuators
        if self.order >= 1:
            q = 1*self.use_quaternions
            ## INTSTATE
            x[12+q:16+q] = self._quantize_input(self._limit_input(x[12+q:16+q]))

        # normalize quaternion
        if self.use_quaternions:
            ## INTSTATE
            x[9:13] = quat_norm(x[9:13])

        return x


    def controls(self):
        """Method which returns the control values.
        
        Returns
        -------
        controls : array
            The controls of the aircraft.
        """

        return self.u


    def aero_angles(self,x):
        """ Method which determine the magnitude of the velocity, the angle of
        attack, and the sideslip angle for the given state.

        Returns
        -------
        V : float
            The velocity of the aircraft.
        
        a : float
            The angle of attack of the aircraft, in radians.

        b : float
            The sideslip angle of the aircraft, in radians.
        """

        # calculate
        ## INTSTATE
        V = (x[0]*x[0]+x[1]*x[1]+x[2]*x[2])**0.5
        a = atan2(x[2],x[0])
        b = asin(x[1]/V)

        return V,a,b


    def _euler_angles(self,x):
        """ Method which determines the euler angles from the member state 
        array.
        
        Returns
        -------
        phi : float
            The bank angle, in radians.
        
        theta : float
            The elevation angle, in radians.
        
        psi : float
            The azimuth angle, in radians.
        """

        ## INTSTATE
        return quat_2_euler(x[9:13])


    def euler_angles(self):
        """ Method which determines the euler angles from the member state 
        array.
        
        Returns
        -------
        phi : float
            The bank angle, in radians.
        
        theta : float
            The elevation angle, in radians.
        
        psi : float
            The azimuth angle, in radians.
        """

        return self._euler_angles(self.x)


    def quat2euler_state(self,x):
        euler = self._euler_angles(x)
        ## INTSTATE
        ex = np.delete(x,12)*1.
        ex[9:12] = euler
        return ex


    def _load_factors(self,x,u,axis="stab"):
        """Method whichs determines the loading on the aircraft, in one of
        three specified types of axes.
        
        Returns
        -------
        axial_load_factor : float
            The load factor on the aircraft in the axial direction.
            
        side_load_factor : float
            The load factor on the aircraft in the side direction.
            
        normal_load_factor : float
            The load factor on the aircraft in the normal direction.
        """
        poss_axes = ["body","wind","stab"]
        if axis not in poss_axes:
            raise ValueError("Load factor axis type '{}'".format(axis) 
                + " not supported. Must be one of " 
                + "[{},{},{}]".format(poss_axes[0],poss_axes[1],poss_axes[2]))

        # determine forces
        Fx,Fy,Fz,_,_,_,g = self._aerodynamics(x,u)

        # multiply by corresponding axis type
        Vu,Vv,Vw = x[0], x[1], x[2]
        a = atan2(Vw,Vu)
        V = (Vu * Vu + Vv * Vv + Vw * Vw)**0.5
        b = asin(Vv/V)
        ca = cos(a); sa = sin(a)
        cb = cos(b); sb = sin(b)
        if axis == "body":
            M = np.eye(3)
        elif axis == "stab":
            M = np.array([
                [ ca,0.0, sa],
                [0.0,1.0,0.0],
                [-sa,0.0, ca]
            ])
        elif axis == "wind":
            M = np.array([
                [ ca*cb,    sb, sa*cb],
                [-ca*sb,    cb,-sa*sb],
                [   -sa,   0.0,    ca]
            ])
        # multiply
        [Fx,Fy,Fz] = np.matmul(M,[Fx,Fy,Fz])

        # read in mass properties
        W = self.inertia_model.W

        return -Fx/W,Fy/W,-Fz/W


    def get_error_scales(self,change_zeros=True):

        # save error scale values used for later
        aeroscale = {}
        inerscale = {}

        if self.is_BIRE:
            # aero errors
            # add errors
            for i,key in enumerate(self.aero_keys):
                i0 = key.split("_")[0]
                subdict= self.aero_model.__dict__[key]
                aeroscale[i0] = {}
                for j,subkey in enumerate(subdict.keys()):
                    i1 = subkey.split("_")[1]
                    i1 = i1.replace("lpha","")
                    i1 = i1.replace("eta","")
                    i1 = i1.replace("bar","")
                    aeroscale[i0][i1] = {}
                    subsubdict = subdict[subkey]
                    for k,subsubkey in enumerate(subsubdict.keys()):
                        if subsubkey in ["A","w","phi","z"]:

                            if change_zeros and not \
                                self.truth_aero_model.__dict__[key][\
                                subkey][subsubkey]:
                                # find a 'comparable' nonzero value
                                n =np.nonzero([self.truth_aero_model.__dict__[\
                                    key][sub][subsubkey] for sub in \
                                    self.truth_aero_model.__dict__[key]])[0]
                                
                                # make it a nonzero value subkey
                                sub = list(self.truth_aero_model.__dict__[key\
                                    ].keys())[n[0]]
                            else:
                                sub = subkey

                            aeroscale[i0][i1][subsubkey] = \
                                abs(self.truth_aero_model.__dict__[key][sub\
                                    ][subsubkey])*1.
            
            # inertia errors
            inerscale["Ixx"] = {}
            inerscale["Iyy"] = {}
            inerscale["Izz"] = {}
            inerscale["Ixy"] = {}
            inerscale["Ixz"] = {}
            inerscale["Iyz"] = {}
            # add errors
            for i,key in enumerate(self.inertia_keys):
                i0,i1 = key.split("_")
                if change_zeros and key in ["Ixx_A","Ixx_w","Ixx_p"]:
                    tkey = "Iyy_" + key[-1]
                elif change_zeros and not self.truth_inertia_model.__dict__[key]:
                    if key[-1] == "p":
                        tkey = "Izz_p"
                    else:
                        tkey = "Iyz_" + key[-1]
                else:
                    tkey = key
                inerscale[i0][i1] = abs(\
                    self.truth_inertia_model.__dict__[tkey])*1.

        else:
            # aero errors
            aeroscale["CL"] = {}
            aeroscale["CS"] = {}
            aeroscale["CD"] = {}
            aeroscale["Cl"] = {}
            aeroscale["Cm"] = {}
            aeroscale["Cn"] = {}
            for i,key in enumerate(self.aero_keys):
                i0 = key[:2]
                i1 = key[2:]
                aeroscale[i0][i1] = abs(self.truth_aero_model.__dict__[key])*1.
            
            # inertia errors
            for i,key in enumerate(self.inertia_keys):
                i0,i1 = key.split("_")
                if change_zeros and key[:3] in ["Ixy","Iyz"]:
                    tkey = "Ixz_z"
                else:
                    tkey = key
                inerscale[i0] = abs(self.truth_inertia_model.__dict__[tkey])*1.
        
        # weight error
        inerscale["W"] = self.truth_inertia_model.W*1.

        # gyroscopic error
        inerscale["hx"] = self.truth_inertia_model.hx*1.
        inerscale["hy"] = self.truth_inertia_model.hx*1.
        inerscale["hz"] = self.truth_inertia_model.hx*1.
        self.aero_scale_errors = aeroscale
        self.iner_scale_errors = inerscale

        return


    def refresh_models_error(self,aero_error={},
        inertia_error={},change_zeros=True):

        # save error values used for later
        self.aero_model.errors = {}
        self.inertia_model.errors = {}
        # retrieve scale values
        aeroscale = self.aero_scale_errors
        inerscale = self.iner_scale_errors

        if self.is_BIRE:
            # aero errors
            # add errors
            for i,key in enumerate(self.aero_keys):
                i0 = key.split("_")[0]
                self.aero_model.errors[i0] = {}
                subdict= self.aero_model.__dict__[key]
                for j,subkey in enumerate(subdict.keys()):
                    i1 = subkey.split("_")[1]
                    i1 = i1.replace("lpha","")
                    i1 = i1.replace("eta","")
                    i1 = i1.replace("bar","")
                    self.aero_model.errors[i0][i1] = {}
                    subsubdict = subdict[subkey]
                    for k,subsubkey in enumerate(subsubdict.keys()):
                        if subsubkey in ["A","w","phi","z"]:
                            # set key err value
                            key_err = self.err_rng.normal(loc=0.0,\
                                scale=aero_error[i0][i1][subsubkey])
                            # save for plots
                            self.aero_model.errors[i0][i1][subsubkey] = key_err
                            # add to model
                            err_val = aeroscale[i0][i1][subsubkey]*key_err + \
                                self.truth_aero_model.__dict__[key][subkey][\
                                subsubkey]                
                            self.aero_model.__dict__[key][subkey][subsubkey] =\
                                err_val
            
            # inertia errors
            # formulate errors dict
            self.inertia_model.errors["Ixx"] = {}
            self.inertia_model.errors["Iyy"] = {}
            self.inertia_model.errors["Izz"] = {}
            self.inertia_model.errors["Ixy"] = {}
            self.inertia_model.errors["Ixz"] = {}
            self.inertia_model.errors["Iyz"] = {}
            # add errors
            for i,key in enumerate(self.inertia_keys):
                i0,i1 = key.split("_")
                key_err = self.err_rng.normal(loc=0.0,\
                    scale=inertia_error[i0][i1])
                # save for plots
                self.inertia_model.errors[i0][i1] = key_err
                # add to model
                err_val = inerscale[i0][i1]*key_err + \
                    self.truth_inertia_model.__dict__[key]
                self.inertia_model.__dict__[key] = err_val

        else:
            # aero errors
            self.aero_model.errors["CL"] = {}
            self.aero_model.errors["CS"] = {}
            self.aero_model.errors["CD"] = {}
            self.aero_model.errors["Cl"] = {}
            self.aero_model.errors["Cm"] = {}
            self.aero_model.errors["Cn"] = {}
            for i,key in enumerate(self.aero_keys):
                i0 = key[:2]
                i1 = key[2:]
                key_err = self.err_rng.normal(loc=0.0,scale=aero_error[i0][i1])
                # save for plots
                self.aero_model.errors[i0][i1] = key_err
                # add to model
                err_val = aeroscale[i0][i1]*key_err + \
                    self.truth_aero_model.__dict__[key]
                self.aero_model.__dict__[key] = err_val
            
            # inertia errors
            for i,key in enumerate(self.inertia_keys):
                i0,i1 = key.split("_")
                key_err = self.err_rng.normal(loc=0.0,scale=inertia_error[i0])
                # save for plots
                self.inertia_model.errors[i0] = key_err
                # add to model
                err_val = inerscale[i0]*key_err + \
                    self.truth_inertia_model.__dict__[key]
                self.inertia_model.__dict__[key] = err_val
        
        # weight error
        key_err = self.err_rng.normal(loc=0.0, scale=inertia_error["W"])
        self.inertia_model.W = inerscale["W"]*key_err + \
            self.truth_inertia_model.W
        self.inertia_model.errors["W"] = key_err

        # gyroscopic error
        key_err = self.err_rng.normal(loc=0.0, scale=inertia_error["hx"])
        self.inertia_model.hx = inerscale["hx"]*key_err + \
            self.truth_inertia_model.hx
        self.inertia_model.errors["hx"] = key_err
        key_err = self.err_rng.normal(loc=0.0, scale=inertia_error["hy"])
        self.inertia_model.hy = inerscale["hy"]*key_err + \
            self.truth_inertia_model.hy
        self.inertia_model.errors["hy"] = key_err
        key_err = self.err_rng.normal(loc=0.0, scale=inertia_error["hz"])
        self.inertia_model.hz = inerscale["hz"]*key_err + \
            self.truth_inertia_model.hz
        self.inertia_model.errors["hz"] = key_err

        return


    def make_errored_models(self,aero_error={},
        inertia_error={}):
        print("Making errored aero and mass models...")
        # pull reference for current aero and inertia models
        self.truth_aero_model = self.aero_model
        del self.aero_model
        self.truth_inertia_model = self.inertia_model
        del self.inertia_model

        # create new aero model
        aero_dict = {
            "inp_dir" : aero_directory,
            "thrust_dir" : aero_directory,
            "use_fitted_thrust_model" : self.use_fitted_thrust,
            "atmosphere_model" : self.stdatm,
            "rho_index_in_model" : 4
        }
        if self.is_BIRE:
            self.aero_model = BIREAero(**aero_dict)
        else:
            self.aero_model = F16Aero(**aero_dict)
        # initialize inertia model
        self.inertia_model = InertiaModel(inp_dir=mass_directory, \
            is_bire=self.is_BIRE)
        
        # determine keys for adding error
        if self.is_BIRE:
            self.aero_keys = [key for key in self.aero_model.__dict__.keys() \
                if key[0] == "C" and key[-7:] == "_coeffs"]
            self.inertia_keys = \
                [key for key in self.inertia_model.__dict__.keys() \
                if key[-2:] in ["_A","_w","_p","_z"]]
        else:
            self.aero_keys = [key for key in self.aero_model.__dict__.keys() \
                if key[0] == "C" and key[-7:] != "_coeffs"]
            self.inertia_keys = \
                [key for key in self.inertia_model.__dict__.keys() \
                if key[-2:] == "_z"]
        
        # determine scales for errors
        self.get_error_scales()

        # add in error
        self.refresh_models_error(aero_error,inertia_error)

        return


    def refresh_FM_error(self,FM_error_percs):

        self.FM_errors = self.err_rng.normal(loc=0.0,\
            scale=FM_error_percs,size=(6,))

        return


    def make_FM_error_model(self,
        FM_error_percs=[0.25,0.25,0.25,0.25,0.25,0.25]):
        print("making FM error models...")

        # add in error
        self.refresh_FM_error(FM_error_percs)

        return 


    def _report_simulation_deltas(self):
        # report max vals from trim
        xs = self.xarr - self.x_trim_euler[:,None]
        us = self.uarr - self.u_trim[:,None]
        state_names = [
            "Vxb","Vyb","Vzb",
            "p","q","r",
            "xf","yf","zf",
            "phi","theta","psi",
            "da","de"+"B"*self.is_BIRE,
            "dB"*self.is_BIRE+"dr"*(not(self.is_BIRE)),"tau",
            "da dot","de"+"B"*self.is_BIRE+" dot",
            "dB"*self.is_BIRE+"dr"*(not(self.is_BIRE))+" dot","tau dot"
        ]
        state_units = ["ft/s"]*3 + ["deg/s"]*3 + ["ft"]*3 + \
            ["deg"]*6 + [""] + ["deg/s"]*3 + [""]
        control_names = [
            "da cmd","de"+"B"*self.is_BIRE+" cmd",
            "dB"*self.is_BIRE+"dr"*(not(self.is_BIRE))+" cmd","tau cmd"
        ]
        control_units = ["deg"]*3 + [""]
        n = 25
        print("-"*(n*2+14))
        print("-"*n + "  max states  " + "-"*n)
        for i in range(xs.shape[0]):
            vals = xs[i,:]
            name = state_names[i]
            unit = state_units[i]
            max_val = np.max(np.abs(vals))
            print("max{:^10s}= {:> 9.3f} {:<5s}".format("\u0394"+name,max_val,\
                unit))
        print("-"*n + " max controls " + "-"*n)
        for i in range(us.shape[0]):
            vals = us[i,:]
            name = control_names[i]
            unit = control_units[i]
            max_val = np.max(vals)
            min_val = np.min(vals)
            print("max{:^10s}= {:> 9.3f} {:<5s}".format("\u0394"+name,max_val,\
                unit),end="")
            print(", min{:^10s}= {:> 9.3f} {:<5s}".format("\u0394"+name,\
                min_val,unit))
        print("-"*(n*2+14))
        print("-"*(n*2+14))
        return


    def _add_to_delta_x0(self,delta_x0):
        return delta_x0


    def run_simulation(self,report_controller=False,report_trim=True,
        save_matrices=True,mrrr=None,mrrc=None,delta_x0=None,
        include_stall_derivatives=False,
        actr_warm_start=False,report_simulation=True,
        report_simulation_deltas=False):

        # build controller if not built yet
        try:
            self.Lin_Model
        except:
            self._build_controller(report=report_controller,
                save_matrices=save_matrices,mrrr=mrrr,mrrc=mrrc,
                include_stall_derivatives=include_stall_derivatives,
                run_freq=False)
        
        if report_trim:
            self._report_trim_solution(self.x_trim,self.u_trim,self.trim_iter)

        # report
        if report_simulation:
            print("running simulation...")

        # add delta to state
        if delta_x0 is not None:
            delta_x0 = np.concatenate((delta_x0,[0.]*self.additional_states))
            delta_x0 = self._add_to_delta_x0(delta_x0)
            x0 = self.x0 + delta_x0
        else:
            x0 = self.x0*1.
        
        # warm start actuator
        if self.order >= 1 and actr_warm_start:
            inputs,_ = self._get_control(0.,x0,is_controlled=not(self.run_unctrl))
            q = 1*self.use_quaternions
            x0[12+q:16+q] = ( np.array(inputs)*1. ).tolist()

        # initialize lists, form of file writing
        self.n_steps = int( self.tf / self.dt) + 1
        self.xarr = np.zeros((self.x_trim.shape[0]-1*self.use_quaternions,\
            self.n_steps))
        if self.use_quaternions:
            ind2kp = [i for i in range(len(self.x_trim)) if i != 12]
        else:
            ind2kp = [i for i in range(len(self.x_trim))]
        x0_euler = x0[ind2kp] * 1.0
        if self.use_quaternions:
            x0_euler[9:12] = self._euler_angles(x0)
        x0_euler[9:12] = np.rad2deg(x0_euler[9:12])
        self.xarr[:,0] = x0_euler*1.
        self.uarr = np.zeros((4,self.n_steps))
        self.uarr[:,0],_ = self._get_control(0.0,x0,
                    is_controlled=not(self.run_unctrl))
        self.tarr = np.zeros((self.n_steps,))
        self.tarr[0] = 0.

        # ###########################
        # # read in sim output, set t array to that
        # self.Lin_Model.K = self.Lin_Model.K*0.0
        # ts_sim = []
        # xs_sim = []
        # us_sim = []
        # with open("sim_output.txt","r") as f:
        #     for line in f.readlines():

        #         simpline =     line.replace("LogTemp: Warning: Time =","")
        #         simpline = simpline.replace("CurrentStates =","")
        #         simpline = simpline.replace("Controls =","")
        #         # print(simpline)
        #         vals = [float(val) for val in simpline.split(",")]
        #         ts_sim.append(vals[0])
        #         xs_sim.append(vals[1:17])
        #         us_sim.append(vals[17:21])
        #     f.close()
        # ts_sim = np.array(ts_sim)
        # xs_sim = np.array(xs_sim).T
        # us_sim = np.array(us_sim).T
        # self.xarr = np.delete(self.xarr,slice(xs_sim.shape[1],self.xarr.shape[1]),axis=1)
        # self.uarr = np.delete(self.uarr,slice(us_sim.shape[1],self.uarr.shape[1]),axis=1)
        # self.n_steps = xs_sim.shape[1]
        # self.tf = ts_sim[-1]
        # xs_sim[3:6,0] = xs_sim[3:6,0] + delta_x0[3:6]
        # # print(xs_sim.shape)
        # # print(self.xarr.shape)
        # # print(us_sim.shape)
        # # print(self.uarr.shape)
        # # quit()
        # ###########################

        # controller timing variables
        self.t_u_next_update = 0.0
        self.can_update = True

        self._reinitialize()
                
        # begin simulation
        counter = 1
        self.x,self.t = self.initialize_sim(x0,\
            atol=1e-10,rtol=1e-10,
            nonlinear=self.use_nonlinear,quat=self.use_quaternions)
        
        if self.integrator == "odeint":
            ts = np.linspace(0.,self.tf,num=self.n_steps,endpoint=False)
            self.tarr = ts*1.
            try:
                xs = odeint(self._dynamics,x0,ts,tfirst=True,
                    atol=1e-10,rtol=1e-10
                    )[1:].T
            except:
                n_sub = min(self.t_gimbal/self.dt,10.)
                t_index = np.argwhere(ts<=self.t_gimbal-self.dt*n_sub)[:,0][-1]
                xs = odeint(self._dynamics,x0,ts[:t_index],tfirst=True,
                    atol=1e-10,rtol=1e-10
                    )[1:].T
                end = np.kron(xs[:,-1],np.ones((ts.shape[0]-t_index,1)))
                xs = np.block([xs,end.T])
            self.t_u_next_update = 0.0
            for i in range(self.n_steps-1):
                # for plots
                # check if we can update the controls or not
                if self.tarr[i] >= self.t_u_next_update:
                    self.t_u_next_update = self.tarr[i] + self.dt_u_update
                    self.can_update = True
                self.uarr[:,i+1],_ = self._get_control(ts[i],xs[:,i],
                    is_controlled=not(self.run_unctrl))
                if self.use_quaternions:
                    self.xarr[9:12,i+1] = self._euler_angles(xs[:,i])
                else:
                    self.xarr[9:12,i+1] = xs[9:12,i]
            self.xarr[9:12,1:] = np.rad2deg(self.xarr[9:12,1:])
            self.xarr[:9,1:] = xs[:9,:]
            if self.use_quaternions:
                self.xarr[12:,1:] = xs[13:,:]
            else:
                self.xarr[12:,1:] = xs[12:,:]
        else:
            ts = np.linspace(0.,self.tf,num=self.n_steps,endpoint=False)
            # ###########################
            # # print("true!!!!!")
            # ts = ts_sim*1.
            # ###########################
            self.tarr = ts*1.
            for i in range(self.n_steps-1):
                # for plots
                
                # run a step
                self.x_old = self.x*1.
                dt = ts[i+1] - ts[i]
                self.x = self.sim_step(dt,self.x)
                u,_ = self._get_control(self.tarr[i],self.x_old)
                # Dx = self.x[self.Lin_Model.Cslice]-self.Lin_Model.xhat_eq
                # Dxn = np.linalg.norm(Dx)
                # if self.t > 2.0 and Dxn <= 1.99:
                #     return self.xarr[:,:i+1],self.uarr[:,:i+1]

                # save state
                x_euler = self.x[ind2kp]*1.
                if self.use_quaternions:
                    x_euler[9:12] = self._euler_angles(self.x)
                x_euler[9:12] = np.rad2deg(x_euler[9:12])
                #################################
                # this is commented out for now
                # if no gimbal lock, return
                if x_euler[10] > 80.:
                    return self.xarr[:,:i+1],self.uarr[:,:i+1]
                ################################
                # save to arrays
                self.xarr[:,counter] = x_euler*1.
                self.uarr[:,counter] = u
                self.tarr[counter] = self.t*1.
                counter = counter + 1
        
        if report_simulation:
            print("finished simulating...")
            # print(np.max(self.xarr[6,:]))
            # print(np.min(self.xarr[6,:]))
        
        # sim deltas
        if report_simulation_deltas:
            self._report_simulation_deltas()

        # calculate total velocity and aero angles
        Vxarr = (self.xarr[0]**2. + self.xarr[1]**2. + self.xarr[2]**2.)**0.5
        axarr = np.rad2deg(np.arctan2(self.xarr[2],self.xarr[0]))
        bxarr = np.rad2deg(np.arcsin(self.xarr[1]/Vxarr)) # experimental beta
        Mxarr = np.array([Vxarr[i]/self.stdatm(-self.xarr[8,i])[5] \
            for i in range(len(self.tarr))])
        self.aerox = np.array([Vxarr,Mxarr,axarr,bxarr])

        # ############################
        # # # remove unreal sim states
        # # self.xarr = self.xarr - xs_sim
        # # self.uarr = self.uarr - us_sim
        # self.xs_sim = xs_sim
        # self.us_sim = us_sim
        # ############################

        # convert to degrees
        xicnv = [3,4,5] + [12,13,14]*(self.order >=1) + \
            [16,17,18]*(self.order >1)
        for xPii in range(len(self.xPi_eul)):
            if self.xPi_eul[xPii] in xicnv + [9,10,11]:
                xicnv += [self.xIi_eul[xPii]]
        self.xicnv = xicnv + [9,10,11]
        uicnv = [0,1,2]
        self.xarr[xicnv,:] = np.rad2deg(self.xarr[xicnv,:])
        self.uarr[uicnv,:] = np.rad2deg(self.uarr[uicnv,:])
        # #####################################
        # xeul = [9,10,11]
        # self.xs_sim[xeul,:] = np.rad2deg(self.xs_sim[xeul,:])
        # self.xs_sim[xicnv,:] = np.rad2deg(self.xs_sim[xicnv,:])
        # self.us_sim[uicnv,:] = np.rad2deg(self.us_sim[uicnv,:])
        # ####################################

        return self.xarr,self.uarr


    def run_uncontrolled_comparison_simulation(self):

        # build controller
        self._build_controller()#mrrr=[0,1,2,6,7,8,9,10,11,12])

        # print("wn da", self.w_da, self.s_da / self.z_da)
        # print("wn de", self.w_de, self.s_de / self.z_de)
        # print("wn dr", self.w_dr, self.s_dr / self.z_dr)
        # print("wn tau", self.w_tau, 1.0 / self.z_tau)

        # print("zt da", self.z_da, self.s_da / self.w_da)
        # print("zt de", self.z_de, self.s_de / self.w_de)
        # print("zt dr", self.z_dr, self.s_dr / self.w_dr)
        # print("zt tau", self.z_tau, 1.0 / self.w_tau)

        self._report_trim_solution(self.x_trim,self.u_trim,self.trim_iter)

        # report
        print("running simulation...")

        # set state control vars as equilibrium control vars
        u_eq = self.u_trim[:4] * 1.0

        # create equilibrium truncated state for controller
        regind = [6,7,8,11,12,16,20]
        if self.order == 0:
            ordind = [13,14,15,17,18,19]
        if self.order == 1:
            ordind = [17,18,19]
        elif self.order == 2:
            ordind = []
        index = [i for i in range(len(self.x)) if i not in (regind + ordind)]
        xhat_eq = self.x_trim[index] * 1.0
        if self.use_quaternions:
            xhat_eq[6:8] = self._euler_angles(self.x_trim)[:2]
        u = self.u[0:4] * 1.0
        v = u * 1.0
        self.z = self.x

        self.u_eq = u_eq * 1.0

        # initialize lists, form of file writing
        self.n_steps = int( self.tf / self.dt) + 1
        self.xarr = np.zeros((self.x.shape[0]-1*self.use_quaternions,\
            self.n_steps))
        ind2kp = [i for i in range(len(self.x)) if i != 12]
        x0_euler = self.x[ind2kp] * 1.0
        if self.use_quaternions:
            x0_euler[9:12] = self._euler_angles(self.x)


        # # self.z = np.append(x0_euler,0.0) * 1.0
        # # self.z[9:12] = [self.phi0,self.theta0,self.psi0]
        # jim = self.z * 1.0
        # if self.use_quaternions:
        #     jim[9:12] = self._euler_angles(self.z * 1.0)
        # self.z,self.t = self.initialize_sim(self.z,\
        #     nonlinear=self.use_nonlinear,quat=self.use_quaternions)
        # # frint("t =",0.0)
        # # rep2D(jim[:12,np.newaxis].reshape((3,4)),"x",predecimals=7,decimals=16)
        # # print()

        x0_euler[9:12] = np.rad2deg(x0_euler[9:12])
        self.xarr[:,0] = x0_euler * 1.0
        self.zarr = np.zeros((self.x.shape[0]-1*self.use_quaternions,\
            self.n_steps))
        self.zarr[:,0] = x0_euler * 1.0
        self.uarr = np.zeros((4,self.n_steps))
        self.uarr[:,0] = u * 1.0
        self.varr = np.zeros((4,self.n_steps))
        self.varr[:,0] = v * 1.0
        self.tarr = np.zeros((self.n_steps,))
        self.tarr[0] = self.t * 1.0

        # # begin simulation of uncontrolled
        # if self.run_unctrl:
        #     counter = 1
        #     for vee in range(self.n_steps-1):
        #         # run a step
        #         self.z = self.sim_step(self.dt,self.z,v)

        #         # save state
        #         z_euler = self.z[ind2kp] * 1.0
        #         if self.use_quaternions:
        #             z_euler[9:12] = self._euler_angles(self.z * 1.0)

        #         # frint("t =",self.t)
        #         # rep2D(z_euler[:12,np.newaxis].reshape((3,4)),"x",predecimals=7,decimals=16)
        #         # print()

        #         z_euler[9:12] = np.rad2deg(z_euler[9:12])

        #         # if counter >= 78000:# 72: # 11
        #         #     quit()
        #         self.zarr[:,counter] = z_euler * 1.0
        #         self.varr[:,counter] = v * 1.0
        #         counter = counter + 1

        #     frint("t =",self.t)
        #     rep2D(z_euler[:12,np.newaxis].reshape((3,4)),"x",predecimals=7,decimals=16)
        #     print()

        # self.uncontrolled_theta_max = np.max(self.zarr[10,:])
        # self.uncontrolled_theta_min = np.min(self.zarr[10,:])
                
        # begin simulation
        counter = 1
        self.x,self.t = self.initialize_sim(self.x0,\
            atol=1e-10,rtol=1e-10,
            nonlinear=self.use_nonlinear,quat=self.use_quaternions)
        for vee in range(self.n_steps-1):
            # for plots
            
            # run a step
            dxhat = np.matmul(self.Lin_Model.C,self.x) - self.Lin_Model.xhat_eq
            u = self.u_eq - np.matmul(self.Lin_Model.K,dxhat)
            self.x = self.sim_step(self.dt,self.x)

            # save state
            x_euler = self.x[ind2kp]*1.
            x_euler[9:12] = np.rad2deg(self._euler_angles(self.x))
            self.xarr[:,counter] = x_euler*1.
            self.uarr[:,counter] = u*1.
            self.tarr[counter] = self.t*1.
            counter = counter + 1
        
        print("finished simulating...")

        # calculate total velocity and aero angles
        Vxarr = (self.xarr[0]**2. + self.xarr[1]**2. + self.xarr[2]**2.)**0.5
        axarr = np.rad2deg(np.arctan2(self.xarr[2],self.xarr[0]))
        bxarr = np.rad2deg(np.arcsin(self.xarr[1]/Vxarr)) # experimental beta
        Mxarr = np.array([Vxarr[i]/self.stdatm(self.xarr[8,i])[5] \
            for i in range(len(self.tarr))])
        self.aerox = np.array([Vxarr,axarr,bxarr])
        Vzarr = (self.zarr[0]**2. + self.zarr[1]**2. + self.zarr[2]**2.)**0.5
        azarr = np.rad2deg(np.arctan2(self.zarr[2],self.zarr[0]))
        if self.run_unctrl:
            bzarr = np.rad2deg(np.arcsin(self.zarr[1]/Vzarr)) # experimental beta
        else:
            bzarr = azarr * 0.0
        self.aeroz = np.array([Vzarr,azarr,bzarr])

        # convert to degrees
        xicnv = [3,4,5] + [12,13,14]*(self.order >=1)
        uicnv = [0,1,2]
        self.xarr[xicnv,:] = np.rad2deg(self.xarr[xicnv,:])
        self.zarr[xicnv,:] = np.rad2deg(self.zarr[xicnv,:])
        self.uarr[uicnv,:] = np.rad2deg(self.uarr[uicnv,:])
        self.varr[uicnv,:] = np.rad2deg(self.varr[uicnv,:])

        return


    def integration_analysis_simulation(self,tf=60.0):

        # report
        print("running integration analysis simulation...")

        # initialize state and input
        v = self.u_trim[0:4] * 1.0
        x0_quat = self.x_trim * 1.0
        x0_euler = self.x_trim * 1.0
        x0_euler[9:12] = self._euler_angles(self.x_trim)

        # set timesteps / tols
        dts = [0.1,0.01,0.001,0.0001]
        tols = [1.e-6,1.e-8,1.e-10]
        # dts = [0.1,0.05,0.01]
        # tols = [1.e-4,5.e-5]

        # initialize arrays
        quats_fix_dt = np.zeros((len(dts),12))
        eulrs_fix_dt = np.zeros((len(dts),12))
        quats_var_dt = np.zeros((len(tols),12))
        eulrs_var_dt = np.zeros((len(tols),12))

        use_nonlinear = True # force nonlinear till I can fix linear

        for use_quats in [True,False]:
            if use_quats:
                print("  quaternions...");  x0 = x0_quat  * 1.0
            else:
                print("  euler angles..."); x0 = x0_euler * 1.0
            self._set_dynamics_function(True,use_quats)
            

            for use_fixed_dt in [True,False]:
                if use_fixed_dt:
                    self._set_integration_method("rk4"); tvals = dts
                    print("    rk4...")
                else:
                    self._set_integration_method("ode"); tvals = tols
                    print("    ode...")
                
                for i,val in enumerate(tvals):
                    if use_fixed_dt:
                        dt = dts[i]
                    else:
                        dt = dts[0]
                    num = int(tf / dt)
                    ts = np.linspace(0.0,self.tf,num=num)
                    if use_fixed_dt:
                        x,t0 = self.initialize_sim(x0,\
                            nonlinear=use_nonlinear,\
                            quat=use_quats)
                        print("      dt  = ",end="")
                    else:
                        x,t0 = self.initialize_sim(x0,atol=val,rtol=val,\
                            nonlinear=use_nonlinear,\
                            quat=use_quats)
                        print("      tol = ",end="")
                    print(val)

                    # integrate through simulation
                    for ti in ts:
                        x = self.sim_step(dt,x,v)
                    
                    # save as euler for final state, in degrees
                    xf = x[:12] * 1.0
                    if use_quats:
                        xf[9:12] = self._euler_angles(x)
                    xf[3: 6] = np.rad2deg(xf[3: 6])
                    xf[9:12] = np.rad2deg(xf[9:12])
                    
                    # save
                    if use_quats:
                        if use_fixed_dt:
                            quats_fix_dt[i] = xf*1.
                        else:
                            quats_var_dt[i] = xf*1.
                    else:
                        if use_fixed_dt:
                            eulrs_fix_dt[i] = xf*1.
                        else:
                            eulrs_var_dt[i] = xf*1.
        
        ## create plots
        predir = self.fldr_prfx + "_" + "plots/integration_analysis/" + \
            self.is_BIRE*"bire_" + (not self.is_BIRE)*"base_"
        show_plots = False
        transparent = False # True # 
        pltform = "pdf" # "pdf" # 
        savedict = dict(transparent=transparent,format=pltform,dpi=300.0)


        cs = ["#F5793A","#A95AA1","#85C0F9","#0F2080"]
        ms = ["o","^","s"]
        ls = ["-","--","-."]
        lbl = [
            r"$u$",r"$v$",r"$w$",
            r"$p$",r"$q$",r"$r$",
            r"$x_f$",r"$y_f$",r"$z_f$",
            r"$\phi$",r"$\theta$",r"$\psi$"
        ]
        
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
        plt.rcParams["xtick.major.width"] = plt.rcParams["ytick.major.width"] = 1.0
        plt.rcParams["xtick.minor.width"] = plt.rcParams["ytick.minor.width"] = 1.0
        plt.rcParams["xtick.major.size"] = plt.rcParams["ytick.major.size"] = 5.0
        plt.rcParams["xtick.minor.size"] = plt.rcParams["ytick.minor.size"] = 2.5
        # plt.rcParams["axes.labelpad"] = "10"
        # plt.rcParams["font.weight"] = "bold"
        plt.rcParams["mathtext.fontset"] = "dejavuserif"
        plt.rcParams['figure.dpi'] = 200.0

        # get percent differences
        ff = len(dts) - 1
        vf = len(tols) - 1
        pd_quats_fix = (quats_fix_dt[ff,:] - quats_fix_dt) / quats_fix_dt[ff,:]
        pd_eulrs_fix = (eulrs_fix_dt[ff,:] - eulrs_fix_dt) / eulrs_fix_dt[ff,:]
        pd_quats_var = (quats_var_dt[vf,:] - quats_var_dt) / quats_var_dt[vf,:]
        pd_eulrs_var = (eulrs_var_dt[vf,:] - eulrs_var_dt) / eulrs_var_dt[vf,:]
        pd_meths_fix = (quats_fix_dt - eulrs_fix_dt) / quats_fix_dt
        pd_meths_var = (quats_var_dt - eulrs_var_dt) / quats_var_dt
        
        fixdt_fig, fixdt_axs = plt.subplots(4,1,tight_layout=True,sharex=True)
        vardt_fig, vardt_axs = plt.subplots(4,1,tight_layout=True,sharex=True)
        fixmh_fig, fixmh_axs = plt.subplots(4,1,tight_layout=True,sharex=True)
        varmh_fig, varmh_axs = plt.subplots(4,1,tight_layout=True,sharex=True)

        sets = {"ls" : "none","mfc" : "none","mew" : 0.7,"ms" : 3.0}

        # fixed dt plot
        for i in range(12):
            c_i = i // 3
            l_i = i % 3
            fixdt_axs[c_i].plot(dts,pd_quats_fix[:,i],c="k",marker=ms[l_i],\
                label=lbl[i],**sets)
            fixdt_axs[c_i].plot(dts,pd_eulrs_fix[:,i],c="0.4",marker=ms[l_i],**sets)
        fixdt_fig.supylabel("Percent Difference")
        fixdt_fig.supxlabel("Timestep, s")
        fixdt_axs[0].invert_xaxis()
        for i in range(4):
            fixdt_axs[i].set_xscale("log")
            # fixdt_axs[i].set_yscale("symlog")
            fixdt_axs[i].legend()
        fixdt_fig.savefig(predir+"fix_dt_tmsteps."+pltform,**savedict)

        # variable dt plot
        for i in range(12):
            c_i = i // 3
            l_i = i % 3
            vardt_axs[c_i].plot(tols,pd_quats_var[:,i],c="k",marker=ms[l_i],\
                label=lbl[i],**sets)
            vardt_axs[c_i].plot(tols,pd_eulrs_var[:,i],c="0.4",marker=ms[l_i],**sets)
        vardt_fig.supylabel("Percent Difference")
        vardt_fig.supxlabel("Timestep, s")
        vardt_axs[0].invert_xaxis()
        for i in range(4):
            vardt_axs[i].set_xscale("log")
            # vardt_axs[i].set_yscale("symlog")
            vardt_axs[i].legend()
        vardt_fig.savefig(predir+"var_dt_tmsteps."+pltform,**savedict)

        # fixed dt method plot
        for i in range(12):
            c_i = i // 3
            l_i = i % 3
            fixmh_axs[c_i].plot(dts,pd_meths_fix[:,i],c="k",marker=ms[l_i],\
                label=lbl[i],**sets)
        fixmh_fig.supylabel("Percent Difference")
        fixmh_fig.supxlabel("Timestep, s")
        fixmh_axs[0].invert_xaxis()
        for i in range(4):
            fixmh_axs[i].set_xscale("log")
            # fixmh_axs[i].set_yscale("symlog")
            fixmh_axs[i].legend()
        fixmh_fig.savefig(predir+"fix_dt_methods."+pltform,**savedict)

        # variable dt method plot
        for i in range(12):
            c_i = i // 3
            l_i = i % 3
            varmh_axs[c_i].plot(tols,pd_meths_var[:,i],c="k",marker=ms[l_i],\
                label=lbl[i],**sets)
        varmh_fig.supylabel("Percent Difference")
        varmh_fig.supxlabel("Timestep, s")
        varmh_axs[0].invert_xaxis()
        for i in range(4):
            varmh_axs[i].set_xscale("log")
            # varmh_axs[i].set_yscale("symlog")
            varmh_axs[i].legend()
        varmh_fig.savefig(predir+"var_dt_methods."+pltform,**savedict)


        if show_plots:
            plt.show()
        else:
            plt.close()


    def check_partials(self,report=True,save_matrices=True,mrrr=[6,7,11],
            mrrc=None,run_freq=True,save_name_end="",
            include_stall_derivatives=False,
            drop_actrs=True):

        # report
        print("checking partial derivatives...")

        self._report_trim_solution(self.x_trim,self.u_trim,0)

        # determine dynamics jacobians
        use_quats_old = self.use_quaternions
        del self.use_quaternions
        self.use_quaternions = False
        x_euler = self.x_trim_euler*1.
        x_full = self.x_trim*1.
        u = self.u_trim*1.
        # randomize
        xrnd = x_euler * 1.0
        xrnd[1] = 10.0; xrnd[3] = xrnd[4] = xrnd[5] = 0.1
        xrnd[8] = 0.0; xrnd[9] = xrnd[10] = xrnd[11] = 0.15
        x = x_euler #+ xrnd * self.rng.normal(scale=1./3.,size=xrnd.size)
        x_full[:6] = x[:6] * 1.0
        x_full[9:13] = euler_2_quat(x[9:12])
        uxrnd = u * 1.0
        uxrnd[0] = uxrnd[2] = 0.15
        u = u #+ uxrnd * self.rng.normal(scale=1./3.,size=uxrnd.size)

        inputs = [0.0,x,True,True,u]
        r = [0,1,2,3,4,5,8,9,10]
        df_dx = (make_jacobian(self._nonlinear_euler_dynamics,inputs,1)[r])[:,r]
        df_du = make_jacobian(self._nonlinear_euler_dynamics,inputs,4)[r]
        Ln = lin(x,u,[0.,0.,0.],is_bire=self.is_BIRE,use_quaternion=False,
            compressible = self.is_compressible,
            use_Anderson = self.use_anderson,
            enforce_stall = self.has_stall,
            include_stall = include_stall_derivatives,
            controller_type = self.controller_type,
            controller_properties = self.control_dict,
            actuators_properties = self.actuators_dict,
            aero_model = self.aero_model,
            use_simple_thrust_model = not self.use_fitted_thrust,
            min_realization_removal_rows = mrrr,
            min_realization_removal_cols = mrrc,
            drop_actuators = drop_actrs,
            run_frequency_analysis = run_freq,
            freq_plots_name_end = save_name_end,
            folder_prefix = self.fldr_prfx)
        A = Ln.A_min; B = Ln.B_min
        # rep2D(  A  ,"  A  ",print_format="f",decimals=12)
        # rep2D(df_dx,"df/dx",print_format="f",decimals=12)

        rows = [0,1,2,3,4,5,6,7,8] # ,9,10,11
        cols = [0,1,2,3,4,5,6,7,8] # ,9,10,11
        rep2D((df_dx[rows])[:,cols],"df_dx",print_format="f",decimals=12)
        rep2D((A[rows])[:,cols],"  A  ",print_format="f",decimals=12)
        diff = df_dx - A
        rep2D((diff[rows])[:,cols],"diff", decimals=9)
        # diff = np.abs(A) * 0.0; diff[abs(A - df_dx) >=1e-4] = 505.0 #*#*#*#*#*#
        # rep2D((diff[rows])[:,cols],"diff",predecimals=3, decimals=0)#*#*#*#*#*#
        # df/dx 'fro' error = 0.00026519179043434256
        # df/du 'fro' error = 2.1589525357787776e-13
        print("df/dx eulr 'fro' error =", np.linalg.norm(A-df_dx,ord="fro"))
        print()
        
        rows = [0,1,2,3,4,5,6,7,8] # ,9,10,11
        cols = [0,1,2,3] # 
        rep2D((df_du[rows])[:,cols],"df_du",predecimals=8,decimals=12)
        rep2D((B[rows])[:,cols],"  B  ",predecimals=8,decimals=12)
        diff = df_du - B
        rep2D((diff[rows])[:,cols],"diff", decimals=9)
        print("df/du eulr 'fro' error =", np.linalg.norm(B-df_du,ord="fro"))
        print()

        # x = x_full
        # inputs = [0.0,x,True,True,u]
        # df_dx = ord2_jacobian(self._nonlinear_quaternion_dynamics,inputs,1)[:13,:13]
        # df_du = ord2_jacobian(self._nonlinear_quaternion_dynamics,inputs,4)[:13]
        # Ln = lin(x[:13],u,[0.,0.,0.],is_bire=self.is_BIRE,use_quaternion=True,\
        #     folder_prefix = self.fldr_prfx)
        # A = Ln.A; B = Ln.B

        # rows = [0,1,2,3,4,5,6,7,8] # ,9,10,11
        # cols = [0,1,2,3,4,5,6,7,8,9,10,11] # 
        # # rep2D((df_dx[rows])[:,cols],"df_dx",decimals=9)
        # # rep2D((A[rows])[:,cols],"  A  ",print_format="f",decimals=9)
        # diff = df_dx - A
        # # rep2D((diff[rows])[:,cols],"diff", decimals=9)
        # print("df/dx quat 'fro' error =", np.linalg.norm(A-df_dx,ord="fro"))

        # rows = [0,1,2,3,4,5,6,7,8,9,10,11] # 
        # cols = [0,1,2,3] # 
        # # rep2D(df_du,"df_du",predecimals=8,decimals=12)
        # # rep2D(B,"  B  ",predecimals=8,decimals=12)
        # diff = df_du - B
        # # rep2D((diff[rows])[:,cols],"diff", decimals=9)
        # print("df/du quat 'fro' error =", np.linalg.norm(B-df_du,ord="fro"))

        self.use_quaternions = use_quats_old
        return


    def _plot_results(self,**kwargs):

        # pull out kwargs
        show = kwargs.get("show", False)
        transparent = kwargs.get("transparent", True)
        format = kwargs.get("format", "pdf")
        deltas = kwargs.get("plot_deltas", False)
        perc_zoom = kwargs.get("percent_zoom", 1.0)
        is_zoomed = perc_zoom < 1.
        i_zoom = int(self.tarr.shape[0]*perc_zoom) + 1
        save_states = kwargs.get("save_states",False)
        plot_full = kwargs.get("plot_full",True)
        first_label = kwargs.get("first_set_label","controlled")
        second_label = kwargs.get("second_set_label","uncontrolled")
        plot_second_set = kwargs.get("plot_second_set",False)
        plot_input_limits_zoomed = kwargs.get("plot_input_limits_zoomed",True)
        plot_norm = kwargs.get("plot_norm",False) # True) # 
        plot_ul_bounds = kwargs.get("plot_upp_and_low",False)

        # determine where to save plots
        folder = kwargs.get("plotting_directory","")
        if deltas:
            # report
            print("\t delta" + " zoom"*is_zoomed + "...")
            folder += "delta"+"_zoom"*is_zoomed+"/"
        else:
            # report
            print("\t full" + " zoom"*is_zoomed + "...")
            folder += "full"+"_zoom"*is_zoomed+"/"
        # whether to prefix the BIRE or F16 name on the plots
        if self.is_BIRE:
            prename = "bire_"
        else:
            prename = "base_"
        predir = folder + prename

        # rename arrays for ease of use
        try:
            xhat_eq = np.array([self.x_tr_deg_itp(t) for t in self.tarr]).T
            uhat_eq = np.array([self.u_tr_deg_itp(t) for t in self.tarr]).T
            # xhat_t2 = self.x_trim2_euler_deg
            # uhat_t2 = self.u_trim2_deg
            # t_gs = self.t_gs
        except:
            print("Gain scheduling trim not used!!!")
            plot_norm = False
            xhat_eq = np.array([self.x_trim_euler_deg*1. for t in self.tarr]).T
            uhat_eq = np.array([self.u_trim_deg for t in self.tarr]).T
            # xhat_t2 = self.x_trim_euler_deg
            # uhat_t2 = self.u_trim_deg
            # t_gs = 0.
        V_eq = (xhat_eq[0,:]**2. + xhat_eq[1,:]**2. +xhat_eq[2,:]**2.)**0.5
        a_eq = np.rad2deg(np.arctan2(xhat_eq[2,:],xhat_eq[0,:]))
        b_eq = np.rad2deg(np.arcsin(xhat_eq[1,:]/V_eq)) # experimental beta
        M_eq = np.array([V_eq[ti]/self.stdatm(-xhat_eq[8,ti])[5] \
            for ti in range(len(self.tarr))])
        aerohat_eq = np.array([V_eq,M_eq,a_eq,b_eq])
        fill = dict(fc="k",alpha=0.1,ec="none")#color="k",alpha=0.2,ec=None)
        fil2 = dict(fc="none",ec="0.5",lw=0.75)#color="k",alpha=0.2,ec=None)
        if deltas:
            # state
            # xhat_t1 = self.x_trim_euler_deg
            # uhat_t1 = self.u_trim_deg
            # t_i = np.argwhere(self.tarr <= t_gs)[-1,0]
            # ps = np.ones(self.tarr.shape)
            # ps[:t_i+1] = np.linspace(0.,1.,t_i+1)
            # xhat_eq = np.array([(1. - pi)*xhat_t1 + pi*xhat_t2 for pi in ps]).T
            # uhat_eq = np.array([(1. - pi)*uhat_t1 + pi*uhat_t2 for pi in ps]).T
            # aero angles / total velocity
            # # input
            # uhat_eq = self.u_trim[0:4]*1.
            # uhat_eq[0:3] = np.rad2deg(uhat_eq[0:3])
            # if self.order != 0:
            #     xhat_eq[12:15] = uhat_eq[0:3]*1.
            
            tarr = self.tarr[:i_zoom+1]
            xarr = self.xarr[:,:i_zoom+1] - xhat_eq[:,:i_zoom+1]
            aerox = self.aerox[:,:i_zoom+1] - aerohat_eq[:,:i_zoom+1]
            uarr = self.uarr[:,:i_zoom+1] - uhat_eq[:,:i_zoom+1]
            if plot_second_set:
                zarr = self.zarr[:,:i_zoom+1] - xhat_eq[:,:i_zoom+1]
                aeroz = self.aeroz[:,:i_zoom+1] - aerohat_eq[:,:i_zoom+1]
                varr = self.varr[:,:i_zoom+1] - uhat_eq[:,:i_zoom+1]
            Del = r"$\Delta$"
            predir = predir + "del_" + "zm_"*is_zoomed

            # upper and lower plotting
            if plot_ul_bounds:
                xupp = self.xarr_upp[:,:i_zoom+1] - xhat_eq[:,:i_zoom+1]
                aupp = self.aerox_upp[:,:i_zoom+1] - aerohat_eq[:,:i_zoom+1]
                uupp = self.uarr_upp[:,:i_zoom+1] - uhat_eq[:,:i_zoom+1]
                xlow = self.xarr_low[:,:i_zoom+1] - xhat_eq[:,:i_zoom+1]
                alow = self.aerox_low[:,:i_zoom+1] - aerohat_eq[:,:i_zoom+1]
                ulow = self.uarr_low[:,:i_zoom+1] - uhat_eq[:,:i_zoom+1]
        else:
            tarr = self.tarr[:i_zoom+1]
            xarr = self.xarr[:,:i_zoom+1]
            aerox = self.aerox[:,:i_zoom+1]
            uarr = self.uarr[:,:i_zoom+1]
            if plot_second_set:
                zarr = self.zarr[:,:i_zoom+1]
                aeroz = self.aeroz[:,:i_zoom+1]
                varr = self.varr[:,:i_zoom+1]
            Del = r""
            predir = predir + "zm_"*is_zoomed
            # upper and lower plotting
            if plot_ul_bounds:
                xupp = self.xarr_upp[:,:i_zoom+1]
                aupp = self.aerox_upp[:,:i_zoom+1]
                uupp = self.uarr_upp[:,:i_zoom+1]
                xlow = self.xarr_low[:,:i_zoom+1]
                alow = self.aerox_low[:,:i_zoom+1]
                ulow = self.uarr_low[:,:i_zoom+1]
        
        # error and integral states
        if self.tracking:
            # get reference for all time
            ref = np.array([self._get_reference(ti) for ti in tarr]).T

            ref[self.xicnv] = np.rad2deg(ref[self.xicnv])
            
            # determine error for all proportional states
            xerr = xarr[self.xPi_eul,:] - ref[self.xPi_eul,:]
            
            # determine integral state for all integral states
            xigr = xarr[self.xIi_eul,:]

            # for plotting upper lower bounds with MC plots
            if plot_ul_bounds:
                eupp = xupp[self.xPi_eul,:] - ref[self.xPi_eul,:]
                elow = xlow[self.xPi_eul,:] - ref[self.xPi_eul,:]
                gupp = xupp[self.xIi_eul,:]
                glow = xlow[self.xIi_eul,:]

            if plot_second_set:
                # determine error for all proportional states
                zerr = zarr[self.xPi_eul,:] - ref[self.xPi_eul,:]
                # determine integral state for all integral states
                zigr = zarr[self.xIi_eul,:]
        
        # determine tick locations
        xticks = np.linspace(0.0,perc_zoom*self.tf,num=6).tolist()

        # change plot text parameters
        plt.rcParams["font.family"] = "Serif"
        plt.rcParams["font.size"] = 8.0
        plt.rcParams["axes.labelsize"] = 8.0
        plt.rcParams['axes.xmargin'] = 0
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
        
        subdict = {
            "figsize" : (3.25,3.5),
            "constrained_layout" : True,
            "sharex" : True
        }
        if deltas:
            n_vel_plots = 1
            ivw = 0
            n_pqr_plots = 1
            iq = ir = 0
            n_eul_plots = 1
            it = ip = 0
            # n_vel_plots = 2
            # ivw = 1
            # n_pqr_plots = 3
            # iq = 1
            # ir = 2
            # n_eul_plots = 3
            # it = 1
            # ip = 2
        else:
            n_vel_plots = 2
            ivw = 1
            n_pqr_plots = 3
            iq = 1
            ir = 2
            n_eul_plots = 3
            it = 1
            ip = 2
        vels_fig, vels_axs = plt.subplots(n_vel_plots,1,**subdict)
        aero_fig, aero_axs = plt.subplots(3 - 1*self.is_rc,1,**subdict)
        rate_fig, rate_axs = plt.subplots(n_pqr_plots,1,**subdict)
        posn_fig, posn_axs = plt.subplots(3,1,**subdict)
        ornt_fig, ornt_axs = plt.subplots(n_eul_plots,1,**subdict)
        if self.tracking:
            errs_fig, errs_axs = plt.subplots(1,1,**subdict)
            igrs_fig, igrs_axs = plt.subplots(1,1,**subdict)
        ctrl_fig, ctrl_axs = plt.subplots(4,1,**subdict)
        surf_fig, surf_axs = plt.subplots(1,1,**subdict)
        if self.is_BIRE:
            de = r"e^B"
            dr = r"B"
        else:
            de = r"e"
            dr = r"r"
        if deltas: 
            vels_axs = [vels_axs]
            rate_axs = [rate_axs]
            ornt_axs = [ornt_axs]

        ilbl = int(0.55*i_zoom)
        ilbl3 = int(0.65*i_zoom)
        ilbl2 = int(0.75*i_zoom)
        lbl_params = dict(ha="center",va="center",size=8.0)
        bbox_dict = dict(facecolor="w",linewidth=0,alpha=0.8,
            boxstyle="Square, pad=0.0")
        savedict = dict(transparent=transparent,format=format,dpi=300.0)
        zrs = 0.*tarr

        # states for error plots
        names = [r"$V_{x_b}$",r"$V_{y_b}$",r"$V_{z_b}$",
            r"$p$",r"$q$",r"$r$",
            r"$x_f$",r"$y_f$",r"$z_f$",
            r"$\phi$",r"$\theta$",r"$\psi$",
            r"$\delta_a$",r"$\delta_"+de+r"$",r"$\delta_"+dr+r"$",r"$\tau$",
            r"$\dot{\delta}_a$",r"$\dot{\delta}_"+de+r"$",
            r"$\dot{\delta}_"+dr+r"$",r"$\dot{\tau}$",
            r"$\ddot{\delta}_a$",r"$\ddot{\delta}_"+de+r"$",
            r"$\ddot{\delta}_"+dr+r"$",r"$\ddot{\tau}$"]

        # initialize parts of plots
        if plot_second_set:
            run_len = 2
        else:
            run_len = 1

        for i in range(run_len):

            # if controlled or uncontrolled
            if i == 0:
                c = "k"
                lbl = first_label
                state = xarr
                aero = aerox
                if self.tracking:
                    err = xerr
                    igr = xigr
            else:
                c = "0.35"
                lbl = second_label
                state = zarr
                aero = aeroz
                if self.tracking:
                    err = zerr
                    igr = zigr
            
            # # uvw plot
            if i==0:
                # if not deltas:
                #     if not self.is_BIRE:
                #         vels_axs[0].set_ylim(537.5,725)
                #         vels_axs[1].set_ylim(-125,125)
                #     else:
                #         vels_axs[0].set_ylim(555,705)
                #         vels_axs[1].set_ylim(-75,75)
                # else:
                #     vels_axs[0].set_ylim(-75,75)
                #     vels_axs[1].set_ylim(-75,75)
                if not deltas:
                    vels_axs[0].spines['bottom'].set_visible(False)
                    vels_axs[ivw].spines['top'].set_visible(False)
                    vels_axs[0].tick_params(which="both",bottom=False,labelbottom=False)
                    vels_axs[ivw].tick_params(which="both",top=False)
                vels_axs[0].grid(which="major",lw=0.6,ls="-",c="0.75")
                # vels_axs[0].grid(which="minor",lw=0.5,ls="dotted",c="0.5")
                vels_axs[ivw].grid(which="major",lw=0.6,ls="-",c="0.75")
                # vels_axs[ivw].grid(which="minor",lw=0.5,ls="dotted",c="0.5")
                vels_fig.supxlabel(r"Time, s")
                vels_fig.supylabel(r"Velocity, ft/s")
                # xticks
                vels_axs[ivw].set_xticks(ticks=xticks)
                if not deltas:
                    d = .015  # how big to make the diagonal lines in axes coordinates
                    # arguments to pass to plot, just so we don't keep repeating them
                    kwargs = dict(transform=vels_axs[0].transAxes, c='k', clip_on=False)
                    vels_axs[0].plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
                    vels_axs[0].plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
                    kwargs.update(transform=vels_axs[1].transAxes)  # switch to the bottom axes
                    vels_axs[ivw].plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
                    vels_axs[ivw].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
                # line var label
                # vels_axs[0].text(tarr[ilbl],state[0,ilbl],
                #     Del+r"$V_{x_b}$",**lbl_params)
                # vels_axs[ivw].text(tarr[ilbl3],state[1,ilbl3],
                #     Del+r"$V_{y_b}$",**lbl_params)
                # vels_axs[ivw].text(tarr[ilbl2],state[2,ilbl2],
                #     Del+r"$V_{z_b}$",**lbl_params)
                if plot_ul_bounds:
                    vels_axs[  0].fill_between(tarr, xupp[0], xlow[0],**fill)
                    vels_axs[ivw].fill_between(tarr, xupp[1], xlow[1],**fill)
                    vels_axs[ivw].fill_between(tarr, xupp[2], xlow[2],**fill)
                    vels_axs[  0].fill_between(tarr, xupp[0], xlow[0],ls="-",**fil2)
                    vels_axs[ivw].fill_between(tarr, xupp[1], xlow[1],ls="--",**fil2)
                    vels_axs[ivw].fill_between(tarr, xupp[2], xlow[2],ls="-.",**fil2)
            if not deltas and plot_norm:
                vels_axs[  0].plot(tarr,xhat_eq[0],c="k",ls="-" ,lw=0.5)
                vels_axs[ivw].plot(tarr,xhat_eq[1],c="k",ls="--",lw=0.5)
                vels_axs[ivw].plot(tarr,xhat_eq[2],c="k",ls="-.",lw=0.5)
            vels_axs[  0].plot(tarr, state[0],c=c,ls="-",
                label=Del+r"$V_{x_b}$")
            vels_axs[ivw].plot(tarr, state[1],c=c,ls="--",
                label=Del+r"$V_{y_b}$")
            vels_axs[ivw].plot(tarr, state[2],c=c,ls="-.",
                label=Del+r"$V_{z_b}$")
            # ################################
            # if not deltas:
            #     vels_axs[  0].plot(tarr, self.xs_sim[0],c="r",ls="-",
            #         label=Del+r"$V_{x_b}$")
            #     vels_axs[ivw].plot(tarr, self.xs_sim[1],c="r",ls="--",
            #         label=Del+r"$V_{y_b}$")
            #     vels_axs[ivw].plot(tarr, self.xs_sim[2],c="r",ls="-.",
            #         label=Del+r"$V_{z_b}$")
            # #################################
            if deltas:
                vels_axs[0].legend()
            else:
                vels_axs[0].text(tarr[ilbl],state[0,ilbl],
                    Del+r"$V_{x_b}$",bbox=bbox_dict,**lbl_params)
                vels_axs[1].legend()
            
            # # Vab plot
            if i==0:
                # aero_axs[0].spines['bottom'].set_visible(False)
                # aero_axs[1].spines['top'].set_visible(False)
                # aero_axs[0].tick_params(which="both",bottom=False,labelbottom=False)
                # aero_axs[1].tick_params(which="both",top=False)
                aero_axs[0].grid(which="major",lw=0.6,ls="-",c="0.75")
                # aero_axs[0].grid(which="minor",lw=0.5,ls="dotted",c="0.5")
                aero_axs[1].grid(which="major",lw=0.6,ls="-",c="0.75")
                # aero_axs[1].grid(which="minor",lw=0.5,ls="dotted",c="0.5")
                if not(self.is_rc):
                    aero_axs[2].grid(which="major",lw=0.6,ls="-",c="0.75")
                    # aero_axs[2].grid(which="minor",lw=0.5,ls="dotted",c="0.5")
                aero_fig.supxlabel(r"Time, s")
                aero_fig.supylabel(r"Velocity, ft/s" + r" / Mach"*(not self.is_rc) \
                    + r" / Angle, deg")
                # xticks
                aero_axs[2 - 1*self.is_rc].set_xticks(ticks=xticks)
                # d = .015  # how big to make the diagonal lines in axes coordinates
                # # arguments to pass to plot, just so we don't keep repeating them
                # kwargs = dict(transform=aero_axs[0].transAxes, c='k', clip_on=False)
                # aero_axs[0].plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
                # aero_axs[0].plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
                # kwargs.update(transform=aero_axs[1].transAxes)  # switch to the bottom axes
                # aero_axs[1].plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
                # aero_axs[1].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
                # line var label
                aero_axs[0].text(tarr[ilbl],aero[0,ilbl],
                    Del+r"$V$",bbox=bbox_dict,**lbl_params)
                if not(self.is_rc):
                    aero_axs[1].text(tarr[ilbl],aero[1,ilbl],
                        Del+r"$M$",bbox=bbox_dict,**lbl_params)
                # aero_axs[1].text(tarr[ilbl],aero[1,ilbl],
                #     Del+r"$\alpha$",**lbl_params)
                # aero_axs[1].text(tarr[ilbl2],aero[2,ilbl2],
                #     Del+r"$\beta$",**lbl_params)
                if plot_ul_bounds:
                    aero_axs[  0].fill_between(tarr, aupp[0], alow[0],**fill)
                    aero_axs[  0].fill_between(tarr, aupp[0], alow[0],ls="-",**fil2)
                    if self.is_rc:
                        aero_axs[1].fill_between(tarr, aupp[2], alow[2],**fill)
                        aero_axs[1].fill_between(tarr, aupp[2], alow[2],ls="-",**fil2)
                        aero_axs[1].fill_between(tarr, aupp[3], alow[3],**fill)
                        aero_axs[1].fill_between(tarr, aupp[3], alow[3],ls="--",**fil2)
                    else:
                        aero_axs[1].fill_between(tarr, aupp[1], alow[1],**fill)
                        aero_axs[1].fill_between(tarr, aupp[1], alow[1],ls="-",**fil2)
                        aero_axs[2].fill_between(tarr, aupp[2], alow[2],**fill)
                        aero_axs[2].fill_between(tarr, aupp[2], alow[2],ls="-",**fil2)
                        aero_axs[2].fill_between(tarr, aupp[3], alow[3],**fill)
                        aero_axs[2].fill_between(tarr, aupp[3], alow[3],ls="--",**fil2)
            if not deltas and plot_norm:
                aero_axs[0].plot(tarr, aerohat_eq[0],c="k",ls="-",lw=0.5)
                if self.is_rc:
                    aero_axs[1].plot(tarr, aerohat_eq[2],c="k",ls="-",lw=0.5)
                    aero_axs[1].plot(tarr, aerohat_eq[3],c="k",ls="--",lw=0.5)
                else:
                    aero_axs[1].plot(tarr, aerohat_eq[1],c="k",ls="-",lw=0.5)
                    aero_axs[2].plot(tarr, aerohat_eq[2],c="k",ls="-",lw=0.5)
                    aero_axs[2].plot(tarr, aerohat_eq[3],c="k",ls="--",lw=0.5)
            aero_axs[0].plot(tarr, aero[0],c=c,ls="-")
            if self.is_rc:
                aero_axs[1].plot(tarr, aero[2],c=c,ls="-",label=Del+r"$\alpha$")
                aero_axs[1].plot(tarr, aero[3],c=c,ls="--",label=Del+r"$\beta$")
                aero_axs[1].legend()
            else:
                aero_axs[1].plot(tarr, aero[1],c=c,ls="-")
                aero_axs[2].plot(tarr, aero[2],c=c,ls="-",label=Del+r"$\alpha$")
                aero_axs[2].plot(tarr, aero[3],c=c,ls="--",label=Del+r"$\beta$")
                aero_axs[2].legend()

            # # pqr plots
            if i==0:
                # if not deltas:
                #     rate_axs[0].set_ylim(-22.5,22.5)
                #     if not self.is_BIRE:
                #         rate_axs[1].set_ylim(-22.5,22.5)
                #         rate_axs[2].set_ylim(-22.5,22.5)
                #     else:
                #         rate_axs[1].set_ylim(-5.0,5.0)
                #         rate_axs[2].set_ylim(-5.0,5.0)
                # else:
                #     if not self.is_BIRE:
                #         rate_axs[0].set_ylim(-37.5,37.5)
                #         rate_axs[1].set_ylim(-37.5,37.5)
                #         rate_axs[2].set_ylim(-37.5,37.5)
                #     else:
                #         rate_axs[0].set_ylim(-25.0,25.0)
                #         rate_axs[1].set_ylim(-5.0,5.0)
                #         rate_axs[2].set_ylim(-5.0,5.0)
                # grid, axis labels, legends
                rate_axs[ 0].grid(which="major",lw=0.6,ls="-",c="0.75")
                # rate_axs[ 0].grid(which="minor",lw=0.5,ls="dotted",c="0.5")
                rate_axs[iq].grid(which="major",lw=0.6,ls="-",c="0.75")
                # rate_axs[iq].grid(which="minor",lw=0.5,ls="dotted",c="0.5")
                rate_axs[ir].grid(which="major",lw=0.6,ls="-",c="0.75")
                # rate_axs[ir].grid(which="minor",lw=0.5,ls="dotted",c="0.5")
                rate_fig.supxlabel(r"Time, s")
                rate_fig.supylabel(r"Rotation Rates, deg/s")
                # xticks
                rate_axs[ir].set_xticks(ticks=xticks)
                # line var labels
                if not deltas:
                    rate_axs[ 0].text(tarr[ilbl ],state[3,ilbl ],
                        Del+r"$p$",bbox=bbox_dict,**lbl_params)
                    rate_axs[iq].text(tarr[ilbl2],state[4,ilbl2],
                        Del+r"$q$",bbox=bbox_dict,**lbl_params)
                    rate_axs[ir].text(tarr[ilbl3],state[5,ilbl3],
                        Del+r"$r$",bbox=bbox_dict,**lbl_params)
                if plot_ul_bounds:
                    rate_axs[ 0].fill_between(tarr, xupp[3], xlow[3],**fill)
                    rate_axs[ 0].fill_between(tarr, xupp[3], xlow[3],ls="-",**fil2)
                    rate_axs[iq].fill_between(tarr, xupp[4], xlow[4],**fill)
                    rate_axs[iq].fill_between(tarr, xupp[4], xlow[4],
                        ls="-"+("" if iq else "-"),**fil2)
                    rate_axs[ir].fill_between(tarr, xupp[5], xlow[5],**fill)
                    rate_axs[ir].fill_between(tarr, xupp[5], xlow[5],
                        ls="-"+("" if ir else "."),**fil2)
            if not deltas and plot_norm:
                rate_axs[ 0].plot(tarr, xhat_eq[3],c="k",
                    ls="-",lw=0.5)
                rate_axs[iq].plot(tarr, xhat_eq[4],c="k",
                    ls="-"+("" if iq else "-"),lw=0.5)
                rate_axs[ir].plot(tarr, xhat_eq[5],c="k",
                    ls="-"+("" if ir else "."),lw=0.5)
            rate_axs[ 0].plot(tarr, state[3],c=c,ls="-",
                label=Del+r"$p$")
            rate_axs[iq].plot(tarr, state[4],c=c,ls="-"+("" if iq else "-"),
                label=Del+r"$q$")
            rate_axs[ir].plot(tarr, state[5],c=c,ls="-"+("" if ir else "."),
                label=Del+r"$r$")
            # ################################
            # if not deltas:
            #     rate_axs[ 0].plot(tarr, self.xs_sim[3],c="r",ls="-",
            #         label=Del+r"$p$")
            #     rate_axs[iq].plot(tarr, self.xs_sim[4],c="r",ls="-"+("" if iq else "-"),
            #         label=Del+r"$q$")
            #     rate_axs[ir].plot(tarr, self.xs_sim[5],c="r",ls="-"+("" if ir else "."),
            #         label=Del+r"$r$")
            # ################################
            if deltas:
                rate_axs[0].legend()
            
            # # error plots
            lsy = ["-","--","-.",":"]
            sfl = dict(color="k",alpha=0.1)
            if self.tracking:
                # axis labels, legends
                errs_fig.supxlabel(r"Time, s")
                errs_fig.supylabel(r"Error")
                # xticks
                errs_axs.set_xticks(ticks=xticks)
                # grid, axis labels, legends
                errs_axs.grid(which="major",lw=0.6,ls="-",c="0.75")
                for j in range(len(self.xPi)):
                    # determine linestyle
                    lsj = lsy[j % len(lsy)]
                    # if not deltas:
                    #     errs_axs.text(tarr[ilbl ],err[j,ilbl ],
                    #         Del+r"$e_{"+names[self.xPi_eul[j]][1:-1]+r"}$",\
                    #         bbox=bbox_dict,**lbl_params)
                    errs_axs.plot(tarr, err[j],c=c,ls=lsj,
                        label=Del+r"$e_{"+names[self.xPi_eul[j]][1:-1]+r"}$")
                    if plot_ul_bounds:
                        errs_axs.fill_between(tarr,eupp[j],elow[j],**fill)
                        errs_axs.fill_between(tarr,eupp[j],elow[j],ls=lsj,**fil2)
                errs_axs.legend()
            
            # # integrator plots
            if self.tracking:
                # axis labels, legends
                igrs_fig.supxlabel(r"Time, s")
                igrs_fig.supylabel(r"Integrator State")
                # xticks
                igrs_axs.set_xticks(ticks=xticks)
                # grid, axis labels, legends
                igrs_axs.grid(which="major",lw=0.6,ls="-",c="0.75")
                for j in range(len(self.xIi)):
                    # determine linestyle
                    lsj = lsy[j % len(lsy)]
                    # line var labels
                    # if not deltas:
                    #     igrs_axs.text(tarr[ilbl ],igr[j,ilbl ],
                    #         Del+r"$\int e_{"+names[self.xPi_eul[j]][1:-1]\
                    #         +r"}\, dt$",bbox=bbox_dict,**lbl_params)
                    igrs_axs.plot(tarr, igr[j],c=c,ls=lsj,
                        label=Del+r"$\int e_{"+names[self.xPi_eul[j]][1:-1]\
                            +r"}\, dt$")
                    if plot_ul_bounds:
                        igrs_axs.fill_between(tarr,gupp[j],glow[j],**fill)
                        igrs_axs.fill_between(tarr,gupp[j],glow[j],ls=lsj,**fil2)
                igrs_axs.legend()

            # # xyz plots
            if i==0:
                # # y limits
                # posn_axs[0].set_ylim(-7500.0,17500.0)
                # posn_axs[1].set_ylim(-150.0,150.0)
                # posn_axs[2].set_ylim(-15150.0,-14850.0)
                # grid, axis labels, legends
                posn_axs[0].grid(which="major",lw=0.6,ls="-",c="0.75")
                # posn_axs[0].grid(which="minor",lw=0.5,ls="dotted",c="0.5")
                posn_axs[1].grid(which="major",lw=0.6,ls="-",c="0.75")
                # posn_axs[1].grid(which="minor",lw=0.5,ls="dotted",c="0.5")
                posn_axs[2].grid(which="major",lw=0.6,ls="-",c="0.75")
                # posn_axs[2].grid(which="minor",lw=0.5,ls="dotted",c="0.5")
                posn_fig.supxlabel(r"Time, s")
                posn_fig.supylabel(r"Position, ft")
                # xticks
                posn_axs[2].set_xticks(ticks=xticks)
                # line var labels
                posn_axs[0].text(tarr[ilbl],state[6,ilbl],
                    Del+r"$x_f$",bbox=bbox_dict,**lbl_params)
                posn_axs[1].text(tarr[ilbl],state[7,ilbl],
                    Del+r"$y_f$",bbox=bbox_dict,**lbl_params)
                posn_axs[2].text(tarr[ilbl],state[8,ilbl],
                    Del+r"$z_f$",bbox=bbox_dict,**lbl_params)
                output_coords = posn_axs[2].transLimits.inverted().transform((
                    tarr[ilbl],state[8,ilbl]))
                if plot_ul_bounds:
                    posn_axs[0].fill_between(tarr, xupp[6], xlow[6],**fill)
                    posn_axs[0].fill_between(tarr, xupp[6], xlow[6],ls="-",**fil2)
                    posn_axs[1].fill_between(tarr, xupp[7], xlow[7],**fill)
                    posn_axs[1].fill_between(tarr, xupp[7], xlow[7],ls="-",**fil2)
                    posn_axs[2].fill_between(tarr, xupp[8], xlow[8],**fill)
                    posn_axs[2].fill_between(tarr, xupp[8], xlow[8],ls="-",**fil2)
            if not deltas and plot_norm:
                # posn_axs[0].plot(tarr, xhat_eq[6],c="k",ls="-",lw=0.5)
                posn_axs[1].plot(tarr, xhat_eq[7],c="k",ls="-",lw=0.5)
                posn_axs[2].plot(tarr, xhat_eq[8],c="k",ls="-",lw=0.5)
            posn_axs[0].plot(tarr, state[6],c=c,ls="-",label=lbl)
            posn_axs[1].plot(tarr, state[7],c=c,ls="-")
            posn_axs[2].plot(tarr, state[8],c=c,ls="-")
            # ################################
            # if not deltas:
            #     posn_axs[0].plot(tarr, self.xs_sim[6],c="r",ls="-",label=lbl)
            #     posn_axs[1].plot(tarr, self.xs_sim[7],c="r",ls="-")
            #     posn_axs[2].plot(tarr, self.xs_sim[8],c="r",ls="-")
            # ################################
            if i==1:
                posn_axs[0].legend()

            # # euler angles plots
            if i==0:
                # if not self.is_BIRE:
                #     ornt_axs[0].set_ylim(-12.5,12.5)
                #     ornt_axs[1].set_ylim(-12.5,12.5)
                #     ornt_axs[2].set_ylim(-12.5,12.5)
                # else:
                #     if not deltas:
                #         ornt_axs[0].set_ylim(-15.0,15.0)
                #         ornt_axs[1].set_ylim(-3.0,7.0)
                #         ornt_axs[2].set_ylim(-7.0,3.0)
                #     else:
                #         ornt_axs[0].set_ylim(-17.5,17.5)
                #         ornt_axs[1].set_ylim(-3.5,3.5)
                #         ornt_axs[2].set_ylim(-3.5,3.5)
                # grid, axis labels, legends
                ornt_axs[ 0].grid(which="major",lw=0.6,ls="-",c="0.75")
                # ornt_axs[ 0].grid(which="minor",lw=0.5,ls="dotted",c="0.5")
                ornt_axs[it].grid(which="major",lw=0.6,ls="-",c="0.75")
                # ornt_axs[it].grid(which="minor",lw=0.5,ls="dotted",c="0.5")
                ornt_axs[ip].grid(which="major",lw=0.6,ls="-",c="0.75")
                # ornt_axs[ip].grid(which="minor",lw=0.5,ls="dotted",c="0.5")
                ornt_fig.supxlabel(r"Time, s")
                ornt_fig.supylabel(r"Orientation, deg")
                # xticks
                ornt_axs[ip].set_xticks(ticks=xticks)
                # line var labels
                if not deltas:
                    ornt_axs[ 0].text(tarr[ilbl ],state[ 9,ilbl ],
                        Del+r"$\phi$",bbox=bbox_dict,**lbl_params)
                    ornt_axs[it].text(tarr[ilbl2],state[10,ilbl2],
                        Del+r"$\theta$",bbox=bbox_dict,**lbl_params)
                    ornt_axs[ip].text(tarr[ilbl3],state[11,ilbl3],
                        Del+r"$\psi$",bbox=bbox_dict,**lbl_params)
                if plot_ul_bounds:
                    ornt_axs[ 0].fill_between(tarr, xupp[ 9], xlow[ 9],**fill)
                    ornt_axs[ 0].fill_between(tarr, xupp[ 9], xlow[ 9],
                        ls="-",**fil2)
                    ornt_axs[it].fill_between(tarr, xupp[10], xlow[10],**fill)
                    ornt_axs[it].fill_between(tarr, xupp[10], xlow[10],
                        ls="-"+("" if it else "-"),**fil2)
                    ornt_axs[ip].fill_between(tarr, xupp[11], xlow[11],**fill)
                    ornt_axs[ip].fill_between(tarr, xupp[11], xlow[11],
                        ls="-"+("" if ip else "."),**fil2)
            if not deltas and plot_norm:
                ornt_axs[ 0].plot(tarr, xhat_eq[ 9],c="k",
                    ls="-",lw=0.5)
                ornt_axs[it].plot(tarr, xhat_eq[10],c="k",
                    ls="-"+("" if it else "-"),lw=0.5)
                ornt_axs[ip].plot(tarr, xhat_eq[11],c="k",
                    ls="-"+("" if ip else "."),lw=0.5)
            ornt_axs[ 0].plot(tarr, state[9],c=c,ls="-",
                label=Del+r"$\phi$")
            ornt_axs[it].plot(tarr, state[10],c=c,ls="-"+("" if it else "-"),
                label=Del+r"$\theta$")
            ornt_axs[ip].plot(tarr, state[11],c=c,ls="-"+("" if ip else "."),
                label=Del+r"$\psi$")
            # ################################
            # if not deltas:
            #     ornt_axs[ 0].plot(tarr, self.xs_sim[9],c="r",ls="-",
            #         label=Del+r"$\phi$")
            #     ornt_axs[it].plot(tarr, self.xs_sim[10],c="r",ls="-"+("" if it else "-"),
            #         label=Del+r"$\theta$")
            #     ornt_axs[ip].plot(tarr, self.xs_sim[11],c="r",ls="-"+("" if ip else "."),
            #         label=Del+r"$\psi$")
            # ################################
            if deltas:
                ornt_axs[0].legend()

        
        svuc = uarr*0.0
        for i in range(run_len + 1):
            # if controlled or uncontrolled
            if i == 0:
                c = "k"
                lbl = "commanded"
                ctrl = uarr
            elif i == 1:
                c = "0.5"
                lbl = "response"
                if self.order == 0:
                    ctrl = uarr
                else:
                    ctrl = xarr[12:16]
                ctrl_i1 = ctrl*1.
                svuc = ctrl*1.
                if plot_ul_bounds:
                    if self.order == 0:
                        cupp = uupp
                        clow = ulow
                    else:
                        cupp = xupp[12:16]
                        clow = xlow[12:16]
            else:
                c = "0.6"
                lbl = second_label + " commanded"
                ctrl = varr

            # # controls plots
            if i==0:
                # grid, axis labels, legends
                ctrl_axs[0].grid(which="major",lw=0.6,ls="-",c="0.75")
                surf_axs   .grid(which="major",lw=0.6,ls="-",c="0.75")
                # ctrl_axs[0].grid(which="minor",lw=0.5,ls="dotted",c="0.5")
                ctrl_axs[1].grid(which="major",lw=0.6,ls="-",c="0.75")
                # ctrl_axs[1].grid(which="minor",lw=0.5,ls="dotted",c="0.5")
                ctrl_axs[2].grid(which="major",lw=0.6,ls="-",c="0.75")
                # ctrl_axs[2].grid(which="minor",lw=0.5,ls="dotted",c="0.5")
                ctrl_axs[3].grid(which="major",lw=0.6,ls="-",c="0.75")
                # ctrl_axs[3].grid(which="minor",lw=0.5,ls="dotted",c="0.5")
                ctrl_fig.supxlabel(r"Time, s")
                ctrl_fig.supylabel(r"Input, deg or per-unit")
                surf_fig.supxlabel(r"Time, s")
                surf_fig.supylabel(r"Input, deg")
                # xticks
                ctrl_axs[3].set_xticks(ticks=xticks)
                surf_axs   .set_xticks(ticks=xticks)
            if i==1:
                # line var labels
                ctrl_axs[0].text(tarr[ilbl],ctrl[0,ilbl],
                    Del+r"$\delta_a$",bbox=bbox_dict,**lbl_params)
                ctrl_axs[1].text(tarr[ilbl],ctrl[1,ilbl],
                    Del+r"$\delta_"+de+r"$",bbox=bbox_dict,**lbl_params)
                ctrl_axs[2].text(tarr[ilbl],ctrl[2,ilbl],
                    Del+r"$\delta_"+dr+r"$",bbox=bbox_dict,**lbl_params)
                ctrl_axs[3].text(tarr[ilbl],ctrl[3,ilbl],
                    Del+r"$\tau$",bbox=bbox_dict,**lbl_params)
                if plot_ul_bounds:
                    ctrl_axs[0].fill_between(tarr, cupp[0], clow[0],**fill)
                    ctrl_axs[0].fill_between(tarr, cupp[0], clow[0],ls="-",**fil2)
                    ctrl_axs[1].fill_between(tarr, cupp[1], clow[1],**fill)
                    ctrl_axs[1].fill_between(tarr, cupp[1], clow[1],ls="-",**fil2)
                    ctrl_axs[2].fill_between(tarr, cupp[2], clow[2],**fill)
                    ctrl_axs[2].fill_between(tarr, cupp[2], clow[2],ls="-",**fil2)
                    ctrl_axs[3].fill_between(tarr, cupp[3], clow[3],**fill)
                    ctrl_axs[3].fill_between(tarr, cupp[3], clow[3],ls="-",**fil2)
                    #
                    sfl = dict(color="k",alpha=0.1)
                    surf_axs.fill_between(tarr, cupp[0], clow[0],**fill)
                    surf_axs.fill_between(tarr, cupp[0], clow[0],ls="-",**fil2)
                    surf_axs.fill_between(tarr, cupp[1], clow[1],**fill)
                    surf_axs.fill_between(tarr, cupp[1], clow[1],ls="--",**fil2)
                    surf_axs.fill_between(tarr, cupp[2], clow[2],**fill)
                    surf_axs.fill_between(tarr, cupp[2], clow[2],ls="-.",**fil2)
            if not deltas and plot_norm:
                ctrl_axs[0].plot(tarr, uhat_eq[0],c="k",ls="-",lw=0.5)
                ctrl_axs[1].plot(tarr, uhat_eq[1],c="k",ls="-",lw=0.5)
                ctrl_axs[2].plot(tarr, uhat_eq[2],c="k",ls="-",lw=0.5)
                ctrl_axs[3].plot(tarr, uhat_eq[3],c="k",ls="-",lw=0.5)
                surf_axs   .plot(tarr, uhat_eq[0],c="k",ls= "-",lw=0.5)
                surf_axs   .plot(tarr, uhat_eq[1],c="k",ls="--",lw=0.5)
                surf_axs   .plot(tarr, uhat_eq[2],c="k",ls="-.",lw=0.5)
            ctrl_axs[0].plot(tarr, ctrl[0],c=c,ls="-",label=lbl)
            ctrl_axs[1].plot(tarr, ctrl[1],c=c,ls="-")
            ctrl_axs[2].plot(tarr, ctrl[2],c=c,ls="-")
            ctrl_axs[3].plot(tarr, ctrl[3],c=c,ls="-")
            surf_axs   .plot(tarr, ctrl[0],c=c,ls= "-",
                label=(i==0)*r"$\delta_a$"       + "")
            surf_axs   .plot(tarr, ctrl[1],c=c,ls="--",
                label=(i==0)*r"$\delta_"+de+r"$" + "")
            surf_axs   .plot(tarr, ctrl[2],c=c,ls="-.",
                label=(i==0)*r"$\delta_"+dr+r"$" + "")
            if i==2:
                ctrl_axs[0].legend()
            if i==0:
                surf_axs   .legend()
        ## limits ##
        if deltas:
            min_da  = np.rad2deg( self.min_da - self.u_trim[0])
            max_da  = np.rad2deg( self.max_da - self.u_trim[0])
            min_de_opt = np.rad2deg( self.min_de - self.u_trim[1])
            max_de_opt = np.rad2deg( self.max_de - self.u_trim[1])
            min_dr  = np.rad2deg( self.min_dr - self.u_trim[2])
            max_dr  = np.rad2deg( self.max_dr - self.u_trim[2])
            min_tau = self.min_tau - self.u_trim[3]
            max_tau = self.max_tau - self.u_trim[3]
        else:
            min_da  = np.rad2deg( self.min_da)
            max_da  = np.rad2deg( self.max_da)
            min_de_opt = np.rad2deg( self.min_de)
            max_de_opt = np.rad2deg( self.max_de)
            min_dr  = np.rad2deg( self.min_dr)
            max_dr  = np.rad2deg( self.max_dr)
            min_tau = self.min_tau
            max_tau = self.max_tau
        # # set elevator limit as a function of aileron value
        # da_to_de = np.abs(ctrl[0])*0.25
        max_de = max_de_opt # - da_to_de
        min_de = min_de_opt # + da_to_de
        if is_zoomed and not(plot_input_limits_zoomed):
            # lylim,uylim = ctrl_axs[3].get_ylim()
            # ctrl_axs[3].set_ylim((min(lylim,-0.05),max(uylim,0.05)))
            ctrl_axs[2].set_ylim((min_dr*0.1,max_dr*0.1))
            ctrl_axs[3].set_ylim((min_tau*0.1,max_tau*0.1))
        else:
            ctrl_axs[0].plot(tarr, zrs + min_da ,c="0.25",ls="--")
            ctrl_axs[0].plot(tarr, zrs + max_da ,c="0.25",ls="--")
            ctrl_axs[1].plot(tarr, zrs + min_de ,c="0.25",ls="--")
            ctrl_axs[1].plot(tarr, zrs + max_de ,c="0.25",ls="--")
            ctrl_axs[2].plot(tarr, zrs + min_dr ,c="0.25",ls="--")
            ctrl_axs[2].plot(tarr, zrs + max_dr ,c="0.25",ls="--")
            ctrl_axs[3].plot(tarr, zrs + min_tau,c="0.25",ls="--")
            ctrl_axs[3].plot(tarr, zrs + max_tau,c="0.25",ls="--")
            ctrl_axs[0].set_ylim((min_da-5.,max_da+5.))
            ctrl_axs[1].set_ylim((min_de_opt-5.,max_de_opt+5.))
            ctrl_axs[2].set_ylim((min_dr-5.,max_dr+5.))
            ctrl_axs[3].set_ylim((min_tau-0.05,max_tau+0.05))
        # surf maxes
        # determine maxes and mins
        if plot_ul_bounds:
            max_delta =  ( (np.max(cupp[:3])+0.1) // 5. + 1.)*5. #,axis=0)
            min_delta =  ( (np.min(clow[:3])+0.1) // 5. - 1)*5. #,axis=0)
        else:
            max_delta =  ( (np.max(ctrl_i1[:3])+0.1) // 5. + 1.)*5. #,axis=0)
            min_delta =  ( (np.min(ctrl_i1[:3])+0.1) // 5. - 1)*5. #,axis=0)
        max_max = np.max([max_da + 5.,max_de + 5.,max_dr + 5.])
        min_max = np.min([min_da - 5.,min_de - 5.,min_dr - 5.])
        max_lim = min(max_delta,max_max)
        min_lim = max(min_delta,min_max)
        surf_axs.set_ylim((min_lim,max_lim))

        # finite diffs (1)st and (2)nd order, (c)entered, (f)orward, (b)ackward
        # (l)ower and (h)igher accuracy
        # centered
        F1cl = lambda a,t : (a[2:]-a[:-2])/(t[2:]-t[:-2])
        F1ch = lambda a,t : (-a[4:]+8.*(a[3:-1]-a[1:-3])+a[:-4])\
            /3./(t[4:]-t[:-4])
        F2cl = lambda a,t : (a[2:]-2.*a[1:-1]+a[:-2])/((t[2:]-t[:-2])/2.)**2.
        F2ch = lambda a,t : (-a[4:]+16.*(a[3:-1]+a[1:-3])-30.*a[2:-2]-a[:-4])\
            /12./((t[4:]-t[:-4])/4.)**2.
        # forward
        F1fl = lambda a,t : (a[1:]-a[:-1])/(t[1:]-t[:-1])
        F1fh = lambda a,t : (-a[2:]+4.*a[1:-1]-3.*a[:-2])/2./(t[1:-1]-t[:-2])
        F2fl = lambda a,t : (a[2:]-2.*a[1:-1]+a[:-2])/(t[1:-1]-t[:-2])**2.
        F2fh = lambda a,t : (-a[3:]+4.*a[2:-1]-5.*a[1:-2]+2.*a[:-3])\
            /(t[1:-2]-t[:-3])**2.
        # backward
        F1bl = lambda a,t : (a[1:]-a[:-1])/(t[1:]-t[:-1])
        F1bh = lambda a,t : (3*a[2:]-4.*a[1:-1]+a[:-2])/2./(t[2:]-t[1:-1])
        F2bl = lambda a,t : (a[2:]-2.*a[1:-1]+a[:-2])/(t[2:]-t[1:-1])**2.
        F2bh = lambda a,t : (2.*a[3:]-5.*a[2:-1]+4.*a[1:-2]-a[:-3])\
            /(t[3:]-t[1:-2])**2.
        #
        _dxl = lambda a,t : np.concatenate((F1fl(a[:2],t[:2]),F1cl(a,t),\
            F1bl(a[-2:],t[-2:])),axis=0)
        _dxh = lambda a,t : np.concatenate((F1fh(a[:4],t[:4]),F1ch(a,t),\
            F1bh(a[-4:],t[-4:])),axis=0)
        ddxl = lambda a,t : np.concatenate((F2fl(a[:3],t[:3]),F2cl(a,t),\
            F2fl(a[-3:],t[-3:])),axis=0)
        ddxh = lambda a,t : np.concatenate((F2fh(a[:5],t[:5]),F2ch(a,t),\
            F2bh(a[-5:],t[-5:])),axis=0)
        # de2 = lambda a,t : F1ca(a,t)

        # # ex
        # plt.close("all")
        # tf = 100.0
        # num = 1001
        # ts   = np.linspace(0.0,tf,num=num)
        # __f = lambda x :  np.sin(x) + 1.0e-4*   x**3. + 1.0e-4*   x**2. + 153.0
        # _df = lambda x :  np.cos(x) + 1.0e-4*3.*x**2. + 1.0e-4*2.*x
        # ddf = lambda x : -np.sin(x) + 1.0e-4*6.*x     + 1.0e-4*2.
        # xs   = __f(ts)
        # dxa  = _df(ts)
        # ddxa = ddf(ts)
        # dxl  = _dxl(xs,ts)
        # ddxl = ddxl(xs,ts)
        # dxh  = _dxh(xs,ts)
        # ddxh = ddxh(xs,ts)

        # h = np.average(ts[1:]-ts[:-1])+ts*0.0

        # fig, axs = plt.subplots(2,1,figsize=(6.0,3.0),
        #     constrained_layout=True,sharex=True)
        # axs[0].plot(ts,dxa,"k") # dxa - 
        # axs[0].plot(ts,dxl,"m") # dxa - 
        # axs[0].plot(ts,dxh,"y") # dxa - 
        # axs[1].plot(ts,ddxa,"k") # ddxa - 
        # axs[1].plot(ts,ddxl,"m") # ddxa - 
        # axs[1].plot(ts,ddxh,"y") # ddxa - 
        # plt.show()
        # quit()
        
        udot_fig, udot_axs = plt.subplots(4,1,**subdict)

        c = "k"
        if self.order == 2:
            udot = xarr[16:20]
            udot_lbl = "input rate"
            if plot_ul_bounds:
                udpp = xupp[16:20]
                udow = xlow[16:20]
        else:
            if self.order == 0:
                inputs = uarr
            else:
                inputs = xarr[12:16]
            if plot_ul_bounds:
                if self.order == 0:
                    iupp = uupp
                    ilow = ulow
                else:
                    iupp = xupp[12:16]
                    ilow = xlow[12:16]
            # udot =  inputs*0.
            # udot[:,1:-1] = ( inputs[:,2:] - inputs[:,:-2] ) \
            #     / ( tarr[2:] - tarr[:-2] )
            # udot[:,0] = udot[:,1]*1.
            # udot[:,-1] = udot[:,-2]*1.
            udot = np.vstack([_dxl(inputs[k],tarr) for k in range(len(inputs))])
            if plot_ul_bounds:
                # udpp =  iupp*0.
                # udpp[:,1:-1] = ( iupp[:,2:] - iupp[:,:-2] ) \
                #     / ( tarr[2:] - tarr[:-2] )
                # udpp[:,0] = udpp[:,1]*1.
                # udpp[:,-1] = udpp[:,-2]*1.
                udpp = np.vstack([_dxl(iupp[k],tarr) \
                    for k in range(len(inputs))])
                # udow =  ilow*0.
                # udow[:,1:-1] = ( ilow[:,2:] - ilow[:,:-2] ) \
                #     / ( tarr[2:] - tarr[:-2] )
                # udow[:,0] = udow[:,1]*1.
                # udow[:,-1] = udow[:,-2]*1.
                udow = np.vstack([_dxl(ilow[k],tarr) \
                    for k in range(len(inputs))])
            udot_lbl = "cent diff resp"
        # grid, axis labels, legends
        udot_axs[0].grid(which="major",lw=0.6,ls="-",c="0.75")
        # udot_axs[0].grid(which="minor",lw=0.5,ls="dotted",c="0.5")
        udot_axs[1].grid(which="major",lw=0.6,ls="-",c="0.75")
        # udot_axs[1].grid(which="minor",lw=0.5,ls="dotted",c="0.5")
        udot_axs[2].grid(which="major",lw=0.6,ls="-",c="0.75")
        # udot_axs[2].grid(which="minor",lw=0.5,ls="dotted",c="0.5")
        udot_axs[3].grid(which="major",lw=0.6,ls="-",c="0.75")
        # udot_axs[3].grid(which="minor",lw=0.5,ls="dotted",c="0.5")
        # line var labels
        udot_axs[0].text(tarr[ilbl],udot[0,ilbl],
            Del+r"$\dot{\delta}_a$",bbox=bbox_dict,**lbl_params)
        udot_axs[1].text(tarr[ilbl],udot[1,ilbl],
            Del+r"$\dot{\delta}_"+de+r"$",bbox=bbox_dict,**lbl_params)
        udot_axs[2].text(tarr[ilbl],udot[2,ilbl],
            Del+r"$\dot{\delta}_"+dr+r"$",bbox=bbox_dict,**lbl_params)
        udot_axs[3].text(tarr[ilbl],udot[3,ilbl],
            Del+r"$\dot{\tau}$",bbox=bbox_dict,**lbl_params)
        udot_fig.supxlabel(r"Time, s")
        udot_fig.supylabel(r"Control Rate, deg/s or per-unit/s")
        # xticks
        udot_axs[3].set_xticks(ticks=xticks)
        if plot_ul_bounds:
            udot_axs[0].fill_between(tarr, udpp[0], udow[0],**fill)
            udot_axs[0].fill_between(tarr, udpp[0], udow[0],ls="-",**fil2)
            udot_axs[1].fill_between(tarr, udpp[1], udow[1],**fill)
            udot_axs[1].fill_between(tarr, udpp[1], udow[1],ls="-",**fil2)
            udot_axs[2].fill_between(tarr, udpp[2], udow[2],**fill)
            udot_axs[2].fill_between(tarr, udpp[2], udow[2],ls="-",**fil2)
            udot_axs[3].fill_between(tarr, udpp[3], udow[3],**fill)
            udot_axs[3].fill_between(tarr, udpp[3], udow[3],ls="-",**fil2)
        udot_axs[0].plot(tarr, udot[0],c=c,ls="-",label=udot_lbl)
        udot_axs[1].plot(tarr, udot[1],c=c,ls="-")
        udot_axs[2].plot(tarr, udot[2],c=c,ls="-")
        udot_axs[3].plot(tarr, udot[3],c=c,ls="-")
        ## limits ##
        if self.order >= 1 and self._limit_input_rates:
            udot_axs[0].plot(tarr,zrs+np.rad2deg(self.min_dadot),c="0.25",ls="--")
            udot_axs[0].plot(tarr,zrs+np.rad2deg(self.max_dadot),c="0.25",ls="--")
            udot_axs[1].plot(tarr,zrs+np.rad2deg(self.min_dedot),c="0.25",ls="--")
            udot_axs[1].plot(tarr,zrs+np.rad2deg(self.max_dedot),c="0.25",ls="--")
            udot_axs[2].plot(tarr,zrs+np.rad2deg(self.min_drdot),c="0.25",ls="--")
            udot_axs[2].plot(tarr,zrs+np.rad2deg(self.max_drdot),c="0.25",ls="--")
            udot_axs[3].plot(tarr,zrs+self.min_taudot,c="0.25",ls="--")
            udot_axs[3].plot(tarr,zrs+self.max_taudot,c="0.25",ls="--")
            udot_axs[0].set_ylim((1.1*np.rad2deg(self.min_dadot),\
                1.1*np.rad2deg(self.max_dadot)))
            udot_axs[1].set_ylim((1.1*np.rad2deg(self.min_dedot),\
                1.1*np.rad2deg(self.max_dedot)))
            udot_axs[2].set_ylim((1.1*np.rad2deg(self.min_drdot),\
                1.1*np.rad2deg(self.max_drdot)))
            udot_axs[3].set_ylim((1.1*self.min_taudot,1.1*self.max_taudot))

        # double derivative of control
        if self.order == 2:
            inputs = xarr[16:20]
            uddt =  inputs * 0.0
            uddt[:,1:-1] = ( inputs[:,2:] - inputs[:,:-2] ) \
                / ( tarr[2:] - tarr[:-2] )
            uddt = np.vstack([_dxl(inputs[k],tarr) \
                for k in range(len(inputs))])
            if plot_ul_bounds:
                derp = xupp[16:20]
                # uddp = derp*0.0
                # uddp[:,1:-1] = ( derp[:,2:] - derp[:,:-2] ) \
                # / ( tarr[2:] - tarr[:-2] )
                uddp = np.vstack([_dxl(derp[k],tarr) \
                    for k in range(len(inputs))])
                derw = xlow[16:20]
                # uddw = derw*0.0
                # uddw[:,1:-1] = ( derw[:,2:] - derw[:,:-2] ) \
                # / ( tarr[2:] - tarr[:-2] )
                uddw = np.vstack([_dxl(derw[k],tarr) \
                    for k in range(len(inputs))])
            uddt_lbl = "1st ord cent diff"
        else:
            if self.order == 0:
                inputs = uarr
            else:
                inputs = xarr[12:16]
            if plot_ul_bounds:
                if self.order == 0:
                    iupp = uupp
                    ilow = ulow
                else:
                    iupp = xupp[12:16]
                    ilow = xlow[12:16]
            # uddt =  inputs * 0.0
            # uddt[:,1:-1] = ( inputs[:,2:] - 2.*inputs[:,1:-1] + \
            #     inputs[:,:-2] ) \
            #     / ( (tarr[2:] - tarr[:-2])/2.)**2.
            uddt = np.vstack([ddxl(inputs[k],tarr) \
                for k in range(len(inputs))])
            if plot_ul_bounds:
                # uddp =  iupp*0.
                # uddp[:,1:-1] = ( iupp[:,2:] - 2.*iupp[:,1:-1] + \
                #     iupp[:,:-2] ) \
                #     / ( (tarr[2:] - tarr[:-2])/2.)**2.
                uddp = np.vstack([ddxl(iupp[k],tarr) \
                    for k in range(len(inputs))])
                # uddw =  ilow*0.
                # uddw[:,1:-1] = ( ilow[:,2:] - 2.*ilow[:,1:-1] + \
                #     ilow[:,:-2] ) \
                #     / ( (tarr[2:] - tarr[:-2])/2.)**2.
                uddw = np.vstack([ddxl(ilow[k],tarr) \
                    for k in range(len(inputs))])
            uddt_lbl = "2nd ord cent diff"
        
        uddt_fig, uddt_axs = plt.subplots(4,1,**subdict)

        # grid, axis labels, legends
        uddt_axs[0].grid(which="major",lw=0.6,ls="-",c="0.75")
        # uddt_axs[0].grid(which="minor",lw=0.5,ls="dotted",c="0.5")
        uddt_axs[1].grid(which="major",lw=0.6,ls="-",c="0.75")
        # uddt_axs[1].grid(which="minor",lw=0.5,ls="dotted",c="0.5")
        uddt_axs[2].grid(which="major",lw=0.6,ls="-",c="0.75")
        # uddt_axs[2].grid(which="minor",lw=0.5,ls="dotted",c="0.5")
        uddt_axs[3].grid(which="major",lw=0.6,ls="-",c="0.75")
        # uddt_axs[3].grid(which="minor",lw=0.5,ls="dotted",c="0.5")
        # line var labels
        uddt_axs[0].text(tarr[ilbl],uddt[0,ilbl],
            Del+r"$\ddot{\delta}_a$",bbox=bbox_dict,**lbl_params)
        uddt_axs[1].text(tarr[ilbl],uddt[1,ilbl],
            Del+r"$\ddot{\delta}_"+de+r"$",bbox=bbox_dict,**lbl_params)
        uddt_axs[2].text(tarr[ilbl],uddt[2,ilbl],
            Del+r"$\ddot{\delta}_"+dr+r"$",bbox=bbox_dict,**lbl_params)
        uddt_axs[3].text(tarr[ilbl],uddt[3,ilbl],
            Del+r"$\ddot{\tau}$",bbox=bbox_dict,**lbl_params)
        uddt_fig.supxlabel(r"Time, s")
        uddt_fig.supylabel(r"Control Acceleration, deg/s$^2$ or per-unit/s$^2$")
        # xticks
        uddt_axs[3].set_xticks(ticks=xticks)
        if plot_ul_bounds:
            uddt_axs[0].fill_between(tarr, uddp[0], uddw[0],**fill)
            uddt_axs[0].fill_between(tarr, uddp[0], uddw[0],ls="-",**fil2)
            uddt_axs[1].fill_between(tarr, uddp[1], uddw[1],**fill)
            uddt_axs[1].fill_between(tarr, uddp[1], uddw[1],ls="-",**fil2)
            uddt_axs[2].fill_between(tarr, uddp[2], uddw[2],**fill)
            uddt_axs[2].fill_between(tarr, uddp[2], uddw[2],ls="-",**fil2)
            uddt_axs[3].fill_between(tarr, uddp[3], uddw[3],**fill)
            uddt_axs[3].fill_between(tarr, uddp[3], uddw[3],ls="-",**fil2)
        uddt_axs[0].plot(tarr, uddt[0],c=c,ls="-",label=uddt_lbl)
        uddt_axs[1].plot(tarr, uddt[1],c=c,ls="-")
        uddt_axs[2].plot(tarr, uddt[2],c=c,ls="-")
        if self.is_BIRE and self.order >= 2:
            ddBc = "0.25"
            # else:
            #     ddBc = "g"
            ddBmax_deg = 0. * tarr + np.rad2deg(self.max_drddot)
            uddt_axs[2].plot(tarr, ddBmax_deg,c=ddBc,ls="--")
            uddt_axs[2].plot(tarr,-ddBmax_deg,c=ddBc,ls="--")
            uddt_axs[2].set_ylim((-np.rad2deg(self.max_drddot)-10.,\
                np.rad2deg(self.max_drddot)+10.))
        uddt_axs[3].plot(tarr, uddt[3],c=c,ls="-")
        
        # set xlimits
        # print(perc_zoom,tarr[-1])
        vels_axs[  0].set_xlim((0.,perc_zoom*self.tf))
        aero_axs[  0].set_xlim((0.,perc_zoom*self.tf))
        rate_axs[  0].set_xlim((0.,perc_zoom*self.tf))
        if self.tracking:
            errs_axs    .set_xlim((0.,perc_zoom*self.tf))
            igrs_axs    .set_xlim((0.,perc_zoom*self.tf))
        posn_axs[  0].set_xlim((0.,perc_zoom*self.tf))
        ornt_axs[  0].set_xlim((0.,perc_zoom*self.tf))
        ctrl_axs[  0].set_xlim((0.,perc_zoom*self.tf))
        surf_axs     .set_xlim((0.,perc_zoom*self.tf))
        udot_axs[  0].set_xlim((0.,perc_zoom*self.tf))
        uddt_axs[  0].set_xlim((0.,perc_zoom*self.tf))
        # set legend font
        fnt = 8.0
        if plot_second_set:
            vels_axs[0].legend(fontsize=fnt)
            aero_axs[0].legend(fontsize=fnt)
            rate_axs[0].legend(fontsize=fnt)
            if self.tracking:
                errs_axs[0].legend(fontsize=fnt)
                igrs_axs[0].legend(fontsize=fnt)
            posn_axs[0].legend(fontsize=fnt)
            ornt_axs[0].legend(fontsize=fnt)
            # ctrl_axs[0].legend(fontsize=fnt)
            # udot_axs[0].legend(fontsize=fnt)
            # uddt_axs[0].legend(fontsize=fnt)
        # save figs
        if not(save_states and not(plot_full)):
            vels_fig.savefig(predir+"velocities."+format,**savedict)
            aero_fig.savefig(predir+"aero_angles."+format,**savedict)
            rate_fig.savefig(predir+"rates."+format,**savedict)
            if self.tracking:
                errs_fig.savefig(predir+"errors."+format,**savedict)
                igrs_fig.savefig(predir+"integrator_states."+format,**savedict)
            posn_fig.savefig(predir+"position."+format,**savedict)
            ornt_fig.savefig(predir+"orientation."+format,**savedict)
            ctrl_fig.savefig(predir+"inputs_all."+format,**savedict)
            surf_fig.savefig(predir+"inputs_surfaces."+format,**savedict)
            udot_fig.savefig(predir+"actuation_rates."+format,**savedict)
            uddt_fig.savefig(predir+"actuation_acceleration."+format,**savedict)

        # save states
        if save_states:
            print("         saving states...")
            # file name
            save_file = folder[:-(len(folder.split("/")[-2])+1)] + prename
            save_file += "states_output.csv"

            # build header
            li = 16
            lh = li + 8
            hd = ""
            # time
            hd += ("{:>{}s},").format("time [s]",lh)
            # velocities
            vel = ["V{}b [ft/s]".format(sub) for sub in ["x","y","z"]]
            hd += ("{:>{}s},"*3).format(vel[0],lh,vel[1],lh,vel[2],lh)
            # fix start and end to align
            hd = hd[3:-1]
            # rates
            pqr = ["{} [deg/s]".format(sub) for sub in ["p","q","r"]]
            hd += (",{:>{}s}"*3).format(pqr[0],lh,pqr[1],lh,pqr[2],lh)
            # position
            psn = ["{}f [ft]".format(sub) for sub in ["x","y","z"]]
            hd += (",{:>{}s}"*3).format(psn[0],lh,psn[1],lh,psn[2],lh)
            # position
            orn = ["{} [deg]".format(sub) for sub in ["phi","theta","psi"]]
            hd += (",{:>{}s}"*3).format(orn[0],lh,orn[1],lh,orn[2],lh)
            # actuator states
            de = de.replace("^","")
            acts_list = ["da","d"+de,"d"+dr,"tau"]
            act = ["{} [deg]".format(sub) for sub in acts_list[:3]]
            act += ["{} [per-unit]".format(acts_list[3])]
            hd += (",{:>{}s}"*4).format(act[0],lh,act[1],lh,act[2],lh,act[3],lh)
            # actuatordot states
            act = ["{}dot [deg/s]".format(sub) for sub in acts_list[:3]]
            act += ["{}dot [per-unit/s]".format(acts_list[3])]
            hd += (",{:>{}s}"*4).format(act[0],lh,act[1],lh,act[2],lh,act[3],lh)
            # actuatordotdot states
            act = ["{}ddot [deg/s/s]".format(sub) for sub in acts_list[:3]]
            act += ["{}ddot [per-unit/s/s]".format(acts_list[3])]
            hd += (",{:>{}s}"*4).format(act[0],lh,act[1],lh,act[2],lh,act[3],lh)

            # build array
            arr = xarr[:12].T*1.0
            # add in time
            arr = np.hstack((tarr[:,np.newaxis],arr))
            # add in control
            arr = np.hstack((arr,svuc.T))
            # add in dot
            arr = np.hstack((arr,udot.T))
            # add in ddot
            arr = np.hstack((arr,uddt.T))

            # save file
            np.savetxt(save_file,arr,fmt="% ."+str(li)+"e",delimiter=", ",
                       header=hd)

            # quit()

        # show or close figures
        if show:
            plt.show()
        else:
            plt.close("all")

        return


    def plot_results(self,**kwargs):

        # report
        print("plotting results...")
        zm_full = kwargs.get("zoom_full",False)
        zm_delt = kwargs.get("zoom_deltas",False)
        zm_frc = kwargs.get("zoom_fraction",1.0)
        plot_full = kwargs.get("plot_full",True)
        plot_delta = kwargs.get("plot_delta",True)
        output_states = kwargs.get("output_states",False)

        # plot full results
        if output_states or plot_full:
            self._plot_results(plot_deltas=False, save_states=output_states, 
                               **kwargs)
        if zm_full:
            self._plot_results(plot_deltas=False, percent_zoom=zm_frc,**kwargs)

        # plot deltas results
        if plot_delta:
            self._plot_results(plot_deltas=True, **kwargs)
        if zm_delt:
            self._plot_results(plot_deltas=True, percent_zoom=zm_frc,**kwargs)



class GainSchedulingAircraft(Aircraft):
    """A default class for calculating and containing the mass properties of a
    Cuboid.

    Parameters
    ----------
    input_vars : dict , optional
        Must be a python dictionary
    """
    def __init__(self,input_dict={}):

        # invoke init of parent
        Aircraft.__init__(self,input_dict,folder_prefix = "stblz")
    

    def _get_control(self,t,x,is_controlled=True,given_control=False,u="o"):
        # build control or pass through
        if not given_control:
            if is_controlled:
                if self.use_quaternions:
                    x_euler = self.quat2euler_state(x)
                else:
                    x_euler = x*1.
                    # reset angles
                    x_euler[9:12] = quat_2_euler(euler_2_quat(x_euler[9:12]))
                # # # uncomment for gain scheduling
                try:
                    # determine time for gain scheduling
                    # tt = t/self.t_gs # 1. if t >= self.t_gs else
                    # # 2 points
                    # x_tr = (1. - tt)*self.Lin_Model.xhat_eq + \
                    #     tt*self.Lin_Model2.xhat_eq
                    # u_tr = (1. - tt)*self.Lin_Model.uhat_eq + \
                    #     tt*self.Lin_Model2.uhat_eq
                    # u = (1. - tt)*self.u_trim + tt*self.u_trim2
                    # K_tr = (1. - tt)*self.Lin_Model.K + tt*self.Lin_Model2.K
                    # # many points
                    # x_tr = np.array([np.interp(\
                    #     tt,self.t_tr,self.x_tr_euler[:,i], \
                    #     # left=, \
                    #     right=self.x_trim2_euler_slf[i] \
                    #     ) \
                    #     for i in self.Lin_Model.Cslice])
                    # K_tr = [[np.interp(tt,self.t_tr,self.K_tr[:,j,i], \
                    #     # left=,
                    #     right=self.K_slf[j,i] \
                    #     ) \
                    #     for i in range(self.K_tr.shape[2])] \
                    #     for j in range(self.K_tr.shape[1])]
                    # # scipy nearest neighbor
                    x_tr = self.x_tr_itp(t)
                    u_tr = self.u_tr_itp(t)
                    K_tr = self.K_tr_itp(t)
                except:
                    raise TypeError("Error! Error! Error!")
                # x_tr = self.Lin_Model.xhat_eq*1.
                # u_tr = self.Lin_Model.uhat_eq*1.
                # K_tr = self.Lin_Model.K
                #
                u = self.u_trim*1.
                # u_tr = self.Lin_Model.uhat_eq*1.
                ###################################
                Dx = x_euler[self.Lin_Model.Cslice] - x_tr
                u[self.Lin_Model.Cuslice] = u_tr - np.matmul(K_tr,Dx)#*0.
                if self.order > 0:
                    q = 1*self.use_quaternions
                    inputs = x[12+q:16+q]*1.
                else:
                    inputs = u*1.
            else:
                inputs = u = self.Lin_Model.uhat_eq*1.
        elif given_control:
            if u[0] == "o":
                raise TypeError("Control input required.")
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



def damped_sinusoid(x,A,s,w,p):
    return A*np.exp(-s*x)*np.sin(w*x + p)

def damped_sinusoid_derivatives(x,A,s,w,p):
    return [
        [np.exp(-s*x)*np.sin(w*x + p)],
        [-x*A*np.exp(-s*x)*np.sin(w*x + p)],
        [x*A*np.exp(-s*x)*np.cos(w*x + p)],
        [A*np.exp(-s*x)*np.cos(w*x + p)]
    ]

def controllability_analysis(filename,H=15000.,M=0.6,aircraft_class=Aircraft):
    # initialze file
    # pull in json file
    input_vars_type = type(filename)
    # dictionary
    if input_vars_type == dict:
        input_dict = filename
    
    # json file
    elif input_vars_type == str and filename.split(".")[-1] == "json":
        # import json file from file path
        json_string = open(filename).read()
        # save to vals dictionary
        input_dict = json.loads(json_string)

    # initialize range of bank angles
    n_bank = 23 # 11 # 3 # 
    n_cgxs = 9 # 5 # 3 # 
    bank_max = 55.
    banks = np.linspace(-bank_max,bank_max,n_bank)
    # initialize range of cg locations
    cgxs = np.linspace(-1.,1.,n_cgxs)
    # initialize controllability array
    Gamma_ranks = np.zeros((n_bank,n_cgxs))

    # get velocity
    _,_,_,_,_,sos = stdatm_english(H)
    V = M*sos    

    # affix trim settings
    initial = {
        "airspeed[ft/s]" : V,
        "longitude[deg]" : 0.0,
        "latitude[deg]" : 0.0,
        "altitude[ft]" : H,
        "heading[deg]" : 0.0,
        "type" : "trim",
        "trim" : {
            "type" : "sct",
            "climb_angle[deg]" : 0.0,
            "bank_angle[deg]" : 0.0,
            "solver" : {
                "finite_difference_step_size" : 0.001,
                "relaxation_factor" : 0.1,
                "tolerance" : 1.0e-9
            },
            "verbose_trim" : False
        }
    }
    input_dict["initial"] = initial
    order = input_dict["actuators"]["order"]
    # input_dict["simulation"]["use_quaternions"] = False
    if input_dict["simulation"]["use_quaternions"]:
        R = 8 + order*4
    else:
        R = 8 + order*4

    # run through range of bank angles and cg locations
    for i in range(banks.shape[0]):
        input_dict["initial"]["trim"]["bank_angle[deg]"] = banks[i]

        for j in range(cgxs.shape[0]):
            input_dict["aircraft"]["CG_shift[ft]"][0] = cgxs[j]

            # report
            print("running phi = {:> 6.2f}, cg[x] = {:> 5.2f}...".format(\
                banks[i],cgxs[j]))
            
            # build aircraft model, run linearization and control design
            try:
                craft = aircraft_class(input_dict)
                craft._build_controller(report=False,save_matrices=False)

                # save controllability
                Gamma_ranks[i,j] = craft.Lin_Model.Gamma_rank_min
            except:
                Gamma_ranks[i,j] = 0
    
    # create plot
        
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
    plt.rcParams["xtick.major.width"] = plt.rcParams["ytick.major.width"] = 1.0
    plt.rcParams["xtick.minor.width"] = plt.rcParams["ytick.minor.width"] = 1.0
    plt.rcParams["xtick.major.size"] = plt.rcParams["ytick.major.size"] = 5.0
    plt.rcParams["xtick.minor.size"] = plt.rcParams["ytick.minor.size"] = 2.5
    # plt.rcParams["axes.labelpad"] = "10"
    # plt.rcParams["font.weight"] = "bold"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams['figure.dpi'] = 300.0


    ## create plots
    is_BIRE = input_dict["simulation"]["BIRE"]
    predir = craft.fldr_prfx + "_" + "plots/controllability/" + \
        is_BIRE*"bire_" + (not is_BIRE)*"base_"
    predir = predir + str(order) + "_ord_"
    show_plots = False
    transparent = True # False # 
    pltform = "pdf" # "pdf" # 
    savedict = dict(transparent=transparent,format=pltform,dpi=300.0)

    ctrb_fig, ctrb_axs = plt.subplots(1,1,tight_layout=True)

    # plot
    props = {
        "ls" : "none",
        "marker" : "o",
        "c" : "k"
    }
    mfcCC = "none"
    mfcnCC = "k"
    legend_elements = [
        Line2D([0],[0],mfc=mfcCC,label="C",**props),
        Line2D([0],[0],mfc=mfcnCC,label="nC",**props)
    ]
    for i in range(banks.shape[0]):
        for j in range(cgxs.shape[0]):
            if Gamma_ranks[i,j] > 0:
                if Gamma_ranks[i,j] < R:
                    mfc = mfcnCC
                else:
                    mfc = mfcCC
                ctrb_axs.plot(cgxs[j],banks[i],mfc=mfc,**props)
                if Gamma_ranks[i,j] < R:
                    ctrb_axs.text(cgxs[j],banks[i],s=str(int(Gamma_ranks[i,j])),\
                        ha="center",va="center",size=6.0,c="w",weight="bold")
    ctrb_fig.supxlabel("x cg location [ft]")
    ctrb_fig.supylabel("bank angle [deg]")
    fnt = 8.0
    ctrb_axs.legend(handles=legend_elements,fontsize=fnt,\
        bbox_to_anchor=(1.05, 1.0), loc='upper left')
    ctrb_fig.savefig(predir+"ctrb."+pltform,**savedict)

    if show_plots:
        plt.show()
    else:
        plt.close()

def check_linearization_assumption(filename,prtb_deg=5.,mrrr=None,
    actr_warm_start=False,aircraft_class=Aircraft,**plot_dict):
    # pull in json file
    input_vars_type = type(filename)
    # dictionary
    if input_vars_type == dict:
        input_dict = filename
    
    # json file
    elif input_vars_type == str and filename.split(".")[-1] == "json":
        # import json file from file path
        json_string = open(filename).read()
        # save to vals dictionary
        input_dict = json.loads(json_string)

    # get linear nonlinear parameter and initialize aircraft
    simulation = input_dict.get("simulation",{})
    simulation["nonlinear_dynamics"] = False
    simulation["use_quaternions"] = True
    simulation["total_time[sec]"] = 10.
    #
    input_dict["simulation"] = simulation
    linear_craft = aircraft_class(input_dict)
    simulation["nonlinear_dynamics"] = True
    input_dict["simulation"] = simulation
    nonlin_craft = aircraft_class(input_dict)
    #
    _,g,_,_,rho,_ = linear_craft.stdatm(linear_craft.H0)

    # setup perturbation shift
    report_trim = False
    shift = np.deg2rad(prtb_deg) # bire can take 1 deg
    #
    a_trim = atan2(linear_craft.x_trim[2],linear_craft.x_trim[0])
    V_trim = ( linear_craft.x_trim[0]**2. + linear_craft.x_trim[1]**2. + 
        linear_craft.x_trim[2]**2. )**0.5
    ushift = V_trim*cos(a_trim+shift)*cos(shift) - linear_craft.x_trim[0]
    vshift = V_trim*sin(shift) - linear_craft.x_trim[1]
    wshift = V_trim*sin(a_trim+shift)*cos(shift) - linear_craft.x_trim[2]
    #
    quat_trim = linear_craft.x_trim[9:13]*1.
    euler_trim = quat_2_euler(quat_trim)
    euler_shift = [euler_trim[0]+shift,euler_trim[1]+shift,euler_trim[2]]
    quat_shift = (np.array(euler_2_quat(euler_shift))-quat_trim).tolist()
    #
    dx0 = np.array([
        ushift, vshift, wshift,
        shift, shift, shift,
        0.0, 0.0, 0.0
    ] + quat_shift + linear_craft.order*[0.0,0.0,0.0,0.0])

    # call run sim
    linear_craft.run_simulation(report_trim=report_trim,mrrr=mrrr,delta_x0=dx0,
        actr_warm_start=actr_warm_start)
    nonlin_craft.run_simulation(report_trim=report_trim,mrrr=mrrr,delta_x0=dx0,
        actr_warm_start=actr_warm_start)

    # set datasets
    nonlin_craft.zarr  = linear_craft.xarr *1.
    nonlin_craft.varr  = linear_craft.uarr *1.
    nonlin_craft.aeroz = linear_craft.aerox*1.
    plot_dict["first_set_label"] = "nonlinear"
    plot_dict["second_set_label"] = "linear"
    plot_dict["plot_second_set"] = True

    # plot using plot_results
    nonlin_craft.plot_results(**plot_dict)

    return


def run_single_simulation(filename,rtdst_1sg=[20.,10.,5.],
    mrrr=None,mrrc=None,
    aero_model_errors=[0.,0.,0.,0.],inertia_model_errors=[0.,0.,0.,0.],
    FM_errors = [0.,0.,0.,0.,0.,0.],
    fixed_FM_errors = "o",
    actr_warm_start=False,num=1,cut_mine=False,save_data=True,name_end="",
    state_threshold="o", has_turbulence=True,turbulence_setting="light",
    rerandomize_turbulence=False,
    final_time=15.0,track_check_time="o",time_step=0.01,
    initial_velocity=634.,
    initial_mach="o",initial_altitude=15000.0,
    start_climbing=False,end_gs_climbing=False,
    trim_climb=0.0,trim_bank=0.0,
    final_velocity="o",
    final_mach="o",final_altitude="o",
    t_gain_schedule=8.,gain_steps=10,trim_steps=2,
    interpolation_type="linear",
    include_stall_derivatives=False,
    random_seed=None,
    turbulence_random_seed=None,
    error_random_seed=None,
    has_model_error=True,
    skip_simulation=False,
    aircraft_class=Aircraft,
    **plot_dict):
    # pull in json file
    input_vars_type = type(filename)
    # dictionary
    if input_vars_type == dict:
        input_dict = filename
    # json file
    elif input_vars_type == str and filename.split(".")[-1] == "json":
        # import json file from file path
        json_string = open(filename).read()
        # save to vals dictionary
        input_dict = json.loads(json_string)
    
    if initial_mach == "o":
        initial_mach = initial_velocity / stdatm_english(initial_altitude)[5]
    if final_altitude == "o":
        final_altitude = initial_altitude
    if final_mach == "o":
        if final_velocity == "o":
            final_mach = initial_mach
        else:
            final_mach = final_velocity / stdatm_english(final_altitude)[5]

    # get linear nonlinear parameter and initialize aircraft
    simulation = input_dict.get("simulation",{})
    # compressible = simulation["include_compressibility"] # True # False # 
    stallable = True # False # 
    use_quats = True
    climb_mult = ( 1. + np.tan(np.deg2rad(trim_climb))**2. )**0.5
    simulation = {
        "constant_density" : True,
        "time_step[sec]" : time_step,
        "total_time[sec]" : final_time,
        "integrator" : simulation.get("integrator","odeint"),
        "nonlinear_dynamics" : True,
        "use_quaternions" : use_quats,
        #############################
        "limit_input" : simulation.get("limit_input",True),
        "limit_input_rates" : simulation.get("limit_input_rates",True),
        "include_compressibility" : simulation["include_compressibility"],
        "use_Anderson_corrections" : simulation["use_Anderson_corrections"],
        "include_stall" : simulation["include_stall"],
        #############################
        "simulate_uncontrolled" : False,
        "use_fitted_thrust_model" : simulation["use_fitted_thrust_model"],
        "BIRE" : simulation["BIRE"],
        "full_scale" : simulation["full_scale"],
        "random_seed" : random_seed,
        "turbulence_random_seed" : turbulence_random_seed,
        "error_random_seed" : error_random_seed
    }
    #
    initial = {
        "mach" : initial_mach*(climb_mult if start_climbing else 1.),
        "longitude[deg]" : 0.0,
        "latitude[deg]" : 0.0,
        "altitude[ft]" : initial_altitude,
        "heading[deg]" : 0.0,
        "type" : "trim",
        "trim" : {
            "type" : "sct",
            "___elevation_angle[deg]" : 0.0,
            "climb_angle[deg]" : trim_climb if start_climbing else 0.,
            "bank_angle[deg]" : trim_bank,
            "___sideslip_angle[deg]" : 0.0,
            "solver" : {
                "finite_difference_step_size" : 0.001,
                "relaxation_factor" : 0.1,
                "tolerance" : 1.0e-9
            },
            "verbose_trim" : False
        },
        "trim_guess" : input_dict.get("initial",{}).get("trim_guess",{})
    }
    #
    input_dict["simulation"] = simulation
    input_dict["initial"] = initial
    #
    disturbance = input_dict.get("disturbance",{})
    if has_turbulence:
        disturbance["type"] = "von_Karman" # "none" # 
    else:
        disturbance["type"] = "none" # "von_Karman" # 
    disturbance["turbulence_intensity"] = turbulence_setting
    input_dict["disturbance"] = disturbance
    #
    if initial_altitude != final_altitude or initial_mach != final_mach:
        aircraft = GainSchedulingAircraft(input_dict)
    else:
        aircraft = aircraft_class(input_dict)
    print()
    if trim_climb != 0. and start_climbing:
        print("steady climbing flight at " + \
            "{} ft altitude, climb angle = {} deg".format(initial_altitude,\
            trim_climb))
    else:
        print("steady level flight at {} ft altitude".format(initial_altitude))
    aircraft._report_trim_solution(aircraft.x_trim,aircraft.u_trim)

    # track check time
    if track_check_time == "o":
        track_check_time = final_time
    
    # trim states calculation
    guesses = {"a_guess":aircraft.a_guess,"b_guess":aircraft.b_guess,
     "phi_guess":aircraft.phi_guess,"u_guess":aircraft.u_guess}
    trim_steps = max(trim_steps,2)
    trim_sts = np.zeros((trim_steps,aircraft.x_trim_euler.shape[0]))
    trim_cos = np.zeros((trim_steps,aircraft.u_trim.shape[0]))
    for i in range(trim_steps):
        perc = float(i/(trim_steps - 1))
        if end_gs_climbing:
            Y = trim_climb
        else:
            Y = (1. - perc)*trim_climb
        aircraft.climb_trim = np.deg2rad(Y)
        H = perc*(final_altitude - initial_altitude)+initial_altitude
        aircraft.H0 = H
        M = perc*(final_mach - initial_mach)+initial_mach
        Y_mult = ( 1. + np.tan(np.deg2rad(Y))**2. )**0.5
        aircraft.V0 = M*aircraft.stdatm(H)[5]*Y_mult
        aircraft._initialize_state(run2=True,no_report=True,**guesses)
        trim_sts[i,:] = aircraft.x_trim2_euler*1.
        trim_cos[i,:] = aircraft.u_trim2[:aircraft.u_trim.shape[0]]*1.
        print()
        if trim_climb != 0.:
            print("steady climbing flight at " + \
                "{} ft altitude, climb angle = {} deg".format(H,Y))
        else:
            print("steady level flight at {} ft altitude".format(H))
        aircraft._report_trim_solution(aircraft.x_trim2,aircraft.u_trim2)

    aircraft.climb_trim = 0.
    aircraft.V0 = final_mach*aircraft.stdatm(final_altitude)[5]
    aircraft._initialize_state(run2=True,no_report=True,**guesses)
    aircraft.x_trim2_euler_slf = aircraft.x_trim2_euler*1.
    aircraft.u_trim2_slf = aircraft.u_trim2*1.
    print()
    print("steady level flight at {} ft altitude".format(final_altitude))
    aircraft._report_trim_solution(aircraft.x_trim2,aircraft.u_trim2)
    aircraft.x0 = aircraft.x_trim*1.
    aircraft.u = aircraft.u_trim*1.
    aircraft.t_gs = t_gain_schedule
    # print(aircraft.x_trim_euler)
    # print(aircraft.u_trim)
    # quit()

    _,g,_,_,rho,_ = aircraft.stdatm(aircraft.H0)
    #
    report_trim = False
    # setup perturbation shift
    shift = np.deg2rad(rtdst_1sg)
    #
    quat_shift = [0., 0., 0.] + aircraft.use_quaternions*[0.]
    #
    dx0 = np.array([
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0
    ] + quat_shift + aircraft.order*[0.0,0.0,0.0,0.0] + [0.0]*len(aircraft.xIi))


    if state_threshold == "o":
        Dx_norm_stable_threshold = 1. # FIX THIS
        raise TypeError("E not calculated for this case yet")
    else:
        Dx_norm_stable_threshold = 1.
        E = np.diag(1./(np.array(state_threshold)**2.))
        l_i = len(aircraft.xIi)
        E = np.diag(1./(np.array(state_threshold + [1.0]*l_i)**2.))

    # check if failed folder exists
    degs = rtdst_1sg
    if aircraft.is_BIRE:
        prename = "bire"
    else:
        prename = "base"
    if aircraft.is_rc:
        prename += "_rc"
    else:
        prename += "_fs"
    run_name = prename + "_" + "p{:d}_q{:d}_r{:d}".format(
        int(abs(degs[0])),int(abs(degs[1])),int(abs(degs[2])))
    run_name = run_name + "_M{:2.1f}".format(initial_mach).replace(".","_") + \
        "_H{:04.1f}".format(initial_altitude/1000.).replace(".","_") + \
        "_P{:02d}".format(int(trim_bank))
    run_name += "_we"*has_model_error + "_ne"*(not has_model_error)
    run_name += "_wD"*has_turbulence + "_nD"*(not has_turbulence)
    run_name += ("_" + turbulence_setting[0])*has_turbulence
    run_name += name_end
    file_folder = aircraft.fldr_prfx + "_" +"plots/single_simulation/"+run_name
    # saving folders
    delta_tf_folder = file_folder + "/delta"
    delta_zm_folder = file_folder + "/delta_zoom"
    full__tf_folder = file_folder + "/full"
    full__zm_folder = file_folder + "/full_zoom"
    snstvty__folder = file_folder + "/sensitivity"
    # assign to plot dict
    plot_dict["plot_delta"] = plot_dict.get("plot_delta",True)
    plot_dict["zoom_deltas"] = plot_dict.get("zoom_deltas",False)
    plot_dict["plot_full"] = plot_dict.get("plot_full",True)
    plot_dict["zoom_full"] = plot_dict.get("zoom_full",False)
    if save_data and path_exists(file_folder):
        # step through and remove every file, then delete folder
        # delete / create folders
        if path_exists(snstvty__folder):
            for filename in listdir(snstvty__folder):
                remove(snstvty__folder + "/" + filename)
        if  not(skip_simulation):
            if path_exists(delta_tf_folder):
                for filename in listdir(delta_tf_folder):
                    remove(delta_tf_folder + "/" + filename)
                # delete folder
                if not(plot_dict["plot_delta"]):
                    rmdir(delta_tf_folder)
            elif plot_dict["plot_delta"]:
                mkdir(delta_tf_folder)
            if path_exists(delta_zm_folder):
                for filename in listdir(delta_zm_folder):
                    remove(delta_zm_folder + "/" + filename)
                # delete folder
                if not(plot_dict["zoom_deltas"]):
                    rmdir(delta_zm_folder)
            elif plot_dict["zoom_deltas"]:
                mkdir(delta_zm_folder)
            if path_exists(full__tf_folder):
                for filename in listdir(full__tf_folder):
                    remove(full__tf_folder + "/" + filename)
                # delete folder
                if not(plot_dict["plot_full"]):
                    rmdir(full__tf_folder)
            elif plot_dict["plot_full"]:
                mkdir(full__tf_folder)
            if path_exists(full__zm_folder):
                for filename in listdir(full__zm_folder):
                    remove(full__zm_folder + "/" + filename)
                # delete folder
                if not(plot_dict["zoom_full"]):
                    rmdir(full__zm_folder)
            elif plot_dict["zoom_full"]:
                mkdir(full__zm_folder)
        # # delete folder
        # rmdir(file_folder)
        for filename in listdir(file_folder):
            if filename not in \
                ["full","sensitivity","delta","delta_zoom","full_zoom"]:
                remove(file_folder + "/" + filename)
    elif save_data:
        mkdir(file_folder)
        mkdir(snstvty__folder)
        if  not(skip_simulation):
            if plot_dict["plot_delta"]:
                mkdir(delta_tf_folder)
            if plot_dict["zoom_deltas"]:
                mkdir(delta_zm_folder)
            if plot_dict["plot_full"]:
                mkdir(full__tf_folder)
            if plot_dict["zoom_full"]:
                mkdir(full__zm_folder)
    plot_dict["plotting_directory"] = file_folder + "/"

    # build controller
    aircraft._build_controller(report=True,save_matrices=save_data,
        filename="matrices",mrrr=mrrr,
        mrrc=mrrc,run_freq=save_data,save_name_end=name_end,
        include_stall_derivatives=include_stall_derivatives,
        drop_actrs=cut_mine,save_folder=file_folder[6:]+"/")
    CTC = np.matmul(aircraft.Lin_Model.C.T,aircraft.Lin_Model.C)
    CEC = np.matmul(CTC,np.matmul(E,CTC))
    # aircraft._build_controller(report=True,save_matrices=True,mrrr=mrrr,
    #     mrrc=mrrc,run_freq=True,save_name_end=name_end,
    #     include_stall_derivatives=include_stall_derivatives,
    #     run2=True,
    #     drop_actrs=cut_mine)
    # determine interpolated trim states and controls
    z21 = np.linspace(0.,1.,num=gain_steps)
    t21 = np.linspace(0.,1.,num=trim_steps)
    # x1, u1 = aircraft.x_trim_euler_climb, aircraft.u_trim_climb
    # x2, u2 = aircraft.x_trim2_euler_climb, aircraft.u_trim2_climb
    # aircraft.x_tr_euler = np.array([(1. - i)*x1 + i*x2 for i in z21])
    # aircraft.u_tr       = np.array([(1. - i)*u1 + i*u2 for i in z21])
    aircraft.x_tr_euler = interp1d(t21,trim_sts,kind="linear",axis=0)(z21)
    aircraft.u_tr       = interp1d(t21,trim_cos,kind="linear",axis=0)(z21)
    K_trs = np.zeros((gain_steps, \
        aircraft.Lin_Model.K.shape[0], aircraft.Lin_Model.K.shape[1]))
    # determine true trim gains
    for i in range(gain_steps):
        # store trim condition
        _,Lin_Model = aircraft._build_controller(
            aircraft.x_tr_euler[i],aircraft.u_tr[i],
            report=False,save_matrices=False,
            mrrr=mrrr,mrrc=mrrc,drop_actrs=True,
            include_stall_derivatives=False,skip_reporting=True,run_freq=False)
        K_trs[i] = Lin_Model.K*1.
    aircraft.K_tr = K_trs*1.
    aircraft.t_tr = z21*t_gain_schedule
    _,Lin_Model = aircraft._build_controller(
        aircraft.x_trim2_euler_slf,aircraft.u_trim2_slf,
        report=False,save_matrices=False,
        mrrr=mrrr,mrrc=mrrc,drop_actrs=True,
        include_stall_derivatives=False,skip_reporting=True,run_freq=False)
    aircraft.K_slf = Lin_Model.K*1.

    # create interpolation functions
    typ = interpolation_type # "linear" # "nearest-up" # 
    nan = float("nan")
    st_r2d_i = [3,4,5,9,10,11]+(aircraft.order>=1)*[12,13,14]
    co_r2d_i = [0,1,2]
    xdeg = aircraft.x_tr_euler*1.
    xdeg[:,st_r2d_i] = np.rad2deg(xdeg[:,st_r2d_i])
    xdsf = aircraft.x_trim2_euler_slf*1.
    xdsf[st_r2d_i] = np.rad2deg(xdsf[st_r2d_i])
    udeg = aircraft.u_tr*1.
    udeg[:,co_r2d_i] = np.rad2deg(udeg[:,co_r2d_i])
    udsf = aircraft.u_trim2_slf*1.
    udsf[co_r2d_i] = np.rad2deg(udsf[co_r2d_i])
    
    aircraft.x_tr_deg_itp = interp1d(aircraft.t_tr,xdeg,kind=typ,\
        axis=0,bounds_error=False,fill_value=(nan,xdsf))
    aircraft.u_tr_deg_itp = interp1d(aircraft.t_tr,udeg,kind=typ,\
        axis=0,bounds_error=False,fill_value=(nan,udsf))

    aircraft.x_tr_euler=np.matmul(aircraft.Lin_Model.C,aircraft.x_tr_euler.T).T
    aircraft.u_tr=np.matmul(aircraft.Lin_Model.C_u,aircraft.u_tr.T).T
    x_slf = np.matmul(aircraft.Lin_Model.C  ,aircraft.x_trim2_euler_slf)
    u_slf = np.matmul(aircraft.Lin_Model.C_u,aircraft.u_trim2_slf)
    aircraft.x_tr_itp = interp1d(aircraft.t_tr,aircraft.x_tr_euler,kind=typ,\
        axis=0,bounds_error=False,fill_value=(nan,x_slf))
    aircraft.u_tr_itp = interp1d(aircraft.t_tr,aircraft.u_tr,kind=typ,\
        axis=0,bounds_error=False,fill_value=(nan,u_slf))
    aircraft.K_tr_itp = interp1d(aircraft.t_tr,aircraft.K_tr,kind=typ,\
        axis=0,bounds_error=False,fill_value=(nan,aircraft.K_slf))

    # aircraft.climb_trim = 0.
    
    
    # # check linearization
    # aircraft.check_partials(report=False,save_matrices=False,mrrr=mrrr,
    #     mrrc=mrrc,run_freq=True,save_name_end=name_end,
    #     include_stall_derivatives=include_stall_derivatives,
    #     drop_actrs=cut_mine)
    
    # skip sim
    print("running case:",run_name)
    if not(skip_simulation):
        # create errored FM
        if type(fixed_FM_errors) == str:
            if has_model_error:
                aircraft.make_FM_error_model(FM_error_percs=FM_errors)
        else:
            aircraft.FM_errors = np.array(fixed_FM_errors)

        # # create errored aero and inertia model
        # aircraft.make_errored_models(aero_model_errors,inertia_model_errors)

        # check if failed folder exists
        degs = rtdst_1sg
        if aircraft.is_BIRE:
            prename = "bire"
        else:
            prename = "base"
        run_name = prename
        run_name += "_quats"*use_quats + "_euler"*(not use_quats)
        run_name += "_{:d}_ord_act_system".format(int(aircraft.order))
        run_name += name_end
        # aircraft._save_controller(aircraft.Lin_Model,file_folder[6:]+"/",\
        #     filename="matrices")

        # create shifting values
        dx0[3:6] = [shift[0],shift[1],shift[2]]
        i = 0
        counter = 0
        header = "   i    "
        names = ["CL","CS","CD","Cl","Cm","Cn"]
        for j in range(6):
            header += " {:^6s}".format(names[j])
        print(header)

        # determine time to check
        ############### tracking checker
        t_track_i = np.argwhere(\
            np.linspace(0.,aircraft.tf,int(aircraft.tf/aircraft.dt))\
            <track_check_time)[-1][0]
        r_track = aircraft._get_reference(aircraft.dt*t_track_i)
        rows = [3,4,5,9,10,11]
        r_track[rows] = np.rad2deg(r_track[rows])
        ###################################

        # test a few
        for i in range(num):
            # renew error
            if type(fixed_FM_errors) == str and has_model_error:
                    aircraft.refresh_FM_error(FM_error_percs=FM_errors)
            
            # rerandomize turbulence
            if has_turbulence and rerandomize_turbulence:
                aircraft.disturbance_model.rebuild_turbulence_phases()

            # run simulation
            xr,ur = aircraft.run_simulation(report_trim=report_trim,
                    mrrr=mrrr,delta_x0=dx0,actr_warm_start=actr_warm_start,
                    save_matrices=False,report_simulation=save_data)
            
            x_zero = xr[:,-1]*1.
            dx = x_zero - (aircraft.x_trim2_euler_deg + r_track)
            Dx_norm = np.matmul(dx.T,np.matmul(CEC,dx))
            if aircraft.tracking:
                dx_track = xr[:,t_track_i] -(aircraft.x_trim_euler_deg+r_track)
                Dx_norm_track = np.matmul(dx_track.T,np.matmul(CEC,dx_track))
                Dx_norm = max(Dx_norm_track,Dx_norm)
            case_run_text = "{:>4d} -- ".format(i+1)
            for j in range(6):
                case_run_text += " {:> 5.3f}".format(aircraft.FM_errors[j])
            case_run_text += " -- |Dx| = {:>9.3f},".format(Dx_norm)
            case_run_text += "   Stable" if Dx_norm <= Dx_norm_stable_threshold \
                else " Unstable"
            print(case_run_text)

            if Dx_norm <= Dx_norm_stable_threshold:
                counter += 1

            if (i+1) % 50 == 0:
                succ = "{:>5d}/{:>5d} cases successful,".format(counter,i+1)
                succ += " est {:>5d}/{:>5d}".format(int(counter*num/(i+1)),num)
                print(succ)
                print(header)
    
        succ = "{:>5d}/{:>5d} cases successful".format(counter,num)
        print(succ)
        
        # report max vals from trim
        # xs = xr - aircraft.x_trim_euler_deg[:,None]
        # us = ur - aircraft.u_trim_deg[:,None]
        xs = xr - np.array([aircraft.x_tr_deg_itp(w) for w in aircraft.tarr]).T
        us = ur - np.array([aircraft.u_tr_deg_itp(w) for w in aircraft.tarr]).T
        state_names = [
            "Vxb","Vyb","Vzb",
            "p","q","r",
            "xf","yf","zf",
            "phi","theta","psi",
            "da","de"+"B"*aircraft.is_BIRE,
            "dB"*aircraft.is_BIRE+"dr"*(not(aircraft.is_BIRE)),"tau",
            "da dot","de"+"B"*aircraft.is_BIRE+" dot",
            "dB"*aircraft.is_BIRE+"dr"*(not(aircraft.is_BIRE))+" dot","tau dot"
        ]
        state_units = ["ft/s"]*3 + ["deg/s"]*3 + ["ft"]*3 + \
            ["deg"]*6 + [""] + ["deg/s"]*3 + [""]
        control_names = [
            "da cmd","de"+"B"*aircraft.is_BIRE+" cmd",
            "dB"*aircraft.is_BIRE+"dr"*(not(aircraft.is_BIRE))+" cmd","tau cmd"
        ]
        control_units = ["deg"]*3 + [""]
        if aircraft.has_turbulence:
            turb_interps = [
                aircraft.disturbance_model.Vgu,
                aircraft.disturbance_model.Vgv,
                aircraft.disturbance_model.Vgw,
                lambda t : np.rad2deg(aircraft.disturbance_model.Wgp(t)),
                lambda t : np.rad2deg(aircraft.disturbance_model.Wgq(t)),
                lambda t : np.rad2deg(aircraft.disturbance_model.Wgr(t))]
        n = 25
        nadd = 16
        print("-"*(n*2+nadd))
        print("-"*(n*2+nadd))
        print("-"*n + "   max states   " + "-"*n)
        for i in range(xs.shape[0] - aircraft.additional_states):
            vals = xs[i,:]
            name = state_names[i]
            unit = state_units[i]
            max_val = np.max(np.abs(vals))
            print("max{:^10s}= {:> 9.3f} {:<5s}".format("\u0394"+name,max_val,\
                unit))
        print("-"*n + "  max controls  " + "-"*n)
        for i in range(us.shape[0]):
            vals = us[i,:]
            name = control_names[i]
            unit = control_units[i]
            max_val = np.max(vals)
            min_val = np.min(vals)
            print("max{:^10s}= {:> 9.3f} {:<5s}".format("\u0394"+name,max_val,\
                unit),end="")
            print(", min{:^10s}= {:> 9.3f} {:<5s}".format("\u0394"+name,\
                min_val,unit))
        if aircraft.has_turbulence:
            print("-"*n + " max turbulence " + "-"*n)
            for i in range(6):
                turb_fun = turb_interps[i]
                vals = [turb_fun(tk) for tk in aircraft.tarr]
                name = state_names[i]
                unit = state_units[i]
                max_val = np.max(np.abs(vals))
                print("max{:^10s}= {:> 9.3f} {:<5s}".format("\u0394"+name,\
                    max_val,unit))
        if aircraft.tracking:
            print("-"*n + " error response " + "-"*n)
            # calc signal
            ref = np.array([aircraft._get_reference(ti) for ti in aircraft.tarr]).T
            ref[aircraft.xicnv] = np.rad2deg(ref[aircraft.xicnv])
            xerr = xr[aircraft.xPi_eul,:] - ref[aircraft.xPi_eul,:]
            info_txt = ("   {:<7s}  & {:^7s} & {:^7s}" + \
                    " & {:^10s} \\\\\n").format(" ","Trs [s]","Tst [s]","%OS")
            for i in range(len(aircraft.xPi_eul)):
                # determine when signal stops changing
                xp = aircraft.ref_data_xp[aircraft.xPi_eul[i]]
                fp = aircraft.ref_data_fp[aircraft.xPi_eul[i]]
                iT = len(fp) - 1
                while fp[iT] == fp[-1] and iT > 0:
                    iT -= 1
                if iT != 0: iT += 1
                iT = np.argwhere(aircraft.tarr >= xp[iT])[0,0]
                # determine error at this timestep
                xe0 = xerr[i,iT]
                # determine rise time 10% -> 90%
                xe0_10 = xe0*0.9; xe0_90 = xe0*0.1
                i_10 = iT; i_90 = len(aircraft.tarr)
                for it in range(iT+1,len(aircraft.tarr)):
                    if (abs(xerr[i,it]) <= abs(xe0_10) or \
                        np.sign(xerr[i,it]) == -np.sign(xe0)):
                        i_10 = it
                        break
                for it in range(iT+1,len(aircraft.tarr)):
                    if (abs(xerr[i,it]) <= abs(xe0_90) or \
                        np.sign(xerr[i,it]) == -np.sign(xe0)):
                        i_90 = it
                        break
                if i_90 == i_10: i_90 += 1
                # calculate
                try:
                    tris = aircraft.tarr[i_90] - aircraft.tarr[i_10]
                except:
                    tris = np.inf
                # settling time
                try:
                    tstl = aircraft.tarr[iT + \
                        np.argwhere(np.abs(xerr[i,iT:]) >=0.05*abs(xe0))[-1,0] + 1]
                    tstl -= aircraft.tarr[iT]
                except:
                    tstl = np.inf
                # overshoot
                posh = (np.max(np.abs(xerr[i,iT:] - xe0)))/abs(xe0) - 1.0
                info_txt+=("$e_{{{:<5s}}}$ & {:> 7.2f} & {:> 7.2f}" + \
                    " & {:> 7.1f} \% \\\\\n").format(
                    state_names[aircraft.xPi_eul[i]],tris,tstl,posh*100.0
                    )
            print(info_txt,end="")
            # save to file
            with open(file_folder+"/signal_info.txt","w") as f:
                f.write(info_txt)
                f.close()
        print("-"*(n*2+nadd))
        print("-"*(n*2+nadd))
        # plot
        if save_data:
            aircraft.plot_results(**plot_dict)

    return


def monte_carlo_perturbations(filename,rtdst_1sg=[5.,5.,5.],
    mrrr=None,mrrc=None,
    actr_warm_start=False,num=20,cut_mine=True,save_data=True,name_end="",
    state_threshold="o", rerandomize_turbulence=False,
    aero_model_errors=[0.,0.,0.,0.],inertia_model_errors=[0.,0.,0.,0.],
    FM_errors = [0.,0.,0.,0.,0.,0.],
    statistical=True,has_turbulence=True,turbulence_setting="light",
    random_seed=None,
    turbulence_random_seed=None,
    error_random_seed=None,
    has_model_error=True,
    final_time=15.0,track_check_time="o",time_step=0.01,
    initial_velocity=634.,initial_mach="o",initial_altitude=15000.0,trim_bank=0.0,
    include_stall_derivatives=False,
    aircraft_class=Aircraft,
    skip_video=False,
    plot_ul_bounds=False,
    **plot_dict):
    # pull in json file
    input_vars_type = type(filename)
    # dictionary
    if input_vars_type == dict:
        input_dict = filename
    
    # json file
    elif input_vars_type == str and filename.split(".")[-1] == "json":
        # import json file from file path
        json_string = open(filename).read()
        # save to vals dictionary
        input_dict = json.loads(json_string)

    if track_check_time == "o":
        track_check_time = final_time

    # get linear nonlinear parameter and initialize aircraft
    simulation = input_dict.get("simulation",{})
    lim_u = True
    lim_du = True
    # compressible = simulation["include_compressibility"] # True # False # 
    stallable = True # False # 
    simulation = {
        "constant_density" : True,
        "time_step[sec]" : time_step,
        "total_time[sec]" : final_time,
        "integrator" : simulation.get("integrator","odeint"),
        "nonlinear_dynamics" : True,
        "use_quaternions" : True,
        #############################
        "limit_input" : lim_u,
        "limit_input_rates" : lim_du,
        "include_compressibility" : simulation["include_compressibility"],
        "use_Anderson_corrections" : simulation["use_Anderson_corrections"],
        "include_stall" : stallable,
        #############################
        "simulate_uncontrolled" : False,
        "use_fitted_thrust_model" : simulation["use_fitted_thrust_model"],
        "BIRE" : simulation["BIRE"],
        "full_scale" : simulation["full_scale"],
        "random_seed" : random_seed,
        "turbulence_random_seed" : turbulence_random_seed,
        "error_random_seed" : error_random_seed
    }
    #
    if initial_mach == "o":
        initial_mach = initial_velocity / stdatm_english(initial_altitude)[5]
    initial = {
        "mach" : initial_mach,
        "longitude[deg]" : 0.0,
        "latitude[deg]" : 0.0,
        "altitude[ft]" : initial_altitude,
        "heading[deg]" : 0.0,
        "type" : "trim",
        "trim" : {
            "type" : "sct",
            "___elevation_angle[deg]" : 0.0,
            "climb_angle[deg]" : 0.0,
            "bank_angle[deg]" : trim_bank,
            "___sideslip_angle[deg]" : 0.0,
            "solver" : {
                "finite_difference_step_size" : 0.001,
                "relaxation_factor" : 0.1,
                "tolerance" : 1.0e-9
            },
            "verbose_trim" : False
        },
        "trim_guess" : input_dict.get("initial",{}).get("trim_guess",{})
    }
    #
    input_dict["simulation"] = simulation
    input_dict["initial"] = initial
    #
    disturbance = input_dict.get("disturbance",{})
    if has_turbulence:
        disturbance["type"] = "von_Karman" # "none" # 
    else:
        disturbance["type"] = "none" # "von_Karman" # 
    if state_threshold == "o":
        Dx_norm_stable_threshold = 1. # FIX THIS
        raise TypeError("E not calculated for this case yet")
    else:
        Dx_norm_stable_threshold = 1.
        controller_dict = input_dict.get("controller",{})
        l_i = len(controller_dict.get("integral_states",[]))
        E = np.diag(1./(np.array(state_threshold + [1.0]*l_i)**2.))
    disturbance["turbulence_intensity"] = turbulence_setting
    input_dict["disturbance"] = disturbance
    #
    aircraft = aircraft_class(input_dict)
    print("trim is \n",aircraft.x_trim,"\n",aircraft.u_trim)
    #
    _,g,_,_,rho,_ = aircraft.stdatm(aircraft.H0)
    #
    report_trim = False
    # setup perturbation shift
    shift = np.deg2rad(rtdst_1sg)
    #
    quat_shift = [0., 0., 0.] + aircraft.use_quaternions*[0.]
    #
    dx0 = np.array([
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0
    ] + quat_shift + aircraft.order*[0.0,0.0,0.0,0.0] + [0.0]*len(aircraft.xIi))

    # determine mean turbulence state
    if state_threshold == "o" and has_turbulence:
        raise TypeError("E not calculated yet")
        u_sig = np.abs(aircraft.disturbance_model.Vgu_signal)
        v_sig = np.abs(aircraft.disturbance_model.Vgv_signal)
        w_sig = np.abs(aircraft.disturbance_model.Vgw_signal)
        p_sig = np.abs(np.rad2deg(aircraft.disturbance_model.Wgp_signal))
        q_sig = np.abs(np.rad2deg(aircraft.disturbance_model.Wgq_signal))
        r_sig = np.abs(np.rad2deg(aircraft.disturbance_model.Wgr_signal))
        u_avg = np.std(u_sig)*3. # np.average(u_sig,weights=u_sig**3./np.max(u_sig)**3.)
        v_avg = np.std(v_sig)*3. # np.average(v_sig,weights=v_sig**3./np.max(v_sig)**3.)
        w_avg = np.std(w_sig)*3. # np.average(w_sig,weights=w_sig**3./np.max(w_sig)**3.)
        p_avg = np.std(p_sig)*3. # np.average(p_sig,weights=p_sig**3./np.max(p_sig)**3.)
        q_avg = np.std(q_sig)*3. # np.average(q_sig,weights=q_sig**3./np.max(q_sig)**3.)
        r_avg = np.std(r_sig)*3. # np.average(r_sig,weights=r_sig**3./np.max(r_sig)**3.)
        # print(u_avg,v_avg,w_avg,p_avg,q_avg,r_avg)
        Dx_add = np.linalg.norm(
            [u_avg, v_avg, w_avg,p_avg, q_avg, r_avg]
        )
        Dx_norm_stable_threshold += Dx_add
    Dx_report = "|Dx| threshold is {:> 10.6}".format(1.)
    # print(Dx_report)
    repstr = Dx_report + "\n"
    repstr += "state_threshold is\n"
    for i in range(len(state_threshold)):
        repstr += "{:> 7.3f}".format(state_threshold[i])
        if (i+1) % 3 == 0:# and i != 0:
            repstr += "\n"
    repstr += "\n"
    print(repstr,end="")
    # quit()

    # determining acceptable threshold values
    # _,_,Sul = aircraft.disturbance_model._interpolate_intensity(initial_altitude,"light")
    # _,_,Sum = aircraft.disturbance_model._interpolate_intensity(initial_altitude,"moderate")
    # _,_,Sus = aircraft.disturbance_model._interpolate_intensity(initial_altitude,"severe")
    # _,_,Su1 = aircraft.disturbance_model._interpolate_intensity(initial_altitude,"1")
    # _,_,Su2 = aircraft.disturbance_model._interpolate_intensity(initial_altitude,"2")
    # _,_,Su3 = aircraft.disturbance_model._interpolate_intensity(initial_altitude,"3")
    # _,_,Su4 = aircraft.disturbance_model._interpolate_intensity(initial_altitude,"4")
    # _,_,Su5 = aircraft.disturbance_model._interpolate_intensity(initial_altitude,"5")
    # _,_,Su6 = aircraft.disturbance_model._interpolate_intensity(initial_altitude,"6")
    # print("mod",25.*Sum/Sul)
    # print("sev",25.*Sus/Sul)
    # print(" 1 ",25.*Su1/Sul)
    # print(" 2 ",25.*Su2/Sul)
    # print(" 3 ",25.*Su3/Sul)
    # print(" 4 ",25.*Su4/Sul)
    # print(" 5 ",25.*Su5/Sul)
    # print(" 6 ",25.*Su6/Sul)
    # quit()

    # create errored FM
    if has_model_error:
        FM_error = FM_errors
    else:
        FM_error = np.zeros((6,)).tolist()
    aircraft.make_FM_error_model(FM_error_percs=FM_error)

    # # create errored aero and inertia model
    # aircraft.make_errored_models(aero_model_errors,inertia_model_errors)

    # check if failed folder exists
    degs = rtdst_1sg
    if aircraft.is_BIRE:
        prename = "bire"
    else:
        prename = "base"
    if aircraft.is_rc:
        prename += "_rc"
    else:
        prename += "_fs"
    run_name = prename + "_" + "p{:d}_q{:d}_r{:d}_n{:d}".format(
        int(abs(degs[0])),int(abs(degs[1])),int(abs(degs[2])),num)
    run_name = run_name + "_M{:2.1f}".format(initial_mach).replace(".","_") + \
        "_H{:04.1f}".format(initial_altitude/1000.).replace(".","_") + \
        "_P{:02d}".format(int(trim_bank))
    # run_name += "_wul"*lim_u + "_nul"*(not lim_u)
    # run_name += "_wrl"*lim_du + "_nrl"*(not lim_du)
    # run_name += "_wc"*compressible + "_nc"*(not compressible)
    # run_name += "_ws"*stallable + "_ns"*(not stallable)
    run_name += "_stat"*statistical + "_cube"*(not statistical)
    run_name += "_we"*has_model_error + "_ne"*(not has_model_error)
    run_name += "_wD"*has_turbulence + "_nD"*(not has_turbulence)
    run_name += ("_" + turbulence_setting[0])*has_turbulence
    run_name += name_end
    ####################################################
    # modify for removing throttle row case
    ####################################################
    file_folder = aircraft.fldr_prfx + "_" + "plots/monte_carlo/"+run_name#+"/"
    fail_folder = file_folder + "/fail_plots"
    # succ_folder = file_folder + "/succ_plots"
    sens_folder = file_folder + "/sensitivity"
    errs_folder = file_folder + "/errs_plots"
    plot_dict["plot_delta"] = plot_dict.get("plot_delta",True)
    plot_dict["zoom_deltas"] = plot_dict.get("zoom_deltas",False)
    plot_dict["plot_full"] = plot_dict.get("plot_full",True)
    plot_dict["zoom_full"] = plot_dict.get("zoom_full",False)
    delta_tf_folder = file_folder + "/delta"
    delta_zm_folder = file_folder + "/delta_zoom"
    full__tf_folder = file_folder + "/full"
    full__zm_folder = file_folder + "/full_zoom"
    if save_data and path_exists(file_folder):
        # step through and remove every file, then delete folder
        # fail folder
        if path_exists(fail_folder):
            for filename in listdir(fail_folder):
                remove(fail_folder + "/" + filename)
            # delete folder
            # rmdir(fail_folder)
        # sens folder
        if path_exists(sens_folder):
            for filename in listdir(sens_folder):
                remove(sens_folder + "/" + filename)
            # delete folder
            rmdir(sens_folder)
        # # succ folder
        # if path_exists(succ_folder):
        #     for filename in listdir(succ_folder):
        #         remove(succ_folder + "/" + filename)
        #     # delete folder
        #     rmdir(succ_folder)
        # errs folder
        if path_exists(errs_folder):
            for filename in listdir(errs_folder):
                remove(errs_folder + "/" + filename)
            # delete folder
            # rmdir(errs_folder)
        
        # delete / create folders
        if path_exists(delta_tf_folder):
            for filename in listdir(delta_tf_folder):
                remove(delta_tf_folder + "/" + filename)
            # delete folder
            if not(plot_dict["plot_delta"]):
                rmdir(delta_tf_folder)
        elif plot_dict["plot_delta"]:
            mkdir(delta_tf_folder)
        if path_exists(delta_zm_folder):
            for filename in listdir(delta_zm_folder):
                remove(delta_zm_folder + "/" + filename)
            # delete folder
            if not(plot_dict["zoom_deltas"]):
                rmdir(delta_zm_folder)
        elif plot_dict["zoom_deltas"]:
            mkdir(delta_zm_folder)
        if path_exists(full__tf_folder):
            for filename in listdir(full__tf_folder):
                remove(full__tf_folder + "/" + filename)
            # delete folder
            if not(plot_dict["plot_full"]):
                rmdir(full__tf_folder)
        elif plot_dict["plot_full"]:
            mkdir(full__tf_folder)
        if path_exists(full__zm_folder):
            for filename in listdir(full__zm_folder):
                remove(full__zm_folder + "/" + filename)
            # delete folder
            if not(plot_dict["zoom_full"]):
                rmdir(full__zm_folder)
        elif plot_dict["zoom_full"]:
            mkdir(full__zm_folder)  
        # other
        for filename in listdir(file_folder):
            if filename not in \
                ["errs_plots","fail_plots","full","sensitivity","delta","delta_zoom","full_zoom"]:
                remove(file_folder + "/" + filename)
        # # delete folder
        # rmdir(file_folder)
    elif save_data:
        mkdir(file_folder)
        mkdir(fail_folder)
        mkdir(sens_folder)
        # mkdir(succ_folder)
        mkdir(errs_folder)
        #
        if plot_dict["plot_delta"]:
            mkdir(delta_tf_folder)
        if plot_dict["zoom_deltas"]:
            mkdir(delta_zm_folder)
        if plot_dict["plot_full"]:
            mkdir(full__tf_folder)
        if plot_dict["zoom_full"]:
            mkdir(full__zm_folder)
    
    
    # build controller
    repstr += aircraft._build_controller(report=False,save_matrices=False,
        drop_actrs=cut_mine,mrrr=mrrr,mrrc=mrrc,run_freq=True,
        include_stall_derivatives=include_stall_derivatives,
        save_name_end=name_end,save_folder=file_folder[6:]+"/")
    CTC = np.matmul(aircraft.Lin_Model.C.T,aircraft.Lin_Model.C)
    CEC = np.matmul(CTC,np.matmul(E,CTC))
    states = [
        "Vx","Vy","Vz"," p"," q"," r","xf","yf","zf"," \u03C6"," \u03B8"," \u03C8"
    ]
    if save_data:
        # save control info
        with open(file_folder+"/"+"terminal_output.txt","a") as f:
            f.write(repstr)
            f.close()

    if save_data:
        aircraft._save_controller(aircraft.Lin_Model,\
            file_folder[6:]+"/",filename="matrices")

    # create shifting values
    if statistical:
        p_vals = aircraft.rng.normal(loc=0.0, scale=shift[0], size=(num,))
        q_vals = aircraft.rng.normal(loc=0.0, scale=shift[1], size=(num,))
        r_vals = aircraft.rng.normal(loc=0.0, scale=shift[2], size=(num,))
    else:
        num_root = int( np.ceil(float(num)**(1./3.)) )
        num = int(float(num_root)**3.)

        # initialize arrays
        p_vals = np.zeros((num,))
        q_vals = np.zeros((num,))
        r_vals = np.zeros((num,))
        p_lins = np.linspace(-shift[0]*3.,shift[0]*3.,num=num_root)
        q_lins = np.linspace(-shift[1]*3.,shift[1]*3.,num=num_root)
        r_lins = np.linspace(-shift[2]*3.,shift[2]*3.,num=num_root)

        # run through, add values
        for i in range(num_root):
            for j in range(num_root):
                for k in range(num_root):
                    m = i*num_root*num_root + j*num_root + k
                    p_vals[m] = p_lins[i]
                    q_vals[m] = q_lins[j]
                    r_vals[m] = r_lins[k]
    
    # # initialize errored models dict
    # aero_err_percs = {}
    # err_dict = aircraft.aero_model.errors
    # for i in err_dict:
    #     aero_err_percs[i] = {}
    #     for j in err_dict[i]:
    #         if simulation["BIRE"]:
    #             aero_err_percs[i][j] = {}
    #             for k in err_dict[i][j]:
    #                 aero_err_percs[i][j][k] = np.zeros((num,))
    #         else:
    #             aero_err_percs[i][j] = np.zeros((num,))
    # # inertia
    # iner_err_percs = {}
    # err_dict = aircraft.inertia_model.errors
    # for i in err_dict:
    #     if simulation["BIRE"] and i not in ["hx","hy","hz","W"]:
    #         iner_err_percs[i] = {}
    #         for j in err_dict[i]:
    #             iner_err_percs[i][j] = np.zeros((num,))
    #     else:
    #         iner_err_percs[i] = np.zeros((num,))
    # FM
    FM_error_dict = {
        "CL" : FM_error[0],
        "CS" : FM_error[1],
        "CD" : FM_error[2],
        "Cl" : FM_error[3],
        "Cm" : FM_error[4],
        "Cn" : FM_error[5]
    }
    FM_err_percs = np.zeros((6,num))

    # initialize plot params arrays
    disturbs = np.zeros((3,num))
    stables = np.zeros((num,),dtype=bool)
    how_stable = np.zeros((num,))

    errors = {}
    # errors["aero"] = aero_model_errors
    # errors["inertia"] = inertia_model_errors
    errors["1sigma_FM"] = FM_error_dict
    if save_data:
        # save error values
        json_obj = json.dumps(errors,indent=4)
        with open(file_folder+"/errors.json","w") as f:
            f.write(json_obj)
            f.close()
        # # save scale values
        # scales = {}
        # # scales["aero"] = aircraft.aero_scale_errors
        # # scales["inertia"] = aircraft.iner_scale_errors
        # json_obj = json.dumps(scales,indent=4)
        # with open(file_folder+"/error_scales.json","w") as f:
        #     f.write(json_obj)
        #     f.close()
        # # save total acceptable error
        # total = {}
        # # run through dicts / values
        # for i in errors:
        #     total[i] = {}
        #     model = errors[i]
        #     for j in model:
        #         model_comp = model[j]
        #         if isinstance(model_comp,dict):
        #             total[i][j] = {}
        #             for k in model_comp:
        #                 model_subcomp = model_comp[k]
        #                 if isinstance(model_subcomp,dict):
        #                     total[i][j][k] = {}
        #                     for l in model_subcomp:
        #                         total[i][j][k][l] = errors[i][j][k][l]*3.*\
        #                             scales[i][j][k][l]
        #                 else:
        #                     total[i][j][k] = errors[i][j][k]*3.*scales[i][j][k]
        #         elif i != "1sigma_FM":
        #             total[i][j] = errors[i][j]*3.*scales[i][j]
        # total["1sigma_FM"] = FM_error_dict
        # json_obj = json.dumps(total,indent=4)
        # with open(file_folder+"/RoA_acceptable_error.json","w") as f:
        #     f.write(json_obj)
        #     f.close()
    
    # plot setup
    plt.rcdefaults()
    plt.rcParams["font.family"] = "Serif"
    plt.rcParams["font.size"] = 8.0
    plt.rcParams["axes.labelsize"] = 8.0
    plt.rcParams['lines.linewidth'] = 1.0
    plt.rcParams['figure.constrained_layout.use'] = True
    # plt.rcParams["xtick.minor.visible"] = True
    # plt.rcParams["ytick.minor.visible"] = False
    # plt.rcParams["xtick.direction"] = plt.rcParams["ytick.direction"] = "in"
    # plt.rcParams["xtick.bottom"] = plt.rcParams["xtick.top"] = False
    # plt.rcParams["ytick.left"] = plt.rcParams["ytick.right"] = False
    # plt.rcParams["xtick.major.width"] = plt.rcParams["ytick.major.width"] = 0.75
    # plt.rcParams["xtick.minor.width"] = plt.rcParams["ytick.minor.width"] = 0.75
    # plt.rcParams["xtick.major.size"] = plt.rcParams["ytick.major.size"] = 5.0
    # plt.rcParams["xtick.minor.size"] = plt.rcParams["ytick.minor.size"] = 2.5
    # # plt.rcParams["axes.labelpad"] = 2.0
    # # plt.rcParams["font.weight"] = "bold"
    # plt.rcParams["figure.constrained_layout.hspace"] = 0.0
    # # plt.rcParams["figure.subplot.hspace"] = 0.0
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams['figure.dpi'] = 300.0

    # run montecarlo
    print("running Monte Carlo {}...".format(run_name))
    run_str = ""
    r50_str = ""
    ffl, afl = plt.subplots(1,1,layout="constrained")
    ctr = afl.twinx()
    counter = 0
    t_min_stable = 0.0
    it1 = int(t_min_stable/aircraft.dt)
    itf = int(aircraft.tf/aircraft.dt)
    i_true = np.arange(0,itf+1,step=1,dtype=int)
    early_counter = 0
    ############### tracking checker
    t_track_i = np.argwhere(\
        np.linspace(0.,aircraft.tf,int(aircraft.tf/aircraft.dt))\
        <track_check_time)[-1][0]
    r_track = aircraft._get_reference(aircraft.dt*t_track_i)
    rows = [3,4,5,9,10,11]
    r_track[rows] = np.rad2deg(r_track[rows])
    ###################################
    # save upper, lower, and average stable responses
    tnum = int(aircraft.tf/aircraft.dt) + 1
    xupp = np.zeros((aircraft.x_trim_euler.shape[0],tnum)) - 1.0e100
    xlow = xupp*0. + 1.0e100; xavg = xupp*0.
    uupp = np.zeros((aircraft.u_trim.shape[0],tnum)) - 1.0e100
    ulow = uupp*0. + 1.0e100; uavg = uupp*0.
    aupp = np.zeros((4,tnum)) - 1.0e100
    alow = aupp*0. + 1.0e100; aavg = aupp*0.
    ###################################
    for i in range(num):
        # create shift
        pshift = p_vals[i]
        qshift = q_vals[i]
        rshift = r_vals[i]
        # add to shifting array
        dx0[3:6] = [pshift,qshift,rshift]
        # renew model error
        # aircraft.refresh_models_error(aero_model_errors,inertia_model_errors)
        aircraft.refresh_FM_error(FM_error_percs=FM_error)

        # report
        pdeg,qdeg,rdeg = np.rad2deg([pshift,qshift,rshift])
        case_run_text = ("{:>4d} Dp = {:> 9.3f} Dq = {:> 9.3f} " + \
            "Dr = {:> 9.3f}").format(i+1,pdeg,qdeg,rdeg)

        # call run sim
        try:
            xr,ur = aircraft.run_simulation(report_trim=report_trim,
                mrrr=mrrr,delta_x0=dx0,actr_warm_start=actr_warm_start,
                save_matrices=False,report_simulation=False)
            
            crw = [12,13,14,15]
            # print(xr[crw,0],xr[crw,1])
            # print((xr[crw,1]-xr[crw,0])/(aircraft.tarr[1]-aircraft.tarr[0]))
            
            # pull out last state, check if zeros
            x_zero = xr[:,-1]*1.
            # Lin = aircraft.Lin_Model
            if aircraft.tracking:
                dx = x_zero - r_track
            else:
                dx = x_zero - (aircraft.x_trim_euler_deg)
            Dx_norm = np.matmul(dx.T,np.matmul(CEC,dx))

            if aircraft.tracking:
                dx = xr[:,t_track_i] - r_track
                Dx_norm_track = np.matmul(dx.T,np.matmul(CEC,dx))
                Dx_norm = max(Dx_norm_track,Dx_norm)
        except:
            xr,ur = xupp*0.0,uupp*0.0
            xr[0] = xr[0] + 1.0
            x_zero = xr[:,-1]*1.
            # Lin = aircraft.Lin_Model
            if aircraft.tracking:
                dx = x_zero - r_track
            else:
                dx = x_zero - (aircraft.x_trim_euler_deg)
            Dx_norm = 13.0
        
        case_run_text += " |Dx| = {:>9.3f},".format(Dx_norm)
        is_stable = Dx_norm <= Dx_norm_stable_threshold
        
        if is_stable:
            case_run_text += "   Stable"
            case_run_text += " ->"
            # sort
            i_sort = np.argsort([dx[s]/(1./E[s,s])**0.5 for s in \
                range(dx.shape[0])])
            rep_count = 0
            max_rep = 3
            for v in i_sort:
                if v not in mrrr and v < 12 and rep_count < max_rep:
                    case_run_text += " " + states[v]
                    rep_count += 1
            print(case_run_text)
            counter += 1

            # determine "how" stable (time to stabilize)
            dxs = xr - aircraft.x_trim_euler_deg[:,None]
            if aircraft.tracking:
                dxs = xr - r_track[:,None]
            Dxs = np.array([np.matmul(dxs[:,j].T,np.matmul(CEC,dxs[:,j])) \
                    for j in range(dxs.shape[1])])
            i_stable = np.argwhere(Dxs[it1:] <= Dx_norm_stable_threshold)[:,0]
            # print(len(i_stable),i_stable)
            i_fixed = i_stable[i_stable == i_true[itf-len(i_stable)+1:]]
            # print(i_stable.shape,i_stable[0],i_fixed[0])#i_stable[-1])
            # print(i_fixed)
            # print(i_true)
            # print(itf-len(i_stable)+1)
            # print(len(i_true[itf-len(i_stable)+1:]),
            #     i_true[itf-len(i_stable)+1:])
            if i_fixed[0] != i_stable[0]:
                early_counter += 1
            i_stable = i_fixed[0]
            how_stable[i] = float(i_stable)/float(itf - it1)
            # if True and save_data:
            #     # plot failure
            #     names=["p","q","r",r"$\phi$",r"$\theta$",r"$z_f$"]
            #     index=[3,4,5,9,10,8]
            #     for ik,k in enumerate(index):
            #         vals = aircraft.xarr[k,:] - aircraft.x_trim_euler_deg[k]
            #         afl.plot(aircraft.tarr,vals,label=names[ik])
            #     # plot alpha, beta
            #     u = aircraft.xarr[0,:]
            #     v = aircraft.xarr[1,:]
            #     w = aircraft.xarr[2,:]
            #     V = (u**2. + v**2. + w**2.)**0.5
            #     u_eq = aircraft.x_trim_euler_deg[0]
            #     v_eq = aircraft.x_trim_euler_deg[1]
            #     w_eq = aircraft.x_trim_euler_deg[2]
            #     V_eq = (u_eq**2. + v_eq**2. + w_eq**2.)**0.5
            #     a_eq = atan2(w_eq,u_eq)
            #     b_eq = asin(v_eq/V_eq)
            #     a = np.arctan2(w,u) - a_eq
            #     b = np.arcsin(v/V) - b_eq
            #     afl.plot(aircraft.tarr,V - V_eq,label=r"$V$")
            #     afl.plot(aircraft.tarr,np.rad2deg(a),label=r"$\alpha$")
            #     afl.plot(aircraft.tarr,np.rad2deg(b),label=r"$\beta$")
            #     afl.legend(loc="upper center")
            #     names=[r"$\delta_a$",r"$\delta_e^B$",r"$\delta_B$",r"$\tau$"]
            #     index=[0,1,2,3]
            #     clrs = ["#5e67bf","#c730b5","#91ad1f","#915a11"]
            #     for ik,k in enumerate(index):
            #         vals = aircraft.uarr[k,:] - aircraft.u_trim_deg[k]
            #         if k == 3:
            #             vals = vals*100.
            #         ctr.plot(aircraft.tarr,vals,linestyle="-.",c=clrs[ik],label=names[ik])
            #     ctr.legend(loc="lower center")
                
            #     # save plots to folder
            #     plt.savefig(succ_folder + "/" + "succ_{:04d}".format(i+1))
            #     afl.cla()
            #     ctr.cla()
            #
            # save max, min, and avg
            if plot_ul_bounds:
                xavg = xavg + xr
                xupp = np.array([xupp,xr]).max(axis=0)
                xlow = np.array([xlow,xr]).min(axis=0)
                uavg = uavg + ur
                uupp = np.array([uupp,ur]).max(axis=0)
                ulow = np.array([ulow,ur]).min(axis=0)
                ar = aircraft.aerox*1.
                aavg = aavg + ar
                aupp = np.array([aupp,ar]).max(axis=0)
                alow = np.array([alow,ar]).min(axis=0)
        else:
            case_run_text += " Unstable"
            case_run_text += " >>"
            # add in states that are maxed
            for s in range(len(states)):
                if dx[s] > (1./E[s,s])**0.5 and s not in mrrr:
                    case_run_text += " " + states[s]
            # report states in closest to max to furthest from max order
            case_run_text += " ->"
            # sort
            i_sort = np.argsort([dx[s]/(1./E[s,s])**0.5 for s in \
                range(dx.shape[0])])
            for v in i_sort:
                if v not in mrrr and v < 12:
                    case_run_text += " " + states[v]
            print(case_run_text)
            how_stable[i] = 1.0
            if save_data:
                # plot failure
                names=["p","q","r",r"$\phi$",r"$\theta$",r"$z_f$"]
                index=[3,4,5,9,10,8]
                for ik,k in enumerate(index):
                    vals = aircraft.xarr[k,:] - aircraft.x_trim_euler_deg[k]
                    afl.plot(aircraft.tarr,vals,label=names[ik])
                # plot alpha, beta
                u = aircraft.xarr[0,:]
                v = aircraft.xarr[1,:]
                w = aircraft.xarr[2,:]
                V = (u**2. + v**2. + w**2.)**0.5
                u_eq = aircraft.x_trim_euler_deg[0]
                v_eq = aircraft.x_trim_euler_deg[1]
                w_eq = aircraft.x_trim_euler_deg[2]
                V_eq = (u_eq**2. + v_eq**2. + w_eq**2.)**0.5
                a_eq = atan2(w_eq,u_eq)
                b_eq = asin(v_eq/V_eq)
                a = np.arctan2(w,u) - a_eq
                b = np.arcsin(v/V) - b_eq
                afl.plot(aircraft.tarr,V - V_eq,label=r"$V$")
                afl.plot(aircraft.tarr,np.rad2deg(a),label=r"$\alpha$")
                afl.plot(aircraft.tarr,np.rad2deg(b),label=r"$\beta$")
                afl.legend(loc="upper center")
                names=[r"$\delta_a$",r"$\delta_e^B$",r"$\delta_B$",r"$\tau$"]
                index=[0,1,2,3]
                clrs = ["#5e67bf","#c730b5","#91ad1f","#915a11"]
                for ik,k in enumerate(index):
                    vals = aircraft.uarr[k,:] - aircraft.u_trim_deg[k]
                    if k == 3:
                        vals = vals*100.
                    ctr.plot(aircraft.tarr,vals,linestyle="-.",c=clrs[ik],label=names[ik])
                ctr.legend(loc="lower center")
                
                # save plots to folder
                plt.savefig(fail_folder + "/" + "fail_{:04d}".format(i+1))
                afl.cla()
                ctr.cla()
                # plt.show()
        # write down in README file
        run_str += case_run_text + "\n"
        r50_str += case_run_text + "\n"
        # add in errors
        # err_dict = aircraft.aero_model.errors
        # for m in aero_err_percs:
        #     for j in aero_err_percs[m]:
        #         if simulation["BIRE"]:
        #             for k in aero_err_percs[m][j]:
        #                 aero_err_percs[m][j][k][i] = err_dict[m][j][k]*1.
        #         else:
        #             aero_err_percs[m][j][i] = err_dict[m][j]*1.
        # # inertia
        # err_dict = aircraft.inertia_model.errors
        # for m in iner_err_percs:
        #     if simulation["BIRE"] and m not in ["hx","hy","hz","W"]:
        #         for j in iner_err_percs[m]:
        #             iner_err_percs[m][j][i] = err_dict[m][j]
        #     else:
        #         iner_err_percs[m][i] = err_dict[m]
        # FM
        FM_err_percs[:,i] = aircraft.FM_errors*1.
        
        # report every 50
        if (i+1) % 25 == 0:
            cases = "{:> 4d}/{:> 4d} conv, est {:> 4d}/{:> 4d}".format(\
                counter,i+1,int(counter*num/(i+1)),num) +" for "+run_name+"\n"
            run_str += cases
            r50_str += cases #.replace(r"\u0394","D")
            if save_data:
                with open(file_folder+"/"+"terminal_output.txt","a") as f:
                    f.write(r50_str)
                    f.close()
                r50_str = ""
            print(cases,end="")

        
        # re-randomize phases in turbulence model
        if has_turbulence and rerandomize_turbulence:
            aircraft.disturbance_model.rebuild_turbulence_phases()


        
        # store plot params
        disturbs[:,i] = [pdeg,qdeg,rdeg]
        stables[i] = is_stable
    stable_num = "{:> 4d}/{:> 4d} cases stable".format(counter,num)
    run_str += stable_num + "\n"
    plt.rcdefaults()
    plt.close()
    if num % 25 == 0:
        pass
    else:
        print(stable_num)
    print(Dx_report)
    run_str += Dx_report + "\n"
    tc_rep = "{:> 4d} cases tc early was not tc true\n".format(early_counter)
    print(tc_rep)
    run_str += tc_rep + "\n"
    if save_data:
        with open(file_folder+"/"+
            "success_{:>04d}_of_{:>04d}.txt".format(counter,num),
            "a") as f:
            f.write(run_str)
            f.close()
        with open(file_folder+"/"+"terminal_output.txt","a") as f:
            f.write(r50_str)
            f.close()
    print("finished simulating {}...".format(run_name))

    # divide average responses
    if counter > 0:
        xavg = xavg/counter
        uavg = uavg/counter
        aavg = aavg/counter

    # determine stable-coloring tuples
    clrs = []
    mec = []
    ms = []
    zord = []
    dis90 = np.array(rtdst_1sg)*3.
    mss = 3.0
    msu = 3.0
    for k in range(num):
        if stables[k]:
            # shades of green
            # clrs.append( (0.0+how_stable[k],0.5*(1. +how_stable[k]),0.0) )
            # black
            clrs.append( "k" ) # (0.0,0.0,0.0) )
            mec.append( "k" )# "w" ) # (1.0,1.0,1.0) )
            ms.append(mss)
            zord.append( k )
        else:
            clrs.append( "0.6" ) # "w" ) # (1.0,1.0,1.0) ) # white
            mec.append( "0.6" ) # "k" ) # (0.0,0.0,0.0) )
            ms.append(msu)
            zord.append( k+num )
    
    # create plot
    if save_data:
        print("creating plots...")

        print("    average response plots...")
        # set plotted line as average
        aircraft.xarr = xavg*1.
        aircraft.uarr = uavg*1.
        aircraft.aerox = aavg*1.
        aircraft.xarr_upp = xupp*1.
        aircraft.uarr_upp = uupp*1.
        aircraft.aerox_upp = aupp*1.
        aircraft.xarr_low = xlow*1.
        aircraft.uarr_low = ulow*1.
        aircraft.aerox_low = alow*1.
        plot_dict["plotting_directory"] = file_folder + "/"
        plot_dict["plot_upp_and_low"] = plot_ul_bounds
        aircraft.plot_results(**plot_dict)

        # model error plots
        print("    model error plots...")
        #
        savedict = dict(transparent=False,format="pdf",dpi=300.0)
        err_counter = 0
        #
        # legend
        # for some reason this was a problem
        # plt.rcParams["text.usetex"] = True
        n_pts = 5
        lgnd_elms_sp = []
        ts = np.linspace(aircraft.tf/n_pts,aircraft.tf,n_pts)
        lgnd_elms_lbls = []
        # for i in range(n_pts):
        #     flt_stable = (ts[i] - t_min_stable)/(aircraft.tf - t_min_stable)
        #     lgnd_elms_sp.append(
        #         Line2D([0], [0], 
        #             c=(0.0+flt_stable,0.5*(1. + flt_stable),0.0),ls="none",
        #             marker="o")
        #     )
        #     lgnd_elms_lbls.append(
        #         "$t_c \leq${:> 2.0f} s".format(ts[i])
        #     )
        lgnd_elms_sp.append(Line2D([0], [0], 
            c="k",ls="none",mec="k",mew=0.5,ms=mss,marker="o"))
        lgnd_elms_sp.append(Line2D([0], [0],
            c="0.6",ls="none",mec="0.6",mew=0.5,ms=msu,marker="o"))
        #     c="w",ls="none",mec="k",mew=0.5,ms=msu,marker="o"))
        # lgnd_elms_lbls.append("converged")
        # lgnd_elms_lbls.append("unconverged")
        lgnd_elms_lbls.append("$t_c \leq${:> 2.0f} s".format(aircraft.tf))
        lgnd_elms_lbls.append("$t_c>${:> 2.0f} s".format(aircraft.tf))
        #
        # FM
        names = ["CL", "CS", "CD", "Cell", "Cm", "Cn"]
        plots = [[i,j] for i in range(6) for j in range(i,6) if i != j]
        # run through plots
        # aero
        fig, ax = plt.subplots(figsize=(3.25,2.4375),layout="constrained")
        for i,plot in enumerate(plots):
            i0,i1 = plot

            # formulate plot
            if FM_error[i0] or FM_error[i1]:
                ax.grid(which="major",axis="both",c="0.2",ls="-.",lw=0.5 )
                x = FM_err_percs[i0,:]
                y = FM_err_percs[i1,:]
                [ax.plot(x[k],y[k],c=clrs[k],marker="o",mec=mec[k],mew=0.5,ms=ms[k],\
                    zorder=zord[k]) for k in range(num)]
                ax.grid(which="major",lw=0.6,ls="-",c="0.75")
                xlabel = r"$\epsilon_{C_" + names[i0][1:] + "}$"
                ylabel = r"$\epsilon_{C_" + names[i1][1:] + "}$"
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                leg = ax.legend(handles=lgnd_elms_sp,labels=lgnd_elms_lbls)#,
                # loc=(1.0,0.05))
                fig.savefig(errs_folder+"/"+run_name+"_"+\
                    "FM_{:02d}_".format(err_counter)+\
                    names[i0]+"__"+names[i1]+".pdf",**savedict)
                err_counter += 1
                ax.cla()
        plt.close(fig)

        #######################################################################
        plt.rcParams["font.family"] = "Serif"
        plt.rcParams["font.size"] = 8.0
        plt.rcParams["axes.labelsize"] = 8.0
        plt.rcParams['lines.linewidth'] = 1.0
        plt.rcParams["xtick.minor.visible"] = False
        plt.rcParams["ytick.minor.visible"] = False
        plt.rcParams["xtick.direction"] = plt.rcParams["ytick.direction"] = "in"
        plt.rcParams["xtick.bottom"] = plt.rcParams["xtick.top"] = True
        plt.rcParams["ytick.left"] = plt.rcParams["ytick.right"] = True
        plt.rcParams["xtick.major.width"] = plt.rcParams["ytick.major.width"] = 0.75
        plt.rcParams["xtick.major.size"] = plt.rcParams["ytick.major.size"] = 5.0
        plt.rcParams["mathtext.fontset"] = "dejavuserif"
        plt.rcParams['figure.dpi'] = 300.0
        plt.rcParams['figure.constrained_layout.use'] = True
        plt.rcParams['axes.labelpad'] = 0.0
        # plt.rcParams['xtick.major.pad'] = 0.1
        #######################################################################            
        # histograms
        print("    histogram...")
        # errs
        print("        FM errors tc plots...")
        time_to_stable = how_stable*(final_time-t_min_stable) + t_min_stable
        p_argsort = np.argsort(disturbs[0,:])
        q_argsort = np.argsort(disturbs[1,:])
        r_argsort = np.argsort(disturbs[2,:])
        fig, ax = plt.subplots(figsize=(3.25,2.4375),layout="constrained")
        FMname = ["Lift Force","Side-Force","Drag Force",
            "Rolling Moment","Pitching Moment","Yawing Moment"]
        for i,coeff in enumerate(names):
            if FM_error[i]:
                err_sort = np.argsort(FM_err_percs[i,:])
                [ax.plot(FM_err_percs[i,k],time_to_stable[k],"o",\
                    c=clrs[k],mec=mec[k],mew=0.5,ms=ms[k]) for k in err_sort]
                leg = ax.legend(handles=lgnd_elms_sp,labels=lgnd_elms_lbls,\
                    loc=(1.0,0.05),borderpad=0.1,handletextpad=0.0)
                ax.set_xlabel(FMname[i] + " Error " + \
                    r"$\epsilon_{C_" + (i == 3)*"\\" + coeff[1:] + "}$")
                ax.set_ylabel(r"Time to Converge, s")
                ax.grid(which="major",lw=0.6,ls="-",c="0.75")
                fig.savefig(file_folder+"/"+run_name+"_"+coeff+\
                    "_tc.pdf",**savedict)
                ax.cla()

        # p q r
        print("        p,q,r tc plots...")
        [ax.plot(disturbs[0,k],time_to_stable[k],"o",\
            c=clrs[k],mec=mec[k],mew=0.5,ms=ms[k]) for k in p_argsort]
        ax.legend(handles=lgnd_elms_sp,labels=lgnd_elms_lbls,loc=(1.0,0.05),
            borderpad=0.1,handletextpad=0.0)
        ax.set_xlabel( r"Initial Roll Rate $\Delta p_0$, deg/s")
        ax.set_ylabel(r"Time to Converge, s")
        ax.grid(which="major",lw=0.6,ls="-",c="0.75")
        fig.savefig(file_folder+"/"+run_name+"_p_tc.pdf",**savedict)
        ax.cla()
        [ax.plot(disturbs[1,k],time_to_stable[k],"o",\
            c=clrs[k],mec=mec[k],mew=0.5,ms=ms[k]) for k in q_argsort]
        ax.legend(handles=lgnd_elms_sp,labels=lgnd_elms_lbls,loc=(1.0,0.05),
            borderpad=0.1,handletextpad=0.0)
        ax.set_xlabel(r"Initial Pitch Rate $\Delta q_0$, deg/s")
        ax.set_ylabel(r"Time to Converge, s")
        ax.grid(which="major",lw=0.6,ls="-",c="0.75")
        fig.savefig(file_folder+"/"+run_name+"_q_tc.pdf",**savedict)
        ax.cla()
        [ax.plot(disturbs[2,k],time_to_stable[k],"o",\
            c=clrs[k],mec=mec[k],mew=0.5,ms=ms[k]) for k in r_argsort]
        ax.legend(handles=lgnd_elms_sp,labels=lgnd_elms_lbls,loc=(1.0,0.05),
            borderpad=0.1,handletextpad=0.0)
        ax.set_xlabel(  r"Initial Yaw Rate $\Delta r_0$, deg/s")
        ax.set_ylabel(r"Time to Converge, s")
        ax.grid(which="major",lw=0.6,ls="-",c="0.75")
        fig.savefig(file_folder+"/"+run_name+"_r_tc.pdf",**savedict)
        ax.cla()

        # # magnitude plot
        print("        weighted magnitude plot...")
        pqr_E = state_threshold[3:6]
        mag = ( (disturbs[0]/pqr_E[0])**2. + (disturbs[1]/pqr_E[1])**2. \
            + (disturbs[2]/pqr_E[2])**2. )**0.5
        [ax.plot(how_stable[k]*aircraft.tf,mag[k],"o",\
            c=clrs[k],mec=mec[k],mew=0.5,ms=ms[k]) for k in r_argsort]
        ax.legend(handles=lgnd_elms_sp,labels=lgnd_elms_lbls,loc=(1.0,0.05),
            borderpad=0.1,handletextpad=0.0)
        ax.set_ylabel(r"Weighted Initial Disturbance")
        ax.set_xlabel(r"Time to Converge, s")
        ax.grid(which="major",lw=0.6,ls="-",c="0.75")
        fig.savefig(file_folder+"/"+run_name+"_mag_tc.pdf",**savedict)
        plt.close(fig)

        # 3D plot
        print("    pqr 3D plot...")
        # fig = plt.figure(figsize=(3.25,2.4375))#layout="constrained")
        # ax = fig.add_subplot(111,projection='3d')#,layout="constrained")
        fig, ax = plt.subplots(figsize=(3.25,2.4375),#constrained_layout=True,
            subplot_kw={"projection":"3d"})
        # plt.subplots_adjust(wspace=0.0,hspace=0.0)
        max_rate = np.max(np.abs(disturbs))
        ax.set_xlabel( r'$\qquad\qquad$Roll Rate, deg/s',labelpad=0.0) # $\Delta p_0$
        ax.set_ylabel(r'Pitch Rate, deg/s$\qquad\qquad$',labelpad=0.0) # $\Delta q_0$
        ax.set_zlabel(  r'Yaw Rate, deg/s',labelpad=0.0) # $\Delta r_0$
        # plt.tight_layout()
        ax.invert_zaxis()
        ax.view_init(0,0)
        start = r"$\sigma_{1 \, "
        end = r"$^\circ$/s, "
        title  = start + \
            r"p,q,r}$ = $\left[ "+"{:> 5.1f} \,\,\, ".format(degs[0])
        title += "{:> 5.1f} \,\,\, ".format(degs[1])
        title += "{:> 5.1f} ".format(degs[2]) + \
            r" \right]^T$ " + end
        title += " {:> 5.1f}".format(float(counter)/float(num)*100.) + \
            r"\% converged"
        # ax.set_title(title)

        # plot
        [ax.plot(disturbs[0,k],disturbs[1,k],disturbs[2,k],c=clrs[k],\
            marker="o",mec=mec[k],mew=0.5,ms=ms[k]) for k in range(num)] # ,zorder=zord[k]
        # return
        
        frames = 200
        az = np.linspace(0.,360.,num=frames)
        el = 22.5*np.cos(np.linspace(0.,2.*np.pi,num=frames))
        total_frames = frames
        time_to_end = 10.

        # legend
        ax.legend(handles=lgnd_elms_sp,labels=lgnd_elms_lbls,#loc=(1.0,0.05),
            borderpad=0.1,handletextpad=0.0)
        # loc=(0.66,-0.1)
        ax.tick_params(pad=0.0)

        def animate(i):
            # rotate view
            ax.view_init(el[i],az[i])
            return
        ax.view_init(22.5,45.)
        fig.savefig(file_folder+"/"+run_name+"_iso.pdf",**savedict)
        ax.view_init( 90.,-90.)
        fig.savefig(file_folder+"/"+run_name+"_xy.pdf",**savedict)
        ax.view_init(  0., 90.)#, roll=180.)
        fig.savefig(file_folder+"/"+run_name+"_xz.pdf",**savedict)
        ax.view_init(  0.,180.)#, roll=180.)
        fig.savefig(file_folder+"/"+run_name+"_yz.pdf",**savedict)
        # plt.show()

        ## SKIP Till finished with integral adding
        if not(skip_video):
            print("building animation...")
            anim = FuncAnimation(fig, animate,frames=total_frames, 
                interval=int(time_to_end/total_frames*1000.), blit=False)
            # plt.show()
            print("saving mp4 to {}.mp4...".format(run_name))
            anim.save(file_folder+"/"+run_name+"_vid_RoA.mp4",dpi=300.)
            ## SKIP End
            # print("saving gif...")
            # anim.save(aircraft.fldr_prfx + "_" + \
            #     "plots/monte_carlo/"+run_name+".gif",writer="imagemagick",dpi=300.)
        plt.close(fig)
        plt.rcdefaults()
    return


def compare_aero_forces(filename,rtdst_1sg=[5.,5.,5.],
    mrrr=None,mrrc=None,
    actr_warm_start=False,num=20,cut_mine=True,save_data=True,name_end="",
    state_threshold="o", get_aero_FM = False,
    aero_model_errors=[0.,0.,0.,0.],inertia_model_errors=[0.,0.,0.,0.],
    FM_error = [0.,0.,0.,0.,0.,0.],
    statistical=True,turbulence_setting="light",
    final_time=15.0,time_step=0.01,
    include_stall_derivatives=False,
    aircraft_class=Aircraft,
    **plot_dict):
    # pull in json file
    input_vars_type = type(filename)
    # dictionary
    if input_vars_type == dict:
        input_dict = filename
    
    # json file
    elif input_vars_type == str and filename.split(".")[-1] == "json":
        # import json file from file path
        json_string = open(filename).read()
        # save to vals dictionary
        input_dict = json.loads(json_string)

    # get linear nonlinear parameter and initialize aircraft
    simulation = input_dict.get("simulation",{})
    lim_u = True
    lim_du = True
    compressible = True # False # 
    stallable = True # False # 
    simulation = {
        "constant_density" : True,
        "time_step[sec]" : time_step,
        "total_time[sec]" : final_time,
        "integrator" : "odeint",
        "nonlinear_dynamics" : True,
        "use_quaternions" : True,
        #############################
        "limit_input" : lim_u,
        "limit_input_rates" : lim_du,
        "include_compressibility" : compressible,
        "use_Anderson_corrections" : compressible,
        "include_stall" : stallable,
        #############################
        "simulate_uncontrolled" : False,
        "use_fitted_thrust_model" : True,
        "BIRE" : simulation["BIRE"]
    }
    #
    input_dict["simulation"] = simulation
    #
    disturbance = input_dict.get("disturbance",{})
    disturbance["type"] = "von_Karman" # "none" #
    if state_threshold == "o":
        raise TypeError("not implemented")
        Dx_norm_stable_threshold = 10.
    else:
        Dx_norm_stable_threshold = state_threshold
    disturbance["turbulence_intensity"] = turbulence_setting
    input_dict["disturbance"] = disturbance
    #
    aircraft = aircraft_class(input_dict)
    #
    report_trim = False
    # setup perturbation shift
    shift = np.deg2rad(rtdst_1sg)
    #
    quat_shift = [0., 0., 0.] + aircraft.use_quaternions*[0.]
    #
    dx0 = np.array([
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0
    ] + quat_shift + aircraft.order*[0.0,0.0,0.0,0.0])

    # determine mean turbulence state
    if state_threshold == "o":
        raise TypeError("not implemented")
        u_sig = np.abs(aircraft.disturbance_model.Vgu_signal)
        v_sig = np.abs(aircraft.disturbance_model.Vgv_signal)
        w_sig = np.abs(aircraft.disturbance_model.Vgw_signal)
        p_sig = np.abs(np.rad2deg(aircraft.disturbance_model.Wgp_signal))
        q_sig = np.abs(np.rad2deg(aircraft.disturbance_model.Wgq_signal))
        r_sig = np.abs(np.rad2deg(aircraft.disturbance_model.Wgr_signal))
        u_avg = np.std(u_sig)*3.
        v_avg = np.std(v_sig)*3.
        w_avg = np.std(w_sig)*3.
        p_avg = np.std(p_sig)*3.
        q_avg = np.std(q_sig)*3.
        r_avg = np.std(r_sig)*3.
        Dx_add = np.linalg.norm(
            [u_avg, v_avg, w_avg,p_avg, q_avg, r_avg]
        )
        Dx_norm_stable_threshold += Dx_add
    Dx_report = "|Dx| threshold is {:> 10.6}".format(Dx_norm_stable_threshold)
    print(Dx_report)

    # build controller
    aircraft._build_controller(report=False,save_matrices=False,
        drop_actrs=cut_mine,mrrr=mrrr,mrrc=mrrc,
        include_stall_derivatives=include_stall_derivatives,
        run_freq=False)

    # create errored FM
    aircraft.make_FM_error_model(FM_error_percs=FM_error)

    # create errored aero and inertia model
    aircraft.make_errored_models(aero_model_errors,inertia_model_errors)

    # check if failed folder exists
    if aircraft.is_BIRE:
        prename = "bire"
    else:
        prename = "base"
    run_name = prename + "_FM" + name_end
    file_folder = aircraft.fldr_prfx + "_"+"plots/forces_comparison/"+ run_name
    if save_data and path_exists(file_folder):
        # step through and remove every file, then delete folder
        # other
        for filename in listdir(file_folder):
            remove(file_folder + "/" + filename)
        # delete folder
        rmdir(file_folder)

    if save_data:
        aircraft._save_controller(aircraft.Lin_Model,\
            "data/forces_comparison/",\
            filename=run_name)

    # create shifting values
    if statistical:
        p_val = aircraft.rng.normal(loc=0.0, scale=shift[0])
        q_val = aircraft.rng.normal(loc=0.0, scale=shift[1])
        r_val = aircraft.rng.normal(loc=0.0, scale=shift[2])
    else:
        p_val = shift[0]
        q_val = shift[1]
        r_val = shift[2]
    dx0[3:6] = [p_val,q_val,r_val]
    
    # initialize errored models dict
    aero_err_percs = {}
    err_dict = aircraft.aero_model.errors
    for i in err_dict:
        aero_err_percs[i] = {}
        for j in err_dict[i]:
            if simulation["BIRE"]:
                aero_err_percs[i][j] = {}
                for k in err_dict[i][j]:
                    aero_err_percs[i][j][k] = np.zeros((num,))
            else:
                aero_err_percs[i][j] = np.zeros((num,))
    # inertia
    iner_err_percs = {}
    err_dict = aircraft.inertia_model.errors
    for i in err_dict:
        if simulation["BIRE"] and i not in ["hx","hy","hz","W"]:
            iner_err_percs[i] = {}
            for j in err_dict[i]:
                iner_err_percs[i][j] = np.zeros((num,))
        else:
            iner_err_percs[i] = np.zeros((num,))
    # FM
    FM_error_dict = {
        "CL" : FM_error[0],
        "CS" : FM_error[1],
        "CD" : FM_error[2],
        "Cl" : FM_error[3],
        "Cm" : FM_error[4],
        "Cn" : FM_error[5]
    }

    errors = {}
    errors["aero"] = aero_model_errors
    errors["inertia"] = inertia_model_errors
    errors["1sigma_FM"] = FM_error_dict
    if save_data:
        mkdir(file_folder)
        # save error values
        json_obj = json.dumps(errors,indent=4)
        with open(file_folder+"/errors.json","w") as f:
            f.write(json_obj)
            f.close()
        # save scale values
        scales = {}
        scales["aero"] = aircraft.aero_scale_errors
        scales["inertia"] = aircraft.iner_scale_errors
        json_obj = json.dumps(scales,indent=4)
        with open(file_folder+"/error_scales.json","w") as f:
            f.write(json_obj)
            f.close()
        # save total acceptable error
        total = {}
        # run through dicts / values
        for i in errors:
            total[i] = {}
            model = errors[i]
            for j in model:
                model_comp = model[j]
                if isinstance(model_comp,dict):
                    total[i][j] = {}
                    for k in model_comp:
                        model_subcomp = model_comp[k]
                        if isinstance(model_subcomp,dict):
                            total[i][j][k] = {}
                            for l in model_subcomp:
                                total[i][j][k][l] = errors[i][j][k][l]*3.*\
                                    scales[i][j][k][l]
                        else:
                            total[i][j][k] = errors[i][j][k]*3.*scales[i][j][k]
                elif i != "1sigma_FM":
                    total[i][j] = errors[i][j]*3.*scales[i][j]
        total["1sigma_FM"] = FM_error_dict
        json_obj = json.dumps(total,indent=4)
        with open(file_folder+"/RoA_acceptable_error.json","w") as f:
            f.write(json_obj)
            f.close()
    
    # initialize zero-turbulence model
    zero = ZeroTurbulence(disturbance,aircraft.bw,aircraft.V0,aircraft.dt)
    Dist_zero = zero.get_precomputed_disturbance
    vnKa = aircraft.disturbance_model
    Dist_vnKa = vnKa.get_precomputed_disturbance
    del aircraft.get_disturbance
    del aircraft.disturbance_model

    # initialize error models
    true_aero = aircraft.truth_aero_model
    true_iner = aircraft.truth_inertia_model
    errd_aero = aircraft.aero_model
    errd_iner = aircraft.inertia_model
    del aircraft.truth_aero_model, aircraft.truth_inertia_model
    del aircraft.aero_model, aircraft.inertia_model

    # initialize FM error models
    true_FMmd = np.zeros((6,))
    errd_FMmd = aircraft.FM_errors*1.
    del aircraft.FM_errors

    # run cases function
    def get_FM(craft,aero,iner,dist,FMer):
        craft.aero_model = aero
        craft.inertia_model = iner
        craft.get_disturbance = dist
        craft.FM_errors = FMer
        # call run sim
        states,_ = craft.run_simulation(report_trim=report_trim,
            mrrr=mrrr,delta_x0=dx0,actr_warm_start=actr_warm_start,
            save_matrices=False,report_simulation=False)
        # pull out last state, check if zeros
        x_zero = states[:,-1]*1.
        Dx_norm = np.linalg.norm(np.matmul(craft.Lin_Model.C,(x_zero - 
            craft.x_trim_euler_deg) ) )
        is_stable = Dx_norm <= Dx_norm_stable_threshold
        case_run_text = " |Dx| = {:>8.3f},".format(Dx_norm)
        if is_stable: case_run_text += "   Stable"
        else: case_run_text += " Unstable"
        print(case_run_text)
        # convert back to quaternion
        xicnv = [3,4,5] + [12,13,14]*(aircraft.order >=1) + \
            [16,17,18]*(aircraft.order >1)
        states[xicnv,:] = np.deg2rad(states[xicnv,:])
        states = np.insert(states,12,np.zeros((1,states.shape[1])),axis=0)
        for i in range(craft.n_steps):
            states[9:13,i] = euler_2_quat(states[9:12,i])
        # get FM
        FM = np.zeros((6,states.shape[1]))
        # determine forces and moments
        for k in range(states.shape[1]):
            x = states[:,k]*1.
            _,inputs = craft._get_control(0.,x)
            # disturbance model
            V = (x[0]**2. + x[1]**2. + x[2]**2.)**0.5
            Du,Dv,Dw,Dp,Dq,Dr = craft.get_disturbance(craft.tarr[k],V)
            Vg = [Du,Dv,Dw]
            Wg = [Dp,Dq,Dr]
            # get forces and moments
            if get_aero_FM:
                Vu,Vv,Vw = x[0]+Vg[0], x[1]+Vg[1], x[2]+Vg[2]
                a = atan2(Vw,Vu)
                V = (Vu * Vu + Vv * Vv + Vw * Vw)**0.5
                b = asin(Vv/V)
                _,g,_,_,rho,sos = craft.stdatm(-x[8])
                # ##############################
                # g = 32.12780074195162
                # ##############################
                M = V / sos

                # nondimensionalize rates
                pbar = (x[3]+Wg[0])*craft.bw/2./V
                qbar = (x[4]+Wg[1])*craft.cw/2./V
                rbar = (x[5]+Wg[2])*craft.bw/2./V

                # pass in controls state
                ail = inputs[0]
                ele = inputs[1]
                rud = inputs[2]
                thr = inputs[3]

                # use aircraft model
                FM[0,k],FM[1,k],FM[2,k],FM[3,k],FM[4,k],FM[5,k] = \
                    craft.aero_model.aero_results(*[
                    a,b,pbar,qbar,rbar,ail,ele,rud,
                    craft.is_compressible,M,craft.use_anderson,craft.has_stall
                ])
            else:
                FM[0,k],FM[1,k],FM[2,k],FM[3,k],FM[4,k],FM[5,k],_ = \
                    craft._aerodynamics(x,inputs,Vg=Vg,Wg=Wg)

        del craft.aero_model, craft.inertia_model, craft.get_disturbance, \
            craft.FM_errors

        return FM

    # run cases
    print(("dispersions: \u0394p = {:> 8.3f} \u0394q = {:> 8.3f} " + \
            "\u0394r = {:> 8.3f}").format(np.rad2deg(p_val),np.rad2deg(q_val),\
            np.rad2deg(r_val)))
    # without turbulence, without error
    print("running wo dist, wo err...")
    ndnanfm = get_FM(aircraft,true_aero,true_iner,Dist_zero,true_FMmd)

    # with turbulence, without error
    print("running w  dist, wo err...")
    wdnanfm = get_FM(aircraft,true_aero,true_iner,Dist_vnKa,true_FMmd)

    # without turbulence, with aero error
    print("running wo dist, wa err...")
    ndwanfm = get_FM(aircraft,errd_aero,errd_iner,Dist_zero,true_FMmd)

    # with turbulence, with aero error
    print("running w  dist, wa err...")
    wdwanfm = get_FM(aircraft,errd_aero,errd_iner,Dist_vnKa,true_FMmd)

    # without turbulence, with aero error
    print("running wo dist, wFMerr...")
    ndnawfm = get_FM(aircraft,true_aero,true_iner,Dist_zero,errd_FMmd)

    # with turbulence, with aero error
    print("running w  dist, wFMerr...")
    wdnawfm = get_FM(aircraft,true_aero,true_iner,Dist_vnKa,errd_FMmd)

    # plot deltas
    t = aircraft.tarr
    dwdnanfm = wdnanfm - ndnanfm
    dndwanfm = ndwanfm - ndnanfm
    dwdwanfm = wdwanfm - ndnanfm
    diff1 = dwdwanfm - dndwanfm
    diff2 = diff1 - dwdnanfm

    # plot forces
    savedict = dict(transparent=False,format="pdf",dpi=300.0)
    if get_aero_FM:
        ylabels = ["$C_L$","$C_S$","$C_D$","$C_l$","$C_m$","$C_n$"]
        units = [""]*6
    else:
        ylabels = ["$F_x$","$F_y$","$F_z$","$M_x$","$M_y$","$M_z$"]
        units = ["lbf","lbf","lbf","lbf-ft","lbf-ft","lbf-ft"]
        units = ["[" + units[i] + "]" for i in range(len(units))]
    folderpre = file_folder + "/" + prename
    print(np.max(ndnanfm,axis=1))
    for i in range(6):
        plt.plot(t,ndnanfm[i,:],label="no turb  no err")
        plt.plot(t,wdnanfm[i,:],label="wi turb  no err")
        plt.plot(t,ndwanfm[i,:],label="no turb  wa err")
        plt.plot(t,wdwanfm[i,:],label="wi turb  wa err")
        plt.plot(t,ndnawfm[i,:],label="no turb  wFMerr")
        plt.plot(t,wdnawfm[i,:],label="wi turb  wFMerr")
        plt.xlabel("Time [sec]")
        plt.ylabel(ylabels[i] + " " + units[i])
        plt.legend()
        plt.tight_layout()
        lbl = ylabels[i][1] + ylabels[i][3]
        plt.savefig(folderpre + "_{:>02d}_".format(i)+ lbl + ".pdf",**savedict)
        plt.cla()
    for i in range(6):
        plt.plot(t,dwdnanfm[i,:],label="wi t  no e")
        plt.plot(t,dndwanfm[i,:],label="no t  wi e")
        plt.plot(t,dwdwanfm[i,:],label="wi t  wi e")
        plt.plot(t,diff1[i,:],label="wi t  wi e - no t  wi e")
        plt.plot(t,diff2[i,:],label="wi t  wi e - no t  wi e - wi t  no e")
        plt.xlabel("Time [sec]")
        plt.ylabel(ylabels[i] + " [" + units[i] + "]")
        plt.legend()
        plt.tight_layout()
        lbl = ylabels[i][1] + ylabels[i][3]
        plt.savefig(folderpre + "_{:>02d}_sub_".format(6+i) + lbl + \
            ".pdf",**savedict)
        plt.cla()

    return


def make_jacobian(function,inputs,index,step_size=0.001):
        
    # call function to check size
    dfdx = function(*inputs)

    J = np.zeros((dfdx.shape[0],inputs[index].shape[0]))

    # develop Jacobian
    for i in range(J.shape[1]):
        # determine forces with each step change
        base = np.array(inputs[index]) * 1.0
        # plus
        base[i] += step_size
        inputs[index] = base * 1.0
        fun_ip1 = function(*inputs)
        # minus
        base[i] -= 2. * step_size
        inputs[index] = base * 1.0
        fun_im1 = function(*inputs)
        # reset
        base[i] += step_size
        inputs[index] = base * 1.0

        # assign to jacobian
        J[:,i] = (fun_ip1 - fun_im1) / 2. / step_size

    return J

def ord2_jacobian_may_be_incorrect(function,inputs,index,step_size=1e-3):
    
    # call function to check size
    dfdx = function(*inputs)

    J = np.zeros((dfdx.shape[0],inputs[index].shape[0]))

    # develop Jacobian
    for i in range(J.shape[1]):
        # determine forces with each step change
        base = np.array(inputs[index]) * 1.0
        # plus
        base[i] += step_size
        inputs[index] = base * 1.0
        fun_ip1 = function(*inputs)
        # plus2
        base[i] += step_size
        inputs[index] = base * 1.0
        fun_ip2 = function(*inputs)
        # minus
        base[i] -= 3. * step_size
        inputs[index] = base * 1.0
        fun_im1 = function(*inputs)
        # minus2
        base[i] -= 1. * step_size
        inputs[index] = base * 1.0
        fun_im2 = function(*inputs)
        # reset
        base[i] += 2. * step_size
        inputs[index] = base * 1.0

        # assign to jacobian
        J[:,i] = (-fun_ip2 + 8.*fun_ip1 - 8.*fun_im1 + fun_im1) \
            / 12. / step_size

    return J

def rep2D(array, name = "ans", predecimals = 5, decimals = 4,print_format="f",
    final_endline=True,print_report=True,np_array=False):

    printname = "{} = {}[[".format(name,"np.array(" if np_array else "")
    lenname = len(printname)
    return_string = ""
    for i in range(array.shape[0]):
        if i == 0:
            return_string += printname
        else:
            return_string += " "*(lenname - 1) + "["
        
        if print_format == "e":
            num = 4
        else:
            num = 0
        
        for j in range(array.shape[1]):
            return_string += "{:> {}.{}{}}".format(array[i,j],\
                decimals+predecimals+num,decimals,print_format)
            if j != array.shape[1]-1:
                return_string += ","
        if i != array.shape[0]-1:
            return_string += "],\n"
        else:
            return_string += "]]{}\n".format(")" if np_array else "")
    if final_endline:
        return_string += "\n"
    
    if print_report:
        print(return_string,end="")
    return return_string

def frint(name,value,predecimals=4,decimals=16):
    print(name + " {:> {}.{}e}".format(value,decimals+predecimals+4,decimals))

def report_latex(M, name="M", predecimals=4, decimals=4, diag=False, 
    align=False,comquad=False,transpose=False,endln=False,add_tab=True,
    sci=False,print_report=True,eigvecs=False):

    # check if 1D
    if len(M.shape) == 1:
        M = M[:,np.newaxis]

    if align:
        char = "&" # "  " # 
    else:
        char = ""
    if endln:
        end = "\\\\"
    elif comquad:
        end = ", \\quad"
    else:
        end = ""
    if diag:
        bef = "\\operatorname{diag} \\left( "
        aft = " \\right)"
    else:
        bef = ""
        aft = ""
    if transpose:
        tran = "^T"
    else:
        tran = ""
    
    if add_tab:
        t = "    "
    else:
        t = ""

    # print name
    if eigvecs:
        return_string = "{}{} {}= {}\\left\\{{".format(t,name,char,bef)
        return_string += "\n"

        # print matrix
        for j in range(M.shape[0]):
            return_string += "{}{}\\begin{{bmatrix}}\n".format(t,t)
            return_string += "{}{}{}".format(t,t,t)
            for i in range(M[0].shape[0]):
                # real
                if np.round(np.abs(np.real(M[i,j])) %1.,decimals=decimals)\
                    == 0.0:
                    dec = 0
                else:
                    dec = decimals
                return_string += "{:> {}.{}f}".format(np.real(M[i,j]),\
                    predecimals+decimals+1,dec)
                # imag
                if np.round(np.imag(M[i,j]) % 1.,decimals=decimals) == 0.0:
                    dec = 0
                else:
                    dec = decimals
                return_string += "{:>+{}.{}f}j".format(np.imag(M[i,j]),\
                    predecimals+decimals,dec)
                
                if i == M[0].shape[0] - 1:
                    return_string += " \\\\" + "\n"
                else:
                    return_string += " \\\\ "
            
            return_string +="{}{}\\end{{bmatrix}}".format(t,t,aft,tran,end)
            if j != M.shape[0]-1:
                return_string += ","
            if j == int(M.shape[0]/2.):
                return_string += "\\right. \\\\\n\\left."
            else:
                return_string += "\n"
        return_string +="{}\\right\\}}{}{} {}".format(t,aft,tran,end)+"\n"
        if not (endln or comquad):
            return_string += "\n" 
    else:
        return_string = "{}{} {}= {}\\begin{{bmatrix}}".format(t,name,char,bef)
        return_string += "\n"

        # print matrix
        for i in range(M.shape[0]):
            return_string += "{}{}".format(t,t)
            for j in range(M[0].shape[0]):
                if sci:
                    string = "{:> {}.{}e}".format(M[i,j],decimals+7,decimals)
                    pre_exp,post_exp = string.split("e")
                    if int(pre_exp.split(".")[1]) == 0:
                        pre_exp = pre_exp.split(".")[0]
                    print_string = pre_exp 
                    if int(post_exp) != 0:
                        print_string += "e" + "^{"
                        print_string += "{:>+3d}".format(int(post_exp)) + "}"
                    return_string += print_string
                elif np.iscomplexobj(M[i,j]):
                    # real
                    if np.round(np.abs(np.real(M[i,j])) %1.,decimals=decimals)\
                        == 0.0:
                        dec = 0
                    else:
                        dec = decimals
                    return_string += "{:> {}.{}f}".format(np.real(M[i,j]),\
                        predecimals+decimals+1,dec)
                    # imag
                    if np.round(np.imag(M[i,j]) % 1.,decimals=decimals) == 0.0:
                        dec = 0
                    else:
                        dec = decimals
                    return_string += "{:>+{}.{}f}j".format(np.imag(M[i,j]),\
                        predecimals+decimals,dec)
                elif not np.iscomplexobj(M[i,j]) and \
                    np.round(abs(M[i,j] % 1.),decimals=decimals) == 0.0:
                    return_string += "{:> {}.{}f}".format(M[i,j],\
                        predecimals+decimals+1,0)
                else:
                    return_string += "{:> {}.{}f}".format(M[i,j],\
                        predecimals+decimals+1,decimals)
                if j == M[0].shape[0] - 1:
                    return_string += " \\\\" + "\n"
                else:
                    return_string += " & "
        return_string +="{}\\end{{bmatrix}}{}{} {}".format(t,aft,tran,end)+"\n"
        if not (endln or comquad):
            return_string += "\n" 
    
    if print_report:
        print(return_string,end="")
    return return_string  

def report_eigprops(eigs,predecs=3, decs=4, edec=3, add_tab=True,
    n_a=0., print_report=True):
    
    if add_tab:
        t = "    "
    else:
        t = ""
    
    return_string = "eigenvalues table :::\n"

    sumdec = predecs+decs
    sumedec = predecs+edec+1

    for e0 in eigs:
        e1 = np.conj(e0)

        # print eigval
        return_string += "{}{}".format(t,t)
        # real
        if np.round(np.abs(np.real(e0)) %1.,decimals=decs) == 0.0:
            dec = 0
        else:
            dec = decs
        return_string += "${:> {}.{}f}".format(np.real(e0),sumdec+1,dec)
        # imag
        if np.round(np.imag(e0) % 1.,decimals=decs) == 0.0:
            dec = 0
        else:
            dec = decs
        return_string += "{:>+{}.{}f}j$".format(np.imag(e0),sumdec,dec)

        # print mode
        return_string += " & mode & "

        # print sigma
        sigma = -np.real(e0)
        return_string += "{:> {}.{}f} & ".format(sigma,sumedec,edec)

        # print wn
        if np.abs(np.imag(e0)) > 1e-12:
            wd = np.abs(np.imag(e0))
            wn = (sigma**2. + wd**2.)**0.5
            return_string += "{:> {}.{}f} & ".format(wn,sumedec,edec)
            zt = sigma/wn
            return_string += "{:> {}.{}f} & ".format(zt,sumedec,edec)
            CAP = wn**2./n_a
            return_string += "{:> {}.{}f} & ".format(CAP,sumedec,edec)
        else:
            return_string += "{:^{}s} & ".format("-",sumedec)
            return_string += "{:^{}s} & ".format("-",sumedec)
            return_string += "{:^{}s} & ".format("-",sumedec)
        
        # tdbl
        if sigma != 0.0 and np.abs(np.imag(e0)) < 1e-12:
            tdbl = -np.log(2)/sigma
            if tdbl > 0:
                return_string+="{:> {}.{}f} & ".format(tdbl,sumedec+1,edec)
            else:
                return_string += "{:^{}s} & ".format("-",sumedec+1)
            tc = 1/sigma
            if tc > 0:
                return_string+="{:> {}.{}f} & ".format(tc,sumedec,edec)
            else:
                return_string += "{:^{}s} & ".format("-",sumedec)
        else:
            return_string += "{:^{}s} & ".format("-",sumedec+1)
            return_string += "{:^{}s} & ".format("-",sumedec)

        # handling quality level
        return_string += "Level  \\\\"
        # add newline
        return_string += "\n"
    # add newline
    return_string += "\n"
    
    if print_report:
        print(return_string,end="")
    return return_string 


if __name__ == "__main__":

    # filenames 
    base_file = "base_fs_in.json"
    bire_file = "bire_fs_in.json"

    # read in json to ensure no file changes while running
    base_dict = json.loads( open(base_file).read() )
    bire_dict = json.loads( open(bire_file).read() )

    plot_vars = {
        "show" : False,
        "plot_full" : True,
        "plot_delta" : True,
        "zoom_deltas" : True,
        # "zoom_fraction" : 0.05,
        "zoom_fraction" : 2./15.,
        "transparent" : False,
        "format" : "pdf"
    }

    # bire aero err dict
    bire_errs = { # 3-sig bounds written at end of line
        "CL" : {
            "0"  : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.1600}, #z+-0.4?
            "a"  : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.0500}, #z(+0.2,-0.15)
            "b"  : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "p"  : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "q"  : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "r"  : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "da" : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "de" : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  }
        },
        "CS" : {
            "0"  : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.0230},#z(+0.069,-0.097)
            "a"  : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "b"  : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "p"  : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "Lp" : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "q"  : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "r"  : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "da" : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "de" : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  }
        },
        "CD" : {
            "0"   : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "L"   : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "L2"  : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "S"   : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "S2"  : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "p"   : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "Sp"  : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "q"   : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "Lq"  : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "L2q" : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "r"   : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "Sr"  : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "da"  : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "Sda" : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "de"  : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "Lde" : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "de2" : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  }
        },
        "Cl" : {
            "0"  : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.0240}, #z(+0.073,-0.097)
            "a"  : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "b"  : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "p"  : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "q"  : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "r"  : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "Lr" : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "da" : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "de" : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  }
        },
        "Cm" : {
            "0"  : {"A":0.0600,"w":0.25  ,"phi":0.1500,"z":0.25  },#A(+0.2,-0.2),p(+0.5,-0.5)
            "a"  : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "b"  : {"A":0.25  ,"w":0.25  ,"phi":0.1000,"z":0.25  },
            "p"  : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "q"  : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "r"  : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "da" : {"A":0.25  ,"w":0.25  ,"phi":0.25  ,"z":0.25  },
            "de" : {"A":0.0333,"w":0.25  ,"phi":0.0667,"z":0.25  }
        },
        "Cn" : {
            "0"   : {"A":0.15  ,"w":0.15  ,"phi":0.0067,"z":0.0002},
            #(z<=-0.0067*p+0.00033)(z>=-0.04*p-0.002)
            "a"   : {"A":0.0333,"w":0.15  ,"phi":0.0033,"z":0.0025},
            #(z<=0.5*p+0.025)(z>=0.5*p-0.025)
            "b"   : {"A":0.0067,"w":0.15  ,"phi":0.0600,"z":0.0067}, #(z<=-0.8A),p(+0.2,-0.2)
            "p"   : {"A":0.15  ,"w":0.15  ,"phi":0.15  ,"z":0.15  },
            "Lp"  : {"A":0.15  ,"w":0.15  ,"phi":0.15  ,"z":0.15  },
            "q"   : {"A":0.15  ,"w":0.15  ,"phi":0.15  ,"z":0.15  },
            "r"   : {"A":0.15  ,"w":0.15  ,"phi":0.15  ,"z":0.15  },
            "da"  : {"A":0.15  ,"w":0.15  ,"phi":0.15  ,"z":0.15  },
            "Lda" : {"A":0.15  ,"w":0.15  ,"phi":0.15  ,"z":0.15  },
            "de"  : {"A":0.0800,"w":0.15  ,"phi":0.0300,"z":0.0333} #z(+0.18,-0.2)
            # linear relationship between errors in Cn,bA and Cn,bz.
            # Cn,bz ~= -1.0 * Cn,bA + 0.15
            # linear relationship bounds between errors in Cn,dep and Cn,dez
            # Cn,dep <= 0.6 * Cn,dez + 0.3
            # Cn,dep >= 0.5 * Cn,dez - 0.2
        }
    }
    # bire inertia
    bire_iner = {
        "Ixx" : {"A":0.25  ,"w":0.25  ,"p":0.25  ,"z":0.25  },
        "Iyy" : {"A":0.25  ,"w":0.25  ,"p":0.25  ,"z":0.25  },
        "Izz" : {"A":0.25  ,"w":0.25  ,"p":0.25  ,"z":0.25  },
        "Ixy" : {"A":0.25  ,"w":0.25  ,"p":0.25  ,"z":0.25  },
        "Ixz" : {"A":0.25  ,"w":0.25  ,"p":0.25  ,"z":0.25  },
        "Iyz" : {"A":0.25  ,"w":0.25  ,"p":0.25  ,"z":0.25  },
        "hx" : 0.25  ,
        "hy" : 0.25  ,
        "hz" : 0.25  ,
        "W" : 0.0667
    }
    # base make f16 aero err dict
    base_errs = {
        "CL" : {
            "0"  : 0.25 ,"a"  : 0.1  ,"q"  : 0.25 ,"de" : 0.25 # a -0.1,+?(all good)
        },
        "CS" : {
            "b"  : 0.25 ,"p"  : 0.25 ,"Lp" : 0.25 ,"r" : 0.25 ,
            "da" : 0.25 ,"dr" : 0.25 
        },
        "CD" : {
            "0"   : 0.25 ,"L"   : 0.25 ,"L2"  : 0.25 ,"S2"  : 0.25 ,
            "Sp"  : 0.25 ,"q"   : 0.25 ,"Lq"  : 0.25 ,"L2q" : 0.25 ,
            "Sr"  : 0.25 ,
            "Sda" : 0.25 ,"de"  : 0.25 ,"Lde" : 0.25 ,"de2" : 0.25 ,"Sdr" : 0.25
        },
        "Cl" : {
            "b" : 0.25 ,
            "p"  : 0.25 ,"r" : 0.25 ,"Lr" : 0.25 ,
            "da" : 0.25 ,"dr" : 0.25 
        },
        "Cm" : {
            "0"  : 0.25 ,"a"  : 0.25 ,"q"  : 0.25 ,"de" : 0.25 
        },
        "Cn" : {
            "b" : 0.25 ,
            "p"  : 0.25 ,"Lp"  : 0.25 ,"r"  : 0.25 ,
            "da" : 0.25 ,"Lda" : 0.25 ,"dr" : 0.25 
        }
    }
    # base inertia
    base_iner = {
        "Ixx" : 0.25 ,
        "Iyy" : 0.25 ,
        "Izz" : 0.25 ,
        "Ixy" : 0.25 ,
        "Ixz" : 0.25 ,
        "Iyz" : 0.25 ,
        "hx" : 0.25 ,
        "hy" : 0.25 ,
        "hz" : 0.25 ,
        "W" : 0.125 # +-0.125
    }
    # bire FM
    bire_FM_errs = [
        0.0700, # CL +0.50,-0.24 ## SCT
        0.25  , # CS
        0.1200, # CD +-0.4
        0.25  , # Cl
        0.25  , # Cm
        0.25   # Cn
    ]
    # base FM
    base_FM_errs = [
        0.0800, # CL +0.6,-0.25
        0.25  , # CS
        0.1200, # CD +-0.4
        0.25  , # Cl
        0.25  , # Cm
        0.25   # Cn
    ]
    # make zeros if desired
    if False:
        var = "Cn"
        svar = ["all"] # ["0bde"] # ["all"] # 
        for i in bire_errs:
            for j in bire_errs[i]: # []:# ["CL","CS","CD","Cl","Cm"]: #
                if True:#i not in [var]:# or \
                    # i == var and j not in svar:#,"b","de"]: # "0","a","b","de"]: # 
                    # "p","Lp","q","r","da","Lda"
                    for k in bire_errs[i][j]:
                        bire_errs[i][j][k] = 0.
        for i in bire_iner:
            if i not in ["hx","hy","hz","W"]:
                for j in bire_iner[i]:
                    bire_iner[i][j] = 0.
            else:
                bire_iner[i] = 0.
        for i in base_errs:
            if i not in []: # ["Cn"]: # ["CL","CS","CD","Cl","Cm"]: # 
                for j in base_errs[i]:
                    base_errs[i][j] = 0.
        for i in base_iner:
            base_iner[i] = 0.
        # FM
        axis = 0
        names = ["CL","CS","CD","Cl","Cm","Cn"]
        name = names[axis]
        for i in range(len(bire_FM_errs)):
            if i not in [axis]: # [0,1,2,3,4,5]
                bire_FM_errs[i] = 0.
        for i in range(len(base_FM_errs)):
            if i not in [axis]: # [0,1,2,3,4,5]
                base_FM_errs[i] = 0.
        #
        print(bire_FM_errs)
        print(base_FM_errs)


    # acceptable threshold values based on intensity
    # lit 25.0
    # mod 64.05489802098029
    # sev 147.98352457025243
    # 1  0.9485557794765515
    # 2  31.030580033403417
    # 3  57.03736141921409
    # 4  80.34850249536807
    # 5  154.93788846269098
    # 6  213.9917005470033

    flight_conditions = {
        "T1" : { "m" : 0.2 , "h" :  1000., "V" : 222., "Re" : 15641000. },
        "T2" : { "m" : 0.19, "h" : 15000., "V" : 201., "Re" :  9919000. },
        "C1" : { "m" : 0.8 , "h" :  1000., "V" : 890., "Re" : 62563000. },
        "C2" : { "m" : 0.6 , "h" : 15000., "V" : 634., "Re" : 31324000. },
        "C3" : { "m" : 0.8 , "h" : 30000., "V" : 796., "Re" : 25828000. }
    }
    f1 = "C2" # 
    f2 = "C3"
    state_threshold = [
        10., 15., 15.,
        20., 10., 10., # 
        1., 1., 50., # 
        25., 10., 1., # 
        5., 5., 5., 0.05
    ]

    run_base = {
        "actr_warm_start" : False,
        "num" : 1000,
        "final_time" : 15., # 120., # 
        "time_step" : 0.01,
        "initial_mach" : flight_conditions[f1]["m"]*1.,
        "initial_altitude" : flight_conditions[f1]["h"]*1.,
        "trim_bank" : 0.0, # 75.5224878, # 78.463041, # 80.4059318, # 60.0, # 
        "trim_climb" : 0.0,
        "start_climbing" : False,
        "end_gs_climbing" : False,
        "final_mach" : flight_conditions[f1]["m"]*1., # f2]["m"]*1., # 
        "final_altitude" : flight_conditions[f1]["h"]*1., # f2]["h"]*1., # 
        "t_gain_schedule" : 0.1, # 90., # 
        "gain_steps" : 30,
        "cut_mine" : True,
        "save_data" : True,
        "statistical" : True,
        "has_turbulence" : False, # True, # 
        "turbulence_setting" : "light", # "moderate", # "severe", # 
        "has_model_error" : False, # True, # 
        "aero_model_errors" : base_errs,
        "inertia_model_errors" : base_iner,
        "FM_errors" : base_FM_errs,
        "state_threshold" : state_threshold, # 64.0, # 
        "random_seed" : 13,
        "turbulence_random_seed" : 14, # 13, # 
        "error_random_seed" : 15, # 13, # 
        "rerandomize_turbulence" : True,
        "mrrr" : [6,7,11], # 0,1,2,8,9,10,
        # "mrrc" : [2,3], # [3], # [2], # 
        "get_aero_FM" : True,
        "include_stall_derivatives" : False, # True, # 
        "skip_simulation" : False, # True, # 
        "skip_video" : True, # False, # 
        "name_end" : "_" + f1 + "_BK_3"#4_wSd" # _1e1pqr" #+ "_" + name
        # 4 -- incr wt on tau, decr wt on da,de
        # 5 -- decr wt on da
    }
    run_bire = {**run_base}
    run_bire["aero_model_errors"] = bire_errs
    run_bire["inertia_model_errors"] = bire_iner
    run_bire["FM_errors"] = bire_FM_errs

    # #####
    # # # for running in unreal sim
    # di = [90.0,10.0,2.5]
    # # di = [90.0,5.0,2.5]
    # # plot_vars["format"] = "png"
    # bire_dict["initial"]["trim_guess"] = {}
    # bire_dict["initial"]["trim_guess"]["BIRE[deg]"] = 1.0
    # bire_dict["initial"]["trim_guess"]["elevator[deg]"] = 25.0
    # # bire_dict["aircraft"]["CG_shift[ft]"] = [1.0,0.0,0.0]
    # bire_dict["simulation"]["include_compressibility"] = True
    # bire_dict["simulation"]["use_Anderson_corrections"] = True
    # bire_dict["simulation"]["use_fitted_thrust_model"] = False
    # run_bire["skip_simulation"] = True
    # run_bire["save_data"] = False
    # run_bire["mrrc"] = [2]
    # run_bire["trim_bank"] = 30.0
    # if run_bire["trim_bank"] == 30.0:
    #     bire_dict["controller"]["LQR"] = {
    #         # # # 
    #         "note" : "_current_sctA", # also sctC
    #         "Q" : [1.0e-5, 1.0e-6, 5.0e-6, # ### hs
    #             1.5e-2, 1.0e+1, 2.0e+0, # 
    #             0.0e+0, 0.0e+0, 1.0e-6, # 
    #             2.0e-3, 5.0e-3, 0.0e+0], # 
    #         "Q1a" : [0.0e0, 0.0e0, 0.0e0, 0.0e0],
    #         "Q2a" : [0.0e0, 0.0e0, 0.0e0, 0.0e0],
    #         "R" : [1.0e+0, 1.0e+0, 1.0e+0, 1.0e+0] # 
    #         # # # #
    #         # "note" : "_current_sctB",
    #         # "Q" : [5.0e-5, 1.0e-6, 1.0e-5, # ### aej
    #         #     5.0e-3, 1.0e+1, 1.0e+0, # 
    #         #     0.0e+0, 0.0e+0, 2.0e-7, # 
    #         #     1.0e-3, 5.0e-3, 0.0e+0], # 
    #         # "Q1a" : [0.0e0, 0.0e0, 0.0e0, 0.0e0],
    #         # "Q2a" : [0.0e0, 0.0e0, 0.0e0, 0.0e0],
    #         # "R" : [1.0e+0, 5.0e+0, 2.0e+0, 1.0e+0] # 
    #         # # # #
    #     }
    #     run_base["name_end"] = run_bire["name_end"] = "_" + f1 + "_BK_aaab" # _aen" # _hs" # 
    # run_bire["num"] = 1
    # run_bire.pop("initial_mach")
    # run_bire.pop("final_mach")
    # run_bire["initial_velocity"] = 634.0
    # run_single_simulation(bire_dict,rtdst_1sg=di,**run_bire,**plot_vars)
    # quit()
    # #####

    bire_dict["controller"]["LQR"] = {
        "note" : "_current",
        "Q" : [1.0e-6, 1.0e-6, 1.0e-6, # ### BK_3
            1.0e0, 1.0e0, 1.0e0,
            0.0, 0.0, 1.0e-6, 
            1.0e0, 1.0e0, 0.0],
        "Q1a" : [0.0e0, 0.0e0, 0.0e0, 0.0e0],
        "Q2a" : [0.0, 0.0, 0.0, 0.0],
        "R" : [5.0e0, 5.0e0, 5.0e0, 5.0e-2]
        # # # #
        # "note" : "_current_sctA",
        # "Q" : [1.0e-5, 1.0e-6, 5.0e-6, # ### hs
        #        1.5e-2, 1.0e+1, 2.0e+0, # 
        #        0.0e+0, 0.0e+0, 1.0e-6, # 
        #        2.0e-3, 5.0e-3, 0.0e+0], # 
        # "Q1a" : [0.0e0, 0.0e0, 0.0e0, 0.0e0],
        # "Q2a" : [0.0e0, 0.0e0, 0.0e0, 0.0e0],
        # "R" : [1.0e+0, 1.0e+0, 1.0e+0, 1.0e+0] # 
        # # # #
        # "note" : "_current_sctB",
        # "Q" : [1.0e-5, 5.0e-7, 2.0e-6, # ### aft 
        #        5.0e-3, 1.0e+1, 1.0e+0, # 
        #        0.0e+0, 0.0e+0, 2.0e-7, # 
        #        1.0e-3, 2.0e-3, 0.0e+0], # 
        # "Q1a" : [0.0e0, 0.0e0, 0.0e0, 0.0e0],
        # "Q2a" : [0.0e0, 0.0e0, 0.0e0, 0.0e0],
        # "R" : [1.0e+0, 5.0e+0, 2.0e+0, 1.0e+0] # 
        # # # #
    }
    base_dict["controller"]["LQR"] = {**bire_dict["controller"]["LQR"]}
    # run_bire["FM_errors"][0] = 0.03
    # run_bire["FM_errors"][2] = 0.1
    run_base["name_end"] = run_bire["name_end"] = "_" + f1 + "_BK_3" # _hs" # 
    run_base["has_turbulence"] = run_bire["has_turbulence"] = False # True # 
    run_base["has_model_error"] = run_bire["has_model_error"] = False # True # 

    if True:
        True
        # SCTA2 controller iters
        # aaac -  acv - 970 - ~___ wD -- run
        # aaab -   hs - 986 - ~___ wD -- run
        # aaaa - BK_3 -~644 - ~___ wD -- run

    if True:
        a = 3

        # SCTA controller iters
        # agz -      - ___ - ~___ wD -- 
        # agy -      - ___ - ~___ wD -- 
        # agx -      - ___ - ~___ wD -- 
        # agw -      - ___ - ~___ wD -- 
        # agv -      - ___ - ~___ wD -- 
        # agu -      - ___ - ~___ wD -- 
        # agt -      - ___ - ~___ wD -- 
        # ags -  aft - 915 - ~___ wD -- decr tau from 1e0 to 5e-1
        # agr -  aft - 915 - ~___ wD -- incr tau from 1e0 to 2e0
        # agq -  aft - 907 - ~___ wD -- decr dB from 2e0 to 1e0
        # agp -  aft - 910 - ~___ wD -- incr dB from 2e0 to 5e0
        # ago -  aft - 888 - ~___ wD -- decr de from 5e0 to 2e0
        # agn -  aft - 895 - ~___ wD -- incr de from 5e0 to 1e1
        # agm -  aft - 890 - ~___ wD -- decr da from 1e0 to 5e-1
        # agl -  aft - 894 - ~___ wD -- incr da from 1e0 to 2e0
        # agk -  aft - 915 - ~___ wD -- decr theta from 2e-3 to 1e-3
        # agj -  aft - 912 - ~___ wD -- decr phi from 1e-3 to 5e-4
        # agi -  aft - 908 - ~___ wD -- incr phi from 1e-3 to 2e-3
        # agh -  aft - 913 - ~___ wD -- decr r from 1e0 to 5e-1
        # agg -  aft - 912 - ~___ wD -- incr r from 1e0 to 2e0
        # agf -  aft - 906 - ~___ wD -- decr q from 1e1 to 5e0
        # age -  aft - 904 - ~___ wD -- incr q from 1e1 to 2e1
        # agd -  aft - 913 - ~___ wD -- decr p from 5e-3 to 2e-3
        # agc -  aft - 900 - ~___ wD -- incr p from 5e-3 to 1e-2
        # agb -  aft - 914 - ~___ wD -- decr Vz from 2e-6 to 1e-6
        # aga -  aft - 913 - ~___ wD -- incr Vz from 2e-6 to 5e-6
        # afz -  aft - 914 - ~___ wD -- decr Vy from 5e-7 to 2e-7
        # afy -  aft - 913 - ~___ wD -- incr Vy from 5e-7 to 1e-6
        # afx -  aft - 914 - ~___ wD -- decr Vx from 1e-5 to 5e-6
        # afw -  aft - 912 - ~___ wD -- incr Vx from 1e-5 to 2e-5
        # afv -  aft - 893 - ~___ wD -- decr zf from 2e-7 to 1e-7
        # afu -  aft - 900 - ~___ wD -- incr zf from 2e-7 to 5e-7
        # aft -  aez - 915 - ~___ wD -- decr theta from 5e-3 to 2e-3 ##########
        # afs -  aez - 913 - ~___ wD -- incr theta from 5e-3 to 1e-2
        # afr -  aez - 912 - ~___ wD -- decr phi from 1e-3 to 5e-4
        # afq -  aez - 908 - ~___ wD -- incr phi from 1e-3 to 2e-3
        # afp -  aez - 912 - ~___ wD -- decr r from 1e0 to 5e-1
        # afo -  aez - 912 - ~___ wD -- incr r from 1e0 to 2e0
        # afn -  aez - 906 - ~___ wD -- decr q from 1e1 to 5e0
        # afm -  aez - 904 - ~___ wD -- incr q from 1e1 to 2e1
        # afl -  aez - 914 - ~___ wD -- decr p from 5e-3 to 2e-3
        # afk -  aez - 900 - ~___ wD -- incr p from 5e-3 to 1e-2
        # afj -  aez - 879 - ~___ wD -- decr ctrl from 1e0,5e0,2e0,1e0 to 5e-1,2e0,1e0,5e-1
        # afi -  aez - 854 - ~___ wD -- incr ctrl from 1e0,5e0,2e0,1e0 to 2e0,1e1,4e0,2e0
        # afh -  aez - 914 - ~___ wD -- decr tau from 1e0 to 5e-1
        # afg -  aez - 915 - ~___ wD -- incr tau from 1e0 to 2e0
        # aff -  aez - 907 - ~___ wD -- decr dB from 2e0 to 1e0
        # afe -  aez - 910 - ~___ wD -- incr dB from 2e0 to 5e0
        # afd -  aez - 888 - ~___ wD -- decr de from 5e0 to 2e0
        # afc -  aez - 895 - ~___ wD -- incr de from 5e0 to 1e1
        # afb -  aez - 890 - ~___ wD -- decr da from 1e0 to 5e-1
        # afa -  aez - 894 - ~___ wD -- incr da from 1e0 to 2e0
        # aez -  acv - 914 - ~___ wD -- decr Vx,Vz from 2e-5,5e-6 to 1e-5,2e-6 ###
        # aey -  acv - 912 - ~___ wD -- decr Vz from 5e-6 to 2e-6
        # aex -  acv - 909 - ~___ wD -- incr Vz from 5e-6 to 1e-5
        # aew -  acv - 911 - ~___ wD -- decr Vy from 5e-7 to 2e-7
        # aev -  acv - 908 - ~___ wD -- incr Vy from 5e-7 to 1e-6
        # aeu -  acv - 912 - ~___ wD -- decr Vx from 2e-5 to 1e-5
        # aet -  acv - 909 - ~___ wD -- incr Vx from 2e-5 to 5e-5
        # aes -  aaz - 825 - ~___ wD -- rerun
        # aer -  abh - 848 - ~___ wD -- rerun
        # aeq -  abo - 857 - ~___ wD -- rerun
        # aep -  aca - 860 - ~___ wD -- rerun
        # aeo -  acd - 894 - ~___ wD -- rerun
        # aen -  ack - 897 - ~___ wD -- rerun
        # aem -  acv - 910 - ~___ wD -- rerun
        # ael -  aej - 905 - ~___ wD -- rerun
        #                               Fixed random seed, 13
        # aem -  aej - 919 - ~___ wD -- rerun #################################
        # ael -  acv - 894 - ~___ wD -- rerun
        # aek -  acv - 898 - ~___ wD -- decr vels from 2e-5,5e-7,5e-6 to 1e-5,2e-7,2e-6
        # aej -  acv - 903 - ~___ wD -- incr vels from 2e-5,5e-7,5e-6 to 5e-5,1e-6,1e-5
        # aei -  acv - 897 - ~___ wD -- decr rtes from 5e-3,1e1,1e0 to 2e-3,5e0,5e-1
        # aeh -  acv - 881 - ~___ wD -- incr rtes from 5e-3,1e1,1e0 to 1e-2,2e1,2e0
        # aeg -  acv - 832 - ~___ wD -- decr ornt from 1e-3,5e-3 to 5e-4,2e-3
        # aef -  acv - 892 - ~___ wD -- incr ornt from 1e-3,5e-3 to 2e-3,1e-2
        # aee -  acv - 865 - ~___ wD -- decr ctrl from 1e0,5e0,2e0,1e0 to 5e-1,2e0,1e0,5e-1
        # aed -  acv - 886 - ~___ wD -- incr ctrl from 1e0,5e0,2e0,1e0 to 2e0,1e1,4e0,2e0
        # aec -  adn - 889 - ~___ wD -- rerun
        # aeb -  acv - 950 - ~___ wD -- smaller q, with error (SLF error!!!)
        # aea -  acv - 989 - ~___ wD -- smaller q bounds
        # adz -  ado - 889 - ~___ wD -- rerun
        # ady -  acv - 897 - ~___ wD -- incr,decr Vy,Vz from 5e-7,5e-6 to 1e-6,2e-6
        # adx -  ack - 893 - ~___ wD -- rerun
        # adw -  acv - 904 - ~___ wD -- decr tau from 1e0 to 5e-1
        # adv -  acv - 901 - ~___ wD -- incr tau from 1e0 to 2e0
        # adu -  acv - 887 - ~___ wD -- decr dB from 2e0 to 1e0
        # adt -  acv - 906 - ~___ wD -- incr dB from 2e0 to 5e0
        # ads -  acv - 892 - ~___ wD -- decr de from 5e0 to 2e0
        # adr -  acv - 893 - ~___ wD -- incr de from 5e0 to 1e1
        # adq -  acv - 879 - ~___ wD -- decr da from 1e0 to 5e-1
        # adp -  acv - 868 - ~___ wD -- incr da from 1e0 to 2e0
        # ado -  acv - 917 - ~___ wD -- decr Vz from 5e-6 to 2e-6
        # adn -  acv - 910 - ~___ wD -- incr Vz from 5e-6 to 1e-5
        # adm -  acv - 904 - ~___ wD -- decr Vy from 5e-7 to 2e-7
        # adl -  acv - 914 - ~___ wD -- incr Vy from 5e-7 to 1e-6
        # adk -  acv - 910 - ~___ wD -- decr Vx from 2e-5 to 1e-5
        # adj -  acv - 899 - ~___ wD -- incr Vx from 2e-5 to 5e-5
        # adi -  acv - 901 - ~___ wD -- rerun
        # adh -  acv - 900 - ~___ wD -- decr zf from 2e-7 to 1e-7
        # adg -  acv - 899 - ~___ wD -- decr r from 1e0 to 5e-1
        # adf -  acv - 902 - ~___ wD -- incr r from 1e0 to 2e0
        # ade -  acv - 893 - ~___ wD -- decr q from 1e1 to 5e0
        # add -  acv - 897 - ~___ wD -- incr q from 1e1 to 2e1
        # adc -  acv - 906 - ~___ wD -- decr p from 5e-3 to 2e-3
        # adb -  acv - 895 - ~___ wD -- incr p from 5e-3 to 1e-2
        # ada -  ack - 865 - ~___ wD -- decr theta from 5e-3 to 2e-3
        # acz -  ack - 878 - ~___ wD -- incr theta from 5e-3 to 1e-2
        # acy -  ack - 887 - ~___ wD -- decr phi from 1e-3 to 5e-4
        # acx -  ack - 895 - ~___ wD -- incr phi from 1e-3 to 2e-3
        # acw -  ack - 877 - ~___ wD -- incr dB from 2e0 to 5e0
        # acv -  ack - 915 - ~___ wD -- decr zf from 5e-7 to 2e-7 #############
        # acu -  ack - 886 - ~___ wD -- incr zf from 5e-7 to 1e-6
        # act -  ack - 868 - ~___ wD -- decr Vz from 5e-6 to 2e-6
        # acs -  ack - 896 - ~___ wD -- incr Vz from 5e-6 to 1e-5
        # acr -  ack - 889 - ~___ wD -- decr Vy from 5e-7 to 2e-7
        # acq -  ack - 891 - ~___ wD -- incr Vy from 5e-7 to 1e-6
        # acp -  ack - 896 - ~___ wD -- decr Vx from 2e-5 to 1e-5
        # aco -  ack - 881 - ~___ wD -- incr Vx from 2e-5 to 5e-5
        # acn -  acd - 867 - ~___ wD -- decr tau from 1e0 to 5e-1
        # acm -  acd - 883 - ~___ wD -- incr tau from 1e0 to 2e0
        # acl -  acd - 852 - ~___ wD -- decr dB from 1e0 to 5e-1
        # ack -  acd - 912 - ~___ wD -- incr dB from 1e0 to 2e0 ###############
        # acj -  acd - 893 - ~___ wD -- incr de from 5e0 to 1e1
        # aci -  acd - 869 - ~___ wD -- decr da from 1e0 to 5e-1
        # ach -  acd - 843 - ~___ wD -- incr da from 1e0 to 2e0
        # acg -  aca - 835 - ~___ wD -- decr dB from 1e0 to 5e-1
        # acf -  aca - 836 - ~___ wD -- incr dB from 1e0 to 2e0
        # ace -  aca - 809 - ~___ wD -- decr de from 2e0 to 1e0
        # acd -  aca - 904 - ~___ wD -- incr de from 2e0 to 5e0 ###############
        # acc -  aca - 859 - ~___ wD -- decr da from 1e0 to 5e-1
        # acb -  aca - 824 - ~___ wD -- incr da from 1e0 to 2e0
        # aca -  abo - 861 - ~___ wD -- in,de Vx,Vy from 1e-5,1e-6 to 2e-5,5e-7 ######
        # abz -  abo - 839 - ~___ wD -- decr Vz from 5e-6 to 2e-6
        # aby -  abo - 844 - ~___ wD -- incr Vz from 5e-6 to 1e-5
        # abx -  abo - 862 - ~___ wD -- decr Vy from 1e-6 to 5e-7
        # abw -  abo - 812 - ~___ wD -- incr Vy from 1e-6 to 2e-6
        # abv -  abo - 841 - ~___ wD -- decr Vx from 1e-5 to 5e-6
        # abu -  abo - 856 - ~___ wD -- incr Vx from 1e-5 to 2e-5
        # abt -  abo - 844 - ~___ wD -- decr q from 1e1 to 5e0
        # abs -  abo - 840 - ~___ wD -- decr theta from 5e-3 to 2e-3
        # abr -  abo - 846 - ~___ wD -- incr theta from 5e-3 to 1e-2
        # abq -  abo - 840 - ~___ wD -- decr phi from 1e-3 to 5e-4
        # abp -  abo - 846 - ~___ wD -- incr phi from 1e-3 to 2e-3
        # abo -  abh - 851 rrn846 wD -- decr p,r from 7.5e-3,2e0 to 5e-3,1e0 ##
        # abn -  abh - 844 - ~___ wD -- decr r from 2e0 to 1e0
        # abm -  abh - 836 - ~___ wD -- incr r from 2e0 to 5e0
        # abl -  abh - 834 - ~___ wD -- decr q from 1e1 to 5e0
        # abk -  abh - 827 - ~___ wD -- incr q from 1e1 to 2e1
        # abj -  abh - 848 - ~___ wD -- decr p from 7.5e-3 to 5e-3
        # abi -  abh - 830 - ~___ wD -- incr p from 7.5e-3 to 1e-2
        # abh -  aaz - 833 - ~___ wD -- decr zf from 1e-6 to 5e-7  ############
        # abg -  aaz - 797 - ~___ wD -- incr zf from 1e-6 to 2e-6
        # abf -  aaz - 773 - ~___ wD -- incr pqr from 7.5e-3,1e1,2e0 to 2e-2,2e1,5e0
        # abe -  aan - 794 - ~___ wD -- decr tau from 1e0 to 5e-1
        # abd -  aan - 787 - ~___ wD -- incr tau from 1e0 to 2e0
        # abc -  aan - 758 - ~___ wD -- decr dB from 1e0 to 5e-1
        # abb -  aan - 795 - ~___ wD -- incr dB from 1e0 to 2e0
        # aba -  aan - 788 - ~___ wD -- decr de from 1e0 to 5e-1
        # aaz -  aan - 823 - ~___ wD -- incr de from 1e0 to 2e0  ##############
        # aay -  aan - 791 - ~___ wD -- decr da from 1e0 to 5e-1
        # aax -  aan - 780 - ~___ wD -- incr da from 1e0 to 2e0
        # aaw -  aan - 967 - ~___ wD -- rerun w/ smaller q bound
        # aav -  aan - 799 - ~___ wD -- decr Vz from 5e-6 to 2e-6
        # aau -  aan - 812 - ~___ wD -- incr Vz from 5e-6 to 1e-5
        # aat -  aan - 767 - ~___ wD -- decr Vx from 1e-5 to 5e-6
        # aas -  aan - 793 - ~___ wD -- incr Vx from 1e-5 to 2e-5
        # aar -  aan - 806 - ~___ wD -- decr Vy from 1e-6 to 5e-7
        # aaq -  aan - 809 - ~___ wD -- incr Vy from 1e-6 to 2e-6
        # aap -  aan - 798 - ~___ wD -- decr theta from 5e-3 to 2e-3
        # aao -  aan - 798 - ~___ wD -- incr theta from 5e-3 to 1e-2
        # aan -  aag - 818 - ~___ wD -- decr phi from 2e-3 to 1e-3  ###########
        # aam -  aag - 782 - ~___ wD -- incr phi from 2e-3 to 5e-3
        # aal -  aag - 791 - ~___ wD -- decr p from 7.5e-3 to 4e-3
        # aak -  aag - 789 - ~___ wD -- decr r from 2e0 to 1e0
        # aaj -  aag - 815 - ~___ wD -- incr r from 2e0 to 5e0
        # aai -  aag - 793 - ~___ wD -- decr q from 1e1 to 5e0
        # aah -  aag - 803 - ~___ wD -- incr q from 1e1 to 2e1
        # aag -  aad - 812 - ~___ wD -- decr p from 1.5e-2 to 7.5e-3  #########
        # aaf -  aad - 723 - ~___ wD -- incr p from 1.5e-2 to 3e-2
        # aae -  aad - 792 - ~___ wD -- incr Vy from 1e-6 to 2e-6
        # aad -   hs - 775 - ~___ wD -- run  ##################################
        # aac -      -~350 - ~___ wD -- arbitrary
        # aab -  aaa -~323 - ~___ wD -- incr Vx Vz from 1e-6 to 1e-4
        # aaa - BK_3 -~368 - ~___ wD -- run  ##################################
        

        
        # hs    - hp    - 987 - ~780 wD -- incr zf from 5e-7 to 1e-6
        # hp    - fq    - 990 - ~660 wD -- incr da from 5e-1 to 1e0
        # fq    - fm    - 974 - ~820 wD -- incr phi from 1e-3 to 2e-3 #########
        # fm    - ew    - 981 - ~740 wD -- decr da from 1e0 to 5e-1
        # ew    - dy    - 990 - ~720 wD -- incr tau from 1e-2 to 1e0
        # dy    - dx    - 983 - ~700 wD -- incr pqr from 7.5e-3, 5.0e0, 1.0e0 to 1.5e-2, 1.0e1, 2.0e0
        # dx    - dw    - 987 - ~620 wD -- incr Vx to 1e-5, Vz to 5e-6
        # dw    - ch    - 979 - ~660 wD -- incr Vy to 1.0e-6
        # # # # # # # change to smaller pqr (q)
        # ch    - ay    - 920 - decr p to 7.5e-1
        # ay    - ao    - 892 - incr q r wt by 5e-1 #############################
        # ao    - ak    - 866 - incr q wt to 1e2 ################################
        # ak    - v     - 846 - decr theta to 5e-1 ##############################
        # v     - q     - 851 - incr Vx to 5e-4 #################################
        # q     - p     - 838 - incr ctrl wts by 5e-1 ###########################
        # p     - o     - 814 - incr ctrl wts by 5e-1 ###########################
        # o     - f     - 678 - incr ctrl wts by 5e-1 ###########################
        # f     - c     - 619 - decr zf wt to 5e-5 ##############################
        # c     - 999   - 611 - decr phi wt to 1e-1 #############################
        # 999   - 993   - 596 - decr phi wt to 5e-1 #############################
        # 993   - 96    - 560 - incr Vy to 1e-5 # reran, 51.6... ################
        # 96    - 90    - 540 - incr q r wt to 5e1 ##############################
        # 90    - 9     - 504 - incr q r wt to 1e1 # _cg1 cgx = 1.0, 78.5% # 90 w/ S_q_1 = 3. 92.3%
        # 9     - 7     - 474 - incr q r wt to 5e0 ##############################
        # 7     - 5     - 417 - incr zf wt to 1e-4 ##############################
        # 5     - 4     - 344 - incr Vz wt to 1e-3 ##############################
        # 4     - 3     - 257 - incr Vx Vz wt to 1e-4 ###########################
        # 3     -       - 240 - SLF #############################################

    if True:
        a = 2
        # SCT controller iters
        # iz    -       - ___ - ~___ wD -- 
        # iy    -       - ___ - ~___ wD -- 
        # ix    -       - ___ - ~___ wD -- 
        # iw    -       - ___ - ~___ wD -- 
        # iv    -       - ___ - ~___ wD -- 
        # iu    -       - ___ - ~___ wD -- 
        # it    -       - ___ - ~___ wD -- 
        # is    -       - ___ - ~___ wD -- 
        # ir    -       - ___ - ~___ wD -- 
        # iq    -       - ___ - ~___ wD -- 
        # ip    -       - ___ - ~___ wD -- 
        # io    -       - ___ - ~___ wD -- 
        # in    -       - ___ - ~___ wD -- 
        # im    -       - ___ - ~___ wD -- 
        # il    -       - ___ - ~___ wD -- 
        # ik    -       - ___ - ~___ wD -- 
        # ij    -       - ___ - ~___ wD -- 
        # ii    -       - ___ - ~___ wD -- 
        # ih    -       - ___ - ~___ wD -- 
        # ig    -       - ___ - ~___ wD -- 
        # if    -       - ___ - ~___ wD -- 
        # ie    -       - ___ - ~___ wD -- 
        # id    - hs    - ___ - ~___ wD -- 
        # ic    - hz    - 978 - ~720 wD -- incr zf from 1e-6 to 1.1e-6
        # ib    - hz    - 968 - ~720 wD -- incr zf from 1e-6 to 2e-6
        # ia    - hs    - 983 - ~780 wD -- rerun  ***  678 for turbulence, only 3 hours!
        # hz    - hs    - 980 - ~760 wD -- decr tau from 1e0 to 9e-1
        # hy    - hs    - 973 - ~680 wD -- incr Vz from 5e-6 to 1e-5
        # hx    - hs    - 974 - ~640 wD -- incr Vy from 1e-6 to 2e-6
        # hw    - hs    - 982 - ~640 wD -- incr Vx from 1e-5 to 2e-5
        # hv    - hs    - 975 - ~660 wD -- incr p from 1.5e-2 to 3e-2
        # hu    - hs    - 977 - ~700 wD -- incr phi from 2e-3 to 4e-3
        # ht    - hs    - 971 - ~720 wD -- incr zf from 1e-6 to 2e-6
        # hs    - hp    - 987 - ~780 wD -- incr zf from 5e-7 to 1e-6
        # hr    - hp    - 989 - ~460 wD -- decr zf from 5e-7 to 2.5e-7
        # hq    - hp    - 984 - ~740 wD -- decr theta from 5e-3 to 2.5e-3
        # hp    - fq    - 990 - ~660 wD -- incr da from 5e-1 to 1e0
        # ho    - fq    - 973 - ~800 wD -- decr theta from 5e-3 to 2.5e-3
        # hn    - fq    - 962 - ~760 wD -- incr r from 2e0 to 4e0
        # hm    - fq    - 973 - ~720 wD -- decr Vx,Vz from 1e-5/5e-6 to 5e-6/2.5e-6
        # hl    - fq    - 964 - ~740 wD -- incr zf from 5e-7 to 1e-6, decr tau from 1e0 to 5e-1
        # hk    - fq    - 977 - ~700 wD -- decr p from 1.5e-2 to 7.5e-3, incr phi from 2e-3 to 4e-3
        # hj    - fq    - 963 - ~540 wD -- incr p from 1.5e-2 to 3e-2, decr phi from 2e-3 to 1e-3
        # hi    - fq    - 971 - ~660 wD -- incr de from 1e0 to 2e0
        # hh    - fq    - 974 - ~700 wD -- incr tau from 1e0 to 2e0
        # hg    - fq    - 970 - ~660 wD -- incr dB from 1e0 to 2e0
        # hf    - fq    - 942 - ~620 wD -- set phi theta equal to p q, Vy from 1e-6 to 5e-6
        # he    - fq    - 939 - ~700 wD -- set phi theta equal to p q
        # hd    - fq    - 977 - ~700 wD -- decr r from 2e0 to 1e0
        # hc    - fq    - 873 - ~___ wD -- larger q bound
        # hb    - fq    - 974 - ~600 wD -- decr q from 1e1 to 5e0
        # ha    - fq    - 993 - ~660 wD -- decr zf from 5e-7 to 2.5e-7
        # gz    - fq    - 965 - ~700 wD -- incr Vx from 1e-5 to 5e-5
        # gy    - fq    - 975 - ~800 wD -- incr Vx from 1e-5 to 2e-5
        # gx    - fq    - 980 - ~720 wD -- incr Vz from 5e-6 to 1e-5
        # gw    - fq    - 983 - ~720 wD -- incr Vy from 1e-6 to 2e-6
        # gv    - fq    - 985 - ~820 wD -- rerun
        # gu    - gg    - 972 - ~720 wD -- incr Vy from 1e-6 to 2e-6
        # gt    - gg    - 979 - ~720 wD -- rerun
        # gs    - gq    - 946 - ~760 wD -- incr p from 1.5e-2 to 3e-2
        # gr    - gq    - 934 - ~560 wD -- incr pqr from 1.5e-2/1e1/2e0 to 3e-2/2e1/4e0
        # gq    - gg    - 942 - ~740 wD -- decr da, de from 5e-1/1e0/5e-1 to 2.5e-1/5e-1/2.5e-1
        # gp    - gg    - 969 - ~720 wD -- incr theta from 5e-3 to 1e-2
        # go    - gg    - 956 - ~660 wD -- incr r from 2e0 to 4e0
        # gn    - gg    - 973 - ~720 wD -- decr de from 1e0 to 5e-1
        # gm    - gg    - 970 - ~700 wD -- incr zf from 5e-7 to 1e-6
        # gl    - gg    - 974 - ~620 wD -- incr p from 1.5e-2 to 3e-2
        # gk    - gg    - 967 - ~720 wD -- incr phi from 2e-3 to 4e-3
        # gj    - fm    - 977 - ~700 wD -- decr dB from 1e0 to 5e-1
        # gi    - fm    - 971 - ~700 wD -- rerun
        # gh    - fq    - 971 - ~680 wD -- incr dB from 1e0 to 2e0
        # gg    - fq    - 971 - ~800 wD -- decr dB from 1e0 to 5e-1
        # gf    - fs    - 957 - ~780 wD -- rerun
        # ge    - fq    - 970 - ~740 wD -- rerun
        # gd    - fq    - ___ - ~___ wD -- cProfile after fixing precomp turb, only Vgu
        # gc    - fq    - ___ - ~___ wD -- cProfile organized by tottime
        # gb    - fq    - ___ - ~___ wD -- cProfile # python -m cProfile con...
        # ga    - fq    - 975 - ~640 wD -- decr tau from 1e0 to 5e-1
        # fz    - fq    - 986 - ~720 wD -- decr de from 1e0 to 5e-1, incr theta from 5e-3 to 1e-2
        # fy    - fq    - 973 - ~660 wD -- decr de from 1e0 to 5e-1
        # fx    - fs    - 954 - ~640 wD -- rerun ...???
        # fw    - fq    - 951 - ~560 wD -- with error
        # fv    - fs    - 946 - ~840 wD -- incr zf from 5e-7 to 1e-6
        # fu    - fq    - 974 - ~700 wD -- incr zf from 5e-7 to 1e-6
        # ft    - fq    - 971 - ~480 wD -- incr p from 1.5e-2 to 3e-2
        # fs    - fq    - 970 - ~840 wD -- decr da from 5e-1 to 2.5e-1
        # fr    - fq    - 980 - ~700 wD -- incr phi from 2e-3 to 5e-3
        # fq    - fm    - 974 - ~820 wD -- incr phi from 1e-3 to 2e-3 #########
        # fp    - fm    - 966 - ~560 wD -- incr p from 1.5e-2 to 3e-2
        # fo    - ew    - 981 - ~660 wD -- decr da from 1e0 to 5e-1, de from 1e0 to 5e-1
        # fn    - ew    - 898 - ~700 wD -- decr da from 1e0 to 1e-1
        # fm    - ew    - 981 - ~740 wD -- decr da from 1e0 to 5e-1
        # fl    - ew    - 983 - ~___ wD -- rerun
        # fk    - ew    - ___ - ~680 wD -- test with increased convergence bounds (zf<=75)
        # fj    - ew    - ___ - ~___ wD -- test ew   control on 4 g turn w turb
        # fi    - ew    - ___ - ~___ wD -- test ew   controller on 4 g turn
        # fh    - BK_3  - ___ - ~___ wD -- test BK_3 controller on 60 deg bank
        # fg    - ew    - ___ - ~___ wD -- test ew   controller on 60 deg bank
        # ff    - ew    - 984 - ~643 wD -- decr phi from 1e-3 to 5e-4
        # fe    - ew    - 979 - ~560 wD -- incr phi from 1e-3 to 2e-3
        # fd    - ew    - 982 - ~720 wD -- decr p from 1.5e-2 to 7.5e-3
        # fc    - ew    - 979 - ~480 wD -- incr p from 1.5e-2 to 3e-2
        # fb    - ew    - 990 - ~600 wD -- decr zf from 5e-7 to 2.5e-7
        # fa    - ew    - 982 - ~700 wD -- incr zf from 5e-7 to 1e-6
        # ez    - ew    - 980 - ~600 wD -- decr dB from 1e0 to 5e-1
        # ey    - ew    - 991 - ~620 wD -- incr tau from 1e0 to 5e0
        # ex    - dy    - 989 - ~650 wD -- incr Vz from 5e-6 to 1e-5
        # ew    - dy    - 990 - ~720 wD -- incr tau from 1e-2 to 1e0
        # ev    - eu    - 883 - ~540 wD -- test with larger q
        # eu    - dy    - 987 - ~660 wD -- incr tau from 1e-2 to 5e-1
        # et    - dy    - 985 - ~580 wD -- incr tau from 1e-2 to 1e-1
        # es    - dy    - 982 - ~620 wD -- incr tau from 1e-2 to 5e-2
        # er    - dy    -~080 - ~___ wD -- incr p from 1.5e-2 to 1e1
        # eq    - dy    -~520 - ~___ wD -- incr p from 1.5e-2 to 1e0
        # ep    - dy    - 990 - ~540 wD -- incr tau from 1e-2 to 2e-2
        # eo    - dy    - 987 - ~640 wD -- incr V's from 1e-5,1e-6,5e-6 to 2e-5,2e-6,1e-5
        # en    - dy    - 988 - ~660 wD -- incr pqr from 1.5e-2,1e1,2e0 to 3e-2,2e1,4e0
        # em    - dy    - 978 - ~380 wD -- decr zf from 5e-7 to 1e-7
        # el    - dy    - 989 - ~520 wD -- decr phi from 1e-3 to 5e-4 and p from 1,5e-2 to 7.5e-3
        # ek    - dy    - 986 - ~560 wD -- decr phi from 1e-3 to 5e-4
        # ej    - dy    - 983 - ~640 wD -- rerun
        # ei    - dy    - 977 - ~640 wD -- incr zf from 5e-7 to 1e-6
        # eh    - dy    - 990 - ~600 wD -- incr Vx from 1e-5 to 5e-5
        # eg    - dy    - 940 - ~500 wD -- use Vx,Vy,Vz 1.0e-3, 1.0e-6, 2.0e-4, 
        # ef    - dx    - 989 - ~593 wD -- rerun
        # ee    - dy    - 981 - ~630 wD -- rerun
        # ed    - ec    - 863 -  ___ wD -- test with larger q
        # ec    - dy    - 981 - ~546 wD -- incr p from 1.5e-2 to 3e-2
        # eb    - dy    - 978 - ~560 wD -- incr phi from 1e-3 to 5e-3
        # ea    - dy    - 954 -  ___ wD -- incr zf from 5e-7 to 5e-6
        # dz    - dy    - 972 - ~460 wD -- incr q from 1e1 to 1e2
        # dy    - dx    - 983 - ~700 wD -- incr pqr from 7.5e-3, 5.0e0, 1.0e0 to 1.5e-2, 1.0e1, 2.0e0
        # dx    - dw    - 987 - ~620 wD -- incr Vx to 1e-5, Vz to 5e-6
        # dw    - ch    - 979 - ~660 wD -- incr Vy to 1.0e-6
        #                     # dropped q dispersion from 12 to 5
        # dv    - ch    - 892 - rerun with successes plotted
        # du    - ch    - 800 - with model error
        # dt    - ch    - 902 - decr VxVyVz from 5e-6 1e-7 1e-6 to 2.5e-6, 5.0e-8, 5.0e-7
        # ds    - ch    - 905 - incr pqr from 7.5e-3, 5.0e0, 1.0e0 to 1.5e-2, 1.0e1, 2.0e0
        # dr    - ch    - 909 - decr pqr from 7.5e-3, 5.0e0, 1.0e0 to 3.75e-3, 2.5e0, 5.0e-1
        # dq    - ch    - 909 - incr VxVyVz from 5e-6 1e-7 1e-6 to 1.0e-5, 5.0e-7, 5.0e-6
        # dp    - ch    - 913 - decr VxVyVz from 5e-6 1e-7 1e-6 to 1.0e-6, 5.0e-8, 5.0e-7
        # do    - ch    - 919 - incr dB from 1.0e0 to 2.0e0
        # dn    - ay    - 903 - rerun
        # dm    - ch    - 911 - rerun
        # dl    - dk    - 912 - rerun
        # dk    - ch    - 905 - drop all wts by 1.0e2
        # dj    - ch    - 909 - incr Vx from 5.0e-4 to 7.5e-4
        # di    - ch    - 919 - decr Vx from 5.0e-4 to 2.5e-4
        # dh    - db    - 913 - decr theta from 5.0e-1 to 4.0e-1
        # dg    - ch    - 917 - incr Vy from 1.0e-5 to 2.5e-5
        # df    - ch    - 916 - decr Vy from 1.0e-5 to 7.5e-6
        # de    - ch    - 899 - incr Vz to 2.5e-4
        # dd    - ch    - 905 - decr Vz to 7.5e-5
        # dc    - ch    - 909 - incr theta to 7.5e-1
        # db    - ch    - 922 - decr theta to 2.5e-1
        # da    - ch    - 919 - incr q to 7.5e2, decr r to 7.5e1
        # cz    - ch    - 913 - incr phi to 2.5e-1, decr p to 5.0e-1
        # cy    - ch    - 905 - incr phi to 2.5e-1
        # cx    - ch    - 905 - decr phi to 7.5e-2
        # cw    - ct    - 907 - rerun
        # cv    - ch    - 911 - decr p to 2.5e-1
        # cu    - ch    - 897 - incr zf to 7.5e-5
        # ct    - ch    - 927 - decr p to 5.0e-1
        # cs    - ch    - 926 - rerun
        # cr    - ay    - 910 - rerun
        # cq    - ch    - 905 - incr r to 2.5e2
        # cp    - ch    - 901 - decr r to 7.5e1
        # co    - ch    - 910 - incr q to 7.5e2
        # cn    - ch    - 876 - decr q to 2.5e2
        # cm    - ay    - 901 - incr r to 2.5e2
        # cl    - ay    - 914 - decr r to 7.5e1
        # ck    - ay    - 865 - decr q to 2.5e2
        # cj    - ay    - 898 - incr q to 7.5e2
        # ci    - ay    - 879 - incr p to 2.5e0
        # ch    - ay    - 920 - decr p to 7.5e-1
        # cg    - cd    - 901 - rerun
        # cf    - ay    - 897 - incr zf to 7.5e-5
        # ce    - ay    - 909 - decr zf to 2.5e-5
        # cd    - ay    - 919 - incr phi to 2.5e-1 *  *  *  *  *
        # cc    - ay    - 893 - decr phi to 7.5e-2
        # cb    - ay    - 881 - incr theta to 7.5e-1
        # ca    - ay    - 915 - decr theta to 2.5e-1
        # bz    - ay    - 836 - is this ay? (NOPE) incr ctrl to [5.0e2, 5.0e2, 5.0e2, 5.0e0]
        # by    - ay    - 910 - incr tau wt to 5e0
        # bx    - ay    - 901 - decr tau wt to 5e-1
        # bw    - ay    - 908 - rerun -- return to ay
        # bv    - bj    - 878 - rerun
        # bu    - bj    - 869 - incr Vx to 1e-3
        # bt    - bj    - 864 - decr Vx to 1e-4
        # bs    - bj    - 707 - error
        # br    - bj    - 878 - incr tau wt to 1e4
        # bq    - bj    - 891 - incr tau wt to 5e0
        # bp    - bj    - 896 - decr tau wt to 5e-1
        # bo    - bj    - 903 - incr dB wt to 5e2
        # bn    - bj    - 881 - decr dB wt to 5e1
        # bm    - ay    - 882 - incr de wt to 5e2
        # bl    - ay    - 902 - decr de wt to 5e1
        # bk    - ay    - 782 - incr da wt to 5e2
        # bj    - ay    - 916 - decr da wt to 5e1 ###############################
        # bi    - ay    - 829 - incr ctrl wts by 5e-1
        # bh    - ay    - 824 - incr p wt to 5e0
        # bg    - ay    - 904 - decr p wt to 5e-1
        # bf    - ay    - 883 - decr zf wt to 1e-5
        # be    - ay    - 887 - incr zf wt to 1e-4
        # bd    - ay    - 888 - incr phi to 5e-1
        # bc    - ay    - 904 - decr phi to 5e-2
        # bb    - ay    - 913 - incr Vx to 1e-3
        # ba    - ay    - 897 - decr Vx to 1e-4
        # az    - ay    - 912 - rerun
        # ay    - ao    - 892 - incr q r wt by 5e-1 #############################
        # ax    - ao    - 823 - incr Vz to 5e-3
        # aw    - ao    - 845 - decr Vz to 5e-4
        # av    - ao    - 854 - incr Vy to 5e-5
        # au    - ao    - 860 - decr Vy to 5e-6
        # at    - ao    - 860 - incr Vx to 1e-3
        # as    - ao    - 818 - decr Vx to 1e-4
        # ar    - ao    - 864 - incr r wt to 1e2
        # aq    - ak    - 840 - incr r wt to 1e2
        # ap    - ak    - 829 - decr r wt to 1e1
        # ao    - ak    - 866 - incr q wt to 1e2 ################################
        # an    - ak    - 802 - decr q wt to 1e1
        # am    - ak    - 821 - rerun
        # al    - v     - 829 - incr theta to 5e0
        # ak    - v     - 846 - decr theta to 5e-1 ##############################
        # aj    - v     - 822 - incr Vz to 5e-3
        # ai    - v     - 801 - decr Vz to 5e-4
        # ah    - v     - 828 - rerun
        # ag    - v     - 782 - decr zf wt to 1e-5
        # af    - v     - 819 - incr zf wt to 1e-4
        # ae    - v     - 726 - incr ctrl wts by 5e-1
        # ad    - v     - 653 - incr p wt to 5e0
        # ac    - v     - 838 - decr p wt to 5e-1
        # ab    - v     - 841 - incr r wt to 1e2
        # aa    - v     - 825 - decr r wt to 1e1
        # z     - v     - 856 - incr q wt to 1e2
        # y     - v     - 805 - decr q wt to 1e1
        # x     - q     - 823 - incr phi to 5e-1
        # w     - q     - 837 - decr phi to 5e-2
        # v     - q     - 851 - incr Vx to 5e-4 #################################
        # u     - q     - 797 - decr Vx to 5e-5
        # t     - q     - 836 - incr q r wt to 1e2
        # s     - q     - 756 - decr q r wt to 1e1
        # r     - p     - 816 - incr Vx Vz ctrl wts by 5e-1
        # q     - p     - 838 - incr ctrl wts by 5e-1 ###########################
        # p     - o     - 814 - incr ctrl wts by 5e-1 ###########################
        # o     - f     - 678 - incr ctrl wts by 5e-1 ###########################
        # n     - f     - 474 - decr ctrl wts by 5e-1
        # m     - f     - 613 - decr q r wt to 1e1
        # l     - f     - 609 - incr q r wt to 1e2
        # k     - f     - 624 - rerun
        # j     - c     - 618 - decr zf wt to 5e-5 incr q wt to 1e2
        # i     - c     - 617 - incr q wt to 1e2
        # h     - c     - 588 - decr q wt to 1e1
        # g     - c     - 529 - incr zf wt to 5e-4
        # f     - c     - 619 - decr zf wt to 5e-5 ##############################
        # e     - c     -~200 - decr da deB wt to 5e-2, incr tau wt to 5e0
        # d     - 999   - 611 - decr phi wt to 5e-2
        # c     - 999   - 611 - decr phi wt to 1e-1 #############################
        # b     - 999   - 578 - incr Vx wt to 5e-4
        # a     - 999   - 586 - decr Vx wt to 5e-5
        # 99998 -       - 596 - run 99991 again
        # 99997 - 999   - 415 - incr Vz wt to 5e-3
        # 99996 - 999   - 584 - decr Vz wt to 5e-4
        # 99995 - 999   - 584 - incr q wt to 1e2, decr r wt to 1e1
        # 99994 - 999   - 599 - decr r wt to 7.5e0
        # 99993 - 999   - 597 - decr r wt to 2.5e1
        # 99992 - 999   -~000 - incr p wt to 5e1
        # 99991 - 999   - 606 - decr r wt to 1e1
        # 99990 - 999   - 524 - incr r wt to 1e2
        # 9999  - 999   - 457 - decr de wt to 1e0
        # 9998  - 999   - 589 - incr de wt to 1e1
        # 9997  - 999   - 571 - decr theta wt to 5e-1
        # 9996  - 999   - 541 - incr theta wt to 5e0
        # 9995  - 999   - 546 - decr q wt to 1e1
        # 9994  - 999   - 571 - incr q wt to 1e2
        # 9993  - 999   - 593 - decr p wt to 5e-1
        # 9992  - 999   - 389 - incr p wt to 5e0
        # 9991  - 993   - 534 - decr theta wt to 5e-1
        # 9990  - 993   - 524 - incr theta wt to 5e0
        # 999   - 993   - 596 - decr phi wt to 5e-1 #############################
        # 998   - 993   - 299 - incr phi wt to 5e0
        # 997   - 993   - 529 - incr dB wt to 1e1
        # 996   - 993   - 526 - decr tau wt to 1e-2
        # 995   - 96    - 505 - incr r wt to 7e1
        # 994   - 96    - 514 - incr r wt to 1e2
        # 993   - 96    - 560 - incr Vy to 1e-5 # reran, 51.6... ################
        # 992   - 96    - 471 - incr zf wt to 1e-3
        # 991   - 96    - 540 - incr q wt to 1e2
        # 99    - 90    - 465 - decr Vy to 1e-8 oops
        # 99    - 90    - 484 - incr Vy to 1e-4
        # 98    - 90    - 484 - incr Vx to 5e-4
        # 97    - 90    - 491 - incr q r wt to 1e2
        # 96    - 90    - 540 - incr q r wt to 5e1 ##############################
        # 95    - 90    -~400 - incr p wt to 2e0
        # 94    - 90    -~189 - decr de wt to 5e-1
        # 93    - 90    - 470 - incr tau wt to 5e-1
        # 92    - 90    - 498 - incr theta wt to 5e0
        # 91    - 9     - 491 - incr q wt to 1e1, (r at 5e0)
        # 90    - 9     - 504 - incr q r wt to 1e1 # _cg1 cgx = 1.0, 78.5% # 90 w/ S_q_1 = 3. 92.3%
        # 9     - 7     - 474 - incr q r wt to 5e0 ##############################
        # 8     - 7     - 427 - incr Vy wt to 1e-4
        # 7     - 5     - 417 - incr zf wt to 1e-4 ##############################
        # 6     - 5     - 152 - incr wt to ????
        # 5     - 4     - 344 - incr Vz wt to 1e-3 ##############################
        # 4     - 3     - 257 - incr Vx Vz wt to 1e-4 ###########################
        # 3     -       - 240 - SLF #############################################
        # throttle wrong at altitude? smaller than expected

    # # controllability
    # controllability_analysis(bire_file,H=flight_conditions[fc]["h"]*1.,
    #     M=flight_conditions[fc]["m"]*1.)
    # controllability_analysis(base_file,H=flight_conditions[fc]["h"]*1.,
    #     M=flight_conditions[fc]["m"]*1.)
    # quit()

    # run GS case
    if False:
        # di = [-1000.,0.,0.] # 
        # di = [-750.,0.,0.] # 
        # di = [-500.,0.,0.] # 
        # di = [-250.,0.,0.] # 
        # di = [90.,10.,2.5]
        # # # # 
        # plot_vars["plot_full"] = False
        # plot_vars["plot_delta"] = True
        # plot_vars["zoom_deltas"] = False
        # plot_vars["plot_norm"] = True
        # # plot_vars["format"] = "png"
        # # plot_vars["zoom_fraction"] = 1./15.
        # # plot_vars["plot_input_limits_zoomed"] = False
        # # # #
        # di = [0.,0.,0.]
        # t_gain = 90.
        # adt = 0. # 90. /2. # 
        # scale = 1. # 105./90. # 
        # offset = 15. # 0. # 
        # run_bire["num"] = 1 # 1000 # 
        # run_bire["final_time"] = (t_gain + adt)*scale + offset # 15. # 
        # run_bire["trim_bank"] = 0.0
        # run_bire["trim_climb"] = 0.0 # 0. # 
        # run_bire["start_climbing"] = False # False # 
        # run_bire["end_gs_climbing"] = False # True # 
        # run_bire["final_mach"] = flight_conditions[f2]["m"]*1. # 
        # run_bire["final_altitude"] = flight_conditions[f2]["h"]*1. # 
        # run_bire["t_gain_schedule"] = t_gain + adt # 0. # 
        # run_bire["gain_steps"] = 40
        # run_bire["trim_steps"] = 40
        # run_bire["interpolation_type"] = "next" # "linear" # "nearest-up" # 
        # run_bire["has_turbulence"] = False # True # 
        # run_bire["turbulence_setting"] = "light" # "moderate", # "severe", # 
        # run_bire["has_model_error"] = False # True # 
        # run_bire["skip_simulation"] = False # True # 
        # run_bire["save_data"] = True # False # 
        # # run_bire["fixed_FM_errors"] = [0.1,0.1,0.1,0.1,0.1,0.1]
        # run_bire["name_end"] = "_" + f1 + "_BK_3_GS"#4_wSd" # _1e1pqr" #+ "_" + name
        # # run_bire["mrrr"] = [1,3,5,6,7,9,11] # [6,7,11] # 
        # # run_bire["mrrc"] = [2] # None # 
        # bire_dict["controller"]["LQR"] = {
        #     "note" : "_almost_current",
        #     "Q" : [1.0e-3, 1.0e-6, 2.0e-4, # 
        #         1.0e0, 1.0e0, 1.0e0,
        #         0.0, 0.0, 5.0e-6,
        #         1.0e0, 1.0e0, 0.0],
        #     "Q1a" : [0.0e0, 0.0e0, 0.0e0, 0.0e0],
        #     "Q2a" : [0.0, 0.0, 0.0, 0.0],
        #     "R" : [5.0e0, 5.0e0, 5.0e0, 5.0e-2]
        # }
        # run_bire["num"] = run_base["num"] = 1
        # run_single_simulation(bire_dict,rtdst_1sg=di,**run_bire,**plot_vars)
        # # run_single_simulation(base_dict,rtdst_1sg=di,**run_base,**plot_vars)
        quit()

    # run single case
    bire_dict["initial"]["trim_guess"] = {}
    bire_dict["initial"]["trim"]["type"] = "sct"
    bire_dict["initial"]["trim"]["bank_angle[deg]"] = 0.0
    # ###############
    # bire_dict["initial"]["trim"]["type"] = "shss"
    # bire_dict["initial"]["trim"].pop("bank_angle[deg]")
    # bire_dict["initial"]["trim"]["sideslip_angle[deg]"] = 11.0
    # # bire_dict["initial"]["airspeed[ft/s]"] = 222.0
    # bire_dict["initial"]["mach"] = 0.2
    # bire_dict["initial"]["altitude[ft]"] = 1000.0
    # bire_dict["initial"]["trim_guess"]["BIRE[deg]"] = -10.0
    # bire_dict["initial"]["trim_guess"]["elevator[deg]"] = 10.0
    # # bire_dict["aircraft"]["CG_shift[ft]"] = [1.0,0.0,0.0]
    # bire = Aircraft(bire_dict)
    # bire._report_trim_solution(bire.x_trim,bire.u_trim,bire.trim_iter)
    # quit()
    # ##############
    di = [90.,10.,2.5]
    # di = [0.,0.,0.0]
    plot_vars["plot_full"] = False # True # 
    plot_vars["plot_delta"] = True # False # 
    plot_vars["zoom_deltas"] = True # False # 
    plot_vars["zoom_full"] = False # True # 
    plot_vars["format"] = "png"
    # plot_vars["format"] = "pdf"
    plot_vars["zoom_fraction"] = 1./15.
    # plot_vars["output_states"] = True # False # 
    run_base["trim_bank"] = run_bire["trim_bank"] = 0.0
    run_base["gain_steps"] = run_bire["gain_steps"] = 2
    run_base["skip_simulation"] = run_bire["skip_simulation"] = False
    # # #
    run_bire["num"] = run_base["num"] = 1
    # ###########################################################################
    # run_base["mrrr"] = run_bire["mrrr"] = [0,2,6,7,8,9,10,11]
    # run_base["mrrc"] = run_bire["mrrc"] = [2,3]
    # ###########################################################################
    run_base["final_time"] = run_bire["final_time"] = 15.0 # 1.0 # 5.0 # 
    # di = [1.0,1.0,1.0]
    # ###########################################################################
    # run_bire["has_turbulence"] = run_base["has_turbulence"] = True # False # 
    # run_bire["has_model_error"] = run_base["has_model_error"] = True # False # 
    # run_bire["fixed_FM_errors"] = run_base["fixed_FM_errors"] = \
    #     [0.1,0.1,0.1,0.1,0.1,0.1]
    run_single_simulation(bire_dict,rtdst_1sg=di,**run_bire,**plot_vars)
    # run_single_simulation(base_dict,rtdst_1sg=di,**run_base,**plot_vars)
    quit()

    # # run forces analysis
    # di = [50.,6.,1.5]
    # compare_aero_forces(bire_dict,rtdst_1sg=di,**run_bire,**plot_vars)
    # # compare_aero_forces(base_dict,rtdst_1sg=di,**run_base,**plot_vars)
    # quit()

    # # run monte carlo perturbation analysis
    # bire_dict["initial"]["trim_guess"] = {}
    # bire_dict["initial"]["trim_guess"]["BIRE[deg]"] = 1.0
    # bire_dict["initial"]["trim_guess"]["BIRE[deg]"] = 60.0
    # bire_dict["initial"]["trim_guess"]["elevator[deg]"] = 25.0
    run_base["trim_bank"] = run_bire["trim_bank"] = 0.0
    run_base["gain_steps"] = run_bire["gain_steps"] = 2
    di = [100.,12.,3.] # SLF
    # di = [100.,5.,3.] # SCT
    # # # di = [120.,0.,0.]
    # di = [0.,80.,0.]
    # # # di = [0.,0.,5.] # [0.,0.,10.] #
    # di = [120.,25.,5.]
    # plot_vars["format"] = "png"
    plot_vars["format"] = "pdf"
    run_base["plot_ul_bounds"] = run_bire["plot_ul_bounds"] = True # False # 
    run_bire["num"] = run_base["num"] = 10 # 1000 # 
    # run_base["has_model_error"] = run_bire["has_model_error"] = True # False # 
    # run_bire["FM_errors"] = [ 0.03, 0.25, 0.10, 0.25, 0.25, 0.25 ]
    monte_carlo_perturbations(bire_dict,rtdst_1sg=di,**run_bire,**plot_vars)
    # monte_carlo_perturbations(base_dict,rtdst_1sg=di,**run_base,**plot_vars)
    quit()
    #
    # # single axis pqr dispersions
    # disa = [[120.,0.,0.],[0.,40.,0.],[0.,0.,5.]] # SLF
    # # disa = [[120.,0.,0.],[0.,10.,0.],[0.,0.,5.]] # SCT
    # # bire_dict["initial"]["trim_guess"] = {}
    # # bire_dict["initial"]["trim_guess"]["BIRE[deg]"] = 1.0
    # # run_base["name_end"] = run_bire["name_end"] = "_" + f1 + "_BK_hs_two"
    # run_base["plot_ul_bounds"] = run_bire["plot_ul_bounds"] = True # False # 
    # for i in range(3): # [0]: # [1]: # 
    #     ds = disa[i]
    #     monte_carlo_perturbations(bire_dict,rtdst_1sg=ds,**run_bire,**plot_vars)
    #     # monte_carlo_perturbations(base_dict,rtdst_1sg=ds,**run_base,**plot_vars)
    # quit()
    # #
    # # single FM error dispersions
    # names = ["CL","CS","CD","Cl","Cm","Cn"]
    # run_base["has_model_error"] = run_bire["has_model_error"] = True
    # run_base["plot_ul_bounds"] = run_bire["plot_ul_bounds"] = True # False # 
    # first_name = True
    # for i in range(len(names)):
    #     name = names[i]
    #     # create FM errors
    #     FM_error_list = np.zeros((6,))
    #     FM_error_list[i] = 0.25
    #     run_base["FM_errors"] = FM_error_list*1.
    #     run_bire["FM_errors"] = FM_error_list*1.
    #     if first_name:
    #         first_name = False
    #         end_i = None
    #     else:
    #         end_i = -3
    #     run_base["name_end"] = run_base["name_end"][:end_i] + "_" + name
    #     run_bire["name_end"] = run_bire["name_end"][:end_i] + "_" + name
    #     monte_carlo_perturbations(bire_dict,rtdst_1sg=di,**run_bire,**plot_vars)
    #     # monte_carlo_perturbations(base_dict,rtdst_1sg=di,**run_base,**plot_vars)
    # quit()
    # #
    # # run for roa plots / diff controllers
    # run_bire["has_turbulence"] = run_base["has_turbulence"] = False # True # 
    # run_bire["has_model_error"] = run_base["has_model_error"] = True # False # 
    # run_base["plot_ul_bounds"] = run_bire["plot_ul_bounds"] = True # False # 
    # monte_carlo_perturbations(bire_dict,rtdst_1sg=di,**run_bire,**plot_vars)
    # # monte_carlo_perturbations(base_dict,rtdst_1sg=di,**run_base,**plot_vars)
    # run_bire["mrrc"] = [2]
    # run_bire["name_end"] = run_bire["name_end"]      + "_nB"
    # monte_carlo_perturbations(bire_dict,rtdst_1sg=di,**run_bire,**plot_vars)
    # run_bire["mrrc"] = [3]
    # run_bire["name_end"] = run_bire["name_end"][:-3] + "_nt"
    # monte_carlo_perturbations(bire_dict,rtdst_1sg=di,**run_bire,**plot_vars)
    # run_bire["mrrc"] = [2,3]
    # run_bire["name_end"] = run_bire["name_end"][:-3] + "_nBt"
    # monte_carlo_perturbations(bire_dict,rtdst_1sg=di,**run_bire,**plot_vars)
    # quit()
    # # # Turbulence cases
    # di = [90.,10.,2.5]
    # run_bire["fixed_FM_errors"] = run_base["fixed_FM_errors"] = \
    #     [0.1,0.1,0.1,0.1,0.1,0.1]
    # run_bire["has_turbulence"] = run_base["has_turbulence"] = True # False # 
    # run_bire["has_model_error"] = run_base["has_model_error"] = True # False # 
    # # run_base["trim_bank"] = run_bire["trim_bank"] = 0.0
    # run_base["num"] = run_bire["num"] = 1
    # run_base["skip_simulation"] = run_bire["skip_simulation"] = False # True # 
    # run_single_simulation(bire_dict,rtdst_1sg=di,**run_bire,**plot_vars)
    # # run_single_simulation(base_dict,rtdst_1sg=di,**run_base,**plot_vars)
    # run_bire["mrrc"] = [2]
    # run_bire["name_end"] = run_bire["name_end"]      + "_nB"
    # run_single_simulation(bire_dict,rtdst_1sg=di,**run_bire,**plot_vars)
    # run_bire["mrrc"] = [3]
    # run_bire["name_end"] = run_bire["name_end"][:-3] + "_nt"
    # run_single_simulation(bire_dict,rtdst_1sg=di,**run_bire,**plot_vars)
    # run_bire["mrrc"] = [2,3]
    # run_bire["name_end"] = run_bire["name_end"][:-3] + "_nBt"
    # run_single_simulation(bire_dict,rtdst_1sg=di,**run_bire,**plot_vars)
    # quit()
    quit()
    # #
    # # test various trim conditions
    # for fc in ["T1","T2","C1","C2","C3"]:
    #     run_bire["initial_mach"] = flight_conditions[fc]["m"]*1.
    #     run_bire["initial_altitude"] = flight_conditions[fc]["h"]*1.
    #     run_bire["name_end"] = (fc != "C2")*("_" + fc) + "_BK_3"
    #     monte_carlo_perturbations(bire_dict,rtdst_1sg=di,**run_bire,**plot_vars)
    # quit()



    # # # test trim with cg forward
    # # bire_dict["aircraft"]["CG_shift[ft]"] = [1.0,0.0,0.0]
    # bire_dict["initial"]["trim_guess"] = {}
    # bire_dict["initial"]["trim_guess"]["BIRE[deg]"] = 0.0
    # bire_dict["initial"]["trim"]["bank_angle[deg]"] = 30.0
    # bire = Aircraft(bire_dict)
    # # bire.run_trim()
    # x0 = bire.x_trim*1.
    # u0 = bire.u_trim*1.
    # bire._report_trim_solution(x0,u0)
    # print()
    # print("running with initial guess of +45 deg")
    # u_guess = np.zeros((4,))
    # u_guess[2] = np.deg2rad(+45.0)
    # bire.verbose_trim = True # False # 
    # bire._initialize_state(u_guess=u_guess,no_report=False)
    # bire._report_trim_solution(bire.x_trim,bire.u_trim)
    # bire._report_trim_solution(x0,u0)
    # quit()
    # dB_max = 60.
    # num = 2*int(dB_max)+1 # 5 # 
    # bire.verbose_trim = False # True # 
    # dB_guesses_deg = np.linspace(-dB_max,dB_max,num=num)
    # dB_guesses = np.deg2rad(dB_guesses_deg)
    # fig,ax1 = plt.subplots()
    # ax2 = ax1.twinx()
    # dBs = np.zeros((num,))
    # des = np.zeros((num,))
    # for i in range(num):
    #     print("running {:>03d}, dB = {:> 6.1f} deg".format(i+1,
    #         dB_guesses_deg[i]))
    #     u_guess[2] = dB_guesses[i]
    #     bire._initialize_state(u_guess=u_guess,no_report=True)
    #     des[i],dBs[i] = np.rad2deg(bire.u_trim[1:3])
    # lns2 = ax2.plot(dB_guesses_deg,des,"b",label="de")
    # lns1 = ax1.plot( dB_guesses_deg,dBs,"r",label="dB")
    # lns1 = ax1.lines; lns2 = ax2.lines
    # lns = lns1 + lns2
    # labs = [lns1[0].get_label(),lns2[0].get_label()]
    # ax1.legend(lns,labs)#loc=0)
    
    # ax1.set_xlabel( "BIRE initial guess [deg]")
    # ax1.set_ylabel( "BIRE trim solution [deg]")
    # ax2.set_ylabel("elevator trim solution [deg]")
    # plt.show()
    # quit()

    # test 'linearity' of K wrt gain scheduling
    n = 10
    gamma = 15.
    x1H = 15000.
    x2H = 20000.
    x1M = 0.6
    x2M = 0.67
    x1P = 0.
    x2P = 30.
    V = (x1M + x2M)/2.*stdatm_english((x1H + x2H)/2.)[5]
    t = (x2H - x1H)*(1. + np.tan(np.deg2rad(gamma))**2. )**0.5/V
    print(t)
    
    # initialize steps
    Hs = np.linspace(x1H,x2H,n)
    Ms = np.linspace(x1M,x2M,n)
    ps = np.linspace( 0., 1.,n)
    # initialize aircraft
    craft = Aircraft(bire_dict)
    # initialize arrays for trim points and gains
    x_trs = np.zeros((n,16))
    u_trs = np.zeros((n, 4))
    K_trs = np.zeros((n, 4, 9))
    K_lns = np.zeros((n, 4, 9))
    # calculate trim at each point
    for i in range(n):
        print(" H0 =",Hs[i],", M0 =",Ms[i])
        craft.H0 = Hs[i]
        craft.M0 = Ms[i]
        craft._initialize_state()
        x_trs[i] = craft.x_trim_euler*1.
        u_trs[i] = craft.u_trim*1.
    # get 'linear' trim terms
    x_lns = np.array([(1. - psi)*x_trs[0] + psi*x_trs[-1] for psi in ps])
    u_lns = np.array([(1. - psi)*u_trs[0] + psi*u_trs[-1] for psi in ps])
    # determine true trim gains
    for i in range(n):
        # store trim condition
        craft.x_trim_euler = x_trs[i]*1.
        craft.u_trim       = u_trs[i]*1.
        craft._build_controller(report=False,save_matrices=False,
            mrrr=[6,7,11],mrrc=None,drop_actrs=True,
            include_stall_derivatives=False,skip_reporting=True,run_freq=False)
        K_trs[i] = craft.Lin_Model.K*1.
    # determine lin trim gains
    for i in range(n):
        # store trim condition
        craft.x_trim_euler = x_lns[i]*1.
        craft.u_trim       = u_lns[i]*1.
        craft._build_controller(report=False,save_matrices=False,
            mrrr=[6,7,11],mrrc=None,drop_actrs=True,
            include_stall_derivatives=False,skip_reporting=True,run_freq=False)
        K_lns[i] = craft.Lin_Model.K*1.
    # get 'linear' gains
    K_LNs = np.array([(1. - psi)*K_trs[0] + psi*K_trs[-1] for psi in ps])

    # plots
    subdict = {
        # "figsize" : (8.,8.),
        "constrained_layout" : True,
        "sharex" : True
    }
    xsf, xsa = plt.subplots(4,3,figsize=(8.,8.),**subdict)
    usf, usa = plt.subplots(2,2,figsize=(8.,8.),**subdict)
    Ksf, Ksa = plt.subplots(4,9,figsize=(18.,8.),**subdict)
    states = ["V_{x_b}","V_{y_b}","V_{z_b}",
    "p","q","r",
    "x_f","y_f","z_f",
    "\phi",r"\theta","\psi"]
    state_units = ["ft/s","ft/s","ft/s",
    "deg/s","deg/s","deg/s",
    "ft","ft","ft",
    "deg","deg","deg"]
    controls = ["\delta_a","\delta_e^B","\delta_B",r"\tau"]
    control_units = ["deg","deg","deg","percent"]
    r2dconv = lambda x : np.rad2deg(x)
    d2rconv = lambda x : np.deg2rad(x)
    nullconv = lambda x : x
    prcconv = lambda x : x*100.
    for i in range(4):
        for j in range(3):
            k = j + i*3
            if k in [3,4,5,9,10,11]:
                stconv = r2dconv
            else:
                stconv = nullconv
            xsa[i,j].plot(stconv(x_trs[:,k]),"b-")
            xsa[i,j].plot(stconv(x_lns[:,k]),"r--")
            xsa[i,j].set_ylabel("$" + states[k] + "$" + " [" + state_units[k] + "]")
    for i in range(2):
        for j in range(2):
            k = j + i*2
            if k in [0,1,2]:
                ctconv = r2dconv
            else:
                ctconv = prcconv
            usa[i,j].plot(ctconv(u_trs[:,k]),"b-")
            usa[i,j].plot(ctconv(u_lns[:,k]),"r--")
            usa[i,j].set_ylabel("$" + controls[k] + "$" + " [" + control_units[k] + "]")
    state_units.pop(11)
    state_units.pop( 7)
    state_units.pop( 6)
    for i in range(4):
        for j in range(9):
            if j in [3,4,5,7,8]:
                stconv = d2rconv
            else:
                stconv = nullconv
            if i in [0,1,2]:
                ctconv = r2dconv
            else:
                ctconv = prcconv
            # Ksa[i,j].plot(ctconv(stconv(K_trs[:,i,j])),"b-")
            # Ksa[i,j].plot(ctconv(stconv(K_lns[:,i,j])),"r--")
            # Ksa[i,j].plot(ctconv(stconv(K_LNs[:,i,j])),"g.-")
            Ksa[i,j].plot(K_trs[:,i,j],"b-")
            Ksa[i,j].plot(K_lns[:,i,j],"r--")
            Ksa[i,j].plot(K_LNs[:,i,j],"g:")
            Ksa[i,j].set_ylabel("$K_{" + str(i+1) + "," + str(j+1) + "}$" 
                # + " [" + control_units[k] + "/" + state_units[k] + "]"
                )
    plt.show()
    
    quit()

    # report A and B at various bank angles
    banks = np.deg2rad(np.linspace(0.,30.,num=2))
    run_bire_trim = False
    if run_bire_trim:
        craft = Aircraft(bire_dict)
        wrd = "\delta_B = "
    else:
        craft = Aircraft(base_dict)
        wrd = "\delta_r = "
    rows = [0,2,4,6,8,1,3,5,7]
    cols = [1,3,0,2]
    mrrc = None # [2,3] # 
    # vars = ["u","v","w","p","q","r","z","phi","theta"]
    # print([vars[row] for row in rows])
    for phi in banks:
        craft.phi_trim = phi
        craft._initialize_state()
        craft._build_controller(report=False,save_matrices=False,
        mrrr=[6,7,11],mrrc=mrrc,drop_actrs=True,
            include_stall_derivatives=False,run_freq=False)
        # print(craft.Lin_Model.A_min.shape,craft.Lin_Model.B_min.shape)
        A,B = craft.Lin_Model.A_min,craft.Lin_Model.B_min
        print(r"\begin{matrix}")
        print("\phi = {:> 3.0f}, \quad".format(np.rad2deg(phi)))
        print(wrd + "{:> 6.2f}, \\\\".format(np.rad2deg(craft.u_trim[2])))
        # report_latex((A[rows,:])[:,rows],"A",add_tab=False,endln=True) # comquad=True,
        report_latex((B[rows,:])[:,cols],"B",add_tab=False,endln=True)
        print(r"\end{matrix}")
    quit()

    BASE = Aircraft(base_dict)
    # BASE._report_trim_solution(BASE.x_trim,BASE.u_trim,BASE.trim_iter)
    # # # BASE.check_partials() # need to fix 
    # # BASE._build_controller(report=False,save_matrices=False,
    # # include_stall_derivatives=False,run_freq=False)
    # # rep2D(BASE.x_trim[:,np.newaxis].T,"x_trim")
    # # rep2D(BASE.u_trim[:,np.newaxis].T,"u_trim")
    # # rep2D(BASE.Lin_Model.A,"A_base")
    # # A_num = make_jacobian(BASE._nonlinear_euler_dynamics,[0.0,BASE.x_trim],1)
    # # B_num = make_jacobian(BASE._nonlinear_euler_dynamics,\
    # #     [0.0,BASE.x_trim,True,True,BASE.u_trim],4)
    # # rep2D(A_num,"A_bsnm")
    # # rep2D(A_num - BASE.Lin_Model.A,"A_diff")
    # # rep2D(BASE.Lin_Model.B,"B_base")
    # # rep2D(B_num,"B_bsnm")
    # # rep2D(B_num - BASE.Lin_Model.B,"B_diff")
    # # rep2D(BASE.Lin_Model.K,"K_base")

    # # BASE.run_uncontrolled_comparison_simulation()
    # # BASE.plot_results(**plot_vars)
    # # # # BASE.integration_analysis_simulation()

    BIRE = Aircraft(bire_dict)
    # BIRE._report_trim_solution(BIRE.x_trim,BIRE.u_trim,BIRE.trim_iter)
    # # # BIRE.check_partials()
    # # BIRE._build_controller(report=False,save_matrices=False,
    # # include_stall_derivatives=False,run_freq=False)
    # # rep2D(BIRE.x_trim[:,np.newaxis].T,"x_trim")
    # # rep2D(BIRE.u_trim[:,np.newaxis].T,"u_trim")
    # # rep2D(BIRE.Lin_Model.A,"A_bire")
    # # A_num = make_jacobian(BIRE._nonlinear_euler_dynamics,[0.0,BIRE.x_trim],1)
    # # B_num = make_jacobian(BIRE._nonlinear_euler_dynamics,\
    # #     [0.0,BASE.x_trim,True,True,BASE.u_trim],4)
    # # rep2D(A_num,"A_brnm")
    # # rep2D(A_num - BIRE.Lin_Model.A,"A_diff")
    # # rep2D(BIRE.Lin_Model.B,"B_bire")
    # # rep2D(B_num,"B_brnm")
    # # rep2D(B_num - BIRE.Lin_Model.B,"B_diff")
    # # rep2D(BIRE.Lin_Model.K,"K_bire")
    # # BIRE.run_uncontrolled_comparison_simulation()
    # # BIRE.plot_results(**plot_vars)
    # # # BIRE.integration_analysis_simulation()

    # # di = loadmat(BIRE.fldr_prfx + "_" + "data/monte_carlo/" + 
    # #     "bire_p60_q30_r10_n500_wul_wrl_wc_ws_HK4_sct.mat")
    # # print(di.keys())
    # # print(di["note"])
    # # print(di["system notes"])
    # # print(di["x_eq"])
    # # print(di["u_eq"])


    ftb, atb = plt.subplots()
    # plot grid
    atb.grid(which="major",lw=0.6,ls="-",c="0.5")
    atb.grid(which="minor",lw=0.5,ls="dotted",c="0.5")
    
    # plot
    atb_y2 = atb.twinx()

    BIR2 = Aircraft(bire_dict)
    banks = np.concatenate((
        np.linspace(0.,55.,num=56),
        np.linspace(55.01,60.,num=500),
        np.linspace(61.,77.,num=17)
    ))
    # print(banks)
    for i in range(len(banks)):
        bank_deg = banks[i]
        BASE.phi_trim = np.deg2rad(bank_deg)
        # BASE._initialize_state(no_report=True)
        # rep2D(BASE.x_trim[:,np.newaxis].T,"x_bs_trim",final_endline=False)
        # rep2D(BASE.u_trim[:,np.newaxis].T,"u_bs_trim",final_endline=False)
        BIRE.phi_trim = np.deg2rad(bank_deg)
        BIRE._initialize_state(
            x_guess=BIRE.x_trim*1,u_guess=BIRE.u_trim*1,
            no_report=True)
        # rep2D(BIRE.x_trim[:,np.newaxis].T,"x_br_trim",final_endline=False)
        # rep2D(BIRE.u_trim[:,np.newaxis].T,"u_br_trim",final_endline=False)
        # print()
        BIR2.phi_trim = np.deg2rad(bank_deg)
        BIR2._initialize_state(
            # x_guess=BIR2.x_trim*1,u_guess=BIR2.u_trim*1,
            no_report=True)
        # aero conditions
        Vu,Vv,Vw = BIRE.x_trim[0], BIRE.x_trim[1], BIRE.x_trim[2]
        a = atan2(Vw,Vu)
        V = (Vu * Vu + Vv * Vv + Vw * Vw)**0.5
        b = asin(Vv/V)
        _,g,_,_,_,_ = BIRE.stdatm(BIRE.H0)
        ph = BIRE.x_trim[ 9]
        th = BIRE.x_trim[10]
        n_BASE = BASE._load_factors(BASE.x_trim,BASE.u_trim)[2]
        print(("SCT trim bank angle = {:> 6.2f}, n = {:> 6.3f}," 
            + " 0 = {:> 6.3f}").format(bank_deg,n_BASE,
            np.max(BIRE._trim_forces(
                a,b,ph,th,g,BIRE.x_trim,BIRE.u_trim
            )),
            ))
        # plot
        while np.rad2deg(BIRE.u_trim[2]) > 90.:
            BIRE.u_trim[2] -= np.deg2rad(180.)
            BIRE.u_trim[1] = - BIRE.u_trim[1]
        while np.rad2deg(BIRE.u_trim[2]) < -90.:
            BIRE.u_trim[2] += np.deg2rad(180.)
            BIRE.u_trim[1] = - BIRE.u_trim[1]
        while np.rad2deg(BIR2.u_trim[2]) > 90.:
            BIR2.u_trim[2] -= np.deg2rad(180.)
            BIR2.u_trim[1] = - BIR2.u_trim[1]
        while np.rad2deg(BIR2.u_trim[2]) < -90.:
            BIR2.u_trim[2] += np.deg2rad(180.)
            BIR2.u_trim[1] = - BIR2.u_trim[1]
        if BIRE.u_trim[1] >= 0.:
            cbr = "r"
            lbl_add = " deB < 0"
        else:
            cbr = "b" 
            lbl_add = " deB >= 0"
        if BIR2.u_trim[1] >= 0.:
            c2r = "r"
            l2l_add = " deB < 0"
        else:
            c2r = "b" 
            l2l_add = " deB >= 0"
        if i == 0:
            lbl_base = "base"
            lbl_bire = "bire"
            lbl_da = "bire da"
            lbl_de = "bire de"
            lbl_ta = "bire ta"
        else:
            lbl_base = ""
            lbl_bire = ""
            lbl_add = ""
            l2l_add = ""
            lbl_da = ""
            lbl_de = ""
            lbl_ta = ""
        # plt.plot(bank_deg,np.rad2deg(BASE.u_trim[2]),"o",ls="none",c="k",
        #     label=lbl_base)
        atb.plot(bank_deg,np.rad2deg(BIRE.u_trim[2]),"o",ls="none",c=cbr,
            label=lbl_bire+lbl_add)
        atb_y2.plot(bank_deg,np.rad2deg(BIRE.u_trim[0]),"o",ls="none",c="k",
            label=lbl_da)
        atb_y2.plot(bank_deg,np.rad2deg(BIRE.u_trim[1]),"o",ls="none",c="m",
            label=lbl_de)
        atb_y2.plot(bank_deg,BIRE.u_trim[3]*10.,"o",ls="none",c="k",
            label=lbl_ta)
        atb.plot(bank_deg,np.rad2deg(BIR2.u_trim[2]),"o",ls="none",c=c2r,
            mfc="None")#,label=lbl_bire+l2l_add)
        atb_y2.plot(bank_deg,np.rad2deg(BIR2.u_trim[0]),"o",ls="none",c="k",
            mfc="None")#,label=lbl_da)
        atb_y2.plot(bank_deg,np.rad2deg(BIR2.u_trim[1]),"o",ls="none",c="m",
            mfc="None")#,label=lbl_de)
        atb_y2.plot(bank_deg,BIR2.u_trim[3]*10.,"o",ls="none",c="k",
            mfc="None")#,label=lbl_ta)
    
    atb_y2.set_ylim((-5.,11.))
    atb.set_xlabel("Bank angle [deg]")
    atb.set_ylabel("BIRE Deflection angle [deg]")
    atb_y2.set_ylabel("non-BIRE control [deg,%/5]")
    h0,_ = atb.get_legend_handles_labels()
    h1,_ = atb_y2.get_legend_handles_labels()
    atb.legend(handles=h0 + h1,loc="center left")
    plt.show()



    # plot eigvals at various bank angles
    numbanks = 60
    banks = np.deg2rad(np.linspace(0.,30.,num=numbanks))
    run_bire_trim = True
    if run_bire_trim:
        craft = Aircraft(bire_dict)
    else:
        craft = Aircraft(base_dict)
    mrrc = None # [2,3] #
    eigs = np.zeros((9,numbanks),dtype=complex)
    for i,phi in enumerate(banks):
        craft.phi_trim = phi
        craft._initialize_state()
        craft._build_controller(report=False,save_matrices=False,
        mrrr=[6,7,11],mrrc=mrrc,drop_actrs=True,
            include_stall_derivatives=False,run_freq=False,skip_reporting=True)
        # print(craft.Lin_Model.A_min.shape,craft.Lin_Model.B_min.shape)
        print(craft.Lin_Model.A_min_eigs)
        eigs[:,i] = np.sort(craft.Lin_Model.A_min_eigs)
    print();print()
    print(eigs)
    cs = ["r","k","g","b","0.5","m","lightblue","pink","teal"]
    for j in range(9):
        # print(j,np.real(eigs[j,:]),np.imag(eigs[j,:]))
        for k in range(numbanks):
            plt.plot(np.real(eigs[j,k]),np.imag(eigs[j,k]),"o",c=cs[j],ms=(k/numbanks)*8.0+2.0)
    plt.show()
    quit()
        

