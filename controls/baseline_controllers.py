import numpy as np
import json
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import mpl_toolkits.mplot3d.axes3d as ax3
from matplotlib.animation import FuncAnimation
from numpy import sign
import control as co
from scipy.linalg import block_diag
from scipy.integrate import ode, odeint
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.io import savemat, loadmat
from scipy.signal import tf2zpk as scipy_tf2zpk
from math import pi, sin, cos, tan, exp, asin, atan2
from std_atm import stdatm_english
from quat import quat_mult, euler_2_quat, quat_2_euler, quat_norm, body_2_fixed, fixed_2_body, eulerdot_2_quatdot, quatdot_2_eulerdot
from linearization import linearization as lin,Anderson_correction_der_coeff,Anderson_correction_der_M

# import sys
# aero_directory = '../aerodynamics_model/'
# # loads_directory = '../trim/'
# mass_directory = '../mass_properties/'
# turb_directory = '../turbulence_models/'

# sys.path.insert(1, aero_directory)
# # sys.path.insert(1, loads_directory)
# sys.path.insert(1, mass_directory)
# sys.path.insert(1, turb_directory)

# from os import mkdir, rmdir, walk, remove, listdir
# from os.path import exists as path_exists

# from f16_aero import F16Aero
# from bire_aero import BIREAero
# from thrust import Propulsion
# from inertia_model import InertiaModel
# from turbulence import ZeroTurbulence, DampedSinusoidGust, VonKarmanTurbulence
# from hunsaker_atm import stdatm_english as stdatm_hunsaker, gravity_english
from controller_simulation import Aircraft,run_single_simulation, \
    monte_carlo_perturbations, report_latex, report_eigprops, rep2D


class FeedbackLinearizationAircraft(Aircraft):
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
        self.aero_val = 0
        self.SCT_at_2s = False
    
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
                ref = self._get_reference(t)
                if self.sct_on_5:
                    _,g,_,_,_,_ = self.stdatm(-x[8])
                    ph = x_euler[ 9]
                    th = x_euler[10]
                    st = np.sin(th); ct = np.cos(th)
                    sp = np.sin(ph); cp = np.cos(ph)
                    frac = g*sp*ct/(x[0]*ct*cp + x[2]*st)
                    ref[4] = frac*sp*ct
                    ref[5] = frac*cp*ct
                    # print(t,np.rad2deg(ref[4]),np.rad2deg(ref[5]))
                ref = ref[self.Lin_Model.Cslice]
                # per dave, full stick should be 270 deg/s in aileron
                # 120 deg/s in elevator
                # 60 deg/s in rudder
                #

                # feedback linearization
                #-------------------#
                # STATE DEFINITIONS #
                #-------------------#
                V_xb    = x_euler[0]
                V_yb    = x_euler[1]
                V_zb    = x_euler[2]
                p       = x_euler[3]
                q       = x_euler[4]
                r       = x_euler[5]
                # x_f     = x_euler[6]
                # y_f     = x_euler[7]
                z_f     = x_euler[8]
                # phi     = x_euler[9]
                # theta   = x_euler[10]
                # psi     = x_euler[11]
                dB      = x_euler[14]
                epI     = x_euler[self.xIi_eul[0]]
                eqI     = x_euler[self.xIi_eul[1]]
                erI     = x_euler[self.xIi_eul[2]]
                # Derived Quantities
                V_tot   = np.sqrt(V_xb**2+V_yb**2+V_zb**2)
                V_xb_ss = self.x_trim[0]
                V_yb_ss = self.x_trim[1]
                V_zb_ss = self.x_trim[2]
                V_ss    = np.sqrt(V_xb_ss**2+V_yb_ss**2+V_zb_ss**2)
                aero = self.aero_val
                if aero == 0:
                    alpha   = np.arctan2(V_zb,V_xb)
                    beta    = sin(V_yb/V_tot)
                    V = V_tot
                elif aero == 1:
                    alpha   = 0.0
                    beta    = 0.0
                    V = V_tot
                elif aero == 2:
                    alpha = np.arctan2(V_zb_ss,V_xb_ss)
                    beta  = sin(V_yb_ss/V_ss)
                    V     = V_tot
                elif aero == 3:
                    alpha = np.arctan2(V_zb_ss,V_xb_ss)
                    beta  = sin(V_yb_ss/V_ss)
                    V = V_xb
                elif aero == 4:
                    alpha = np.arctan2(V_zb_ss,V_xb_ss)
                    beta  = sin(V_yb_ss/V_ss)
                    V = V_ss
                    if self.SCT_at_2s and t > 2.0:
                        alpha = np.deg2rad(6.4817083551896602)
                        beta = np.deg2rad(0.0337237957384821)
                _,g,_,_,rho,_ = self.stdatm(-z_f)
                # other componets
                S_w = self.Sw
                b_w = self.bw
                cbar_w = self.cw
                h_xb,h_yb,h_zb = self.inertia_model.angular_momentum_results()
                H = np.array([
                    [0, -h_zb, h_yb], [h_zb, 0, -h_xb], [-h_yb, h_xb, 0]])
                # Weight and inertia (See Table A.2)
                # W = self.inertia_model.W
                Ixx,Iyy,Izz,Ixy,Ixz,Iyz = \
                    self.inertia_model.inertia_results(dB)
                # angles
                Sa = np.sin(alpha); Ca = np.cos(alpha)
                Sb = np.sin( beta); Cb = np.cos( beta)

                # define aero derivs
                # force
                C_L_0 = self.aero_model.CL0
                C_L_alpha = self.aero_model.CLa
                C_L_1 = C_L_0 + C_L_alpha*alpha
                C_L_qbar = self.aero_model.CLq
                C_L_de = self.aero_model.CLde
                C_S_beta = self.aero_model.CSb
                C_S_1 = C_S_beta*beta
                C_S_Lpbar = self.aero_model.CSLp
                C_S_pbar = self.aero_model.CSp
                C_S_rbar = self.aero_model.CSr
                C_S_da = self.aero_model.CSda
                C_S_dr = self.aero_model.CSdr
                C_D_0 = self.aero_model.CD0
                C_D_L = self.aero_model.CDL
                C_D_L2 = self.aero_model.CDL2
                C_D_S2 = self.aero_model.CDS2
                C_D_Spbar = self.aero_model.CDSp
                C_D_qbar = self.aero_model.CDq
                C_D_Lqbar = self.aero_model.CDLq
                C_D_L2qbar = self.aero_model.CDL2q
                C_D_Srbar = self.aero_model.CDSr
                C_D_Sda = self.aero_model.CDSda
                C_D_de = self.aero_model.CDde
                C_D_de2 = self.aero_model.CDde2
                C_D_Lde = self.aero_model.CDLde
                C_D_Sdr = self.aero_model.CDSdr
                # moment
                C_ell_beta = self.aero_model.Clb
                C_ell_pbar = self.aero_model.Clp
                C_ell_rbar = self.aero_model.Clr
                C_ell_Lrbar = self.aero_model.ClLr
                C_ell_delta_a = self.aero_model.Clda
                C_ell_delta_r = self.aero_model.Cldr
                C_m_0 = self.aero_model.Cm0
                C_m_alpha = self.aero_model.Cma
                C_m_qbar = self.aero_model.Cmq
                C_m_delta_e = self.aero_model.Cmde
                C_n_beta = self.aero_model.Cnb
                C_n_pbar = self.aero_model.Cnp
                C_n_Lpbar = self.aero_model.CnLp
                C_n_rbar = self.aero_model.Cnr
                C_n_delta_a = self.aero_model.Cnda
                C_n_Ldelta_a = self.aero_model.CnLda
                C_n_delta_r = self.aero_model.Cndr

                # define components
                # I    = self.inertia_model.inertia_tensor(dB)
                Iinv = self.inertia_model.inverse_tensor(dB)
                ref_diag = np.diag([b_w,cbar_w,b_w])
                G = 0.5*rho*V**2.*S_w*ref_diag
                C0ab = np.array([
                    C_ell_beta*beta,
                    C_m_0 + C_m_alpha*alpha,
                    C_n_beta*beta
                ])
                Cstates = 1./2./V*np.matmul(np.array([
                    [C_ell_pbar, 0, (C_ell_Lrbar*C_L_1 + C_ell_rbar)],
                    [0, C_m_qbar, 0],
                    [(C_n_Lpbar*C_L_1+C_n_pbar), 0, C_n_rbar]
                ]),ref_diag)
                ##
                Dcg = self.cgshift
                Delta = np.array([
                    [   0.0, Dcg[2],-Dcg[1]],
                    [-Dcg[2],   0.0, Dcg[0]],
                    [ Dcg[1],-Dcg[0],   0.0]
                ])
                Theta = 0.5*rho*V**2.*S_w*np.array([
                    [    Sa,-Ca*Sb,-Ca*Cb],
                    [   0.0,    Cb,   -Sb],
                    [   -Ca,-Sa*Sb,-Sa*Cb]
                ])
                CF0 = np.array([
                    C_L_1,
                    C_S_beta*beta,
                    C_D_0 + (C_D_L + C_D_L2*C_L_1)*C_L_1 + C_D_S2*C_S_1*C_S_1
                ])
                CFs = 1./2./V*np.matmul(np.array([
                    [0.0, C_L_qbar, 0.0],
                    [(C_S_Lpbar*C_L_1 + C_S_pbar), 0.0, C_S_rbar],
                    [C_D_Spbar*C_S_1, 
                        ((C_D_L2qbar*C_L_1 + C_D_Lqbar)*C_L_1 + C_D_qbar), 
                        C_D_Srbar*C_S_1]
                ]),ref_diag)
                
                ###############################################################
                w  = np.array([ p, q, r])
                eI = np.array([epI,eqI,erI])
                Omega = np.array([
                    (Iyy-Izz)*q*r + Iyz*(q**2-r**2) + Ixz*p*q - Ixy*p*r,
                    (Izz-Ixx)*p*r + Ixz*(r**2-p**2) + Ixy*q*r - Iyz*p*q,
                    (Ixx-Iyy)*p*q + Ixy*(p**2-q**2) + Iyz*p*r - Ixz*q*r])
                Ccontrol = np.array([
                    [C_ell_delta_a, 0.0, C_ell_delta_r],
                    [0.0, C_m_delta_e, 0.0],
                    [(C_n_Ldelta_a*C_L_1 + C_n_delta_a), 0.0, C_n_delta_r]
                ])
                CFc = np.array([
                    [0.0, C_L_de, 0.0],
                    [C_S_da, 0.0, C_S_dr],
                    [C_D_Sda*C_S_1, 
                        (C_D_Lde*C_L_1 + C_D_de), C_D_Sdr*C_S_1] # + 2.*C_D_de2
                ])
                Tv = np.array([
                    self.aero_model.get_thrust(self.u_trim[3],-z_f,V),0.0,0.0
                ])

                # build linear system
                DT = np.matmul(Delta,Theta)
                A    = np.matmul(Iinv,
                    np.matmul(G,Cstates) + np.matmul(DT,CFs) + H)
                B    = np.matmul(Iinv,np.matmul(G,Ccontrol) +np.matmul(DT,CFc))
                eta  = np.matmul(Iinv,np.matmul(G,C0ab) 
                    + np.matmul(DT,CF0) + np.matmul(Delta,Tv) + Omega)
                Binv = np.linalg.solve(B,np.eye(3))

                # define controller
                e = w - ref
                # print(A,B,Binv,eta,e)
                # print()
                v = - np.matmul(self.Lin_Model.K,e) \
                    - np.matmul(self.Lin_Model.KI,eI)
                # zt = 0.7
                # wn = 10.0
                # v = - np.matmul(np.diag([2.*zt*wn]*3),ref-w) \
                #     - np.matmul(np.diag([wn**2.]*3),wI)
                refdot = 0.0
                delta = np.matmul(Binv, \
                    - np.matmul(A,e) - np.matmul(A,ref) - eta + refdot + v)
                u = np.concatenate((delta,[self.u_trim[3]]))


                # # integral states
                # if len(self.xIi_eul):
                #     u[uslc] = u[uslc] - np.matmul(K_I,intg)
                #     # u[uslc] = u[uslc] - [kI*intg[0],kI*intg[1],kI*intg[2]]
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
        # u = self._limit_input(u)
        inputs = self._limit_input(inputs)
        if self.order > 0:
            q = 1*self.use_quaternions
            x[12+q:16+q] = np.array(inputs)*1.
        # quantize actuators
        inputs = self._quantize_input(inputs)

        return u,inputs


class DynamicInversionAircraft(Aircraft):
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
        self.about_SCT = False # True # 
        self.is_MC = True # False # 
    
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
                # per dave, full stick should be 270 deg/s in aileron
                # 120 deg/s in elevator
                # 60 deg/s in rudder
                #

                # feedback linearization

                #-------------------#
                # STATE DEFINITIONS #
                #-------------------#
                p       = x_euler[3]
                q       = x_euler[4]
                r       = x_euler[5]
                epI     = x_euler[self.xIi_eul[0]]
                eqI     = x_euler[self.xIi_eul[1]]
                erI     = x_euler[self.xIi_eul[2]]
                w  = np.array([  p,  q,  r])
                eI = np.array([epI,eqI,erI])
                if self.about_SCT: # t > 2.0 and 
                    x_trim = self.x_sct_trim_euler
                else:
                    x_trim = self.x_trim
                dref = ref - x_trim[3:6]
                e = w - ref
                A = self.Lin_Model.A_min
                Binv = self.Lin_Model.Binv_min
                v = - np.matmul(self.Lin_Model.K,e) \
                    - np.matmul(self.Lin_Model.KI,eI)
                
                delta = np.matmul(Binv,-np.matmul(A,e) - np.matmul(A,dref) + v)
                u = np.concatenate((delta + self.u_trim[0:3],[self.u_trim[3]]))


                # # integral states
                # if len(self.xIi_eul):
                #     u[uslc] = u[uslc] - np.matmul(K_I,intg)
                #     # u[uslc] = u[uslc] - [kI*intg[0],kI*intg[1],kI*intg[2]]
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
        # u = self._limit_input(u)
        inputs = self._limit_input(inputs)
        if self.order > 0:
            q = 1*self.use_quaternions
            x[12+q:16+q] = np.array(inputs)*1.
        # quantize actuators
        inputs = self._quantize_input(inputs)

        return u,inputs


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
            # initialize run2 trim (SCT w/ phitrim = 30 deg)
            dum = self.phi_trim*1.
            if (self.is_MC and not self.about_SCT) or \
                (not self.is_MC and not self.about_SCT):
                self.phi_trim = 0.0
            else:
                self.phi_trim = np.deg2rad(30.0)
            self._initialize_state()
            self.x_sct_trim_euler = self.x_trim_euler*1.
            self.u_sct_trim = self.u_trim*1.
            self.phi_trim = dum
            # return to original state
            self._initialize_state(self.a_guess,self.b_guess,self.phi_guess,
                self.u_guess)
            if self.is_MC or self.about_SCT: # True: # 
                x_trim_euler = self.x_sct_trim_euler*1.
                u_trim = self.u_sct_trim*1.
            else:
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



class LinearAdaptiveAircraft(Aircraft):
    """A default class for calculating and containing the mass properties of a
    Cuboid.

    Parameters
    ----------
    input_vars : dict , optional
        Must be a python dictionary
    """
    def __init__(self,input_dict={}):

        # build Adaptive Controller
        Abar = np.array([[-12.,0.,3.],[0.,-3.,0.],[0.,0.,-1.]])
        Bbar = np.array([[-150.,0.,0.],[0.,-60.,0.],[0.,0.,-25.]])
        # Bbar = np.array([[-150.,0.,30.],[0.,-60.,0.],[-6.,0.,-25.]])
        self.Am = np.diag([-2.]*3) # np.diag([-18.] + [-9.] + [-27.]) # 
        self.Q  = np.diag([45.]*3) # np.diag([45.]*3) # 
        self.P = co.lyap(self.Am.T,self.Q)
        Bbarinv = np.linalg.solve(Bbar,np.eye(3))
        self.Lshape = (3,3)
        self.Lflat = (9,)
        self.Lhat0 = Bbarinv.reshape(self.Lflat)
        self.Kshape = (3,3)
        self.Kflat = (9,)
        self.Khat0 = np.matmul(Bbarinv,Abar).reshape(self.Kflat)

        # invoke init of parent
        Aircraft.__init__(self,input_dict,folder_prefix = "track")
        self.tracking = True
        self.additional_states = self.Kflat[0] + self.Lflat[0]
        Kstart = self.x_trim.shape[0] - self.additional_states
        Lstart = self.x_trim.shape[0] - self.Lflat[0]
        self.Kinds = list(range(Kstart,Lstart))
        self.Linds = list(range(Lstart,self.x_trim.shape[0]))
        self.Kinds_eul = list(range(Kstart-1,Lstart-1))
        self.Linds_eul = list(range(Lstart-1,self.x_trim.shape[0]-1))

        # add in additional states to ref
        self.r_ints += [lambda j,t_i : 0.0]*self.additional_states
    
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
                # per dave, full stick should be 270 deg/s in aileron
                # 120 deg/s in elevator
                # 60 deg/s in rudder
                #

                # adaptive controller
                #-------------------#
                # STATE DEFINITIONS #
                #-------------------#
                p       = x_euler[3]
                q       = x_euler[4]
                r       = x_euler[5]
                K = x_euler[self.Kinds_eul].reshape(self.Kshape)
                L = x_euler[self.Linds_eul].reshape(self.Lshape)
                w  = np.array([  p,  q,  r])
                Dw = w - self.x_trim_euler[3:6]
                Dwref = ref - self.x_trim_euler[3:6]
                drefdot = 0.0
                e = Dw - Dwref
                
                delta = - np.matmul(K,Dw) \
                    + np.matmul(L,np.matmul(self.Am,e) + drefdot)
                u = np.concatenate((delta + self.u_trim[0:3],[self.u_trim[3]]))


                # # integral states
                # if len(self.xIi_eul):
                #     u[uslc] = u[uslc] - np.matmul(K_I,intg)
                #     # u[uslc] = u[uslc] - [kI*intg[0],kI*intg[1],kI*intg[2]]
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
        # u = self._limit_input(u)
        inputs = self._limit_input(inputs)
        if self.order > 0:
            q = 1*self.use_quaternions
            x[12+q:16+q] = np.array(inputs)*1.
        # quantize actuators
        inputs = self._quantize_input(inputs)

        return u,inputs


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
        
        # add in L0 and K0
        x_trim = np.concatenate((x_trim,self.Khat0,self.Lhat0))
        x_trim_euler = np.concatenate((x_trim_euler,self.Khat0,self.Lhat0))
        x_trim_euler_deg = np.concatenate((x_trim_euler_deg,self.Khat0,self.Lhat0))
        #
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
        elif self.order == 2:
            dx[13:21] = self._actuation_dynamics(x,u)
        
        # integral states
        r = self._get_reference(t)[self.xPi]
        e = x[self.xPi] - r
        dx[self.xIi] = e

        # Khat
        eta = -1.
        dw = (x[self.xPi] - self.x_trim[self.xPi])[:,np.newaxis]
        dwref = (r - self.x_trim[self.xPi])[:,np.newaxis]
        e = dw - dwref
        Khatdot = eta*np.matmul(np.matmul(self.P,e),dw.T)
        dx[self.Kinds] = Khatdot.reshape(self.Kflat)

        # Lhat
        dwrefdot = 0.0
        Lhatdot = -eta*np.matmul(self.P,np.matmul(e,(np.matmul(self.Am,e) 
            + dwrefdot).T))
        dx[self.Linds] = Lhatdot.reshape(self.Lflat)

        return dx



class ModelReferenceAdaptiveAircraft(Aircraft):
    """A default class for calculating and containing the mass properties of a
    Cuboid.

    Parameters
    ----------
    input_vars : dict , optional
        Must be a python dictionary
    """
    def __init__(self,input_dict={}):

        # build Adaptive Controller
        Abar = np.array([[-12.,0.,3.],[0.,-3.,0.],[0.,0.,-1.]])
        Bbar = np.array([[-150.,0.,0.],[0.,-60.,0.],[0.,0.,-25.]])
        self.Am = np.diag([-5.] + [-2.]*2)
        self.Bm = np.diag([1.]*3)
        self.Q  = np.diag([10.] + [5.]*2)
        self.P = co.lyap(self.Am.T,self.Q)
        self.K_MRAC = np.diag([0.5]*3) # np.diag([2.]*3) # 
        Bbarinv = np.linalg.solve(Bbar,np.eye(3))
        self.Bminv = np.linalg.solve(self.Bm,np.eye(3))
        self.Lshape = (3,3)
        self.Lflat = (9,)
        self.Lhat0 = np.matmul(Bbarinv,self.Bm).reshape(self.Lflat)
        self.Kshape = (3,3)
        self.Kflat = (9,)
        self.Khat0 = np.matmul(Bbarinv,Abar - self.Am).reshape(self.Kflat)
        self.xm0 = np.zeros((3,))

        # invoke init of parent
        Aircraft.__init__(self,input_dict,folder_prefix = "track")
        self.tracking = True
        self.additional_states = len(self.xPi) + self.Kflat[0] + self.Lflat[0]
        xmstart = self.x_trim.shape[0] - self.additional_states
        Kstart = self.x_trim.shape[0] - self.additional_states + len(self.xPi)
        Lstart = self.x_trim.shape[0] - self.Lflat[0]
        self.xminds = list(range(xmstart,Kstart))
        self.Kinds = list(range(Kstart,Lstart))
        self.Linds = list(range(Lstart,self.x_trim.shape[0]))
        self.xminds_eul = list(range(xmstart-1,Kstart-1))
        self.Kinds_eul = list(range(Kstart-1,Lstart-1))
        self.Linds_eul = list(range(Lstart-1,self.x_trim.shape[0]-1))

        # add in additional states to ref
        self.r_ints += [lambda j,t_i : 0.0]*self.additional_states
    
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
                # per dave, full stick should be 270 deg/s in aileron
                # 120 deg/s in elevator
                # 60 deg/s in rudder
                #

                # adaptive controller
                #-------------------#
                # STATE DEFINITIONS #
                #-------------------#
                p       = x_euler[3]
                q       = x_euler[4]
                r       = x_euler[5]
                xm = x_euler[self.xminds_eul]
                Khat = x_euler[self.Kinds_eul].reshape(self.Kshape)
                Lhat = x_euler[self.Linds_eul].reshape(self.Lshape)
                w  = np.array([  p,  q,  r])
                um = np.matmul(self.Bminv,(-np.matmul(self.Am,xm) 
                    - np.matmul(self.K_MRAC,(xm - ref + self.x_trim[self.xPi]))))
                
                delta = - np.matmul(Khat,w - self.x_trim[self.xPi]) \
                    + np.matmul(Lhat,um)
                u = np.concatenate((delta + self.u_trim[0:3],[self.u_trim[3]]))


                # # integral states
                # if len(self.xIi_eul):
                #     u[uslc] = u[uslc] - np.matmul(K_I,intg)
                #     # u[uslc] = u[uslc] - [kI*intg[0],kI*intg[1],kI*intg[2]]
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
        # u = self._limit_input(u)
        inputs = self._limit_input(inputs)
        if self.order > 0:
            q = 1*self.use_quaternions
            x[12+q:16+q] = np.array(inputs)*1.
        # quantize actuators
        inputs = self._quantize_input(inputs)

        return u,inputs


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
        
        # add in L0 and K0
        x_trim = np.concatenate((x_trim,self.xm0,self.Khat0,self.Lhat0))
        x_trim_euler = np.concatenate((x_trim_euler,self.xm0,self.Khat0,self.Lhat0))
        x_trim_euler_deg = np.concatenate((x_trim_euler_deg,self.xm0,self.Khat0,self.Lhat0))
        #
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
        elif self.order == 2:
            dx[13:21] = self._actuation_dynamics(x,u)
        
        # integral states
        r = self._get_reference(t)[self.xPi]
        dx[self.xIi] = x[self.xPi] - r

        # xm
        xm = x[self.xminds]
        um = np.matmul(self.Bminv,(-np.matmul(self.Am,xm) 
            - np.matmul(self.K_MRAC,(xm - r + self.x_trim[self.xPi]))))
        dx[self.xminds] = np.matmul(self.Am,xm) + np.matmul(self.Bm,um)

        # Khat
        eta = -1.
        dw = (x[self.xPi] - self.x_trim[self.xPi])[:,np.newaxis]
        e = dw - xm[:,np.newaxis] # + self.x_trim[self.xPi]
        Khatdot = eta*np.matmul(np.matmul(np.matmul(self.Bm.T,self.P),e),dw.T)
        dx[self.Kinds] = Khatdot.reshape(self.Kflat)

        # Lhat
        um = um[:,np.newaxis]
        Lhatdot = -eta*np.matmul(self.Bm.T,np.matmul(self.P,np.matmul(e,um.T)))
        dx[self.Linds] = Lhatdot.reshape(self.Lflat)

        return dx


    def _add_to_delta_x0(self,delta_x0):
        delta_x0[self.xminds] = delta_x0[self.xPi]*1.
        return delta_x0

    

if __name__ == "__main__":

    # filenames 
    base_fs_file = "base_fs_in.json"
    bire_fs_file = "bire_fs_in.json"
    base_rc_file = "base_rc_in.json"
    bire_rc_file = "bire_rc_in.json"

    # read in json to ensure no file changes while running
    base_fs_dict = json.loads( open(base_fs_file).read() )
    bire_fs_dict = json.loads( open(bire_fs_file).read() )
    base_rc_dict = json.loads( open(base_rc_file).read() )
    bire_rc_dict = json.loads( open(bire_rc_file).read() )

    # # build controller
    # build_base_controller(base_rc_dict,"RC_base_control_design")
    # quit()

    plot_vars = {
        "show" : False,
        "plot_full" : True,
        "plot_delta" : True,
        "zoom_deltas" : True,
        # "zoom_fraction" : 0.05,
        "zoom_fraction" : 0.13333333333333333333333,
        "transparent" : False,
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
    # RC
    # bire FM
    bire_rc_FM_errs = [
        0.25  , # CL
        0.25  , # CS
        0.25  , # CD
        0.25  , # Cl
        0.25  , # Cm
        0.25   # Cn
    ]
    # base FM
    base_rc_FM_errs = [
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
        "C3" : { "m" : 0.8 , "h" : 30000., "V" : 796., "Re" : 25828000. }
    }
    f1 = "C2"
    f2 = "C3"
    state_threshold = [
        10., 15., 15.,
        0.5, 0.5, 0.5, # 20., 10., 10., # 
        1., 1., 50.,
        25., 10., 1.,
        5., 5., 5., 0.05
    ]

    run_base_fs = {
        "aircraft_class" : DynamicInversionAircraft,
        "actr_warm_start" : False,
        "num" : 1000,
        "final_time" : 5., # 120., # 
        "track_check_time" : 1.,
        # "time_step" : 0.01,
        "initial_velocity" : 100.,
        "initial_mach" : flight_conditions[f1]["m"],
        "initial_altitude" : flight_conditions[f1]["h"], # 4500., # 
        "trim_bank" : 0.0,
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
        "include_stall_derivatives" : False, # True, # 
        "skip_simulation" : False, # True, # 
        "skip_video" : True, # False, # 
        "name_end" : "_" + f1 + "_DI_1" # "_MRAC_2" # "_LAC_1" # "_TK_8" # 
        # "name_end" : "_" + f1 + "_TK_8_no_act_no_cgshift"
        # 4 -- incr wt on tau, decr wt on da,de
        # 5 -- decr wt on da
    }
    run_bire_fs = {**run_base_fs}
    run_bire_fs["FM_errors"] = bire_fs_FM_errs
    run_base_rc = {**run_base_fs}
    run_base_rc.pop("initial_mach")
    run_base_rc["initial_velocity"] = 100.
    run_base_rc["initial_altitude"] = 4500.
    run_base_rc["FM_errors"] = base_rc_FM_errs
    run_base_rc["name_end"] = "_" + "LGN" + run_base_fs["name_end"][3:]
    run_bire_rc = {**run_base_rc}
    run_bire_rc["FM_errors"] = bire_rc_FM_errs
    run_bire_rc["mrrc"] = [3]

    # per dave, max throws would be p=270deg/s,q=120deg/s,r=60deg/s
    # from 2nd to last flight test:
    # about 1/6 throw was max commanded in flight
    # run_base_rc["track_check_time"] = 10.0
    # # 30 deg bank no cgshift
    # p_tr_deg = -0.9325759719346527
    # q_tr_deg =  5.2711291315150177
    # r_tr_deg =  9.1298634690404246
    # 30 deg bank cgshift = [0.126,0,0]
    p_tr_deg = -1.0380901589473488
    q_tr_deg =  5.2594941135694429
    r_tr_deg =  9.1097110268117127
    p_comm = 22.0 # 15.0 # 23.0 # 17.0 # 20.0 # 
    base_rc_dict["reference"] = {
        "deg2rad_states" : [3,4,5],
        "3" : [
            [ 0.0,p_comm],
            [ 2.0,p_comm],
            [ 2.0, p_tr_deg]
        ],
        "4" : [
            [ 0.0, 0.0],
            [ 2.0, 0.0],
            [ 2.0, q_tr_deg]
        ],
        "5" : [
            [ 0.0, 0.0],
            [ 2.0, 0.0],
            [ 2.0, r_tr_deg]
        ],
        "sct_on_5" : False
    }
    # # ZERO COMMANDED RATES
    # base_rc_dict["reference"] = {
    #     "deg2rad_states" : [3,4,5],
    #     "3" : [ [ 0.0, 0.0], [ 2.0, 0.0] ],
    #     "4" : [ [ 0.0, 0.0], [ 2.0, 0.0] ],
    #     "5" : [ [ 0.0, 0.0], [ 2.0, 0.0] ],
    #     "sct_on_5" : False
    # }

    # run single case
    # di = [-1000.,0.,0.] # 
    # di = [-750.,0.,0.] # 
    # di = [-500.,0.,0.] # 
    # di = [-250.,0.,0.] # 
    # di = [10.,2.,2.]
    # # 
    plot_vars["plot_full"] = True # False # 
    plot_vars["plot_delta"] = False # True # 
    plot_vars["zoom_deltas"] = False
    plot_vars["format"] = "png" # "pdf" # 
    plot_vars["format"] = "pdf" # "png" # 
    plot_vars["plot_norm"] = False # True # 
    #
    di = [0.,0.,0.]
    # di = [0.,1.,0.]
    # di = [14.,10.,4.] # see below
    # di = [0.,0.,1.]
    run_base_fs["num"] = run_bire_fs["num"] = \
        run_base_rc["num"] = run_bire_rc["num"] = 1  
    ##
    # # TK_6
    # zt_p,zt_q,zt_r =  0.7 , 0.7 , 0.7 # Me
    # wn_p,wn_q,wn_r =  8.0 , 8.0 , 8.0 # Me
    # run_base_rc["aircraft_class"] = FeedbackLinearizationAircraft
    # # TK_8 & TK_9 & TK_11
    # zt_p,zt_q,zt_r =  0.6 , 0.6 , 0.6 # Dr Harris
    # wn_p,wn_q,wn_r =  8.0 , 8.0 , 8.0 # Dr Harris
    # # TK_10
    # zt_p,zt_q,zt_r =  0.7 , 0.7 , 0.7 # Me
    # wn_p,wn_q,wn_r =  8.0 , 8.0 , 8.0 # Me
    #
    run_base_rc["aircraft_class"] = DynamicInversionAircraft
    # DI_1 & DI_2
    zt_p,zt_q,zt_r =  0.6 , 0.6 , 0.6 # Dr Harris
    wn_p,wn_q,wn_r =  8.0 , 8.0 , 8.0 # Dr Harris
    
    # base_rc_dict["controller"]["gains"][ "K"] = \
    #     bire_rc_dict["controller"]["gains"][ "K"] = np.diag([
    #     2.*zt_p*wn_p,2.*zt_q*wn_q,2.*zt_r*wn_r
    # ]).tolist()
    # base_rc_dict["controller"]["gains"]["KI"] = \
    #     bire_rc_dict["controller"]["gains"]["KI"] = np.diag([
    #     wn_p**2.,wn_q**2.,wn_r**2.
    # ]).tolist()
    # run_base_rc["aircraft_class"] = LinearAdaptiveAircraft
    # # LAC_1
    # run_base_rc["state_threshold"] += [1.]*18
    # run_base_rc["aircraft_class"] = ModelReferenceAdaptiveAircraft
    # # MRAC_1
    # run_base_rc["state_threshold"] += [1.]*21
    # base_rc_dict["simulation"]["integrator"] = "rk4" 
    run_base_rc["track_check_time"] = \
        run_base_fs["final_time"] = run_base_rc["final_time"] = 10.0 # 10.0 # 
    # base_rc_dict["simulation"]["include_stall"] = \
    #     bire_rc_dict["simulation"]["include_stall"] = False
    # base_rc_dict["actuators"]["order"] = 0
    # run_base_rc["state_threshold"] = [
    #     10., 15., 15.,
    #     0.5, 0.5, 0.5, # 20., 10., 10., # 
    #     1., 1., 50.,
    #     25., 10., 1.,
    #     # 5., 5., 5., 0.05
    # ] # + [1.]*18 #  # 
    # base_rc_dict["aircraft"]["CG_shift[ft]"] = [0.]*3
    # run_base_rc["has_turbulence"] = True
    # #########################################################################
    # base_rc_dict["reference"] = {
    #     "deg2rad_states" : [3,4,5],
    #     "3" : [[ 0.0, p_tr_deg],[ 2.0, p_tr_deg]],
    #     "4" : [[ 0.0, q_tr_deg],[ 2.0, q_tr_deg]],
    #     "5" : [[ 0.0, r_tr_deg],[ 2.0, r_tr_deg]],
    #     "sct_on_5" : False
    # }
    # run_base_rc["trim_bank"] = 30.0
    # run_single_simulation(bire_fs_dict,rtdst_1sg=di,**run_bire_fs,**plot_vars)
    run_single_simulation(base_fs_dict,rtdst_1sg=di,**run_base_fs,**plot_vars)
    # run_single_simulation(bire_rc_dict,rtdst_1sg=di,**run_bire_rc,**plot_vars)
    # run_single_simulation(base_rc_dict,rtdst_1sg=di,**run_base_rc,**plot_vars)
    quit()

    # # # # run monte carlo perturbation analysis
    # base_rc_dict["reference"] = {
    #     "deg2rad_states" : [3,4,5],
    #     "3" : [[ 0.0, p_tr_deg],[ 2.0, p_tr_deg]],
    #     "4" : [[ 0.0, q_tr_deg],[ 2.0, q_tr_deg]],
    #     "5" : [[ 0.0, r_tr_deg],[ 2.0, r_tr_deg]],
    #     "sct_on_5" : False
    # }
    # run_base_rc["trim_bank"] = 30.0
    # num = 1000
    # run_base_fs["num"] = run_bire_fs["num"] = \
    #     run_base_rc["num"] = run_bire_rc["num"] = num
    # # di = [5.,10.,7.] # RC F-16 # change r to 6
    # di = [900. ,130.,100.] # TK_8
    # di = [300. ,100., 60.] # TK_9
    # di = [300. ,130., 50.] # DI_1
    # di = [300. ,130., 50.] # DI_2
    # # di = [ 50. , 30.,  3.] # LAC_1
    # # di = [  1.5,  5.,  1.] # MRAC_1
    # # # di = [0.,0.,0.]
    # # monte_carlo_perturbations(bire_fs_dict,rtdst_1sg=di,**run_bire_fs,**plot_vars)
    # # monte_carlo_perturbations(base_fs_dict,rtdst_1sg=di,**run_base_fs,**plot_vars)
    # # monte_carlo_perturbations(bire_rc_dict,rtdst_1sg=di,**run_bire_rc,**plot_vars)
    # monte_carlo_perturbations(base_rc_dict,rtdst_1sg=di,**run_base_rc,**plot_vars)
    # quit()
    # #
    # # single axis pqr dispersions
    # ###########################################################################
    # base_rc_dict["reference"] = {
    #     "deg2rad_states" : [3,4,5],
    #     "3" : [[ 0.0, p_tr_deg],[ 2.0, p_tr_deg]],
    #     "4" : [[ 0.0, q_tr_deg],[ 2.0, q_tr_deg]],
    #     "5" : [[ 0.0, r_tr_deg],[ 2.0, r_tr_deg]],
    #     "sct_on_5" : False
    # }
    # run_base_rc["trim_bank"] = 30.0
    # run_base_fs["num"] = run_base_rc["num"] = 1000 # 10 # 
    # ###########################################################################
    # disa = [[1000.,0.,0.],[0.,300.,0.],[0.,0.,175.]] # TK_8
    # disa = [[ 500.,0.,0.],[0.,200.,0.],[0.,0.,120.]] # TK_9
    # disa = [[ 500.,0.,0.],[0.,200.,0.],[0.,0.,120.]] # DI_1
    # disa = [[ 500.,0.,0.],[0.,200.,0.],[0.,0.,120.]] # DI_2
    # disa = [[ 100.,0.,0.],[0., 50.,0.],[0.,0., 50.]] # LAC_1
    # disa = [[  20.,0.,0.],[0., 10.,0.],[0.,0., 10.]] # MRAC_1
    # disa = [[  20.,0.,0.],[0., 10.,0.],[0.,0., 10.]] # MRAC_2
    # for i in [0]: # range(3): # 
    #     ds = disa[i]
    #     # monte_carlo_perturbations(bire_fs_dict,rtdst_1sg=ds,**run_bire_fs,**plot_vars)
    #     # monte_carlo_perturbations(base_fs_dict,rtdst_1sg=ds,**run_base_fs,**plot_vars)
    #     # monte_carlo_perturbations(bire_rc_dict,rtdst_1sg=ds,**run_bire_rc,**plot_vars)
    #     monte_carlo_perturbations(base_rc_dict,rtdst_1sg=ds,**run_base_rc,**plot_vars)
    # quit()
    # #
    # # single FM error dispersions
    # names = ["CL","CS","CD","Cl","Cm","Cn"]
    # run_base_fs["has_model_error"] = run_bire_fs["has_model_error"] = \
    #     run_base_rc["has_model_error"] = run_bire_rc["has_model_error"] = True
    # f1 = "LGN"
    # for i in range(len(names)):
    #     name = names[i]
    #     # create FM errors
    #     FM_error_list = np.zeros((6,))
    #     FM_error_list[i] = 0.25
    #     run_base_fs["FM_errors"] = run_bire_fs["FM_errors"] = \
    #         run_base_rc["FM_errors"] = run_bire_rc["FM_errors"] = \
    #         FM_error_list*1.
    #     run_base_fs["name_end"] = run_bire_fs["name_end"] = \
    #         run_base_rc["name_end"] = run_bire_rc["name_end"] = \
    #         (f1 != "C2")*("_" + f1) + "_TK_3" + "_" + name
    #     # monte_carlo_perturbations(bire_fs_dict,rtdst_1sg=di,**run_bire_fs,**plot_vars)
    #     # monte_carlo_perturbations(base_fs_dict,rtdst_1sg=di,**run_base_fs,**plot_vars)
    #     # monte_carlo_perturbations(bire_rc_dict,rtdst_1sg=di,**run_bire_rc,**plot_vars)
    #     monte_carlo_perturbations(base_rc_dict,rtdst_1sg=di,**run_base_rc,**plot_vars)
    # quit()
    # #
    # #
    # # run for roa plots / diff controllers
    # run_bire_fs["has_turbulence"] = run_base_fs["has_turbulence"] = \
    #     run_bire_rc["has_turbulence"] = run_base_rc["has_turbulence"] = False # True # 
    # run_bire_fs["has_model_error"] = run_base_fs["has_model_error"] = \
    #     run_bire_rc["has_model_error"] = run_base_rc["has_model_error"] = True # False # 
    # # monte_carlo_perturbations(bire_fs_dict,rtdst_1sg=di,**run_bire_fs,**plot_vars)
    # # monte_carlo_perturbations(base_fs_dict,rtdst_1sg=di,**run_base_fs,**plot_vars)
    # # monte_carlo_perturbations(bire_rc_dict,rtdst_1sg=di,**run_bire_rc,**plot_vars)
    # monte_carlo_perturbations(base_rc_dict,rtdst_1sg=di,**run_base_rc,**plot_vars)
    # quit()
    # # # Turbulence cases
    # plot_vars["plot_full"] = True # False # 
    # plot_vars["plot_delta"] = True # False # 
    # plot_vars["zoom_deltas"] = False
    # plot_vars["format"] = "pdf" # "png" # 
    # plot_vars["plot_norm"] = False # True # 
    # di = [5.,10.,7.]
    # run_bire_fs["fixed_FM_errors"] = run_base_fs["fixed_FM_errors"] = \
    #     run_bire_rc["fixed_FM_errors"] = run_base_rc["fixed_FM_errors"] = \
    #     [0.1,0.1,0.1,0.1,0.1,0.1]
    # run_bire_fs["has_turbulence"] = run_base_fs["has_turbulence"] = \
    #     run_bire_rc["has_turbulence"] = run_base_rc["has_turbulence"] = False # True # 
    # run_bire_fs["has_model_error"] = run_base_fs["has_model_error"] = \
    #     run_bire_rc["has_model_error"] = run_base_rc["has_model_error"] = False # True # 
    # # run_base_fs["trim_bank"] = run_bire_fs["trim_bank"] = \
    # #     run_base_rc["trim_bank"] = run_bire_rc["trim_bank"] = 0.0
    # run_base_fs["num"] = run_bire_fs["num"] = \
    #     run_base_rc["num"] = run_bire_rc["num"] = 1
    # run_base_fs["skip_simulation"] = run_bire_fs["skip_simulation"] = \
    #     run_base_rc["skip_simulation"] = run_bire_rc["skip_simulation"] = False # True # 
    # # run_single_simulation(bire_fs_dict,rtdst_1sg=di,**run_bire_fs,**plot_vars)
    # # run_single_simulation(base_fs_dict,rtdst_1sg=di,**run_base_fs,**plot_vars)
    # # run_single_simulation(bire_rc_dict,rtdst_1sg=di,**run_bire_rc,**plot_vars)
    # run_single_simulation(base_rc_dict,rtdst_1sg=di,**run_base_rc,**plot_vars)
    # quit()



