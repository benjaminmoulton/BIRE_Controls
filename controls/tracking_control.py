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
from controller_simulation import Aircraft,run_single_simulation,\
    monte_carlo_perturbations, report_latex,rep2D


class TrackingAircraft(Aircraft):
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
                r = self._get_reference(t)[self.Lin_Model.Cslice]
                # per dave, full stick should be 270 deg/s in aileron
                # 120 deg/s in elevator
                # 60 deg/s in rudder
                #
                #
                # modify r for SCT
                if self.sct_on_5:
                    V = ( x[0]*x[0] + x[1]*x[1] + x[2]*x[2] )**0.5
                    r[2] = (self.g*np.sin(x_euler[9])*np.cos(x_euler[10]))/V
                
                # pull out parts of controller
                u_d = np.matmul(self.Lin_Model.nBiA_min,r)
                u_tr = self.Lin_Model.uhat_eq*1.
                K_tr = self.Lin_Model.K
                K_I = self.Lin_Model.KI
                u = self.u_trim*1.
                # kP = 9.8
                # kI = kP/0.2
                kD = 0.1 # kP*0.1
                ###################################
                err = r - x_euler[self.Lin_Model.Cslice]
                intg = x_euler[self.xIi_eul]
                # dt = t - self.t
                # intg = self.prev_integral + err*self.dt
                self.prev_integral = intg
                deriv = (err - self.prev_error) / self.dt
                self.prev_error = err
                uslc = self.Lin_Model.Cuslice
                #
                #
                u[uslc] = u_tr  + u_d - np.matmul(K_tr,err)
                # u[uslc] = u_tr  + u_d - [kP*err[0],kP*err[1],kP*err[2]]
                # integral states
                if len(self.xIi_eul):
                    u[uslc] = u[uslc] - np.matmul(K_I,intg)
                    # u[uslc] = u[uslc] - [kI*intg[0],kI*intg[1],kI*intg[2]]
                if self.order > 0:
                    q = 1*self.use_quaternions
                    inputs = x[12+q:16+q]*1.
                else:
                    inputs = u*1.
                # derivative states
                # u[uslc] = u[uslc] - [kD*deriv[0],kD*deriv[1],kD*deriv[2]]
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
                pI      = x_euler[self.xIi_eul[0]]
                qI      = x_euler[self.xIi_eul[1]]
                rI      = x_euler[self.xIi_eul[2]]
                # Derived Quantities
                V_tot   = np.sqrt(V_xb**2+V_yb**2+V_zb**2)
                V_xb_ss = self.x_trim[0]
                V_yb_ss = self.x_trim[1]
                V_zb_ss = self.x_trim[2]
                V_ss    = np.sqrt(V_xb_ss**2+V_yb_ss**2+V_zb_ss**2)
                aero = 0
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
                _,g,_,_,rho,_ = self.stdatm(-z_f)
                # other componets
                S_w = self.Sw
                b_w = self.bw
                cbar_w = self.cw
                h_xb,h_yb,h_zb = self.inertia_model.angular_momentum_results()
                hmat = np.array([
                    [0, -h_zb, h_yb], [h_zb, 0, -h_xb], [-h_yb, h_xb, 0]])
                # Weight and inertia (See Table A.2)
                W = self.inertia_model.W
                Ixx,Iyy,Izz,Ixy,Ixz,Iyz = \
                    self.inertia_model.inertia_results(dB)

                # define aero derivs
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
                C_L_0 = self.aero_model.CL0
                C_L_alpha = self.aero_model.CLa
                C_L_1 = C_L_0 + C_L_alpha*alpha
                pbar = p*b_w/(2*V)
                qbar = q*cbar_w/(2*V)
                rbar = r*b_w/(2*V)

                # define components
                I    = self.inertia_model.inertia_tensor(dB)
                Iinv = self.inertia_model.inverse_tensor(dB)
                G = 0.5*rho*V**2.*S_w*np.diag([b_w,cbar_w,b_w])
                Cstates = np.array([
                    C_ell_beta*beta + C_ell_pbar*pbar + \
                        (C_ell_Lrbar*C_L_1 + C_ell_rbar)*rbar,
                    C_m_0 + C_m_alpha*alpha + C_m_qbar*qbar,
                    C_n_beta*beta + (C_n_Lpbar*C_L_1+C_n_pbar)*pbar \
                        + C_n_rbar*rbar
                ])
                # print(Cstates)
                # print(self.aero_model.aero_results(alpha,beta,pbar,qbar,rbar,\
                #     0.,0.,0.,False,0.,False,False)[3:])
                # quit()
                ###############################################################
                w  = np.array([ p, q, r])
                wI = np.array([pI,qI,rI])
                Imult = np.array([
                    (Iyy-Izz)*q*r + Iyz*(q**2-r**2) + Ixz*p*q - Ixy*p*r,
                    (Izz-Ixx)*p*r + Ixz*(r**2-p**2) + Ixy*q*r - Iyz*p*q,
                    (Ixx-Iyy)*p*q + Ixy*(p**2-q**2) + Iyz*p*r - Ixz*q*r])
                Sigma = np.matmul(hmat,w) + Imult
                Ccontrol = np.array([
                    [C_ell_delta_a, 0.0, C_ell_delta_r],
                    [0.0, C_m_delta_e, 0.0],
                    [(C_n_Ldelta_a*C_L_1 + C_n_delta_a), 0.0, C_n_delta_r]
                ])
                GCcontrol = np.matmul(G,Ccontrol)
                IinvGCcontrol = np.matmul(Iinv,GCcontrol)
                IinvGCcontrol_inv = np.linalg.solve(IinvGCcontrol,np.eye(3))

                # # control law design
                # wdot = I^-1*(G*Cstates + Sigma) + I^-1*G*Ccontrol*delta
                # delta = (I^-1*G*Ccontrol)^-1 ( v - I^-1*(G*Cstates + Sigma) )
                # v = - k_p*p - k_q*q - k_r*r

                # define controller
                # print(ref)
                # print(self.Lin_Model.K)
                # print(self.Lin_Model.KI)
                # print()
                v = - np.matmul(self.Lin_Model.K,ref-w) \
                    - np.matmul(self.Lin_Model.KI,wI)
                # zt = 0.7
                # wn = 10.0
                # v = - np.matmul(np.diag([2.*zt*wn]*3),ref-w) \
                #     - np.matmul(np.diag([wn**2.]*3),wI)
                delta = - np.matmul(IinvGCcontrol_inv, \
                    ( v + np.matmul(Iinv,(np.matmul(G,Cstates) + Sigma)) ) )
                u = np.concatenate((self.u_trim[:3] + delta,[self.u_trim[3]]))


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



class BIREFeedbackLinearizationAircraft(Aircraft):
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


        I = np.eye(3)
        Z = np.zeros((3,3))
        A = np.block([[Z,I,Z],[Z,Z,I],[Z,Z,Z]])
        B = np.block([[Z],[Z],[I]])
        Q = np.diag([0.1]*3 + [1000.]*3 + [10.]*3)
        R = np.diag([0.1]*2 + [1.])
        K,_,K_eigs = co.lqr(A,B,Q,R)
        self.K_FB_2 = K
        # print(K)
        # rep2D(K,"K",decimals=15,np_array=True)
        report_latex(K,"K_{lqr}")
        report_latex(K_eigs,"\lambda_{cl \, lqr}")

        zt = 0.7
        wn = 10.0
        pv = 1.0
        k1 = pv*wn**2. # inte
        k2 = wn**2. + 2.*wn*zt*pv# e
        k3 = 2.*wn*zt + pv # edot
        K1 = np.diag([k1]*3)
        K2 = np.diag([k2]*3) # self.Lin_Model.KI
        K3 = np.diag([k3]*3) # self.Lin_Model.K
        K = np.block([K1,K2,K3])
        K_eigs,_ = np.linalg.eig(A - np.matmul(B,K))
        report_latex(K,"K_{3ord}")
        report_latex(K_eigs,"\lambda_{cl \, 3ord}")
        quit()
    
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
                V_xb    = x_euler[ 0] #  self.x_trim_euler[ 0] # 
                V_yb    = x_euler[ 1] #  self.x_trim_euler[ 1] # 
                V_zb    = x_euler[ 2] #  self.x_trim_euler[ 2] # 
                p       = x_euler[ 3] #  self.x_trim_euler[ 3] # 
                q       = x_euler[ 4] #  self.x_trim_euler[ 4] # 
                r       = x_euler[ 5] #  self.x_trim_euler[ 5] # 
                z_f     = x_euler[ 8] #  self.x_trim_euler[ 8] # 
                da      = x_euler[12] #  self.x_trim_euler[12] # 
                de      = x_euler[13] #  self.x_trim_euler[13] # 
                dB      = x_euler[14] #  self.x_trim_euler[14] # 
                tau     = x_euler[15] #  self.x_trim_euler[15] # 
                pI      = x_euler[self.xIi_eul[0]]
                qI      = x_euler[self.xIi_eul[1]]
                rI      = x_euler[self.xIi_eul[2]]
                # Derived Quantities
                V_tot   = np.sqrt(V_xb**2+V_yb**2+V_zb**2)
                V_xb_ss = self.x_trim[0]
                V_yb_ss = self.x_trim[1]
                V_zb_ss = self.x_trim[2]
                V_ss    = np.sqrt(V_xb_ss**2+V_yb_ss**2+V_zb_ss**2)
                aero = 0
                if aero == 0:
                    a   = np.arctan2(V_zb,V_xb)
                    b   = sin(V_yb/V_tot)
                    V = V_tot
                elif aero == 1:
                    a   = 0.0
                    b   = 0.0
                    V = V_tot
                elif aero == 2:
                    a = np.arctan2(V_zb_ss,V_xb_ss)
                    b = sin(V_yb_ss/V_ss)
                    V     = V_tot
                elif aero == 3:
                    a = np.arctan2(V_zb_ss,V_xb_ss)
                    b = sin(V_yb_ss/V_ss)
                    V = V_xb
                elif aero == 4:
                    a = np.arctan2(V_zb_ss,V_xb_ss)
                    b = sin(V_yb_ss/V_ss)
                    V = V_ss
                _,g,_,_,rho,sos = self.stdatm(-z_f)
                pbar = p*self.bw/2./V
                qbar = q*self.cw/2./V
                rbar = r*self.bw/2./V
                params = a, b, pbar, qbar, rbar, da, de, dB
                # pull out parts of state
                # preliminaries
                BAM = self.aero_model
                Sw = self.Sw
                bw = self.bw
                cw = self.cw
                h_xb,h_yb,h_zb = self.inertia_model.angular_momentum_results()
                hmat = np.array([
                    [0, -h_zb, h_yb], [h_zb, 0, -h_xb], [-h_yb, h_xb, 0]])
                Ixx,Iyy,Izz,Ixy,Ixz,Iyz = \
                    self.inertia_model.inertia_results(dB)
                I     = self.inertia_model.inertia_tensor(dB)
                Iinv  = self.inertia_model.inverse_tensor(dB)
                dIinv = self.inertia_model.inverse_tensor_derivative(dB)
                dIxx,dIyy,dIzz,dIxy,dIxz,dIyz = \
                    self.inertia_model.inertia_derivative_results(dB)
                G = 0.5*rho*V**2.*Sw*np.diag([bw,cw,bw])
                Imult = np.array([
                    (Iyy-Izz)*q*r + Iyz*(q**2-r**2) + Ixz*p*q - Ixy*p*r,
                    (Izz-Ixx)*p*r + Ixz*(r**2-p**2) + Ixy*q*r - Iyz*p*q,
                    (Ixx-Iyy)*p*q + Ixy*(p**2-q**2) + Iyz*p*r - Ixz*q*r])
                Sigma = np.matmul(hmat,[p,q,r]) + Imult
                # matrices
                ###############################################################
                # pull out evaluating condition
                Dxcg, Dycg, Dzcg = self.cgshift
                C = self.aero_model

                # values for later use
                Ca = cos(a); Sa = sin(a)
                Cb = cos(b); Sb = sin(b)
                #
                Rlon = C.c_w/2./V
                Rlat = C.b_w/2./V
                #
                M = V / sos
                #
                # component derivatives for later use
                a_u = - V_zb/(V_xb**2. + V_zb**2.)
                a_w =   V_xb/(V_xb**2. + V_zb**2.)
                b_u = - V_xb*V_yb/V**2./(V_xb**2. + V_zb**2.)**0.5
                b_v = (V_xb**2. + V_zb**2.)**0.5/V**2.
                b_w = - V_yb*V_zb/V**2./(V_xb**2. + V_zb**2.)**0.5
                #
                pbar_u = - pbar*V_xb/V**2.
                pbar_v = - pbar*V_yb/V**2.
                pbar_w = - pbar*V_zb/V**2.
                #
                qbar_u = - qbar*V_xb/V**2.
                qbar_v = - qbar*V_yb/V**2.
                qbar_w = - qbar*V_zb/V**2.
                #
                rbar_u = - rbar*V_xb/V**2.
                rbar_v = - rbar*V_yb/V**2.
                rbar_w = - rbar*V_zb/V**2.
                #
                Q_u = rho*C.S_w*V_xb
                Q_v = rho*C.S_w*V_yb
                Q_w = rho*C.S_w*V_zb
                Qlon_u = Q_u*C.c_w
                Qlon_v = Q_v*C.c_w
                Qlon_w = Q_w*C.c_w
                Qlat_u = Q_u*C.b_w
                Qlat_v = Q_v*C.b_w
                Qlat_w = Q_w*C.b_w
                #
                Qdyn = 0.5*rho*V**2.*C.S_w
                Qlon = Qdyn*C.c_w
                Qlat = Qdyn*C.b_w
                # get forces and moments at the specified condition
                [CL, CS, CD, Cl, Cm, Cn] = \
                    C.aero_results(*params,M=M,**{
                        "compressible" : self.is_compressible,
                        "use_Anderson" : self.use_anderson,
                        "enforce_stall" : self.has_stall
                })

                # state aerodynamic force derivatives
                # evaluate BIRE angle values
                C.evaluate_coeffs(dB)
                # for use
                CL1 = C.CL0 + C.CLa*a
                CS1 = C.CS0 + C.CSb*b
                # lift
                oCL_u = C.CLa*a_u + C.CLb*b_u + C.CLp*pbar_u + C.CLq*qbar_u + \
                    + C.CLr*rbar_u
                oCL_v = C.CLb*b_v + C.CLp*pbar_v + C.CLq*qbar_v + C.CLr*rbar_v
                oCL_w = C.CLa*a_w + C.CLb*b_w + C.CLp*pbar_w + C.CLq*qbar_w + \
                    + C.CLr*rbar_w
                oCL_p = C.CLp*Rlat
                oCL_q = C.CLq*Rlon
                oCL_r = C.CLr*Rlat
                # side
                oCS_u = C.CSa*a_u + C.CSb*b_u + C.CSLp*C.CLa*a_u*pbar + \
                    + (C.CSLp*CL1 + C.CSp)*pbar_u + C.CSq*qbar_u + C.CSr*rbar_u
                oCS_v = C.CSb*b_v + \
                    + (C.CSLp*CL1 + C.CSp)*pbar_v + C.CSq*qbar_v + C.CSr*rbar_v
                oCS_w = C.CSa*a_w + C.CSb*b_w + C.CSLp*C.CLa*a_w*pbar + \
                    + (C.CSLp*CL1 + C.CSp)*pbar_w + C.CSq*qbar_w + C.CSr*rbar_w
                oCS_p = (C.CSLp*CL1 + C.CSp)*Rlat
                oCS_q = C.CSq*Rlon
                oCS_r = C.CSr*Rlat
                # drag
                oCD_u = (C.CDL + 2.*C.CDL2*CL1 + (2.*C.CDL2q*CL1 + C.CDLq)*qbar + \
                    + C.CDLde*de)*C.CLa*a_u + (C.CDS + 2.*C.CDS2*CS1 + \
                    + C.CDSp*pbar + C.CDSr*rbar + C.CDSda*da)*C.CSb*b_u + \
                    + (C.CDSp*CS1 + C.CDp)*pbar_u + (C.CDL2q*CL1**2. + \
                    + C.CDLq*CL1 + C.CDq)*qbar_u + (C.CDSr*CS1 + C.CDr)*rbar_u
                oCD_v = (C.CDS + 2.*C.CDS2*CS1 + \
                    + C.CDSp*pbar + C.CDSr*rbar + C.CDSda*da)*C.CSb*b_v + \
                    + (C.CDSp*CS1 + C.CDp)*pbar_v + (C.CDL2q*CL1**2. + \
                    + C.CDLq*CL1 + C.CDq)*qbar_v + (C.CDSr*CS1 + C.CDr)*rbar_v
                oCD_w = (C.CDL + 2.*C.CDL2*CL1 + (2.*C.CDL2q*CL1 + C.CDLq)*qbar + \
                    + C.CDLde*de)*C.CLa*a_w + (C.CDS + 2.*C.CDS2*CS1 + \
                    + C.CDSp*pbar + C.CDSr*rbar + C.CDSda*da)*C.CSb*b_w + \
                    + (C.CDSp*CS1 + C.CDp)*pbar_w + (C.CDL2q*CL1**2. + \
                    + C.CDLq*CL1 + C.CDq)*qbar_w + (C.CDSr*CS1 + C.CDr)*rbar_w
                oCD_p = (C.CDSp*CS1 + C.CDp)*Rlat
                oCD_q = (C.CDL2q*CL1**2. + C.CDLq*CL1 + C.CDq)*Rlon
                oCD_r = (C.CDSr*CS1 + C.CDr)*Rlat

                # state aerodynamic moment derivatives
                # roll
                oCl_u = C.Cla*a_u + C.Clb*b_u + C.Clp*pbar_u + C.Clq*qbar_u + \
                    + C.ClLr*C.CLa*a_u*rbar + (C.ClLr*CL1 + C.Clr)*rbar_u
                oCl_v = C.Clb*b_v + C.Clp*pbar_v + C.Clq*qbar_v + \
                    + (C.ClLr*CL1 + C.Clr)*rbar_v
                oCl_w = C.Cla*a_w + C.Clb*b_w + C.Clp*pbar_w + C.Clq*qbar_w + \
                    + C.ClLr*C.CLa*a_w*rbar + (C.ClLr*CL1 + C.Clr)*rbar_w
                oCl_p = C.Clp*Rlat # C.CLr
                oCl_q = C.Clq*Rlon
                oCl_r = (C.ClLr*CL1 + C.Clr)*Rlat
                # pitch
                oCm_u = C.Cma*a_u + C.Cmb*b_u + C.Cmp*pbar_u + C.Cmq*qbar_u + \
                    + C.Cmr*rbar_u
                oCm_v = C.Cmb*b_v + C.Cmp*pbar_v + C.Cmq*qbar_v + C.Cmr*rbar_v
                oCm_w = C.Cma*a_w + C.Cmb*b_w + C.Cmp*pbar_w + C.Cmq*qbar_w + \
                    + C.Cmr*rbar_w
                oCm_p = C.Cmp*Rlat
                oCm_q = C.Cmq*Rlon
                oCm_r = C.Cmr*Rlat
                # yaw
                oCn_u = ((C.CnLp*pbar + C.CnLda*da)*C.CLa + C.Cna)*a_u + \
                    + C.Cnb*b_u + (C.CnLp*CL1 + C.Cnp)*pbar_u + C.Cnq*qbar_u + \
                    + C.Cnr*rbar_u
                oCn_v = C.Cnb*b_v + (C.CnLp*CL1 + C.Cnp)*pbar_v + C.Cnq*qbar_v + \
                    + C.Cnr*rbar_v
                oCn_w = ((C.CnLp*pbar + C.CnLda*da)*C.CLa + C.Cna)*a_w + \
                    + C.Cnb*b_w + (C.CnLp*CL1 + C.Cnp)*pbar_w + C.Cnq*qbar_w + \
                    + C.Cnr*rbar_w
                oCn_p = (C.CnLp*CL1 + C.Cnp)*Rlat
                oCn_q = C.Cnq*Rlon
                oCn_r = C.Cnr*Rlat

                # Stall corrections
                aCL_u,aCL_v,aCL_w = oCL_u,oCL_v,oCL_w
                aCL_p,aCL_q,aCL_r = oCL_p,oCL_q,oCL_r
                aCS_u,aCS_v,aCS_w = oCS_u,oCS_v,oCS_w
                aCS_p,aCS_q,aCS_r = oCS_p,oCS_q,oCS_r
                aCD_u,aCD_v,aCD_w = oCD_u,oCD_v,oCD_w
                aCD_p,aCD_q,aCD_r = oCD_p,oCD_q,oCD_r
                aCl_u,aCl_v,aCl_w = oCl_u,oCl_v,oCl_w
                aCl_p,aCl_q,aCl_r = oCl_p,oCl_q,oCl_r
                aCm_u,aCm_v,aCm_w = oCm_u,oCm_v,oCm_w
                aCm_p,aCm_q,aCm_r = oCm_p,oCm_q,oCm_r
                aCn_u,aCn_v,aCn_w = oCn_u,oCn_v,oCn_w
                aCn_p,aCn_q,aCn_r = oCn_p,oCn_q,oCn_r


                # Compressibility corrections
                if self.is_compressible:
                    # Mach derivatives
                    M_u = 2.*V_xb/V/sos
                    M_v = 2.*V_yb/V/sos
                    M_w = 2.*V_zb/V/sos
                    M_p = M_q = M_r = 0.

                    # incompressible coefficients
                    [aCL, aCS, aCD, aCl, aCm, aCn] = \
                        C.aero_results(*params,M=M,**{
                        "compressible" : False,
                        "use_Anderson" : False,
                        "enforce_stall" : self.has_stall
                    })

                    # Mach correction derivatives
                    if M <= 1.0: # subsonic
                        if self.use_anderson:
                            L_w, L_h, L_v = C.Lam_w, C.Lam_h, C.Lam_v
                            R_w, R_h, R_v = C.RA_w, C.RA_h, C.RA_v

                            # derivatives wrt incompressible coefficients
                            CL_aCL = Anderson_correction_der_coeff(aCL,L_w,R_w,M)
                            Cm_aCm = Anderson_correction_der_coeff(aCm,L_w,R_w,M)
                            if self.is_BIRE:
                                CS_aCS = Anderson_correction_der_coeff(aCS,L_h,R_h,M)
                                Cl_aCl = Anderson_correction_der_coeff(aCl,L_w,R_w,M)
                                Cn_aCn = Anderson_correction_der_coeff(aCn,L_h,R_h,M)
                            else:
                                CS_aCS = Anderson_correction_der_coeff(aCS,L_v,R_v,M)
                                Cl_aCl = Anderson_correction_der_coeff(aCl,L_v,R_v,M)
                                Cn_aCn = Anderson_correction_der_coeff(aCn,L_v,R_v,M)

                            # derivatives wrt mach number
                            CL_M = Anderson_correction_der_M(aCL,L_w,R_w,M)
                            Cm_M = Anderson_correction_der_M(aCm,L_w,R_w,M)
                            if self.is_BIRE:
                                CS_M = Anderson_correction_der_M(aCS,L_h,R_h,M)
                                Cl_M = Anderson_correction_der_M(aCl,L_w,R_w,M)
                                Cn_M = Anderson_correction_der_M(aCn,L_h,R_h,M)
                            else:
                                CS_M = Anderson_correction_der_M(aCS,L_v,R_v,M)
                                Cl_M = Anderson_correction_der_M(aCl,L_v,R_v,M)
                                Cn_M = Anderson_correction_der_M(aCn,L_v,R_v,M)
                        else:
                            CL_aCL = CS_aCS = Cl_aCl = Cm_aCm = Cn_aCn = \
                                1. / (1. - M**2.)**0.5
                            K = M / (1. - M**2.)**1.5
                            CL_M = aCL*K
                            CS_M = aCS*K
                            Cl_M = aCl*K
                            Cm_M = aCm*K
                            Cn_M = aCn*K
                    else: # supersonic
                        CL_aCL = CS_aCS = Cl_aCl = Cm_aCm = Cn_aCn = \
                            1. / (M**2. - 1.)**0.5
                        K = - M / (M**2. - 1.)**1.5
                        CL_M = aCL*K
                        CS_M = aCS*K
                        Cl_M = aCl*K
                        Cm_M = aCm*K
                        Cn_M = aCn*K
                    
                    # apply corrections
                    # lift
                    CL_u = CL_aCL*aCL_u + CL_M*M_u
                    CL_v = CL_aCL*aCL_v + CL_M*M_v
                    CL_w = CL_aCL*aCL_w + CL_M*M_w
                    CL_p = CL_aCL*aCL_p + CL_M*M_p
                    CL_q = CL_aCL*aCL_q + CL_M*M_q
                    CL_r = CL_aCL*aCL_r + CL_M*M_r
                    # side
                    CS_u = CS_aCS*aCS_u + CS_M*M_u
                    CS_v = CS_aCS*aCS_v + CS_M*M_v
                    CS_w = CS_aCS*aCS_w + CS_M*M_w
                    CS_p = CS_aCS*aCS_p + CS_M*M_p
                    CS_q = CS_aCS*aCS_q + CS_M*M_q
                    CS_r = CS_aCS*aCS_r + CS_M*M_r
                    # drag
                    CD_u = aCD_u
                    CD_v = aCD_v
                    CD_w = aCD_w
                    CD_p = aCD_p
                    CD_q = aCD_q
                    CD_r = aCD_r
                    # roll
                    Cl_u = Cl_aCl*aCl_u + Cl_M*M_u
                    Cl_v = Cl_aCl*aCl_v + Cl_M*M_v
                    Cl_w = Cl_aCl*aCl_w + Cl_M*M_w
                    Cl_p = Cl_aCl*aCl_p + Cl_M*M_p
                    Cl_q = Cl_aCl*aCl_q + Cl_M*M_q
                    Cl_r = Cl_aCl*aCl_r + Cl_M*M_r
                    # pitch
                    Cm_u = Cm_aCm*aCm_u + Cm_M*M_u
                    Cm_v = Cm_aCm*aCm_v + Cm_M*M_v
                    Cm_w = Cm_aCm*aCm_w + Cm_M*M_w
                    Cm_p = Cm_aCm*aCm_p + Cm_M*M_p
                    Cm_q = Cm_aCm*aCm_q + Cm_M*M_q
                    Cm_r = Cm_aCm*aCm_r + Cm_M*M_r
                    # yaw
                    Cn_u = Cn_aCn*aCn_u + Cn_M*M_u
                    Cn_v = Cn_aCn*aCn_v + Cn_M*M_v
                    Cn_w = Cn_aCn*aCn_w + Cn_M*M_w
                    Cn_p = Cn_aCn*aCn_p + Cn_M*M_p
                    Cn_q = Cn_aCn*aCn_q + Cn_M*M_q
                    Cn_r = Cn_aCn*aCn_r + Cn_M*M_r
                else:
                    # no compressibility
                    CL_u,CL_v,CL_w,CL_p,CL_q,CL_r = aCL_u,aCL_v,aCL_w,aCL_p,aCL_q,aCL_r
                    CS_u,CS_v,CS_w,CS_p,CS_q,CS_r = aCS_u,aCS_v,aCS_w,aCS_p,aCS_q,aCS_r
                    CD_u,CD_v,CD_w,CD_p,CD_q,CD_r = aCD_u,aCD_v,aCD_w,aCD_p,aCD_q,aCD_r
                    Cl_u,Cl_v,Cl_w,Cl_p,Cl_q,Cl_r = aCl_u,aCl_v,aCl_w,aCl_p,aCl_q,aCl_r
                    Cm_u,Cm_v,Cm_w,Cm_p,Cm_q,Cm_r = aCm_u,aCm_v,aCm_w,aCm_p,aCm_q,aCm_r
                    Cn_u,Cn_v,Cn_w,Cn_p,Cn_q,Cn_r = aCn_u,aCn_v,aCn_w,aCn_p,aCn_q,aCn_r

                # thrust state derivatives
                TM = BAM.Prop
                if self.Lin_Model.use_simple_thrust:
                    T_V = tau*(rho/TM.rho_0)**TM.a*(TM.T1 + 2.*TM.T2*V)
                    T_z = 0.
                else:
                    if tau <= 0.77:
                        P1 = 64.94*tau
                    else:
                        P1 = 217.38*tau - 117.38
                    # pull out each setting derivative
                    ia,_,iT1,iT2 = TM.idle_coefs(-z_f)
                    Tidle_V = (rho/TM.rho_0)**ia*(iT1 + 2.*iT2*V)
                    la,_,lT1,lT2 = TM.mil_coefs(-z_f)
                    Tmil_V = (rho/TM.rho_0)**la*(lT1 + 2.*lT2*V)
                    ma,_,mT1,mT2 = TM.max_coefs(-z_f)
                    Tmax_V = (rho/TM.rho_0)**ma*(mT1 + 2.*mT2*V)
                    # get full derivative
                    if P1 < 50.:
                        T_V = Tidle_V + (Tmil_V - Tidle_V)*P1/50.
                    else:
                        T_V = Tmil_V + (Tmax_V - Tmil_V)*(P1-50.)/50.
                # body-fixed force derivatives wrt state
                CFx0 = CL*Sa - CS*Ca*Sb - CD*Ca*Cb
                Fx_u = Q_u*CFx0 + Qdyn*(CL_u*Sa + CL*Ca*a_u - CS_u*Ca*Sb + \
                    + CS*Sa*Sb*a_u - CS*Ca*Cb*b_u - CD_u*Ca*Cb + CD*Sa*Cb*a_u + \
                    + CD*Ca*Sb*b_u) + T_V*V_xb/V
                Fx_v = Q_v*CFx0 + Qdyn*(CL_v*Sa - CS_v*Ca*Sb + \
                    - CS*Ca*Cb*b_v - CD_v*Ca*Cb + \
                    + CD*Ca*Sb*b_v) + T_V*V_yb/V
                Fx_w = Q_w*CFx0 + Qdyn*(CL_w*Sa + CL*Ca*a_w - CS_w*Ca*Sb + \
                    + CS*Sa*Sb*a_w - CS*Ca*Cb*b_w - CD_w*Ca*Cb + CD*Sa*Cb*a_w + \
                    + CD*Ca*Sb*b_w) + T_V*V_zb/V
                #
                Fx_p = Qdyn*(CL_p*Sa - CS_p*Ca*Sb - CD_p*Ca*Cb)
                Fx_q = Qdyn*(CL_q*Sa - CS_q*Ca*Sb - CD_q*Ca*Cb)
                Fx_r = Qdyn*(CL_r*Sa - CS_r*Ca*Sb - CD_r*Ca*Cb)
                #
                #
                CFy0 = CS*Cb - CD*Sb
                Fy_u = Q_u*CFy0 + Qdyn*(CS_u*Cb - CS*Sb*b_u - CD_u*Sb - CD*Cb*b_u)
                Fy_v = Q_v*CFy0 + Qdyn*(CS_v*Cb - CS*Sb*b_v - CD_v*Sb - CD*Cb*b_v)
                Fy_w = Q_w*CFy0 + Qdyn*(CS_w*Cb - CS*Sb*b_w - CD_w*Sb - CD*Cb*b_w)
                #
                Fy_p = Qdyn*(CS_p*Cb - CD_p*Sb)
                Fy_q = Qdyn*(CS_q*Cb - CD_q*Sb)
                Fy_r = Qdyn*(CS_r*Cb - CD_r*Sb)
                #
                #
                CFz0 = - CL*Ca - CS*Sa*Sb - CD*Sa*Cb
                Fz_u = Q_u*CFz0 + Qdyn*(-CL_u*Ca + CL*Sa*a_u - CS_u*Sa*Sb + \
                    - CS*Ca*Sb*a_u - CS*Sa*Cb*b_u - CD_u*Sa*Cb - CD*Ca*Cb*a_u + \
                    + CD*Sa*Sb*b_u)
                Fz_v = Q_v*CFz0 + Qdyn*(-CL_v*Ca - CS_v*Sa*Sb + \
                    - CS*Sa*Cb*b_v - CD_v*Sa*Cb + \
                    + CD*Sa*Sb*b_v)
                Fz_w = Q_w*CFz0 + Qdyn*(-CL_w*Ca + CL*Sa*a_w - CS_w*Sa*Sb + \
                    - CS*Ca*Sb*a_w - CS*Sa*Cb*b_w - CD_w*Sa*Cb - CD*Ca*Cb*a_w + \
                    + CD*Sa*Sb*b_w)
                #
                Fz_p = Qdyn*(- CL_p*Ca - CS_p*Sa*Sb - CD_p*Sa*Cb)
                Fz_q = Qdyn*(- CL_q*Ca - CS_q*Sa*Sb - CD_q*Sa*Cb)
                Fz_r = Qdyn*(- CL_r*Ca - CS_r*Sa*Sb - CD_r*Sa*Cb)

                # body-fixed moment derivatives wrt state
                Mx_u = Qlat_u*Cl + Qlat*Cl_u + Fy_u*Dzcg - Fz_u*Dycg
                Mx_v = Qlat_v*Cl + Qlat*Cl_v + Fy_v*Dzcg - Fz_v*Dycg
                Mx_w = Qlat_w*Cl + Qlat*Cl_w + Fy_w*Dzcg - Fz_w*Dycg
                #
                Mx_p = Qlat*Cl_p + Fy_p*Dzcg - Fz_p*Dycg
                Mx_q = Qlat*Cl_q + Fy_q*Dzcg - Fz_q*Dycg
                Mx_r = Qlat*Cl_r + Fy_r*Dzcg - Fz_r*Dycg
                #
                #
                My_u = Qlon_u*Cm + Qlon*Cm_u + Fz_u*Dxcg - Fx_u*Dzcg
                My_v = Qlon_v*Cm + Qlon*Cm_v + Fz_v*Dxcg - Fx_v*Dzcg
                My_w = Qlon_w*Cm + Qlon*Cm_w + Fz_w*Dxcg - Fx_w*Dzcg
                #
                My_p = Qlon*Cm_p + Fz_p*Dxcg - Fx_p*Dzcg
                My_q = Qlon*Cm_q + Fz_q*Dxcg - Fx_q*Dzcg
                My_r = Qlon*Cm_r + Fz_r*Dxcg - Fx_r*Dzcg
                #
                #
                Mz_u = Qlat_u*Cn + Qlat*Cn_u + Fx_u*Dycg - Fy_u*Dxcg
                Mz_v = Qlat_v*Cn + Qlat*Cn_v + Fx_v*Dycg - Fy_v*Dxcg
                Mz_w = Qlat_w*Cn + Qlat*Cn_w + Fx_w*Dycg - Fy_w*Dxcg
                #
                Mz_p = Qlat*Cn_p + Fx_p*Dycg - Fy_p*Dxcg
                Mz_q = Qlat*Cn_q + Fx_q*Dycg - Fy_q*Dxcg
                Mz_r = Qlat*Cn_r + Fx_r*Dycg - Fy_r*Dxcg
                ###############################################################
                dfdw = np.matmul(Iinv,(np.array([
                    [Mx_p, Mx_q, Mx_r],
                    [My_p, My_q, My_r],
                    [Mz_p, Mz_q, Mz_r]
                ]) + hmat + np.array([
                    [Ixz*q - Ixy*r, (Iyy - Izz)*r + 2.*Iyz*q + Ixz*p, 
                                            (Iyy - Izz)*q - 2.*Iyz*r - Ixy*p],
                    [(Izz - Ixx)*r - 2.*Ixz*p - Iyz*q, Ixy*r - Iyz*p, 
                                            (Izz - Ixx)*p + 2.*Ixz*r + Ixy*q],
                    [(Ixx - Iyy)*q + 2.*Ixy*p + Iyz*r, 
                                (Ixx - Iyy)*p - 2.*Ixy*q -Ixz*r, Iyz*p - Ixz*q]
                ])))
                #
                dfdy = np.matmul(Iinv,np.array([
                    [Mx_u, Mx_v, Mx_w],
                    [My_u, My_v, My_w],
                    [Mz_u, Mz_v, Mz_w]
                ]))
                #
                ###############################################################
                # input aerodynamic force derivatives
                # evaluate derivatives wrt bire angle
                C.evaluate_derivatives(dB)
                # for use
                dCL1 = C.dCL0 + C.dCLa * a
                dCS1 = C.dCS0 + C.dCSb * b
                # lift
                oCL_dB = C.dCL0 + C.dCLa*a + C.dCLb*b + C.dCLp*pbar + C.dCLq*qbar +\
                    + C.dCLr*rbar + C.dCLda*da + C.dCLde*de
                # side
                oCS_dB = C.dCS0 + C.dCSa*a + C.dCSb*b + (C.dCSLp*CL1 + \
                    + C.CSLp*dCL1 + C.dCSp)*pbar + C.dCSq*qbar + C.dCSr*rbar + \
                    + C.dCSda*da + C.dCSde*de
                # drag
                oCD_da = C.CDSda*CS1 + C.CDda
                oCD_de = C.CDLde*CL1 + C.CDde + 2.*C.CDde2*de
                oCD_dB = C.dCD0 + C.dCDL*CL1 + C.CDL*dCL1 + C.dCDL2*CL1**2. + \
                    + 2.*C.CDL2*CL1*dCL1 + C.dCDS*CS1 + C.CDS*dCS1 + \
                    + C.dCDS2*CS1**2. + 2.*C.CDS2*CS1*dCS1 + (C.dCDSp*CS1 + \
                    + C.CDSp*dCS1 + C.dCDp)*pbar + (C.dCDL2q*CL1**2. + \
                    + 2.*C.CDL2q*CL1*dCL1 + C.dCDLq*CL1 + C.CDLq*dCL1 + \
                    + C.dCDq)*qbar + (C.dCDSr*CS1 + C.CDSr*dCS1 + C.dCDr)*rbar + \
                    + (C.dCDSda*CS1 + C.CDSda*dCS1 + C.dCDda)*da + \
                    + (C.dCDLde*CL1 + C.CDLde*dCL1 + C.dCDde)*de + C.dCDde2*de**2.
                # equated values
                oCL_da, oCL_de, oCS_da, oCS_de = C.CLda, C.CLde, C.CSda, C.CSde
                
                # input aerodynamic moment derivatives
                # roll
                oCl_dB = C.dCl0 + C.dCla*a + C.dClb*b + C.dClp*pbar + C.dClq*qbar +\
                    + (C.dClLr*CL1 + C.ClLr*dCL1 + C.dClr)*rbar + C.dClda*da + \
                    + C.dClde*de
                # pitch
                oCm_dB = C.dCm0 + C.dCma*a + C.dCmb*b + C.dCmp*pbar + C.dCmq*qbar +\
                    + C.dCmr*rbar + C.dCmda*da + C.dCmde*de
                # yaw
                oCn_da = C.CnLda*CL1 + C.Cnda
                oCn_dB = C.dCn0 + C.dCna*a + C.dCnb*b + (C.dCnLp*CL1 + \
                    + C.CnLp*dCL1 + C.dCnp)*pbar + C.dCnq*qbar + C.dCnr*rbar + \
                    + (C.dCnLda*CL1 + C.CnLda*dCL1 + C.dCnda)*da + C.dCnde*de
                # equated values
                oCl_da, oCl_de, oCm_da, oCm_de = C.Clda, C.Clde, C.Cmda, C.Cmde
                oCn_de = C.Cnde

                # Stall corrections
                aCL_da,aCL_de = oCL_da,oCL_de
                aCS_da,aCS_de = oCS_da,oCS_de
                aCD_da,aCD_de = oCD_da,oCD_de
                aCl_da,aCl_de = oCl_da,oCl_de
                aCm_da,aCm_de = oCm_da,oCm_de
                aCn_da,aCn_de = oCn_da,oCn_de
                # bire
                aCL_dB,aCS_dB,aCD_dB = oCL_dB,oCS_dB,oCD_dB
                aCl_dB,aCm_dB,aCn_dB = oCl_dB,oCm_dB,oCn_dB
                
                # Compressibility corrections
                if self.is_compressible:
                    # incompressible coefficients
                    [aCL, aCS, aCD, aCl, aCm, aCn] = \
                        C.aero_results(*params,M=M,**{
                        "compressible" : False,
                        "use_Anderson" : False,
                        "enforce_stall" : self.has_stall
                    })

                    # Mach correction derivatives
                    if M <= 1.0: # subsonic
                        if self.use_anderson:
                            L_w, L_h, L_v = C.Lam_w, C.Lam_h, C.Lam_v
                            R_w, R_h, R_v = C.RA_w, C.RA_h, C.RA_v

                            # derivatives wrt incompressible coefficients
                            CL_aCL = Anderson_correction_der_coeff(aCL,L_w,R_w,M)
                            Cm_aCm = Anderson_correction_der_coeff(aCm,L_w,R_w,M)
                            if self.is_BIRE:
                                CS_aCS = Anderson_correction_der_coeff(aCS,L_h,R_h,M)
                                Cl_aCl = Anderson_correction_der_coeff(aCl,L_w,R_w,M)
                                Cn_aCn = Anderson_correction_der_coeff(aCn,L_h,R_h,M)
                            else:
                                CS_aCS = Anderson_correction_der_coeff(aCS,L_v,R_v,M)
                                Cl_aCl = Anderson_correction_der_coeff(aCl,L_v,R_v,M)
                                Cn_aCn = Anderson_correction_der_coeff(aCn,L_v,R_v,M)
                        else:
                            CL_aCL = CS_aCS = Cl_aCl = Cm_aCm = Cn_aCn = \
                                1. / (1. - M**2.)**0.5
                    else: # supersonic
                        CL_aCL = CS_aCS = Cl_aCl = Cm_aCm = Cn_aCn = \
                            1. / (M**2. - 1.)**0.5
                    
                    # apply corrections
                    # lift
                    CL_da = CL_aCL*aCL_da
                    CL_de = CL_aCL*aCL_de
                    # side
                    CS_da = CS_aCS*aCS_da
                    CS_de = CS_aCS*aCS_de
                    # drag
                    CD_da = aCD_da
                    CD_de = aCD_de
                    # roll
                    Cl_da = Cl_aCl*aCl_da
                    Cl_de = Cl_aCl*aCl_de
                    # pitch
                    Cm_da = Cm_aCm*aCm_da
                    Cm_de = Cm_aCm*aCm_de
                    # yaw
                    Cn_da = Cn_aCn*aCn_da
                    Cn_de = Cn_aCn*aCn_de
                    # BIRE
                    CL_dB = CL_aCL*aCL_dB
                    CS_dB = CS_aCS*aCS_dB
                    CD_dB = aCD_dB
                    Cl_dB = Cl_aCl*aCl_dB
                    Cm_dB = Cm_aCm*aCm_dB
                    Cn_dB = Cn_aCn*aCn_dB
                else:
                    # no compressibility
                    CL_da,CL_de = aCL_da,aCL_de
                    CS_da,CS_de = aCS_da,aCS_de
                    CD_da,CD_de = aCD_da,aCD_de
                    Cl_da,Cl_de = aCl_da,aCl_de
                    Cm_da,Cm_de = aCm_da,aCm_de
                    Cn_da,Cn_de = aCn_da,aCn_de
                    # bire
                    CL_dB,CS_dB,CD_dB = aCL_dB,aCS_dB,aCD_dB
                    Cl_dB,Cm_dB,Cn_dB = aCl_dB,aCm_dB,aCn_dB
                    
                # thrust state derivatives
                TM = C.Prop

                # body-fixed force derivatives wrt input
                Fx_da = Qdyn*(CL_da*Sa - CS_da*Ca*Sb - CD_da*Ca*Cb)
                Fx_de = Qdyn*(CL_de*Sa - CS_de*Ca*Sb - CD_de*Ca*Cb)
                Fx_dB = Qdyn*(CL_dB*Sa - CS_dB*Ca*Sb - CD_dB*Ca*Cb)
                #
                Fy_da = Qdyn*(CS_da*Cb - CD_da*Sb)
                Fy_de = Qdyn*(CS_de*Cb - CD_de*Sb)
                Fy_dB = Qdyn*(CS_dB*Cb - CD_dB*Sb)
                #
                Fz_da = Qdyn*(- CL_da*Ca - CS_da*Sa*Sb - CD_da*Sa*Cb)
                Fz_de = Qdyn*(- CL_de*Ca - CS_de*Sa*Sb - CD_de*Sa*Cb)
                Fz_dB = Qdyn*(- CL_dB*Ca - CS_dB*Sa*Sb - CD_dB*Sa*Cb)
                
                # body-fixed moment derivatives wrt input
                Mx_da = Qlat*Cl_da + Fy_da*Dzcg - Fz_da*Dycg
                Mx_de = Qlat*Cl_de + Fy_de*Dzcg - Fz_de*Dycg
                Mx_dB = Qlat*Cl_dB + Fy_dB*Dzcg - Fz_dB*Dycg
                #
                My_da = Qlon*Cm_da + Fz_da*Dxcg - Fx_da*Dzcg
                My_de = Qlon*Cm_de + Fz_de*Dxcg - Fx_de*Dzcg
                My_dB = Qlon*Cm_dB + Fz_dB*Dxcg - Fx_dB*Dzcg
                #
                Mz_da = Qlat*Cn_da + Fx_da*Dycg - Fy_da*Dxcg
                Mz_de = Qlat*Cn_de + Fx_de*Dycg - Fy_de*Dxcg
                Mz_dB = Qlat*Cn_dB + Fx_dB*Dycg - Fy_dB*Dxcg

                # evaluate at condtion for Mx, My, Mz
                T = TM.get_thrust(tau,-z_f,V)
                #
                Fx = Qdyn*(CL*Sa - CS*Ca*Sb - CD*Ca*Cb) + T
                Fy = Qdyn*(CS*Cb - CD*Sb)
                Fz = Qdyn*(- CL*Ca - CS*Sa*Sb - CD*Sa*Cb)
                #
                Mx = Qlat*Cl + Fy*Dzcg - Fz*Dycg
                My = Qlon*Cm + Fz*Dxcg - Fx*Dzcg
                Mz = Qlat*Cn + Fx*Dycg - Fy*Dxcg
                
                # assemble components
                wdot = (
                    np.array([Mx,My,Mz]) +
                    np.matmul(hmat, np.array([p,q,r])) + 
                    np.array([
                        ( Iyy- Izz)*q*r +  Iyz*(q**2.-r**2.)+ Ixz*p*q- Ixy*p*r,
                        ( Izz- Ixx)*p*r +  Ixz*(r**2.-p**2.)+ Ixy*q*r- Iyz*p*q,
                        ( Ixx- Iyy)*p*q +  Ixy*(p**2.-q**2.)+ Iyz*p*r- Ixz*q*r
                    ])
                )
                wdot_dB = (
                    np.array([Mx_dB,My_dB,Mz_dB]) +
                    np.array([
                        (dIyy-dIzz)*q*r + dIyz*(q**2.-r**2.)+dIxz*p*q-dIxy*p*r,
                        (dIzz-dIxx)*p*r + dIxz*(r**2.-p**2.)+dIxy*q*r-dIyz*p*q,
                        (dIxx-dIyy)*p*q + dIxy*(p**2.-q**2.)+dIyz*p*r-dIxz*q*r
                    ])
                )
                r3lin = (np.matmul(Iinv,wdot_dB) + np.matmul(dIinv,wdot) )
                ###############################################################
                dfdd = np.concatenate((np.matmul(Iinv,np.array([
                    [Mx_da, Mx_de],
                    [My_da, My_de],
                    [Mz_da, Mz_de]
                ])), r3lin[:,np.newaxis]),axis=1)
                #
                S = np.diag([self.s_da,self.s_de,self.s_dr])
                N = dfddS = np.matmul(dfdd,S)
                Nadj = np.array([
                    [ (N[1,1]*N[2,2]-N[1,2]*N[2,1]),
                        -(N[0,1]*N[2,2]-N[0,2]*N[2,1]),
                         (N[0,1]*N[1,2]-N[0,2]*N[1,1])],
                    [-(N[1,0]*N[2,2]-N[1,2]*N[2,0]),
                         (N[0,0]*N[2,2]-N[0,2]*N[2,0]),
                        -(N[0,0]*N[1,2]-N[0,2]*N[1,0])],
                    [ (N[1,0]*N[2,1]-N[1,1]*N[2,0]),
                        -(N[0,0]*N[2,1]-N[0,1]*N[2,0]),
                         (N[0,0]*N[1,1]-N[0,1]*N[1,0])]
                ])
                Ndet = N[0,0]*(N[1,1]*N[2,2] - N[1,2]*N[2,1]) \
                    -  N[0,1]*(N[1,0]*N[2,2] - N[1,2]*N[2,0]) \
                    +  N[0,2]*(N[1,0]*N[2,1] - N[1,1]*N[2,0])
                dfddSinv = Nadj/Ndet
                #

                T = self._get_thrust_model(tau,tau,-z_f,V,M,False)
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
                #
                ph,th,ps = x_euler[9],x_euler[10],x_euler[11]
                cp = cos(ph); sp = sin(ph)
                ct = cos(th); st = sin(th)
                cs = cos(ps); ss = sin(ps)
                # u,v,w
                ## INTSTATE
                W = self.inertia_model.W
                dy = np.array([
                    g/W*Fx - g*st    + r*V_yb - q*V_zb,
                    g/W*Fy + g*sp*ct + p*V_zb - r*V_xb,
                    g/W*Fz + g*cp*ct + q*V_xb - p*V_yb
                ])
                # vectors
                z2 = x_euler[self.Lin_Model.Cslice] - ref
                M = np.matmul(G,[
                    # BAM._Cl(*params),BAM._Cm(*params),BAM._Cn(*params)
                    Cl, Cm, Cn
                    ])
                fx1x2 = np.matmul(Iinv,M + Sigma)
                z3 = fx1x2
                z1 = np.array([pI,qI,rI])
                x2 = x_euler[12:15] #- self.u_trim[:3]
                # controller
                # K1 = self.Lin_Model.KI
                # K2 = self.Lin_Model.K
                # K1 = np.diag([100.0]*3) # self.Lin_Model.KI
                # K2 = np.diag([14.0]*3) # self.Lin_Model.K
                zt = 0.7
                wn = 10.0
                pv = 1.0
                k1 = pv*wn**2. # inte
                k2 = wn**2. + 2.*wn*zt*pv# e
                k3 = 2.*wn*zt + pv # edot
                K1 = np.diag([k1]*3)
                K2 = np.diag([k2]*3) # self.Lin_Model.KI
                K3 = np.diag([k3]*3) # self.Lin_Model.K
                # rest = - np.matmul(dfdy,dy) \
                #     - np.matmul(dfdw,z3) + np.matmul(dfddS,x2) \
                #     - np.matmul(K1,z2) - np.matmul(K2,z3) - np.matmul(K3,z1)
                # v = np.matmul(dfddSinv,rest)

                K = np.block([K1,K2,K3])
                # K = self.K_FB_2
                z = np.concatenate((z1,z2,z3))
                v_cl = - np.matmul(K,z)
                # - np.matmul(K1,z2) - np.matmul(K2,z3) - np.matmul(K3,z1)

                rest = - np.matmul(dfdw,z3) + np.matmul(dfddS,x2) \
                    + v_cl # - np.matmul(dfdy,dy)
                v = np.matmul(dfddSinv,rest)
                # print(np.rad2deg(v))
                # v = 0.0
                # self.u_trim[:3] + 
                u = np.concatenate((v,[self.u_trim[3]]))


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



class BIRELyapunovAircraft(Aircraft):
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
                # pull out states
                p = x_euler[3]
                q = x_euler[4]
                r = x_euler[5]
                ade = x_euler[13] - self.x_trim_euler[13]
                adB = x_euler[14] - self.x_trim_euler[14]
                # calculate control
                # t = -10.0
                # s = - 2.0310956805628173*t - 0.4182774140005106 - 10.0
                t = 4.0
                s = 10.0
                u = self.u_trim*1.
                u[0] += s*p
                # u[1] += t*x[4]*abs(self.max_dr - adB)/self.max_dr
                # u[1] += np.sign(adB)*( r )*abs(q)*2.0
                defun = (              t )*q + (adB)*(r) # np.sign(adB)*
                dBfun = ( 2.*(ade)*t +              (q) )*r # np.sign(ade)* 
                u[1] += defun
                u[2] += np.arcsin(max(-1.,min(1.,dBfun)))
                print("de = {:> 6.2f} deg, dB = {:> 6.2f} deg".format(np.rad2deg(u[1]),np.rad2deg(u[2])))
                
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


def _float2str(value):
    # _num_format = ':.4g'
    # return f"{value:{_num_format}}"
    return "{:.4f}".format(value)

def _tf_factorized_polynomial_to_string(roots, gain=1, var='s'):
    """Convert a factorized polynomial to a string"""

    if roots.size == 0:
        return _float2str(gain)

    factors = []
    for root in sorted(roots, reverse=True):
        if np.isreal(root):
            if root == 0:
                factor = f"{var}"
                factors.append(factor)
            elif root > 0:
                factor = f"{var} - {_float2str(np.abs(root))}"
                factors.append(factor)
            else:
                factor = f"{var} + {_float2str(np.abs(root))}"
                factors.append(factor)
        elif np.isreal(root * 1j):
            if root.imag > 0:
                factor = f"{var} - {_float2str(np.abs(root))}j"
                factors.append(factor)
            else:
                factor = f"{var} + {_float2str(np.abs(root))}j"
                factors.append(factor)
        else:
            if root.real > 0:
                factor = f"{var} - ({_float2str(root)})"
                factors.append(factor)
            else:
                factor = f"{var} + ({_float2str(-root)})"
                factors.append(factor)

    multiplier = ''
    if round(gain, 4) != 1.0:
        multiplier = _float2str(gain) + " "

    if len(factors) > 1 or multiplier:
        factors = [f"({factor})" for factor in factors]

    return multiplier + " ".join(factors)

def print_tf(tf,num_name="",den_name="",var="s"):

    z,p,k = scipy_tf2zpk(tf.num[0][0],tf.den[0][0])

    numstr = _tf_factorized_polynomial_to_string(z, gain=k, var=var)
    denstr = _tf_factorized_polynomial_to_string(p, var=var)

    # Figure out the length of the separating line
    dashcount = max(len(numstr), len(denstr))
    dashes = '-' * dashcount

    # Center the numerator or denominator
    if len(numstr) < dashcount:
        numstr = ' ' * ((dashcount - len(numstr)) // 2) + numstr
    if len(denstr) < dashcount:
        denstr = ' ' * ((dashcount - len(denstr)) // 2) + denstr
    
    # add label
    namecount = max(len(num_name),len(den_name))
    numstr = "{:^{}s}   ".format(num_name,namecount) + numstr
    dashes = "-"*namecount + " = " + dashes
    denstr = "{:^{}s}   ".format(den_name,namecount) + denstr

    outstr = "\n" + numstr + "\n" + dashes + "\n" + denstr + "\n"

    print(outstr)

    return outstr

def build_base_controller(base_dict,save_folder):

    # build linearized model
    mrrr = [6,7,8,11]
    mrrc = [3]
    # for i,dicti in enumerate([base_dict]):#,bire_rc_dict]):
    #     if i == 0:
    #         print("Baseline")
    #     else:
    #         print("BIRE")
    #     base = TrackingAircraft(dicti)
    #     base._report_trim_solution(base.x_trim,base.u_trim)
    #     _,Lin_Model = base._build_controller(
    #         base.x_trim_euler,base.u_trim,
    #         report=False,save_matrices=False,
    #         mrrr=mrrr,mrrc=mrrc,drop_actrs=True,
    #         include_stall_derivatives=False,skip_reporting=True,run_freq=False)
    #     print("A_min", Lin_Model.A_min)
    #     print("B_min", Lin_Model.B_min)
    #     print("B_min^-1", np.linalg.solve(Lin_Model.B_min,np.eye(3)))
    #     print("A_eigs", Lin_Model.A_eigs)
    #     print("A_BK_eigs", Lin_Model.A_BK_eigs)
    #     print("K", Lin_Model.K)
    #     print()
    print("Baseline")
    base_dict["controller"]["type"] = "none"
    base = TrackingAircraft(base_dict)
    base._report_trim_solution(base.x_trim,base.u_trim)
    # base.use_quaternions = False
    _,Lin_Model = base._build_controller(
        base.x_trim_euler,base.u_trim,
        report=False,save_matrices=False,
        mrrr=mrrr,mrrc=mrrc,drop_actrs=True,
        # use_VAB_format=True,
        # use_numerical_linearization=True,
        # numerical_dynamics=base._nonlinear_euler_dynamics,
        include_stall_derivatives=False,skip_reporting=True,run_freq=False)
    # base.use_quaternions = True
    aa = Lin_Model.A_min*1.
    print(np.linalg.eig(aa)[0])
    ba = Lin_Model.B_min*1.
    ca = np.block([np.zeros((3,3)),np.eye(3),np.zeros((3,2))])
    # add in integrator states # # p q r pI qI rI
    Z3 = np.zeros((aa.shape[0],3))
    I3 = np.eye(3)
    Zbf = np.zeros((3,3))
    Zaf = np.zeros((3,2))
    A = np.block([[aa,Z3],[Zbf,I3,Zaf,Zbf]])
    B = np.block([[ba],[Zbf]])
    print(B.shape)
    C = np.block([[ca,Zbf],[Z3.T,np.eye(3)]]) # np.block([ca,Z3]) # 
    D = 0.0
    # report
    print(np.linalg.eig(A)[0])
    report_latex(A,"A",decimals=5,predecimals=5,print_report=True)
    report_latex(B,"B",decimals=5,predecimals=5,print_report=True)
    report_latex(C,"C",decimals=5,predecimals=5,print_report=True)

    # longitudinal eigenvalues
    row_lon = [0,2,4,7]
    aa_lon = (A[row_lon,:])[:,row_lon]
    eval_lon,evec_lon = np.linalg.eig(aa_lon)
    en = eval_lon[:,np.newaxis]
    report_latex(en,"\lambda_{lon}",decimals=5,predecimals=5,print_report=True)
    # lateral eigenvalues
    row_lat = [1,3,5,6]
    aa_lat = (A[row_lat,:])[:,row_lat]
    eval_lat,evec_lat = np.linalg.eig(aa_lat)
    et = eval_lat[:,np.newaxis]
    report_latex(et,"\lambda_{lat}",decimals=5,predecimals=5,print_report=True)
    
    # run through by loop
    # p
    t_c_max = 1.0
    s_max = 1./t_c_max
    #
    row = [3,6,8]
    kpP = -0.045 # -0.045
    acl = A - np.matmul(B,np.block([
        [np.array([0.0]*3 + [kpP] + [0.0]*7)],
        [np.zeros((2,A.shape[0]))]
    ]))
    sys = co.ss((acl[row,:])[:,row],B[row,0],C[3,row],D)
    tf = co.ss2tf(sys)
    # print(co.ssdata(sys))
    print_tf(tf,"p","da")
    k = -np.logspace(-1,0.2,1000)
    # print(k)
    kpI = -1.15
    kai = np.argmin(np.abs(k-kpI)) # 
    # print(k[kai])
    r,_ = co.rlocus(sys,kvect=k)
    # print(r.shape)
    plt.close()
    fig,axs = plt.subplots(figsize=(3.25,3.5),dpi=300.0,constrained_layout=True)
    # print(r.shape,k.shape)
    for i in range(r.shape[1]):
        ri = r[:,i]
        axs.plot(np.real(ri),np.imag(ri))#,c="k")
        axs.plot(np.real(ri[0]),np.imag(ri[0]),c="k",marker="x")
        axs.plot(np.real(ri[kai]),np.imag(ri[kai]),c="0.5",marker=".")
    # # plot wn,z
    # wn = 7.0 # rad/s
    # zt = 0.6
    # th = np.linspace(0.0,2.*np.pi,100)
    # wnx = 

    # axs.set_xlim((-1,1))
    # axs.set_ylim((-1,1))
    axs.set_xlabel("Real [s]")
    axs.set_ylabel("Imaginary [s]")
    axs.set_title("Root Locus")
    axs.grid(which="major",lw=0.6,ls="-",c="0.75")
    fig.savefig(save_folder + "/p_rlocus.png",dpi=300.0)
    show_p_rlocus = False
    if show_p_rlocus:
        plt.show()
    else:
        plt.close("all")
    #
    #
    # new closed loop system
    acl = acl - np.matmul(B,np.block([
        [np.array([0.0]*8 + [kpI] + [0.0]*2)],
        [np.zeros((2,A.shape[0]))]
    ]))
    # eigenvalues
    row_lat = [1,3,5,6,8,10]
    aa_lat = (acl[row_lat,:])[:,row_lat]
    eval_lat,evec_lat = np.linalg.eig(aa_lat)
    et = eval_lat[:,np.newaxis]
    report_latex(et,"\lambda_{lat}",decimals=5,predecimals=5,print_report=True)
    #
    #
    #
    # q
    W = base.inertia_model.W
    V = base.V0
    rho = base.rho0
    Sw = base.Sw
    CW = W/0.5/rho/V**2./Sw
    CAP = np.array([0.28, 3.6])
    wn_sp_lim = (CAP*base.aero_model.CLa/CW)**0.5
    zn_sp_lim = np.array([0.35, 1.3])
    # phugoid
    zt_ph_min = 0.04
    #
    row = [0,2,4,7,9]
    kqP = -0.01
    acl = A - np.matmul(B,np.block([
        [np.zeros((1,A.shape[0]))],
        [np.array([0.0]*4 + [kqP] + [0.0]*6)],
        [np.zeros((1,A.shape[0]))]
    ]))
    sys = co.ss((acl[row,:])[:,row],B[row,1],C[4,row],D)
    tf = co.ss2tf(sys)
    # print(tf)
    # print(co.ssdata(sys))
    print_tf(tf,"q","de")
    k = -np.logspace(-2,0,1000)
    # print(k)
    kqI = -0.1 # 0.01
    kai = np.argmin(np.abs(k-kqI)) # 
    # print(k[kai])
    r,_ = co.rlocus(sys,kvect=k)
    print("evals at goal =",r[kai])
    # report wn, zt
    rat = r[kai]
    sg = -np.real(rat)
    wd = np.abs(np.imag(rat))
    e0s = np.array([complex(-sg[j],wd[j]) for j in range(len(sg))])
    e1s = np.conjugate(e0s)
    wn = np.sqrt(e0s*e1s)
    zt = - (e0s + e1s)/2./np.sqrt(e0s*e1s)
    print("   wn at goal =",wn)
    print(" zeta at goal =",zt)
    print("sp    wn lims =",wn_sp_lim[0]," ",wn_sp_lim[1])
    print("sp  zeta lims =",zn_sp_lim[0]," ",zn_sp_lim[1])
    print("ph  zeta min  =",zt_ph_min)
    # print(r.shape)
    plt.close()
    fig,axs = plt.subplots(figsize=(3.25,3.5),dpi=300.0,constrained_layout=True)
    # print(r.shape,k.shape)
    for i in range(r.shape[1]):
        ri = r[:,i]
        axs.plot(np.real(ri),np.imag(ri))#,c="k")
        axs.plot(np.real(ri[0]),np.imag(ri[0]),c="k",marker="x")
        axs.plot(np.real(ri[kai]),np.imag(ri[kai]),c="0.5",marker=".")
    # plot limits
    # sg = np.outer(-wn,zn)
    # # print(sg)
    # wd = np.abs(np.outer(wn,(1. - np.array(zn,dtype=complex))**0.5))
    # # print(wd)
    #
    axs.set_xlabel("Real [s]")
    axs.set_ylabel("Imaginary [s]")
    axs.set_title("Root Locus")
    axs.grid(which="major",lw=0.6,ls="-",c="0.75")
    fig.savefig(save_folder + "/q_rlocus.png",dpi=300.0)
    show_q_rlocus = False
    if show_q_rlocus:
        plt.show()
    else:
        plt.close("all")
    #
    #
    # new closed loop system
    acl = acl - np.matmul(B,np.block([
        [np.zeros((1,A.shape[0]))],
        [np.array([0.0]*9 + [kqI] + [0.0]*1)],
        [np.zeros((1,A.shape[0]))]
    ]))
    # eigenvalues
    row_lon = [0,2,4,7,9]
    aa_lon = (acl[row_lon,:])[:,row_lon]
    eval_lon,evec_lon = np.linalg.eig(aa_lon)
    en = eval_lon[:,np.newaxis]
    report_latex(en,"\lambda_{lon}",decimals=5,predecimals=5,print_report=True)
    #
    #
    #
    # r
    # dutch roll lims, spiral
    zn_dr_min = 0.4
    sg_dr_min = 0.4
    wn_dr_min = 1.0
    #
    row = [1,3,5,6,8,10]
    krP = -0.8
    acl = A - np.matmul(B,np.block([
        [np.zeros((2,A.shape[0]))],
        [np.array([0.0]*5 + [krP] + [0.0]*5)]
    ]))
    sys = co.ss((acl[row,:])[:,row],B[row,2],C[5,row],D)
    tf = co.ss2tf(sys)
    # print(tf)
    # print(co.ssdata(sys))
    print_tf(tf,"r","dr")
    k = -np.logspace(-1,2,1000)
    # print(k)
    krI = -15.0
    kai = np.argmin(np.abs(k-krI)) # 
    # print(k[kai])
    r,_ = co.rlocus(sys,kvect=k)
    print("evals at goal =",r[kai])
    # report wn, zt
    rat = r[kai]
    sg = -np.real(rat)
    wd = np.abs(np.imag(rat))
    e0s = np.array([complex(-sg[j],wd[j]) for j in range(len(sg))])
    e1s = np.conjugate(e0s)
    wn = np.sqrt(e0s*e1s)
    zt = - (e0s + e1s)/2./np.sqrt(e0s*e1s)
    t2d = -np.log(2.0)/sg
    print("   wn at goal =",wn)
    print(" zeta at goal =",zt)
    print("t2dbl at goal =",t2d)
    print("dr    wn min  =",wn_dr_min)
    print("dr  zeta min  =",zn_dr_min)
    print("dr    sg min  =",sg_dr_min)
    print("sl t2dbl min  =",20.0)
    # print(r.shape)
    plt.close()
    fig,axs = plt.subplots(figsize=(3.25,3.5),dpi=300.0,constrained_layout=True)
    # print(r.shape,k.shape)
    for i in range(r.shape[1]):
        ri = r[:,i]
        axs.plot(np.real(ri),np.imag(ri))#,c="k")
        axs.plot(np.real(ri[0]),np.imag(ri[0]),c="k",marker="x")
        axs.plot(np.real(ri[kai]),np.imag(ri[kai]),c="0.5",marker=".")
    #
    axs.set_xlabel("Real [s]")
    axs.set_ylabel("Imaginary [s]")
    axs.set_title("Root Locus")
    axs.grid(which="major",lw=0.6,ls="-",c="0.75")
    fig.savefig(save_folder + "/r_rlocus.png",dpi=300.0)
    show_r_rlocus = False
    if show_r_rlocus:
        plt.show()
    else:
        plt.close("all")
    #
    #
    # new closed loop system
    acl = acl - np.matmul(B,np.block([
        [np.zeros((2,A.shape[0]))],
        [np.array([0.0]*10 + [krI])]
    ]))
    # eigenvalues
    row_lat = [1,3,5,6,8,10]
    aa_lat = (acl[row_lat,:])[:,row_lat]
    eval_lat,evec_lat = np.linalg.eig(aa_lat)
    et = eval_lat[:,np.newaxis]
    report_latex(et,"\lambda_{lat}",decimals=5,predecimals=5,print_report=True)
    # all eigvals
    eval,evec = np.linalg.eig(acl)
    et = eval[:,np.newaxis]
    report_latex(et,"\lambda_{cl}",decimals=5,predecimals=5,print_report=True)
    print("kpP =",-kpP)
    print("kpI =",-kpI)
    print("kqP =",-kqP)
    print("kqI =",-kqI)
    print("krP =",-krP)
    print("krI =",-krI)

    # actual subsystem
    rows = [3,4,5,8,9,10]
    aa_cl = (acl[rows,:])[:,rows]
    print(aa_cl)
    eval,evec = np.linalg.eig(aa_cl)
    et = eval[:,np.newaxis]
    report_latex(et,"\lambda_{cl}",decimals=5,predecimals=5,print_report=True)
    

    return


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
        "aircraft_class" : TrackingAircraft,
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
        "name_end" : "_" + f1 + "_FB_1" # "_TK_3" # 
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
    run_bire_rc["track_check_time"] = run_base_rc["track_check_time"] = 5.0
    # bire_rc_dict["reference"] = base_rc_dict["reference"] = {
    #     "deg2rad_states" : [3,4,5],
    #     "3" : [
    #         [ 0.0, 1.0],
    #         [90.0, 1.0]
    #     ],
    #     # "4" : [
    #     #     [ 0.0, 1.0],
    #     #     [90.0, 1.0]
    #     # ],
    #     "sct_on_5" : False
    # }

    # run single case
    # di = [-1000.,0.,0.] # 
    # di = [-750.,0.,0.] # 
    # di = [-500.,0.,0.] # 
    # di = [-250.,0.,0.] # 
    di = [10.,2.,2.]
    # # 
    plot_vars["plot_full"] = True # False # 
    plot_vars["plot_delta"] = False # True # 
    plot_vars["zoom_deltas"] = False
    # plot_vars["format"] = "png" # "pdf" # 
    plot_vars["plot_norm"] = False # True # 
    #
    di = [0.,0.,0.]
    # di = [0.,1.,0.]
    # di = [5.,10.,7.] # see below
    # di = [0.,0.,1.]
    run_base_fs["num"] = run_bire_fs["num"] = \
        run_base_rc["num"] = run_bire_rc["num"] = 1  
    ##
    # # TK_3
    # zt_p,zt_q,zt_r =  0.7 , 0.7 , 0.7 
    # wn_p,wn_q,wn_r =  7.0 , 7.0 , 7.0 
    # # TK_5 ^^ and vv
    # base_rc_dict["simulation"]["integrator"] = "rk4"
    # # TK_4
    # zt_p,zt_q,zt_r =  0.9 , 0.7 , 0.9 
    # wn_p,wn_q,wn_r =  4.0 , 4.0 , 4.0 
    # # TK_6
    # zt_p,zt_q,zt_r =  0.7 , 0.7 , 0.7 
    # wn_p,wn_q,wn_r = 10.0 ,10.0 ,10.0 
    # # TK_7 -- FBK
    # zt_p,zt_q,zt_r =  0.7 , 0.7 , 0.7 
    # wn_p,wn_q,wn_r = 10.0 ,10.0 ,10.0 
    # # TK_8 -- FBk
    # zt_p,zt_q,zt_r =  0.8 , 0.8 , 0.8 
    # wn_p,wn_q,wn_r = 15.0 ,15.0 ,15.0 
    # BIRE FB_1
    zt_p,zt_q,zt_r =  0.7 , 0.7 , 0.7 
    wn_p,wn_q,wn_r =  7.0 , 7.0 , 7.0
    bire_rc_dict["simulation"]["integrator"] = "rk4" 
    #
    base_rc_dict["controller"]["gains"][ "K"] = \
        bire_rc_dict["controller"]["gains"][ "K"] = np.diag([
        2.*zt_p*wn_p,2.*zt_q*wn_q,2.*zt_r*wn_r
    ]).tolist()
    base_rc_dict["controller"]["gains"]["KI"] = \
        bire_rc_dict["controller"]["gains"]["KI"] = np.diag([
        wn_p**2.,wn_q**2.,wn_r**2.
    ]).tolist()
    # base_rc_dict["simulation"]["include_stall"] = \
    #     bire_rc_dict["simulation"]["include_stall"] = False
    # base_rc_dict["actuators"]["order"] = \
    #     bire_rc_dict["actuators"]["order"] = 0 # 1 # 2 # 
    # run_base_rc["state_threshold"] = [
    #     10., 15., 15.,
    #     0.5, 0.5, 0.5, # 20., 10., 10., # 
    #     1., 1., 50.,
    #     25., 10., 1.,
    #     # 5., 5., 5., 0.05
    # ]
    # run_base_rc["aircraft_class"] = FeedbackLinearizationAircraft
    run_bire_rc["aircraft_class"] = BIREFeedbackLinearizationAircraft
    run_bire_fs["aircraft_class"] = BIREFeedbackLinearizationAircraft
    bire_fs_dict["simulation"]["integrator"] = "rk4" 
    bire_fs_dict["controller"] = {
        "enforce_update_frequency" : False,
        "update_frequency[hz]" : 100.0,
        "type" : "gains",
        "name" : "gains",
        "integral_states" : [3,4,5],
        "gains" : {
            "K" : [ [ -10.0,  0.0,  12.0],
                    [  0.0, -5.0, -4.0],
                    [  0.0,  4.0, 30.0]],
            "KI" :[ [ -1.0,  0.0,  0.0],
                    [  0.0, -5.0,  0.0],
                    [  0.0,  0.0,  5.0]]
        }
    }
    # bire_rc_dict["reference"] = bire_fs_dict["reference"] = {
    #     "deg2rad_states" : [3,4,5],
    #     "3" : [
    #         [ 0.0, 1.0],
    #         [ 5.0, 1.0],
    #         # [ 5.0, 0.0]
    #     ],
    #     "4" : [
    #         [ 0.0, 1.0],
    #         [ 5.0, 1.0],
    #         # [ 5.0, 0.0]
    #     ],
    #     # "5" : [
    #     #     [ 0.0, 1.0],
    #     #     [90.0, 1.0]
    #     # ],
    #     "sct_on_5" : False
    # }
    # SLF throttle :         0.2732387390088981
    # 15 deg bank throttle : 0.2757778155728541
    # # 30 deg bank
    # p_tr_deg = -0.0820880039056245
    # q_tr_deg =  0.8352580178704386
    # r_tr_deg =  1.4467093243808735
    # # 15 deg bank
    # p_tr_deg = -0.0361891562749016
    # q_tr_deg =  0.2007714630167870
    # r_tr_deg =  0.7492893006885849
    # 10 deg bank fullscale
    p_tr_deg = -0.0236847366216922
    q_tr_deg =  0.0886486340380570
    r_tr_deg =  0.5027513865539764
    # # 10 deg bank rc
    # p_tr_deg = -0.3336392202221911
    # q_tr_deg =  0.5580877578924874
    # r_tr_deg =  3.1650729550868739
    p_comm = 5.0 # 7.5 # 
    p_time = 2.0
    t_end = 30.0 # 25.0 # 
    tf = t_end + p_time + 8.0
    run_bire_fs["final_time"] = run_base_fs["final_time"] = \
        run_bire_rc["final_time"] = run_base_rc["final_time"] = tf
    bire_fs_dict["reference"] = bire_rc_dict["reference"] = {
        "deg2rad_states" : [3,4,5],
        # "3" : [
        #     [   0.0, p_comm],
        #     [p_time, p_comm],
        #     [p_time, p_tr_deg]
        # ],
        # "4" : [
        #     [   0.0, 0.0],
        #     [p_time, 0.0],
        #     [p_time, q_tr_deg]
        # ],
        # "5" : [
        #     [   0.0, 0.0],
        #     [p_time, 0.0],
        #     [p_time, r_tr_deg]
        # ],
        # "3" : [
        #     [ 0.0, p_tr_deg],
        #     [ 1.0, p_tr_deg]
        # ],
        # "4" : [
        #     [ 0.0, q_tr_deg],
        #     [ 1.0, q_tr_deg]
        # ],
        # "5" : [
        #     [ 0.0, r_tr_deg],
        #     [ 1.0, r_tr_deg]
        # ],
        "3" : [
            [         0.0,   p_comm],
            [      p_time,   p_comm],
            [      p_time, p_tr_deg],
            [       t_end, p_tr_deg],
            [       t_end,  -p_comm],
            [t_end+p_time,  -p_comm],
            [t_end+p_time,      0.0]
        ],
        "4" : [
            [   0.0, 0.0],
            [p_time, 0.0],
            [p_time, q_tr_deg],
            [ t_end, q_tr_deg],
            [ t_end, 0.0]
        ],
        "5" : [
            [ 0.0 , 0.0],
            [p_time, 0.0],
            [p_time, r_tr_deg],
            [ t_end, r_tr_deg],
            [ t_end, 0.0]
        ],
        "sct_on_5" : False
    }
    # run_bire_fs["trim_bank"] = run_bire_rc["trim_bank"] = 15.0 # 10.0 # 
    # run_bire_fs["has_turbulence"] = run_base_rc["has_turbulence"] = True
    # run_bire_fs["aircraft_class"] = BIRELyapunovAircraft 
    # run_bire_fs["final_time"] = 1. 
    # bire_fs_dict["simulation"]["include_compressibility"] = False
    # # # bire_fs_dict["simulation"]["limit_input"] = False
    # # # bire_fs_dict["simulation"]["limit_input_rates"] = False
    #
    run_single_simulation(bire_fs_dict,rtdst_1sg=di,**run_bire_fs,**plot_vars)
    # run_single_simulation(base_fs_dict,rtdst_1sg=di,**run_base_fs,**plot_vars)
    # run_single_simulation(bire_rc_dict,rtdst_1sg=di,**run_bire_rc,**plot_vars)
    # run_single_simulation(base_rc_dict,rtdst_1sg=di,**run_base_rc,**plot_vars)
    quit()

    # # # run monte carlo perturbation analysis
    # num = 1000
    # run_base_fs["num"] = run_bire_fs["num"] = \
    #     run_base_rc["num"] = run_bire_rc["num"] = num
    di = [2.,2.,2.] # RC F-16 # change r to 6
    # # di = [0.,0.,0.]
    di = [8.,8.,0.2] # FS BIRE
    run_bire_fs["state_threshold"] = [
        10., 15., 15.,
        2.0, 2.0, 2.0, # 20., 10., 10., # 
        1., 1., 50.,
        25., 10., 1.,
        5., 5., 5., 0.05
    ]
    run_bire_fs["final_time"] = run_base_fs["final_time"] = \
        run_bire_rc["final_time"] = run_base_rc["final_time"] = 15.0
    run_base_fs["num"] = run_bire_fs["num"] = \
        run_base_rc["num"] = run_bire_rc["num"] = 1000
    # run_bire_fs["has_model_error"] = run_base_fs["has_model_error"] = \
    #     run_bire_rc["has_model_error"] = run_base_rc["has_model_error"] = True # False # 
    # monte_carlo_perturbations(bire_fs_dict,rtdst_1sg=di,**run_bire_fs,**plot_vars)
    # # monte_carlo_perturbations(base_fs_dict,rtdst_1sg=di,**run_base_fs,**plot_vars)
    # # monte_carlo_perturbations(bire_rc_dict,rtdst_1sg=di,**run_bire_rc,**plot_vars)
    # # monte_carlo_perturbations(base_rc_dict,rtdst_1sg=di,**run_base_rc,**plot_vars)
    # quit()
    # #
    # # single axis pqr dispersions
    # disa = [[10.,0.,0.],[0.,20.,0.],[0.,0.,10.]]
    # disa = [[10.,0.,0.],[0.,10.,0.],[0.,0.,1.0]]
    # for i in [2]: # [1]: # range(3): # 
    #     ds = disa[i]
    #     monte_carlo_perturbations(bire_fs_dict,rtdst_1sg=ds,**run_bire_fs,**plot_vars)
    #     # monte_carlo_perturbations(base_fs_dict,rtdst_1sg=ds,**run_base_fs,**plot_vars)
    #     # monte_carlo_perturbations(bire_rc_dict,rtdst_1sg=ds,**run_bire_rc,**plot_vars)
    #     # monte_carlo_perturbations(base_rc_dict,rtdst_1sg=ds,**run_base_rc,**plot_vars)
    # quit()
    # #
    # # single FM error dispersions
    # names = ["CL","CS","CD","Cl","Cm","Cn"]
    # run_base_fs["has_model_error"] = run_bire_fs["has_model_error"] = \
    #     run_base_rc["has_model_error"] = run_bire_rc["has_model_error"] = True
    # f1 = "LGN"
    # for i in [5]: # [1]: # [0]: # range(len(names)): # 
    #     name = names[i]
    #     # create FM errors
    #     FM_error_list = np.zeros((6,))
    #     FM_error_list[i] = 0.25
    #     run_base_fs["FM_errors"] = run_bire_fs["FM_errors"] = \
    #         run_base_rc["FM_errors"] = run_bire_rc["FM_errors"] = \
    #         FM_error_list*1.
    #     run_base_fs["name_end"] = run_bire_fs["name_end"] = \
    #         run_base_rc["name_end"] = run_bire_rc["name_end"] = \
    #         (f1 != "C2")*("_" + f1) + "_FB_1" + "_" + name
    #     monte_carlo_perturbations(bire_fs_dict,rtdst_1sg=di,**run_bire_fs,**plot_vars)
    #     # monte_carlo_perturbations(base_fs_dict,rtdst_1sg=di,**run_base_fs,**plot_vars)
    #     # monte_carlo_perturbations(bire_rc_dict,rtdst_1sg=di,**run_bire_rc,**plot_vars)
    #     # monte_carlo_perturbations(base_rc_dict,rtdst_1sg=di,**run_base_rc,**plot_vars)
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







    # # tracking BIRE transformed system
    # zt = 0.7
    # wn = 7.0
    # k1 = np.diag([wn**2.]*3)
    # k2 = np.diag([2.*zt*wn]*3)
    # Z = np.zeros((3,3))
    # I = np.eye(3)
    # A = np.block([[Z,I],[-k1,-k2]])
    # print(np.linalg.eig(A)[0])
    # #
    # x0 = np.array([1.]*6)
    # dyn = lambda t,x : np.matmul(A,x)
    # ts = np.linspace(0.0,5.0,num=500)
    # xs = odeint(dyn,x0,ts,tfirst=True).T
    # fig,axs = plt.subplots(6,
    #     figsize=(3.5,3.5),dpi=300.0,sharex=True,constrained_layout=True)
    # for i in range(xs.shape[0]):
    #     axs[i].plot(ts,xs[i],"k",lw=1.0)
    # axs[5].set_xlim(ts[0],ts[-1])
    
    # plt.show()