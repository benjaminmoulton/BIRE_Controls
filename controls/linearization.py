import numpy as np
import control as co
from matplotlib import pyplot as plt
from math import pi, sin, cos, tan, exp, asin, atan, atan2
from scipy.linalg import block_diag
# from slycot.exceptions import SlycotArithmeticError
from std_atm import stdatm_english, stdatm_derivative_english
import warnings as wrn

import sys
aero_directory = '../aerodynamics_model/'
mass_directory = '../mass_properties/'

sys.path.insert(1, aero_directory)
sys.path.insert(1, mass_directory)

from os import mkdir, rmdir, remove, listdir
from os.path import exists as path_exists

from f16_aero import F16Aero
from bire_aero import BIREAero
from inertia_model import InertiaModel


class linearization:
    """ A class which simulates the flight of an aircraft.

    Parameters
    ----------
    trim_state : np.ndarray
        The trim state, formatted as
        [u, v, w, p, q, r, x, y, z, phi, theta, psi] for euler angles or for 
        quaternions as
        [u, v, w, p, q, r, x, y, z, e0, ex, ey, ez].

    trim_input : np.ndarray
        The trim input, formatted as [da, de, dr, tau].

    use_quaternion : bool, optional
        Whether to return a linearization that uses quaternions. Defaults to
        True.
    
    is_bire : bool, optional
        Whether to linearize the equations of motion for the BIRE aircraft.
        Defaults to False.
    
    Raises
    ------
    IOError
        If input filepath or filename is invalid.
    """

    def __init__(self, x_eq, u_eq, cg_shift = [0.,0.,0.], **kwargs):


        # pull out key-word arguments
        self.use_quats = False # kwargs.get("use_quaternion", True) # 
        self.is_bire = kwargs.get("is_bire", False)
        self.is_rc = kwargs.get("is_rc", False)
        self.is_stevens_and_lewis = kwargs.get("is_stevens_and_lewis",False)
        self.compressible = kwargs.get("compressible", True)
        self.use_Anderson = kwargs.get("use_Anderson", True)
        self.enforce_stall = kwargs.get("enforce_stall", True)
        self.include_stall = kwargs.get("include_stall",False)
        self.use_simple_thrust = kwargs.get("use_simple_thrust_model", False)
        self.given_aero_model = kwargs.get("aero_model","None")
        self.aero_dict = {
            "compressible" : self.compressible,
            "use_Anderson" : self.use_Anderson,
            "enforce_stall" : self.enforce_stall
        }
        self.use_numerical = kwargs.get("use_numerical_linearization",False)
        self.numerical_dynamics = kwargs.get("numerical_dynamics",None)
        self.use_VAB_format = kwargs.get("use_VAB_format",False)
        self.turn_off_warnings = kwargs.get("turn_off_warnings",False)
        self.controller_type = kwargs.get("controller_type","LQR")
        self.xIi = kwargs.get("integral_states",[])
        self.xPi = kwargs.get("principal_states",[])
        self.additional_states = kwargs.get("additional_states",0)
        self.controller_dict = kwargs.get("controller_properties",{})
        self.actuators_dict = kwargs.get("actuators_properties",{})
        self.mrrr = kwargs.get("min_realization_removal_rows",None)
        self.mrrc = kwargs.get("min_realization_removal_cols",None)
        self.drop_actrs = kwargs.get("drop_actuators",True)
        run_frequency_analysis = kwargs.get("run_frequency_analysis",True)
        self.report = kwargs.get("report",True)
        self.freq_folder = kwargs.get("freq_folder","./")
        controller_name = kwargs.get("controller_name","")

        # report
        method = self.use_quats*"quaternion" +(not self.use_quats)*"   euler  "
        aircraft = self.is_bire*"BIRE" + (not self.is_bire)*"F-16"
        if self.report:
            print("building "+method+" linearization for the " +aircraft+"...")

        # make x_eq, u_eq, and cg_shift numpy arrays
        x_eq = np.array(x_eq)
        u_eq = np.array(u_eq)
        cg_shift = np.array(cg_shift)
        # put in VAB format if requested
        if self.use_VAB_format:
            V = (x_eq[0]**2. + x_eq[1]**2. + x_eq[2]**2.)**0.5
            a = atan2(x_eq[2],x_eq[0])
            b = asin(x_eq[1]/V)
            x_eq[0] = V*1.
            x_eq[1] = a*1.
            x_eq[2] = b*1.

        # pull in parameters
        self._get_input_variables(x_eq,u_eq)

        # run linearization
        self.A,self.B = self.build_jacobians(x_eq,u_eq,cg_shift,
            numerical = self.use_numerical,
            numerical_dynamics = self.numerical_dynamics)
        self.D = np.zeros((self.C.shape[0],self.B.shape[1]))

        # truncate x,y,z (and psi for euler) rows / cols from A,B
        self.A_min = np.matmul(np.matmul(  self.C,self.A),  self.C.T)
        self.B_min = np.matmul(np.matmul(  self.C,self.B),self.C_u.T)
        self.A_full = (self.A[self.Xslice,:])[:,self.Xslice]
        self.B_full = (self.B[self.Xslice,:])
        if self.controller_type == "LQR":
            self.Q_min = np.matmul(np.matmul(  self.C,self.Q),  self.C.T)
            self.R_min = np.matmul(np.matmul(self.C_u,self.R),self.C_u.T)
        
        # try to make (B^-1)(-A)
        try:
            self.Binv_min = np.linalg.solve(self.B_min,np.eye(3))
            self.nBiA_min = -np.matmul(self.Binv_min,self.A_min)
        except:
            self.nBiA_min = "did not work!!"
            print("(B^-1)(-A) " + self.nBiA_min)
            pass

        # run controllability
        if self.controller_type != "none":
            self.Gamma_min, self.Gamma_rank_min, self.A_min_eigs, \
                self.A_min_evecs = \
                self._analyze_controllability(self.A_min,self.B_min)
            self.Gamma, self.Gamma_rank, self.A_eigs, self.A_evecs = \
                self._analyze_controllability(self.A,self.B,is_minimal=False)
        
        # build lqr state feedback matrix
        self.KI = np.array(self.controller_dict.get("KI",
            np.eye(len(self.xIi))*0.))
        if self.controller_type == "LQR":
            rest = [self.Q_min,self.R_min]
        else:
            rest = [1,1]
        if len(self.xIi):
            rest = rest + [self.KI]
        else:
            rest = rest + [1]
        # build and analyze controller
        if self.controller_type != "none":
            self.K,self.P,self.Acl,self.A_BK_eigs,self.A_BK_evecs = \
                self._build_feedback(self.A_min,self.B_min,*rest)

            # analyze control design
            if run_frequency_analysis:
                self._analyze_sensitivity(controller_name)


    def _get_input_variables(self, x_eq, u_eq):

        # pre-check order of actuator dynamics
        actuators = self.actuators_dict
        self.order = actuators.get("order",1)

        # check to make sure arrays are the correct length
        l_i = len(self.xIi)
        add = self.additional_states
        if self.use_quats and x_eq.shape[0] != 13 + 4*self.order + l_i + add:
            raise TypeError("x array is not in full state quaternion format")
        if not self.use_quats and x_eq.shape[0] != 12 + 4*self.order + l_i+add:
            raise TypeError("x array is not in full state euler format")
        if u_eq.shape[0] != 4:
            raise TypeError("u array is not in full input (4 values) format")

        # create aero model
        if self.given_aero_model == "None":
            if self.is_bire:
                self.aero_model = BIREAero(inp_dir=aero_directory,
                    thrust_dir=aero_directory,use_rc_thrust_model=self.is_rc)
            else:
                self.aero_model = F16Aero(inp_dir=aero_directory,
                    thrust_dir=aero_directory,use_rc_thrust_model=self.is_rc)
        else:
            self.aero_model = self.given_aero_model
        self.inertia_model = InertiaModel(inp_dir=mass_directory, \
            is_bire=self.is_bire,is_rc=self.is_rc)

        # LQR sensitivity matrices
        if self.controller_type == "LQR":
            if self.is_bire:
                if self.use_quats:
                    Q_diag = self.controller_dict.get("Q",[
                        0.0, 0.0, 0.0, 20.0, 10.0, 10.0, 
                        0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 0.0
                    ])
                else:
                    Q_diag = self.controller_dict.get("Q",[
                        0.0, 0.0, 0.0, 20.0, 10.0, 10.0, 
                        0.0, 0.0, 0.0, 10.0, 10.0, 0.0
                    ])
                Q1a_diag = self.controller_dict.get("Q1a",[0.0, 0.0, 0.0])
                Q2a_diag = self.controller_dict.get("Q2a",[0.0, 0.0, 0.0])
                self.R = np.diag( self.controller_dict.get("R",
                    [2.0, 0.1, 0.1, 1000.]) )
            else:
                if self.use_quats:
                    Q_diag = self.controller_dict.get("Q",[
                        0.0, 0.0, 0.0, 10.0, 10.0, 10.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.0
                    ])
                else:
                    Q_diag = self.controller_dict.get("Q",[
                        0.0, 0.0, 0.0, 10.0, 10.0, 10.0,
                        0.0, 0.0, 0.0, 0.0, 20.0, 0.0
                    ])
                Q1a_diag = self.controller_dict.get("Q1a",[0.0, 0.0, 0.0])
                Q2a_diag = self.controller_dict.get("Q2a",[0.0, 0.0, 0.0])
                self.R = np.diag( self.controller_dict.get("R",
                    [2.0, 1.0, 2.0, 1000.]) )
            
            # actuator dynamics con't
            if self.order == 0:
                self.Q = np.diag(Q_diag)
            if self.order == 1:
                self.Q = np.diag(Q_diag + Q1a_diag)
            elif self.order == 2:
                self.Q = np.diag(Q_diag + Q1a_diag + Q2a_diag)
        elif self.controller_type == "gains":
            self.K_in = np.array(
                self.controller_dict.get("K",-2.*np.eye(x_eq.shape[0])))
        elif self.controller_type == "none":
            pass
        else:
            raise TypeError("Invalid controller type: {}".format(
                self.controller_type) + ", should be 'LQR' or 'gains'")
        ##
        if self.order >= 0:
            aileron = actuators.get("aileron",{})
            self.s_da = 1. / aileron.get("lag[s]", 0.0495)
            ##
            elevator = actuators.get("elevator",{})
            self.s_de = 1. / elevator.get("lag[s]", 0.0495)
            ##
            if self.is_bire:
                yaw_surface = actuators.get("BIRE",{})
                self.s_dB = 1. / yaw_surface.get("lag[s]", 0.0495)
            else:
                yaw_surface = actuators.get("rudder",{})
                self.s_dr = 1. / yaw_surface.get("lag[s]", 0.0495)
            ##
            throttle = actuators.get("throttle",{})
            if self.order == 2:
                self.z_da = aileron.get("damping_ratio", 0.7)
                self.w_da = aileron.get("bandwidth[rad/s]", 30.)
                ##
                self.z_de = elevator.get("damping_ratio", 0.7)
                self.w_de = elevator.get("bandwidth[rad/s]", 30.)
                ##
                if self.is_bire:
                    self.z_dB = yaw_surface.get("damping_ratio", 0.7)
                    self.w_dB = yaw_surface.get("bandwidth[rad/s]", 30.0)
                else:
                    self.z_dr = yaw_surface.get("damping_ratio", 0.7)
                    self.w_dr = yaw_surface.get("bandwidth[rad/s]", 30.0)
                ##
                self.z_tau = throttle.get("damping_ratio", 0.7)
                self.w_tau = throttle.get("bandwidth[rad/s]", 2000.)
        
        # minimum realization removal rows
        og1 = self.order >= 1
        oe2 = self.order == 2
        if self.mrrr == None:
            if self.use_quats:
                self.mrrr = [6,7,9,12]
            else:
                self.mrrr = [6,7,11]
        self.mrrr += self.xIi
        if add != 0:
            self.mrrr += list(range(x_eq.shape[0]))[-add:]
        q = self.use_quats*1
        self.mrrr_mdn = self.mrrr + og1*[12+q,13+q,14+q,15+q] + \
            oe2*[16+q,17+q,18+q,19+q]
        if self.is_stevens_and_lewis and self.use_numerical:
            self.mrrr_mdn = self.mrrr + og1*[12+q,13+q,14+q] + \
                oe2*[16+q,17+q,18+q]
        #####
        if self.use_quats:
            first_bit = [6,7,9,12]
        else:
            first_bit = [6,7,11]
        self.Xslice = first_bit + og1*[12+q,13+q,14+q,15+q] + \
            oe2*[16+q,17+q,18+q,19+q] + self.xIi
        self.Xslice = np.delete(np.arange(x_eq.shape[0]),self.Xslice)
        #####
        C = np.eye(x_eq.shape[0],dtype=int)
        C_mdn = np.delete(C,self.mrrr_mdn,axis=0)
        C = np.delete(C,self.mrrr,axis=0)
        if self.drop_actrs:
            C = C_mdn
        else:
            C = C
        self.Cslice = np.matmul(C,np.arange(x_eq.shape[0]),dtype=int).tolist()
        self.C = C
        if self.controller_type == "LQR":
            H_inds = np.delete(np.arange(12+self.use_quats*1+self.order*4),\
                np.nonzero(np.diag(self.Q))).tolist() + self.mrrr_mdn
        elif self.controller_type == "gains":
            H_inds = self.mrrr_mdn
        else:
            H_inds = []
        H = np.delete(np.eye(x_eq.shape[0],dtype=int),H_inds,axis=0)
        self.Hslice = np.matmul(H,np.arange(x_eq.shape[0]),dtype=int).tolist()
        self.H = H
        # minimum realization removal cols
        if self.mrrc == None:
            self.mrrc = []
        C_u = np.eye(u_eq.shape[0],dtype=int)
        C_u = np.delete(C_u,self.mrrc,axis=0)
        self.Cuslice = np.matmul(C_u,np.arange(u_eq.shape[0]),dtype=int).tolist()
        self.C_u = C_u
        # save xhat
        self.xhat_eq = np.matmul(self.C,x_eq)
        self.uhat_eq = np.matmul(self.C_u,u_eq)


    def _throttle_gain(self,tau):
        if tau <= 0.3:
            return 1.0
        elif 0.3 <= tau <= 0.5:
            return 1. / (2.35 - 4.5 * tau)
        else: # tau >= 0.5
            return 10.0


    def _build_state_jacobian(self, x_eq, u_eq, cg_shift = [0.,0.,0.]):
        # report
        if self.report:
            print("    building state jacobian...")

        # pull out evaluating condition
        if self.use_quats: u,v,w,p,q,r,x,y,z, e0, ex, ey,ez = x_eq[0:13]
        else:              u,v,w,p,q,r,x,y,z,phi,tht,psi    = x_eq[0:12]
        if self.is_bire: da, de, dB, tau = u_eq
        else:            da, de, dr, tau = u_eq
        Dxcg, Dycg, Dzcg = cg_shift
        C = self.aero_model
        IM = self.inertia_model

        # values for later use
        a = atan2(w,u)
        V = (u*u + v*v + w*w)**0.5
        b = asin(v/V)
        #
        Ca = cos(a); Sa = sin(a)
        Cb = cos(b); Sb = sin(b)
        #
        if self.use_quats:
            e02, ex2, ey2, ez2 = e0*e0, ex*ex, ey*ey, ez*ez
            e0ex, e0ey, e0ez = e0*ex, e0*ey, e0*ez
            exey, exez, eyez = ex*ey, ex*ez, ey*ez
            e0u, e0v, e0w = e0*u, e0*v, e0*w
            exu, exv, exw = ex*u, ex*v, ex*w
            eyu, eyv, eyw = ey*u, ey*v, ey*w
            ezu, ezv, ezw = ez*u, ez*v, ez*w
            e0upeywmezv = e0u + eyw - ezv
            exupeyvpezw = exu + eyv + ezw
            e0wpexvmeyu = e0w + exv - eyu
            e0vmexwpezu = e0v - exw + ezu
        else:
            Cp = cos(phi); Sp = sin(phi)
            Ct = cos(tht); St = sin(tht)
            Cs = cos(psi); Ss = sin(psi)
        #
        Rlon = C.c_w/2./V
        Rlat = C.b_w/2./V
        #
        pbar = p*Rlat
        qbar = q*Rlon
        rbar = r*Rlat
        #
        _,g,_,_, rho,sos = stdatm_english( -z)
        _,_,_,_,rho0,_   = stdatm_english(0.0)
        _,g_H,_,_,R_H,S_H = stdatm_derivative_english( -z)
        g_z = -g_H; rho_z = -R_H; sos_z = -S_H
        # #####################
        # g = 32.12780074195162
        # print("g =",g)
        # #####################
        #
        M = V / sos
        #
        Qdyn = 0.5*rho*V**2.*C.S_w
        Qlon = Qdyn*C.c_w
        Qlat = Qdyn*C.b_w
        #
        m = IM.W/g
        if self.is_bire:
            Iinv = np.array(IM.inverse_tensor(dB))
            [Ixx, Iyy, Izz, Ixy, Ixz, Iyz] = IM.inertia_results(dB)
        else:
            Iinv = np.array(IM.inverse_tensor(0.))
            [Ixx, Iyy, Izz, Ixy, Ixz, Iyz] = IM.inertia_results(0.)
        hx, hy, hz = IM.angular_momentum_results()
        
        # component derivatives for later use
        a_u = - w/(u**2.0 + w**2.0)
        a_w =   u/(u**2.0 + w**2.0)
        b_u = - u*v/V**2.0/(u**2.0 + w**2.0)**0.5
        b_v = (u**2.0 + w**2.0)**0.5/V**2.0
        b_w = - v*w/V**2./(u**2.0 + w**2.0)**0.5
        #
        pbar_u = - pbar*u/V**2.0
        pbar_v = - pbar*v/V**2.0
        pbar_w = - pbar*w/V**2.0
        #
        qbar_u = - qbar*u/V**2.0
        qbar_v = - qbar*v/V**2.0
        qbar_w = - qbar*w/V**2.0
        #
        rbar_u = - rbar*u/V**2.0
        rbar_v = - rbar*v/V**2.0
        rbar_w = - rbar*w/V**2.0
        #
        Q_u = rho*C.S_w*u
        Q_v = rho*C.S_w*v
        Q_w = rho*C.S_w*w
        Qlon_u = Q_u*C.c_w
        Qlon_v = Q_v*C.c_w
        Qlon_w = Q_w*C.c_w
        Qlat_u = Q_u*C.b_w
        Qlat_v = Q_v*C.b_w
        Qlat_w = Q_w*C.b_w
        #
        Q_z = 0.5*rho_z*V**2.0*C.S_w
        Qlon_z = Q_z*C.c_w
        Qlat_z = Q_z*C.b_w

        # state aerodynamic force derivatives
        if self.is_bire:
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
        else:
            # for use
            CL1 = C.CL0 + C.CLa*a
            CS1 = C.CSb*b
            # lift
            oCL_u = C.CLa * a_u + C.CLq * qbar_u
            oCL_v = C.CLq * qbar_v
            oCL_w = C.CLa * a_w + C.CLq * qbar_w
            oCL_q = C.CLq * Rlon
            # side
            oCS_u = C.CSb*b_u + C.CSLp*C.CLa*a_u*pbar + \
                + (C.CSLp*CL1 + C.CSp)*pbar_u + C.CSr*rbar_u
            oCS_v = C.CSb*b_v + (C.CSLp*CL1 + C.CSp)*pbar_v + C.CSr*rbar_v
            oCS_w = C.CSb*b_w + C.CSLp*C.CLa*a_w*pbar + \
                + (C.CSLp*CL1 + C.CSp)*pbar_w + C.CSr*rbar_w
            oCS_p = (C.CSLp*CL1 + C.CSp)*Rlat
            oCS_r = C.CSr*Rlat
            # drag
            oCD_u = (C.CDL + 2.*C.CDL2*CL1 + (2.*C.CDL2q*CL1 + C.CDLq)*qbar + \
                + C.CDLde*de)*C.CLa*a_u + (2.*C.CDS2*CS1 + C.CDSp*pbar + \
                + C.CDSr*rbar + C.CDSda*da + C.CDSdr*dr)*C.CSb*b_u + \
                + C.CDSp*CS1*pbar_u + (C.CDL2q*CL1**2. + C.CDLq*CL1 + \
                + C.CDq)*qbar_u + C.CDSr*CS1*rbar_u
            oCD_v = (2.*C.CDS2*CS1 + C.CDSp*pbar + \
                + C.CDSr*rbar + C.CDSda*da + C.CDSdr*dr)*C.CSb*b_v + \
                + C.CDSp*CS1*pbar_v + (C.CDL2q*CL1**2. + C.CDLq*CL1 + \
                + C.CDq)*qbar_v + C.CDSr*CS1*rbar_v
            oCD_w = (C.CDL + 2.*C.CDL2*CL1 + (2.*C.CDL2q*CL1 + C.CDLq)*qbar + \
                + C.CDLde*de)*C.CLa*a_w + (2.*C.CDS2*CS1 + C.CDSp*pbar + \
                + C.CDSr*rbar + C.CDSda*da + C.CDSdr*dr)*C.CSb*b_w + \
                + C.CDSp*CS1*pbar_w + (C.CDL2q*CL1**2. + C.CDLq*CL1 + \
                + C.CDq)*qbar_w + C.CDSr*CS1*rbar_w
            oCD_p = C.CDSp*CS1*Rlat
            oCD_q = (C.CDL2q*CL1**2. + C.CDLq*CL1 + C.CDq)*Rlon
            oCD_r = C.CDSr*CS1*Rlat
            # zero values
            oCL_p = oCL_r = oCS_q = 0.0

        # state aerodynamic moment derivatives
        if self.is_bire:
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
        else:
            # roll
            oCl_u = C.Clb*b_u + C.Clp*pbar_u + C.ClLr*C.CLa*a_u*rbar + \
                + (C.ClLr*CL1 + C.Clr)*rbar_u
            oCl_v = C.Clb*b_v + C.Clp*pbar_v + (C.ClLr*CL1 + C.Clr)*rbar_v
            oCl_w = C.Clb*b_w + C.Clp*pbar_w + C.ClLr*C.CLa*a_w*rbar + \
                + (C.ClLr*CL1 + C.Clr)*rbar_w
            oCl_p = C.Clp*Rlat
            oCl_r = (C.ClLr*CL1 + C.Clr)*Rlat
            # pitch
            oCm_u = C.Cma*a_u + C.Cmq*qbar_u
            oCm_v = C.Cmq*qbar_v
            oCm_w = C.Cma*a_w + C.Cmq*qbar_w
            oCm_q = C.Cmq*Rlon
            # yaw
            oCn_u = (C.CnLp*pbar + C.CnLda*da)*C.CLa*a_u + \
                + C.Cnb*b_u + (C.CnLp*CL1 + C.Cnp)*pbar_u + C.Cnr*rbar_u
            oCn_v = C.Cnb*b_v + (C.CnLp*CL1 + C.Cnp)*pbar_v + C.Cnr*rbar_v
            oCn_w = (C.CnLp*pbar + C.CnLda*da)*C.CLa*a_w + \
                + C.Cnb*b_w + (C.CnLp*CL1 + C.Cnp)*pbar_w + C.Cnr*rbar_w
            oCn_p = (C.CnLp*CL1 + C.Cnp)*Rlat
            oCn_r = C.Cnr*Rlat
            # zero values
            oCl_q = oCm_p = oCm_r = oCn_q = 0.0
        
        # get forces and moments at the specified condition
        if self.is_bire:
            params = [a, b, pbar, qbar, rbar, da, de, dB]
        else:
            params = [a, b, pbar, qbar, rbar, da, de, dr]
        [CL, CS, CD, Cl, Cm, Cn] = C.aero_results(*params,M=M,**self.aero_dict)

        # Stall corrections
        if self.include_stall and self.enforce_stall: # False: # 

            # unstallable incompressible coefficients
            [oCL, _, oCD, _, oCm, _] = \
                C.aero_results(*params,M=M,**{
                "compressible" : False,
                "use_Anderson" : False,
                "enforce_stall" : False
            })

            # pull out stall model coeffs
            CLp,CDp,Cmp,S,CLp_a,CDp_a,Cmp_a,S_a = \
                C._stall_correction_derivatives(a,oCL,oCD,oCm)
            
            # implement
            # lift
            aCL_u = (1. - S)*oCL_u + (S_a*(CLp - oCL) + S*CLp_a)*a_u
            aCL_v = (1. - S)*oCL_v
            aCL_w = (1. - S)*oCL_w + (S_a*(CLp - oCL) + S*CLp_a)*a_w
            aCL_p = (1. - S)*oCL_p
            aCL_q = (1. - S)*oCL_q
            aCL_r = (1. - S)*oCL_r
            # drag
            aCD_u = (1. - S)*oCD_u + (S_a*(CDp - oCD) + S*CDp_a)*a_u
            aCD_v = (1. - S)*oCD_v
            aCD_w = (1. - S)*oCD_w + (S_a*(CDp - oCD) + S*CDp_a)*a_w
            aCD_p = (1. - S)*oCD_p
            aCD_q = (1. - S)*oCD_q
            aCD_r = (1. - S)*oCD_r
            # pitch
            aCm_u = (1. - S)*oCm_u + (S_a*(Cmp - oCm) + S*Cmp_a)*a_u
            aCm_v = (1. - S)*oCm_v
            aCm_w = (1. - S)*oCm_w + (S_a*(Cmp - oCm) + S*Cmp_a)*a_w
            aCm_p = (1. - S)*oCm_p
            aCm_q = (1. - S)*oCm_q
            aCm_r = (1. - S)*oCm_r

            # coefficients which are not in stall model
            aCS_u,aCS_v,aCS_w = oCS_u,oCS_v,oCS_w
            aCS_p,aCS_q,aCS_r = oCS_p,oCS_q,oCS_r
            aCl_u,aCl_v,aCl_w = oCl_u,oCl_v,oCl_w
            aCl_p,aCl_q,aCl_r = oCl_p,oCl_q,oCl_r
            aCn_u,aCn_v,aCn_w = oCn_u,oCn_v,oCn_w
            aCn_p,aCn_q,aCn_r = oCn_p,oCn_q,oCn_r
        else:
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
        if self.compressible:
            # Mach derivatives
            M_u = 2.*u/V/sos
            M_v = 2.*v/V/sos
            M_w = 2.*w/V/sos
            M_p = M_q = M_r = 0.
            M_z = -V/sos/sos*sos_z

            # incompressible coefficients
            [aCL, aCS, aCD, aCl, aCm, aCn] = \
                C.aero_results(*params,M=M,**{
                "compressible" : False,
                "use_Anderson" : False,
                "enforce_stall" : self.enforce_stall
            })

            # Mach correction derivatives
            if M <= 1.0: # subsonic
                if self.use_Anderson:
                    L_w, L_h, L_v = C.Lam_w, C.Lam_h, C.Lam_v
                    R_w, R_h, R_v = C.RA_w, C.RA_h, C.RA_v

                    # derivatives wrt incompressible coefficients
                    CL_aCL = Anderson_correction_der_coeff(aCL,L_w,R_w,M)
                    Cm_aCm = Anderson_correction_der_coeff(aCm,L_w,R_w,M)
                    if self.is_bire:
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
                    if self.is_bire:
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
            CL_z =              + CL_M*M_z
            # side
            CS_u = CS_aCS*aCS_u + CS_M*M_u
            CS_v = CS_aCS*aCS_v + CS_M*M_v
            CS_w = CS_aCS*aCS_w + CS_M*M_w
            CS_p = CS_aCS*aCS_p + CS_M*M_p
            CS_q = CS_aCS*aCS_q + CS_M*M_q
            CS_r = CS_aCS*aCS_r + CS_M*M_r
            CS_z =              + CS_M*M_z
            # drag
            CD_u = aCD_u
            CD_v = aCD_v
            CD_w = aCD_w
            CD_p = aCD_p
            CD_q = aCD_q
            CD_r = aCD_r
            CD_z = 0.0
            # roll
            Cl_u = Cl_aCl*aCl_u + Cl_M*M_u
            Cl_v = Cl_aCl*aCl_v + Cl_M*M_v
            Cl_w = Cl_aCl*aCl_w + Cl_M*M_w
            Cl_p = Cl_aCl*aCl_p + Cl_M*M_p
            Cl_q = Cl_aCl*aCl_q + Cl_M*M_q
            Cl_r = Cl_aCl*aCl_r + Cl_M*M_r
            Cl_z =              + Cl_M*M_z
            # pitch
            Cm_u = Cm_aCm*aCm_u + Cm_M*M_u
            Cm_v = Cm_aCm*aCm_v + Cm_M*M_v
            Cm_w = Cm_aCm*aCm_w + Cm_M*M_w
            Cm_p = Cm_aCm*aCm_p + Cm_M*M_p
            Cm_q = Cm_aCm*aCm_q + Cm_M*M_q
            Cm_r = Cm_aCm*aCm_r + Cm_M*M_r
            Cm_z =              + Cm_M*M_z
            # yaw
            Cn_u = Cn_aCn*aCn_u + Cn_M*M_u
            Cn_v = Cn_aCn*aCn_v + Cn_M*M_v
            Cn_w = Cn_aCn*aCn_w + Cn_M*M_w
            Cn_p = Cn_aCn*aCn_p + Cn_M*M_p
            Cn_q = Cn_aCn*aCn_q + Cn_M*M_q
            Cn_r = Cn_aCn*aCn_r + Cn_M*M_r
            Cn_z =              + Cn_M*M_z
        else:
            # no compressibility
            CL_u,CL_v,CL_w,CL_p,CL_q,CL_r = aCL_u,aCL_v,aCL_w,aCL_p,aCL_q,aCL_r
            CS_u,CS_v,CS_w,CS_p,CS_q,CS_r = aCS_u,aCS_v,aCS_w,aCS_p,aCS_q,aCS_r
            CD_u,CD_v,CD_w,CD_p,CD_q,CD_r = aCD_u,aCD_v,aCD_w,aCD_p,aCD_q,aCD_r
            Cl_u,Cl_v,Cl_w,Cl_p,Cl_q,Cl_r = aCl_u,aCl_v,aCl_w,aCl_p,aCl_q,aCl_r
            Cm_u,Cm_v,Cm_w,Cm_p,Cm_q,Cm_r = aCm_u,aCm_v,aCm_w,aCm_p,aCm_q,aCm_r
            Cn_u,Cn_v,Cn_w,Cn_p,Cn_q,Cn_r = aCn_u,aCn_v,aCn_w,aCn_p,aCn_q,aCn_r
            #
            CL_z = CS_z = CD_z = Cl_z = Cm_z = Cn_z = 0.0

        # thrust state derivatives
        T_V,T_H = C.Prop.T_V_H_ders(tau,-z,V)
        T_z = -T_H
        # TM = C.Prop
        # if self.use_simple_thrust:
        #     T_V = tau*(rho/TM.rho_0)**TM.a*(TM.T1 + 2.*TM.T2*V)
        #     T_z = 0.
        # else:
        #     if tau <= 0.77:
        #         P1 = 64.94*tau
        #     else:
        #         P1 = 217.38*tau - 117.38
        #     # pull out each setting derivative
        #     ia,_,iT1,iT2 = TM.idle_coefs(-z)
        #     Tidle_V = (rho/TM.rho_0)**ia*(iT1 + 2.*iT2*V)
        #     la,_,lT1,lT2 = TM.mil_coefs(-z)
        #     Tmil_V = (rho/TM.rho_0)**la*(lT1 + 2.*lT2*V)
        #     ma,_,mT1,mT2 = TM.max_coefs(-z)
        #     Tmax_V = (rho/TM.rho_0)**ma*(mT1 + 2.*mT2*V)
        #     # get full derivative
        #     if P1 < 50.:
        #         T_V = Tidle_V + (Tmil_V - Tidle_V)*P1/50.
        #     else:
        #         T_V = Tmil_V + (Tmax_V - Tmil_V)*(P1-50.)/50.

        # body-fixed force derivatives wrt state
        CFx0 = CL*Sa - CS*Ca*Sb - CD*Ca*Cb
        Fx_u = Q_u*CFx0 + Qdyn*(CL_u*Sa + CL*Ca*a_u - CS_u*Ca*Sb + \
            + CS*Sa*Sb*a_u - CS*Ca*Cb*b_u - CD_u*Ca*Cb + CD*Sa*Cb*a_u + \
            + CD*Ca*Sb*b_u) + T_V*u/V
        Fx_v = Q_v*CFx0 + Qdyn*(CL_v*Sa - CS_v*Ca*Sb + \
            - CS*Ca*Cb*b_v - CD_v*Ca*Cb + \
            + CD*Ca*Sb*b_v) + T_V*v/V
        Fx_w = Q_w*CFx0 + Qdyn*(CL_w*Sa + CL*Ca*a_w - CS_w*Ca*Sb + \
            + CS*Sa*Sb*a_w - CS*Ca*Cb*b_w - CD_w*Ca*Cb + CD*Sa*Cb*a_w + \
            + CD*Ca*Sb*b_w) + T_V*w/V
        #
        Fx_p = Qdyn*(CL_p*Sa - CS_p*Ca*Sb - CD_p*Ca*Cb)
        Fx_q = Qdyn*(CL_q*Sa - CS_q*Ca*Sb - CD_q*Ca*Cb)
        Fx_r = Qdyn*(CL_r*Sa - CS_r*Ca*Sb - CD_r*Ca*Cb)
        #
        Fx_z = Q_z*CFx0 + Qdyn*(CL_z*Sa - CS_z*Ca*Sb - CD_z*Ca*Cb) + T_z
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
        Fy_z = Q_z*CFy0 + Qdyn*(CS_z*Cb - CD_z*Sb)
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
        #
        Fz_z = Q_z*CFz0 + Qdyn*(- CL_z*Ca - CS_z*Sa*Sb - CD_z*Sa*Cb)

        # body-fixed moment derivatives wrt state
        Mx_u = Qlat_u*Cl + Qlat*Cl_u + Fy_u*Dzcg - Fz_u*Dycg
        Mx_v = Qlat_v*Cl + Qlat*Cl_v + Fy_v*Dzcg - Fz_v*Dycg
        Mx_w = Qlat_w*Cl + Qlat*Cl_w + Fy_w*Dzcg - Fz_w*Dycg
        #
        Mx_p = Qlat*Cl_p + Fy_p*Dzcg - Fz_p*Dycg
        Mx_q = Qlat*Cl_q + Fy_q*Dzcg - Fz_q*Dycg
        Mx_r = Qlat*Cl_r + Fy_r*Dzcg - Fz_r*Dycg
        #
        Mx_z = Qlat_z*Cl + Qlat*Cl_z + Fy_z*Dzcg - Fz_z*Dycg
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
        My_z = Qlon_z*Cm + Qlon*Cm_z + Fz_z*Dxcg - Fx_z*Dzcg
        #
        #
        Mz_u = Qlat_u*Cn + Qlat*Cn_u + Fx_u*Dycg - Fy_u*Dxcg
        Mz_v = Qlat_v*Cn + Qlat*Cn_v + Fx_v*Dycg - Fy_v*Dxcg
        Mz_w = Qlat_w*Cn + Qlat*Cn_w + Fx_w*Dycg - Fy_w*Dxcg
        #
        Mz_p = Qlat*Cn_p + Fx_p*Dycg - Fy_p*Dxcg
        Mz_q = Qlat*Cn_q + Fx_q*Dycg - Fy_q*Dxcg
        Mz_r = Qlat*Cn_r + Fx_r*Dycg - Fy_r*Dxcg
        #
        Mz_z = Qlat_z*Cn + Qlat*Cn_z + Fx_z*Dycg - Fy_z*Dxcg

        # evaluate at condition
        T = C.Prop.get_thrust(tau,-z,V)
        Fx = Qdyn*CFx0 + T
        Fy = Qdyn*CFy0
        Fz = Qdyn*CFz0

        # initialize jacobian
        A = np.zeros((x_eq.shape[0],x_eq.shape[0]))

        # set range values for ease of use
        rV = [0,1,2]
        rw = [3,4,5]
        rx = [6,7,8]
        if self.use_quats:
            rq = [9,10,11,12]
        else:
            re = [9,10,11]
        
        # assemble components
        A[0:3,rV] = [
            [Fx_u/m    , Fx_v/m + r, Fx_w/m - q],
            [Fy_u/m - r, Fy_v/m    , Fy_w/m + p],
            [Fz_u/m + q, Fz_v/m - p, Fz_w/m    ]
        ]
        #
        A[0:3,rw] = [
            [Fx_p/m    , Fx_q/m - w, Fx_r/m + v],
            [Fy_p/m + w, Fy_q/m    , Fy_r/m - u],
            [Fz_p/m - v, Fz_q/m + u, Fz_r/m    ]
        ]
        #
        if self.use_quats:
            A[0:3,rx] = [
                [0.0, 0.0, g_z/IM.W*Fx + g/IM.W*Fx_z + g_z*2.0*(exez - e0ey)],
                [0.0, 0.0, g_z/IM.W*Fy + g/IM.W*Fy_z + g_z*2.0*(eyez - e0ex)],
                [0.0, 0.0, g_z/IM.W*Fz + g/IM.W*Fz_z + g_z*(ez2 + e02 - ex2 - ey2)]
            ]
            #
            A[0:3,rq] = 2.*g*np.array([
                [-ey,  ez, -e0, ex],
                [ ex,  e0,  ez, ey],
                [ e0, -ex, -ey, ez]
            ])
        else:
            A[0:3,rx] = [
                [0.0, 0.0, g_z/IM.W*Fx + g/IM.W*Fx_z - g_z*St],
                [0.0, 0.0, g_z/IM.W*Fy + g/IM.W*Fy_z + g_z*Sp*Ct],
                [0.0, 0.0, g_z/IM.W*Fz + g/IM.W*Fz_z + g_z*Cp*Ct]
            ]
            #
            A[0:3,re] = g*np.array([
                [    0.,    -Ct, 0.],
                [ Cp*Ct, -Sp*St, 0.],
                [-Sp*Ct, -Cp*St, 0.]
            ])
        #
        #
        A[3:6,rV] = np.matmul(Iinv,np.array([
            [Mx_u, Mx_v, Mx_w],
            [My_u, My_v, My_w],
            [Mz_u, Mz_v, Mz_w]
        ]))
        #
        A[3:6,rw] = np.matmul(Iinv,(np.array([
            [Mx_p, Mx_q, Mx_r],
            [My_p, My_q, My_r],
            [Mz_p, Mz_q, Mz_r]
        ]) + np.array([
            [ 0., -hz,  hy],
            [ hz,  0., -hx],
            [-hy,  hx,  0.]
        ]) + np.array([
            [Ixz*q - Ixy*r, (Iyy - Izz)*r + 2.*Iyz*q + Ixz*p, 
                                            (Iyy - Izz)*q - 2.*Iyz*r - Ixy*p],
            [(Izz - Ixx)*r - 2.*Ixz*p - Iyz*q, Ixy*r - Iyz*p, 
                                            (Izz - Ixx)*p + 2.*Ixz*r + Ixy*q],
            [(Ixx - Iyy)*q + 2.*Ixy*p + Iyz*r, (Ixx - Iyy)*p - 2.*Ixy*q -Ixz*r,
                                            Iyz*p - Ixz*q]
        ])))
        #
        A[3:6,rx] = np.matmul(Iinv,np.array([
            [0.0, 0.0, Mx_z],
            [0.0, 0.0, My_z],
            [0.0, 0.0, Mz_z]
        ]))
        #
        if self.use_quats:
            A[6:9,rV] = [
                [e02 + ex2 - ey2 - ez2, -2.*e0ez + 2.*exey, 2.*e0ey + 2.*exez],
                [2.*e0ez + 2.*exey, e02 - ex2 + ey2 - ez2, -2.*e0ex + 2.*eyez],
                [-2.*e0ey + 2.*exez, 2.*e0ex + 2.*eyez, e02 - ex2 - ey2 + ez2]
            ]
            A[6:9,rq] = 2. * np.array([ # I know it's ugly, but it works!
                [e0upeywmezv,  exupeyvpezw,  e0wpexvmeyu, -e0vmexwpezu],
                [e0vmexwpezu, -e0wpexvmeyu,  exupeyvpezw,  e0upeywmezv],
                [e0wpexvmeyu,  e0vmexwpezu, -e0upeywmezv,  exupeyvpezw]
            ])
        else:
            A[6:9,rV] = [
                [Ct*Cs, Sp*St*Cs - Cp*Ss, Cp*St*Cs + Sp*Ss],
                [Ct*Ss, Sp*St*Ss + Cp*Cs, Cp*St*Ss - Sp*Cs],
                [  -St,            Sp*Ct,            Cp*Ct]
            ]
            #
            Vy = Cp*v - Sp*w
            Vz = Sp*v + Cp*w
            A[6:9,re] = [
                [Vy*St*Cs +Vz*Ss, Vz*Ct*Cs -St*Cs*u, -Vy*Cs -(Vz*St +Ct*u)*Ss],
                [Vy*St*Ss -Vz*Cs, Vz*Ct*Ss -St*Ss*u, -Vy*Ss +(Vz*St +Ct*u)*Cs],
                [          Vy*Ct,      -Ct*u -Vz*St,                       0.]
            ]
        #
        if self.use_quats:
            A[9:13,rw] = 0.5 * np.array([
                [-ex, -ey, -ez],
                [ e0, -ez,  ey],
                [ ez,  e0, -ex],
                [-ey,  ex,  e0]
            ])
            A[9:13,rq] = 0.5 * np.array([
                [0., -p, -q, -r],
                [ p, 0.,  r, -q],
                [ q, -r, 0.,  p],
                [ r,  q, -p, 0.]
            ])
        else:
            A[9:12,rw] = [
                [1., Sp*St/Ct, Cp*St/Ct],
                [0.,       Cp,      -Sp],
                [0.,    Sp/Ct,    Cp/Ct]
            ]
            A[9:12,re] = [
                [Cp*St/Ct*q - Sp*St/Ct*r,    (Sp*q + Cp*r)/Ct**2., 0.],
                [           -Sp*q - Cp*r,                      0., 0.],
                [       (Cp*q - Sp*r)/Ct, (Sp*q + Cp*r)*St/Ct**2., 0.],
            ]
        
        return A
        

    def _build_input_jacobian(self, x_eq, u_eq, cg_shift = [0.,0.,0.]):
        # report
        if self.report:
            print("    building input jacobian...")

        # pull out evaluating condition
        if self.use_quats: u,v,w,p,q,r,x,y,z, e0, ex, ey,ez = x_eq[0:13]
        else:              u,v,w,p,q,r,x,y,z,phi,tht,psi    = x_eq[0:12]
        if self.is_bire: da, de, dB, tau = u_eq
        else:            da, de, dr, tau = u_eq
        Dxcg, Dycg, Dzcg = cg_shift
        C = self.aero_model
        IM = self.inertia_model

        # values for latter use
        a = atan2(w,u)
        V = (u*u + v*v + w*w)**0.5
        b = asin(v/V)
        #
        Ca = cos(a); Sa = sin(a)
        Cb = cos(b); Sb = sin(b)
        #
        Rlon = C.c_w/2./V
        Rlat = C.b_w/2./V
        #
        pbar = p*Rlat
        qbar = q*Rlon
        rbar = r*Rlat
        #
        _,g,_,_, rho,sos = stdatm_english( -z)
        _,_,_,_,rho0,_   = stdatm_english(0.0)
        # #####################
        # g = 32.12780074195162
        # print("g =",g)
        # #####################
        #
        M = V / sos
        #
        Qdyn = 0.5*rho*V**2.*C.S_w
        Qlon = Qdyn*C.c_w
        Qlat = Qdyn*C.b_w
        #
        m = IM.W/g
        minv = g/IM.W
        if self.is_bire:
            Iinv = IM.inverse_tensor(dB)
            dIinv = IM.inverse_tensor_derivative(dB)
            [ Ixx, Iyy, Izz, Ixy, Ixz, Iyz] = IM.inertia_results(dB)
            [dIxx,dIyy,dIzz,dIxy,dIxz,dIyz] = IM.inertia_derivative_results(dB)
            hx, hy, hz = IM.angular_momentum_results()
        else:
            Iinv = IM.inverse_tensor(0.)
        
        # input aerodynamic force derivatives
        if self.is_bire:
            # evaluate derivatives wrt bire angle
            C.evaluate_derivatives(dB)
            # for use
            CL1 = C.CL0 + C.CLa * a
            CS1 = C.CS0 + C.CSb * b
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
        else:
            # for use
            CL1 = C.CL0 + C.CLa * a
            CS1 = C.CSb * b
            # drag
            oCD_da = C.CDSda*CS1
            oCD_de = C.CDLde*CL1 + C.CDde + 2.*C.CDde2*de
            oCD_dr = C.CDSdr*CS1
            # equated values
            oCL_de, oCS_da, oCS_dr = C.CLde, C.CSda, C.CSdr
            # zero values
            oCL_da = oCL_dr = oCS_de = 0.0
        
        # input aerodynamic moment derivatives
        if self.is_bire:
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
        else:
            # yaw
            oCn_da = C.CnLda*CL1 + C.Cnda
            # equated values
            oCl_da, oCl_dr, oCm_de, oCn_dr = C.Clda, C.Cldr, C.Cmde, C.Cndr
            # zero values
            oCl_de = oCm_da = oCm_dr = oCn_de = 0.0

        # get forces and moments at the specified condition
        if self.is_bire:
            params = [a, b, pbar, qbar, rbar, da, de, dB]
        else:
            params = [a, b, pbar, qbar, rbar, da, de, dr]
        [CL, CS, CD, Cl, Cm, Cn] = C.aero_results(*params,M=M,**self.aero_dict)

        # Stall corrections
        if self.include_stall and self.enforce_stall: # False: # 

            # pull out stall model coeff
            _,_,_,S,_,_,_,_ = \
                C._stall_correction_derivatives(a,0.,0.,0.)
            
            # implement
            # lift
            aCL_da = (1. - S)*oCL_da
            aCL_de = (1. - S)*oCL_de
            # drag
            aCD_da = (1. - S)*oCD_da
            aCD_de = (1. - S)*oCD_de
            # pitch
            aCm_da = (1. - S)*oCm_da
            aCm_de = (1. - S)*oCm_de
            # bire vs rudder
            if self.is_bire:
                aCL_dB = (1. - S)*oCL_dB
                aCD_dB = (1. - S)*oCD_dB
                aCm_dB = (1. - S)*oCm_dB
            else:
                aCL_dr = (1. - S)*oCL_dr
                aCD_dr = (1. - S)*oCD_dr
                aCm_dr = (1. - S)*oCm_dr

            # coefficients which are not in stall model
            aCS_da,aCS_de = oCS_da,oCS_de
            aCl_da,aCl_de = oCl_da,oCl_de
            aCn_da,aCn_de = oCn_da,oCn_de
            # bire vs rudder
            if self.is_bire:
                aCS_dB,aCl_dB,aCn_dB = oCS_dB,oCl_dB,oCn_dB
            else:
                aCS_dr,aCl_dr,aCn_dr = oCS_dr,oCl_dr,oCn_dr
        else:
            aCL_da,aCL_de = oCL_da,oCL_de
            aCS_da,aCS_de = oCS_da,oCS_de
            aCD_da,aCD_de = oCD_da,oCD_de
            aCl_da,aCl_de = oCl_da,oCl_de
            aCm_da,aCm_de = oCm_da,oCm_de
            aCn_da,aCn_de = oCn_da,oCn_de
            # bire vs rudder
            if self.is_bire:
                aCL_dB,aCS_dB,aCD_dB = oCL_dB,oCS_dB,oCD_dB
                aCl_dB,aCm_dB,aCn_dB = oCl_dB,oCm_dB,oCn_dB
            else:
                aCL_dr,aCS_dr,aCD_dr = oCL_dr,oCS_dr,oCD_dr
                aCl_dr,aCm_dr,aCn_dr = oCl_dr,oCm_dr,oCn_dr
        
        # Compressibility corrections
        if self.compressible:
            # incompressible coefficients
            [aCL, aCS, aCD, aCl, aCm, aCn] = \
                C.aero_results(*params,M=M,**{
                "compressible" : False,
                "use_Anderson" : False,
                "enforce_stall" : self.enforce_stall
            })

            # Mach correction derivatives
            if M <= 1.0: # subsonic
                if self.use_Anderson:
                    L_w, L_h, L_v = C.Lam_w, C.Lam_h, C.Lam_v
                    R_w, R_h, R_v = C.RA_w, C.RA_h, C.RA_v

                    # derivatives wrt incompressible coefficients
                    CL_aCL = Anderson_correction_der_coeff(aCL,L_w,R_w,M)
                    Cm_aCm = Anderson_correction_der_coeff(aCm,L_w,R_w,M)
                    if self.is_bire:
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
            if self.is_bire:
                CL_dB = CL_aCL*aCL_dB
                CS_dB = CS_aCS*aCS_dB
                CD_dB = aCD_dB
                Cl_dB = Cl_aCl*aCl_dB
                Cm_dB = Cm_aCm*aCm_dB
                Cn_dB = Cn_aCn*aCn_dB
            else:
                CL_dr = CL_aCL*aCL_dr
                CS_dr = CS_aCS*aCS_dr
                CD_dr = aCD_dr
                Cl_dr = Cl_aCl*aCl_dr
                Cm_dr = Cm_aCm*aCm_dr
                Cn_dr = Cn_aCn*aCn_dr
        else:
            # no compressibility
            CL_da,CL_de = aCL_da,aCL_de
            CS_da,CS_de = aCS_da,aCS_de
            CD_da,CD_de = aCD_da,aCD_de
            Cl_da,Cl_de = aCl_da,aCl_de
            Cm_da,Cm_de = aCm_da,aCm_de
            Cn_da,Cn_de = aCn_da,aCn_de
            # bire vs rudder
            if self.is_bire:
                CL_dB,CS_dB,CD_dB = aCL_dB,aCS_dB,aCD_dB
                Cl_dB,Cm_dB,Cn_dB = aCl_dB,aCm_dB,aCn_dB
            else:
                CL_dr,CS_dr,CD_dr = aCL_dr,aCS_dr,aCD_dr
                Cl_dr,Cm_dr,Cn_dr = aCl_dr,aCm_dr,aCn_dr
            
        # thrust state derivatives
        T_tau = C.Prop.T_der_tau(tau,-z,V)

        # body-fixed force derivatives wrt input
        Fx_da = Qdyn*(CL_da*Sa - CS_da*Ca*Sb - CD_da*Ca*Cb)
        Fx_de = Qdyn*(CL_de*Sa - CS_de*Ca*Sb - CD_de*Ca*Cb)
        if self.is_bire:
            Fx_dB = Qdyn*(CL_dB*Sa - CS_dB*Ca*Sb - CD_dB*Ca*Cb)
        else:
            Fx_dr = Qdyn*(CL_dr*Sa - CS_dr*Ca*Sb - CD_dr*Ca*Cb)
        Fx_tau = T_tau
        #
        Fy_da = Qdyn*(CS_da*Cb - CD_da*Sb)
        Fy_de = Qdyn*(CS_de*Cb - CD_de*Sb)
        if self.is_bire:
            Fy_dB = Qdyn*(CS_dB*Cb - CD_dB*Sb)
        else:
            Fy_dr = Qdyn*(CS_dr*Cb - CD_dr*Sb)
        #
        Fz_da = Qdyn*(- CL_da*Ca - CS_da*Sa*Sb - CD_da*Sa*Cb)
        Fz_de = Qdyn*(- CL_de*Ca - CS_de*Sa*Sb - CD_de*Sa*Cb)
        if self.is_bire:
            Fz_dB = Qdyn*(- CL_dB*Ca - CS_dB*Sa*Sb - CD_dB*Sa*Cb)
        else:
            Fz_dr = Qdyn*(- CL_dr*Ca - CS_dr*Sa*Sb - CD_dr*Sa*Cb)
        
        # body-fixed moment derivatives wrt input
        Mx_da = Qlat*Cl_da + Fy_da*Dzcg - Fz_da*Dycg
        Mx_de = Qlat*Cl_de + Fy_de*Dzcg - Fz_de*Dycg
        if self.is_bire:
            Mx_dB = Qlat*Cl_dB + Fy_dB*Dzcg - Fz_dB*Dycg
        else:
            Mx_dr = Qlat*Cl_dr + Fy_dr*Dzcg - Fz_dr*Dycg
        #
        My_da = Qlon*Cm_da + Fz_da*Dxcg - Fx_da*Dzcg
        My_de = Qlon*Cm_de + Fz_de*Dxcg - Fx_de*Dzcg
        if self.is_bire:
            My_dB = Qlon*Cm_dB + Fz_dB*Dxcg - Fx_dB*Dzcg
        else:
            My_dr = Qlon*Cm_dr + Fz_dr*Dxcg - Fx_dr*Dzcg
        My_tau = - Fx_tau*Dzcg
        #
        Mz_da = Qlat*Cn_da + Fx_da*Dycg - Fy_da*Dxcg
        Mz_de = Qlat*Cn_de + Fx_de*Dycg - Fy_de*Dxcg
        if self.is_bire:
            Mz_dB = Qlat*Cn_dB + Fx_dB*Dycg - Fy_dB*Dxcg
        else:
            Mz_dr = Qlat*Cn_dr + Fx_dr*Dycg - Fy_dr*Dxcg
        Mz_tau = Fx_tau*Dycg

        # evaluate at condtion for Mx, My, Mz
        T = C.Prop.get_thrust(tau,-z,V)
        #
        Fx = Qdyn*(CL*Sa - CS*Ca*Sb - CD*Ca*Cb) + T
        Fy = Qdyn*(CS*Cb - CD*Sb)
        Fz = Qdyn*(- CL*Ca - CS*Sa*Sb - CD*Sa*Cb)
        #
        Mx = Qlat*Cl + Fy*Dzcg - Fz*Dycg
        My = Qlon*Cm + Fz*Dxcg - Fx*Dzcg
        Mz = Qlat*Cn + Fx*Dycg - Fy*Dxcg

        # initialize jacobian
        B = np.zeros((x_eq.shape[0],u_eq.shape[0]))

        # set range values for ease of use
        rV = [0,1,2]
        rw = [3,4,5]
        ru = [0,1,3]
        r3 = 2
        
        # assemble components
        B[0:3,ru] = minv*np.array([
            [Fx_da, Fx_de, Fx_tau],
            [Fy_da, Fy_de,     0.],
            [Fz_da, Fz_de,     0.]
        ])
        B[3:6,ru] = np.matmul(Iinv,np.array([
            [Mx_da, Mx_de,     0.],
            [My_da, My_de, My_tau],
            [Mz_da, Mz_de, Mz_tau]
        ]))
        if self.is_bire:
            B[0:3,r3] = np.array([Fx_dB, Fy_dB, Fz_dB])/m
            wdot = (
                np.array([Mx,My,Mz]) +
                np.matmul(np.array([
                [ 0., -hz,  hy],
                [ hz,  0., -hx],
                [-hy,  hx,  0.]
                ]), np.array([p,q,r])) + 
                np.array([
                    ( Iyy- Izz)*q*r +  Iyz*(q**2.-r**2.) +  Ixz*p*q -  Ixy*p*r,
                    ( Izz- Ixx)*p*r +  Ixz*(r**2.-p**2.) +  Ixy*q*r -  Iyz*p*q,
                    ( Ixx- Iyy)*p*q +  Ixy*(p**2.-q**2.) +  Iyz*p*r -  Ixz*q*r
                ])
            )
            wdot_dB = (
                np.array([Mx_dB,My_dB,Mz_dB]) +
                np.array([
                    (dIyy-dIzz)*q*r + dIyz*(q**2.-r**2.) + dIxz*p*q - dIxy*p*r,
                    (dIzz-dIxx)*p*r + dIxz*(r**2.-p**2.) + dIxy*q*r - dIyz*p*q,
                    (dIxx-dIyy)*p*q + dIxy*(p**2.-q**2.) + dIyz*p*r - dIxz*q*r
                ])
            )
            B[3:6,r3] = ( 
            np.matmul(Iinv,wdot_dB) + 
            np.matmul(dIinv,wdot) ) # 
        else:
            B[0:3,r3] = np.array([Fx_dr, Fy_dr, Fz_dr])/m
            B[3:6,r3] = np.matmul(Iinv,np.array([
                Mx_dr, My_dr, Mz_dr
            ])[:,np.newaxis])[:,0]

        return B


    def _calculate_jacobian(self,function,input,step_size=0.001):
            
        # call function to check size
        dfdx = function(input)

        J = np.zeros((dfdx.shape[0],input.shape[0]))

        # develop Jacobian
        for i in range(J.shape[1]):
            # determine forces with each step change
            base = np.array(input) * 1.0
            # plus
            base[i] += step_size
            input = base * 1.0
            fun_ip1 = function(input)
            # minus
            base[i] -= 2. * step_size
            input = base * 1.0
            fun_im1 = function(input)
            # reset
            base[i] += step_size
            input = base * 1.0

            # assign to jacobian
            J[:,i] = (fun_ip1 - fun_im1) / 2. / step_size

        return J


    def _calculate_complex_jacobian(self,function,input,step_size=0.001):
            
        # call function to check size
        dfdx = function(input)

        J = np.zeros((dfdx.shape[0],input.shape[0]))

        # develop Jacobian
        for i in range(J.shape[1]):
            # determine forces with each step change
            base = np.array(input,dtype=complex)*1.0
            # plus
            base[i] += complex(0.0,step_size)
            input = base * 1.0

            # assign to jacobian
            J[:,i] = np.imag(function(input)) / step_size

        return J


    def build_jacobians(self, x_eq, u_eq, cg_shift = [0.,0.,0.], 
        numerical = False, numerical_dynamics = None):

        # build state and input
        if numerical:
            if numerical_dynamics == None:
                raise TypeError("Dynamics function required")
            if False: # use_complex_jacobian: # 
                jacob = self._calculate_complex_jacobian
            else:
                jacob = self._calculate_jacobian
            dyn_fun = lambda x,u : numerical_dynamics(0.0,x,True,True,u)
            A = jacob(lambda x:dyn_fun(x,u_eq),x_eq)
            dyn_fun = lambda x,u : numerical_dynamics(0.0,x,True,True,u,True)
            B = jacob(lambda u:dyn_fun(x_eq,u),u_eq)#,step_size=0.0001)
            # print(A)
            # print(B)
            # quit()
            # pass
        else:
            A = self._build_state_jacobian(x_eq,u_eq,cg_shift)
            B = self._build_input_jacobian(x_eq,u_eq,cg_shift)

        if not self.drop_actrs:
            # append actuator dynamics
            if self.order == 1:
                # get throttle
                s_tau = self._throttle_gain(u_eq[3])
                # define actuation parameters
                if self.is_bire: s_d3 = self.s_dB
                else:            s_d3 = self.s_dr
                S1 = np.diag([self.s_da,self.s_de,s_d3,     s_tau])
                
                # build new matrices
                A[:-4,-4:] = B[:-4]*1.
                A[-4:,-4:] = -S1
                B = B*0.
                B[-4:] = S1
            elif self.order == 2:
                # define actuation parameters
                if self.is_bire:
                    w_d3 = self.w_dB
                    z_d3 = self.z_dB
                else:
                    w_d3 = self.w_dr
                    z_d3 = self.z_dr
                Wn = np.diag([self.w_da,self.w_de,w_d3,self.w_tau])
                Zt = np.diag([self.z_da,self.z_de,z_d3,self.z_tau])
                Wn2 = np.matmul(Wn,Wn)
                
                # add actuation to LQR matrices
                I = np.eye(u_eq.shape[0])
                Z = np.zeros((u_eq.shape[0],u_eq.shape[0]))
                block = np.block([
                    [Z,I],
                    [-Wn2,-2.*np.matmul(Zt,Wn)]
                ])

                # build new matrices #### THIS NEEDS TO BE FIXED
                A[-8:,-8:] = block        ### <<<<<<
                B[-8:] = np.vstack(Z,Wn2) ### <<<<<<
            else: # order 0
                pass
            
        return A,B
            

    def _analyze_controllability(self,A,B,is_minimal=True):
        # run controllability
        Gamma = co.ctrb(A,B)

        Gamma_rank = np.linalg.matrix_rank(Gamma)

        # get A eigenvalues
        if is_minimal and len(self.xIi):
            In = len(self.xIi)
            Pn = A.shape[0]
            M = np.block([[np.zeros((In,In)),np.eye(In)],[np.zeros((Pn,In)),-A]])
        else:
            M = A
        eigs,evecs = np.linalg.eig(M)

        if not self.turn_off_warnings and is_minimal and Gamma_rank != A.shape[0]:
            raise ValueError("Controllability matrix is rank " + 
                "{} < A size {}. System is Uncontrollable.".format(
                Gamma_rank,A.shape[0]))

        return Gamma, Gamma_rank, eigs, evecs


    def _analyze_observability(self,A,C,is_minimal=True):
        # run controllability
        O = co.obsv(A,C)

        O_rank = np.linalg.matrix_rank(O)

        if not self.turn_off_warnings and is_minimal and O_rank != A.shape[0]:
            raise ValueError("Observability matrix is rank " + 
                "{} < A size {}. System is Unobserveable.".format(
                O_rank,A.shape[0]))

        return O, O_rank


    def _build_feedback(self,A,B,Q,R,KI):
        # build feedback
        # try:
        if self.controller_type == "LQR":
            # check observability
            _,_ = self._analyze_observability(A,np.sqrt(Q),is_minimal=True)
            K,P,_ = co.lqr(A,B,Q,R,method="scipy")
        elif self.controller_type == "gains":
            K = self.K_in*1.
            P = 1.
        
        if len(self.xIi):
            In = len(self.xIi)
            Pn = A.shape[0]
            Acl = np.block([[np.zeros((In,In)),np.eye(Pn)],
                [np.matmul(B,KI), A + np.matmul(B,K)]])
        else:
            Acl = A - np.matmul(B,K)
        eig,evec = np.linalg.eig(Acl)
        # except SlycotArithmeticError:
        #     print("    lqr failed due to SlycotArithmeticError...")
        #     K,P,eig,evec = None,None,None,None
        
        # check eigenvalues are in LHP
        if np.max(np.real(eig)) >= 0. and not self.turn_off_warnings:
            wrn.warn("Eigenvalues are not all less than zero:"+str(eig))
        return K,P,Acl,eig,evec


    def _analyze_sensitivity(self,case_name):

        # create 's'
        snum = 100
        s = np.logspace(-5.0,5.0,num=snum)
        # iep2 = int(0.67*snum)
        iep2 = int(1.00*snum)
        # identity
        I = np.eye(self.A.shape[0])

        # plotting variables
        # input
        if self.is_bire:
            ins = [r"$\delta_a$",r"$\delta_e^B$",r"$\delta_B$",r"$\tau$"]
        else:
            ins = [r"$\delta_a$",r"$\delta_e$",r"$\delta_r$",r"$\tau$"]
        # outputs
        ous = [
            r"$V_{x_b}$",r"$V_{y_b}$",r"$V_{z_b}$",
            r"$p$",r"$q$",r"$r$",
            r"$x_f$",r"$y_f$",r"$z_f$",
            r"$\phi$",r"$\theta$",r"$\psi$"
            ]
        if self.order > 0:
            ous = ous + [
                r"$\dot{\delta}_f$",r"$\dot{\delta}_s^B$",
                r"$\dot{\delta}_B$",r"$\dot{\tau}$"
            ]
        if self.order > 1:
            ous = ous + [
                r"$\ddot{\delta}_f$",r"$\ddot{\delta}_s^B$",
                r"$\ddot{\delta}_B$",r"$\ddot{\tau}$"
            ]
        if len(self.xIi):
            ous = ous + ["${"+ous[self.xPi[I]][1:-1]+"}_I$" \
                for I in range(len(self.xPi))]
        # apply C
        ous = [ous[j] for j in self.Hslice]
        ins = [ins[j] for j in self.Cuslice]
        H = self.H*1.
        # H[1,8] = 0.001 # check to see if it should be the actual values...
        K  = np.matmul(self.K,np.matmul(H,self.C.T).T) # I believe this is correct...
        if len(self.xIi):
            KI = np.matmul(self.KI,np.matmul(H,self.C.T).T)
        
        # plot one array
        plt1  = np.zeros((self.B_min.shape[1],snum))
        plt2  = np.zeros((self.H.shape[0],snum))
        Lrows = np.zeros((self.H.shape[0],snum))
        Lcols = np.zeros((self.H.shape[0],snum))
        
        # build transfer function
        for i,sval in enumerate(s):
            sImAinv = np.linalg.solve(sval*I - self.A,I)
            G = np.matmul(H,np.matmul(sImAinv,np.matmul(self.B,self.C_u.T)))
            Li = np.matmul(K,G)
            Lo = np.matmul(G,K)

            Ui,Si,Vhi = np.linalg.svd(Li)
            Uo,So,Vho = np.linalg.svd(Lo)
            plt1[:,i] = Si
            plt2[:,i] = So

            # get norm of rows and cols of Lo
            Lrows[:,i] = np.linalg.norm(Lo,axis=1)
            Lcols[:,i] = np.linalg.norm(Lo,axis=0)
        # print(Ui)
        # for j in range(len(Uo[0])):
        #     print(Uo[:,j])
        
        # plot setup
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

        cs = ["k","0.5"]
        ls = ["-","--","-."]

        in_labels = []
        ou_labels = []

        # determine labels
        # in
        for j in range(Ui.shape[1]):
            sublabel = ""
            col = Ui[:,j]
            powers = np.floor(np.log10(np.abs(col)))
            mags_inds = np.argsort(np.abs(col))
            max_power = np.max(powers)
            for k in mags_inds:
                if powers[k] == max_power:
                    if sublabel != "":
                        end = ", "
                    else:
                        end = ""
                    sublabel += end + ins[k]
            in_labels.append(sublabel)
        # out
        for j in range(Uo.shape[1]):
            sublabel = ""
            col = Uo[:,j]
            powers = np.floor(np.log10(np.abs(col)))
            mags_inds = np.argsort(np.abs(col))
            max_power = np.max(powers)
            for k in mags_inds:
                if powers[k] == max_power:
                    if sublabel != "":
                        end = ", "
                    else:
                        end = ""
                    sublabel += end + ous[k]
            ou_labels.append(sublabel)
        
        # plotting
        pt = "pdf" # "png" # 
        folder = self.freq_folder + "sensitivity/"
        transparent = False
        savedict = dict(transparent=transparent,format=pt,dpi=300.0)
        figdict = dict(figsize=(3.25,3.5),constrained_layout=True)
        ins_fig, ins_ax = plt.subplots(**figdict)
        out_fig, out_ax = plt.subplots(**figdict)
        ouc_fig, ouc_ax = plt.subplots(**figdict)
        our_fig, our_ax = plt.subplots(**figdict)

        # build new folder / delete old
        if not path_exists(folder):
        #     # step through and remove every file, then delete folder
        #     # folder
        #     for filename in listdir(folder):
        #         remove(folder + "/" + filename)
        #     # delete folder
        # else:
            mkdir(folder)
        
        if case_name != "":
            case_name += "_"
        
        # input loop transfer
        for i in range(plt1.shape[0]):
            c_i = cs[i % len(cs)]
            l_i = ls[(i // len(cs)) % len(ls)]
            ins_ax.plot(s[:iep2],10.*np.log10(plt1[i,:iep2]),c=c_i,ls=l_i,
                label=in_labels[i])
        ins_ax.grid(which="major",lw=0.6,ls="-",c="0.75")
        ins_ax.set_xscale("log")
        ins_ax.set_xlim((s[0],s[iep2-1]))
        ins_ax.set_xlabel("Frequency, rad/s")
        ins_ax.set_ylabel("Singular Values, dB")
        ins_ax.legend()
        ins_fig.savefig(folder +case_name+ "input_singular_values."+pt,
            **savedict)
        # ins_ax.cla()
        
        # output loop transfer
        for i in range(plt2.shape[0]):
            c_i = cs[i % len(cs)]
            l_i = ls[(i // len(cs)) % len(ls)]
            out_ax.plot(s,10.*np.log10(plt2[i,:]),c=c_i,ls=l_i,label=ou_labels[i])
        out_ax.grid(which="major",lw=0.6,ls="-",c="0.75")
        out_ax.set_xscale("log")
        out_ax.set_xlim((s[0],s[-1]))
        out_ax.set_xlabel("Frequency, rad/s")
        out_ax.set_ylabel("Singular Values, dB")
        out_ax.legend()
        out_fig.savefig(folder +case_name+ "output_singular_values."+pt,**savedict)
        # out_ax.cla()

        # output loop transfer channels columns
        for i in range(plt2.shape[0]):
            c_i = cs[i % len(cs)]
            l_i = ls[(i // len(cs)) % len(ls)]
            ouc_ax.plot(s,10.*np.log10(Lcols[i,:]),c=c_i,ls=l_i,label=ous[i])
        ouc_ax.grid(which="major",lw=0.6,ls="-",c="0.75")
        ouc_ax.set_xscale("log")
        ouc_ax.set_xlim((s[0],s[-1]))
        ouc_ax.set_xlabel("Frequency, rad/s")
        ouc_ax.set_ylabel("Singular Values, dB")
        ouc_ax.legend()
        ouc_fig.savefig(folder +case_name+ "output_rowerr_channels."+pt,**savedict)
        # ouc_ax.cla()

        # output loop transfer channels rows
        for i in range(plt2.shape[0]):
            c_i = cs[i % len(cs)]
            l_i = ls[(i // len(cs)) % len(ls)]
            our_ax.plot(s,10.*np.log10(Lrows[i,:]),c=c_i,ls=l_i,label=ous[i])
        our_ax.grid(which="major",lw=0.6,ls="-",c="0.75")
        our_ax.set_xscale("log")
        our_ax.set_xlim((s[0],s[-1]))
        our_ax.set_xlabel("Frequency, rad/s")
        our_ax.set_ylabel("Singular Values, dB")
        our_ax.legend()
        our_fig.savefig(folder +case_name+ "output_colerr_channels."+pt,**savedict)
        # our_ax.cla()
        plt.close("all")


def Anderson_correction_der_coeff(coeff, Lambda, RA, M):
    CL = cos(Lambda)
    set0 = 1. - M**2.*CL**2. + (coeff*CL/pi/RA)**2.
    num = CL*( pi*RA*set0**0.5 - coeff*CL)
    denom = coeff*CL*set0**0.5 + pi*RA*set0
    return num/denom

def Anderson_correction_der_M(coeff, Lambda, RA, M):
    CL = cos(Lambda)
    set0 = 1. - M**2.*CL**2. + (coeff*CL/pi/RA)**2.
    num = coeff*CL
    denom = coeff*CL/pi/RA*set0**0.5 + set0
    return num/denom



def report_latex(M, name="M", predecimals=4, decimals=4, diag=False,
    align=False, endln=False):

    if align:
        char = "&"
    else:
        char = ""
    if endln:
        end = "\\\\"
    else:
        end = ""
    if diag:
        bef = "\\operatorname{diag} \\left( "
        aft = " \\right)"
    else:
        bef = ""
        aft = ""
    
    t = "    "

    # print name
    print("{}{} {}= {}\\begin{{bmatrix}}".format(t,name,char,bef))

    # print matrix
    for i in range(M.shape[0]):
        print("{}{}".format(t,t),end="")
        for j in range(M[0].shape[0]):
            print("{:> {}.{}f}".format(M[i,j],predecimals+decimals+1,\
                decimals),end="")
            if j == M[0].shape[0] - 1:
                print(" \\\\")
            else:
                print(" & ",end="")
    print("{}\\end{{bmatrix}}{} {}".format(t,aft,end))
    if not endln:
        print()    

def rep2D(array, name = "ans", predecimals = 5, decimals = 4,print_format="f"):

    printname = "{} = ".format(name)
    lenname = len(printname)
    for i in range(array.shape[0]):
        if i == 0:
            print(printname,end="")
        else:
            print(" "*lenname,end="")
        
        if print_format == "e":
            num = 4
        else:
            num = 0
        
        for j in range(array.shape[1]):
            print("{:> {}.{}{}}".format(array[i,j],decimals+predecimals+num,\
                decimals,print_format),end="")
            if j != array.shape[1]-1:
                print(",",end="")
        print()
    print()



if __name__ == "__main__":
    np.set_printoptions(precision=16)
    # test linearization
    x = np.ones((12,))
    u = np.ones((4,)) / 10.
    cg = np.ones((3,))

    # Christian trim result for the Baseline : 20 deg bank, 15000 ft, sct, M=0.6
    x = np.array([
        633.44608380577915340837,
        3.31652081762194361758,
        34.86135820251543293580,
        -0.00098324452597210884,
        0.00628505820338688334,
        0.01726805549376349974,
        0.00000000000000000000,
        0.00000000000000000000,
        -15_000.000_000_000_000_000_000_00,
        0.34906585039886589561,
        0.05345520291649863420,
        0.00000000000000000000
    ])
    u = np.array([
        0.09413859573040345152,
        -0.03506196601240764432,
        -0.00899426274124131252,
         0.28068662443453590294
    ])

    # dict of other vals
    inputs = {
        "compressible" : True,
        "use_Anderson" : True,
        "enforce_stall" : True,
        "actuators_properties" : {
            "order" : 0
        }
    }

    # BASE_euler = linearization(x,u,cg,use_quaternion=False,is_bire=False,**inputs)
    # rep2D(BASE_euler.A[:,:],"  A  ",decimals=16)
    # rng = [0,1,2,3]
    # rep2D(BASE_euler.B[:,rng],"  B  ",decimals=16)

    # Christian trim result for the BIRE : 20 deg bank, 15000 ft, sct, M=0.6
    x = np.array([
        633.47460914603436776815,
         0.88226106112795721348,
        34.48761919348824989129,
        -0.00094904556134772181,
         0.00628628244500431706,
         0.01727141906996335768,
         0.00000000000000000000,
         0.00000000000000000000,
         -15_000.000_000_000_000_000_000_00,
         0.34906585039886589561,
         0.05158926408385627188,
         0.00000000000000000000
    ])
    u = np.array([
        0.07970154685909836001,
        -0.03323488524681716960,
        -0.17293856925365588828,
         0.27818160272351094564
    ])

    BIRE_euler = linearization(x,u,cg,use_quaternion=False,is_bire=True,**inputs)
    rep2D(BIRE_euler.A[:,:],"  A  ",decimals=4) # 16)
    rng = [0,1,2,3] # [2] # 
    rep2D(BIRE_euler.B[:,rng],"  B  ",decimals=4) # 16)
    quit()

    x = np.append(x,1.0)

    BASE_quat = linearization(x,u,cg,use_quaternion=True,is_bire=False,**inputs)
    BIRE_quat = linearization(x,u,cg,use_quaternion=True,is_bire=True,**inputs)


    # cases for Troy
    x_troy = [
        633.1809060552918, 
        0.0, 
        32.21707943001307, 
        0.0, 
        0.0, 
        0.0, 
        0.0, 
        0.0, 
        -15000., 
        0.0, 
        np.deg2rad(2.91277291217), 
        0.0
    ]
    u_troy = [
        0.0, 
        np.deg2rad(-1.8396287977), 
        0.0, 
        0.1 # he didn't give me throttle, assume 10%
    ]

    # BASE_troy = linearization(x_troy,u_troy,[1.,0.,0.],use_quaternion=False,is_bire=False,**inputs)


    x_troy = [
        633.2028389854763, 
        0.0, 
        31.78308828186817, 
        0.0, 
        0.0, 
        0.0, 
        0.0, 
        0.0, 
        -15000.0, 
        0.0, 
        2.87350225624, 
        0.0
    ]
    u_troy = [
        0.0, 
        np.deg2rad(-1.60158898737), 
        0.0, 
        0.1 # he didn't give me throttle, assume 10%
    ]
    # BASE_troy = linearization(x_troy,u_troy,[1.,0.,0.],use_quaternion=False,is_bire=True,**inputs)

    x_troy = [
        629.9220543318561, 
        0.11444714781324784, 
        71.7927041431407, 
        np.deg2rad(-0.285042939922), 
        np.deg2rad(4.31996204643), 
        np.deg2rad(2.49413125039), 
        0.0,
        0.0,
        -15000.,
        np.deg2rad(60.0), 
        np.deg2rad(3.27048069272), 
        0.0

    ]
    u_troy = [
        np.deg2rad(0.0546001205051), 
        np.deg2rad(-2.81781600619), 
        np.deg2rad(-0.504654161921), 
        0.1 # he didn't give me throttle, assume 10%
    ]
    # BASE_troy = linearization(x_troy,u_troy,[1.,0.,0.],use_quaternion=False,is_bire=True,**inputs)


        
    # print("Fx,u =", Fx_u)
    # print("Fx,v =", Fx_v)
    # print("Fx,w =", Fx_w)
    # print("Fx,p =", Fx_p)
    # print("Fx,q =", Fx_q)
    # print("Fx,r =", Fx_r)
    # print("Fy,u =", Fy_u)
    # print("Fy,v =", Fy_v)
    # print("Fy,w =", Fy_w)
    # print("Fy,p =", Fy_p)
    # print("Fy,q =", Fy_q)
    # print("Fy,r =", Fy_r)
    # print("Fz,u =", Fz_u)
    # print("Fz,v =", Fz_v)
    # print("Fz,w =", Fz_w)
    # print("Fz,p =", Fz_p)
    # print("Fz,q =", Fz_q)
    # print("Fz,r =", Fz_r)
    # print("Mx,u =", Mx_u)
    # print("Mx,v =", Mx_v)
    # print("Mx,w =", Mx_w)
    # print("Mx,p =", Mx_p)
    # print("Mx,q =", Mx_q)
    # print("Mx,r =", Mx_r)
    # print("My,u =", My_u)
    # print("My,v =", My_v)
    # print("My,w =", My_w)
    # print("My,p =", My_p)
    # print("My,q =", My_q)
    # print("My,r =", My_r)
    # print("Mz,u =", Mz_u)
    # print("Mz,v =", Mz_v)
    # print("Mz,w =", Mz_w)
    # print("Mz,p =", Mz_p)
    # print("Mz,q =", Mz_q)
    # print("Mz,r =", Mz_r)