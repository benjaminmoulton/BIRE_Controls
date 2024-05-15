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

from controller_simulation import Aircraft,run_single_simulation, \
    monte_carlo_perturbations, report_latex, report_eigprops, rep2D


class ProjectBIREFeedbackLinearizationAircraft(Aircraft):
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
        # quit()
    
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
                epI     = x_euler[self.xIi_eul[0]]
                eqI     = x_euler[self.xIi_eul[1]]
                erI     = x_euler[self.xIi_eul[2]]
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
                # I     = self.inertia_model.inertia_tensor(dB)
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
                # FP = T  * self.T_dir
                # MP = [
                #     FP[2] * self.T_loc[1] - FP[1] * self.T_loc[2],
                #     FP[0] * self.T_loc[2] - FP[2] * self.T_loc[0],
                #     FP[1] * self.T_loc[0] - FP[0] * self.T_loc[1]
                # ]

                # aero forces
                # ca = cos(a); sa = sin(a)
                # cb = cos(b); sb = sin(b)
                # dynF = 0.5 * rho * V*V * self.Sw
                # Fx = FP[0] + dynF * (  CL*sa - CS*ca*sb - CD*ca*cb)
                # Fy = FP[1] + dynF * (  CS*cb - CD*sb)
                # Fz = FP[2] + dynF * (- CL*ca - CS*sa*sb - CD*sa*cb)
                #
                ph,th,ps = x_euler[9],x_euler[10],x_euler[11]
                cp = cos(ph); sp = sin(ph)
                ct = cos(th); st = sin(th)
                # cs = cos(ps); ss = sin(ps)
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
                    Cl, Cm, Cn
                    ])
                fx1x2 = np.matmul(Iinv,M + Sigma)
                z3 = fx1x2
                z1 = np.array([epI,eqI,erI])
                x2 = x_euler[12:15]
                #
                K = self.K_FB_2
                z = np.concatenate((z1,z2,z3))
                v_cl = - np.matmul(K,z) # - np.matmul(K1,z2) - np.matmul(K2,z3) - np.matmul(K3,z1)

                rest = - np.matmul(dfdw,z3) + np.matmul(dfddS,x2) \
                    + v_cl - np.matmul(dfdy,dy)
                v = np.matmul(dfddSinv,rest)
                #
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
        # u = self._limit_input(u)
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
        q_z1 = 0.1; q_z2 = 1000.0; q_z3 = 10.0
        qqr1 = qpr1 = -q_z1/5.; qqr2 = qpr2 = -q_z2/5.; qqr3 = qpr3 = -q_z3/5.
        q1 = np.array([[q_z1,0.0,qpr1],[0.0,q_z1,qqr1],[qpr1,qqr1,q_z1]])
        q2 = np.array([[q_z2,0.0,qpr2],[0.0,q_z2,qqr2],[qpr2,qqr2,q_z2]])
        q3 = np.array([[q_z3,0.0,qpr3],[0.0,q_z3,qqr3],[qpr3,qqr3,q_z3]])
        Q = block_diag(q1,q2,q3)
        r_d1 = 0.1; r_d2 = 0.1; r_d3 = 1.0
        r_aB = 0.01; r_eB = 0.05
        R = np.array([[r_d1,0.0,r_aB],[0.0,r_d2,r_eB],[r_aB,r_eB,r_d3]])
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
        K1 = np.diag([k1]*3)#; K1[0,2] = k1**2.
        K2 = np.diag([k2]*3)#; K2[0,2] = k2**2. # self.Lin_Model.KI
        K3 = np.diag([k3]*3)#; K3[0,2] = k3**2. # self.Lin_Model.K
        K = np.block([K1,K2,K3])
        K_eigs,_ = np.linalg.eig(A - np.matmul(B,K))
        report_latex(K,"K_{3ord}")
        report_latex(K_eigs,"\lambda_{cl \, 3ord}")
        # quit()
    
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
                epI     = x_euler[self.xIi_eul[0]]
                eqI     = x_euler[self.xIi_eul[1]]
                erI     = x_euler[self.xIi_eul[2]]
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
                    V_xb_in = V_xb*1.; V_yb_in = V_yb*1.; V_zb_in = V_zb*1.
                elif aero == 1:
                    a   = 0.0
                    b   = 0.0
                    V = V_tot
                    V_xb_in = V_tot*1.; V_yb_in = 0.0; V_zb_in = 0.0
                elif aero == 2:
                    a = np.arctan2(V_zb_ss,V_xb_ss)
                    b = sin(V_yb_ss/V_ss)
                    V     = V_tot
                    V_xb_in = V*np.cos(a)*np.cos(b)
                    V_yb_in = V          *np.sin(b)
                    V_zb_in = V*np.sin(a)*np.cos(b)
                elif aero == 3:
                    a = np.arctan2(V_zb_ss,V_xb_ss)
                    b = sin(V_yb_ss/V_ss)
                    V = V_xb
                    V_xb_in = V_xb*1.; V_yb_in = V_yb_ss*1.; V_zb_in = V_zb_ss*1.
                elif aero == 4:
                    a = np.arctan2(V_zb_ss,V_xb_ss)
                    b = sin(V_yb_ss/V_ss)
                    V = V_ss
                    V_xb_in = V_xb_ss*1.; V_yb_in = V_yb_ss*1.; V_zb_in = V_zb_ss*1.
                #
                # pull in matrices from linearization code
                x_in = x_euler*1.0
                x_in[0:3] = [V_xb_in,V_yb_in,V_zb_in]
                u_in = np.array([da,de,dB,tau])
                self.Lin_Model.report = False
                A,B = self.Lin_Model.build_jacobians(x_in, u_in,self.cgshift)
                # An,Bn = self.Lin_Model.build_jacobians(x_in, u_in,self.cgshift,True,self._nonlinear_euler_dynamics)
                # A = (A[3:6])[:,0:6]; An = (An[3:6])[:,0:6]
                # B = (B[3:6])[:,0:3]; Bn = (Bn[3:6])[:,0:3]
                # Anonz = A*1.; Anonz[A==0.0] = 1.0
                # Bnonz = B*1.; Bnonz[B==0.0] = 1.0
                # print(A)
                # print()
                # print(An)
                # print()
                # print((A-An)/Anonz*100.)
                # print(); print()
                # print(B)
                # print()
                # print(Bn)
                # print()
                # print((B-Bn)/Bnonz*100.)
                # print(); print()
                # quit()
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
                Iinv  = self.inertia_model.inverse_tensor(dB)
                G = 0.5*rho*V**2.*Sw*np.diag([bw,cw,bw])
                Imult = np.array([
                    (Iyy-Izz)*q*r + Iyz*(q**2-r**2) + Ixz*p*q - Ixy*p*r,
                    (Izz-Ixx)*p*r + Ixz*(r**2-p**2) + Ixy*q*r - Iyz*p*q,
                    (Ixx-Iyy)*p*q + Ixy*(p**2-q**2) + Iyz*p*r - Ixz*q*r])
                Sigma = np.matmul(hmat,[p,q,r]) + Imult
                #
                # values for later use
                Ca = cos(a); Sa = sin(a)
                Cb = cos(b); Sb = sin(b)
                #
                M = V / sos
                #
                Qdyn = 0.5*rho*V**2.*Sw
                # get forces and moments at the specified condition
                [CL, CS, CD, Cl, Cm, Cn] = \
                    self.aero_model.aero_results(*params,M=M,**{
                        "compressible" : self.is_compressible,
                        "use_Anderson" : self.use_anderson,
                        "enforce_stall" : self.has_stall
                })
                #
                # # thrust state derivatives
                TM = BAM.Prop
                #
                dfdw = A[3:6,3:6]
                #
                dfdy = A[3:6,0:3]
                #
                # evaluate at condtion for Mx, My, Mz
                T = TM.get_thrust(tau,-z_f,V)
                #
                Fx = Qdyn*(CL*Sa - CS*Ca*Sb - CD*Ca*Cb) + T
                Fy = Qdyn*(CS*Cb - CD*Sb)
                Fz = Qdyn*(- CL*Ca - CS*Sa*Sb - CD*Sa*Cb)
                #
                dfdd = B[3:6,0:3]
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
                #
                ph,th,ps = x_euler[9],x_euler[10],x_euler[11]
                cp = cos(ph); sp = sin(ph)
                ct = cos(th); st = sin(th)
                # cs = cos(ps); ss = sin(ps)
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
                    Cl, Cm, Cn
                    ]) + np.array([Fy * self.cgshift[2] - Fz * self.cgshift[1],
                    Fz * self.cgshift[0] - Fx * self.cgshift[2],
                    Fx * self.cgshift[1] - Fy * self.cgshift[0]])
                fx1x2 = np.matmul(Iinv,M + Sigma)
                z3 = fx1x2
                z1 = np.array([epI,eqI,erI])
                delta = x_euler[12:15]
                #
                K = self.K_FB_2
                z = np.concatenate((z1,z2,z3))
                v_cl = - np.matmul(K,z) # - np.matmul(K1,z2) - np.matmul(K2,z3) - np.matmul(K3,z1)

                rest = - np.matmul(dfdw,z3) + np.matmul(dfddS,delta) \
                    + v_cl - np.matmul(dfdy,dy)
                v = np.matmul(dfddSinv,rest)
                #
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
        # u = self._limit_input(u)
        inputs = self._limit_input(inputs)
        if self.order > 0:
            q = 1*self.use_quaternions
            x[12+q:16+q] = np.array(inputs)*1.
        # quantize actuators
        inputs = self._quantize_input(inputs)

        return u,inputs


class BIREDynamicInversionAircraft(Aircraft):
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
        self.is_MC = False # True # 
    
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
                # A = np.block([[A,np.zeros((3,3))],[np.zeros((3,3)),np.eye(3)]])
                # B = np.block([[self.Lin_Model.B_min],[np.zeros((3,3))]])
                # Q = np.diag([1.]*3 + [100.]*3)
                # R = np.diag([1.]*2 + [100.])
                # K,_,_ = co.lqr(A,B,Q,R)
                # print(K)
                # quit()

                v = - np.matmul(self.Lin_Model.K,e) \
                    - np.matmul(self.Lin_Model.KI,eI)
                
                delta = np.matmul(Binv, - np.matmul(A,e) - np.matmul(A,dref) + v)
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
            if not self.about_SCT: #(self.is_MC and not self.about_SCT) or \
                # (not self.is_MC and self.about_SCT):
                self.phi_trim = 0.0
            else:
                self.phi_trim = np.deg2rad(10.0)
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
        "aircraft_class" : BIREFeedbackLinearizationAircraft,
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
        "include_stall_derivatives" : False, # True, # 
        "skip_simulation" : False, # True, # 
        "skip_video" : True, # False, # 
        "name_end" : "_" + f1 + "_FB_1" # "_DI_1" # 
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
    plot_vars["plot_norm"] = False # True # 
    #
    di = [0.,0.,0.]
    # di = [5.,10.,7.] # see below
    run_base_fs["num"] = run_bire_fs["num"] = \
        run_base_rc["num"] = run_bire_rc["num"] = 1  
    ##
    # # FB_1
    run_bire_fs["aircraft_class"] = BIREFeedbackLinearizationAircraft
    run_bire_fs["name_end"] = "_" + f1 + "_FB_1"
    # zt_p,zt_q,zt_r =  0.6 , 0.6 , 0.6
    # wn_p,wn_q,wn_r =  8.0 , 8.0 , 8.0 
    #
    # run_bire_rc["aircraft_class"] = \
    #     run_bire_fs["aircraft_class"] = BIREDynamicInversionAircraft
    # run_bire_fs["name_end"] = "_" + f1 + "_DI_1"
    # # DI_1 & DI_2
    # zt_p,zt_q,zt_r =  0.7 , 0.7 , 0.7 
    # wn_p,wn_q,wn_r = 10.0 ,10.0 ,10.0 
    # #
    # bire_rc_dict["controller"]["gains"][ "K"] = \
    #     bire_fs_dict["controller"]["gains"][ "K"] = np.array([
    #     [2.*zt_p*wn_p,         0.0,  -zt_r*wn_r],
    #     [         0.0,2.*zt_q*wn_q,  -zt_r*wn_r],
    #     [         0.0,         0.0,2.*zt_r*wn_r]
    # ]).tolist()
    # bire_rc_dict["controller"]["gains"]["KI"] = \
    #     bire_fs_dict["controller"]["gains"]["KI"] = np.array([
    #     [wn_p**2.,     0.0,wn_r**2.],
    #     [     0.0,wn_q**2.,     0.0],
    #     [     0.0,     0.0,wn_r**2.]
    # ]).tolist()
    # #
    # run_bire_fs["aircraft_class"] = LinearAdaptiveAircraft
    # # LAC_1
    # run_bire_fs["state_threshold"] += [1.]*18
    # run_bire_fs["aircraft_class"] = ModelReferenceAdaptiveAircraft
    # # MRAC_1
    # run_bire_fs["state_threshold"] += [1.]*21
    # # # 
    # # # 
    # # 30 deg bank fullscale BIRE
    # p_tr_deg = -0.0820880039056245
    # q_tr_deg =  0.8352580178704386
    # r_tr_deg =  1.4467093243808735
    # # 30 deg bank fullscale BIRE w/o stall
    # p_tr_deg = -0.0811715035429339
    # q_tr_deg =  0.8353105093734944
    # r_tr_deg =  1.4468002423311315
    # # 30 deg bank fullscale BIRE w/o comp
    # p_tr_deg = -0.0897324311209177
    # q_tr_deg =  0.8348006599067213
    # r_tr_deg =  1.4459171571504688
    # # 30 deg bank fullscale BIRE w/o comp w/o stall
    # p_tr_deg = -0.0887219675261609
    # q_tr_deg =  0.8348639332754728
    # r_tr_deg =  1.4460267498399122
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # 10 deg bank fullscale BIRE
    p_tr_deg = -0.0236847366216922
    q_tr_deg =  0.0886486340380570
    r_tr_deg =  0.5027513865539764
    # 10 deg bank fullscale BIRE w/o stall
    p_tr_deg = -0.0234005498745413
    q_tr_deg =  0.0886532662096017
    r_tr_deg =  0.5027776569042431
    # # 10 deg bank fullscale BIRE w/o stall w/Dcg = [1.0, 2.0, -1.0]
    # p_tr_deg = -0.0258643620770918
    # q_tr_deg =  0.0886114412475169
    # r_tr_deg =  0.5025404557571655
    # # 10 deg bank fullscale BIRE w/o compressibility
    # p_tr_deg = -0.0259965332970090
    # q_tr_deg =  0.0886089113144201
    # r_tr_deg =  0.5025261077935890
    # # 10 deg bank fullscale BIRE w/o stall w/o compressibility
    # p_tr_deg = -0.0256961439719502
    # q_tr_deg =  0.0886143006077343
    # r_tr_deg =  0.5025566719947823
    p_comm = 5.0 # 15.0 # 7.5 # 
    t_zero = 0.0
    p_time = t_zero + 2.0
    t_end = 0.0 # 25.0 # 
    tf = t_end + p_time + 8.0
    bire_fs_dict["reference"] = bire_rc_dict["reference"] = {
        "deg2rad_states" : [3,4,5],
        "3" : [ [0.0, 0.0], [t_zero, 0.0], [t_zero, p_comm], [p_time, p_comm], [p_time, p_tr_deg] ],
        "4" : [ [0.0, 0.0], [t_zero, 0.0], [t_zero,    0.0], [p_time,    0.0], [p_time, q_tr_deg] ],
        "5" : [ [0.0, 0.0], [t_zero, 0.0], [t_zero,    0.0], [p_time,    0.0], [p_time, r_tr_deg] ],
        "sct_on_5" : False
    }
    run_bire_fs["track_check_time"] = run_bire_rc["track_check_time"] = \
        run_bire_fs["final_time"] = run_bire_rc["final_time"] = tf # 200.0 # 10.0 # 
    bire_fs_dict["simulation"]["include_stall"] = \
        bire_rc_dict["simulation"]["include_stall"] = False
    # bire_fs_dict["simulation"]["include_compressibility"] = False
    bire_fs_dict["simulation"]["integrator"] = \
        bire_rc_dict["simulation"]["integrator"] = "rk4"
    # bire_fs_dict["actuators"]["order"] = 0
    # run_bire_fs["state_threshold"] = run_bire_fs["state_threshold"][:-4]
    # bire_fs_dict["aircraft"]["CG_shift[ft]"] = [1.0, 2.0, -1.0]
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
    # # run_single_simulation(base_fs_dict,rtdst_1sg=di,**run_base_fs,**plot_vars)
    # # run_single_simulation(bire_rc_dict,rtdst_1sg=di,**run_bire_rc,**plot_vars)
    # # run_single_simulation(base_rc_dict,rtdst_1sg=di,**run_base_rc,**plot_vars)
    # quit()

    # # # # run monte carlo perturbation analysis
    # # num = 1000
    # # run_base_fs["num"] = run_bire_fs["num"] = \
    # #     run_base_rc["num"] = run_bire_rc["num"] = num
    # di = [2.,2.,2.] # RC F-16 # change r to 6
    # # # di = [0.,0.,0.]
    # di = [8.,8.,0.2] # FS BIRE
    # run_bire_fs["state_threshold"] = [
    #     10., 15., 15.,
    #     2.0, 2.0, 2.0, # 20., 10., 10., # 
    #     1., 1., 50.,
    #     25., 10., 1.,
    #     5., 5., 5., 0.05
    # ]
    # run_bire_fs["final_time"] = run_base_fs["final_time"] = \
    #     run_bire_rc["final_time"] = run_base_rc["final_time"] = 15.0
    # run_base_fs["num"] = run_bire_fs["num"] = \
    #     run_base_rc["num"] = run_bire_rc["num"] = 1000
    # # run_bire_fs["has_model_error"] = run_base_fs["has_model_error"] = \
    # #     run_bire_rc["has_model_error"] = run_base_rc["has_model_error"] = True # False # 
    # # monte_carlo_perturbations(bire_fs_dict,rtdst_1sg=di,**run_bire_fs,**plot_vars)
    # # # monte_carlo_perturbations(base_fs_dict,rtdst_1sg=di,**run_base_fs,**plot_vars)
    # # # monte_carlo_perturbations(bire_rc_dict,rtdst_1sg=di,**run_bire_rc,**plot_vars)
    # # # monte_carlo_perturbations(base_rc_dict,rtdst_1sg=di,**run_base_rc,**plot_vars)
    # # quit()
    # #
    # single axis pqr dispersions
    ###########################################################################
    bire_fs_dict["reference"] = {
        "deg2rad_states" : [3,4,5],
        "3" : [[ 0.0, p_tr_deg],[ 2.0, p_tr_deg]],
        "4" : [[ 0.0, q_tr_deg],[ 2.0, q_tr_deg]],
        "5" : [[ 0.0, r_tr_deg],[ 2.0, r_tr_deg]],
        "sct_on_5" : False
    }
    run_bire_fs["trim_bank"] = 10.0
    run_bire_fs["num"] = 3 # 10 # 1000 # 
    ###########################################################################
    disa = [[ 10.,0.,0.],[0., 20.,0.],[0.,0.,  1.]] # FB_1
    for i in [0]: # [1]: # range(3): # 
        ds = disa[i]
        monte_carlo_perturbations(bire_fs_dict,rtdst_1sg=ds,**run_bire_fs,**plot_vars)
        # monte_carlo_perturbations(base_fs_dict,rtdst_1sg=ds,**run_base_fs,**plot_vars)
        # monte_carlo_perturbations(bire_rc_dict,rtdst_1sg=ds,**run_bire_rc,**plot_vars)
        # monte_carlo_perturbations(base_rc_dict,rtdst_1sg=ds,**run_base_rc,**plot_vars)
    quit()
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





    # into and out of SCT
    # bire_fs_dict["reference"] = bire_rc_dict["reference"] = {
    #     "deg2rad_states" : [3,4,5],
    #     # "3" : [ [ 0.0, p_tr_deg], [ 1.0, p_tr_deg] ],
    #     # "4" : [ [ 0.0, q_tr_deg], [ 1.0, q_tr_deg] ],
    #     # "5" : [ [ 0.0, r_tr_deg], [ 1.0, r_tr_deg] ],
    #     # "3" : [
    #     #     [         0.0,   p_comm],
    #     #     [      p_time,   p_comm],
    #     #     [      p_time, p_tr_deg],
    #     #     [       t_end, p_tr_deg],
    #     #     [       t_end,  -p_comm],
    #     #     [t_end+p_time,  -p_comm],
    #     #     [t_end+p_time,      0.0]
    #     # ],
    #     # "4" : [
    #     #     [   0.0, 0.0],
    #     #     [p_time, 0.0],
    #     #     [p_time, q_tr_deg],
    #     #     [ t_end, q_tr_deg],
    #     #     [ t_end, 0.0]
    #     # ],
    #     # "5" : [
    #     #     [ 0.0 , 0.0],
    #     #     [p_time, 0.0],
    #     #     [p_time, r_tr_deg],
    #     #     [ t_end, r_tr_deg],
    #     #     [ t_end, 0.0]
    #     # ],
    #     "sct_on_5" : False
    # }