import numpy as np
import control as co
from matplotlib import pyplot as plt
from sympy import pi, sin, cos, tan, exp, asin, atan, atan2, Matrix as array
from sympy.matrices.dense import zeros
from scipy.linalg import block_diag
# from slycot.exceptions import SlycotArithmeticError
from std_atm import stdatm_english
import warnings as wrn
import json

import sys
aero_directory = '../aerodynamics_model/'
mass_directory = '../mass_properties/'

sys.path.insert(1, aero_directory)
sys.path.insert(1, mass_directory)

from os import mkdir, rmdir, remove, listdir
from os.path import exists as path_exists

from f16_aero import F16Aero
from bire_aero import BIREAero
# from inertia_model import InertiaModel


import sympy as sy

def _build_input_jacobian(x_eq, u_eq, aero_model, inertia_model, aero_dict, 
    include_stall,include_compressible,use_Anderson, cg_shift = [0.,0.,0.]):
    # report
    if True:
        print("    building input jacobian...")

    # pull out evaluating condition
    if False: u,v,w,p,q,r,x,y,z, e0, ex, ey,ez = x_eq[0:13]
    else:              u,v,w,p,q,r,x,y,z,phi,tht,psi    = x_eq[0:12]
    if True: da, de, dB, tau = u_eq
    else:            da, de, dr, tau = u_eq
    Dxcg, Dycg, Dzcg = cg_shift
    C = aero_model
    IM = inertia_model

    # values for latter use
    a = atan2(w,u)
    V = sy.sqrt(u*u + v*v + w*w) # **0.5
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
    g, rho,sos = sy.symbols("g rho sos")
    # #####################
    # g = 32.12780074195162
    # print("g =",g)
    # #####################
    #
    M = V / sos
    #
    Qdyn = 0.5*rho*V**2*C.S_w
    Qlon = Qdyn*C.c_w
    Qlat = Qdyn*C.b_w
    #
    m = IM.W/g
    minv = g/IM.W
    if True:
        Iinv = IM.inverse_tensor(dB)
        dIinv = IM.inverse_tensor_derivative(dB)
        [ Ixx, Iyy, Izz, Ixy, Ixz, Iyz] = IM.inertia_results(dB)
        [dIxx,dIyy,dIzz,dIxy,dIxz,dIyz] = IM.inertia_derivative_results(dB)
        hx, hy, hz = IM.angular_momentum_results()
    else:
        Iinv = IM.inverse_tensor(0.)
    
    # input aerodynamic force derivatives
    if True:
        # evaluate derivatives wrt bire angle
        C.evaluate_coeffs(dB)
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
        oCD_dB = C.dCD0 + C.dCDL*CL1 + C.CDL*dCL1 + C.dCDL2*CL1**2 + \
            + 2.*C.CDL2*CL1*dCL1 + C.dCDS*CS1 + C.CDS*dCS1 + \
            + C.dCDS2*CS1**2 + 2.*C.CDS2*CS1*dCS1 + (C.dCDSp*CS1 + \
            + C.CDSp*dCS1 + C.dCDp)*pbar + (C.dCDL2q*CL1**2 + \
            + 2.*C.CDL2q*CL1*dCL1 + C.dCDLq*CL1 + C.CDLq*dCL1 + \
            + C.dCDq)*qbar + (C.dCDSr*CS1 + C.CDSr*dCS1 + C.dCDr)*rbar + \
            + (C.dCDSda*CS1 + C.CDSda*dCS1 + C.dCDda)*da + \
            + (C.dCDLde*CL1 + C.CDLde*dCL1 + C.dCDde)*de + C.dCDde2*de**2
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
    if True:
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
    if True:
        params = [a, b, pbar, qbar, rbar, da, de, dB]
    else:
        params = [a, b, pbar, qbar, rbar, da, de, dr]
    [CL, CS, CD, Cl, Cm, Cn] = C.aero_results(*params,M=M,**aero_dict)

    # Stall corrections
    if include_stall and include_stall: # False: # 

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
        if True:
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
        if True:
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
        if True:
            aCL_dB,aCS_dB,aCD_dB = oCL_dB,oCS_dB,oCD_dB
            aCl_dB,aCm_dB,aCn_dB = oCl_dB,oCm_dB,oCn_dB
        else:
            aCL_dr,aCS_dr,aCD_dr = oCL_dr,oCS_dr,oCD_dr
            aCl_dr,aCm_dr,aCn_dr = oCl_dr,oCm_dr,oCn_dr
    
    # Compressibility corrections
    if include_compressible:
        # incompressible coefficients
        [aCL, aCS, aCD, aCl, aCm, aCn] = \
            C.aero_results(*params,M=M,**{
            "compressible" : False,
            "use_Anderson" : False,
            "enforce_stall" : include_stall
        })

        # Mach correction derivatives
        if True: # subsonic
            if use_Anderson:
                L_w, L_h, L_v = C.Lam_w, C.Lam_h, C.Lam_v
                R_w, R_h, R_v = C.RA_w, C.RA_h, C.RA_v

                # derivatives wrt incompressible coefficients
                CL_aCL = Anderson_correction_der_coeff(aCL,L_w,R_w,M)
                Cm_aCm = Anderson_correction_der_coeff(aCm,L_w,R_w,M)
                if True:
                    CS_aCS = Anderson_correction_der_coeff(aCS,L_h,R_h,M)
                    Cl_aCl = Anderson_correction_der_coeff(aCl,L_w,R_w,M)
                    Cn_aCn = Anderson_correction_der_coeff(aCn,L_h,R_h,M)
                else:
                    CS_aCS = Anderson_correction_der_coeff(aCS,L_v,R_v,M)
                    Cl_aCl = Anderson_correction_der_coeff(aCl,L_v,R_v,M)
                    Cn_aCn = Anderson_correction_der_coeff(aCn,L_v,R_v,M)
            else:
                CL_aCL = CS_aCS = Cl_aCl = Cm_aCm = Cn_aCn = \
                    1. / sy.sqrt(1. - M**2) # **0.5
        else: # supersonic
            CL_aCL = CS_aCS = Cl_aCl = Cm_aCm = Cn_aCn = \
                1. / (M**2 - 1.)**0.5
        
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
        if True:
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
        if True:
            CL_dB,CS_dB,CD_dB = aCL_dB,aCS_dB,aCD_dB
            Cl_dB,Cm_dB,Cn_dB = aCl_dB,aCm_dB,aCn_dB
        else:
            CL_dr,CS_dr,CD_dr = aCL_dr,aCS_dr,aCD_dr
            Cl_dr,Cm_dr,Cn_dr = aCl_dr,aCm_dr,aCn_dr
        
    # thrust state derivatives
    TM = C.Prop
    if False:
        T_tau = (rho/TM.rho_0)**TM.a*(TM.T0 + TM.T1*V + TM.T2*V*V)
    else:
        # pull out each setting
        Tmil = TM._T_mil(rho,V,-z)
        if tau <= 0.77:
            P1_tau = 64.94
            Tidle = TM._T_idle(rho,V,-z)
            T_tau = (Tmil - Tidle)*P1_tau/50.
        else:
            P1_tau = 217.38
            Tmax = TM._T_max(rho,V,-z)
            T_tau = (Tmax - Tmil)*P1_tau/50.

    # body-fixed force derivatives wrt input
    Fx_da = Qdyn*(CL_da*Sa - CS_da*Ca*Sb - CD_da*Ca*Cb)
    Fx_de = Qdyn*(CL_de*Sa - CS_de*Ca*Sb - CD_de*Ca*Cb)
    if True:
        Fx_dB = Qdyn*(CL_dB*Sa - CS_dB*Ca*Sb - CD_dB*Ca*Cb)
    else:
        Fx_dr = Qdyn*(CL_dr*Sa - CS_dr*Ca*Sb - CD_dr*Ca*Cb)
    Fx_tau = T_tau
    #
    Fy_da = Qdyn*(CS_da*Cb - CD_da*Sb)
    Fy_de = Qdyn*(CS_de*Cb - CD_de*Sb)
    if True:
        Fy_dB = Qdyn*(CS_dB*Cb - CD_dB*Sb)
    else:
        Fy_dr = Qdyn*(CS_dr*Cb - CD_dr*Sb)
    #
    Fz_da = Qdyn*(- CL_da*Ca - CS_da*Sa*Sb - CD_da*Sa*Cb)
    Fz_de = Qdyn*(- CL_de*Ca - CS_de*Sa*Sb - CD_de*Sa*Cb)
    if True:
        Fz_dB = Qdyn*(- CL_dB*Ca - CS_dB*Sa*Sb - CD_dB*Sa*Cb)
    else:
        Fz_dr = Qdyn*(- CL_dr*Ca - CS_dr*Sa*Sb - CD_dr*Sa*Cb)
    
    # body-fixed moment derivatives wrt input
    Mx_da = Qlat*Cl_da + Fy_da*Dzcg - Fz_da*Dycg
    Mx_de = Qlat*Cl_de + Fy_de*Dzcg - Fz_de*Dycg
    if True:
        Mx_dB = Qlat*Cl_dB + Fy_dB*Dzcg - Fz_dB*Dycg
    else:
        Mx_dr = Qlat*Cl_dr + Fy_dr*Dzcg - Fz_dr*Dycg
    #
    My_da = Qlon*Cm_da + Fz_da*Dxcg - Fx_da*Dzcg
    My_de = Qlon*Cm_de + Fz_de*Dxcg - Fx_de*Dzcg
    if True:
        My_dB = Qlon*Cm_dB + Fz_dB*Dxcg - Fx_dB*Dzcg
    else:
        My_dr = Qlon*Cm_dr + Fz_dr*Dxcg - Fx_dr*Dzcg
    My_tau = - Fx_tau*Dzcg
    #
    Mz_da = Qlat*Cn_da + Fx_da*Dycg - Fy_da*Dxcg
    Mz_de = Qlat*Cn_de + Fx_de*Dycg - Fy_de*Dxcg
    if True:
        Mz_dB = Qlat*Cn_dB + Fx_dB*Dycg - Fy_dB*Dxcg
    else:
        Mz_dr = Qlat*Cn_dr + Fx_dr*Dycg - Fy_dr*Dxcg
    Mz_tau = Fx_tau*Dycg

    # evaluate at condtion for Mx, My, Mz
    T = TM.get_thrust(tau,-z,V)
    #
    Fx = Qdyn*(CL*Sa - CS*Ca*Sb - CD*Ca*Cb) + T
    Fy = Qdyn*(CS*Cb - CD*Sb)
    Fz = Qdyn*(- CL*Ca - CS*Sa*Sb - CD*Sa*Cb)
    #
    Mx = Qlat*Cl + Fy*Dzcg - Fz*Dycg
    My = Qlon*Cm + Fz*Dxcg - Fx*Dzcg
    Mz = Qlat*Cn + Fx*Dycg - Fy*Dxcg

    # initialize jacobian
    B = zeros(x_eq.shape[0],u_eq.shape[0])

    # set range values for ease of use
    rV = [0,1,2]
    rw = [3,4,5]
    ru = [0,1,3]
    r3 = 2
    
    # assemble components
    Fder = minv*sy.Matrix([
        [Fx_da, Fx_de, Fx_tau],
        [Fy_da, Fy_de,     0.],
        [Fz_da, Fz_de,     0.]
    ])
    Mder = np.matmul(Iinv,sy.Matrix([
        [Mx_da, Mx_de,     0.],
        [My_da, My_de, My_tau],
        [Mz_da, Mz_de, Mz_tau]
    ]))
    B[0:3,0] = Fder[:,0]
    B[0:3,1] = Fder[:,1]
    B[0:3,3] = Fder[:,2]
    B[3:6,0] = Mder[:,0]
    B[3:6,1] = Mder[:,1]
    B[3:6,3] = Mder[:,2]
    if True:
        B[0:3,r3] = sy.Matrix([Fx_dB, Fy_dB, Fz_dB])/m
        wdot = (
            sy.Matrix([Mx,My,Mz]) +
            np.matmul(sy.Matrix([
            [ 0., -hz,  hy],
            [ hz,  0., -hx],
            [-hy,  hx,  0.]
            ]), sy.Matrix([p,q,r])) + 
            sy.Matrix([
                ( Iyy- Izz)*q*r +  Iyz*(q**2-r**2) +  Ixz*p*q -  Ixy*p*r,
                ( Izz- Ixx)*p*r +  Ixz*(r**2-p**2) +  Ixy*q*r -  Iyz*p*q,
                ( Ixx- Iyy)*p*q +  Ixy*(p**2-q**2) +  Iyz*p*r -  Ixz*q*r
            ])
        )
        wdot_dB = (
            sy.Matrix([Mx_dB,My_dB,Mz_dB]) +
            sy.Matrix([
                (dIyy-dIzz)*q*r + dIyz*(q**2-r**2) + dIxz*p*q - dIxy*p*r,
                (dIzz-dIxx)*p*r + dIxz*(r**2-p**2) + dIxy*q*r - dIyz*p*q,
                (dIxx-dIyy)*p*q + dIxy*(p**2-q**2) + dIyz*p*r - dIxz*q*r
            ])
        )
        B[3:6,r3] = ( 
        np.matmul(Iinv,wdot_dB) + 
        np.matmul(dIinv,wdot) ) # 
    else:
        B[0:3,r3] = sy.Matrix([Fx_dr, Fy_dr, Fz_dr])/m
        B[3:6,r3] = np.matmul(Iinv,sy.Matrix([
            Mx_dr, My_dr, Mz_dr
        ])[:,np.newaxis])[:,0]

    return B


class InertiaModel:
    def __init__(self, inp_dir='./', **kwargs):
        is_bire = kwargs.get("is_bire",False)
        is_rc = kwargs.get("is_rc", False)
        is_SAL = kwargs.get("is_SAL", False)
        if is_SAL:
            fn_def = "f16_SAL_inertial_properties.json"
        else:
            if is_rc:
                if is_bire:
                    fn_def = "bire_rc_inertial_properties.json"
                else:
                    fn_def = "f16_rc_inertial_properties.json"
            else:
                if is_bire:
                    fn_def = "bire_inertial_properties.json"
                else:
                    fn_def = "f16_inertial_properties.json"
        fn = kwargs.get('fn', fn_def)
        self.model_coeffs_dict = json.load(open(inp_dir + fn))

        self.W = self.model_coeffs_dict["weight"]

        self.inertia_coeffs = self.model_coeffs_dict["inertia"]
        self.Ixx_coeffs = self.inertia_coeffs["Ixx"]
        self.Iyy_coeffs = self.inertia_coeffs["Iyy"]
        self.Izz_coeffs = self.inertia_coeffs["Izz"]
        self.Ixy_coeffs = self.inertia_coeffs["Ixy"]
        self.Ixz_coeffs = self.inertia_coeffs["Ixz"]
        self.Iyz_coeffs = self.inertia_coeffs["Iyz"]
        self.Ixx_A = self.Ixx_coeffs["A"]
        self.Ixx_w = self.Ixx_coeffs["w"]
        self.Ixx_p = self.Ixx_coeffs["phi"]
        self.Ixx_z = self.Ixx_coeffs["z"]
        self.Iyy_A = self.Iyy_coeffs["A"]
        self.Iyy_w = self.Iyy_coeffs["w"]
        self.Iyy_p = self.Iyy_coeffs["phi"]
        self.Iyy_z = self.Iyy_coeffs["z"]
        self.Izz_A = self.Izz_coeffs["A"]
        self.Izz_w = self.Izz_coeffs["w"]
        self.Izz_p = self.Izz_coeffs["phi"]
        self.Izz_z = self.Izz_coeffs["z"]
        self.Ixy_A = self.Ixy_coeffs["A"]
        self.Ixy_w = self.Ixy_coeffs["w"]
        self.Ixy_p = self.Ixy_coeffs["phi"]
        self.Ixy_z = self.Ixy_coeffs["z"]
        self.Ixz_A = self.Ixz_coeffs["A"]
        self.Ixz_w = self.Ixz_coeffs["w"]
        self.Ixz_p = self.Ixz_coeffs["phi"]
        self.Ixz_z = self.Ixz_coeffs["z"]
        self.Iyz_A = self.Iyz_coeffs["A"]
        self.Iyz_w = self.Iyz_coeffs["w"]
        self.Iyz_p = self.Iyz_coeffs["phi"]
        self.Iyz_z = self.Iyz_coeffs["z"]

        self.h_coeffs = self.model_coeffs_dict["angular_momentum"]
        self.hx = self.h_coeffs["hx"]
        self.hy = self.h_coeffs["hy"]
        self.hz = self.h_coeffs["hz"]

        if is_rc:
            self._Iyz = lambda dB : self.Iyz_A*sin(self.Iyz_w*dB + self.Iyz_p)\
                + self.Iyz_z
            self._dIyz = lambda dB : self.Iyz_A*self.Iyz_w*cos(\
                self.Iyz_w*dB + self.Iyz_p)

    def _Ixx(self, dB):
        Ixx = self.Ixx_A*sin(self.Ixx_w*dB + self.Ixx_p) + self.Ixx_z
        return Ixx

    def _dIxx(self, dB):
        dIxx = self.Ixx_A*self.Ixx_w*cos(self.Ixx_w*dB + self.Ixx_p)
        return dIxx

    def _Iyy(self, dB):
        Iyy = self.Iyy_A*sin(self.Iyy_w*dB + self.Iyy_p) + self.Iyy_z
        return Iyy

    def _dIyy(self, dB):
        dIyy = self.Iyy_A*self.Iyy_w*cos(self.Iyy_w*dB + self.Iyy_p)
        return dIyy

    def _Izz(self, dB):
        Izz = self.Izz_A*sin(self.Izz_w*dB + self.Izz_p) + self.Izz_z
        return Izz

    def _dIzz(self, dB):
        dIzz = self.Izz_A*self.Izz_w*cos(self.Izz_w*dB + self.Izz_p)
        return dIzz

    def _Ixy(self, dB):
        Ixy = self.Ixy_A*sin(self.Ixy_w*dB + self.Ixy_p) + self.Ixy_z
        return Ixy

    def _dIxy(self, dB):
        dIxy = self.Ixy_A*self.Ixy_w*cos(self.Ixy_w*dB + self.Ixy_p)
        return dIxy

    def _Ixz(self, dB):
        Ixz = self.Ixz_A*sin(self.Ixz_w*dB + self.Ixz_p) + self.Ixz_z
        return Ixz

    def _dIxz(self, dB):
        dIxz = self.Ixz_A*self.Ixz_w*cos(self.Ixz_w*dB + self.Ixz_p)
        return dIxz

    def _Iyz(self, dB):
        Iyz = self.Iyz_A*abs(sin(self.Iyz_w*dB + self.Iyz_p)) + self.Iyz_z
        return Iyz

    def _dIyz(self, dB):
        if dB // pi != 0.0:
            O = self.Iyz_w*dB + self.Iyz_p
            dIyz = self.Iyz_A*self.Iyz_w*sin(O)*cos(O)/abs(sin(O))
        else:
            dIyz = 0.0
        return dIyz
    
    def _determinant(self, dB):
        Ixx = self._Ixx(dB)
        Iyy = self._Iyy(dB)
        Izz = self._Izz(dB)
        Ixy = self._Ixy(dB)
        Ixz = self._Ixz(dB)
        Iyz = self._Iyz(dB)
        det = ( Ixx*(Iyy*Izz - Iyz**2) - Ixy*Ixz*Iyz \
            - (Ixy**2*Izz + Ixz**2*Iyy) )
        return det
    
    def _determinant_derivative(self, dB):
        # get values
        Ixx = self._Ixx(dB)
        Iyy = self._Iyy(dB)
        Izz = self._Izz(dB)
        Ixy = self._Ixy(dB)
        Ixz = self._Ixz(dB)
        Iyz = self._Iyz(dB)
        # get derivatives
        dIxx = self._dIxx(dB)
        dIyy = self._dIyy(dB)
        dIzz = self._dIzz(dB)
        dIxy = self._dIxy(dB)
        dIxz = self._dIxz(dB)
        dIyz = self._dIyz(dB)
        ddet = ( dIxx*(Iyy*Izz - Iyz**2) + Ixx*(dIyy*Izz + Iyy*dIzz \
            - 2.*Iyz*dIyz) - dIxy*Ixz*Iyz - Ixy*dIxz*Iyz - Ixy*Ixz*dIyz
            -(2.*Ixy*dIxy*Izz + Ixy**2*dIzz + 2.*Ixz*dIxz*Iyy + Ixz**2*dIyy))
        return ddet
    
    def _adjoint(self, dB):
        Ixx = self._Ixx(dB)
        Iyy = self._Iyy(dB)
        Izz = self._Izz(dB)
        Ixy = self._Ixy(dB)
        Ixz = self._Ixz(dB)
        Iyz = self._Iyz(dB)
        return [
            [ Iyy*Izz - Iyz**2, Ixy*Izz + Ixz*Iyz, Ixy*Iyz + Ixz*Iyy],
            [ Ixy*Izz + Iyz*Ixz, Ixx*Izz - Ixz**2, Ixx*Iyz + Ixy*Ixz],
            [ Ixy*Iyz + Ixz*Iyy, Ixx*Iyz + Ixz*Ixy, Ixx*Iyy - Ixy**2]
        ]
    
    def _adjoint_derivative(self, dB):
        # get values
        Ixx = self._Ixx(dB)
        Iyy = self._Iyy(dB)
        Izz = self._Izz(dB)
        Ixy = self._Ixy(dB)
        Ixz = self._Ixz(dB)
        Iyz = self._Iyz(dB)
        # get derivatives
        dIxx = self._dIxx(dB)
        dIyy = self._dIyy(dB)
        dIzz = self._dIzz(dB)
        dIxy = self._dIxy(dB)
        dIxz = self._dIxz(dB)
        dIyz = self._dIyz(dB)
        # initialize and assign
        dadj = zeros(3,3)
        dadj[0,0] = dIyy*Izz + Iyy*dIzz - 2.*Iyz*dIyz
        dadj[0,1] = dadj[1,0] = dIxy*Izz + Ixy*dIzz + dIxz*Iyz + Ixz*dIyz
        dadj[1,1] = dIxx*Izz + Ixx*dIzz - 2.*Ixz*dIxz
        dadj[0,2] = dadj[2,0] = dIxy*Iyz + Ixy*dIyz + dIxz*Iyy + Ixz*dIyy
        dadj[2,2] = dIxx*Iyy + Ixx*dIyy - 2.*Ixy*dIxy
        dadj[1,2] = dadj[2,1] = dIxx*Iyz + Ixx*dIyz + dIxy*Ixz + Ixy*dIxz
        return dadj

    def inertia_results(self, dB):
        return [self._Ixx(dB), self._Iyy(dB), self._Izz(dB),
        self._Ixy(dB), self._Ixz(dB), self._Iyz(dB)]

    def inertia_derivative_results(self, dB):
        return [self._dIxx(dB), self._dIyy(dB), self._dIzz(dB),
        self._dIxy(dB), self._dIxz(dB), self._dIyz(dB)]

    def angular_momentum_results(self):
        return [self.hx, self.hy, self.hz]

    def inertia_tensor(self, dB):
        Ixx = self._Ixx(dB)
        Iyy = self._Iyy(dB)
        Izz = self._Izz(dB)
        Ixy = self._Ixy(dB)
        Ixz = self._Ixz(dB)
        Iyz = self._Iyz(dB)
        return [
            [  Ixx, -Ixy, -Ixz],
            [ -Ixy,  Iyy, -Iyz],
            [ -Ixz, -Iyz,  Izz]
        ]

    def inverse_tensor(self, dB):
        # return ( array(self._adjoint(dB))/self._determinant(dB) ).tolist()
        adj = self._adjoint(dB)
        det = self._determinant(dB)
        return [
            [adj[0][0]/det, adj[0][1]/det, adj[0][2]/det],
            [adj[1][0]/det, adj[1][1]/det, adj[1][2]/det],
            [adj[2][0]/det, adj[2][1]/det, adj[2][2]/det]
        ]

    def inverse_tensor_derivative(self, dB):
        # get pieces
        Iinv = array(self.inverse_tensor(dB))
        det = self._determinant(dB)
        dadj = array(self._adjoint_derivative(dB))
        ddet = self._determinant_derivative(dB)

        return ( (dadj - Iinv*ddet)/det ).tolist()


class InertiaModel_zerodB:
    def __init__(self, inp_dir='./', **kwargs):
        is_bire = kwargs.get("is_bire",False)
        is_rc = kwargs.get("is_rc", False)
        is_SAL = kwargs.get("is_SAL", False)
        if is_SAL:
            fn_def = "f16_SAL_inertial_properties.json"
        else:
            if is_rc:
                if is_bire:
                    fn_def = "bire_rc_inertial_properties.json"
                else:
                    fn_def = "f16_rc_inertial_properties.json"
            else:
                if is_bire:
                    fn_def = "bire_inertial_properties.json"
                else:
                    fn_def = "f16_inertial_properties.json"
        fn = kwargs.get('fn', fn_def)
        self.model_coeffs_dict = json.load(open(inp_dir + fn))

        self.W = self.model_coeffs_dict["weight"]

        self.inertia_coeffs = self.model_coeffs_dict["inertia"]
        self.Ixx_coeffs = self.inertia_coeffs["Ixx"]
        self.Iyy_coeffs = self.inertia_coeffs["Iyy"]
        self.Izz_coeffs = self.inertia_coeffs["Izz"]
        self.Ixy_coeffs = self.inertia_coeffs["Ixy"]
        self.Ixz_coeffs = self.inertia_coeffs["Ixz"]
        self.Iyz_coeffs = self.inertia_coeffs["Iyz"]
        self.Ixx_A = self.Ixx_coeffs["A"]
        self.Ixx_w = self.Ixx_coeffs["w"]
        self.Ixx_p = self.Ixx_coeffs["phi"]
        self.Ixx_z = self.Ixx_coeffs["z"]
        self.Iyy_A = self.Iyy_coeffs["A"]
        self.Iyy_w = self.Iyy_coeffs["w"]
        self.Iyy_p = self.Iyy_coeffs["phi"]
        self.Iyy_z = self.Iyy_coeffs["z"]
        self.Izz_A = self.Izz_coeffs["A"]
        self.Izz_w = self.Izz_coeffs["w"]
        self.Izz_p = self.Izz_coeffs["phi"]
        self.Izz_z = self.Izz_coeffs["z"]
        self.Ixy_A = self.Ixy_coeffs["A"]
        self.Ixy_w = self.Ixy_coeffs["w"]
        self.Ixy_p = self.Ixy_coeffs["phi"]
        self.Ixy_z = self.Ixy_coeffs["z"]
        self.Ixz_A = self.Ixz_coeffs["A"]
        self.Ixz_w = self.Ixz_coeffs["w"]
        self.Ixz_p = self.Ixz_coeffs["phi"]
        self.Ixz_z = self.Ixz_coeffs["z"]
        self.Iyz_A = self.Iyz_coeffs["A"]
        self.Iyz_w = self.Iyz_coeffs["w"]
        self.Iyz_p = self.Iyz_coeffs["phi"]
        self.Iyz_z = self.Iyz_coeffs["z"]

        self.h_coeffs = self.model_coeffs_dict["angular_momentum"]
        self.hx = self.h_coeffs["hx"]
        self.hy = self.h_coeffs["hy"]
        self.hz = self.h_coeffs["hz"]

        if is_rc:
            self._Iyz = lambda dB : self.Iyz_A*sin(self.Iyz_w*dB + self.Iyz_p)\
                + self.Iyz_z
            self._dIyz = lambda dB : self.Iyz_A*self.Iyz_w*cos(\
                self.Iyz_w*dB + self.Iyz_p)

    def _Ixx(self, dB):
        dB = 0.0
        Ixx = self.Ixx_A*sin(self.Ixx_w*dB + self.Ixx_p) + self.Ixx_z
        return Ixx

    def _dIxx(self, dB):
        dIxx = self.Ixx_A*self.Ixx_w*cos(self.Ixx_w*dB + self.Ixx_p)
        return 0.0 # dIxx

    def _Iyy(self, dB):
        dB = 0.0
        Iyy = self.Iyy_A*sin(self.Iyy_w*dB + self.Iyy_p) + self.Iyy_z
        return Iyy

    def _dIyy(self, dB):
        dIyy = self.Iyy_A*self.Iyy_w*cos(self.Iyy_w*dB + self.Iyy_p)
        return 0.0 # dIyy

    def _Izz(self, dB):
        dB = 0.0
        Izz = self.Izz_A*sin(self.Izz_w*dB + self.Izz_p) + self.Izz_z
        return Izz

    def _dIzz(self, dB):
        dIzz = self.Izz_A*self.Izz_w*cos(self.Izz_w*dB + self.Izz_p)
        return 0.0 # dIzz

    def _Ixy(self, dB):
        dB = 0.0
        Ixy = self.Ixy_A*sin(self.Ixy_w*dB + self.Ixy_p) + self.Ixy_z
        return Ixy

    def _dIxy(self, dB):
        dIxy = self.Ixy_A*self.Ixy_w*cos(self.Ixy_w*dB + self.Ixy_p)
        return 0.0 # dIxy

    def _Ixz(self, dB):
        dB = 0.0
        Ixz = self.Ixz_A*sin(self.Ixz_w*dB + self.Ixz_p) + self.Ixz_z
        return Ixz

    def _dIxz(self, dB):
        dIxz = self.Ixz_A*self.Ixz_w*cos(self.Ixz_w*dB + self.Ixz_p)
        return 0.0 # dIxz

    def _Iyz(self, dB):
        dB = 0.0
        Iyz = self.Iyz_A*abs(sin(self.Iyz_w*dB + self.Iyz_p)) + self.Iyz_z
        return Iyz

    def _dIyz(self, dB):
        if dB // pi != 0.0:
            O = self.Iyz_w*dB + self.Iyz_p
            dIyz = self.Iyz_A*self.Iyz_w*sin(O)*cos(O)/abs(sin(O))
        else:
            dIyz = 0.0
        return 0.0 # dIyz
    
    def _determinant(self, dB):
        Ixx = self._Ixx(dB)
        Iyy = self._Iyy(dB)
        Izz = self._Izz(dB)
        Ixy = self._Ixy(dB)
        Ixz = self._Ixz(dB)
        Iyz = self._Iyz(dB)
        det = ( Ixx*(Iyy*Izz - Iyz**2) - Ixy*Ixz*Iyz \
            - (Ixy**2*Izz + Ixz**2*Iyy) )
        return det
    
    def _determinant_derivative(self, dB):
        # get values
        Ixx = self._Ixx(dB)
        Iyy = self._Iyy(dB)
        Izz = self._Izz(dB)
        Ixy = self._Ixy(dB)
        Ixz = self._Ixz(dB)
        Iyz = self._Iyz(dB)
        # get derivatives
        dIxx = self._dIxx(dB)
        dIyy = self._dIyy(dB)
        dIzz = self._dIzz(dB)
        dIxy = self._dIxy(dB)
        dIxz = self._dIxz(dB)
        dIyz = self._dIyz(dB)
        ddet = ( dIxx*(Iyy*Izz - Iyz**2) + Ixx*(dIyy*Izz + Iyy*dIzz \
            - 2.*Iyz*dIyz) - dIxy*Ixz*Iyz - Ixy*dIxz*Iyz - Ixy*Ixz*dIyz
            -(2.*Ixy*dIxy*Izz + Ixy**2*dIzz + 2.*Ixz*dIxz*Iyy + Ixz**2*dIyy))
        return ddet
    
    def _adjoint(self, dB):
        Ixx = self._Ixx(dB)
        Iyy = self._Iyy(dB)
        Izz = self._Izz(dB)
        Ixy = self._Ixy(dB)
        Ixz = self._Ixz(dB)
        Iyz = self._Iyz(dB)
        return [
            [ Iyy*Izz - Iyz**2, Ixy*Izz + Ixz*Iyz, Ixy*Iyz + Ixz*Iyy],
            [ Ixy*Izz + Iyz*Ixz, Ixx*Izz - Ixz**2, Ixx*Iyz + Ixy*Ixz],
            [ Ixy*Iyz + Ixz*Iyy, Ixx*Iyz + Ixz*Ixy, Ixx*Iyy - Ixy**2]
        ]
    
    def _adjoint_derivative(self, dB):
        # get values
        Ixx = self._Ixx(dB)
        Iyy = self._Iyy(dB)
        Izz = self._Izz(dB)
        Ixy = self._Ixy(dB)
        Ixz = self._Ixz(dB)
        Iyz = self._Iyz(dB)
        # get derivatives
        dIxx = self._dIxx(dB)
        dIyy = self._dIyy(dB)
        dIzz = self._dIzz(dB)
        dIxy = self._dIxy(dB)
        dIxz = self._dIxz(dB)
        dIyz = self._dIyz(dB)
        # initialize and assign
        dadj = zeros(3,3)
        dadj[0,0] = dIyy*Izz + Iyy*dIzz - 2.*Iyz*dIyz
        dadj[0,1] = dadj[1,0] = dIxy*Izz + Ixy*dIzz + dIxz*Iyz + Ixz*dIyz
        dadj[1,1] = dIxx*Izz + Ixx*dIzz - 2.*Ixz*dIxz
        dadj[0,2] = dadj[2,0] = dIxy*Iyz + Ixy*dIyz + dIxz*Iyy + Ixz*dIyy
        dadj[2,2] = dIxx*Iyy + Ixx*dIyy - 2.*Ixy*dIxy
        dadj[1,2] = dadj[2,1] = dIxx*Iyz + Ixx*dIyz + dIxy*Ixz + Ixy*dIxz
        return dadj

    def inertia_results(self, dB):
        return [self._Ixx(dB), self._Iyy(dB), self._Izz(dB),
        self._Ixy(dB), self._Ixz(dB), self._Iyz(dB)]

    def inertia_derivative_results(self, dB):
        return [self._dIxx(dB), self._dIyy(dB), self._dIzz(dB),
        self._dIxy(dB), self._dIxz(dB), self._dIyz(dB)]

    def angular_momentum_results(self):
        return [self.hx, self.hy, self.hz]

    def inertia_tensor(self, dB):
        Ixx = self._Ixx(dB)
        Iyy = self._Iyy(dB)
        Izz = self._Izz(dB)
        Ixy = self._Ixy(dB)
        Ixz = self._Ixz(dB)
        Iyz = self._Iyz(dB)
        return [
            [  Ixx, -Ixy, -Ixz],
            [ -Ixy,  Iyy, -Iyz],
            [ -Ixz, -Iyz,  Izz]
        ]

    def inverse_tensor(self, dB):
        # return ( array(self._adjoint(dB))/self._determinant(dB) ).tolist()
        adj = self._adjoint(dB)
        det = self._determinant(dB)
        return [
            [adj[0][0]/det, adj[0][1]/det, adj[0][2]/det],
            [adj[1][0]/det, adj[1][1]/det, adj[1][2]/det],
            [adj[2][0]/det, adj[2][1]/det, adj[2][2]/det]
        ]

    def inverse_tensor_derivative(self, dB):
        # get pieces
        Iinv = array(self.inverse_tensor(dB))
        det = self._determinant(dB)
        dadj = array(self._adjoint_derivative(dB))
        ddet = self._determinant_derivative(dB)

        return ( (dadj - Iinv*ddet)/det ).tolist()


class BIREAero:
    def __init__(self, inp_dir='./', **kwargs):
        self.model_coeffs_dict = json.load(open(inp_dir + 'bire_model_adj.json'))
        self.CL_coeffs = self.model_coeffs_dict["CL"]
        self.CS_coeffs = self.model_coeffs_dict["CS"]
        self.CD_coeffs = self.model_coeffs_dict["CD"]
        self.Cl_coeffs = self.model_coeffs_dict["Cell"]
        self.Cm_coeffs = self.model_coeffs_dict["Cm"]
        self.Cn_coeffs = self.model_coeffs_dict["Cn"]
        self.deriv=False

        # pull out coefficients
        self._extract_coefficients()

        # initialize thrust model
        self.Prop = Propulsion(inp_dir=kwargs.get("thrust_dir",inp_dir),**kwargs)

        # store stall model characteristics
        stall_model = self.model_coeffs_dict.get("stall_model",{})
        self.S_M = stall_model.get("blending_rate",7.0)
        self.S_ab = np.deg2rad(stall_model.get("stall_transition[deg]",45.0))

        # add in properties for compressibility
        is_rc = kwargs.get("use_rc_thrust_model",False)
        if is_rc:
            fn_props = kwargs.get("fn_props", "f16_rc_props.json") # same for F16 / BIRE
        else:
            fn_props = kwargs.get("fn_props", "f16_props.json") # same for F16 / BIRE
        self.props_dict = json.load(open(inp_dir + fn_props))
        self.geom_coeffs = self.props_dict["geometry"]
        self.S_w = self.geom_coeffs["S_w"]
        self.b_w = self.geom_coeffs["b_w"]
        self.c_w = self.geom_coeffs["c_w"]
        self.l_h = self.geom_coeffs["l_h"]
        self.Lam_w = self.geom_coeffs["Lam_w"]
        self.RA_w = self.geom_coeffs["RA_w"]
        self.Lam_v = self.geom_coeffs["Lam_v"]
        self.RA_v = self.geom_coeffs["RA_v"]
        self.Lam_h = self.geom_coeffs["Lam_h"]
        self.RA_h = self.geom_coeffs["RA_h"]

    def _extract_coefficients(self):

        # CL
        # CL0
        Cdict = self.CL_coeffs["CL_0"]
        [self.CL_0_A, self.CL_0_w, self.CL_0_p, self.CL_0_z, self.CL_0_s,
        self.CL_0_d] = [Cdict[c] for c in Cdict]
        # CL,beta
        Cdict = self.CL_coeffs["CL_alpha"]
        [self.CL_a_A, self.CL_a_w, self.CL_a_p, self.CL_a_z, self.CL_a_s,
        self.CL_a_d] = [Cdict[c] for c in Cdict]
        # CL,beta
        Cdict = self.CL_coeffs["CL_beta"]
        [self.CL_b_A, self.CL_b_w, self.CL_b_p, self.CL_b_z, self.CL_b_s,
        self.CL_b_d] = [Cdict[c] for c in Cdict]
        # CL,pbar
        Cdict = self.CL_coeffs["CL_pbar"]
        [self.CL_p_A, self.CL_p_w, self.CL_p_p, self.CL_p_z, self.CL_p_s,
        self.CL_p_d] = [Cdict[c] for c in Cdict]
        # CL,qbar
        Cdict = self.CL_coeffs["CL_qbar"]
        [self.CL_q_A, self.CL_q_w, self.CL_q_p, self.CL_q_z, self.CL_q_s,
        self.CL_q_d] = [Cdict[c] for c in Cdict]
        # CL,rbar
        Cdict = self.CL_coeffs["CL_rbar"]
        [self.CL_r_A, self.CL_r_w, self.CL_r_p, self.CL_r_z, self.CL_r_s,
        self.CL_r_d] = [Cdict[c] for c in Cdict]
        # CL,da
        Cdict = self.CL_coeffs["CL_da"]
        [self.CL_da_A, self.CL_da_w, self.CL_da_p, self.CL_da_z, self.CL_da_s,
        self.CL_da_d] = [Cdict[c] for c in Cdict]
        # CL,de
        Cdict = self.CL_coeffs["CL_de"]
        [self.CL_de_A, self.CL_de_w, self.CL_de_p, self.CL_de_z, self.CL_de_s,
        self.CL_de_d] = [Cdict[c] for c in Cdict]

        # CS
        # CS0
        Cdict = self.CS_coeffs["CS_0"]
        [self.CS_0_A, self.CS_0_w, self.CS_0_p, self.CS_0_z, self.CS_0_s,
        self.CS_0_d] = [Cdict[c] for c in Cdict]
        # CS,alpha
        Cdict = self.CS_coeffs["CS_alpha"]
        [self.CS_a_A, self.CS_a_w, self.CS_a_p, self.CS_a_z, self.CS_a_s,
        self.CS_a_d] = [Cdict[c] for c in Cdict]
        # CS,beta
        Cdict = self.CS_coeffs["CS_beta"]
        [self.CS_b_A, self.CS_b_w, self.CS_b_p, self.CS_b_z, self.CS_b_s,
        self.CS_b_d] = [Cdict[c] for c in Cdict]
        # CS,pbar
        Cdict = self.CS_coeffs["CS_pbar"]
        [self.CS_p_A, self.CS_p_w, self.CS_p_p, self.CS_p_z, self.CS_p_s,
        self.CS_p_d] = [Cdict[c] for c in Cdict]
        # CS,Lpbar
        Cdict = self.CS_coeffs["CS_Lpbar"]
        [self.CS_Lp_A, self.CS_Lp_w, self.CS_Lp_p, self.CS_Lp_z, self.CS_Lp_s,
        self.CS_Lp_d] = [Cdict[c] for c in Cdict]
        # CS,qbar
        Cdict = self.CS_coeffs["CS_qbar"]
        [self.CS_q_A, self.CS_q_w, self.CS_q_p, self.CS_q_z, self.CS_q_s,
        self.CS_q_d] = [Cdict[c] for c in Cdict]
        # CS,rbar
        Cdict = self.CS_coeffs["CS_rbar"]
        [self.CS_r_A, self.CS_r_w, self.CS_r_p, self.CS_r_z, self.CS_r_s,
        self.CS_r_d] = [Cdict[c] for c in Cdict]
        # CS,da
        Cdict = self.CS_coeffs["CS_da"]
        [self.CS_da_A, self.CS_da_w, self.CS_da_p, self.CS_da_z, self.CS_da_s,
        self.CS_da_d] = [Cdict[c] for c in Cdict]
        # CS,de
        Cdict = self.CS_coeffs["CS_de"]
        [self.CS_de_A, self.CS_de_w, self.CS_de_p, self.CS_de_z, self.CS_de_s,
        self.CS_de_d] = [Cdict[c] for c in Cdict]

        # CD
        # CD0
        Cdict = self.CD_coeffs["CD_0"]
        [self.CD_0_A, self.CD_0_w, self.CD_0_p, self.CD_0_z, self.CD_0_s,
        self.CD_0_d] = [Cdict[c] for c in Cdict]
        # CDL
        Cdict = self.CD_coeffs["CD_L"]
        [self.CD_L_A, self.CD_L_w, self.CD_L_p, self.CD_L_z, self.CD_L_s,
        self.CD_L_d] = [Cdict[c] for c in Cdict]
        # CDL2
        Cdict = self.CD_coeffs["CD_L2"]
        [self.CD_L2_A, self.CD_L2_w, self.CD_L2_p, self.CD_L2_z, self.CD_L2_s,
        self.CD_L2_d] = [Cdict[c] for c in Cdict]
        # CDS
        Cdict = self.CD_coeffs["CD_S"]
        [self.CD_S_A, self.CD_S_w, self.CD_S_p, self.CD_S_z, self.CD_S_s,
        self.CD_S_d] = [Cdict[c] for c in Cdict]
        # CDS2
        Cdict = self.CD_coeffs["CD_S2"]
        [self.CD_S2_A, self.CD_S2_w, self.CD_S2_p, self.CD_S2_z, self.CD_S2_s,
        self.CD_S2_d] = [Cdict[c] for c in Cdict]
        # CD,pbar
        Cdict = self.CD_coeffs["CD_pbar"]
        [self.CD_p_A, self.CD_p_w, self.CD_p_p, self.CD_p_z, self.CD_p_s,
        self.CD_p_d] = [Cdict[c] for c in Cdict]
        # CD,Spbar
        Cdict = self.CD_coeffs["CD_Spbar"]
        [self.CD_Sp_A, self.CD_Sp_w, self.CD_Sp_p, self.CD_Sp_z, self.CD_Sp_s,
        self.CD_Sp_d] = [Cdict[c] for c in Cdict]
        # CD,qbar
        Cdict = self.CD_coeffs["CD_qbar"]
        [self.CD_q_A, self.CD_q_w, self.CD_q_p, self.CD_q_z, self.CD_q_s,
        self.CD_q_d] = [Cdict[c] for c in Cdict]
        # CD,Lqbar
        Cdict = self.CD_coeffs["CD_Lqbar"]
        [self.CD_Lq_A, self.CD_Lq_w, self.CD_Lq_p, self.CD_Lq_z, self.CD_Lq_s,
        self.CD_Lq_d] = [Cdict[c] for c in Cdict]
        # CD,L2qbar
        Cdict = self.CD_coeffs["CD_L2qbar"]
        [self.CD_L2q_A, self.CD_L2q_w, self.CD_L2q_p, self.CD_L2q_z, self.CD_L2q_s,
        self.CD_L2q_d] = [Cdict[c] for c in Cdict]
        # CD,rbar
        Cdict = self.CD_coeffs["CD_rbar"]
        [self.CD_r_A, self.CD_r_w, self.CD_r_p, self.CD_r_z, self.CD_r_s,
        self.CD_r_d] = [Cdict[c] for c in Cdict]
        # CD,Srbar
        Cdict = self.CD_coeffs["CD_Srbar"]
        [self.CD_Sr_A, self.CD_Sr_w, self.CD_Sr_p, self.CD_Sr_z, self.CD_Sr_s,
        self.CD_Sr_d] = [Cdict[c] for c in Cdict]
        # CD,da
        Cdict = self.CD_coeffs["CD_da"]
        [self.CD_da_A, self.CD_da_w, self.CD_da_p, self.CD_da_z, self.CD_da_s,
        self.CD_da_d] = [Cdict[c] for c in Cdict]
        # CD,Sda
        Cdict = self.CD_coeffs["CD_Sda"]
        [self.CD_Sda_A, self.CD_Sda_w, self.CD_Sda_p, self.CD_Sda_z, self.CD_Sda_s,
        self.CD_Sda_d] = [Cdict[c] for c in Cdict]
        # CD,de
        Cdict = self.CD_coeffs["CD_de"]
        [self.CD_de_A, self.CD_de_w, self.CD_de_p, self.CD_de_z, self.CD_de_s,
        self.CD_de_d] = [Cdict[c] for c in Cdict]
        # CD,Lde
        Cdict = self.CD_coeffs["CD_Lde"]
        [self.CD_Lde_A, self.CD_Lde_w, self.CD_Lde_p, self.CD_Lde_z, self.CD_Lde_s,
        self.CD_Lde_d] = [Cdict[c] for c in Cdict]
        # CD,de2
        Cdict = self.CD_coeffs["CD_de2"]
        [self.CD_de2_A, self.CD_de2_w, self.CD_de2_p, self.CD_de2_z, self.CD_de2_s,
        self.CD_de2_d] = [Cdict[c] for c in Cdict]

        # Cl
        # Cl0
        Cdict = self.Cl_coeffs["Cl_0"]
        [self.Cl_0_A, self.Cl_0_w, self.Cl_0_p, self.Cl_0_z, self.Cl_0_s,
        self.Cl_0_d] = [Cdict[c] for c in Cdict]
        # Cl,alpha
        Cdict = self.Cl_coeffs["Cl_alpha"]
        [self.Cl_a_A, self.Cl_a_w, self.Cl_a_p, self.Cl_a_z, self.Cl_a_s,
        self.Cl_a_d] = [Cdict[c] for c in Cdict]
        # Cl,beta
        Cdict = self.Cl_coeffs["Cl_beta"]
        [self.Cl_b_A, self.Cl_b_w, self.Cl_b_p, self.Cl_b_z, self.Cl_b_s,
        self.Cl_b_d] = [Cdict[c] for c in Cdict]
        # Cl,pbar
        Cdict = self.Cl_coeffs["Cl_pbar"]
        [self.Cl_p_A, self.Cl_p_w, self.Cl_p_p, self.Cl_p_z, self.Cl_p_s,
        self.Cl_p_d] = [Cdict[c] for c in Cdict]
        # Cl,qbar
        Cdict = self.Cl_coeffs["Cl_qbar"]
        [self.Cl_q_A, self.Cl_q_w, self.Cl_q_p, self.Cl_q_z, self.Cl_q_s,
        self.Cl_q_d] = [Cdict[c] for c in Cdict]
        # Cl,rbar
        Cdict = self.Cl_coeffs["Cl_rbar"]
        [self.Cl_r_A, self.Cl_r_w, self.Cl_r_p, self.Cl_r_z, self.Cl_r_s,
        self.Cl_r_d] = [Cdict[c] for c in Cdict]
        # Cl,Lpbar
        Cdict = self.Cl_coeffs["Cl_Lrbar"]
        [self.Cl_Lr_A, self.Cl_Lr_w, self.Cl_Lr_p, self.Cl_Lr_z, self.Cl_Lr_s,
        self.Cl_Lr_d] = [Cdict[c] for c in Cdict]
        # Cl,da
        Cdict = self.Cl_coeffs["Cl_da"]
        [self.Cl_da_A, self.Cl_da_w, self.Cl_da_p, self.Cl_da_z, self.Cl_da_s,
        self.Cl_da_d] = [Cdict[c] for c in Cdict]
        # Cl,de
        Cdict = self.Cl_coeffs["Cl_de"]
        [self.Cl_de_A, self.Cl_de_w, self.Cl_de_p, self.Cl_de_z, self.Cl_de_s,
        self.Cl_de_d] = [Cdict[c] for c in Cdict]

        # Cm
        # Cm0
        Cdict = self.Cm_coeffs["Cm_0"]
        [self.Cm_0_A, self.Cm_0_w, self.Cm_0_p, self.Cm_0_z, self.Cm_0_s,
        self.Cm_0_d] = [Cdict[c] for c in Cdict]
        # Cm,alpha
        Cdict = self.Cm_coeffs["Cm_alpha"]
        [self.Cm_a_A, self.Cm_a_w, self.Cm_a_p, self.Cm_a_z, self.Cm_a_s,
        self.Cm_a_d] = [Cdict[c] for c in Cdict]
        # Cm,beta
        Cdict = self.Cm_coeffs["Cm_beta"]
        [self.Cm_b_A, self.Cm_b_w, self.Cm_b_p, self.Cm_b_z, self.Cm_b_s,
        self.Cm_b_d] = [Cdict[c] for c in Cdict]
        # Cm,pbar
        Cdict = self.Cm_coeffs["Cm_pbar"]
        [self.Cm_p_A, self.Cm_p_w, self.Cm_p_p, self.Cm_p_z, self.Cm_p_s,
        self.Cm_p_d] = [Cdict[c] for c in Cdict]
        # Cm,qbar
        Cdict = self.Cm_coeffs["Cm_qbar"]
        [self.Cm_q_A, self.Cm_q_w, self.Cm_q_p, self.Cm_q_z, self.Cm_q_s,
        self.Cm_q_d] = [Cdict[c] for c in Cdict]
        # Cm,rbar
        Cdict = self.Cm_coeffs["Cm_rbar"]
        [self.Cm_r_A, self.Cm_r_w, self.Cm_r_p, self.Cm_r_z, self.Cm_r_s,
        self.Cm_r_d] = [Cdict[c] for c in Cdict]
        # Cm,da
        Cdict = self.Cm_coeffs["Cm_da"]
        [self.Cm_da_A, self.Cm_da_w, self.Cm_da_p, self.Cm_da_z, self.Cm_da_s,
        self.Cm_da_d] = [Cdict[c] for c in Cdict]
        # Cm,de
        Cdict = self.Cm_coeffs["Cm_de"]
        [self.Cm_de_A, self.Cm_de_w, self.Cm_de_p, self.Cm_de_z, self.Cm_de_s,
        self.Cm_de_d] = [Cdict[c] for c in Cdict]

        # Cn
        # Cn0
        Cdict = self.Cn_coeffs["Cn_0"]
        [self.Cn_0_A, self.Cn_0_w, self.Cn_0_p, self.Cn_0_z, self.Cn_0_s,
        self.Cn_0_d] = [Cdict[c] for c in Cdict]
        # Cn,alpha
        Cdict = self.Cn_coeffs["Cn_alpha"]
        [self.Cn_a_A, self.Cn_a_w, self.Cn_a_p, self.Cn_a_z, self.Cn_a_s,
        self.Cn_a_d] = [Cdict[c] for c in Cdict]
        # Cn,beta
        Cdict = self.Cn_coeffs["Cn_beta"]
        [self.Cn_b_A, self.Cn_b_w, self.Cn_b_p, self.Cn_b_z, self.Cn_b_s,
        self.Cn_b_d] = [Cdict[c] for c in Cdict]
        # Cn,pbar
        Cdict = self.Cn_coeffs["Cn_pbar"]
        [self.Cn_p_A, self.Cn_p_w, self.Cn_p_p, self.Cn_p_z, self.Cn_p_s,
        self.Cn_p_d] = [Cdict[c] for c in Cdict]
        # Cn,Lpbar
        Cdict = self.Cn_coeffs["Cn_Lpbar"]
        [self.Cn_Lp_A, self.Cn_Lp_w, self.Cn_Lp_p, self.Cn_Lp_z, self.Cn_Lp_s,
        self.Cn_Lp_d] = [Cdict[c] for c in Cdict]
        # Cn,qbar
        Cdict = self.Cn_coeffs["Cn_qbar"]
        [self.Cn_q_A, self.Cn_q_w, self.Cn_q_p, self.Cn_q_z, self.Cn_q_s,
        self.Cn_q_d] = [Cdict[c] for c in Cdict]
        # Cn,rbar
        Cdict = self.Cn_coeffs["Cn_rbar"]
        [self.Cn_r_A, self.Cn_r_w, self.Cn_r_p, self.Cn_r_z, self.Cn_r_s,
        self.Cn_r_d] = [Cdict[c] for c in Cdict]
        # Cn,da
        Cdict = self.Cn_coeffs["Cn_da"]
        [self.Cn_da_A, self.Cn_da_w, self.Cn_da_p, self.Cn_da_z, self.Cn_da_s,
        self.Cn_da_d] = [Cdict[c] for c in Cdict]
        # Cn,Lda
        Cdict = self.Cn_coeffs["Cn_Lda"]
        [self.Cn_Lda_A, self.Cn_Lda_w, self.Cn_Lda_p, self.Cn_Lda_z, self.Cn_Lda_s,
        self.Cn_Lda_d] = [Cdict[c] for c in Cdict]
        # Cn,de
        Cdict = self.Cn_coeffs["Cn_de"]
        [self.Cn_de_A, self.Cn_de_w, self.Cn_de_p, self.Cn_de_z, self.Cn_de_s,
        self.Cn_de_d] = [Cdict[c] for c in Cdict]

        return

    def evaluate_coeffs(self, d_B):
        self.CL0 = self._CL0(d_B)
        self.CLa = self._CL_alpha(d_B)
        self.CLb = self._CL_beta(d_B)
        self.CLp = self._CL_pbar(d_B)
        self.CLq = self._CL_qbar(d_B)
        self.CLr = self._CL_rbar(d_B)
        self.CLda = self._CL_da(d_B)
        self.CLde = self._CL_de(d_B)

        self.CS0 = self._CS0(d_B)
        self.CSa = self._CS_alpha(d_B)
        self.CSb = self._CS_beta(d_B)
        self.CSp = self._CS_pbar(d_B)
        self.CSLp = self._CS_Lpbar(d_B)
        self.CSq = self._CS_qbar(d_B)
        self.CSr = self._CS_rbar(d_B)
        self.CSda = self._CS_da(d_B)
        self.CSde = self._CS_de(d_B)

        self.CD0 = self._CD0(d_B)
        self.CDL = self._CD_L(d_B)
        self.CDL2 = self._CD_L2(d_B)
        self.CDS = self._CD_S(d_B)
        self.CDS2 = self._CD_S2(d_B)
        self.CDp = self._CD_pbar(d_B)
        self.CDSp = self._CD_Spbar(d_B)
        self.CDq = self._CD_qbar(d_B)
        self.CDLq = self._CD_Lqbar(d_B)
        self.CDL2q = self._CD_L2qbar(d_B)
        self.CDr = self._CD_rbar(d_B)
        self.CDSr = self._CD_Srbar(d_B)
        self.CDda = self._CD_da(d_B)
        self.CDSda = self._CD_Sda(d_B)
        self.CDde = self._CD_de(d_B)
        self.CDLde = self._CD_Lde(d_B)
        self.CDde2 = self._CD_de2(d_B)

        self.Cl0 = self._Cl0(d_B)
        self.Cla = self._Cl_alpha(d_B)
        self.Clb = self._Cl_beta(d_B)
        self.Clp = self._Cl_pbar(d_B)
        self.Clq = self._Cl_qbar(d_B)
        self.Clr = self._Cl_rbar(d_B)
        self.ClLr = self._Cl_Lrbar(d_B)
        self.Clda = self._Cl_da(d_B)
        self.Clde = self._Cl_de(d_B)

        self.Cm0 = self._Cm0(d_B)
        self.Cma = self._Cm_alpha(d_B)
        self.Cmb = self._Cm_beta(d_B)
        self.Cmp = self._Cm_pbar(d_B)
        self.Cmq = self._Cm_qbar(d_B)
        self.Cmr = self._Cm_rbar(d_B)
        self.Cmda = self._Cm_da(d_B)
        self.Cmde = self._Cm_de(d_B)

        self.Cn0 = self._Cn0(d_B)
        self.Cna = self._Cn_alpha(d_B)
        self.Cnb = self._Cn_beta(d_B)
        self.Cnp = self._Cn_pbar(d_B)
        self.CnLp = self._Cn_Lpbar(d_B)
        self.Cnq = self._Cn_qbar(d_B)
        self.Cnr = self._Cn_rbar(d_B)
        self.Cnda = self._Cn_da(d_B)
        self.CnLda = self._Cn_Lda(d_B)
        self.Cnde = self._Cn_de(d_B)

    def evaluate_derivatives(self, d_B):
        self.deriv = True
        self.dCL0 = self._CL0(d_B)
        self.dCLa = self._CL_alpha(d_B)
        self.dCLb = self._CL_beta(d_B)
        self.dCLp = self._CL_pbar(d_B)
        self.dCLq = self._CL_qbar(d_B)
        self.dCLr = self._CL_rbar(d_B)
        self.dCLda = self._CL_da(d_B)
        self.dCLde = self._CL_de(d_B)

        self.dCS0 = self._CS0(d_B)
        self.dCSa = self._CS_alpha(d_B)
        self.dCSb = self._CS_beta(d_B)
        self.dCSp = self._CS_pbar(d_B)
        self.dCSLp = self._CS_Lpbar(d_B)
        self.dCSq = self._CS_qbar(d_B)
        self.dCSr = self._CS_rbar(d_B)
        self.dCSda = self._CS_da(d_B)
        self.dCSde = self._CS_de(d_B)

        self.dCD0 = self._CD0(d_B)
        self.dCDL = self._CD_L(d_B)
        self.dCDL2 = self._CD_L2(d_B)
        self.dCDS = self._CD_S(d_B)
        self.dCDS2 = self._CD_S2(d_B)
        self.dCDp = self._CD_pbar(d_B)
        self.dCDSp = self._CD_Spbar(d_B)
        self.dCDq = self._CD_qbar(d_B)
        self.dCDLq = self._CD_Lqbar(d_B)
        self.dCDL2q = self._CD_L2qbar(d_B)
        self.dCDr = self._CD_rbar(d_B)
        self.dCDSr = self._CD_Srbar(d_B)
        self.dCDda = self._CD_da(d_B)
        self.dCDSda = self._CD_Sda(d_B)
        self.dCDde = self._CD_de(d_B)
        self.dCDLde = self._CD_Lde(d_B)
        self.dCDde2 = self._CD_de2(d_B)

        self.dCl0 = self._Cl0(d_B)
        self.dCla = self._Cl_alpha(d_B)
        self.dClb = self._Cl_beta(d_B)
        self.dClp = self._Cl_pbar(d_B)
        self.dClq = self._Cl_qbar(d_B)
        self.dClr = self._Cl_rbar(d_B)
        self.dClLr = self._Cl_Lrbar(d_B)
        self.dClda = self._Cl_da(d_B)
        self.dClde = self._Cl_de(d_B)

        self.dCm0 = self._Cm0(d_B)
        self.dCma = self._Cm_alpha(d_B)
        self.dCmb = self._Cm_beta(d_B)
        self.dCmp = self._Cm_pbar(d_B)
        self.dCmq = self._Cm_qbar(d_B)
        self.dCmr = self._Cm_rbar(d_B)
        self.dCmda = self._Cm_da(d_B)
        self.dCmde = self._Cm_de(d_B)

        self.dCn0 = self._Cn0(d_B)
        self.dCna = self._Cn_alpha(d_B)
        self.dCnb = self._Cn_beta(d_B)
        self.dCnp = self._Cn_pbar(d_B)
        self.dCnLp = self._Cn_Lpbar(d_B)
        self.dCnq = self._Cn_qbar(d_B)
        self.dCnr = self._Cn_rbar(d_B)
        self.dCnda = self._Cn_da(d_B)
        self.dCnLda = self._Cn_Lda(d_B)
        self.dCnde = self._Cn_de(d_B)
        self.deriv = False

    def _make_double_derivative_model(self):

        # CL
        self.CL_0_A = -self.CL_0_A*self.CL_0_w**2; self.CL_0_z = self.CL_0_d = 0.0
        self.CL_a_A = -self.CL_a_A*self.CL_a_w**2; self.CL_a_z = self.CL_a_d = 0.0
        self.CL_b_A = -self.CL_b_A*self.CL_b_w**2; self.CL_b_z = self.CL_b_d = 0.0
        self.CL_p_A = -self.CL_p_A*self.CL_p_w**2; self.CL_p_z = self.CL_p_d = 0.0
        self.CL_q_A = -self.CL_q_A*self.CL_q_w**2; self.CL_q_z = self.CL_q_d = 0.0
        self.CL_r_A = -self.CL_r_A*self.CL_r_w**2; self.CL_r_z = self.CL_r_d = 0.0
        self.CL_da_A = -self.CL_da_A*self.CL_da_w**2; self.CL_da_z = self.CL_da_d = 0.0
        self.CL_de_A = -self.CL_de_A*self.CL_de_w**2; self.CL_de_z = self.CL_de_d = 0.0

        # CS
        self.CS_0_A = -self.CS_0_A*self.CS_0_w**2; self.CS_0_z = self.CS_0_d = 0.0
        self.CS_a_A = -self.CS_a_A*self.CS_a_w**2; self.CS_a_z = self.CS_a_d = 0.0
        self.CS_b_A = -self.CS_b_A*self.CS_b_w**2; self.CS_b_z = self.CS_b_d = 0.0
        self.CS_p_A = -self.CS_p_A*self.CS_p_w**2; self.CS_p_z = self.CS_p_d = 0.0
        self.CS_Lp_A = -self.CS_Lp_A*self.CS_Lp_w**2; self.CS_Lp_z = self.CS_Lp_d = 0.0
        self.CS_q_A = -self.CS_q_A*self.CS_q_w**2; self.CS_q_z = self.CS_q_d = 0.0
        self.CS_r_A = -self.CS_r_A*self.CS_r_w**2; self.CS_r_z = self.CS_r_d = 0.0
        self.CS_da_A = -self.CS_da_A*self.CS_da_w**2; self.CS_da_z = self.CS_da_d = 0.0
        self.CS_de_A = -self.CS_de_A*self.CS_de_w**2; self.CS_de_z = self.CS_de_d = 0.0

        # CD
        self.CD_0_A = -self.CD_0_A*self.CD_0_w**2; self.CD_0_z = self.CD_0_d = 0.0
        self.CD_L_A = -self.CD_L_A*self.CD_L_w**2; self.CD_L_z = self.CD_L_d = 0.0
        self.CD_L2_A = -self.CD_L2_A*self.CD_L2_w**2; self.CD_L2_z = self.CD_L2_d = 0.0
        self.CD_S_A = -self.CD_S_A*self.CD_S_w**2; self.CD_S_z = self.CD_S_d = 0.0
        self.CD_S2_A = -self.CD_S2_A*self.CD_S2_w**2; self.CD_S2_z = self.CD_S2_d = 0.0
        self.CD_p_A = -self.CD_p_A*self.CD_p_w**2; self.CD_p_z = self.CD_p_d = 0.0
        self.CD_Sp_A = -self.CD_Sp_A*self.CD_Sp_w**2; self.CD_Sp_z = self.CD_Sp_d = 0.0
        self.CD_q_A = -self.CD_q_A*self.CD_q_w**2; self.CD_q_z = self.CD_q_d = 0.0
        self.CD_Lq_A = -self.CD_Lq_A*self.CD_Lq_w**2; self.CD_Lq_z = self.CD_Lq_d = 0.0
        self.CD_L2q_A = -self.CD_L2q_A*self.CD_L2q_w**2; self.CD_L2q_z = self.CD_L2q_d = 0.0
        self.CD_r_A = -self.CD_r_A*self.CD_r_w**2; self.CD_r_z = self.CD_r_d = 0.0
        self.CD_Sr_A = -self.CD_Sr_A*self.CD_Sr_w**2; self.CD_Sr_z = self.CD_Sr_d = 0.0
        self.CD_da_A = -self.CD_da_A*self.CD_da_w**2; self.CD_da_z = self.CD_da_d = 0.0
        self.CD_Sda_A = -self.CD_Sda_A*self.CD_Sda_w**2; self.CD_Sda_z = self.CD_Sda_d = 0.0
        self.CD_de_A = -self.CD_de_A*self.CD_de_w**2; self.CD_de_z = self.CD_de_d = 0.0
        self.CD_Lde_A = -self.CD_Lde_A*self.CD_Lde_w**2; self.CD_Lde_z = self.CD_Lde_d = 0.0
        self.CD_de2_A = -self.CD_de2_A*self.CD_de2_w**2; self.CD_de2_z = self.CD_de2_d = 0.0

        # Cl
        self.Cl_0_A = -self.Cl_0_A*self.Cl_0_w**2; self.Cl_0_z = self.Cl_0_d = 0.0
        self.Cl_a_A = -self.Cl_a_A*self.Cl_a_w**2; self.Cl_a_z = self.Cl_a_d = 0.0
        self.Cl_b_A = -self.Cl_b_A*self.Cl_b_w**2; self.Cl_b_z = self.Cl_b_d = 0.0
        self.Cl_p_A = -self.Cl_p_A*self.Cl_p_w**2; self.Cl_p_z = self.Cl_p_d = 0.0
        self.Cl_q_A = -self.Cl_q_A*self.Cl_q_w**2; self.Cl_q_z = self.Cl_q_d = 0.0
        self.Cl_r_A = -self.Cl_r_A*self.Cl_r_w**2; self.Cl_r_z = self.Cl_r_d = 0.0
        self.Cl_Lr_A = -self.Cl_Lr_A*self.Cl_Lr_w**2; self.Cl_Lr_z = self.Cl_Lr_d = 0.0
        self.Cl_da_A = -self.Cl_da_A*self.Cl_da_w**2; self.Cl_da_z = self.Cl_da_d = 0.0
        self.Cl_de_A = -self.Cl_de_A*self.Cl_de_w**2; self.Cl_de_z = self.Cl_de_d = 0.0

        # Cm
        self.Cm_0_A = -self.Cm_0_A*self.Cm_0_w**2; self.Cm_0_z = self.Cm_0_d = 0.0
        self.Cm_a_A = -self.Cm_a_A*self.Cm_a_w**2; self.Cm_a_z = self.Cm_a_d = 0.0
        self.Cm_b_A = -self.Cm_b_A*self.Cm_b_w**2; self.Cm_b_z = self.Cm_b_d = 0.0
        self.Cm_p_A = -self.Cm_p_A*self.Cm_p_w**2; self.Cm_p_z = self.Cm_p_d = 0.0
        self.Cm_q_A = -self.Cm_q_A*self.Cm_q_w**2; self.Cm_q_z = self.Cm_q_d = 0.0
        self.Cm_r_A = -self.Cm_r_A*self.Cm_r_w**2; self.Cm_r_z = self.Cm_r_d = 0.0
        self.Cm_da_A = -self.Cm_da_A*self.Cm_da_w**2; self.Cm_da_z = self.Cm_da_d = 0.0
        self.Cm_de_A = -self.Cm_de_A*self.Cm_de_w**2; self.Cm_de_z = self.Cm_de_d = 0.0

        # Cn
        self.Cn_0_A = -self.Cn_0_A*self.Cn_0_w**2; self.Cn_0_z = self.Cn_0_d = 0.0
        self.Cn_a_A = -self.Cn_a_A*self.Cn_a_w**2; self.Cn_a_z = self.Cn_a_d = 0.0
        self.Cn_b_A = -self.Cn_b_A*self.Cn_b_w**2; self.Cn_b_z = self.Cn_b_d = 0.0
        self.Cn_p_A = -self.Cn_p_A*self.Cn_p_w**2; self.Cn_p_z = self.Cn_p_d = 0.0
        self.Cn_Lp_A = -self.Cn_Lp_A*self.Cn_Lp_w**2; self.Cn_Lp_z = self.Cn_Lp_d = 0.0
        self.Cn_q_A = -self.Cn_q_A*self.Cn_q_w**2; self.Cn_q_z = self.Cn_q_d = 0.0
        self.Cn_r_A = -self.Cn_r_A*self.Cn_r_w**2; self.Cn_r_z = self.Cn_r_d = 0.0
        self.Cn_da_A = -self.Cn_da_A*self.Cn_da_w**2; self.Cn_da_z = self.Cn_da_d = 0.0
        self.Cn_Lda_A = -self.Cn_Lda_A*self.Cn_Lda_w**2; self.Cn_Lda_z = self.Cn_Lda_d = 0.0
        self.Cn_de_A = -self.Cn_de_A*self.Cn_de_w**2; self.Cn_de_z = self.Cn_de_d = 0.0

        return

    def _CL0(self, d_B):
        if not self.deriv:
            return self.CL_0_A*sin(self.CL_0_w*d_B + self.CL_0_p) + \
                self.CL_0_z + self.CL_0_d
        else:
            return self.CL_0_A*self.CL_0_w*cos(self.CL_0_w*d_B + self.CL_0_p)

    def _CL_alpha(self, d_B):
        if not self.deriv:
            return self.CL_a_A*sin(self.CL_a_w*d_B + self.CL_a_p) + \
                self.CL_a_z + self.CL_a_d
        else:
            return self.CL_a_A*self.CL_a_w*cos(self.CL_a_w*d_B + self.CL_a_p)

    def _CL_beta(self, d_B):
        if not self.deriv:
            return self.CL_b_A*sin(self.CL_b_w*d_B + self.CL_b_p) + \
                self.CL_b_z + self.CL_b_d
        else:
            return self.CL_b_A*self.CL_b_w*cos(self.CL_b_w*d_B + self.CL_b_p)

    def _CL_pbar(self, d_B):
        if not self.deriv:
            return self.CL_p_A*sin(self.CL_p_w*d_B + self.CL_p_p) + \
                self.CL_p_z + self.CL_p_d
        else:
            return self.CL_p_A*self.CL_p_w*cos(self.CL_p_w*d_B + self.CL_p_p)

    def _CL_qbar(self, d_B, deriv=False):
        if not self.deriv:
            return self.CL_q_A*sin(self.CL_q_w*d_B + self.CL_q_p) + \
                self.CL_q_z + self.CL_q_d
        else:
            return self.CL_q_A*self.CL_q_w*cos(self.CL_q_w*d_B + self.CL_q_p)

    def _CL_rbar(self, d_B):
        if not self.deriv:
            return self.CL_r_A*sin(self.CL_r_w*d_B + self.CL_r_p) + \
                self.CL_r_z + self.CL_r_d
        else:
            return self.CL_r_A*self.CL_r_w*cos(self.CL_r_w*d_B + self.CL_r_p)

    def _CL_da(self, d_B):
        if not self.deriv:
            return self.CL_da_A*sin(self.CL_da_w*d_B + self.CL_da_p) + \
                self.CL_da_z + self.CL_da_d
        else:
            return self.CL_da_A*self.CL_da_w*cos(self.CL_da_w*d_B + self.CL_da_p)

    def _CL_de(self, d_B):
        if not self.deriv:
            return self.CL_de_A*sin(self.CL_de_w*d_B + self.CL_de_p) + \
                self.CL_de_z + self.CL_de_d
        else:
            return self.CL_de_A*self.CL_de_w*cos(self.CL_de_w*d_B + self.CL_de_p)

    def _CS0(self, d_B):
        if not self.deriv:
            return self.CS_0_A*sin(self.CS_0_w*d_B + self.CS_0_p) + \
                self.CS_0_z + self.CS_0_d
        else:
            return self.CS_0_A*self.CS_0_w*cos(self.CS_0_w*d_B + self.CS_0_p)

    def _CS_alpha(self, d_B):
        if not self.deriv:
            return self.CS_a_A*sin(self.CS_a_w*d_B + self.CS_a_p) + \
                self.CS_a_z + self.CS_a_d
        else:
            return self.CS_a_A*self.CS_a_w*cos(self.CS_a_w*d_B + self.CS_a_p)

    def _CS_beta(self, d_B):
        if not self.deriv:
            return self.CS_b_A*sin(self.CS_b_w*d_B + self.CS_b_p) + \
                self.CS_b_z + self.CS_b_d
        else:
            return self.CS_b_A*self.CS_b_w*cos(self.CS_b_w*d_B + self.CS_b_p)

    def _CS_pbar(self, d_B):
        if not self.deriv:
            return self.CS_p_A*sin(self.CS_p_w*d_B + self.CS_p_p) + \
                self.CS_p_z + self.CS_p_d
        else:
            return self.CS_p_A*self.CS_p_w*cos(self.CS_p_w*d_B + self.CS_p_p)

    def _CS_Lpbar(self, d_B):
        if not self.deriv:
            return self.CS_Lp_A*sin(self.CS_Lp_w*d_B + self.CS_Lp_p) + \
                self.CS_Lp_z + self.CS_Lp_d
        else:
            return self.CS_Lp_A*self.CS_Lp_w*cos(self.CS_Lp_w*d_B + self.CS_Lp_p)

    def _CS_qbar(self, d_B):
        if not self.deriv:
            return self.CS_q_A*sin(self.CS_q_w*d_B + self.CS_q_p) + \
                self.CS_q_z + self.CS_q_d
        else:
            return self.CS_q_A*self.CS_q_w*cos(self.CS_q_w*d_B + self.CS_q_p)

    def _CS_rbar(self, d_B):
        if not self.deriv:
            return self.CS_r_A*sin(self.CS_r_w*d_B + self.CS_r_p) + \
                self.CS_r_z + self.CS_r_d
        else:
            return self.CS_r_A*self.CS_r_w*cos(self.CS_r_w*d_B + self.CS_r_p)

    def _CS_da(self, d_B):
        if not self.deriv:
            return self.CS_da_A*sin(self.CS_da_w*d_B + self.CS_da_p) + \
                self.CS_da_z + self.CS_da_d
        else:
            return self.CS_da_A*self.CS_da_w*cos(self.CS_da_w*d_B + self.CS_da_p)

    def _CS_de(self, d_B):
        if not self.deriv:
            return self.CS_de_A*sin(self.CS_de_w*d_B + self.CS_de_p) + \
                self.CS_de_z + self.CS_de_d
        else:
            return self.CS_de_A*self.CS_de_w*cos(self.CS_de_w*d_B + self.CS_de_p)

    def _CD0(self, d_B):
        if not self.deriv:
            return self.CD_0_A*sin(self.CD_0_w*d_B + self.CD_0_p) + \
                self.CD_0_z + self.CD_0_d
        else:
            return self.CD_0_A*self.CD_0_w*cos(self.CD_0_w*d_B + self.CD_0_p)

    def _CD_L(self, d_B):
        if not self.deriv:
            return self.CD_L_A*sin(self.CD_L_w*d_B + self.CD_L_p) + \
                self.CD_L_z + self.CD_L_d
        else:
            return self.CD_L_A*self.CD_L_w*cos(self.CD_L_w*d_B + self.CD_L_p)

    def _CD_L2(self, d_B):
        if not self.deriv:
            return self.CD_L2_A*sin(self.CD_L2_w*d_B + self.CD_L2_p) + \
                self.CD_L2_z + self.CD_L2_d
        else:
            return self.CD_L2_A*self.CD_L2_w*cos(self.CD_L2_w*d_B + self.CD_L2_p)

    def _CD_S(self, d_B):
        if not self.deriv:
            return self.CD_S_A*sin(self.CD_S_w*d_B + self.CD_S_p) + \
                self.CD_S_z + self.CD_S_d
        else:
            return self.CD_S_A*self.CD_S_w*cos(self.CD_S_w*d_B + self.CD_S_p)

    def _CD_S2(self, d_B):
        if not self.deriv:
            return self.CD_S2_A*sin(self.CD_S2_w*d_B + self.CD_S2_p) + \
                self.CD_S2_z + self.CD_S2_d
        else:
            return self.CD_S2_A*self.CD_S2_w*cos(self.CD_S2_w*d_B + self.CD_S2_p)

    def _CD_pbar(self, d_B):
        if not self.deriv:
            return self.CD_p_A*sin(self.CD_p_w*d_B + self.CD_p_p) + \
                self.CD_p_z + self.CD_p_d
        else:
            return self.CD_p_A*self.CD_p_w*cos(self.CD_p_w*d_B + self.CD_p_p)

    def _CD_Spbar(self, d_B):
        if not self.deriv:
            return self.CD_Sp_A*sin(self.CD_Sp_w*d_B + self.CD_Sp_p) + \
                self.CD_Sp_z + self.CD_Sp_d
        else:
            return self.CD_Sp_A*self.CD_Sp_w*cos(self.CD_Sp_w*d_B + self.CD_Sp_p)

    def _CD_qbar(self, d_B, deriv=False):
        if not self.deriv:
            return self.CD_q_A*sin(self.CD_q_w*d_B + self.CD_q_p) + \
                self.CD_q_z + self.CD_q_d
        else:
            return self.CD_q_A*self.CD_q_w*cos(self.CD_q_w*d_B + self.CD_q_p)

    def _CD_Lqbar(self, d_B):
        if not self.deriv:
            return self.CD_Lq_A*sin(self.CD_Lq_w*d_B + self.CD_Lq_p) + \
                self.CD_Lq_z + self.CD_Lq_d
        else:
            return self.CD_Lq_A*self.CD_Lq_w*cos(self.CD_Lq_w*d_B + self.CD_Lq_p)

    def _CD_L2qbar(self, d_B):
        if not self.deriv:
            return self.CD_L2q_A*sin(self.CD_L2q_w*d_B + self.CD_L2q_p) + \
                self.CD_L2q_z + self.CD_L2q_d
        else:
            return self.CD_L2q_A*self.CD_L2q_w*cos(self.CD_L2q_w*d_B + self.CD_L2q_p)

    def _CD_rbar(self, d_B):
        if not self.deriv:
            return self.CD_r_A*sin(self.CD_r_w*d_B + self.CD_r_p) + \
                self.CD_r_z + self.CD_r_d
        else:
            return self.CD_r_A*self.CD_r_w*cos(self.CD_r_w*d_B + self.CD_r_p)

    def _CD_Srbar(self, d_B):
        if not self.deriv:
            return self.CD_Sr_A*sin(self.CD_Sr_w*d_B + self.CD_Sr_p) + \
                self.CD_Sr_z + self.CD_Sr_d
        else:
            return self.CD_Sr_A*self.CD_Sr_w*cos(self.CD_Sr_w*d_B + self.CD_Sr_p)

    def _CD_da(self, d_B):
        if not self.deriv:
            return self.CD_da_A*sin(self.CD_da_w*d_B + self.CD_da_p) + \
                self.CD_da_z + self.CD_da_d
        else:
            return self.CD_da_A*self.CD_da_w*cos(self.CD_da_w*d_B + self.CD_da_p)

    def _CD_Sda(self, d_B):
        if not self.deriv:
            return self.CD_Sda_A*sin(self.CD_Sda_w*d_B + self.CD_Sda_p) + \
                self.CD_Sda_z + self.CD_Sda_d
        else:
            return self.CD_Sda_A*self.CD_Sda_w*cos(self.CD_Sda_w*d_B + self.CD_Sda_p)

    def _CD_de(self, d_B):
        if not self.deriv:
            return self.CD_de_A*sin(self.CD_de_w*d_B + self.CD_de_p) + \
                self.CD_de_z + self.CD_de_d
        else:
            return self.CD_de_A*self.CD_de_w*cos(self.CD_de_w*d_B + self.CD_de_p)

    def _CD_Lde(self, d_B):
        if not self.deriv:
            return self.CD_Lde_A*sin(self.CD_Lde_w*d_B + self.CD_Lde_p) + \
                self.CD_Lde_z + self.CD_Lde_d
        else:
            return self.CD_Lde_A*self.CD_Lde_w*cos(self.CD_Lde_w*d_B + self.CD_Lde_p)

    def _CD_de2(self, d_B):
        if not self.deriv:
            return self.CD_de2_A*sin(self.CD_de2_w*d_B + self.CD_de2_p) + \
                self.CD_de2_z + self.CD_de2_d
        else:
            return self.CD_de2_A*self.CD_de2_w*cos(self.CD_de2_w*d_B + self.CD_de2_p)

    def _Cl0(self, d_B):
        if not self.deriv:
            return self.Cl_0_A*sin(self.Cl_0_w*d_B + self.Cl_0_p) + \
                self.Cl_0_z + self.Cl_0_d
        else:
            return self.Cl_0_A*self.Cl_0_w*cos(self.Cl_0_w*d_B + self.Cl_0_p)

    def _Cl_alpha(self, d_B):
        if not self.deriv:
            return self.Cl_a_A*sin(self.Cl_a_w*d_B + self.Cl_a_p) + \
                self.Cl_a_z + self.Cl_a_d
        else:
            return self.Cl_a_A*self.Cl_a_w*cos(self.Cl_a_w*d_B + self.Cl_a_p)

    def _Cl_beta(self, d_B):
        # return 0.0
        if not self.deriv:
            return self.Cl_b_A*sin(self.Cl_b_w*d_B + self.Cl_b_p) + \
                self.Cl_b_z + self.Cl_b_d
        else:
            return self.Cl_b_A*self.Cl_b_w*cos(self.Cl_b_w*d_B + self.Cl_b_p)

    def _Cl_pbar(self, d_B):
        if not self.deriv:
            return self.Cl_p_A*sin(self.Cl_p_w*d_B + self.Cl_p_p) + \
                self.Cl_p_z + self.Cl_p_d
        else:
            return self.Cl_p_A*self.Cl_p_w*cos(self.Cl_p_w*d_B + self.Cl_p_p)

    def _Cl_qbar(self, d_B):
        if not self.deriv:
            return self.Cl_q_A*sin(self.Cl_q_w*d_B + self.Cl_q_p) + \
                self.Cl_q_z + self.Cl_q_d
        else:
            return self.Cl_q_A*self.Cl_q_w*cos(self.Cl_q_w*d_B + self.Cl_q_p)

    def _Cl_rbar(self, d_B):
        if not self.deriv:
            return self.Cl_r_A*sin(self.Cl_r_w*d_B + self.Cl_r_p) + \
                self.Cl_r_z + self.Cl_r_d
        else:
            return self.Cl_r_A*self.Cl_r_w*cos(self.Cl_r_w*d_B + self.Cl_r_p)

    def _Cl_Lrbar(self, d_B):
        if not self.deriv:
            return self.Cl_Lr_A*sin(self.Cl_Lr_w*d_B + self.Cl_Lr_p) + \
                self.Cl_Lr_z + self.Cl_Lr_d
        else:
            return self.Cl_Lr_A*self.Cl_Lr_w*cos(self.Cl_Lr_w*d_B + self.Cl_Lr_p)

    def _Cl_da(self, d_B):
        if not self.deriv:
            return self.Cl_da_A*sin(self.Cl_da_w*d_B + self.Cl_da_p) + \
                self.Cl_da_z + self.Cl_da_d
        else:
            return self.Cl_da_A*self.Cl_da_w*cos(self.Cl_da_w*d_B + self.Cl_da_p)

    def _Cl_de(self, d_B):
        if not self.deriv:
            return self.Cl_de_A*sin(self.Cl_de_w*d_B + self.Cl_de_p) + \
                self.Cl_de_z + self.Cl_de_d
        else:
            return self.Cl_de_A*self.Cl_de_w*cos(self.Cl_de_w*d_B + self.Cl_de_p)

    def _Cm0(self, d_B):
        if not self.deriv:
            return self.Cm_0_A*sin(self.Cm_0_w*d_B + self.Cm_0_p) + \
                self.Cm_0_z + self.Cm_0_d
        else:
            return self.Cm_0_A*self.Cm_0_w*cos(self.Cm_0_w*d_B + self.Cm_0_p)

    def _dCm0_dB(self, d_B):
        return self.Cm_0_A*self.Cm_0_w*cos(self.Cm_0_w*d_B + self.Cm_0_p)

    def _Cm_alpha(self, d_B):
        if not self.deriv:
            return self.Cm_a_A*sin(self.Cm_a_w*d_B + self.Cm_a_p) + \
                self.Cm_a_z + self.Cm_a_d
        else:
            return self.Cm_a_A*self.Cm_a_w*cos(self.Cm_a_w*d_B + self.Cm_a_p)

    def _dCma_dB(self, d_B):
        return self.Cm_a_A*self.Cm_a_w*cos(self.Cm_a_w*d_B + self.Cm_a_p)

    def _Cm_beta(self, d_B):
        if not self.deriv:
            return self.Cm_b_A*sin(self.Cm_b_w*d_B + self.Cm_b_p) + \
                self.Cm_b_z + self.Cm_b_d
        else:
            return self.Cm_b_A*self.Cm_b_w*cos(self.Cm_b_w*d_B + self.Cm_b_p)

    def _dCmb_dB(self, d_B):
        return self.Cm_b_A*self.Cm_b_w*cos(self.Cm_b_w*d_B + self.Cm_b_p)

    def _Cm_pbar(self, d_B):
        if not self.deriv:
            return self.Cm_p_A*sin(self.Cm_p_w*d_B + self.Cm_p_p) + \
                self.Cm_p_z + self.Cm_p_d
        else:
            return self.Cm_p_A*self.Cm_p_w*cos(self.Cm_p_w*d_B + self.Cm_p_p)

    def _dCmp_dB(self, d_B):
        return self.Cm_p_A*self.Cm_p_w*cos(self.Cm_p_w*d_B + self.Cm_p_p)

    def _Cm_qbar(self, d_B):
        if not self.deriv:
            return self.Cm_q_A*sin(self.Cm_q_w*d_B + self.Cm_q_p) + \
                self.Cm_q_z + self.Cm_q_d
        else:
            return self.Cm_q_A*self.Cm_q_w*cos(self.Cm_q_w*d_B + self.Cm_q_p)

    def _dCmq_dB(self, d_B):
        return self.Cm_q_A*self.Cm_q_w*cos(self.Cm_q_w*d_B + self.Cm_q_p)

    def _Cm_rbar(self, d_B):
        if not self.deriv:
            return self.Cm_r_A*sin(self.Cm_r_w*d_B + self.Cm_r_p) + \
                self.Cm_r_z + self.Cm_r_d
        else:
            return self.Cm_r_A*self.Cm_r_w*cos(self.Cm_r_w*d_B + self.Cm_r_p)

    def _dCmr_dB(self, d_B):
        return self.Cm_r_A*self.Cm_r_w*cos(self.Cm_r_w*d_B + self.Cm_r_p)

    def _Cm_da(self, d_B):
        if not self.deriv:
            return self.Cm_da_A*sin(self.Cm_da_w*d_B + self.Cm_da_p) + \
                self.Cm_da_z + self.Cm_da_d
        else:
            return self.Cm_da_A*self.Cm_da_w*cos(self.Cm_da_w*d_B + self.Cm_da_p)

    def _dCmda_dB(self, d_B):
        return self.Cm_da_A*self.Cm_da_w*cos(self.Cm_da_w*d_B + self.Cm_da_p)

    def _Cm_de(self, d_B):
        if not self.deriv:
            return self.Cm_de_A*sin(self.Cm_de_w*d_B + self.Cm_de_p) + \
                self.Cm_de_z + self.Cm_de_d
        else:
            return self.Cm_de_A*self.Cm_de_w*cos(self.Cm_de_w*d_B + self.Cm_de_p)

    def _dCmde_dB(self, d_B):
        return self.Cm_de_A*self.Cm_de_w*cos(self.Cm_de_w*d_B + self.Cm_de_p)

    def _Cn0(self, d_B):
        if not self.deriv:
            return self.Cn_0_A*sin(self.Cn_0_w*d_B + self.Cn_0_p) + \
                self.Cn_0_z + self.Cn_0_d
        else:
            return self.Cn_0_A*self.Cn_0_w*cos(self.Cn_0_w*d_B + self.Cn_0_p)

    def _dCn0_dB(self, d_B):
        return self.Cn_0_A*self.Cn_0_w*cos(self.Cn_0_w*d_B + self.Cn_0_p)

    def _Cn_alpha(self, d_B):
        if not self.deriv:
            return self.Cn_a_A*sin(self.Cn_a_w*d_B + self.Cn_a_p) + \
                self.Cn_a_z + self.Cn_a_d
        else:
            return self.Cn_a_A*self.Cn_a_w*cos(self.Cn_a_w*d_B + self.Cn_a_p)

    def _dCna_dB(self, d_B):
        return self.Cn_a_A*self.Cn_a_w*cos(self.Cn_a_w*d_B + self.Cn_a_p)

    def _Cn_beta(self, d_B):
        # return 0.0
        if not self.deriv:
            return self.Cn_b_A*sin(self.Cn_b_w*d_B + self.Cn_b_p) + \
                self.Cn_b_z + self.Cn_b_d
        else:
            return self.Cn_b_A*self.Cn_b_w*cos(self.Cn_b_w*d_B + self.Cn_b_p)

    def _dCnb_dB(self, d_B):
        return self.Cn_b_A*self.Cn_b_w*cos(self.Cn_b_w*d_B + self.Cn_b_p)

    def _Cn_pbar(self, d_B):
        if not self.deriv:
            return self.Cn_p_A*sin(self.Cn_p_w*d_B + self.Cn_p_p) + \
                self.Cn_p_z + self.Cn_p_d
        else:
            return self.Cn_p_A*self.Cn_p_w*cos(self.Cn_p_w*d_B + self.Cn_p_p)

    def _dCnp_dB(self, d_B):
        return self.Cn_p_A*self.Cn_p_w*cos(self.Cn_p_w*d_B + self.Cn_p_p)

    def _Cn_Lpbar(self, d_B):
        if not self.deriv:
            return self.Cn_Lp_A*sin(self.Cn_Lp_w*d_B + self.Cn_Lp_p) + \
                self.Cn_Lp_z + self.Cn_Lp_d
        else:
            return self.Cn_Lp_A*self.Cn_Lp_w*cos(self.Cn_Lp_w*d_B + self.Cn_Lp_p)

    def _dCnLp_dB(self, d_B): 
        CnLp = self._Cn_Lpbar(d_B)
        C1 = self.Cn_Lp_A*self.Cn_Lp_w*cos(self.Cn_Lp_w*d_B + self.Cn_Lp_p)*\
            (self.CL_0_A*sin(self.CL_0_w*d_B + self.CL_0_p) + self.CL_0_z + self.CL_0_d)
        C2 = self.CL_0_A*self.CL_0_w*cos(self.CL_0_w*d_B + self.CL_0_p)*\
            CnLp
        C3 = self.Cn_Lp_A*self.Cn_Lp_w*cos(self.Cn_Lp_w*d_B + self.Cn_Lp_p)*\
            (self.CL_a_A*sin(self.CL_a_w*d_B + self.CL_a_p) + self.CL_a_z + self.CL_a_d)
        C4 = self.CL_a_A*self.CL_a_w*cos(self.CL_a_w*d_B + self.CL_a_p)*\
            CnLp
        return [C1, C2, C3, C4]

    def _Cn_qbar(self, d_B):
        if not self.deriv:
            return self.Cn_q_A*sin(self.Cn_q_w*d_B + self.Cn_q_p) + \
                self.Cn_q_z + self.Cn_q_d
        else:
            return self.Cn_q_A*self.Cn_q_w*cos(self.Cn_q_w*d_B + self.Cn_q_p)

    def _dCnq_dB(self, d_B):
        return self.Cn_q_A*self.Cn_q_w*cos(self.Cn_q_w*d_B + self.Cn_q_p)

    def _Cn_rbar(self, d_B):
        if not self.deriv:
            return self.Cn_r_A*sin(self.Cn_r_w*d_B + self.Cn_r_p) + \
                self.Cn_r_z + self.Cn_r_d
        else:
            return self.Cn_r_A*self.Cn_r_w*cos(self.Cn_r_w*d_B + self.Cn_r_p)

    def _dCnr_dB(self, d_B):
        return self.Cn_r_A*self.Cn_r_w*cos(self.Cn_r_w*d_B + self.Cn_r_p)

    def _Cn_da(self, d_B):
        if not self.deriv:
            return self.Cn_da_A*sin(self.Cn_da_w*d_B + self.Cn_da_p) + \
                self.Cn_da_z + self.Cn_da_d
        else:
            return self.Cn_da_A*self.Cn_da_w*cos(self.Cn_da_w*d_B + self.Cn_da_p)

    def _dCnda_dB(self, d_B):
        Cnda = self._Cn_da(d_B)
        C1 = self.Cn_da_A*self.Cn_da_w*cos(self.Cn_da_w*d_B + self.Cn_da_p)*\
            (self.CL_0_A*sin(self.CL_0_w*d_B + self.CL_0_p) + self.CL_0_z + self.CL_0_d)
        C2 = self.CL_0_A*self.CL_0_w*cos(self.CL_0_w*d_B + self.CL_0_p)*\
            Cnda
        C3 = self.Cn_da_A*self.Cn_da_w*cos(self.Cn_da_w*d_B + self.Cn_da_p)*\
            (self.CL_a_A*sin(self.CL_a_w*d_B + self.CL_a_p) + self.CL_a_z + self.CL_a_d)
        C4 = self.CL_a_A*self.CL_a_w*cos(self.CL_a_w*d_B + self.CL_a_p)*\
            Cnda
        return [C1, C2, C3, C4]

    def _Cn_Lda(self, d_B):
        if not self.deriv:
            return self.Cn_Lda_A*sin(self.Cn_Lda_w*d_B + self.Cn_Lda_p) + \
                self.Cn_Lda_z + self.Cn_Lda_d
        else:
            return self.Cn_Lda_A*self.Cn_Lda_w*cos(self.Cn_Lda_w*d_B + self.Cn_Lda_p)

    def _Cn_de(self, d_B):
        if not self.deriv:
            return self.Cn_de_A*sin(self.Cn_de_w*d_B + self.Cn_de_p) + \
                self.Cn_de_z + self.Cn_de_d
        else:
            return self.Cn_de_A*self.Cn_de_w*cos(self.Cn_de_w*d_B + self.Cn_de_p)

    def _dCnde_dB(self, d_B):
        return self.Cn_de_A*self.Cn_de_w*cos(self.Cn_de_w*d_B + self.Cn_de_p)

    def _CL(self, alpha, beta, pbar, qbar, rbar, da, de, dB):
        CL1 = self._CL0(dB) + self._CL_alpha(dB)*alpha
        CL = (CL1 + self._CL_beta(dB)*beta + self._CL_pbar(dB)*pbar +
              self._CL_qbar(dB)*qbar + self._CL_rbar(dB)*rbar +
              self._CL_da(dB)*da + self._CL_de(dB)*de)
        return CL

    def _CS(self, alpha, beta, pbar, qbar, rbar, da, de, dB):
        CL1 = self._CL0(dB) + self._CL_alpha(dB)*alpha
        CS = (self._CS0(dB) + self._CS_alpha(dB)*alpha +
              self._CS_beta(dB)*beta +
              (self._CS_pbar(dB) + self._CS_Lpbar(dB)*CL1)*pbar +
              self._CS_qbar(dB)*qbar + self._CS_rbar(dB)*rbar +
              self._CS_da(dB)*da + self._CS_de(dB)*de)
        return CS

    def _CD(self, alpha, beta, pbar, qbar, rbar, da, de, dB):
        CL1 = self._CL0(dB) + self._CL_alpha(dB)*alpha
        CS1 = self._CS0(dB) + self._CS_beta(dB)*beta
        CD = (self._CD0(dB) + self._CD_L(dB)*CL1 + self._CD_L2(dB)*CL1**2 +
              self._CD_S(dB)*CS1 + self._CD_S2(dB)*CS1**2 +
              (self._CD_pbar(dB) + self._CD_Spbar(dB)*CS1)*pbar +
              (self._CD_qbar(dB) + self._CD_Lqbar(dB)*CL1 + 
              self._CD_L2qbar(dB)*CL1**2)*qbar +
              (self._CD_rbar(dB) + self._CD_Srbar(dB)*CS1)*rbar +
              (self._CD_da(dB) + self._CD_Sda(dB)*CS1)*da +
              (self._CD_de(dB) + self._CD_Lde(dB)*CL1)*de +
              self._CD_de2(dB)*de**2)
        return CD

    def _Cl(self, alpha, beta, pbar, qbar, rbar, da, de, dB):
        CL1 = self._CL0(dB) + self._CL_alpha(dB)*alpha
        Cl = (self._Cl0(dB) + self._Cl_alpha(dB)*alpha +
              self._Cl_beta(dB)*beta + self._Cl_pbar(dB)*pbar +
              self._Cl_qbar(dB)*qbar +
              (self._Cl_rbar(dB) + self._Cl_Lrbar(dB)*CL1)*rbar +
              self._Cl_da(dB)*da + self._Cl_de(dB)*de)
        return Cl

    def _Cm(self, alpha, beta, pbar, qbar, rbar, da, de, dB):
        Cm = (self._Cm0(dB) + self._Cm_alpha(dB)*alpha +
              self._Cm_beta(dB)*beta + self._Cm_pbar(dB)*pbar +
              self._Cm_qbar(dB)*qbar + self._Cm_rbar(dB)*rbar +
              self._Cm_da(dB)*da + self._Cm_de(dB)*de)
        return Cm

    def _dCm_dB(self, alpha, beta, pbar, qbar, rbar, da, de, dB):
        dCmdB = (self._dCm0_dB(dB) + self._dCma_dB(dB)*alpha +
                 self._dCmb_dB(dB)*beta + self._dCmp_dB(dB)*pbar +
                 self._dCmq_dB(dB)*qbar + self._dCmr_dB(dB)*rbar +
                 self._dCmda_dB(dB)*da + self._dCmde_dB(dB)*de)
        return dCmdB

    def _Cn(self, alpha, beta, pbar, qbar, rbar, da, de, dB):
        CL1 = self._CL0(dB) + self._CL_alpha(dB)*alpha
        Cn = (self._Cn0(dB) + self._Cn_alpha(dB)*alpha +
              self._Cn_beta(dB)*beta +
              (self._Cn_pbar(dB) + self._Cn_Lpbar(dB)*CL1)*pbar +
              self._Cn_qbar(dB)*qbar + self._Cn_rbar(dB)*rbar +
              (self._Cn_da(dB) + self._Cn_Lda(dB)*CL1)*da +
              self._Cn_de(dB)*de)
        return Cn

    def _dCn_dB(self, alpha, beta, pbar, qbar, rbar, da, de, dB):
        dCnLp_dB = self._dCnLp_dB(dB)
        dCnda_dB = self._dCnda_dB(dB)
        dCndB = (self._dCn0_dB(dB) + self._dCna_dB(dB)*alpha +
                 self._dCnb_dB(dB)*beta +
                 (dCnLp_dB[0] + dCnLp_dB[1] + alpha*(dCnLp_dB[2] + dCnLp_dB[3]))*pbar +
                 self._dCnq_dB(dB)*qbar + self._dCnr_dB(dB)*rbar +
                 (dCnda_dB[0] + dCnda_dB[1] + alpha*(dCnda_dB[2] + dCnda_dB[3]))*da +
                 self._dCnde_dB(dB)*de)
        return dCndB

    def _inc_aero_results(self, alpha, beta, pbar, qbar, rbar, da, de, dB):
        params = alpha, beta, pbar, qbar, rbar, da, de, dB
        return [self._CL(*params), self._CS(*params), self._CD(*params),
                self._Cl(*params), self._Cm(*params), self._Cn(*params)]

    def _stall_correction(self,a,CL,CD,Cm):
        # determine flat plate forces and moment
        CLplate = 2. *sy.sign(a) * sin(a)**2 * cos(a)
        CDplate = 2. * sin(abs(a))**sy.Rational(3,2) # 1.5
        Cmplate = -0.8 * sin(a)

        # determine stall sigmoid
        expMmin = exp(-self.S_M*(a-self.S_ab))
        expMplu = exp(self.S_M*(a+self.S_ab))
        sig = (1. + expMmin + expMplu) / (1. + expMmin) / (1. + expMplu)

        # add stall effects
        CL = (1. - sig) * CL + sig * CLplate
        CD = (1. - sig) * CD + sig * CDplate
        Cm = (1. - sig) * Cm + sig * Cmplate

        return CL,CD,Cm
    
    def _stall_correction_derivatives(self,a,CL,CD,Cm):
        # determine flat plate forces and moment
        CLplate = 2. *sy.sign(a) * sin(a)**2 * cos(a)
        CDplate = 2. * sin(abs(a))**sy.Rational(3,2) # 1.5
        Cmplate = -0.8 * sin(a)

        # determine stall sigmoid
        expMmin = exp(-self.S_M*(a-self.S_ab))
        expMplu = exp(self.S_M*(a+self.S_ab))
        sig = (1. + expMmin + expMplu) / (1. + expMmin) / (1. + expMplu)

        # derivatives
        CLplate_a =sy.sign(a) * sin(a) * (3.*cos(2.*a) + 1.)
        CDplate_a = 3.*a/abs(a)*sy.sqrt(sin(abs(a))) if a != 0. else 0.
        Cmplate_a = -0.8*cos(a)

        # stall sigmoid derivative
        expMpmn = exp(self.S_M*(a-self.S_ab))
        expMa = exp(2.*self.S_M*a)
        sig_a = (self.S_M*expMplu*(expMa -1.))/\
            ((1. + expMpmn)**2*(1. + expMplu)**2)

        return CLplate,CDplate,Cmplate,sig,CLplate_a,CDplate_a,Cmplate_a,sig_a

    def _Anderson_correction(self, coeff, Lambda, RA, M):
        num = coeff*cos(Lambda)
        denom_2 = num/(pi*RA)
        denom_1 = sy.sqrt(1. - M**2*cos(Lambda)**2 + denom_2**2) # **0.5
        denom = denom_1 + denom_2
        return num/denom

    def _Prandtl_Glauert_subsonic_correction(self, coeff, M):
        return coeff / sy.sqrt(1. - M**2) # **0.5

    def _Prandtl_Glauert_supersonic_correction(self, coeff, M):
        return coeff / sy.sqrt(M**2 - 1.) # **0.5

    def aero_results(self, alpha, beta, pbar, qbar, rbar, da, de, dB, 
    compressible=True, M=113.0, use_Anderson=True, enforce_stall=True):
        params = alpha, beta, pbar, qbar, rbar, da, de, dB

        # run incompressible
        [CL, CS, CD, Cl, Cm, Cn] = self._inc_aero_results(*params)

        # stall
        if enforce_stall:
            # implement stall effects
            [CL,CD,Cm] = self._stall_correction(alpha,CL,CD,Cm)

        if not compressible:
            return [CL, CS, CD, Cl, Cm, Cn]
        else:
            # if not given mach number, throw error
            if M == 113.0:
                raise ValueError("Mach number not specified")
            elif True:
                if use_Anderson:
                    CL = self._Anderson_correction(CL,self.Lam_w,self.RA_w,M)
                    CS = self._Anderson_correction(CS,self.Lam_h,self.RA_h,M)
                    Cl = self._Anderson_correction(Cl,self.Lam_w,self.RA_w,M)
                    Cm = self._Anderson_correction(Cm,self.Lam_w,self.RA_w,M)
                    Cn = self._Anderson_correction(Cn,self.Lam_h,self.RA_h,M)
                else:
                    CL = self._Prandtl_Glauert_subsonic_correction(CL,M)
                    CS = self._Prandtl_Glauert_subsonic_correction(CS,M)
                    Cl = self._Prandtl_Glauert_subsonic_correction(Cl,M)
                    Cm = self._Prandtl_Glauert_subsonic_correction(Cm,M)
                    Cn = self._Prandtl_Glauert_subsonic_correction(Cn,M)
            else:
                CL = self._Prandtl_Glauert_supersonic_correction(CL,M)
                CS = self._Prandtl_Glauert_supersonic_correction(CS,M)
                Cl = self._Prandtl_Glauert_supersonic_correction(Cl,M)
                Cm = self._Prandtl_Glauert_supersonic_correction(Cm,M)
                Cn = self._Prandtl_Glauert_supersonic_correction(Cn,M)
            
            # return
            return [CL, CS, CD, Cl, Cm, Cn]

    def _uncorrect_Anderson(self, coeff, Lambda, RA, M):
        num = coeff*sy.sqrt(1./cos(Lambda)**2 - M**2)
        denom = sy.sqrt(1. - 2.*coeff/pi/RA)
        return num/denom

    def _uncorrect_Prandtl_Glauert_subsonic(self, coeff, M):
        return coeff * sy.sqrt(1. - M**2) # **0.5

    def _uncorrect_Prandtl_Glauert_supersonic(self, coeff, M):
        return coeff * sy.sqrt(M**2 - 1.) # **0.5

    def _uncorrect_stall(self,a,CL,CD,Cm):
        # determine flat plate forces and moment
        CLplate = 2. *sy.sign(a) * sin(a)**2 * cos(a)
        CDplate = 2. * sin(abs(a))**sy.Rational(3,2) # 1.5
        Cmplate = -0.8 * sin(a)

        # determine stall sigmoid
        expMmin = exp(-self.S_M*(a-self.S_ab))
        expMplu = exp(self.S_M*(a+self.S_ab))
        sig = (1. + expMmin + expMplu) / (1. + expMmin) / (1. + expMplu)

        # add stall effects
        CL = (CL - sig * CLplate)/(1. - sig)
        CD = (CD - sig * CDplate)/(1. - sig)
        Cm = (Cm - sig * Cmplate)/(1. - sig)

        return CL,CD,Cm
    
    def uncorrect_M(self,CMs,alpha,
    compressible=True, M=113.0, use_Anderson=True, enforce_stall=True):
        # pull out CMs
        [Cl, Cm, Cn] = CMs

        # uncorrect for compressibility
        if compressible:
            # if not given mach number, throw error
            if M == 113.0:
                raise ValueError("Mach number not specified")
            elif True:
                if use_Anderson:
                    Cl = self._uncorrect_Anderson(Cl,self.Lam_w,self.RA_w,M)
                    Cm = self._uncorrect_Anderson(Cm,self.Lam_w,self.RA_w,M)
                    Cn = self._uncorrect_Anderson(Cn,self.Lam_h,self.RA_h,M)
                else:
                    Cl = self._uncorrect_Prandtl_Glauert_subsonic(Cl,M)
                    Cm = self._uncorrect_Prandtl_Glauert_subsonic(Cm,M)
                    Cn = self._uncorrect_Prandtl_Glauert_subsonic(Cn,M)
            else:
                Cl = self._uncorrect_Prandtl_Glauert_supersonic(Cl,M)
                Cm = self._uncorrect_Prandtl_Glauert_supersonic(Cm,M)
                Cn = self._uncorrect_Prandtl_Glauert_supersonic(Cn,M)
        
        # uncorrect for stall
        if enforce_stall:
            # implement stall effects
            [_,_,Cm] = self._uncorrect_stall(alpha,0.0,0.0,Cm)
        
        return [Cl, Cm, Cn]

    def get_thrust(self,tau,H,V):
        return self.Prop.get_thrust(tau,H,V)

    def aero_CG_offset_results(self, alpha, beta, pbar, qbar, rbar, da, de, dB, tau, 
                                V, H, rho_0, rho, cg_shift=[0., 0., 0.], compressible=True,
                                M=113.0, use_Anderson=True, enforce_stall=True, thrust_off = False):
        
        [CL, CS, CD, Cl, Cm, Cn] = self.aero_results(alpha, beta, pbar, qbar, rbar, da, de, dB, 
                                                      compressible, M, use_Anderson, enforce_stall)
        
        x_shift, y_shift, z_shift = cg_shift
        
        nondim_const = 0.5*rho*V*V*self.S_w

        # body fixed force and moment coefficients
        CX = -CD*cos(alpha)*cos(beta) - CS*cos(alpha)*sin(beta) + CL*sin(alpha)
        CY = CS*cos(beta) - CD*sin(beta)
        CZ = -CD*sin(alpha)*cos(beta) - CS*sin(alpha)*sin(beta) - CL*cos(alpha)

        
        T_dir = array([1.0,0.0,0.0])
        
        
        if thrust_off == False:
            FP = self.get_thrust(tau, H, V)*T_dir
        else:
            FP = self.get_thrust(tau, H, V)*array([0.0, 0.0, 0.0])
        
        # FP = self.get_thrust(tau, H, V)*T_dir
        
        Fx = CX*nondim_const + FP[0]
        Fy = CY*nondim_const + FP[1]
        Fz = CZ*nondim_const + FP[2]
        Mx = Cl*nondim_const*self.b_w - Fz*y_shift + Fy*z_shift
        My = Cm*nondim_const*self.c_w - Fx*z_shift + Fz*x_shift
        Mz = Cn*nondim_const*self.b_w - Fy*x_shift + Fx*y_shift
                
        FM = [Fx, Fy, Fz, Mx, My, Mz]
        return FM

    def Cn_dB(self, alpha, beta, pbar, qbar, rbar, da, de, dB, method='fd', h=0.001):
        if method=='fd':
            params_p = alpha, beta, pbar, qbar, rbar, da, de, dB + h
            params_m = alpha, beta, pbar, qbar, rbar, da, de, dB - h
            Cn_plus = self._Cn(*params_p)
            Cn_minus = self._Cn(*params_m)
            Cn_dB = (Cn_plus - Cn_minus)/(2.*h)
            return Cn_dB
        elif method=='complex-step':
            h = 1e-16
            params_complex = alpha, beta, pbar, qbar, rbar, da, de, complex(dB, h)
            Cn_complex = self._Cn(*params_complex)
            Cn_dB = Cn_complex.imag/h
            return Cn_dB
        elif method=='fit':
            A = -0.0199462
            w = 2.
            phi = pi/2.
            z = 0.
            return A*sin(w*dB + phi) + z
        elif method=='analytic':
            Cdict = self.Cn_coeffs["Cn_0"]
            [A0, w0, phi0, z0] = [Cdict[c] for c in Cdict]
            Cdict = self.Cn_coeffs["Cn_alpha"]
            [Aa, wa, phia, za] = [Cdict[c] for c in Cdict]
            Cdict = self.Cn_coeffs["Cn_beta"]
            [Ab, wb, phib, zb] = [Cdict[c] for c in Cdict]
            Cdict = self.Cn_coeffs["Cn_qbar"]
            [Aq, wq, phiq, zq] = [Cdict[c] for c in Cdict]
            Cdict = self.Cn_coeffs["Cn_rbar"]
            [Ar, wr, phir, zr] = [Cdict[c] for c in Cdict]
            Cdict = self.Cn_coeffs["Cn_de"]
            [Ade, wde, phide, zde] = [Cdict[c] for c in Cdict]
            dCn0 = A0*cos(w0*dB + phi0)
            dCna = Aa*cos(wa*dB + phia)
            dCnb = Ab*cos(wb*dB + phib)
            dCnq = Aq*cos(wq*dB + phiq)
            dCnr = Ar*cos(wr*dB + phir)
            dCnde = Ade*cos(wde*dB + phide)
            Cn_dB = dCn0 + dCna*alpha + dCnb*beta + dCnq*qbar + dCnr*rbar + dCnde*de
            return Cn_dB

    def control_matrix(self,  alpha, beta, pbar, qbar, rbar, da, de, dB):
        A = zeros((2, 2))
        A[0, 0] = self._dCm_dB(alpha, beta, pbar, qbar, rbar, da, de, dB)
        A[0, 1] = self._Cm_de(dB)
        A[1, 0] = self._dCn_dB(alpha, beta, pbar, qbar, rbar, da, de, dB)
        A[1, 1] = self._Cn_de(dB)
        return A


class Propulsion:
    def __init__(self, inp_dir="./", **kwargs):
        use_fitted_thrust_model = kwargs.get("use_fitted_thrust_model",True)
        fn = kwargs.get('thrust_model_file_name', '0')
        use_rc = kwargs.get('use_rc_thrust_model', False)
        if fn == "0":
            if use_fitted_thrust_model:
                fn = "thrust_model.json"
            else: # simple model
                if use_rc:
                    fn = "thrust_model_rc.json"
                else:
                    fn = "thrust_model_simple.json"
        self.model_coeffs_dict = json.load(open(inp_dir + fn))
        if use_fitted_thrust_model:
            # idle
            idle_dict =  self.model_coeffs_dict["idle"]
            self.a_idle_c0 = idle_dict["a"]["c0"]
            self.a_idle_c1 = idle_dict["a"]["c1"]
            self.a_idle_c2 = idle_dict["a"]["c2"]
            self.T0_idle_c0 = idle_dict["T0"]["c0"]
            self.T0_idle_c1 = idle_dict["T0"]["c1"]
            self.T0_idle_c2 = idle_dict["T0"]["c2"]
            self.T1_idle_c0 = idle_dict["T1"]["c0"]
            self.T1_idle_c1 = idle_dict["T1"]["c1"]
            self.T1_idle_c2 = idle_dict["T1"]["c2"]
            self.T2_idle_c0 = idle_dict["T2"]["c0"]
            self.T2_idle_c1 = idle_dict["T2"]["c1"]
            self.T2_idle_c2 = idle_dict["T2"]["c2"]
            # mil
            mil_dict =  self.model_coeffs_dict["mil"]
            self.a_mil_c0 = mil_dict["a"]["c0"]
            self.a_mil_c1 = mil_dict["a"]["c1"]
            self.a_mil_c2 = mil_dict["a"]["c2"]
            self.T0_mil_c0 = mil_dict["T0"]["c0"]
            self.T0_mil_c1 = mil_dict["T0"]["c1"]
            self.T0_mil_c2 = mil_dict["T0"]["c2"]
            self.T1_mil_c0 = mil_dict["T1"]["c0"]
            self.T1_mil_c1 = mil_dict["T1"]["c1"]
            self.T1_mil_c2 = mil_dict["T1"]["c2"]
            self.T2_mil_c0 = mil_dict["T2"]["c0"]
            self.T2_mil_c1 = mil_dict["T2"]["c1"]
            self.T2_mil_c2 = mil_dict["T2"]["c2"]
            # max
            max_dict =  self.model_coeffs_dict["max"]
            self.a_max_c0 = max_dict["a"]["c0"]
            self.a_max_c1 = max_dict["a"]["c1"]
            self.a_max_c2 = max_dict["a"]["c2"]
            self.T0_max_c0 = max_dict["T0"]["c0"]
            self.T0_max_c1 = max_dict["T0"]["c1"]
            self.T0_max_c2 = max_dict["T0"]["c2"]
            self.T1_max_c0 = max_dict["T1"]["c0"]
            self.T1_max_c1 = max_dict["T1"]["c1"]
            self.T1_max_c2 = max_dict["T1"]["c2"]
            self.T2_max_c0 = max_dict["T2"]["c0"]
            self.T2_max_c1 = max_dict["T2"]["c1"]
            self.T2_max_c2 = max_dict["T2"]["c2"]
        else:
            mod_dict = self.model_coeffs_dict["model"]
            self.a = mod_dict["a"]
            self.T0 = mod_dict["T0"]
            self.T1 = mod_dict["T1"]
            self.T2 = mod_dict["T2"]
        
        # initialize atmosphere model
        self.atm_model = kwargs.get("atmosphere_model","use_hunsakers")
        if self.atm_model == "use_hunsakers":
            self.atm_model = stdatm_english
            self.rho_i = 3
        else:
            self.rho_i = kwargs.get("rho_index_in_model")
        
        # initialize sea level rho
        self.rho_0 = self.atm_model(0.0)[self.rho_i]
        
        self.use_fitted_thrust_model = use_fitted_thrust_model
        if use_fitted_thrust_model:
            self.get_thrust = self.T_fitted
        else:
            self.get_thrust = self.T_simple


    def _a_idle(self,H):
        return self.a_idle_c0 + self.a_idle_c1*H + self.a_idle_c2*H*H

    def _T0_idle(self,H):
        return self.T0_idle_c0 + self.T0_idle_c1*H + self.T0_idle_c2*H*H

    def _T1_idle(self,H):
        return self.T1_idle_c0 + self.T1_idle_c1*H + self.T1_idle_c2*H*H

    def _T2_idle(self,H):
        return self.T2_idle_c0 + self.T2_idle_c1*H + self.T2_idle_c2*H*H

    def idle_coefs(self,H):
        return self._a_idle(H),self._T0_idle(H),self._T1_idle(H),self._T2_idle(H)

    def _a_mil(self,H):
        return self.a_mil_c0 + self.a_mil_c1*H + self.a_mil_c2*H*H

    def _T0_mil(self,H):
        return self.T0_mil_c0 + self.T0_mil_c1*H + self.T0_mil_c2*H*H

    def _T1_mil(self,H):
        return self.T1_mil_c0 + self.T1_mil_c1*H + self.T1_mil_c2*H*H

    def _T2_mil(self,H):
        return self.T2_mil_c0 + self.T2_mil_c1*H + self.T2_mil_c2*H*H

    def mil_coefs(self,H):
        return self._a_mil(H),self._T0_mil(H),self._T1_mil(H),self._T2_mil(H)


    def _a_max(self,H):
        return self.a_max_c0 + self.a_max_c1*H + self.a_max_c2*H*H

    def _T0_max(self,H):
        return self.T0_max_c0 + self.T0_max_c1*H + self.T0_max_c2*H*H

    def _T1_max(self,H):
        return self.T1_max_c0 + self.T1_max_c1*H + self.T1_max_c2*H*H

    def _T2_max(self,H):
        return self.T2_max_c0 + self.T2_max_c1*H + self.T2_max_c2*H*H

    def max_coefs(self,H):
        return self._a_max(H),self._T0_max(H),self._T1_max(H),self._T2_max(H)


    def _T_idle(self,rho,V,H):
        # get coefficients
        a  =  self.a_idle_c0 +  self.a_idle_c1*H +  self.a_idle_c2*H*H
        T0 = self.T0_idle_c0 + self.T0_idle_c1*H + self.T0_idle_c2*H*H
        T1 = self.T1_idle_c0 + self.T1_idle_c1*H + self.T1_idle_c2*H*H
        T2 = self.T2_idle_c0 + self.T2_idle_c1*H + self.T2_idle_c2*H*H
        return (rho/self.rho_0)**a*(T0 + T1*V + T2*V*V)

    def _T_mil(self,rho,V,H):
        # get coefficients
        a =   self.a_mil_c0 +  self.a_mil_c1*H +  self.a_mil_c2*H*H
        T0 = self.T0_mil_c0 + self.T0_mil_c1*H + self.T0_mil_c2*H*H
        T1 = self.T1_mil_c0 + self.T1_mil_c1*H + self.T1_mil_c2*H*H
        T2 = self.T2_mil_c0 + self.T2_mil_c1*H + self.T2_mil_c2*H*H
        return (rho/self.rho_0)**a*(T0 + T1*V + T2*V*V)

    def _T_max(self,rho,V,H):
        # get coefficients
        a =   self.a_max_c0 +  self.a_max_c1*H +  self.a_max_c2*H*H
        T0 = self.T0_max_c0 + self.T0_max_c1*H + self.T0_max_c2*H*H
        T1 = self.T1_max_c0 + self.T1_max_c1*H + self.T1_max_c2*H*H
        T2 = self.T2_max_c0 + self.T2_max_c1*H + self.T2_max_c2*H*H
        return (rho/self.rho_0)**a*(T0 + T1*V + T2*V*V)


    def T_fitted(self,tau,H,V):
        # calculate P1
        if 0. <= tau <= 0.77:
            P1 = 64.94*tau
        elif 0.77 < tau <= 1.:
            P1 = 217.38*tau - 117.38
        elif tau <= 0.:
            P1 = 0.
        else:
            P1 = 100.
        
        # keep above ground
        if False:
            H = 0.
        
        # calculate rho
        rho = sy.Symbol("rho")
        
        # get total thrust
        T_mil = self._T_mil(rho,V,H)
        if P1 < 50.:
            T_idle = self._T_idle(rho,V,H)
            T = T_idle + (T_mil - T_idle)*P1/50.
        else:
            T_max = self._T_max(rho,V,H)
            T = T_mil + (T_max - T_mil)*(P1-50.)/50.
        
        return T
    
    def T_simple(self,tau,H,V):
        # calculate rho
        rho = self.atm_model(H)[self.rho_i]
        return tau*(rho/self.rho_0)**self.a*(self.T0 + self.T1*V + self.T2*V*V)


def Anderson_correction_der_coeff(coeff, Lambda, RA, M):
    CL = cos(Lambda)
    set0 = 1. - M**2*CL**2 + (coeff*CL/pi/RA)**2
    num = CL*( pi*RA*sy.sqrt(set0) - coeff*CL)
    denom = coeff*CL*sy.sqrt(set0) + pi*RA*set0
    return num/denom

def Anderson_correction_der_M(coeff, Lambda, RA, M):
    CL = cos(Lambda)
    set0 = 1. - M**2*CL**2 + (coeff*CL/pi/RA)**2
    num = coeff*CL
    denom = coeff*CL/pi/RA*sy.sqrt(set0) + set0
    return num/denom



if __name__ == "__main__":
    # create x symbolic
    V,a,b,p,q,r,z,P,T,da,de,dB = sy.symbols("V a b p q r z P T da de dB")
    vxb = V*sy.cos(a)*sy.cos(b)
    vyb = V*sy.sin(b)
    vzb = V*sy.sin(a)*sy.cos(b)
    x = sy.Matrix([vxb,vyb,vzb,p,q,r,0.0,0.0,z,P,T,0.0] + [0.0]*4)
    u = sy.Matrix([da,de,dB,0.0])
    aero_model = BIREAero(inp_dir=aero_directory,
                    thrust_dir=aero_directory,use_rc_thrust_model=False)
    # inertia_model = InertiaModel(inp_dir=mass_directory, \
    #         is_bire=True,is_rc=False)
    inertia_model = InertiaModel_zerodB(inp_dir=mass_directory, \
            is_bire=True,is_rc=False)
    aero_dict = dict(compressible=False, use_Anderson=True, enforce_stall=False)
    B = _build_input_jacobian(x,u,aero_model,inertia_model,aero_dict,
        aero_dict["enforce_stall"],aero_dict["compressible"],aero_dict["use_Anderson"])
    
    B = B[3:6,0:3]
    # print(B)
    # quit()
    print("simplifying...")
    for i in range(3):
        for j in range(3):
            print("i =",i,", j =",j)
            print(B[i,j])
            print()
            B[i,j] = sy.simplify(B[i,j])
            print(B[i,j])
            print()
    print("full")
    print(B)
    print()
    print("calculating determinant...")
    Bdet = B.det()
    print(Bdet)
    print()
    print("simplifying...")
    Bdet = sy.simplify(Bdet)
    print(Bdet)
    print()

