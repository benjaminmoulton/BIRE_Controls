#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 16:18:40 2021

@author: christian
"""

from math import sin,cos,pi,exp
from numpy import sign, deg2rad
import json
from hunsaker_atm import stdatm_english
from thrust import Propulsion

class F16Aero:
    def __init__(self, inp_dir='./', **kwargs):
        fn = kwargs.get('fn', 'mux_model_adj.json')
        self.model_coeffs_dict = json.load(open(inp_dir + fn))
        self.CL_coeffs = self.model_coeffs_dict["CL"]
        self.CS_coeffs = self.model_coeffs_dict["CS"]
        self.CD_coeffs = self.model_coeffs_dict["CD"]
        self.Cl_coeffs = self.model_coeffs_dict["Cell"]
        self.Cm_coeffs = self.model_coeffs_dict["Cm"]
        self.Cn_coeffs = self.model_coeffs_dict["Cn"]
        self.CL0 = self.CL_coeffs["CL_0"]
        self.CLa = self.CL_coeffs["CL_alpha"]
        self.CLq = self.CL_coeffs["CL_qbar"]
        self.CLde = self.CL_coeffs["CL_de"]
        self.CSb = self.CS_coeffs["CS_beta"]
        self.CSp = self.CS_coeffs["CS_pbar"]
        self.CSLp = self.CS_coeffs["CS_Lpbar"]
        self.CSr = self.CS_coeffs["CS_rbar"]
        self.CSda = self.CS_coeffs["CS_da"]
        self.CSdr = self.CS_coeffs["CS_dr"]
        self.CD0 = self.CD_coeffs["CD_0"]
        self.CDL = self.CD_coeffs["CD_L"]
        self.CDL2 = self.CD_coeffs["CD_L2"]
        self.CDS2 = self.CD_coeffs["CD_S2"]
        self.CDSp = self.CD_coeffs["CD_Spbar"]
        self.CDq = self.CD_coeffs["CD_qbar"]
        self.CDLq = self.CD_coeffs["CD_Lqbar"]
        self.CDL2q = self.CD_coeffs["CD_L2qbar"]
        self.CDSr = self.CD_coeffs["CD_Srbar"]
        self.CDde = self.CD_coeffs["CD_de"]
        self.CDLde = self.CD_coeffs["CD_Lde"]
        self.CDde2 = self.CD_coeffs["CD_de2"]
        self.CDSda = self.CD_coeffs["CD_Sda"]
        self.CDSdr = self.CD_coeffs["CD_Sdr"]
        self.Clb = self.Cl_coeffs["Cl_beta"]
        self.Clp = self.Cl_coeffs["Cl_pbar"]
        self.Clr = self.Cl_coeffs["Cl_rbar"]
        self.ClLr = self.Cl_coeffs["Cl_Lrbar"]
        self.Clda = self.Cl_coeffs["Cl_da"]
        self.Cldr = self.Cl_coeffs["Cl_dr"]
        self.Cm0 = self.Cm_coeffs["Cm_0"]
        self.Cma = self.Cm_coeffs["Cm_alpha"]
        self.Cmq = self.Cm_coeffs["Cm_qbar"]
        self.Cmde = self.Cm_coeffs["Cm_de"]
        self.Cnb = self.Cn_coeffs["Cn_beta"]
        self.Cnp = self.Cn_coeffs["Cn_pbar"]
        self.CnLp = self.Cn_coeffs["Cn_Lpbar"]
        self.Cnr = self.Cn_coeffs["Cn_rbar"]
        self.Cnda = self.Cn_coeffs["Cn_da"]
        self.CnLda = self.Cn_coeffs["Cn_Lda"]
        self.Cndr = self.Cn_coeffs["Cn_dr"]

        # initialize thrust model
        self.Prop = Propulsion(inp_dir=kwargs.get("thrust_dir",inp_dir),**kwargs)

        # store stall model characteristics
        stall_model = self.model_coeffs_dict.get("stall_model",{})
        self.S_M = stall_model.get("blending_rate",7.0)
        self.S_ab = deg2rad(stall_model.get("stall_transition[deg]",45.0))

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

    def _reevaluate_coeffs(self):
        self.CL0 = self.CL_coeffs["CL_0"]
        self.CLa = self.CL_coeffs["CL_alpha"]
        self.CLq = self.CL_coeffs["CL_qbar"]
        self.CLde = self.CL_coeffs["CL_de"]
        self.CSb = self.CS_coeffs["CS_beta"]
        self.CSp = self.CS_coeffs["CS_pbar"]
        self.CSLp = self.CS_coeffs["CS_Lpbar"]
        self.CSr = self.CS_coeffs["CS_rbar"]
        self.CSda = self.CS_coeffs["CS_da"]
        self.CSdr = self.CS_coeffs["CS_dr"]
        self.CD0 = self.CD_coeffs["CD_0"]
        self.CDL = self.CD_coeffs["CD_L"]
        self.CDL2 = self.CD_coeffs["CD_L2"]
        self.CDS2 = self.CD_coeffs["CD_S2"]
        self.CDSp = self.CD_coeffs["CD_Spbar"]
        self.CDq = self.CD_coeffs["CD_qbar"]
        self.CDLq = self.CD_coeffs["CD_Lqbar"]
        self.CDL2q = self.CD_coeffs["CD_L2qbar"]
        self.CDSr = self.CD_coeffs["CD_Srbar"]
        self.CDde = self.CD_coeffs["CD_de"]
        self.CDLde = self.CD_coeffs["CD_Lde"]
        self.CDde2 = self.CD_coeffs["CD_de2"]
        self.CDSda = self.CD_coeffs["CD_Sda"]
        self.CDSdr = self.CD_coeffs["CD_Sdr"]
        self.Clb = self.Cl_coeffs["Cl_beta"]
        self.Clp = self.Cl_coeffs["Cl_pbar"]
        self.Clr = self.Cl_coeffs["Cl_rbar"]
        self.ClLr = self.Cl_coeffs["Cl_Lrbar"]
        self.Clda = self.Cl_coeffs["Cl_da"]
        self.Cldr = self.Cl_coeffs["Cl_dr"]
        self.Cm0 = self.Cm_coeffs["Cm_0"]
        self.Cma = self.Cm_coeffs["Cm_alpha"]
        self.Cmq = self.Cm_coeffs["Cm_qbar"]
        self.Cmde = self.Cm_coeffs["Cm_de"]
        self.Cnb = self.Cn_coeffs["Cn_beta"]
        self.Cnp = self.Cn_coeffs["Cn_pbar"]
        self.CnLp = self.Cn_coeffs["Cn_Lpbar"]
        self.Cnr = self.Cn_coeffs["Cn_rbar"]
        self.Cnda = self.Cn_coeffs["Cn_da"]
        self.CnLda = self.Cn_coeffs["Cn_Lda"]
        self.Cndr = self.Cn_coeffs["Cn_dr"]

    def _CL(self, alpha, beta, pbar, qbar, rbar, da, de, dr):
        CL = (self.CL0 + self.CLa*alpha + self.CLq*qbar + self.CLde*de)
        return CL

    def _CS(self, alpha, beta, pbar, qbar, rbar, da, de, dr):
        CL1 = self._CL(alpha, 0., 0., 0., 0., 0., 0., 0.)
        CS = (self.CSb*beta + (self.CSp + self.CSLp*CL1)*pbar +
              self.CSr*rbar + self.CSda*da + self.CSdr*dr)
        return CS

    def _CD(self, alpha, beta, pbar, qbar, rbar, da, de, dr):
        CL1 = self._CL(alpha, 0., 0., 0., 0., 0., 0., 0.)
        CS1 = self._CS(0., beta, 0., 0., 0., 0., 0., 0.) ## Christian model
        # CS1 = self._CS(alpha, beta, pbar, qbar, rbar, da, de, dr) ## Hunsaker model
        CD = (self.CD0 + self.CDL*CL1 + self.CDL2*CL1**2 + self.CDS2*CS1**2 +
              (self.CDSp*CS1)*pbar +
              (self.CDq + self.CDLq*CL1 + self.CDL2q*CL1**2)*qbar +
              (self.CDSr*CS1)*rbar +
              (self.CDde + self.CDLde*CL1)*de + self.CDde2*de**2 +
              (self.CDSda*CS1)*da +
              (self.CDSdr*CS1)*dr)
        return CD

    def _Cl(self, alpha, beta, pbar, qbar, rbar, da, de, dr):
        CL1 = self._CL(alpha, 0., 0., 0., 0., 0., 0., 0.)
        Cl = (self.Clb*beta + self.Clp*pbar + (self.Clr + self.ClLr*CL1)*rbar +
              self.Clda*da + self.Cldr*dr)
        return Cl

    def _Cm(self, alpha, beta, pbar, qbar, rbar, da, de, dr):
        Cm = (self.Cm0 + self.Cma*alpha + self.Cmq*qbar + self.Cmde*de)
        return Cm

    def _Cn(self, alpha, beta, pbar, qbar, rbar, da, de, dr):
        CL1 = self._CL(alpha, 0., 0., 0., 0., 0., 0., 0.)
        Cn = (self.Cnb*beta + (self.Cnp + self.CnLp*CL1)*pbar + self.Cnr*rbar +
              (self.Cnda + self.CnLda*CL1)*da + self.Cndr*dr)
        return Cn

    def _inc_aero_results(self, alpha, beta, pbar, qbar, rbar, da, de, dr):
        params = alpha, beta, pbar, qbar, rbar, da, de, dr
        return [self._CL(*params), self._CS(*params), self._CD(*params),
                self._Cl(*params), self._Cm(*params), self._Cn(*params)]
    
    def _stall_correction(self,a,CL,CD,Cm):
        # determine flat plate forces and moment
        CLplate = 2. * sign(a) * sin(a)**2. * cos(a)
        CDplate = 2. * sin(abs(a))**1.5
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
        CLplate = 2. * sign(a) * sin(a)**2. * cos(a)
        CDplate = 2. * sin(abs(a))**1.5
        Cmplate = -0.8 * sin(a)

        # determine stall sigmoid
        expMmin = exp(-self.S_M*(a-self.S_ab))
        expMplu = exp(self.S_M*(a+self.S_ab))
        sig = (1. + expMmin + expMplu) / (1. + expMmin) / (1. + expMplu)

        # derivatives
        CLplate_a = sign(a) * sin(a) * (3.*cos(2.*a) + 1.)
        CDplate_a = 3.*a/abs(a)*sin(abs(a))**0.5 if a != 0. else 0.
        Cmplate_a = -0.8*cos(a)

        # stall sigmoid derivative
        expMpmn = exp(self.S_M*(a-self.S_ab))
        expMa = exp(2.*self.S_M*a)
        sig_a = (self.S_M*expMplu*(expMa -1.))/\
            ((1. + expMpmn)**2.*(1. + expMplu)**2.)

        return CLplate,CDplate,Cmplate,sig,CLplate_a,CDplate_a,Cmplate_a,sig_a

    def _Anderson_correction(self, coeff, Lambda, RA, M):
        num = coeff*cos(Lambda)
        denom_2 = num/(pi*RA)
        denom_1 = (1. - M**2*cos(Lambda)**2 + denom_2**2)**0.5
        denom = denom_1 + denom_2
        return num/denom

    def _Prandtl_Glauert_subsonic_correction(self, coeff, M):
        return coeff / (1. - M**2.)**0.5

    def _Prandtl_Glauert_supersonic_correction(self, coeff, M):
        return coeff / (M**2. - 1.)**0.5

    def aero_results(self, alpha, beta, pbar, qbar, rbar, da, de, dr, 
    compressible=True, M=113.0, use_Anderson=True, enforce_stall=True):
        params = alpha, beta, pbar, qbar, rbar, da, de, dr

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
            elif M < 1.:
                if use_Anderson:
                    CL = self._Anderson_correction(CL,self.Lam_w,self.RA_w,M)
                    CS = self._Anderson_correction(CS,self.Lam_v,self.RA_v,M)
                    Cl = self._Anderson_correction(Cl,self.Lam_v,self.RA_v,M) # w w
                    Cm = self._Anderson_correction(Cm,self.Lam_w,self.RA_w,M)
                    Cn = self._Anderson_correction(Cn,self.Lam_v,self.RA_v,M)
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
        CX = -(CD*cos(alpha)*cos(beta) + CS*cos(alpha)*sin(beta) - CL*sin(alpha))
        CY = CS*cos(beta) - CD*sin(beta)
        CZ = -(CD*sin(alpha)*cos(beta) + CS*sin(alpha)*sin(beta) + CL*cos(alpha))

        if thrust_off == False:
            thrust = self.get_thrust(tau, H, V)
        else:
            thrust = 0.0

        Fx = CX*nondim_const + thrust
        Fy = CY*nondim_const
        Fz = CZ*nondim_const
        Mx = Cl*nondim_const*self.b_w - Fz*y_shift + Fy*z_shift
        My = Cm*nondim_const*self.c_w - Fx*z_shift + Fz*x_shift
        Mz = Cn*nondim_const*self.b_w - Fy*x_shift + Fx*y_shift
        FM = [Fx, Fy, Fz, Mx, My, Mz]
        return FM

    def report_coefficients_for_latex_table(self):

        t = "    "
        print()

        # run through coefficients, report
        for coeff in self.model_coeffs_dict:
            coeff_base = coeff[0] + "_{"
            if coeff == "Cell":
                coeff_base = coeff_base + "\ell"
                print()
            else:
                coeff_base = coeff_base + coeff[1]
            
            coeff_dict = self.model_coeffs_dict[coeff]
            for coeff_der in coeff_dict:
                der = coeff_der.split("_")[1]

                # replace certain things
                der = der.replace("alpha","\\alpha")
                der = der.replace("beta","\\beta")
                der = der.replace("pbar","\\bar{p}")
                der = der.replace("qbar","\\bar{q}")
                der = der.replace("rbar","\\bar{r}")
                der = der.replace("de","\\delta_s")
                der = der.replace("da","\\delta_f")
                der = der.replace("dr","\\delta_r")
                der = der.replace("2","^2")
                
                if der == "0":
                    der = "_" + der
                    use_comma = False
                else:
                    use_comma = True
                full_coeff = "$" + coeff_base + ","*use_comma + der + "}" + "$"

                print("{}{}{:<20} & {:> 7.4f} \\\\".format(t,t,\
                    full_coeff,coeff_dict[coeff_der]))
        print()



if __name__ == "__main__":
    case = F16Aero()
    # params = deg2rad([10., 10., 10., 10., 10., 10., 10., 10.])
    V = 634.
    bw = 30
    cw = 11.32
    params = deg2rad([6.488734, 0.01528474, (-0.2848369)*bw*0.5/V, (4.320014)*cw*0.5/V, (2.494161)*bw*0.5/V, 0.5335053, -2.829207, -0.8212136])
    [CL, CS, CD, Cl, Cm, Cn] = case.aero_results(*params, compressible=True, enforce_stall=True, M=0.59960910465664)
    print(CL, CS, CD, Cl, Cm, Cn)

