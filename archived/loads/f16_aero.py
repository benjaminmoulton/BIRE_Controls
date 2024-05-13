#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 16:18:40 2021

@author: christian
"""

import numpy as np
import json

class F16Aero:
    def __init__(self, inp_dir='./'):
        self.model_coeffs_dict = json.load(open(inp_dir + 'mux_model_adj.json'))
        self.CL_coeffs = self.model_coeffs_dict["CL"]
        self.CS_coeffs = self.model_coeffs_dict["CS"]
        self.CD_coeffs = self.model_coeffs_dict["CD"]
        self.Cl_coeffs = self.model_coeffs_dict["Cell"]
        self.Cm_coeffs = self.model_coeffs_dict["Cm"]
        self.Cn_coeffs = self.model_coeffs_dict["Cn"]
        self.CL0 = self.CL_coeffs["CL_0"]
        self.CLa = self.CL_coeffs["CL_alpha"]
        self.CLq = self.CL_coeffs["CL_qbar"] + 22.5115
        self.CLde = self.CL_coeffs["CL_de"]
        self.CSb = self.CS_coeffs["CS_beta"] - 0.4679
        self.CSp = self.CS_coeffs["CS_pbar"]
        self.CSLp = self.CS_coeffs["CS_Lpbar"]
        self.CSr = self.CS_coeffs["CS_rbar"] + 0.2586
        self.CSda = self.CS_coeffs["CS_da"]
        self.CSdr = self.CS_coeffs["CS_dr"]
        self.CD0 = self.CD_coeffs["CD_0"]*2.5676
        self.CDL = self.CD_coeffs["CD_L"] - 0.0298
        self.CDL2 = self.CD_coeffs["CD_L2"] + 0.1127
        self.CDS2 = self.CD_coeffs["CD_S2"]
        self.CDSp = self.CD_coeffs["CD_Spbar"]
        self.CDq = self.CD_coeffs["CD_qbar"] - 0.8737
        self.CDLq = self.CD_coeffs["CD_Lqbar"] + 3.2152
        self.CDL2q = self.CD_coeffs["CD_L2qbar"] + 4.5003
        self.CDSr = self.CD_coeffs["CD_Srbar"] + 0.3992
        self.CDde = self.CD_coeffs["CD_de"] + 0.0021
        self.CDLde = self.CD_coeffs["CD_Lde"]
        self.CDde2 = self.CD_coeffs["CD_de2"]
        self.CDSda = self.CD_coeffs["CD_Sda"]
        self.CDSdr = self.CD_coeffs["CD_Sdr"]
        self.Clb = self.Cl_coeffs["Cl_beta"] - 0.0264
        self.Clp = self.Cl_coeffs["Cl_pbar"]
        self.Clr = self.Cl_coeffs["Cl_rbar"]
        self.ClLr = self.Cl_coeffs["Cl_Lrbar"]
        self.Clda = self.Cl_coeffs["Cl_da"] - 0.0487
        self.Cldr = self.Cl_coeffs["Cl_dr"]
        self.Cm0 = self.Cm_coeffs["Cm_0"] + 0.0256
        self.Cma = self.Cm_coeffs["Cm_alpha"]
        self.Cmq = self.Cm_coeffs["Cm_qbar"] - 0.3599
        self.Cmde = self.Cm_coeffs["Cm_de"]
        self.Cnb = self.Cn_coeffs["Cn_beta"]
        self.Cnp = self.Cn_coeffs["Cn_pbar"]
        self.CnLp = self.Cn_coeffs["Cn_Lpbar"]
        self.Cnr = self.Cn_coeffs["Cn_rbar"] - 0.1077
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
        CS1 = self._CS(0., beta, 0., 0., 0., 0., 0., 0.)
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

    def aero_results(self, alpha, beta, pbar, qbar, rbar, da, de, dr):
        params = alpha, beta, pbar, qbar, rbar, da, de, dr
        return [self._CL(*params), self._CS(*params), self._CD(*params),
                self._Cl(*params), self._Cm(*params), self._Cn(*params)]

if __name__ == "__main__":
    case = F16Aero()
    params = np.deg2rad([10., 10., 0., 0., 0., 0., 0., 0.])
    [CL, CS, CD, Cl, Cm, Cn] = case.aero_results(*params)
    print(CL, CS, CD, Cl, Cm, Cn)
