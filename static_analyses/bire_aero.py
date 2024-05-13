#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 15:15:06 2021

@author: christian
"""

import numpy as np
import json
import scipy.optimize as optimize

class BIREAero:
    def __init__(self, inp_dir='./'):
        self.model_coeffs_dict = json.load(open(inp_dir + 'bire_model_adj.json'))
        self.CL_coeffs = self.model_coeffs_dict["CL"]
        self.CS_coeffs = self.model_coeffs_dict["CS"]
        self.CD_coeffs = self.model_coeffs_dict["CD"]
        self.Cl_coeffs = self.model_coeffs_dict["Cell"]
        self.Cm_coeffs = self.model_coeffs_dict["Cm"]
        self.Cn_coeffs = self.model_coeffs_dict["Cn"]
        self.deriv=False

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


    def _CL0(self, d_B):
        Cdict = self.CL_coeffs["CL_0"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        if not self.deriv:
            return A*np.sin(w*d_B + phi) + z
        else:
            return A*w*np.cos(w*d_B + phi)

    def _CL_alpha(self, d_B):
        Cdict = self.CL_coeffs["CL_alpha"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        if not self.deriv:
            return A*np.sin(w*d_B + phi) + z
        else:
            return A*w*np.cos(w*d_B + phi)

    def _CL_beta(self, d_B):
        Cdict = self.CL_coeffs["CL_beta"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        if not self.deriv:
            return A*np.sin(w*d_B + phi) + z
        else:
            return A*w*np.cos(w*d_B + phi)

    def _CL_pbar(self, d_B):
        Cdict = self.CL_coeffs["CL_pbar"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        if not self.deriv:
            return A*np.sin(w*d_B + phi) + z
        else:
            return A*w*np.cos(w*d_B + phi)

    def _CL_qbar(self, d_B, deriv=False):
        Cdict = self.CL_coeffs["CL_qbar"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        z += delta
        if not self.deriv:
            return A*np.sin(w*d_B + phi) + z
        else:
            return A*w*np.cos(w*d_B + phi)

    def _CL_rbar(self, d_B):
        Cdict = self.CL_coeffs["CL_rbar"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        if not self.deriv:
            return A*np.sin(w*d_B + phi) + z
        else:
            return A*w*np.cos(w*d_B + phi)

    def _CL_da(self, d_B):
        Cdict = self.CL_coeffs["CL_da"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        if not self.deriv:
            return A*np.sin(w*d_B + phi) + z
        else:
            return A*w*np.cos(w*d_B + phi)

    def _CL_de(self, d_B):
        Cdict = self.CL_coeffs["CL_de"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        if not self.deriv:
            return A*np.sin(w*d_B + phi) + z
        else:
            return A*w*np.cos(w*d_B + phi)

    def _CS0(self, d_B):
        Cdict = self.CS_coeffs["CS_0"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        if not self.deriv:
            return A*np.sin(w*d_B + phi) + z
        else:
            return A*w*np.cos(w*d_B + phi)

    def _CS_alpha(self, d_B):
        Cdict = self.CS_coeffs["CS_alpha"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        if not self.deriv:
            return A*np.sin(w*d_B + phi) + z
        else:
            return A*w*np.cos(w*d_B + phi)

    def _CS_beta(self, d_B):
        Cdict = self.CS_coeffs["CS_beta"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        z += delta
        if not self.deriv:
            return A*np.sin(w*d_B + phi) + z
        else:
            return A*w*np.cos(w*d_B + phi)

    def _CS_pbar(self, d_B):
        Cdict = self.CS_coeffs["CS_pbar"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        if not self.deriv:
            return A*np.sin(w*d_B + phi) + z
        else:
            return A*w*np.cos(w*d_B + phi)

    def _CS_Lpbar(self, d_B):
        Cdict = self.CS_coeffs["CS_Lpbar"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        if not self.deriv:
            return A*np.sin(w*d_B + phi) + z
        else:
            return A*w*np.cos(w*d_B + phi)

    def _CS_qbar(self, d_B):
        Cdict = self.CS_coeffs["CS_qbar"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        if not self.deriv:
            return A*np.sin(w*d_B + phi) + z
        else:
            return A*w*np.cos(w*d_B + phi)

    def _CS_rbar(self, d_B):
        Cdict = self.CS_coeffs["CS_rbar"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        z += delta
        if not self.deriv:
            return A*np.sin(w*d_B + phi) + z
        else:
            return A*w*np.cos(w*d_B + phi)

    def _CS_de(self, d_B):
        Cdict = self.CS_coeffs["CS_de"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        if not self.deriv:
            return A*np.sin(w*d_B + phi) + z
        else:
            return A*w*np.cos(w*d_B + phi)

    def _CS_da(self, d_B):
        Cdict = self.CS_coeffs["CS_da"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        if not self.deriv:
            return A*np.sin(w*d_B + phi) + z
        else:
            return A*w*np.cos(w*d_B + phi)

    def _CD0(self, d_B):
        Cdict = self.CD_coeffs["CD_0"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        A = A*sigma
        z = z*sigma
        if not self.deriv:
            return A*np.sin(w*d_B + phi) + z
        else:
            return A*w*np.cos(w*d_B + phi)

    def _CD_L(self, d_B):
        Cdict = self.CD_coeffs["CD_L"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        z += delta
        if not self.deriv:
            return A*np.sin(w*d_B + phi) + z
        else:
            return A*w*np.cos(w*d_B + phi)

    def _CD_L2(self, d_B):
        Cdict = self.CD_coeffs["CD_L2"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        z += delta
        if not self.deriv:
            return A*np.sin(w*d_B + phi) + z
        else:
            return A*w*np.cos(w*d_B + phi)

    def _CD_S(self, d_B):
        Cdict = self.CD_coeffs["CD_S"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        if not self.deriv:
            return A*np.sin(w*d_B + phi) + z
        else:
            return A*w*np.cos(w*d_B + phi)

    def _CD_S2(self, d_B):
        Cdict = self.CD_coeffs["CD_S2"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        if not self.deriv:
            return A*np.sin(w*d_B + phi) + z
        else:
            return A*w*np.cos(w*d_B + phi)

    def _CD_pbar(self, d_B):
        Cdict = self.CD_coeffs["CD_pbar"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        if not self.deriv:
            return A*np.sin(w*d_B + phi) + z
        else:
            return A*w*np.cos(w*d_B + phi)

    def _CD_Spbar(self, d_B):
        Cdict = self.CD_coeffs["CD_Spbar"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        if not self.deriv:
            return A*np.sin(w*d_B + phi) + z
        else:
            return A*w*np.cos(w*d_B + phi)

    def _CD_qbar(self, d_B):
        Cdict = self.CD_coeffs["CD_qbar"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        z += delta
        if not self.deriv:
            return A*np.sin(w*d_B + phi) + z
        else:
            return A*w*np.cos(w*d_B + phi)

    def _CD_Lqbar(self, d_B):
        Cdict = self.CD_coeffs["CD_Lqbar"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        z += delta
        if not self.deriv:
            return A*np.sin(w*d_B + phi) + z
        else:
            return A*w*np.cos(w*d_B + phi)

    def _CD_L2qbar(self, d_B):
        Cdict = self.CD_coeffs["CD_L2qbar"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        z += delta
        if not self.deriv:
            return A*np.sin(w*d_B + phi) + z
        else:
            return A*w*np.cos(w*d_B + phi)

    def _CD_rbar(self, d_B):
        Cdict = self.CD_coeffs["CD_rbar"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        if not self.deriv:
            return A*np.sin(w*d_B + phi) + z
        else:
            return A*w*np.cos(w*d_B + phi)

    def _CD_Srbar(self, d_B):
        Cdict = self.CD_coeffs["CD_Srbar"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        z += delta
        if not self.deriv:
            return A*np.sin(w*d_B + phi) + z
        else:
            return A*w*np.cos(w*d_B + phi)

    def _CD_da(self, d_B):
        Cdict = self.CD_coeffs["CD_da"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        if not self.deriv:
            return A*np.sin(w*d_B + phi) + z
        else:
            return A*w*np.cos(w*d_B + phi)

    def _CD_Sda(self, d_B):
        Cdict = self.CD_coeffs["CD_Sda"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        if not self.deriv:
            return A*np.sin(w*d_B + phi) + z
        else:
            return A*w*np.cos(w*d_B + phi)

    def _CD_de(self, d_B):
        Cdict = self.CD_coeffs["CD_de"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        z += delta
        if not self.deriv:
            return A*np.sin(w*d_B + phi) + z
        else:
            return A*w*np.cos(w*d_B + phi)

    def _CD_Lde(self, d_B):
        Cdict = self.CD_coeffs["CD_Lde"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        if not self.deriv:
            return A*np.sin(w*d_B + phi) + z
        else:
            return A*w*np.cos(w*d_B + phi)

    def _CD_de2(self, d_B):
        Cdict = self.CD_coeffs["CD_de2"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        if not self.deriv:
            return A*np.sin(w*d_B + phi) + z
        else:
            return A*w*np.cos(w*d_B + phi)

    def _Cl0(self, d_B):
        Cdict = self.Cl_coeffs["Cl_0"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        if not self.deriv:
            return A*np.sin(w*d_B + phi) + z
        else:
            return A*w*np.cos(w*d_B + phi)

    def _Cl_alpha(self, d_B):
        Cdict = self.Cl_coeffs["Cl_alpha"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        if not self.deriv:
            return A*np.sin(w*d_B + phi) + z
        else:
            return A*w*np.cos(w*d_B + phi)

    def _Cl_beta(self, d_B):
        Cdict = self.Cl_coeffs["Cl_beta"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        z += delta
        if not self.deriv:
            return A*np.sin(w*d_B + phi) + z
        else:
            return A*w*np.cos(w*d_B + phi)

    def _Cl_pbar(self, d_B):
        Cdict = self.Cl_coeffs["Cl_pbar"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        if not self.deriv:
            return A*np.sin(w*d_B + phi) + z
        else:
            return A*w*np.cos(w*d_B + phi)

    def _Cl_qbar(self, d_B):
        Cdict = self.Cl_coeffs["Cl_qbar"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        if not self.deriv:
            return A*np.sin(w*d_B + phi) + z
        else:
            return A*w*np.cos(w*d_B + phi)

    def _Cl_rbar(self, d_B):
        Cdict = self.Cl_coeffs["Cl_rbar"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        if not self.deriv:
            return A*np.sin(w*d_B + phi) + z
        else:
            return A*w*np.cos(w*d_B + phi)

    def _Cl_Lrbar(self, d_B):
        Cdict = self.Cl_coeffs["Cl_Lrbar"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        if not self.deriv:
            return A*np.sin(w*d_B + phi) + z
        else:
            return A*w*np.cos(w*d_B + phi)

    def _Cl_da(self, d_B):
        Cdict = self.Cl_coeffs["Cl_da"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        z += delta
        if not self.deriv:
            return A*np.sin(w*d_B + phi) + z
        else:
            return A*w*np.cos(w*d_B + phi)

    def _Cl_de(self, d_B):
        Cdict = self.Cl_coeffs["Cl_de"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        if not self.deriv:
            return A*np.sin(w*d_B + phi) + z
        else:
            return A*w*np.cos(w*d_B + phi)

    def _Cm0(self, d_B):
        Cdict = self.Cm_coeffs["Cm_0"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        z += delta
        if not self.deriv:
            return A*np.sin(w*d_B + phi) + z
        else:
            return A*w*np.cos(w*d_B + phi)

    def _dCm0_dB(self, d_B):
        Cdict = self.Cm_coeffs["Cm_0"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        return A*w*np.cos(w*d_B + phi)

    def _Cm_alpha(self, d_B):
        Cdict = self.Cm_coeffs["Cm_alpha"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        if not self.deriv:
            return A*np.sin(w*d_B + phi) + z
        else:
            return A*w*np.cos(w*d_B + phi)

    def _dCma_dB(self, d_B):
        Cdict = self.Cm_coeffs["Cm_alpha"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        return A*w*np.cos(w*d_B + phi)

    def _Cm_beta(self, d_B):
        Cdict = self.Cm_coeffs["Cm_beta"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        if not self.deriv:
            return A*np.sin(w*d_B + phi) + z
        else:
            return A*w*np.cos(w*d_B + phi)

    def _dCmb_dB(self, d_B):
        Cdict = self.Cm_coeffs["Cm_beta"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        return A*w*np.cos(w*d_B + phi)

    def _Cm_pbar(self, d_B):
        Cdict = self.Cm_coeffs["Cm_pbar"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        if not self.deriv:
            return A*np.sin(w*d_B + phi) + z
        else:
            return A*w*np.cos(w*d_B + phi)

    def _dCmp_dB(self, d_B):
        Cdict = self.Cm_coeffs["Cm_pbar"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        return A*w*np.cos(w*d_B + phi)

    def _Cm_qbar(self, d_B):
        Cdict = self.Cm_coeffs["Cm_qbar"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        z += delta
        if not self.deriv:
            return A*np.sin(w*d_B + phi) + z
        else:
            return A*w*np.cos(w*d_B + phi)

    def _dCmq_dB(self, d_B):
        Cdict = self.Cm_coeffs["Cm_qbar"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        return A*w*np.cos(w*d_B + phi)

    def _Cm_rbar(self, d_B):
        Cdict = self.Cm_coeffs["Cm_rbar"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        if not self.deriv:
            return A*np.sin(w*d_B + phi) + z
        else:
            return A*w*np.cos(w*d_B + phi)

    def _dCmr_dB(self, d_B):
        Cdict = self.Cm_coeffs["Cm_rbar"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        return A*w*np.cos(w*d_B + phi)

    def _Cm_da(self, d_B):
        Cdict = self.Cm_coeffs["Cm_da"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        if not self.deriv:
            return A*np.sin(w*d_B + phi) + z
        else:
            return A*w*np.cos(w*d_B + phi)

    def _dCmda_dB(self, d_B):
        Cdict = self.Cm_coeffs["Cm_da"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        return A*w*np.cos(w*d_B + phi)

    def _Cm_de(self, d_B):
        Cdict = self.Cm_coeffs["Cm_de"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        if not self.deriv:
            return A*np.sin(w*d_B + phi) + z
        else:
            return A*w*np.cos(w*d_B + phi)

    def _dCmde_dB(self, d_B):
        Cdict = self.Cm_coeffs["Cm_de"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        return A*w*np.cos(w*d_B + phi)

    def _Cn0(self, d_B):
        Cdict = self.Cn_coeffs["Cn_0"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        if not self.deriv:
            return A*np.sin(w*d_B + phi) + z
        else:
            return A*w*np.cos(w*d_B + phi)

    def _dCn0_dB(self, d_B):
        Cdict = self.Cn_coeffs["Cn_0"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        return A*w*np.cos(w*d_B + phi)

    def _Cn_alpha(self, d_B):
        Cdict = self.Cn_coeffs["Cn_alpha"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        if not self.deriv:
            return A*np.sin(w*d_B + phi) + z
        else:
            return A*w*np.cos(w*d_B + phi)

    def _dCna_dB(self, d_B):
        Cdict = self.Cn_coeffs["Cn_alpha"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        return A*w*np.cos(w*d_B + phi)

    def _Cn_beta(self, d_B):
        Cdict = self.Cn_coeffs["Cn_beta"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        if not self.deriv:
            return A*np.sin(w*d_B + phi) + z
        else:
            return A*w*np.cos(w*d_B + phi)

    def _dCnb_dB(self, d_B):
        Cdict = self.Cn_coeffs["Cn_beta"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        return A*w*np.cos(w*d_B + phi)

    def _Cn_pbar(self, d_B):
        Cdict = self.Cn_coeffs["Cn_pbar"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        if not self.deriv:
            return A*np.sin(w*d_B + phi) + z
        else:
            return A*w*np.cos(w*d_B + phi)

    def _dCnp_dB(self, d_B):
        Cdict = self.Cn_coeffs["Cn_pbar"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        return A*w*np.cos(w*d_B + phi)

    def _Cn_Lpbar(self, d_B):
        Cdict = self.Cn_coeffs["Cn_Lpbar"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        if not self.deriv:
            return A*np.sin(w*d_B + phi) + z
        else:
            return A*w*np.cos(w*d_B + phi)

    def _dCnLp_dB(self, d_B):
        CnLdict = self.Cn_coeffs["Cn_Lpbar"]
        [A_nL, w_nL, phi_nL, z_nL] = [CnLdict[c] for c in CnLdict]
        CL0dict = self.CL_coeffs["CL_0"]
        [A_0, w_0, phi_0, z_0] = [CL0dict[c] for c in CL0dict]
        CLadict = self.CL_coeffs["CL_alpha"]
        [A_a, w_a, phi_a, z_a] = [CLadict[c] for c in CLadict]
        C1 = A_nL*w_nL*np.cos(w_nL*d_B + phi_nL)*(A_0*np.sin(w_0*d_B + phi_0) + z_0)
        C2 = A_0*w_0*np.cos(w_0*d_B + phi_0)*(A_nL*np.sin(w_nL*d_B + phi_nL) + z_nL)
        C3 = A_nL*w_nL*np.cos(w_nL*d_B + phi_nL)*(A_a*np.sin(w_a*d_B + phi_a) + z_a)
        C4 = A_a*w_a*np.cos(w_a*d_B + phi_a)*(A_nL*np.sin(w_nL*d_B + phi_nL) + z_nL)
        return [C1, C2, C3, C4]

    def _Cn_qbar(self, d_B):
        Cdict = self.Cn_coeffs["Cn_qbar"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        if not self.deriv:
            return A*np.sin(w*d_B + phi) + z
        else:
            return A*w*np.cos(w*d_B + phi)

    def _dCnq_dB(self, d_B):
        Cdict = self.Cn_coeffs["Cn_qbar"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        return A*w*np.cos(w*d_B + phi)

    def _Cn_rbar(self, d_B):
        Cdict = self.Cn_coeffs["Cn_rbar"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        z += delta
        if not self.deriv:
            return A*np.sin(w*d_B + phi) + z
        else:
            return A*w*np.cos(w*d_B + phi)

    def _dCnr_dB(self, d_B):
        Cdict = self.Cn_coeffs["Cn_rbar"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        return A*w*np.cos(w*d_B + phi)

    def _Cn_da(self, d_B):
        Cdict = self.Cn_coeffs["Cn_da"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        if not self.deriv:
            return A*np.sin(w*d_B + phi) + z
        else:
            return A*w*np.cos(w*d_B + phi)

    def _dCnda_dB(self, d_B):
        Cndadict = self.Cn_coeffs["Cn_da"]
        [A_da, w_da, phi_da, z_da] = [Cndadict[c] for c in Cndadict]
        CL0dict = self.CL_coeffs["CL_0"]
        [A_0, w_0, phi_0, z_0] = [CL0dict[c] for c in CL0dict]
        CLadict = self.CL_coeffs["CL_alpha"]
        [A_a, w_a, phi_a, z_a] = [CLadict[c] for c in CLadict]
        C1 = A_da*w_da*np.cos(w_da*d_B + phi_da)*(A_0*np.sin(w_0*d_B + phi_0) + z_0)
        C2 = A_0*w_0*np.cos(w_0*d_B + phi_0)*(A_da*np.sin(w_da*d_B + phi_da) + z_da)
        C3 = A_da*w_da*np.cos(w_da*d_B + phi_da)*(A_a*np.sin(w_a*d_B + phi_a) + z_a)
        C4 = A_a*w_a*np.cos(w_a*d_B + phi_a)*(A_da*np.sin(w_da*d_B + phi_da) + z_da)
        return [C1, C2, C3, C4]

    def _Cn_Lda(self, d_B):
        Cdict = self.Cn_coeffs["Cn_Lda"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        if not self.deriv:
            return A*np.sin(w*d_B + phi) + z
        else:
            return A*w*np.cos(w*d_B + phi)

    def _Cn_de(self, d_B):
        Cdict = self.Cn_coeffs["Cn_de"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        if not self.deriv:
            return A*np.sin(w*d_B + phi) + z
        else:
            return A*w*np.cos(w*d_B + phi)

    def _dCnde_dB(self, d_B):
        Cdict = self.Cn_coeffs["Cn_de"]
        [A, w, phi, z, sigma, delta] = [Cdict[c] for c in Cdict]
        return A*w*np.cos(w*d_B + phi)

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
              (self._CD_qbar(dB) + self._CD_Lqbar(dB)*CL1 + self._CD_L2qbar(dB)*CL1**2)*qbar +
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

    def aero_results(self, alpha, beta, pbar, qbar, rbar, da, de, dB):
        params = alpha, beta, pbar, qbar, rbar, da, de, dB
        return [self._CL(*params), self._CS(*params), self._CD(*params),
                self._Cl(*params), self._Cm(*params), self._Cn(*params)]

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
            phi = np.pi/2.
            z = 0.
            return A*np.sin(w*dB + phi) + z
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
            dCn0 = A0*np.cos(w0*dB + phi0)
            dCna = Aa*np.cos(wa*dB + phia)
            dCnb = Ab*np.cos(wb*dB + phib)
            dCnq = Aq*np.cos(wq*dB + phiq)
            dCnr = Ar*np.cos(wr*dB + phir)
            dCnde = Ade*np.cos(wde*dB + phide)
            Cn_dB = dCn0 + dCna*alpha + dCnb*beta + dCnq*qbar + dCnr*rbar + dCnde*de
            return Cn_dB

    def control_matrix(self,  alpha, beta, pbar, qbar, rbar, da, de, dB):
        A = np.zeros((2, 2))
        A[0, 0] = self._dCm_dB(alpha, beta, pbar, qbar, rbar, da, de, dB)
        A[0, 1] = self._Cm_de(dB)
        A[1, 0] = self._dCn_dB(alpha, beta, pbar, qbar, rbar, da, de, dB)
        A[1, 1] = self._Cn_de(dB)
        return A


if __name__ == "__main__":
    case = BIREAero()
    params = np.deg2rad([10., 10., 10., 10., 10., 10., 10., 10.])
    [CL, CS, CD, Cl, Cm, Cn] = case.aero_results(*params)
    print(CL, CS, CD, Cl, Cm, Cn)

