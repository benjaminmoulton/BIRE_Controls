#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 15:15:06 2021

@author: christian
"""

from math import sin,cos,pi,exp
from numpy import sign, deg2rad, zeros, array
import json
from thrust import Propulsion
from math import cos, sin, tan, atan2

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
            elif M < 1.:
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
    
    def report_coefficients_for_latex_table(self):

        t = "    "
        print()

        # run through coefficients, report
        for coeff in self.model_coeffs_dict:
            coeff_base = "\\hat{" + coeff[0] + "}_{"
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
                der = der.replace("de","\\delta_s^B")
                der = der.replace("da","\\delta_f")
                der = der.replace("dr","\\delta_r")
                der = der.replace("2","^2")
                
                if der == "0":
                    der = "_" + der
                    use_comma = False
                else:
                    use_comma = True
                if der[-4:] == "^B^2":
                    der = der[:-4] + "^{B \, 2}"
                full_coeff = "$" + coeff_base + ","*use_comma + der + "}" + "$"

                subdict = coeff_dict[coeff_der]
                A = subdict["A"]
                w = subdict["w"]
                p = subdict["phi"]
                z = subdict["z"] + subdict["delta"]

                if p == 0.0:
                    p_str = "{:> 7.0f}"
                else:
                    p_str = "{:> 7.4f}"
                if A == 0.0:
                    A_str = "{:> 7.0f}"
                else:
                    A_str = "{:> 7.4f}"
                if z == 0.0:
                    z_str = "{:> 7.0f}"
                else:
                    z_str = "{:> 7.4f}"

                print(("{}{}{:<32} & " + A_str + " & {:> 1.0f} & " + p_str \
                    + " & " + z_str + " \\\\").format(t,t,full_coeff,A,w,p,z))
        print()


if __name__ == "__main__":
    print("initializing...")
    case = BIREAero()

    params = deg2rad([10., 10., 10., 10., 10., 10., 10., 10.])
    M = 0.0
    [CL, CS, CD, Cl, Cm, Cn] = case.aero_results(M=M,*params)
    # print(CL, CS, CD, Cl, Cm, Cn)

    # # report coefficients in latex table format
    # case.report_coefficients_for_latex_table()

    # params = deg2rad([10., 10., 10., 10., 10., 10., 10., 10.])
    # [CL, CS, CD, Cl, Cm, Cn] = case.aero_results(*params)
    # print(CL, CS, CD, Cl, Cm, Cn)
    
    # params = deg2rad([2.873502, 0., 0., 0., 0., 0., -1.601589, 0.])
    # [CL, CS, CD, Cl, Cm, Cn] = case.aero_results(*params,compressible=True, enforce_stall=True, M=0.59960910465664)
    # print(CL, CS, CD, Cl, Cm, Cn)

    # from numpy.random import random as nprand
    # from time import time as tm
    # num = 48229500
    # num =     5000
    # dB = deg2rad( (nprand(size=(num,))*2. - 1.)*20. )
    # results = zeros((dB.shape[0],6))
    # a,b,da,de = deg2rad( (nprand(size=(4,))*2. - 1.)*20. )
    # pb,qb,rb = (nprand(size=(3,))*2. - 1.)*0.01


    # # test old
    # print("running...")
    # r_old = results*0.
    # start = tm()
    # for i in range(dB.shape[0]):
    #     r_old[i,0] = case._old_CL(a,b,pb,qb,rb,da,de,dB[i])
    #     r_old[i,1] = case._old_CS(a,b,pb,qb,rb,da,de,dB[i])
    #     r_old[i,2] = case._old_CD(a,b,pb,qb,rb,da,de,dB[i])
    #     r_old[i,3] = case._old_Cl(a,b,pb,qb,rb,da,de,dB[i])
    #     r_old[i,4] = case._old_Cm(a,b,pb,qb,rb,da,de,dB[i])
    #     r_old[i,5] = case._old_Cn(a,b,pb,qb,rb,da,de,dB[i])
    # t_old = tm() - start
    # print("report")
    # print("t old {:> 10.6f}".format(t_old))

    # # test new
    # r_new = results*0.
    # start = tm()
    # for i in range(dB.shape[0]):
    #     r_new[i,0] = case._CL(a,b,pb,qb,rb,da,de,dB[i])
    #     r_new[i,1] = case._CS(a,b,pb,qb,rb,da,de,dB[i])
    #     r_new[i,2] = case._CD(a,b,pb,qb,rb,da,de,dB[i])
    #     r_new[i,3] = case._Cl(a,b,pb,qb,rb,da,de,dB[i])
    #     r_new[i,4] = case._Cm(a,b,pb,qb,rb,da,de,dB[i])
    #     r_new[i,5] = case._Cn(a,b,pb,qb,rb,da,de,dB[i])
    # t_new = tm() - start
    # print("t new {:> 10.6f}".format(t_new))
    
    # # test old
    # print("running...")
    # C_old = results*0.
    # start = tm()
    # for i in range(dB.shape[0]):
    #     C_old[i,:] = case._old_inc_aero_results(a,b,pb,qb,rb,da,de,dB[i])
    # t_old = tm() - start
    # print("report")
    # print("C old {:> 10.6f}".format(t_old))

    # # test new
    # C_new = results*0.
    # start = tm()
    # for i in range(dB.shape[0]):
    #     C_new[i,:] = case._inc_aero_results(a,b,pb,qb,rb,da,de,dB[i])
    # t_new = tm() - start
    # print("C new {:> 10.6f}".format(t_new))

    # # report
    # print()
    # print("compare")
    # Cs = ["CL","CS","CD","Cl","Cm","Cn"]
    # for i in range(results.shape[1]):
    #     print("old and new {} {:> 10.6e}".format(Cs[i],max(r_old[:,i] - r_new[:,i])))
    # print("--------------")
    # for i in range(results.shape[1]):
    #     print("old CCC new {} {:> 10.6e}".format(Cs[i],max(C_old[:,i] - C_new[:,i])))



    # # test old
    # print()
    # print()
    # print("derivatives")
    # r_old = results*0.
    # start = tm()
    # for i in range(dB.shape[0]):
    #     r_old[i,4] = case._old_dCm_dB(a,b,pb,qb,rb,da,de,dB[i])
    #     r_old[i,5] = case._old_dCn_dB(a,b,pb,qb,rb,da,de,dB[i])
    # t_old = tm() - start
    # print("report")
    # print("t old {:> 10.6f}".format(t_old))

    # # test new
    # r_new = results*0.
    # start = tm()
    # for i in range(dB.shape[0]):
    #     r_new[i,4] = case._dCm_dB(a,b,pb,qb,rb,da,de,dB[i])
    #     r_new[i,5] = case._dCn_dB(a,b,pb,qb,rb,da,de,dB[i])
    # t_new = tm() - start
    # print("t new {:> 10.6f}".format(t_new))

    # # report
    # print()
    # print("compare")
    # for i in range(results.shape[1]):
    #     print("old and new {} {:> 10.6e}".format(Cs[i],max(r_old[:,i] - r_new[:,i])))
