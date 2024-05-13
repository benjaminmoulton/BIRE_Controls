#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15

@author: Ben Moulton
"""

import json
from hunsaker_atm import stdatm_english

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
        if H <= 0.:
            H = 0.
        
        # calculate rho
        rho = self.atm_model(H)[self.rho_i]
        
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


if __name__ == "__main__":
    P = Propulsion()
    tau = 0.5
    H = 15_000.
    V = 634.
    print(P.get_thrust(tau,H,V))

    P_simp = Propulsion(use_fitted_thrust_model=False)
    tau = 0.5
    H = 15_000.
    V = 634.
    print(P_simp.get_thrust(tau,H,V))



