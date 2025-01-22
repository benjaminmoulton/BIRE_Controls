#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15

@author: Ben Moulton
"""

import json
from hunsaker_atm import stdatm_english, moulton_stdatm_derivative_english
from numpy import log as ln

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
        #
        if use_fitted_thrust_model:
            self.T_der_V   = self.T_der_V_fitted
            self.T_der_H   = self.T_der_H_fitted
            self.T_der_tau = self.T_der_tau_fitted
        else:
            self.T_der_V   = self.T_der_V_simple
            self.T_der_H   = self.T_der_H_simple
            self.T_der_tau = self.T_der_tau_simple
        
        # initialize atmosphere model
        self.atm_model = kwargs.get("atmosphere_model","use_hunsakers")
        self.atm_der_model = kwargs.get("atmosphere_derivative_model",
                                        "use_hunsakers")
        if self.atm_model == "use_hunsakers":
            self.atm_model = stdatm_english
            self.rho_i = 3
        else:
            self.rho_i = kwargs.get("rho_index_in_model")
        if self.atm_der_model == "use_hunsakers":
            self.atm_der_model = moulton_stdatm_derivative_english
            self.rho_der_i = 4
        else:
            self.rho_der_i = kwargs.get("rho_index_in_model")
        
        # initialize sea level rho
        self.rho_0 = self.atm_model(0.0)[self.rho_i]
        
        self.use_fitted_thrust_model = use_fitted_thrust_model
        if use_fitted_thrust_model:
            self.get_thrust = self.T_fitted
        else:
            self.get_thrust = self.T_simple


    def _a_idle(self,H):
        return self.a_idle_c0 + self.a_idle_c1*H + self.a_idle_c2*H*H

    def _a_idle_dH(self,H):
        return self.a_idle_c1 + 2.0*self.a_idle_c2*H

    def _T0_idle(self,H):
        return self.T0_idle_c0 + self.T0_idle_c1*H + self.T0_idle_c2*H*H

    def _T0_idle_dH(self,H):
        return self.T0_idle_c1 + 2.0*self.T0_idle_c2*H

    def _T1_idle(self,H):
        return self.T1_idle_c0 + self.T1_idle_c1*H + self.T1_idle_c2*H*H

    def _T1_idle_dH(self,H):
        return self.T1_idle_c1 + 2.0*self.T1_idle_c2*H

    def _T2_idle(self,H):
        return self.T2_idle_c0 + self.T2_idle_c1*H + self.T2_idle_c2*H*H

    def _T2_idle_dH(self,H):
        return self.T2_idle_c1 + 2.0*self.T2_idle_c2*H

    def idle_coefs(self,H):
        return self._a_idle(H),self._T0_idle(H),self._T1_idle(H),self._T2_idle(H)

    def idle_coefs_dH(self,H):
        return self._a_idle_dH(H),self._T0_idle_dH(H),self._T1_idle_dH(H),self._T2_idle_dH(H)

    def _a_mil(self,H):
        return self.a_mil_c0 + self.a_mil_c1*H + self.a_mil_c2*H*H

    def _a_mil_dH(self,H):
        return self.a_mil_c1 + 2.0*self.a_mil_c2*H

    def _T0_mil(self,H):
        return self.T0_mil_c0 + self.T0_mil_c1*H + self.T0_mil_c2*H*H

    def _T0_mil_dH(self,H):
        return self.T0_mil_c1 + 2.0*self.T0_mil_c2*H

    def _T1_mil(self,H):
        return self.T1_mil_c0 + self.T1_mil_c1*H + self.T1_mil_c2*H*H

    def _T1_mil_dH(self,H):
        return self.T1_mil_c1 + 2.0*self.T1_mil_c2*H

    def _T2_mil(self,H):
        return self.T2_mil_c0 + self.T2_mil_c1*H + self.T2_mil_c2*H*H

    def _T2_mil_dH(self,H):
        return self.T2_mil_c1 + 2.0*self.T2_mil_c2*H

    def mil_coefs(self,H):
        return self._a_mil(H),self._T0_mil(H),self._T1_mil(H),self._T2_mil(H)

    def mil_coefs_dH(self,H):
        return self._a_mil_dH(H),self._T0_mil_dH(H),self._T1_mil_dH(H),self._T2_mil_dH(H)


    def _a_max(self,H):
        return self.a_max_c0 + self.a_max_c1*H + self.a_max_c2*H*H

    def _a_max_dH(self,H):
        return self.a_max_c1 + 2.0*self.a_max_c2*H

    def _T0_max(self,H):
        return self.T0_max_c0 + self.T0_max_c1*H + self.T0_max_c2*H*H

    def _T0_max_dH(self,H):
        return self.T0_max_c1 + 2.0*self.T0_max_c2*H

    def _T1_max(self,H):
        return self.T1_max_c0 + self.T1_max_c1*H + self.T1_max_c2*H*H

    def _T1_max_dH(self,H):
        return self.T1_max_c1 + 2.0*self.T1_max_c2*H

    def _T2_max(self,H):
        return self.T2_max_c0 + self.T2_max_c1*H + self.T2_max_c2*H*H

    def _T2_max_dH(self,H):
        return self.T2_max_c1 + 2.0*self.T2_max_c2*H

    def max_coefs(self,H):
        return self._a_max(H),self._T0_max(H),self._T1_max(H),self._T2_max(H)

    def max_coefs_dH(self,H):
        return self._a_max_dH(H),self._T0_max_dH(H),self._T1_max_dH(H),self._T2_max_dH(H)


    def _T_idle(self,rho,V,H):
        # get coefficients
        a  =  self.a_idle_c0 +  self.a_idle_c1*H +  self.a_idle_c2*H*H
        T0 = self.T0_idle_c0 + self.T0_idle_c1*H + self.T0_idle_c2*H*H
        T1 = self.T1_idle_c0 + self.T1_idle_c1*H + self.T1_idle_c2*H*H
        T2 = self.T2_idle_c0 + self.T2_idle_c1*H + self.T2_idle_c2*H*H
        return (rho/self.rho_0)**a*(T0 + T1*V + T2*V*V)

    def _T_idle_dH(self,rho,V,H,rho_H):
        # get coefficients
        a  =  self.a_idle_c0 +  self.a_idle_c1*H +  self.a_idle_c2*H*H
        T0 = self.T0_idle_c0 + self.T0_idle_c1*H + self.T0_idle_c2*H*H
        T1 = self.T1_idle_c0 + self.T1_idle_c1*H + self.T1_idle_c2*H*H
        T2 = self.T2_idle_c0 + self.T2_idle_c1*H + self.T2_idle_c2*H*H
        #
        a_H  =  self.a_idle_c1 + 2.0* self.a_idle_c2*H
        T0_H = self.T0_idle_c1 + 2.0*self.T0_idle_c2*H
        T1_H = self.T1_idle_c1 + 2.0*self.T1_idle_c2*H
        T2_H = self.T2_idle_c1 + 2.0*self.T2_idle_c2*H
        #
        Tpart = T0 + T1*V + T2*V*V
        #
        return (rho/self.rho_0)**a*(ln(rho/self.rho_0)*a_H + a*rho_H/rho)*Tpart \
            + (rho/self.rho_0)**a*(T0_H + T1_H*V + T2_H*V*V)

    def _T_mil(self,rho,V,H):
        # get coefficients
        a =   self.a_mil_c0 +  self.a_mil_c1*H +  self.a_mil_c2*H*H
        T0 = self.T0_mil_c0 + self.T0_mil_c1*H + self.T0_mil_c2*H*H
        T1 = self.T1_mil_c0 + self.T1_mil_c1*H + self.T1_mil_c2*H*H
        T2 = self.T2_mil_c0 + self.T2_mil_c1*H + self.T2_mil_c2*H*H
        return (rho/self.rho_0)**a*(T0 + T1*V + T2*V*V)

    def _T_mil_dH(self,rho,V,H,rho_H):
        # get coefficients
        a  =  self.a_mil_c0 +  self.a_mil_c1*H +  self.a_mil_c2*H*H
        T0 = self.T0_mil_c0 + self.T0_mil_c1*H + self.T0_mil_c2*H*H
        T1 = self.T1_mil_c0 + self.T1_mil_c1*H + self.T1_mil_c2*H*H
        T2 = self.T2_mil_c0 + self.T2_mil_c1*H + self.T2_mil_c2*H*H
        #
        a_H  =  self.a_mil_c1 + 2.0* self.a_mil_c2*H
        T0_H = self.T0_mil_c1 + 2.0*self.T0_mil_c2*H
        T1_H = self.T1_mil_c1 + 2.0*self.T1_mil_c2*H
        T2_H = self.T2_mil_c1 + 2.0*self.T2_mil_c2*H
        #
        Tpart = T0 + T1*V + T2*V*V
        #
        return (rho/self.rho_0)**a*(ln(rho/self.rho_0)*a_H + a*rho_H/rho)*Tpart \
            + (rho/self.rho_0)**a*(T0_H + T1_H*V + T2_H*V*V)

    def _T_max(self,rho,V,H):
        # get coefficients
        a =   self.a_max_c0 +  self.a_max_c1*H +  self.a_max_c2*H*H
        T0 = self.T0_max_c0 + self.T0_max_c1*H + self.T0_max_c2*H*H
        T1 = self.T1_max_c0 + self.T1_max_c1*H + self.T1_max_c2*H*H
        T2 = self.T2_max_c0 + self.T2_max_c1*H + self.T2_max_c2*H*H
        return (rho/self.rho_0)**a*(T0 + T1*V + T2*V*V)

    def _T_max_dH(self,rho,V,H,rho_H):
        # get coefficients
        a  =  self.a_max_c0 +  self.a_max_c1*H +  self.a_max_c2*H*H
        T0 = self.T0_max_c0 + self.T0_max_c1*H + self.T0_max_c2*H*H
        T1 = self.T1_max_c0 + self.T1_max_c1*H + self.T1_max_c2*H*H
        T2 = self.T2_max_c0 + self.T2_max_c1*H + self.T2_max_c2*H*H
        #
        a_H  =  self.a_max_c1 + 2.0* self.a_max_c2*H
        T0_H = self.T0_max_c1 + 2.0*self.T0_max_c2*H
        T1_H = self.T1_max_c1 + 2.0*self.T1_max_c2*H
        T2_H = self.T2_max_c1 + 2.0*self.T2_max_c2*H
        #
        Tpart = T0 + T1*V + T2*V*V
        #
        return (rho/self.rho_0)**a*(ln(rho/self.rho_0)*a_H + a*rho_H/rho)*Tpart \
            + (rho/self.rho_0)**a*(T0_H + T1_H*V + T2_H*V*V)


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
            return T_idle + (T_mil - T_idle)*P1/50.
        else:
            T_max = self._T_max(rho,V,H)
            return T_mil + (T_max - T_mil)*(P1-50.)/50.


    def T_simple(self,tau,H,V):
        # calculate rho
        rho = self.atm_model(H)[self.rho_i]
        return tau*(rho/self.rho_0)**self.a*(self.T0 + self.T1*V + self.T2*V*V)


    def T_der_V_fitted(self,tau,H,V):
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
        #
        # pull out each setting derivative
        la,_,lT1,lT2 = self.mil_coefs(H)
        Tmil_V = (rho/self.rho_0)**la*(lT1 + 2.*lT2*V)
        # get full derivative
        if P1 < 50.:
            ia,_,iT1,iT2 = self.idle_coefs(H)
            Tidle_V = (rho/self.rho_0)**ia*(iT1 + 2.*iT2*V)
            return Tidle_V + (Tmil_V - Tidle_V)*P1/50.
        else:
            ma,_,mT1,mT2 = self.max_coefs(H)
            Tmax_V = (rho/self.rho_0)**ma*(mT1 + 2.*mT2*V)
            return Tmil_V + (Tmax_V - Tmil_V)*(P1-50.)/50.


    def T_der_H_fitted(self,tau,H,V):
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
        rho_H = self.atm_der_model(H)[self.rho_der_i]
        #
        # get total thrust
        T_mil_dH = self._T_mil_dH(rho,V,H,rho_H)
        if P1 < 50.:
            T_idle_dH = self._T_idle_dH(rho,V,H,rho_H)
            return T_idle_dH + (T_mil_dH - T_idle_dH)*P1/50.
        else:
            T_max_dH = self._T_max_dH(rho,V,H,rho_H)
            return T_mil_dH + (T_max_dH - T_mil_dH)*(P1-50.)/50.


    def T_der_tau_fitted(self,tau,H,V):
        # calculate P1
        if 0.0 <= tau <= 0.77:
            P1_tau = 64.94
        elif 0.77 < tau <= 1.0:
            P1_tau = 217.38
        elif tau <= 0.0:
            P1_tau = 0.0
        else:
            P1_tau = 217.38
        
        # keep above ground
        if H <= 0.:
            H = 0.
        
        # calculate rho
        rho = self.atm_model(H)[self.rho_i]
        
        # get total thrust
        T_mil = self._T_mil(rho,V,H)
        if tau <= 0.77:
            T_idle = self._T_idle(rho,V,H)
            return (T_mil - T_idle)*P1_tau/50.
        else:
            T_max = self._T_max(rho,V,H)
            return (T_max - T_mil)*P1_tau/50.


    def T_der_V_simple(self,tau,H,V):
        rho = self.atm_model(H)[self.rho_i]
        return tau*(rho/self.rho_0)**self.a*(self.T1 + 2.*self.T2*V)


    def T_der_H_simple(self,tau,H,V):
        rho   = self.atm_model    (H)[self.rho_i]
        rho_H = self.atm_der_model(H)[self.rho_der_i]
        return tau*(
            self.a*(rho/self.rho_0)**(self.a - 1.0)*rho_H/self.rho_0
            )*(self.T0 + self.T1*V + self.T2*V*V)


    def T_der_tau_simple(self,tau,H,V):
        # calculate rho
        rho = self.atm_model(H)[self.rho_i]
        return (rho/self.rho_0)**self.a*(self.T0 + self.T1*V + self.T2*V*V)


    def T_V_H_ders(self,tau,H,V):
        return self.T_der_V(tau,H,V),self.T_der_H(tau,H,V)

if __name__ == "__main__":
    P = Propulsion()
    tau = 0.5
    H = 15_000.
    V = 634.
    print(P.get_thrust(tau,H,V))
    print(P.T_V_H_ders(tau,H,V))
    print(P.T_der_tau(tau,H,V))

    P_simp = Propulsion(use_fitted_thrust_model=False)
    tau = 0.5
    H = 15_000.
    V = 634.
    print(P_simp.get_thrust(tau,H,V))
    print(P.T_V_H_ders(tau,H,V))
    print(P.T_der_tau(tau,H,V))



