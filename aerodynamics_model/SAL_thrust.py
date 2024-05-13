#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15

@author: Ben Moulton
"""

import json
from numpy import array as nparray, zeros as npzeros

class Propulsion:
    def __init__(self, inp_dir="./", **kwargs):
        fn = kwargs.get('thrust_model_file_name', '0')
        if fn == "0":
            fn = "SAL_thrust_model.json"
        self.model_coeffs_dict = json.load(open(inp_dir + fn))
        # read in tables
        thrust_dict = self.model_coeffs_dict.get("THRUST",{})
        self.THRUST_A = nparray(thrust_dict.get("A",npzeros((6,6)))).T
        self.THRUST_B = nparray(thrust_dict.get("B",npzeros((6,6)))).T # mil
        self.THRUST_C = nparray(thrust_dict.get("C",npzeros((6,6)))).T # max

    
    def get_thrust(self,POW,ALT,RMACH):
        # Engine thrust model
        #
        H = .0001*ALT
        I = int(H)
        if I >= 5: I = 4
        DH = H - float(I)
        RM = 5.0*RMACH
        M  = int(RM)
        if M >= 5: M = 4
        DM = RM - float(M)
        CDH = 1.0 - DH
        S= self.THRUST_B[I,M  ]*CDH + self.THRUST_B[I+1,M  ]*DH
        T= self.THRUST_B[I,M+1]*CDH + self.THRUST_B[I+1,M+1]*DH
        TMIL= S + (T - S)*DM
        #
        if POW <= 50.0:
            S= self.THRUST_A[I,M  ]*CDH + self.THRUST_A[I+1,M  ]*DH
            T= self.THRUST_A[I,M+1]*CDH + self.THRUST_A[I+1,M+1]*DH
            TIDL= S + (T - S)*DM
            THRUST=TIDL + (TMIL - TIDL)*POW*.02
        else:
            S= self.THRUST_C[I,M  ]*CDH + self.THRUST_C[I+1,M  ]*DH
            T= self.THRUST_C[I,M+1]*CDH + self.THRUST_C[I+1,M+1]*DH
            TMAX= S + (T - S)*DM
            THRUST=TMIL + (TMAX - TMIL)*(POW - 50.0)*.02
        
        return THRUST


def TGEAR(THTL):
    # Power command v. thtl. relationship
    if THTL <= 0.77:
        TGEAR = 64.94*THTL
    else:
        TGEAR = 217.38*THTL - 117.38
    
    return TGEAR

def RTAU(DP):
    # used by function PDOT
    #
    if DP <= 25.0:
        RTAU = 1.0 # reciprocal time constant
    elif DP >= 50.:
        RTAU = 0.1
    else:
        RTAU = 1.9 - 0.036*DP

    return RTAU

def PDOT(P3,P1):
    # PDOT = rate of change of power
    # P3 = actual power, P1 = power command
    if P1 >= 50.0:
        if P3 >= 50.0:
            T  = 5.0
            P2 = P1
        else:
            P2 = 60.0
            T  = RTAU(P2-P3)
    else:
        if P3 >= 50.0:
            T  = 5.0
            P2 = 40.0
        else:
            P2 = P1
            T  = RTAU(P2-P3)

    PDOT = T*(P2 - P3)
    
    return PDOT

if __name__ == "__main__":
    P = Propulsion()
    H = 15_000.
    M = 0.6
    POW = 70.0
    print(P.get_thrust(POW,H,M))


