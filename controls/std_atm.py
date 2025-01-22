import numpy as np

def stdatm_si(H):
    """Given the geometric altitude (m), calculates and returns the
    geopotential altitude (m), gravitational acceleration (m/s^2),
    temperature (K), pressure (N/m^2), density (kg/m^3), and 
    speed of sound (m/s).

    Parameters
    ----------
    H : float
        geometric altitude (m).
    
    Returns
    -------
    Z : float
        geopotential altitude (m).
    
    G : float
        gravitational acceleration (m/s^2)
    
    T : float
        temperature (K).
    
    P : float
        pressure (N/m^2).
    
    R : float
        density (kg/m^3).
    
    A : float
        speed of sound (m/s).
    """

    # calculate geopotential altitude
    Z = (6356766.0 * H) / (6356766.0 + H)

    # calculate gravitational constant
    G = 3.96271559301397625e+14 / (6356766.0 + H)**2.
    # G = 9.80665 * (6356766.0 / (6356766.0 + H))**2. # Hunsaker gravity

    # check range, determine T and P
    if   Z <    0.0:
        Z = 0.0
        # calculate T and P
        T = 288.150
        P = 1.013250E+05
        # raise ValueError("Z is not within range 0 <= Z <= inf, Z={}".format(Z))
    elif Z < 11000.0:
        # calculate T and P
        T = 288.150 - 0.0065 * Z
        P = 1.013250E+05 * (T / 288.150) ** (5.2558784146492056E+00)
    elif Z < 20000.0:
        # calculate T and P
        T = 216.650
        P = 2.2632049118994407E+04 * np.exp(\
            -1.5768848232273173E-04 * Z + \
            1.7345733055500489E+00)
    elif Z < 32000.0:
        # calculate T and P
        T = 216.650 + 0.001  * Z - 2.00E+01
        P = 5.4748816740651100E+03 * (T / 216.650) ** \
            (-3.4163209695219833E+01)
    elif Z < 47000.0:
        # calculate T and P
        T = 228.650 + 0.0028 * Z - 8.96E+01
        P = 8.6801687564243229E+02 * (T / 228.650) ** \
            (-1.2201146319721370E+01)
    elif Z < 52000.0:
        # calculate T and P
        T = 270.650
        P = 1.1090597448788811E+02 * np.exp(\
            -1.2622652760103394E-04 * Z + \
            5.9326467972485952E+00)
    elif Z < 61000.0:
        # calculate T and P
        T = 270.650 - 0.002  * Z + 1.04E+02
        P = 5.9000748345616187E+01 * (T / 270.650) ** \
            (1.7081604847609916E+01)
    elif Z < 79000.0:
        # calculate T and P
        T = 252.650 - 0.004  * Z + 2.44E+02
        P = 1.8210004975660250E+01 * (T / 252.650) ** \
            (8.5408024238049582E+00)
    else:
        # calculate T and P
        T = 180.650
        P = 1.0377065335528297E+00 * np.exp(\
            -1.8911270243686593E-04 * Z + \
            1.4939903492512408E+01)

    # calculate density
    R = P / 287.0528 / T

    # calculate speed of sound
    A = (1.4 * 287.0528 * T)**0.5
    
    return Z,G,T,P,R,A


def stdatm_english(H):
    """Given the geometric altitude (ft), calculates and returns the
    geopotential altitude (ft), gravitational acceleration (ft/s^2), 
    temperature (deg R), pressure (lbf/ft^2), density (slugs/ft^3), and 
    speed of sound (ft/s).

    Parameters
    ----------
    H : float
        geometric altitude (ft).
    
    Returns
    -------
    Z : float
        geopotential altitude (ft).
    
    G : float
        gravitational acceleration (ft/s^2)
    
    T : float
        temperature (deg R).
    
    P : float
        pressure (lbf/ft^2).
    
    R : float
        density (slugs/ft^3).
    
    A : float
        speed of sound (ft/s).
    """

    # calculate values
    Z,G,T,P,R,A = stdatm_si(H * 0.3048)
    return Z / 0.3048,G / 0.3048,T * 1.8,P * 0.020885434304801722,R * 0.00194032032363104,A / 0.3048


def stdatm_derivative_slow_si(H):
    """Given the geometric altitude (m), calculates and returns the
    derivative of the
    geopotential altitude (-), gravitational acceleration (1/s^2),
    temperature (K/m), pressure (N/m^3), density (kg/m^4), and 
    speed of sound (1/s).

    Parameters
    ----------
    H : float
        geometric altitude (m).
    
    Returns
    -------
    dZdH : float
        derivative of geopotential altitude wrt geometric altitude (-).
    
    dGdH : float
        derivative of gravitational acceleration wrt geometric altitude (1/s^2)
    
    dTdH : float
        derivative of temperature wrt geometric altitude (K/m).
    
    dPdH : float
        derivative of pressure wrt geometric altitude (N/m^3).
    
    dRdH : float
        derivative of density wrt geometric altitude (kg/m^4).
    
    dAdH : float
        derivative of speed of sound wrt geometric altitude (1/s).
    """
    # get current properties
    Z,G,T,P,R,A = stdatm_si(H)
    RE = 6356766.0
    g0 = 9.806645
    y = 1.4
    RGC = 287.0528

    # geopotential altitude derivative
    dZdH = RE*RE/(RE+H)/(RE+H)

    # gravitation acceleration derivative
    dGdH = 2.0*g0*RE/(RE+H)*(-RE/(RE+H)/(RE+H))

    # temperature derivative
    if   Z < 11000.0: dTdZ = -0.0065
    elif Z < 20000.0: dTdZ = +0.0000
    elif Z < 32000.0: dTdZ = +0.0010
    elif Z < 47000.0: dTdZ = +0.0028
    elif Z < 52000.0: dTdZ = +0.0000
    elif Z < 61000.0: dTdZ = -0.0020
    elif Z < 79000.0: dTdZ = -0.0040
    elif Z < 90000.0: dTdZ = +0.0000
    else            : dTdZ = +0.0000
    dTdH = dTdZ*dZdH

    # pressure derivative
    dPdH = -G*P/RGC/T

    # density derivative
    dRdH = dPdH/RGC/T - P/RGC/T/T*dTdH

    # speed of sound derivative
    dAdH = 0.5*y*RGC*(y*RGC*T)**-0.5*dTdH
    # dAdH = 0.5*(y*RGC/T)**0.5*dTdH
    
    return dZdH,dGdH,dTdH,dPdH,dRdH,dAdH


def stdatm_derivative_si(H):
    """Given the geometric altitude (m), calculates and returns the
    derivative of the
    geopotential altitude (-), gravitational acceleration (1/s^2),
    temperature (K/m), pressure (N/m^3), density (kg/m^4), and 
    speed of sound (1/s) with respect to the geometric altitude.

    Parameters
    ----------
    H : float
        geometric altitude (m).
    
    Returns
    -------
    dZdH : float
        derivative of geopotential altitude wrt geometric altitude (-).
    
    dGdH : float
        derivative of gravitational acceleration wrt geometric altitude (1/s^2)
    
    dTdH : float
        derivative of temperature wrt geometric altitude (K/m).
    
    dPdH : float
        derivative of pressure wrt geometric altitude (N/m^3).
    
    dRdH : float
        derivative of density wrt geometric altitude (kg/m^4).
    
    dAdH : float
        derivative of speed of sound wrt geometric altitude (1/s).
    """
    # get current properties
    Z,G,T,P,R,A = stdatm_si(H)

    # geopotential altitude derivative
    dZdH = 40408473978756.0/(6356766.0+H)/(6356766.0+H)

    # gravitation acceleration derivative
    dGdH = -19.61329*dZdH/(6356766.0+H)

    # temperature derivative
    if   Z < 11000.0: dTdZ = -0.0065
    elif Z < 20000.0: dTdZ = +0.0000
    elif Z < 32000.0: dTdZ = +0.0010
    elif Z < 47000.0: dTdZ = +0.0028
    elif Z < 52000.0: dTdZ = +0.0000
    elif Z < 61000.0: dTdZ = -0.0020
    elif Z < 79000.0: dTdZ = -0.0040
    elif Z < 90000.0: dTdZ = +0.0000
    else            : dTdZ = +0.0000
    dTdH = dTdZ*dZdH

    # pressure derivative
    dPdH = -G*P/287.0528/T

    # density derivative
    dRdH = dPdH/287.0528/T - P/287.0528/T/T*dTdH

    # speed of sound derivative
    dAdH = 10.023396629885498*T**-0.5*dTdH
    
    return dZdH,dGdH,dTdH,dPdH,dRdH,dAdH


def stdatm_derivative_english(H):
    """Given the geometric altitude (ft), calculates and returns the
    derivative of the
    geopotential altitude (-), gravitational acceleration (1/s^2), 
    temperature (deg R/ft), pressure (lbf/ft^3), density (slugs/ft^4), and 
    speed of sound (1/s) with respect to the geometric altitude.

    Parameters
    ----------
    H : float
        geometric altitude (ft).
    
    Returns
    -------
    dZdH : float
        derivative of geopotential altitude wrt geometric altitude (-).
    
    dGdH : float
        derivative of gravitational acceleration wrt geometric altitude (1/s^2)
    
    dTdH : float
        derivative of temperature wrt geometric altitude (deg R/ft).
    
    dPdH : float
        derivative of pressure wrt geometric altitude (lbf/ft^3).
    
    dRdH : float
        derivative of density wrt geometric altitude (slugs/ft^4).
    
    dAdH : float
        derivative of speed of sound wrt geometric altitude (1/s).
    """

    # calculate values
    dZdH,dGdH,dTdH,dPdH,dRdH,dAdH = stdatm_derivative_si(H*0.3048)
    return dZdH, dGdH, dTdH*0.54864, dPdH*0.006365880376103566, dRdH*0.000591409634642741, dAdH



if __name__ == "__main__":
    # check values
    print(stdatm_english(15000.0))

    # # # # TEST DERIVATIVES
    english = True
    if english:
        stdatmf = stdatm_english
        stdatm_derf = stdatm_derivative_english
    else:
        stdatmf = stdatm_si
        stdatm_derf = stdatm_derivative_si

    H = 5000.0; dH = 1.0 # 0.001 # 
    Z ,G ,T ,P ,R ,A  = stdatmf(H     )
    Zm,Gm,Tm,Pm,Rm,Am = stdatmf(H - dH) # + dH) # 
    Zp,Gp,Tp,Pp,Rp,Ap = stdatmf(H + dH) # - dH) # 
    # diff
    dZdH_n = (Zp - Zm)/dH/2.0
    dGdH_n = (Gp - Gm)/dH/2.0
    dTdH_n = (Tp - Tm)/dH/2.0
    dPdH_n = (Pp - Pm)/dH/2.0
    dRdH_n = (Rp - Rm)/dH/2.0
    dAdH_n = (Ap - Am)/dH/2.0

    # analytic
    dZdH_a,dGdH_a,dTdH_a,dPdH_a,dRdH_a,dAdH_a = stdatm_derf(H)
    # dZdH_a,dGdH_a,dTdH_a,dPdH_a,dRdH_a,dAdH_a = -dZdH_a,-dGdH_a,-dTdH_a,-dPdH_a,-dRdH_a,-dAdH_a

    # percent difference
    pd_Z = (dZdH_n - dZdH_a)/dZdH_n if dZdH_n != 0.0 else (dZdH_n - dZdH_a)
    pd_G = (dGdH_n - dGdH_a)/dGdH_n if dGdH_n != 0.0 else (dGdH_n - dGdH_a)
    pd_T = (dTdH_n - dTdH_a)/dTdH_n if dTdH_n != 0.0 else (dTdH_n - dTdH_a)
    pd_P = (dPdH_n - dPdH_a)/dPdH_n if dPdH_n != 0.0 else (dPdH_n - dPdH_a)
    pd_R = (dRdH_n - dRdH_a)/dRdH_n if dRdH_n != 0.0 else (dRdH_n - dRdH_a)
    pd_A = (dAdH_n - dAdH_a)/dAdH_n if dAdH_n != 0.0 else (dAdH_n - dAdH_a)

    # compare
    n_dash = 30
    print("-"*n_dash)
    print("Z =",Z)
    print("G =",G)
    print("T =",T)
    print("P =",P)
    print("R =",R)
    print("A =",A)
    print("-"*n_dash)
    print("dZdH_n = {:>+22.16f}; dZdH_a = {:>+22.16f}".format(dZdH_n,dZdH_a))
    print("dGdH_n = {:>+22.16f}; dGdH_a = {:>+22.16f}".format(dGdH_n,dGdH_a))
    print("dTdH_n = {:>+22.16f}; dTdH_a = {:>+22.16f}".format(dTdH_n,dTdH_a))
    print("dPdH_n = {:>+22.16f}; dPdH_a = {:>+22.16f}".format(dPdH_n,dPdH_a))
    print("dRdH_n = {:>+22.16f}; dRdH_a = {:>+22.16f}".format(dRdH_n,dRdH_a))
    print("dAdH_n = {:>+22.16f}; dAdH_a = {:>+22.16f}".format(dAdH_n,dAdH_a))
    print("-"*n_dash)
    print("dZ %df =",pd_Z)
    print("dG %df =",pd_G)
    print("dT %df =",pd_T)
    print("dP %df =",pd_P)
    print("dR %df =",pd_R)
    print("dA %df =",pd_A)
    print("-"*n_dash)


    # from time import time as time
    # n = 1000000
    # start = time()
    # for i in range(n):
    #     stdatm_derivative_slow_si(H+i/100)
    # duration = time() - start
    # print("slow time [s] =",duration)
    # start = time()
    # for i in range(n):
    #     stdatm_derivative_si(H+i/100)
    # duration = time() - start
    # print("fast time [s] =",duration)
    english = True
    if english:
        stdatmf = stdatm_english
        stdatm_derf = stdatm_derivative_english
        print("testing english")
        dv = 3.0
    else:
        stdatmf = stdatm_si
        stdatm_derf = stdatm_derivative_si
        print("testing si")
        dv = 1.0

    # # test at various heights
    # H_final = 100000.0 # 1000.0 # 
    # num = 1000 # 10 # 
    # Hs = np.linspace(0.0,H_final,num)[1:]
    # dH = 1.0
    # pd_thr = 1.0e-8
    # for h in Hs:
    #     h_test = h*dv
    #     Z ,G ,T ,P ,R ,A  = stdatmf(h_test     )
    #     Zm,Gm,Tm,Pm,Rm,Am = stdatmf(h_test - dH)
    #     Zp,Gp,Tp,Pp,Rp,Ap = stdatmf(h_test + dH)
    #     # diff
    #     dZdH_n = (Zp - Zm)/dH/2.0
    #     dGdH_n = (Gp - Gm)/dH/2.0
    #     dTdH_n = (Tp - Tm)/dH/2.0
    #     dPdH_n = (Pp - Pm)/dH/2.0
    #     dRdH_n = (Rp - Rm)/dH/2.0
    #     dAdH_n = (Ap - Am)/dH/2.0

    #     # analytic
    #     dZdH_a,dGdH_a,dTdH_a,dPdH_a,dRdH_a,dAdH_a = stdatm_derf(h_test)

    #     # percent difference
    #     pd_Z = (dZdH_n - dZdH_a)/dZdH_n if dZdH_n != 0.0 else (dZdH_n - dZdH_a)
    #     pd_G = (dGdH_n - dGdH_a)/dGdH_n if dGdH_n != 0.0 else (dGdH_n - dGdH_a)
    #     pd_T = (dTdH_n - dTdH_a)/dTdH_n if dTdH_n != 0.0 else (dTdH_n - dTdH_a)
    #     pd_P = (dPdH_n - dPdH_a)/dPdH_n if dPdH_n != 0.0 else (dPdH_n - dPdH_a)
    #     pd_R = (dRdH_n - dRdH_a)/dRdH_n if dRdH_n != 0.0 else (dRdH_n - dRdH_a)
    #     pd_A = (dAdH_n - dAdH_a)/dAdH_n if dAdH_n != 0.0 else (dAdH_n - dAdH_a)

    #     if abs(pd_Z) > pd_thr: print(h,"pd_Z =",pd_Z)
    #     if abs(pd_G) > pd_thr: print(h,"pd_G =",pd_G)
    #     if abs(pd_T) > pd_thr: print(h,"pd_T =",pd_T)
    #     if abs(pd_P) > pd_thr: print(h,"pd_P =",pd_P)
    #     if abs(pd_R) > pd_thr: print(h,"pd_R =",pd_R)
    #     if abs(pd_A) > pd_thr: print(h,"pd_A =",pd_A)
    #     print(h,dRdH_a)



    