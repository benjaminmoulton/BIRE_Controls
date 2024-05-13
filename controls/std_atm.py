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


if __name__ == "__main__":
    # check values
    print(stdatm_english(15000.0))