
def SAL_stdatm_english(ALT):
    """Given the geometric altitude (ft), calculates and returns the
    geopotential altitude (ft), gravitational acceleration (ft/s^2), 
    temperature (deg R), pressure (lbf/ft^2), density (slugs/ft^3), and 
    speed of sound (ft/s).

    Parameters
    ----------
    ALT : float
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
    # initialize vars
    R0 = 2.377e-3

    # calculate geopotential altitude
    Z = (6356766.0*ALT*0.3048)/(6356766.0 + ALT*0.3048)/0.3048

    # calculate gravitational constant
    G = 3.96271559301397625e+14/(6356766.0 + ALT*0.3048)**2./0.3048
    # fixed in SAL sims
    G = 32.17

    # calculate T
    TFAC = 1.0 - 0.703e-5*ALT
    T = 519.0*TFAC
    if ALT >= 35000.:
        T = 390.0
    
    # calculate R
    R = R0*(TFAC**4.14)

    # calculate P
    P = 1715.0*R*T

    # calculate speed of sound
    A = (1.4*1716.3*T)**0.5
    quit()
    return Z,G,T,P,R,A

if __name__ == "__main__":
    # check values
    print(SAL_stdatm_english(15000.0))