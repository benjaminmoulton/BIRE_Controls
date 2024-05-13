from math import cos, sin, atan, atan2, asin, sqrt, exp, pi

#-----------------------------------------------------------------------------#
#Quaternion Operation Functions

def NormQuat(q):

    '''normalizes quaternion using Eq. (11.10.7) from Phillips

    Parameters
    ----------
    q: array
        quaternion values

    Returns
    -------
    list of normalized quaternion values
    '''

    q0, q1, q2, q3 = q

    norm_coeff = 1.5 - 0.5*(q0*q0 + q1*q1 + q2*q2 + q3*q3)

    return [q0*norm_coeff, q1*norm_coeff, q2*norm_coeff,
            q3*norm_coeff]

def NormQuat2(q):

    '''normalizes quaternion using Eq. (11.10.5) from Phillips

    Parameters
    ----------
    q: array
        quaternion values

    Returns
    -------
    list of normalized quaternion values
    '''
    q0, q1, q2, q3 = q

    norm_den = sqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3)

    return [q0/norm_den, q1/norm_den, q2/norm_den,
            q3/norm_den]

def Euler2Quat(ea):

    '''converts euler angles to quaternion using Eq. (11.7.8)

    Parameters
    ----------
    ea: array
        euler angles

    Returns
    -------
    list of quaternion values
    '''

    ea05 = ea[0]*0.5
    ea15 = ea[1]*0.5
    ea25 = ea[2]*0.5

    CP = cos(ea05)
    CT = cos(ea15)
    CS = cos(ea25)

    SP = sin(ea05)
    ST = sin(ea15)
    SS = sin(ea25)
    c1 = CP*CT
    c2 = SP*ST
    c3 = SP*CT
    c4 = CP*ST

    return [(c1*CS + c2*SS), (c3*CS - c4*SS),
            (c4*CS + c3*SS), (c1*SS - c2*CS)]

def Quat2Euler(q, psiprev=0):

    '''converts quaternion to euler angles using Eq. (11.7.11)

    Parameters
    ----------
    q: array
        quaternion values
    psiprev: float
        previous psi value

    Returns
    -------
    list of euler angles
    '''

    q0, q1, q2, q3 = q

    check = q0*q2 - q1*q2

    if (check) == 0.5:
        phi = 2*asin(q1/cos(pi/4)) + psiprev
        theta = pi/2
        psi = psiprev
        return [phi,theta,psi]
    elif (check) == -0.5:
        phi = 2*asin(q1/cos(pi/4)) - psiprev
        theta = -pi/2
        psi = psiprev
        return [phi,theta,psi]
    else:
        q00 = q0*q0
        q11 = q1*q1
        q22 = q2*q2
        q33 = q3*q3

        return [atan2(2*(q0*q1 + q2*q3),
                    (q00 + q33 - q11 - q22)),
                asin(2*(q0*q2 - q1*q3)),
                atan2(2*(q0*q3 + q1*q2),
                         (q00 + q11 - q22 - q33))]

def Fixed2Body(v, e):

    '''converts earth fixed vector to body fixed vector using reduced form 
    of Eq. (11.6.8)

    Parameters
    ----------
    v: array
        vector to be converted
    e: array
        quaternion

    Returns
    -------
    list of body fixed vector components
    '''

    v0, v1, v2, v3 = v
    e0, e1, e2, e3 = e

    ve00 = v0*e0
    ve01 = v0*e1
    ve02 = v0*e2
    ve03 = v0*e3
    ve10 = v1*e0
    ve11 = v1*e1
    ve12 = v1*e2
    ve13 = v1*e3
    ve20 = v2*e0
    ve21 = v2*e1
    ve22 = v2*e2
    ve23 = v2*e3

    return [(e0*(ve00 + ve13 - ve22) - 
                  e1*(-ve01 - ve12 - ve23) -
                  e2*(ve02 - ve11 + ve20) +
                  e3*(-ve03 + ve10 + ve21)),
                (e0*(-ve03 + ve10 + ve21) + 
                 e1*(ve02 - ve11 + ve20) -
                 e2*(-ve01 - ve12 - ve23) -
                 e3*(ve00 + ve13 - ve22)),
                (e0*(ve02 - ve11 + ve20) - 
                 e1*(-ve03 + ve10 + ve21) +
                 e2*(ve00 + ve13 - ve22) -
                 e3*(-ve01 - ve12 - ve23))]

def Body2Fixed(v, e):

    '''converts body fixed vector to earth fixed vector using reduced form 
    of Eq. (11.6.8) inversed

    Parameters
    ----------
    v: array
        vector to be converted
    e: array
        quaternion

    Returns
    -------
    list of earth fixed vector components
    '''

    e0, e1, e2, e3 = e
    v0, v1, v2 = v

    e00 = e0*e0
    e0x = e0*e1
    e0y = e0*e2
    e0z = e0*e3
    exx = e1*e1
    exy = e1*e2
    exz = e1*e3
    eyy = e2*e2
    eyz = e2*e3
    ezz = e3*e3
    exzv2 = exz*v2
    exyv0 = exy*v0
    e0zv0 = e0z*v0
    e0yv0 = e0y*v0
    exzv0 = exz*v0
    exyv1 = exy*v1
    e0zv1 = e0z*v1
    eyzv1 = eyz*v1
    e0xv1 = e0x*v1
    e0yv2 = e0y*v2
    eyzv2 = eyz*v2
    e0xv2 = e0x*v2

    return [(e00*v0 - e0zv1 + e0yv2 + 
                  exx*v0 + exyv1 + exzv2 +
                  eyy*-v0 + exyv1 + e0yv2 -
                  ezz*v0 - e0zv1 + exzv2),
                (e0zv0 + e00*v1 - e0xv2 + 
                 exyv0 - exx*v1 - e0xv2 +
                 exyv0 + eyy*v1 + eyzv2 +
                 e0zv0 - ezz*v1 + eyzv2),
                (-e0yv0 + e0xv1 + e00*v2 + 
                 exzv0 + e0xv1 - exx*v2 -
                 e0yv0 + eyzv1 - eyy*v2 +
                 exzv0 + eyzv1 + ezz*v2)]