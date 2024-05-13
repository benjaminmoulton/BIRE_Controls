import numpy as np
from atmospheric_functions import statsi, statee, gravity_si, gravity_english


class TrimSolution:
    # object for storing a trim solution
    def __init__(self):
        self.FM = np.zeros(6) # forces and moments
        self.FM_dim = np.zeros(6) # dimensional body fixed forces and moments
        self.rates = np.zeros(3) # rotation rates
        self.velocity = np.zeros(3) # u v w
        self.load = 0.
        self.load_s = 0.
        self.x = np.zeros(6) # trim solution [tau, alpha, beta, da, de, dB]
        self.orient = np.zeros(3) # phi theta psi
        self.num_iters = 0.
        self.vehicle = "Baseline" # BIRE or baseline (F-16)

def climb2theta(V_vec,gamma,phi):
    '''Converts a given climb angle to an elevation angle for a given bank
    angle and body-fixed velocity vector.

    Parameters
    -----------
    V: array_like
        body-fixed velocity vector
    gamma: float
        climb angle in radians
    phi: float
        bank angle in radians
    '''
    u,v,w = V_vec
    Vmag = np.sqrt(u*u + v*v + w*w)
    
    vswc = v*np.sin(phi) + w*np.cos(phi)
    num_sqrt = np.sqrt(u*u + vswc*vswc - Vmag*Vmag*np.sin(gamma)*np.sin(gamma))
    denom = u*u + vswc*vswc
    
    theta1 = np.arcsin((u*Vmag*np.sin(gamma) + vswc*num_sqrt)/denom)
    theta2 = np.arcsin((u*Vmag*np.sin(gamma) - vswc*num_sqrt)/denom)
    
    check_st1 = u*np.sin(theta1) - vswc*np.cos(theta1)
    check_st2 = u*np.sin(theta2) - vswc*np.cos(theta2)
    
    if abs(check_st1 - Vmag*np.sin(gamma)) < 1e-8:
        return theta1
    elif abs(check_st2 - Vmag*np.sin(gamma)) < 1e-8:
        return theta2
    
def SCT_rot_rates(V_vec,phi,theta,g):
    '''Calculates the equilibrium body-fixed rotation rates in a
    steady-coordinated turn.

    Parameters
    -----------
    V: array_like
        body-fixed velocity vector
    phi: float
        bank angle in radians
    theta: float
        elevation angle in radians
    g: float
        gravitational acceleration at altitude
    '''
    u,v,w = V_vec
    coeff = (g*np.sin(phi)*np.cos(theta))/(u*np.cos(theta)*np.cos(phi) + w*np.sin(theta))
    
    p,q,r = coeff*np.array([-np.sin(theta), np.sin(phi)*np.cos(theta), np.cos(phi)*np.cos(theta)])
    
    return np.array([p,q,r])

def solve_Vb_vector(alpha, beta, V):
    # body fixed velocity components from angle of attack and sideslip angle
    u = V*np.cos(alpha)*np.cos(beta)
    v = V*np.sin(beta)
    w = V*np.sin(alpha)*np.cos(beta)
    return u, v, w

def _f1(Fx, theta, phi, pqr, uvw, props):
    # handbook Eq 16.56
    [u, v, w] = uvw
    [p, q, r] = pqr
    return Fx - props.W*np.sin(theta) + (r*v - q*w)*props.W/props.g

def _f2(Fy, theta, phi, pqr, uvw, props):
    # handbook Eq 16.56
    [u, v, w] = uvw
    [p, q, r] = pqr
    return Fy + props.W*np.sin(phi)*np.cos(theta) + (p*w - r*u)*props.W/props.g

def _f3(Fz, theta, phi, pqr, uvw, props):
    # handbook Eq 16.56
    [u, v, w] = uvw
    [p, q, r] = pqr
    return Fz + props.W*np.cos(phi)*np.cos(theta) + (q*u - p*v)*props.W/props.g

def _f4(Mx, theta, phi, pqr, uvw, props):
    # handbook Eq 16.56
    [p, q, r] = pqr
    C1 = Mx - props.hz*q + props.hy*r + (props.Iyy - props.Izz)*q*r
    C2 = props.Iyz*(q*q - r*r) + props.Ixz*p*q - props.Ixy*p*r
    return C1 + C2

def _f5(My, theta, phi, pqr, uvw, props):
    # handbook Eq 16.56
    [p, q, r] = pqr
    C1 = My + props.hz*p - props.hx*r + (props.Izz - props.Ixx)*p*r
    C2 = props.Ixz*(r*r - p*p) + props.Ixy*q*r - props.Iyz*p*q
    return C1 + C2

def _f6(Mz, theta, phi, pqr, uvw, props):
    # handbook Eq 16.56
    [p, q, r] = pqr
    C1 = Mz - props.hy*p + props.hx*q + (props.Ixx - props.Iyy)*p*q
    C2 = props.Ixy*(p*p - q*q) + props.Iyz*p*r - props.Ixz*q*r
    return C1 + C2

def _calc_forces(state, phi, gamma, aero_model, aircraft_props, cg_shift, shss, compressible, stall):
    # find forces and moments using the aero model
    V = aircraft_props.V
    g = aircraft_props.g
    M = aircraft_props.M
    
    bw = aircraft_props.b_w
    cw = aircraft_props.c_w
    H = aircraft_props.H
    rho_0 = aircraft_props.rho_0
    rho = aircraft_props.rho
    
    [tau, alpha, beta, da, de, dr] = state
    
    V_vec = solve_Vb_vector(alpha, beta, V)
    u, v, w = V_vec
    
    theta = climb2theta(V_vec, gamma, phi)
    
    if not shss:
        p, q, r = SCT_rot_rates(V_vec,phi,theta,g)
        pbar = p*bw/(2.*V)
        qbar = q*cw/(2.*V)
        rbar = r*bw/(2.*V)
    else:
        p, q, r = [0., 0., 0.]
        pbar, qbar, rbar = [0., 0., 0.]

    FM = aero_model.aero_CG_offset_results(alpha, beta, pbar, qbar, rbar, da, de, dr, tau,
                                           V, H, rho_0, rho, cg_shift, compressible = compressible,
                                           M = M, use_Anderson = True, enforce_stall=stall)

    return FM, [u, v, w], [p, q, r], theta

def solve_jacobian(trim_state, phi, theta, gamma, aero_model, aircraft_props, cg_shift, shss, compressible, stall, delta=0.001):
    # generate the jacobian for Newtons method
    [tau, alpha, beta, de, da, dr] = trim_state
    J = np.zeros((6, 6))
    
    f = [_f1, _f2, _f3, _f4, _f5, _f6]
    for i in range(6):
        delta_state = np.zeros(6)
        delta_state[i] = delta
        FM_p, vcomp_p, rotrates_p, theta_p = _calc_forces([t + d for t,d in zip(trim_state, delta_state)], phi, gamma, aero_model, aircraft_props, cg_shift, shss, compressible, stall)
        FM_m, vcomp_m, rotrates_m, theta_m = _calc_forces([t - d for t,d in zip(trim_state, delta_state)], phi, gamma, aero_model, aircraft_props, cg_shift, shss, compressible, stall)
        for j in range(6):
            f_p = f[j](FM_p[j], theta_p, phi, rotrates_p, vcomp_p, aircraft_props)
            f_m = f[j](FM_m[j], theta_m, phi, rotrates_m, vcomp_m, aircraft_props)
            J[j, i] = (f_p - f_m)/(2.*delta)
    return J

def solve_trim(aero_model, aircraft_props, gamma = 0.0, phi = 0.0, psi = 0.0, cg_shift=[0., 0., 0.], shss = False, compressible = False, stall = False, **kwargs):
    
    '''
    Solves for trim in either a steady coordinated turn or a steady-heading sidelsip.
    
    '''
    threshold = kwargs.get("tol", 1e-9)
    
    verbose=True 
    # fixed_point=False # revisit if we ever need to used the fixed point solution method
    Gamma = kwargs.get("Gamma", 0.1)
    
    '''Atmospheric/flight condition properties'''
    h,z,t,p,d,a = statee(aircraft_props.H)
    M = aircraft_props.M
    
    # tau, alpha, beta, da, de, dr 
    trim_0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # set current trim state to intial guess
    trim_state = trim_0

    [tau, alpha, beta, da, de, dr] = trim_state
    # [tau, alpha, beta, da, de, dB] = trim_state

    # intiallize rotations rates, dimensional and nondimensional
    p, q, r = [0., 0., 0.]
    pbar, qbar, rbar = p, q, r
    
    error = 100.
    number_of_iterations = 0

    # iterate until error is low enough or max iterations is reached
    while (error > threshold)*(number_of_iterations <= 1000):

        number_of_iterations += 1
        
        FM, [u, v, w], [p, q, r], theta = _calc_forces(trim_state, phi, gamma, aero_model, aircraft_props, cg_shift, shss, compressible, stall)
        [Fx, Fy, Fz, Mx, My, Mz] = FM
        
        if aircraft_props.BIRE == True:
            aircraft_props.calc_BIRE_inertia(trim_state[-1])
            
        # took this from Christians method
        f = [_f1, _f2, _f3, _f4, _f5, _f6]
        nums = np.array([f(FM[idx], theta, phi, [p, q, r], [u, v, w], aircraft_props) for idx, f in enumerate(f)])
        try:
            J = solve_jacobian(trim_state, phi, theta, gamma, aero_model, aircraft_props, cg_shift, shss, compressible, stall)
            D_G = np.linalg.solve(-J, nums)
        except np.linalg.LinAlgError:
            J = solve_jacobian(trim_state, phi, theta, gamma, aero_model, aircraft_props, cg_shift, shss, compressible, stall, delta=0.01)
            D_G = np.linalg.solve(-J, nums)
        
        # does Gamma need to be an input that can be specified?
        trimstate_p1 = trim_state + Gamma*D_G
        error = np.max(np.abs(nums))
        trim_state = trimstate_p1
        [tau, alpha, beta, da, de, dr] = trim_state
        
    # FINAL FORCE/MOMENT SOLUTION AT TRIM
    FM, [u, v, w], [p, q, r], theta = _calc_forces(trim_state, phi, gamma, aero_model, aircraft_props, cg_shift, shss, compressible, stall)
    [Fx, Fy, Fz, Mx, My, Mz] = FM
    
    pbar = p*aircraft_props.b_w/(2.*aircraft_props.V)
    qbar = q*aircraft_props.c_w/(2.*aircraft_props.V)
    rbar = r*aircraft_props.b_w/(2.*aircraft_props.V)

    CFM = aero_model.aero_results(alpha, beta, pbar, qbar, rbar, da, de, dr, compressible = compressible, M = M,enforce_stall=stall)
    [CL, CS, CD, Cl, Cm, Cn] = CFM

    if verbose:
        print("------ Trim Solution ------")
        if aircraft_props.BIRE:
            print(f"------ BIRE ------")
        else:
            print(f"------ F-16 ------")
        print(f"Elevation Angle (deg.) : {theta*180./np.pi:1.16g}")
        print(f"Bank Angle (deg.) : {phi*180./np.pi:1.16g}")
        print(f"Alpha (deg.) : {alpha*180./np.pi:1.16g}")
        print(f"Beta (deg.) : {beta*180./np.pi:1.16g}")
        print(f"p (deg./s) : {p*180./np.pi:1.16g}")
        print(f"q (deg./s) : {q*180./np.pi:1.16g}")
        print(f"r (deg./s) : {r*180./np.pi:1.16g}")
        print(f"Aileron (deg.) : {da*180./np.pi:1.16g}")
        print(f"Elevator (deg.) : {de*180./np.pi:1.16g}")
        print(f"Rudder (deg.) : {dr*180./np.pi:1.16g}")
        # if bire:
        #     print(f"BIRE Rotation (deg.) : {dr*180./np.pi:1.12g}")
        # else:
        #     print(f"Rudder (deg.) : {dr*180./np.pi:1.12g}")
        print(f"Throttle : {tau:1.16g}")
        # print(f"Thrust (lbf.) : {T:1.12f}")
        # print(f"Load Factor : {n_a:1.12f}")
        # print(f"Stability Axis Load Factor : {n_sa:1.12f}")
        print(f"Number of Iterations : {number_of_iterations:d}")
        
    # initialize trim solution object
    solution = TrimSolution()
    
    # updated trim object values
    solution.FM = np.array([CL, CS, CD, Cl, Cm, Cn])
    solution.FM_dim = np.array([Fx, Fy, Fz, Mx, My, Mz])
    
    # # load factors
    # solution.load = n_a
    # solution.load_s = n_sa
    
    # replace the following
    solution.x = trim_state
    
    solution.inputs = np.array([tau, da, de, dr])
    
    solution.num_iters = number_of_iterations
    solution.orient = np.array([phi, theta, 0.])
    solution.velocity = np.array([u, v, w])
    solution.rates = np.array([p, q, r])
    solution.states = np.array([u, v, w, p, q, r, phi, theta])
    
    solution.aero = aero_model
    
    return solution

if __name__ == "__main__":

    import sys
    aero_directory = '../aerodynamics_model/'
    sys.path.insert(1, aero_directory)

    from f16_aero import F16Aero
    from bire_aero import BIREAero
    
    from aircraft_properties import AircraftProperties
    
    gamma = np.deg2rad(0.0) #rad
    phi = np.deg2rad(10.0) #rad
    
    Vinf = 634 #ft/s
    H = 15000
    Gamma = 0.1
    bire = True
    SHSS = False

    AircraftProps = AircraftProperties(Vinf, H, Gamma, bire=bire)
    
    if bire == True:
        AeroModel = BIREAero(aero_directory)
    else:
        AeroModel = F16Aero(aero_directory)

    solution = solve_trim(AeroModel, AircraftProps, gamma, phi, cg_shift=[1.0,0.0,0.0],
                          shss = SHSS, compressible = True, stall = True, Gamma = Gamma, tol = 1e-10)
    print('Forces at trim: ', solution.FM_dim)
    print('Forces at trim: ', solution.FM)
    # bw = 30
    # cw = 11.32
    # #alpha, beta, pbar, qbar, rbar, da, de, dr
    # params = np.deg2rad([3.10448578547155, 0.000514555800702851, (-0.05371348518135819)*bw*0.5/Vinf, (0.3604380623364666)*cw*0.5/Vinf, (0.9902954373814199)*bw*0.5/Vinf, 0.01237144097185099, -1.68732272858867, -0.05150450949196807])
    # params = np.array([0.054183498538, 0.000008980693, -0.000022180059, 0.000056161046, 0.000408925456, 0.000215922378, -0.029449337158, -0.000898923270])
    # print('alpha, beta, pbar, qbar, rbar, da, de, dr')
    # print('{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}'.format(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7]))
    
    # tau = 0.277031917216823
    # rho_0 = 0.0023768930112381755
    # rho = 0.00149615667353899
    # Mach = 0.59960910465664
    
    # print('rho_0: ', rho_0)
    # print('rho: ', rho)
    # print('Mach: ', Mach)
    
    # FM_test = AeroModel.aero_CG_offset_results(*params, tau, 
    #                             Vinf, H, rho_0, rho, cg_shift=[1., 0., 0.], compressible=True,
    #                             M=Mach, use_Anderson=True, enforce_stall=True)
    
    # print('FM Test: ', FM_test)
    
    # FM_test = AeroModel.aero_results(*params, compressible=True,
    #                             M=Mach, use_Anderson=True, enforce_stall=True)
    
    # print('FM Test: ', FM_test)