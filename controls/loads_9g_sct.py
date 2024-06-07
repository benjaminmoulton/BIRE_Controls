import numpy as np
import json
from matplotlib import pyplot as plt
from controller_simulation import Aircraft,monte_carlo_perturbations,run_single_simulation
from quat import quat_mult, euler_2_quat, quat_2_euler
import machupX as mux

def correction(self,FM,enforce_stall,compressible,use_Anderson,M):
    [CL, CS, CD, Cl, Cm, Cn] = FM
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
                CS = self._Anderson_correction(CS,self.Lam_v,self.RA_v,M)
                Cl = self._Anderson_correction(Cl,self.Lam_v,self.RA_v,M) # w w
                Cm = self._Anderson_correction(Cm,self.Lam_w,self.RA_w,M)
                Cn = self._Anderson_correction(Cn,self.Lam_v,self.RA_v,M)
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


if __name__ == "__main__":

    # filenames 
    base_file = "base_fs_in.json"
    bire_file = "bire_fs_in.json"

    # read in json to ensure no file changes while running
    base_dict = json.loads( open(base_file).read() )
    bire_dict = json.loads( open(bire_file).read() )

    ## trim case parameters
    # run at SCT
    trim_type = "sct"
    # altitude
    H = 0.0
    # mach
    M = 0.8
    # gs of the case # 9.08 ~ 9 gs (add a bit more because relation on line 27 is approx)
    #     see trim output results (stbly load factor) to ammend for accuracy
    n = 9.08
    # simplification to put in terms of bank angle
    phi_trim = np.rad2deg(np.arccos(1.0/n))
    # cg shift (in ft) from nominal location in NASA report
    cg_shift = [0.0,0.0,0.0]
    # left stabilator wing root location wrt nominal location (do not add cgshift to this number)
    ls_loc = [-13.1,-0.37*9.2,0.05]
    # weight
    W = 20500.0
    # #
    ## compressibility parameters
    include_compressibility = False # True # 
    # if compressible, whether to use Anderson (True) or Prandtl-Glauert (False) corrections
    use_anderson = True # False # 
    # whether to include a the stall model (probably want this to stay False)
    include_stall = False # True # 

    ###########################################################################
    ###########################################################################
    ###########################################################################

    # base dict
    # pull out ref lengths and area
    Sw = base_dict["aircraft"]["wing_area[ft^2]"]
    bw = base_dict["aircraft"]["wing_span[ft]"]
    cw = base_dict["aircraft"]["wing_aerodynamic_mean_chord[ft]"]
    # # edit base dict
    # compress and stall
    base_dict["simulation"]["include_compressibility"] = include_compressibility
    base_dict["simulation"]["use_Anderson_corrections"] = use_anderson
    base_dict["simulation"]["include_stall"] = include_stall
    # cg
    base_dict["aircraft"]["CG_shift[ft]"] = cg_shift
    # trim params
    base_dict["initial"] = {
        "mach" : M,
        "longitude[deg]" : 0.0,
        "latitude[deg]" : 0.0,
        "altitude[ft]" : H,
        "heading[deg]" : 0.0,
        "type" : "trim",
        "trim" : {
            "type" : trim_type,
            "climb_angle[deg]" : 0.0,
            "bank_angle[deg]" : phi_trim,
            "solver" : {
                "finite_difference_step_size" : 0.001,
                "relaxation_factor" : 0.1,
                "tolerance" : 1.0e-9
            },
            "verbose_trim" : False
        }
    }
    
    # initialize, fix weight
    base = Aircraft(base_dict)
    base.inertia_model.W = W
    # run trim
    base._initialize_state(base.a_guess,base.b_guess,base.phi_guess,
        base.u_guess)
    base.x,base.t = base.initialize_sim(base.x0)
    base._report_trim_solution()
    # determine air density
    _,_,_,_,rho,_ = base.stdatm(H)

    # pull out state & control info
    Vxb = base.x_trim[0]
    Vyb = base.x_trim[1]
    Vzb = base.x_trim[2]
    p   = base.x_trim[3] # keep in rad
    q   = base.x_trim[4]
    r   = base.x_trim[5]
    da  = np.rad2deg(base.u_trim[0])
    de  = np.rad2deg(base.u_trim[1])
    dr  = np.rad2deg(base.u_trim[2])
    # translate to V,a,b
    alpha = np.arctan2(Vzb,Vxb)
    V = (Vxb * Vxb + Vyb * Vyb + Vzb * Vzb)**0.5
    beta = np.arcsin(Vyb/V)
    # determine orientation
    e = base.x_trim[9:13]
    phi,theta,psi = np.rad2deg(quat_2_euler(e))

    # initialize baseline aircraft and scene
    print("initializing mux...")
    base_scene_file = "../aerodynamics_model/F16_input.json"
    base_airpl_file = "../aerodynamics_model/F16_airplane.json"
    base_scene_dict = json.loads( open(base_scene_file).read() )
    base_airpl_dict = json.loads( open(base_airpl_file).read() )
    
    base_airpl_dict["CG"] = (np.array(cg_shift) + np.array(ls_loc)).tolist()
    base_airpl_dict["weight"] = W
    base_airpl_dict["airfoils"]["NACA_64A204"]["geometry"]["outline_points"] =\
        "../aerodynamics_model/64A204.txt"
    base_scene_dict["scene"]["aircraft"]["F16"]["file"] = base_airpl_dict
    #
    base_mux_ls = mux.Scene(base_scene_dict)
    # base_mux_ls.display_wireframe()
    # update for right stab root for cg loc
    base_airpl_dict["CG"] = (np.array(cg_shift) 
        + np.array([ls_loc[0],-ls_loc[1],ls_loc[2]])).tolist()
    base_scene_dict["scene"]["aircraft"]["F16"]["file"] = base_airpl_dict
    base_mux_rs = mux.Scene(base_scene_dict)

    state = {
        "velocity" : V, # total velocity
        "alpha" : alpha, # angle of attack in degrees
        "beta" : beta,
        "orientation" : [phi, theta, psi], # Earth-fixed orientation, bank (phi), elevation (theta), heading (psi) in degrees
        "angular_rates" : [p,q,r], # body-fixed rotation rates, roll (p), pitch (q), yaw (r) radians
        "angular_rates_frame" : "body"
    }
    ctrl_state = {
        "aileron" : da, # setting in degrees
        "elevator" : de, # setting in degrees
        "rudder" : dr # setting in degrees
    }
    forces_settings = dict(body_frame=True,stab_frame=False,wind_frame=True,\
        report_by_segment=True,dimensional=False,non_dimensional=True)
    #
    # set aircraft and control state, solve forces
    print("setting state, evaluating forces and moments...")
    # left
    base_mux_ls.set_aircraft_state(state)
    base_mux_ls.set_aircraft_control_state(ctrl_state)
    # base_mux_ls.display_wireframe()
    FMls = base_mux_ls.solve_forces(**forces_settings)["F16"]
    # right
    base_mux_rs.set_aircraft_state(state)
    base_mux_rs.set_aircraft_control_state(ctrl_state)
    FMrs = base_mux_rs.solve_forces(**forces_settings)["F16"]
    # combine viscous and inviscid results
    # print(json.dumps(FMls,indent=4))
    CL_ls = FMls["inviscid"]["CL"]["h_stab_left"] + FMls["viscous"]["CL"]["h_stab_left"]
    CS_ls = FMls["inviscid"]["CS"]["h_stab_left"] + FMls["viscous"]["CS"]["h_stab_left"]
    CD_ls = FMls["inviscid"]["CD"]["h_stab_left"] + FMls["viscous"]["CD"]["h_stab_left"]
    Cl_ls = FMls["inviscid"]["Cl"]["h_stab_left"] + FMls["viscous"]["Cl"]["h_stab_left"]
    Cm_ls = FMls["inviscid"]["Cm"]["h_stab_left"] + FMls["viscous"]["Cm"]["h_stab_left"]
    Cn_ls = FMls["inviscid"]["Cn"]["h_stab_left"] + FMls["viscous"]["Cn"]["h_stab_left"]
    fmls = [CL_ls,CS_ls,CD_ls,Cl_ls,Cm_ls,Cn_ls]
    CL_rs = FMrs["inviscid"]["CL"]["h_stab_left"] + FMrs["viscous"]["CL"]["h_stab_left"]
    CS_rs = FMrs["inviscid"]["CS"]["h_stab_left"] + FMrs["viscous"]["CS"]["h_stab_left"]
    CD_rs = FMrs["inviscid"]["CD"]["h_stab_left"] + FMrs["viscous"]["CD"]["h_stab_left"]
    Cl_rs = FMrs["inviscid"]["Cl"]["h_stab_left"] + FMrs["viscous"]["Cl"]["h_stab_left"]
    Cm_rs = FMrs["inviscid"]["Cm"]["h_stab_left"] + FMrs["viscous"]["Cm"]["h_stab_left"]
    Cn_rs = FMrs["inviscid"]["Cn"]["h_stab_left"] + FMrs["viscous"]["Cn"]["h_stab_left"]
    fmrs = [CL_rs,CS_rs,CD_rs,Cl_rs,Cm_rs,Cn_rs]
    
    # correct for compressibility, stall
    print("correcting for compressibility, stall (if requested)...")
    fmls = correction(base.aero_model,fmls,
        include_stall,include_compressibility,use_anderson,M)
    fmrs = correction(base.aero_model,fmrs,
        include_stall,include_compressibility,use_anderson,M)
    
    # convert forces from wind to body fixed
    print("converting forces to body fixed frame...")
    ca = np.cos(alpha); sa = np.sin(alpha)
    cb = np.cos( beta); sb = np.sin( beta)
    M_w2bf = np.array([
        [ ca*cb,    sb, sa*cb],
        [-ca*sb,    cb,-sa*sb],
        [   -sa,   0.0,    ca]
    ]).T
    fmls[0:3] = np.matmul(M_w2bf,fmls[0:3])
    fmrs[0:3] = np.matmul(M_w2bf,fmrs[0:3])
    # print(fmls)
    fmls = np.array(fmls)
    fmrs = np.array(fmrs)

    # redimensionalize
    print("redimensionalizing forces and moments...")
    Q = 0.5*rho*V**2.*Sw
    G = np.diag([bw,cw,bw])
    # forces
    fmls[0:3] = Q*fmls[0:3]
    fmrs[0:3] = Q*fmrs[0:3]
    # moments
    fmls[3:6] = np.matmul(Q*G,fmls[3:6])
    fmrs[3:6] = np.matmul(Q*G,fmrs[3:6])

    # report
    print()
    print("reporting...")
    nast = 35
    print("*"*nast)
    print("Conditions:")
    ###########################################################################
    print("trim_type =", trim_type)
    print("H =", H)
    print("M =", M)
    print("n =", n)
    print("phi_trim =", phi_trim)
    print("cg_shift [ft]", cg_shift)
    print("left stabilator root location from nominal [ft]", ls_loc)
    print("W [lbf] =", W)
    print("include_compressibility =", include_compressibility)
    print("use_anderson_correction =", use_anderson)
    print("include_stall =", include_stall)
    base._report_trim_solution()
    ###########################################################################
    side = "left "
    fmnam = ["Fx","Fy","Fz","Mx","My","Mz"]
    unnam = ["lbf"]*3 + ["lbf-ft"]*3
    print("{:^{}s}".format(side,nast))
    lde = de-da/4.
    print("{} stabilator deflected {:> 7.3f} deg (trailing edge {:^4s})".format(
        side,lde,"down" if np.sign(lde)>=0.0 else "up"
    ))
    for i in range(6):
        print("{}_hstab_{} = {:> 10.3f} {}".format(fmnam[i],side,fmls[i],unnam[i]))
    print()
    side = "right"
    print("{:^{}s}".format(side,nast))
    rde = de+da/4.
    print("{} stabilator deflected {:> 7.3f} deg (trailing edge {:^4s})".format(
        side,rde,"down" if np.sign(rde)>=0.0 else "up"
    ))
    for i in range(6):
        print("{}_hstab_{} = {:> 10.3f} {}".format(fmnam[i],side,fmrs[i],unnam[i]))
    print("*"*nast)