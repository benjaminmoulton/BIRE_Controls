import machupX as mx
import numpy as np
import json
from matplotlib import pyplot as plt
from std_atm import stdatm_english

if __name__ == "__main__":
    f16_scene_file = "F16_input.json"
    f16_craft_file = "F16_airplane.json"
    base_fs_dict = json.loads( open(f16_craft_file).read() )
    base_fs_scene = json.loads( open(f16_scene_file).read() )
    # print(base_fs_dict["CG"])
    base_fs_dict["CG"] = [-1.755,0.0,0.0] # [0.0,0.0,0.0] # 

    # change aircraft state
    # aerodynamic angles
    H = 20000.0
    W = 27303.83 # 20500.0 # 
    base_fs_dict["weight"] = W
    # f16._aircraft["F16"].W = W
    _,g,_,_,rho,sos = stdatm_english(H)
    M = 0.9
    V = M*sos
    alpha = 13.5 # 21.25 # 0.0 # 
    beta = 0.0
    phi = 0.0
    theta = alpha
    psi = 0.0
    nz = 8.34
    p = np.deg2rad( 0.0 ) # p in deg/s converted to rad/s
    q = np.deg2rad( 10.0 ) # q in deg/s converted to rad/s
    r = np.deg2rad( 0.0 ) # r in deg/s converted to rad/s
    # build scene
    base_fs_scene["scene"]["aircraft"]["F16"]["file"] = base_fs_dict
    f16 = mx.Scene(base_fs_scene)
    # f16.display_wireframe()
    # FM = f16.solve_forces(body_frame=True,stab_frame=False,wind_frame=False,\
    #     report_by_segment=True,dimensional=True,non_dimensional=False)
    # print(json.dumps(FM,indent=4))
    state = {
        "velocity" : V, # total velocity
        "alpha" : alpha, # angle of attack in degrees
        "beta" : beta,
        "orientation" : [phi, theta, psi], # Earth-fixed orientation, bank (phi), elevation (theta), heading (psi) in degrees
        "angular_rates" : [p,q,r], # body-fixed rotation rates, roll (p), pitch (q), yaw (r) radians
        "angular_rates_frame" : "body"
    }
    ctrl_state = {
        "aileron" : 0.0, # setting in degrees
        "elevator" : -18.0, # setting in degrees
        "rudder" : 0.0 # setting in degrees
    }
    forces_settings = dict(body_frame=True,stab_frame=False,wind_frame=False,\
        report_by_segment=True,dimensional=True,non_dimensional=False)
    #
    # set aircraft and control state, solve forces
    f16.set_aircraft_state(state)
    f16.set_aircraft_control_state(ctrl_state)
    # f16.display_wireframe()
    FM = f16.solve_forces(**forces_settings)
    Fz = FM["F16"]["total"]["Fz"]
    # print(Fz)
    NZ = - Fz / W
    print("NZ =", NZ)
    FM["F16"]["combined"] = {}
    for fom in FM["F16"]["inviscid"]:
        FM["F16"]["combined"][fom] = {}
        for surface in FM["F16"]["inviscid"][fom]:
            invisc = FM["F16"]["inviscid"][fom][surface]
            visc   = FM["F16"][ "viscous"][fom][surface]
            FM["F16"]["combined"][fom][surface] = invisc + visc
    # correct for compressibility
    # /np.sqrt(1. - M**2)
    print(json.dumps(FM,indent=4))
    Fz_hstab_left  = FM["F16"]["combined"]["Fz"]["h_stab_left"]
    Fz_hstab_right = FM["F16"]["combined"]["Fz"]["h_stab_right"]
    print("CG =",base_fs_dict["CG"])
    print("Fz_hstab_left  =",Fz_hstab_left)
    print("Fz_hstab_right =",Fz_hstab_right)