import numpy as np
import json
from matplotlib import pyplot as plt
from controller_simulation import Aircraft,monte_carlo_perturbations,run_single_simulation


if __name__ == "__main__":

    # filenames 
    base_file = "base_fs_in.json"
    bire_file = "bire_fs_in.json"

    # read in json to ensure no file changes while running
    base_dict = json.loads( open(base_file).read() )
    bire_dict = json.loads( open(bire_file).read() )

    flight_conditions = {
        "T1" : { "m" : 0.2 , "h" :  1000., "V" : 222., "Re" : 15641000. },
        "T2" : { "m" : 0.19, "h" : 15000., "V" : 201., "Re" :  9919000. },
        "C1" : { "m" : 0.8 , "h" :  1000., "V" : 890., "Re" : 62563000. },
        "C2" : { "m" : 0.6 , "h" : 15000., "V" : 634., "Re" : 31324000. },
        "C3" : { "m" : 0.8 , "h" : 30000., "V" : 796., "Re" : 25828000. }
    }
    f1 = "C2"

    # run single case
    bire_dict["initial"]["trim_guess"] = {}
    bire_dict["initial"]["trim"]["type"] = "shss" # "sct" # 
    bire_dict["initial"]["trim"].pop("bank_angle[deg]")
    bire_dict["initial"]["trim"]["bank_angle[deg]"] = 5.0 # 10.0 # 0.0 # 
    # bire_dict["initial"]["trim"]["sideslip_angle[deg]"] = 11.0
    bire_dict["initial"].pop("mach")
    bire_dict["initial"]["airspeed[ft/s]"] = 222.0 # 350.0 # 
    # bire_dict["initial"]["mach"] = 0.2
    bire_dict["initial"]["altitude[ft]"] = 1000.0 # 15000.0 # 
    # bire_dict["initial"]["trim_guess"]["BIRE[deg]"] = -10.0
    # bire_dict["initial"]["trim_guess"]["elevator[deg]"] = 10.0
    bire_dict["aircraft"]["CG_shift[ft]"] = [0.0,0.0,0.0] # [1.0,0.0,0.0] # 
    bire_dict["simulation"]["include_compressibility"] = False
    bire_dict["simulation"]["use_Anderson_corrections"] = False
    bire_dict["simulation"]["include_stall"] = False
    bire_dict["simulation"]["use_fitted_thrust_model" ] = False
    bire = Aircraft(bire_dict)
    bire._report_trim_solution(bire.x_trim,bire.u_trim,bire.trim_iter)
    quit()
    ##############