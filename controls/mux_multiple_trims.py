import machupX as mux
import numpy as np

from os import mkdir, rmdir, walk, remove, listdir
from scipy.optimize import minimize as minmiz
import json

if __name__ == "__main__":
    # # filenames 
    # base_file = "base_fs_in.json"
    # bire_file = "bire_fs_in.json"

    # # read in json to ensure no file changes while running
    # base_dict = json.loads( open(base_file).read() )
    # bire_dict = json.loads( open(bire_file).read() )
    
    # flight conditions
    u1M = 0.331014489952403
    flight_conditions = {
        "A1" : { "m" : 0.2 , "h" :     0., "V" : 222., "Re" :        0. },
        "T1" : { "m" : 0.2 , "h" :  1000., "V" : 222., "Re" : 15641000. },
        "T2" : { "m" : 0.19, "h" : 15000., "V" : 201., "Re" :  9919000. },
        "C1" : { "m" : 0.8 , "h" :  1000., "V" : 890., "Re" : 62563000. },
        "C2" : { "m" : 0.6 , "h" : 15000., "V" : 634., "Re" : 31324000. },
        "C3" : { "m" : 0.8 , "h" : 30000., "V" : 796., "Re" : 25828000. },
        "U1" : { "m" : u1M , "h" : 15000., "V" : 350., "Re" : "unkn"    }, # no compr no stall
        "F1" : { "m" : 0.6 , "h" : 15000., "V" : 634., "Re" : 31324000. }, # no compr no stall
    }

    # settings 
    ## Continue from bire_fs_shss_T1_M02_H010_CGp10p00p00_B14
    run_bire = True # False # 
    run_sct  = True # False # 
    run_fs = True
    skip_run = False # True # 
    skip_DOC = True # False # 
    if run_sct: trim_bank_degs = np.linspace(0.0,60.0,num=13).tolist() # [0.0] # np.linspace(0.0,75.0,num=16).tolist() # [10.0] # [60.0] # np.linspace(0.0,75.0,num=16).tolist() # 
    else: trim_beta_degs = [6.0] # np.linspace(0.0,16.0,num=9).tolist() # np.linspace(0.0,16.0,num=9).tolist() # [14.0,16.0] # [0.0] # 
    fc = "F1" # "U1" # "C2" # "T1" # "A1" # 
    cgshift = [0.0,0.0,0.0] # [1.0,0.0,0.0] # [0.5,0.0,0.0] # [0.5,0.0,0.0] # 
    include_compressibility =  False # True # 
    use_Anderson_corrections =  True # False # 
    include_stall =  False # True # 
    plotting_xcgs = [0.0,0.5,1.0]
    plot_inverted_trims = False # True # 
    plot_alternate_trims = True # False # 
    show_plots = False # True # 
    plot_format = "pdf" # "png" # 
    plot_transparent = True if plot_format == "pdf" else False # False # True # 
    plot_dark = True # False # 
    #
    # other settings
    run_num = 30 # 1000 # 
    trim_iter = 1000 # 1000
    mfc = flight_conditions[fc]["m"] # 0.2 # 
    hfc = flight_conditions[fc]["h"] # 1000.0 # 
    a_scale = 20.0 # 0.02 # 0.0 # 
    b_scale = 20.0 # 0.2 # 0.0 # 
    p_scale = 180.0 # 0.0 # 
    u_scale = np.array([20.0,20.0,70.0,1.0]) # np.array([0.1,0.2,20.0,0.02]) # np.array([0.0]*4) # 
    a_shift = 0.0 # 3.17 # 
    b_shift = 0.0 # 0.0 # 
    p_shift = 0.0 # 0.0 # 
    u_shift = np.array([0.0,0.0,0.0,0.0]) # np.array([0.0,-0.05,0.0,0.276]) # 
    # set up run
    # craftdict = bire_dict if run_bire else base_dict
    trim_type = "sct" if run_sct else "shss"
    scale_type = "fs" if run_fs  else "rc"
    folder = "trim_files/" + fc + "_" + trim_type + "/"

    # initialize MUX model -- read in JSON
    input_file = "../aerodynamics_model/BIRE Inputs/BIRE_input.json"
    input_dict = json.loads( open(input_file).read() )
    craft_file = "../aerodynamics_model/BIRE Inputs/BIRE_airplane.json"
    craft_dict = json.loads( open(craft_file).read() )

    # optimizer runs
    for filename in listdir(folder):
        file_split = filename.replace(".json","").split("_")
        if file_split[1:4] == [scale_type, trim_type, fc]:
            # open file
            cases_dict = json.loads( open(folder + filename).read() )
            for case in cases_dict:
                print(folder,filename,"yeah!")
                print(cases_dict[case])
                x_trim = cases_dict[case]["x_trim_euler"]
                u_trim = cases_dict[case]["u_trim"]

                # update aircraft state in mux aircraft dict

                # initialize mux

                # define optimizer cost function

                # run optimizer

                # print success or no...

