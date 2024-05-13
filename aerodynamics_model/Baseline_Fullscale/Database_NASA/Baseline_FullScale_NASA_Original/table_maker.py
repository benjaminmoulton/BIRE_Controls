import os
import time
import numpy as np
import json
import subprocess as sp

RERUN_INTERPOLATOR = True


def run_interpolator_to_gen_tables(input_directory, input_variables):
    csv_file = input_directory+"/tables/CD_and_CL_table.csv"
    input_dictionary = {
        "aerodynamics" : {
            "database_directory" : "F16-NASAData/",
            "database_list" : ["C(x,y,z,l,m,n)(dh,b,a).csv",
                            "C(x,y,z,l,m,n),lef(b,a).csv",
                            "dC(x,z,m),sb(a).csv",
                            "dC(x,z,m)_q,lef(a).csv",
                            "C(x,z,m)_q(a).csv",
                            "C(y,n,l),da=20(b,a).csv",
                            "C(y,n,l),da=20,lef(b,a).csv",
                            "C(y,n,l),dr=30(b,a).csv",
                            "C(y,n,l)_r(a).csv",
                            "dC(y,n,l)_r,lef(a).csv",
                            "C(y,n,l)_p(a).csv",
                            "dC(y,n,l)_p,lef(a).csv", 
                            "dC(n,l)_b(a).csv",
                            "ndh(dh).csv",
                            "dCm(a).csv",
                            "dCm,ds(dh,a).csv"
                            ],
            "input_variables" : {
                "alpha[deg]" : input_variables[0],
                "beta[deg]" : input_variables[1],
                "dh[deg]" : input_variables[2],
                "lef[deg]" : input_variables[3],
                "sb[deg]" : input_variables[4],
                "da[deg]" : input_variables[5],
                "dr[deg]" : input_variables[6],
                "cBar[ft]" : input_variables[7],
                "b[ft]" : input_variables[8],
                "p[deg/s]" : input_variables[9],
                "q[deg/s]" : input_variables[10],
                "r[deg/s]" : input_variables[11],
                "V[ft/s]" : input_variables[12],
                "xcgShift" : input_variables[13]
            }
        }
    }
    # Dump the json vals
    input_file = input_directory + ".json"
    create_input_file(input_dictionary, input_file)

    # Run the interpolator
    output = run_interpolator(input_file, run = RERUN_INTERPOLATOR)

    # parse out the values
    # Parse output to extract the array values
    ans = [float(x) for x in output.strip().split()]

    # Pull out desired force coefficients (CL and CD for now)
    CF = np.array(ans[6:8]) # because CD and CL are indices 6 and 7 in the other file

    return CF


def create_input_file(input_dictionary, input_name):
    """Writes the given input dictionary to the given file location (so the code will run and the json will change in between runs)"""

    with open(input_name, 'w') as input_handle:
        json.dump(input_dictionary, input_handle, indent = 4)


def run_interpolator(input_name, delete_input=True, run=True):
    """Runs the interpolator code"""
    if run:
        # Capture output of subprocess
        result = sp.run(["./main", input_name], capture_output=True, text=True)
        output_lines = result.stdout.strip().split('\n')
        print(len(output_lines))
        forces_and_moments = output_lines[-2]  # Assuming the forces and moments in the cpp file are the second to last print statement
        for i in range(3,28):
            print(output_lines[-i])
        print("NUMERICAL DATA", forces_and_moments)
        return forces_and_moments
    
    # deletes old input
    if delete_input:
        os.remove(input_name)
    return []

def create_csv(array, output_file_name):

    if os.path.isfile(output_file_name):
        try:
            with open(output_file_name, 'wt') as f:
                pass
        except PermissionError:
            print("Error: File '{}' is likely open on your computer. Try closing the file and re-running the code.".format(output_file_name))
    np.savetxt(output_file_name + ".csv", array, delimiter=',', fmt='%s') # the fmt thing is just saying that it's printing strings out to the csv file. That way it doesn't get mad that there are floats and strings in the csv

    
if __name__ =="__main__":
    start = time.time()

    input_directory = "f16"

    input_variables = np.zeros((14, 20)) # 14 is the number of parameters, 20 is the number of alphas we're looping through here.
    input_variables[0] = np.array([-20.0,-15.0,-10.0,-5.0,0.0,5.0,10.0,15.0,20.0,25.0,30.0,35.0,40.0,45.0,50.0,55.0,60.0,70.0,80.0,90.0])
    input_variables[1] = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    input_variables[2] = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    input_variables[3] = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    input_variables[4] = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    input_variables[5] = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    input_variables[6] = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    input_variables[7] = np.array([11.32,11.32,11.32,11.32,11.32,11.32,11.32,11.32,11.32,11.32,11.32,11.32,11.32,11.32,11.32,11.32,11.32,11.32,11.32,11.32])
    input_variables[8] = np.array([30.0,30.0,30.0,30.0,30.0,30.0,30.0,30.0,30.0,30.0,30.0,30.0,30.0,30.0,30.0,30.0,30.0,30.0,30.0,30.0])
    input_variables[9] = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    input_variables[10] = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    input_variables[11] = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    input_variables[12] = np.array([200.0,200.0,200.0,200.0,200.0,200.0,200.0,200.0,200.0,200.0,200.0,200.0,200.0,200.0,200.0,200.0,200.0,200.0,200.0,200.0])
    input_variables[13] = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])

    alpha_CD_CL_list = []
    alpha_CD_CL_list.append(["numberIndependentVariables", "=", "1"])
    alpha_CD_CL_list.append(["alpha[deg]", "CD", "CL"])
    for i in range(input_variables.shape[1]):
        print("\n")
        CD, CL = run_interpolator_to_gen_tables(input_directory, input_variables[:,i])
        print("CD: ", CD, "\n", "CL: ", CL)
        print("\n")
        alpha_CD_CL_list.append([input_variables[0,i], CD, CL])
    alpha_CD_CL_array = np.array(alpha_CD_CL_list)

    # Write to CSV file
    output_file_name = "C(D,L)(alpha)" # you don't need to put the .csv in there. That's covered in the create_csv function
    create_csv(alpha_CD_CL_array, output_file_name)
    print("Data printed to a csv")
    print("Data written to '{}.csv' file.".format(output_file_name))


