import os
import time
import numpy as np
import json
import subprocess as sp

RERUN_INTERPOLATOR = True


def run_interpolator_to_gen_tables(input_directory, input_variables): ## output[-6:] in inputs for this???
    csv_file = input_directory+"/tables/CD_and_CL_table.csv"
    input_dictionary = {
        "aerodynamics" : {
            "database_directory" : "F16-NASAData/",
            "database_list": [
                "C(x,y,z,l,m,n)(elevator,beta,alpha).csv",
                "C(x,y,z,l,m,n),lef(beta,alpha).csv",
                "dC(x,z,m),speedbrake(alpha).csv",
                "C(x,z,m)_qbar(alpha).csv",
                "dC(x,z,m)_qbar,lef(alpha).csv",
                "C(y,n,l),aileron(beta,alpha).csv",
                "C(y,n,l),aileron,lef(beta,alpha).csv",
                "C(y,n,l),rudder(beta,alpha).csv",
                "C(y,n,l)_rbar(alpha).csv",
                "dC(y,n,l)_rbar,lef(alpha).csv",
                "C(y,n,l)_pbar(alpha).csv",
                "dC(y,n,l)_pbar,lef(alpha).csv",
                "dC(n,l)_beta(alpha).csv",
                "etaElevator(elevator).csv",
                "dCm(alpha).csv",
                "dCm,ds(elevator,alpha).csv"
            ],
            "input_variables" : {
                "alpha[deg]" : input_variables[0],
                "beta[deg]" : input_variables[1],
                "elevator[deg]" : input_variables[2],
                "lef[deg]" : input_variables[3],
                "sb[deg]" : input_variables[4],
                 "aileron[deg]" : input_variables[5],
                "rudder[deg]" : input_variables[6],
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



    # Parse output to extract the array values
    ans = []

    for line in output[-6:]: # CHANGE HERE EACH TIME THIS IS HOW MANY PRINT STATEMENTS BACK WE GO TO GET OUR VALUES
        ans.extend([float(x) for x in line.split()])

    CF = np.array(ans)

    return CF


def create_input_file(input_dictionary, input_name):
    """Writes the given input dictionary to the given file location (so the code will run and the json will change in between runs)"""

    with open(input_name, 'w') as input_handle:
        json.dump(input_dictionary, input_handle, indent = 4)


def run_interpolator(input_name, delete_input=True, run=True): ## consider putting the number in the bracket in output_lines[-6:] in the inputs for this
    """Runs the interpolator code"""
    if run:
        # Capture output of subprocess
        result = sp.run(["./main", input_name], capture_output=True, text=True)
        output_lines = result.stdout.strip().split('\n')
        # print(len(output_lines))
        # print(output_lines)
        forces_and_moments = output_lines[-6:]  # CHANGE HERE EACH TIME THIS IS HOW MANY PRINT STATEMENTS BACK WE GO TO GET OUR VALUES

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

def populate_inputs(input_array, value):
    """This function takes in a numpy array called input_array and fills it with the same value"""
    for i in range(len(input_array)):
        input_array[i] = value
    return input_array

if __name__ =="__main__":
    start = time.time()

    input_directory = "f16"


    beta_range = np.array([0.0])

    # beta_range = np.array([-30.0, -25.0, -20.0, -15.0, -10.0, -8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0])
    
    # alpha_range = np.array([-20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0])

    alpha_range = np.array([-20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 70.0, 80.0, 90.0])
    
    elevator_range = np.array([0.0]) #### Change this for your given table

    # elevator_range = np.array([-25.0, -10.0, 0.0, 10.0, 25.0])
    
    lef_range_inputs = np.zeros(len(beta_range))
    
    sb_range_inputs = np.zeros(len(beta_range))
    
    aileron_range_inputs = np.zeros(len(beta_range))
    
    rudder_range_inputs = np.zeros(len(beta_range))
    
    cBar_range_inputs = np.zeros(len(beta_range))
    cBar_range_inputs = populate_inputs(cBar_range_inputs, 11.32)
    
    b_range_inputs = np.zeros(len(beta_range))
    b_range_inputs = populate_inputs(b_range_inputs, 30.0)
    
    p_range_inputs = np.zeros(len(beta_range))
    
    q_range_inputs = np.zeros(len(beta_range))
    
    r_range_inputs = np.zeros(len(beta_range))
    
    V_range_inputs = np.zeros(len(beta_range))
    V_range_inputs = populate_inputs(V_range_inputs, 200.0)
    
    xcg_shift_inputs = np.zeros(len(beta_range))

    num_major_independent_variables = 3
    num_dependent_variables = 6

    new_table = np.empty((len(elevator_range) * len(alpha_range) * len(beta_range) + 2, 9), dtype=object)
    new_table[0] = ["numberIndependentVariables", "=", "1", 0, 0, 0, 0, 0, 0] #### change num independent variables
    new_table[1] = ["elevator[deg]", "alpha[deg]", "beta[deg]", "DClbeta", "DCnbeta", "x", "x", "x", "x"] #### change these values to the ones you want on the table
    # Create an empty 2D array to store the combinations
    independent_variables = np.empty((len(elevator_range) * len(alpha_range) * len(beta_range) + 2, 14), dtype=object) # the dtype thing allows for non numbers to go into the csv
    independent_variables[0] = ["mainIndependentVariables", "=", "1", 0, 0, 0, 0, 0, 0,0,0,0,0,0]
    independent_variables[1] = ["alpha[deg]", "beta[deg]", "elevator[deg]", "f[deg]", "sb[deg]", "aileron[deg]", "rudder[deg]", "cBar[ft]", "b[ft]", "p[deg/s]", "q[deg/s]", "r[deg/s]", "V[ft/s]", "xcgShift"]  # Titles for columns
    index = 2
    for i in range(len(elevator_range)):
        for j in range(len(alpha_range)):
            for k in range(len(beta_range)):
                independent_variables[index] = [alpha_range[j], beta_range[k], elevator_range[i], lef_range_inputs[k], sb_range_inputs[k], aileron_range_inputs[k], rudder_range_inputs[k], cBar_range_inputs[k], b_range_inputs[k], p_range_inputs[k], q_range_inputs[k], r_range_inputs[k], V_range_inputs[k], xcg_shift_inputs[k]]
                new_table[index] = [elevator_range[i], alpha_range[j], beta_range[k], 0, 0, 0, 0, 0, 0]
                index += 1
    
    create_csv(independent_variables, "test_file")
    
    for i in range(2, independent_variables.shape[0]): # shape[0] is just the number of rows rather than shape[1] being the number of columns
        # print("\n")
        Cx, Cy, Cz, Cl, Cm, Cn = run_interpolator_to_gen_tables(input_directory, independent_variables[i,:]) 
        force_list = [Cx, Cy, Cz, Cl, Cm, Cn] 
        for j in range(3, len(new_table[i])):
            new_table[i][j] = force_list[j-3]  
 
    # Write to CSV file
    output_file_name = "DC(l,n),beta(alpha)" #### change these values to the ones you want on the table
    create_csv(new_table, output_file_name)
    print("Data written to '{}.csv' file.".format(output_file_name))


