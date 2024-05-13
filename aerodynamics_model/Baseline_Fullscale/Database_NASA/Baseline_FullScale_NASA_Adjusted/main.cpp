#include "aero.h"

int main (int argc, char * const argv[]) {
    if (argc != 2) {
        cout<<"Please enter a filename on the command line in the form: " << argv[0] << " <filename.json>" << endl;
        return 1;
    }

    const char* filename = argv[1];
    //// cout << "Program started with file: " << filename << endl;

    Atmosphere* myAtm = new Atmosphere();
    myAtm->print_tables();

    // Parse JSON file
    ifstream f(filename);
    json input = json::parse(f);
    Aero* myAero = new Aero(input["aerodynamics"]);

    // Extract parameters from JSON
    json parametersJson = input["aerodynamics"]["input_variables"];

    double inputVariables[14]; 
    inputVariables[0] = parametersJson["alpha[deg]"]; // angle of attack
    inputVariables[1] = parametersJson["beta[deg]"]; // sideslip angle
    inputVariables[2] = parametersJson["elevator[deg]"]; // elevator deflection 
    inputVariables[3] = parametersJson["lef[deg]"]; // leading edge flap deflection
    inputVariables[4] = parametersJson["sb[deg]"]; // speed brake deflection
    inputVariables[5] = parametersJson["aileron[deg]"]; // aileron deflection
    inputVariables[6] = parametersJson["rudder[deg]"]; // rudder deflection
    inputVariables[7] = parametersJson["cBar[ft]"]; // mean chord 
    inputVariables[8] = parametersJson["b[ft]"]; // span
    inputVariables[9] = parametersJson["p[deg/s]"]; // roll rate
    inputVariables[10] = parametersJson["q[deg/s]"]; // pitch rate
    inputVariables[11] = parametersJson["r[deg/s]"]; // yaw rate
    inputVariables[12] = parametersJson["V[ft/s]"]; // Freestream velocity
    inputVariables[13] = parametersJson["xcgShift"]; // center of gravity shift

    // cout << "HERE!!!!!!!!!!!!!!!!!" <<endl;
    double ans[14];
    myAero->get_FM_coefficients(inputVariables, ans);

    // myAero->populateInputFile()

    
    // for (int i = 8; i < 14; ++i) {
    //     cout << ans[i] << " ";
    // }
    // cout << endl;

    // cout<<"Program completed: "<<endl;
    cout << endl;
    return 0;
}



