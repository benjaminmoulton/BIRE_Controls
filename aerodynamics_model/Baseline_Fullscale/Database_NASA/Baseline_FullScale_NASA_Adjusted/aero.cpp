#include "aero.h"
// #include <iostream>
// #include <iomanip> // Include the <iomanip> header for setprecision

Aero::Aero(json dictionary) {
    string directory = dictionary["database_directory"];
    int numDatabase = dictionary["database_list"].size();
    json parameters = dictionary["input_variables"];

    for (int i = 0; i < numDatabase; i++) {
        string filename = directory + dictionary["database_list"][i].get<string>();
        Dataset* newDataset = new Dataset();
        newDataset->create_from_file(filename);
        mvDataset.push_back(newDataset);
        mvDatasetName.push_back(filename);
    }
}

double extra_bit_of_FM = 6;

void Aero::get_FM_coefficients(double inputVariables[14], double FM[8]) // CHANGE THIS TO FIT WHAT YOU NEED
{
    // Initialize FM to zero
    double extra_bit_of_FM = 6;
    double alpha = inputVariables[0];
    double beta = inputVariables[1];
    double elevator = inputVariables[2];
    double lef = inputVariables[3]*PI/180;
    double sb = inputVariables[4]*PI/180;
    double da = inputVariables[5]*PI/180; // 
    double dr = inputVariables[6]*PI/180; // 
    double cBar = inputVariables[7]; // [ft]
    double b = inputVariables[8]; // [ft]
    double p = inputVariables[9]*PI/180; // [rad/s]
    double q = inputVariables[10]*PI/180; // [rad/s]
    double r = inputVariables[11]*PI/180; // [rad/s]
    double V = inputVariables[12]; // [ft/s]
    double xcgShift = inputVariables[13];

    // Deflection evaluation points (in degrees)
    double beta_radians = beta*PI/180;
    double lef_from_table = 25.0*PI/180; // [rad]
    double sb_from_table = 60.0*PI/180; // [rad]
    double da_from_table = 20.0*PI/180; // [rad]
    double dr_from_table = 30.0*PI/180; // [rad]
    double pbar = p*b/(2*V); // [dimensionless]
    double qbar = q*cBar/(2*V); // [dimensionless]
    double rbar = r*b/(2*V); // [dimensionless]

    // NASA paper coefficients
    double Cx, Cxo, Cxlef, DCxlef, dCxsb, DCxsb, DCxq, Cxq, dCxqlef, DCxqlef;
    double Cy, Cyo,  Cylef, dCylef, DCylef, Cyda, Cydalef, dCyda, DCyda, dCydalef, DCydalef, Cydr, dCydr, DCydr, Cyr, DCyr, dCyrlef, DCyrlef, Cyp, DCyp, dCyplef, DCyplef;
    double Cz, Czo, Czlef, DCzlef, dCzsb, DCzsb, dCzqlef, DCzqlef, Czq, DCzq, dCzlef;
    double Cl, Clo, Cllef, dCllef, DCllef, Clda, dClda, DClda, dCldalef, DCldalef, dCldr, DCldr, Cldalef, Cldr, Clr, DClr, dClrlef, DClrlef, Clp, DClp, dClplef, DClplef, dClb, DClb;
    double Cm, Cmlef, dCmsb, dCmqlef, Cmq, dCmlef, dCm, dCmds, Cmo, DCmlef, DCmsb, DCmq, DCmqlef, DCm, DCmds;
    double Cn, Cnlef, Cnda, Cndalef, Cndr, Cnr, dCnrlef, Cnp, dCnplef, dCnb, dCnlef, dCnda, dCndalef, dCndr, Cno, DCnlef, DCnda, DCndalef, DCndr, DCnr, DCnrlef, DCnp, DCnplef, DCnb;
    double eta_elevator;


    for (int i=0; i < mvDataset.size(); i++) {
        darray inputs;
        darray ans;
        if (mvDatasetName[i] == "new_tables/C(x,y,z,l,m,n)o(alpha,beta,elevator).csv") {
            inputs.push_back(elevator);
            inputs.push_back(alpha);
            inputs.push_back(beta);
            ans = mvDataset[i]->linear_interpolate(inputs,0);
            Cxo = ans[0];
            Cyo = ans[1];
            Czo = ans[2];
            Clo = ans[3];
            Cmo = ans[4];
            Cno = ans[5];
        } 
        else if (mvDatasetName[i] == "new_tables/DC(x,y,z,l,m,n),f(alpha,beta).csv") {
            inputs.push_back(alpha);
            inputs.push_back(beta);
            ans = mvDataset[i]->linear_interpolate(inputs,0);
            DCxlef = ans[0];
            DCylef = ans[1];
            DCzlef = ans[2];
            DCllef = ans[3];
            DCmlef = ans[4];
            DCnlef = ans[5];
        } 
        else if (mvDatasetName[i] == "new_tables/DC(x,z,m),sb(alpha).csv") {
            inputs.push_back(alpha);
            ans = mvDataset[i]->linear_interpolate(inputs,0);
            DCxsb = ans[0];
            DCzsb = ans[1];
            DCmsb = ans[2];
        } 
        else if (mvDatasetName[i] == "new_tables/DC(x,z,m),qbar(alpha).csv") {
            inputs.push_back(alpha);
            ans = mvDataset[i]->linear_interpolate(inputs,0);
            DCxq = ans[0];//*180/PI;
            DCzq = ans[1];//*180/PI;
            DCmq = ans[2];//*180/PI;
        } 
        else if (mvDatasetName[i] == "new_tables/DC(x,z,m)qbar,f(alpha).csv") {
            inputs.push_back(alpha);
            ans = mvDataset[i]->linear_interpolate(inputs,0);
            DCxqlef = ans[0];//*180/PI;
            DCzqlef = ans[1];//*180/PI;
            DCmqlef = ans[2];//*180/PI;
        } 
        else if (mvDatasetName[i] == "new_tables/DC(y,l,n),aileron(alpha,beta).csv") {
            inputs.push_back(alpha);
            inputs.push_back(beta);
            ans = mvDataset[i]->linear_interpolate(inputs,0);
            DCyda = ans[0];
            DClda = ans[1];
            DCnda = ans[2];
        } 
        else if (mvDatasetName[i] == "new_tables/DC(y,l,n)aileron,f(alpha,beta).csv") {
            inputs.push_back(alpha);
            inputs.push_back(beta);
            ans = mvDataset[i]->linear_interpolate(inputs,0);
            DCydalef = ans[0];
            DCldalef = ans[1];
            DCndalef = ans[2];
        }
        else if (mvDatasetName[i] == "new_tables/DC(y,l,n),rudder(alpha,beta).csv") {
            inputs.push_back(alpha);
            inputs.push_back(beta);
            ans = mvDataset[i]->linear_interpolate(inputs,0);
            DCydr = ans[0];
            DCldr = ans[1];
            DCndr = ans[2];
        }
        else if (mvDatasetName[i] == "new_tables/DC(y,l,n),rbar(alpha).csv") {
            inputs.push_back(alpha);
            ans = mvDataset[i]->linear_interpolate(inputs,0);
            DCyr = ans[0];//*180/PI;
            DClr = ans[1];//*180/PI;
            DCnr = ans[2];//*180/PI;
        }
        else if (mvDatasetName[i] == "new_tables/DC(y,l,n)rbar,f(alpha).csv") {
            inputs.push_back(alpha);
            ans = mvDataset[i]->linear_interpolate(inputs,0);
            DCyrlef = ans[0];//*180/PI;
            DClrlef = ans[1];//*180/PI;
            DCnrlef = ans[2];//*180/PI;
        }
        else if (mvDatasetName[i] == "new_tables/DC(y,l,n),pbar(alpha).csv") {
            inputs.push_back(alpha);
            ans = mvDataset[i]->linear_interpolate(inputs,0);
            DCyp = ans[0];//*180/PI;
            DClp = ans[1];//*180/PI;
            DCnp = ans[2];//*180/PI;
        }
        else if (mvDatasetName[i] == "new_tables/DC(y,l,n)pbar,f(alpha).csv") {
            inputs.push_back(alpha);
            ans = mvDataset[i]->linear_interpolate(inputs,0);
            DCyplef = ans[0];//*180/PI;
            DClplef = ans[1];//*180/PI;
            DCnplef = ans[2];//*180/PI;
        }
        else if (mvDatasetName[i] == "new_tables/DC(l,n),beta(alpha).csv") {
            inputs.push_back(alpha);
            ans = mvDataset[i]->linear_interpolate(inputs,0);      
            DClb = ans[0];//*180/PI;
            DCnb = ans[1];//*180/PI;
        }
        else if (mvDatasetName[i] == "new_tables/DCm(alpha).csv") {
            inputs.push_back(alpha);
            ans = mvDataset[i]->linear_interpolate(inputs,0);
            DCm = ans[0];
        }
        else if (mvDatasetName[i] == "new_tables/DCm,ds(alpha,elevator).csv") {
            inputs.push_back(elevator);
            inputs.push_back(alpha);
            ans = mvDataset[i]->linear_interpolate(inputs,0);
            DCmds = ans[0];
        }  
        else {
            //cout << "Error: Unsupported dataset name in get_FM_coefficients" << endl;
        }
        // cout<<"Interpolated values for dataset "<<mvDatasetName[i]<<endl;
        // darray_print(ans);
        // cout<<endl;
    }
    //cout<<"Interpolated values for dataset at elevator = 0: "<<endl;

    // darray inputs;
    // inputs.push_back(0.0); // elevator
    // inputs.push_back(beta);
    // inputs.push_back(alpha);
    // darray temp = mvDataset[0]->linear_interpolate(inputs,0);

    //cout<<"Interpolated values for dataset at elevator = 0: "<<mvDatasetName[0]<<endl;
    // darray_print(temp);
    //cout<<endl;
    // temp[0] = temp[0]*PI/180;
    //cout<<"Here are the independent variables and aircraft states "<<endl;
    //cout<<"Alpha = "<<inputVariables[0]<< "[deg]" <<endl;
    //cout<<"Beta = "<<inputVariables[1]<< "[deg]" <<endl;
    //cout<<"elevator = "<<inputVariables[2]<< "[deg]" <<endl;
    //cout<<"lef = "<<inputVariables[3]<< "[deg]" <<endl;
    //cout<<"sb = "<<inputVariables[4]<< "[deg]" <<endl;
    //cout<<"aileron = "<<inputVariables[5]<< "[deg]" <<endl;
    //cout<<"rudder = "<<inputVariables[6]<< "[deg]" <<endl;
    //cout<<"cBar = "<<inputVariables[7]<< "[ft]" <<endl;
    //cout<<"b = "<<inputVariables[8]<< "[ft]" <<endl;
    //cout<<"p = "<<inputVariables[9]<< "[deg/s]" <<endl;
    //cout<<"q = "<<inputVariables[10]<< "[deg/s]" <<endl;
    //cout<<"r = "<<inputVariables[11]<< "[deg/s]" <<endl;
    //cout<<"V = "<<inputVariables[12]<< "[ft/s]" <<endl;
    //cout<<"xcgShift = "<<inputVariables[13]<< "[ft]" <<endl;
    //cout<<endl;

    //cout<<"Here are the resulting forces and moments "<<endl;
    // These go into new tables
    // Here is the new equation for the total x force coefficient, Cx
    FM[0] = Cxo + DCxlef*lef + DCxsb*sb + DCxq*qbar + DCxqlef*qbar*lef;
    //cout<<"Cx_new = "<<FM[0]<<endl;

    // Here is the equation for total y force coefficient, Cy
    FM[1] = Cyo + DCylef*lef + DCyda*da + DCydalef*da*lef + DCydr*dr + DCyr*rbar + DCyrlef*rbar*lef + DCyp*pbar + DCyplef*pbar*lef;
    //cout<<"Cy_new = "<<FM[1]<<endl;

    // Here is the new equation for Cz
    FM[2] = Czo + DCzlef*lef + DCzsb*sb + DCzq*qbar + DCzqlef*qbar*lef;
    //cout<<"Cz_new = "<<FM[2]<<endl;

    // Here is the new equation for total rolling moment coefficient, Cl
    FM[3] = Clo + DCllef*lef + DClda*da + DCldalef*da*lef + DCldr*dr + DClr*rbar + DClrlef*rbar*lef + DClp*pbar + DClplef*pbar*lef + DClb*beta_radians;
    //cout<<"Cl_new = "<<FM[3]<<endl;

    // Here is the new equation for total pitching moment coefficient, Cm
    FM[4] = Cmo + DCmlef*lef + DCmsb*sb + DCmq*qbar + DCmqlef*qbar*lef + DCm + DCmds;
    //cout<<"Cm_new = "<<FM[4]<<endl;

    // Here is the equation for total yawing moment coefficient, Cn
    FM[5] = Cno + DCnlef*lef + DCnda*da + DCndalef*da*lef + DCndr*dr + DCnr*rbar + DCnrlef*rbar*lef + DCnp*pbar + DCnplef*pbar*lef + DCnb*beta_radians;
    //cout<<"Cn_new = "<<FM[5]<<endl;

    // Here is the equation for total drag coefficient CD (see equations 4.23 and 4.24 in Christian's dissertation)
    FM[6] = -FM[0]*cos(alpha*PI/180)*cos(beta*PI/180) - FM[1]*sin(beta*PI/180) - FM[2]*sin(alpha*PI/180)*cos(beta*PI/180); 
    //cout<<"CD = "<<FM[6]<<endl;

    // Here is the equation for total lift coefficient CL (see equations 4.23 and 4.24 in Christian's dissertation)
    FM[7] = -FM[2]*cos(alpha*PI/180) + FM[0]*sin(alpha*PI/180);
    //cout<<"CL = "<<FM[7]<<endl;
    
    //// change these 
    cout<<"Cx: "<< FM[0]<<endl; 
    cout<<"Cy: "<< FM[1]<<endl; 
    cout<<"Cz: "<< FM[2]<<endl; 
    cout<<"Cl: "<< FM[3]<<endl; 
    cout<<"Cm: "<< FM[4]<<endl; 
    cout<<"Cn: "<< FM[5]<<endl; 
}