#include "aero.h"

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


void Aero::get_FM_coefficients(double inputVariables[14], double FM[14])
{
    // Initialize FM to zero
    double alpha = inputVariables[0];
    double beta = inputVariables[1];
    double dh = inputVariables[2];
    double lef = inputVariables[3];
    double sb = inputVariables[4];
    double da = inputVariables[5];
    double dr = inputVariables[6];
    double cBar = inputVariables[7];
    double b = inputVariables[8];
    double p = inputVariables[9];
    double q = inputVariables[10];
    double r = inputVariables[11];
    double V = inputVariables[12];
    double xcgShift = inputVariables[13];

    // Deflection evaluation points (in degrees)
    double lef_from_table = 25.0; // [deg]
    double sb_from_table = 60.0; // [deg]
    double da_from_table = 20.0; // [deg]
    double dr_from_table = 30.0; // [deg]

    // NASA paper coefficients
    double Cx, Cxlef, dCxlef, dCxsb, Cxq, dCxqlef;
    double Cy, Cylef, dCylef, Cyda, Cydalef, dCyda, dCydalef, Cydr, dCydr, Cyr, dCyrlef, Cyp, dCyplef;
    double Cz, Czlef, dCzsb, dCzqlef, Czq, dCzlef;
    double Cl, Cllef, dCllef, Clda, dClda, dCldalef, dCldr, Cldalef, Cldr, Clr, dClrlef, Clp, dClplef, dClb;
    double Cm, Cmlef, dCmsb, dCmqlef, Cmq, dCmlef, dCm, dCmds;
    double Cn, Cnlef, Cnda, Cndalef, Cndr, Cnr, dCnrlef, Cnp, dCnplef, dCnb, dCnlef, dCnda, dCndalef, dCndr;
    double eta_dh;

    // New paper coefficients 
    double Cxo, dCxlefnew, dCxsbnew, dCxqnew, dCxqlefnew, DCxq;
    double Cyo, dCylefnew, dCydanew, dCydalefnew, DCyda, dCydrnew, dCyrnew, dCyrlefnew, DCyr, dCypnew, dCyplefnew, DCyp;
    double Czo, dCzlefnew, dCzsbnew, dCzqnew, dCzqlefnew, DCzq;

    for (int i=0; i < mvDataset.size(); i++) {
        darray inputs;
        darray ans;
        if (mvDatasetName[i] == "F16-NASAData/C(x,y,z,l,m,n)(dh,b,a).csv") {
            inputs.push_back(dh);
            inputs.push_back(beta);
            inputs.push_back(alpha);
            ans = mvDataset[i]->linear_interpolate(inputs,0);
            Cx = ans[0];
            Cy = ans[1];
            Cz = ans[2];
            Cl = ans[3];
            Cm = ans[4];
            Cn = ans[5];
        } 
        else if (mvDatasetName[i] == "F16-NASAData/C(x,y,z,l,m,n),lef(b,a).csv") {
            inputs.push_back(beta);
            inputs.push_back(alpha);
            ans = mvDataset[i]->linear_interpolate(inputs,0);

            Cxlef = ans[0];
            Cylef = ans[1];
            Czlef = ans[2];
            Cllef = ans[3];
            Cmlef = ans[4];
            Cnlef = ans[5];
        } 
        else if (mvDatasetName[i] == "F16-NASAData/dC(x,z,m),sb(a).csv") {
            inputs.push_back(alpha);
            ans = mvDataset[i]->linear_interpolate(inputs,0);
            dCxsb = ans[0];
            dCzsb = ans[1];
            dCmsb = ans[2];
        } 
        else if (mvDatasetName[i] == "F16-NASAData/C(x,z,m)_q(a).csv") {
            inputs.push_back(alpha);
            ans = mvDataset[i]->linear_interpolate(inputs,0);
            Cxq = ans[0];
            Czq = ans[1];
            Cmq = ans[2];
        } 
        else if (mvDatasetName[i] == "F16-NASAData/dC(x,z,m)_q,lef(a).csv") {
            inputs.push_back(alpha);
            ans = mvDataset[i]->linear_interpolate(inputs,0);
            dCxqlef = ans[0];
            dCzqlef = ans[1];
            dCmqlef = ans[2];
        } 
        else if (mvDatasetName[i] == "F16-NASAData/C(y,n,l),da=20(b,a).csv") {
            inputs.push_back(beta);
            inputs.push_back(alpha);
            ans = mvDataset[i]->linear_interpolate(inputs,0);
            Cyda = ans[0];
            Cnda = ans[1];
            Clda = ans[2];
        } 
        else if (mvDatasetName[i] == "F16-NASAData/C(y,n,l),da=20,lef(b,a).csv") {
            inputs.push_back(beta);
            inputs.push_back(alpha);
            ans = mvDataset[i]->linear_interpolate(inputs,0);

            Cydalef = ans[0];
            Cndalef = ans[1];
            Cldalef = ans[2];
        }
        else if (mvDatasetName[i] == "F16-NASAData/C(y,n,l),dr=30(b,a).csv") {
            inputs.push_back(beta);
            inputs.push_back(alpha);
            ans = mvDataset[i]->linear_interpolate(inputs,0);

            Cydr = ans[0];
            Cndr = ans[1];
            Cldr = ans[2];
        }
        else if (mvDatasetName[i] == "F16-NASAData/C(y,n,l)_r(a).csv") {
            inputs.push_back(alpha);
            ans = mvDataset[i]->linear_interpolate(inputs,0);
            Cyr = ans[0];
            Cnr = ans[1];
            Clr = ans[2];
        }
        else if (mvDatasetName[i] == "F16-NASAData/dC(y,n,l)_r,lef(a).csv") {
            inputs.push_back(alpha);
            ans = mvDataset[i]->linear_interpolate(inputs,0);

            dCyrlef = ans[0];
            dCnrlef = ans[1];
            dClrlef = ans[2];
        }
        else if (mvDatasetName[i] == "F16-NASAData/C(y,n,l)_p(a).csv") {
            inputs.push_back(alpha);
            ans = mvDataset[i]->linear_interpolate(inputs,0);

            Cyp = ans[0];
            Cnp = ans[1];
            Clp = ans[2];
        }
        else if (mvDatasetName[i] == "F16-NASAData/dC(y,n,l)_p,lef(a).csv") {
            inputs.push_back(alpha);
            ans = mvDataset[i]->linear_interpolate(inputs,0);

            dCyplef = ans[0];
            dCnplef = ans[1];
            dClplef = ans[2];
        }
        else if (mvDatasetName[i] == "F16-NASAData/dC(n,l)_b(a).csv") {
            inputs.push_back(alpha);
            ans = mvDataset[i]->linear_interpolate(inputs,0);
            
            dCnb = ans[0];
            dClb = ans[1];
        }
        else if (mvDatasetName[i] == "F16-NASAData/ndh(dh).csv") {
            inputs.push_back(dh);
            ans = mvDataset[i]->linear_interpolate(inputs,0);

            eta_dh = ans[0];
        }
        else if (mvDatasetName[i] == "F16-NASAData/dCm(a).csv") {
            inputs.push_back(alpha);
            ans = mvDataset[i]->linear_interpolate(inputs,0);

            dCm = ans[0];
        }
        else if (mvDatasetName[i] == "F16-NASAData/dCm,ds(dh,a).csv") {
            inputs.push_back(dh);
            inputs.push_back(alpha);
            ans = mvDataset[i]->linear_interpolate(inputs,0);

            dCmds = ans[0];
        }  
        else {
            cout << "Error: Unsupported dataset name in get_FM_coefficients" << endl;
        }
        // cout<<"Interpolated values for dataset "<<mvDatasetName[i]<<endl;
        // darray_print(ans);
        // cout<<endl;
    }

    darray inputs;
    inputs.push_back(0.0); // dh
    inputs.push_back(beta);
    inputs.push_back(alpha);
    darray temp = mvDataset[0]->linear_interpolate(inputs,0);

    cout<<"Interpolated values for dataset at dh = 0: "<<mvDatasetName[0]<<endl;
    darray_print(temp);
    cout<<endl;

    cout<<"Here are the independent variables and aircraft states "<<endl;
    cout<<"Alpha = "<<inputVariables[0]<< "[deg]" <<endl;
    cout<<"Beta = "<<inputVariables[1]<< "[deg]" <<endl;
    cout<<"dh = "<<inputVariables[2]<< "[deg]" <<endl;
    cout<<"lef = "<<inputVariables[3]<< "[deg]" <<endl;
    cout<<"sb = "<<inputVariables[4]<< "[deg]" <<endl;
    cout<<"da = "<<inputVariables[5]<< "[deg]" <<endl;
    cout<<"dr = "<<inputVariables[6]<< "[deg]" <<endl;
    cout<<"cBar = "<<inputVariables[7]<< "[ft]" <<endl;
    cout<<"b = "<<inputVariables[8]<< "[ft]" <<endl;
    cout<<"p = "<<inputVariables[9]<< "[deg/s]" <<endl;
    cout<<"q = "<<inputVariables[10]<< "[deg/s]" <<endl;
    cout<<"r = "<<inputVariables[11]<< "[deg/s]" <<endl;
    cout<<"V = "<<inputVariables[12]<< "[ft/s]" <<endl;
    cout<<"xcgShift = "<<inputVariables[13]<< "[ft]" <<endl;
    cout<<endl;

    cout<<"Here are the resulting forces and moments "<<endl;
    // Here is the equation for total x force coefficient, Cx,t
    dCxlef = Cxlef - temp[0];
    FM[0] = Cx + dCxlef*(1.0 - lef/25.0) + dCxsb*(sb/60) + (cBar*q/(2*V))*(Cxq + dCxqlef*(1-lef/25)); // (double check) see appendix B of NASA paper for this Cx,t equation 
    cout<<"Cx = "<<FM[0]<<endl;

    // Here is the equation for total y force coefficient, Cy,t
    dCylef = Cylef - temp[1];
    dCyda = Cyda - temp[1];  // Check to see if Cy is independent of dh...
    dCydalef = Cydalef - Cylef - (Cyda - temp[1]);
    dCydr = Cydr - temp[1];
    FM[1] = Cy + dCylef*(1-lef/25.0) + (dCyda+dCydalef*(1-lef/25.0))*(da/20.0) + dCydr*(dr/30.0) + (b/(2*V))*((Cyr+dCyrlef*(1-lef/25.0))*r + (Cyp+dCyplef*(1-lef/25.0))*p); // (double check) see appendix B of NASA paper for this Cy,t equation
    cout<<"Cy = "<<FM[1]<<endl;

    // Here is the equation for total z force coefficient, Cz,t
    dCzlef = Czlef - temp[2];
    FM[2] = Cz + dCzlef*(1-lef/25) + dCzsb*(sb/60) + ((cBar*q)/(2*V))*(Czq + dCzqlef*(1 - lef/25)); // (double check) see appendix B of NASA paper for this Cz,t equation 
    cout<<"Cz = "<<FM[2]<<endl;

    // Here is the equation for total rolling moment coefficient, Cl,t
    dCllef = Cllef - temp[3];
    dClda = Clda - temp[3];
    dCldalef = Cldalef - Cllef - (Clda - temp[3]);
    dCldr = Cldr - temp[3];
    FM[3] = Cl + dCllef*(1-lef/25) + (dClda + dCldalef*(1-lef/25))*(da/20) + dCldr*(dr/30) + (b/(2*V))*((Clr+dClrlef*(1-lef/25))*r+(Clp+dClplef*(1-lef/25))*p) + dClb*beta; // (double check) see appendix B of NASA paper for this Cl,t equation 
    cout<<"Cl = "<<FM[3]<<endl;

    // Here is the equation for total pitching moment coefficient, Cm,t
    dCmlef = Cmlef - temp[4];
    FM[4] = Cm*eta_dh + FM[2]*(xcgShift) + dCmlef*(1-lef/25) + dCmsb*(sb/60) + ((cBar*q)/(2*V))*(Cmq+dCmqlef*(1-lef/25)) + dCm + dCmds; // (double check) see appendix B of NASA paper for this Cm,t equation
    cout<<"Cm = "<<FM[4]<<endl;

    // Here is the equation for total yawing moment coefficient, Cn,t
    dCnlef = Cnlef - temp[5];
    dCnda = Cnda - temp[5]; // Question... There is data for this in the tables, but is defined here also. Why? 
    dCndalef = Cndalef - Cnlef - (Cnda - temp[5]);
    dCndr = Cndr - temp[5];
    FM[5] = Cn + dCnlef*(1 - lef/25) - FM[1]*(xcgShift)*(cBar/b) + (dCnda + dCndalef*(1-lef/25))*(da/20) + dCndr*(dr/30) +(b/(2*V))*((Cnr + dCnrlef*(1-lef/25))*r+(Cnp+dCnplef*(1-lef/25))*p) + dCnb*beta; // (double check) see appendix B of NASA paper for this Cn,t equation
    cout<<"Cn = "<<FM[5]<<endl;

    // Here is the equation for total drag coefficient CD (see equations 4.23 and 4.24 in Christian's dissertation)
    FM[6] = -FM[0]*cos(alpha*PI/180)*cos(beta*PI/180) - FM[1]*sin(beta*PI/180) - FM[2]*sin(alpha*PI/180)*cos(beta*PI/180); 
    cout<<FM[6]<<endl;

    // Here is the equation for total lift coefficient CL (see equations 4.23 and 4.24 in Christian's dissertation)
    FM[7] = -FM[2]*cos(alpha*PI/180) + FM[0]*sin(alpha*PI/180);
    cout<<FM[7]<<endl;

    // Here is the new equation for Cx
    // These go into new tables
    Cxo = Cx + Cxlef - temp[0];
    dCxlefnew = (temp[0] - Cxlef)/lef_from_table;
    dCxsbnew = dCxsb/sb_from_table;
    dCxqnew = Cxq + dCxqlef;
    dCxqlefnew = dCxqlef/lef_from_table;
    // end of new tables
    DCxq = dCxqnew + dCxqlefnew*lef;
    FM[8] = Cxo + dCxlefnew*lef + dCxsbnew*sb + DCxq*q;
    // cout<<"Cxnew = "<<FM[8]<<endl;

    // Here is the new equation for Cy
    // Below go into new tables
    Cyo = Cy + Cylef - temp[1];
    dCylefnew = (temp[1] - Cylef)/lef_from_table;
    dCydanew = Cyda - Cy + Cydalef - Cylef - (Cyda - temp[1]);
    dCydalefnew = (Cylef - Cydalef + (Cyda - temp[1]))/(da_from_table*lef_from_table);
    // DCyda not in new tables
    DCyda = dCydanew + dCydalefnew*lef;
    // Below go into new tables
    dCydrnew = (Cydr - Cy)/dr_from_table;
    dCyrnew = Cyr + dCyrlef;
    dCyrlefnew = dCyrlef/lef_from_table;
    // DCyr not in new tables
    DCyr = dCydrnew - dCyrlefnew*lef;
    // Below go into new tables
    dCypnew = Cyp + dCyplef;
    dCyplefnew = dCyplef/lef_from_table;
    // end of new tables 
    DCyp = dCypnew - dCyplefnew*lef;
    FM[9] = Cyo + dCylefnew*lef + DCyda*da + dCydrnew*dr + DCyr*r + DCyp*p;
    // cout<<"Cynew = "<<FM[9]<<endl;

    // Here is the new equation for Cx
    // These go into new tables
    Czo = Cz + Czlef - temp[2];
    dCzlefnew = (temp[0] - Czlef)/lef_from_table;
    dCzsbnew = dCzsb/sb_from_table;
    dCzqnew = Czq + dCzqlef;
    dCzqlefnew = dCzqlef/lef_from_table;
    // end of new tables
    DCzq = dCzqnew + dCzqlefnew*lef;
    FM[10] = Czo + dCzlefnew*lef + dCzsbnew*sb + DCzq*q;
    // cout<<"Cznew = "<<FM[10]<<endl;





    // cout<<" new Cyo = "<< Cyo  << " old Cyo" << Cy + dCylef << " diff " << Cyo - (Cy + dCylef)  <<endl;


}