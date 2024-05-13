#ifndef AERO_H
#define AERO_H

#include "atmosphere.h"

class Aero
{
	
public:
    Aero(json dictionary);
    ~Aero(){};
    
    void get_FM_coefficients(double inputVariables[5], double ans[6]);

private:
	//Private Variables
    vector<string> mvDatasetName;   
    vector<Dataset*> mvDataset;   
};

#endif


