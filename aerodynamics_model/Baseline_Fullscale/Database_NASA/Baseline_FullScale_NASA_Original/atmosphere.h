#ifndef ATM_H
#define ATM_H

#include "dataset.h"

class Atmosphere
{
	
public:
    Atmosphere(){};
    ~Atmosphere(){};
    
    double get_gravity_si(double altitude);
    double get_gravity_english(double altitude);
    void set_atmospheric_properties_si(double altitude);
    void set_atmospheric_properties_english(double altitude);
    void print_tables();
    double get_density();
    double get_viscosity();


private:
	//Private Variables
    double mGeopotentialAltitude;
    double mTemperature;
    double mPressure;
    double mDensity;
    double mSpeedOfSound;
    double mViscosity;
};

#endif


