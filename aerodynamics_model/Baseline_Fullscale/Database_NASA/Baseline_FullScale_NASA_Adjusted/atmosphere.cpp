#include "atmosphere.h"

double Atmosphere::get_gravity_si(double altitude)
{
   return 9.80665*pow((6356766./(6356766. + altitude)),2);
}

double Atmosphere::get_gravity_english(double altitude)
{
   return 9.80665*pow((6356766./(6356766. + altitude*0.3048)),2)/0.3048;
}

void Atmosphere::set_atmospheric_properties_si(double altitude)
{
	double levels[9] = {0.0, 11000.0, 20000.0, 32000.0, 47000.0, 52000.0, 61000.0, 79000.0, 90000.0};
	double temps[9] = {288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 252.65, 180.65, 180.65};
	double tprime[9] = {-6.5, 0.0, 1.0, 2.8, 0.0, -2.0, -4.0, 0.0};

    int i;
    double Z,T,p,rho,a;

    for (i = 0; i < 8; i++) {
        tprime[i] /= 1000.0;
    }
    double go = 9.80665;
    double R = 287.052874;
    double RE = 6356766.0;
    double po = 101325.0;
    double gamma = 1.4;
   
    Z = RE*altitude/(RE + altitude);

    Z = max(Z,0.0);

    if(Z > 90000) {
        T = 180.650;
        p = 0.0;
    }
    else{
        i = 0;
        while (Z >= levels[i]){
            if(tprime[i] == 0){
                if(Z < levels[i+1]){
                    T = temps[i];
                    p = po*exp(-go*(Z-levels[i])/R/temps[i]);
                }
                else{
                    po = po*exp(-go*(levels[i+1]-levels[i])/R/temps[i]);
                }
            }
            else{
                if(Z < levels[i+1]){
                    T = temps[i] + tprime[i]*(Z-levels[i]);
                    p = po*pow(((temps[i] + tprime[i]*(Z-levels[i]))/temps[i]),(-go/R/tprime[i]));
                }
                else{
                    po = po*pow(((temps[i] + tprime[i]*(levels[i+1]-levels[i]))/temps[i]),(-go/R/tprime[i]));
                }
            }
            i = i + 1;
        }
    }
    rho = p/R/T;
    a = sqrt(gamma*R*T);

    //Dynamic Viscosity https://www.cfd-online.com/Wiki/Sutherland%27s_law
    double T0 = 273.15; //Kelvin
    double mu0 = 0.00001716; //kg/m-s
    double C = 110.4; //Sutherland's Constant for air
    double mu = mu0*(T0+C)/(T+C)*pow((T/T0),1.5);

    mGeopotentialAltitude = Z;
    mTemperature = T;
    mPressure = p;
    mDensity = rho;
    mSpeedOfSound = a;
    mViscosity = mu;
}

void Atmosphere::set_atmospheric_properties_english(double altitude)
{
    altitude *= 0.3048;
    set_atmospheric_properties_si(altitude);
    mGeopotentialAltitude /= 0.3048;
    mTemperature *= 1.8;
    mPressure *= 0.0208854342331501; //0.020885434304801722;
    mDensity *= 0.00194032033197972; //0.00194032032363104;
    mSpeedOfSound /= 0.3048;
    mViscosity *=0.0208854342331501; // was divide by 47.88025898; //slugs/(ft-s)
}

void Atmosphere::print_tables()
{
    double H; // Altitude
    double Z; // Geopotential Altitude
    double RE = 6356766.0; // Radius of earth

    FILE *outputFile = fopen("stdatmos_si.txt", "w");
    fprintf(outputFile,"     Geometric_Altitude[m]   Geopotential_Altitude[m]   Temperature[K]          Pressure[N/m^2]         Density[kg/m^3]         Speed_of_Sound[m/s] Dynamic_Viscosity[kg/(m-s)]\n");
    for (int i = 0; i < 105000; i+= 5000)
    {
        H = double(i);
        set_atmospheric_properties_si(H);
        fprintf(outputFile,"%20.12e%20.12e%20.12e%20.12e%20.12e%20.12e%20.12e\n",H,mGeopotentialAltitude,mTemperature,mPressure,mDensity,mSpeedOfSound,mViscosity);
    }
    fclose(outputFile);

    // Print properties at level breaks
    outputFile = fopen("stdatmos_si_levels.txt", "w");
    fprintf(outputFile,"     Geometric_Altitude[m]   Geopotential_Altitude[m]   Temperature[K]          Pressure[N/m^2]\n");

    Z = 0.0;
    H = RE*Z/(RE-Z);
    set_atmospheric_properties_si(H);
    fprintf(outputFile,"%20.12e%20.12e%20.12e%22.14e\n",H,mGeopotentialAltitude,mTemperature,mPressure);

    Z = 11000.0;
    H = RE*Z/(RE-Z);
    set_atmospheric_properties_si(H);
    fprintf(outputFile,"%20.12e%20.12e%20.12e%22.14e\n",H,mGeopotentialAltitude,mTemperature,mPressure);

    Z = 20000.0;
    H = RE*Z/(RE-Z);
    set_atmospheric_properties_si(H);
    fprintf(outputFile,"%20.12e%20.12e%20.12e%22.14e\n",H,mGeopotentialAltitude,mTemperature,mPressure);

    Z = 32000.0;
    H = RE*Z/(RE-Z);
    set_atmospheric_properties_si(H);
    fprintf(outputFile,"%20.12e%20.12e%20.12e%22.14e\n",H,mGeopotentialAltitude,mTemperature,mPressure);

    Z = 47000.0;
    H = RE*Z/(RE-Z);
    set_atmospheric_properties_si(H);
    fprintf(outputFile,"%20.12e%20.12e%20.12e%22.14e\n",H,mGeopotentialAltitude,mTemperature,mPressure);

    Z = 52000.0;
    H = RE*Z/(RE-Z);
    set_atmospheric_properties_si(H);
    fprintf(outputFile,"%20.12e%20.12e%20.12e%22.14e\n",H,mGeopotentialAltitude,mTemperature,mPressure);

    Z = 61000.0;
    H = RE*Z/(RE-Z);
    set_atmospheric_properties_si(H);
    fprintf(outputFile,"%20.12e%20.12e%20.12e%22.14e\n",H,mGeopotentialAltitude,mTemperature,mPressure);

    Z = 79000.0;
    H = RE*Z/(RE-Z);
    set_atmospheric_properties_si(H);
    fprintf(outputFile,"%20.12e%20.12e%20.12e%22.14e\n",H,mGeopotentialAltitude,mTemperature,mPressure);

    fclose(outputFile);

    outputFile = fopen("stdatmos_english.txt", "w");
    fprintf(outputFile,"     Geometric_Altitude[ft]   Geopotential_Altitude[ft]   Temperature[R]          Pressure[lbf/ft^2]         Density[slugs/ft^3]         Speed_of_Sound[ft/s] Dynamic_Viscosity[slugs/(ft-s)]\n");
    for (int i = 0; i < 205000; i+= 10000)
    {
        H = double(i);
        set_atmospheric_properties_english(H);
        fprintf(outputFile,"%20.12e%20.12e%20.12e%20.12e%20.12e%20.12e%20.12e\n",H,mGeopotentialAltitude,mTemperature,mPressure,mDensity,mSpeedOfSound,mViscosity);
    }
    fclose(outputFile);
}

double Atmosphere::get_density()
{
    return mDensity;
}

double Atmosphere::get_viscosity()
{
    return mViscosity;
}