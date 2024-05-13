// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

// #include "CoreMinimal.h"
// #include "GameFramework/Pawn.h"
#include "Eigen/Dense"
#include "Aircraft.generated.h"


UCLASS()
class FLIGHTSIMUE4_API AAircraft : public APawn
{
	GENERATED_BODY()

	static constexpr double Pi = 3.1415926535897932384626433832795;
	static constexpr double RE_ft = 20888146.3254593;

public:
	// Sets default values for this pawn's properties
	AAircraft();

	// Called every frame
	virtual void Tick(float DeltaTime) override;

	// Called to bind functionality to input
	virtual void SetupPlayerInputComponent(class UInputComponent* PlayerInputComponent) override;

	// Used to switch between F16 and BIRE aircraft at runtime
	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	bool bIsBireAircraft = false;

	// Used to clamp control surface deflections in blueprints
	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	float daMax = 1.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	float deMax = 1.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	float drMax = 1.0f;


protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

	// BEGIN FlightSim Functions
	/** Initializes variables and calculates initial state based on JSON file 
	* @param ConfigFileName = .JSON file path
	*/
	UFUNCTION(BlueprintCallable, Category = "FlightSim")
	void InitializeAircraftFromJSON(FString ConfigFileName);
	/** Calculates atmospheric properties (SI units) based in an input geometric altitude, H (m) 
	* @param H = Altitude (m)
	* @return [Z, T, p, rho, a] = [geopotential altitude (m), temperature (K), pressure (N/m^2), density (kg/m^3), speed of sound (m/s)]
	*/
	TArray<double> CalculateStdAtmProperties_SI(double H);

	/** Wrapper for CalculateStdAtmProperties_SI that calculates atmospheric properties (english units) based in an input geometric altitude, H (ft)
	* @param H = Altitude (ft)
	* @return [Z, T, p, rho, a] = [geopotential altitude (ft), temperature (R), pressure (lbf/ft^2), density (slugs/ft^3), speed of sound (ft/s)]
	* @see CalculateStdAtmProperties_SI()
	*/
	TArray<double> CalculateStdAtmProperties_English(double H);

	/** Uses aircraft equations of motion to calculate the change in aircraft states
	* IntegrateStates_RK4() uses this function to integrate AircraftStates forward in time
	* @param t = current time (s)
	* @param States = current AircraftStates member var
	* @param Controls = current AircraftControls member var
	* @return Change in state formatted like AircraftStates member var
	* @see IntegrateStates_RK4()
	*/
	Eigen::Matrix<double, 13, 1> CalculateStatesChange(double t, Eigen::Matrix<double, 13, 1> States, Eigen::Vector4d Controls);

	/** Integrates AircraftStates forward in time using the Runge-Kutta 4 integration method
	* @param t0 = current time (s)
	* @param States = current AircraftStates member var to be passed into CalculateStatesChange()
	* @param Controls = current AircraftControls member var to be passed into CalculateStatesChange()
	* @param dt = delta time, or time interval of integration (s)
	* @return New AircraftStates var after integration
	* @see CalculateStatesChange()
	*/
	Eigen::Matrix<double, 13, 1> IntegrateStates_RK4(double t0, Eigen::Matrix<double, 13, 1> States, Eigen::Vector4d Controls, double dt);

	/** Calculates pseudo aerodynamic forces and moments given the aircraft's current states/controls
	* @param States = current AircraftStates member var
	* @param Controls = current AircraftControls member var
	* @return [Fx, Fy, Fz, Mx, My, Mz] = pseudo aerodynamic forces [Fx, Fy, Fz] and moments [Mx, My, Mz] in body fixed frame (x-forward, y-right, z-down)
	*/
	TArray<double> CalculateAeroForces(Eigen::Matrix<double, 13, 1> States, Eigen::Vector4d Controls);

	/** Calculates residual used in the newton's method step of InitializeFromTrim()
	* @param G = trim solution [alpha, beta, da, de, dr, tau] = [angle of attack (rad), sideslip angle (rad), [AircraftControls]]
	* @param pqr = rotation rates [p, q, r] (rad/s)
	* @return Size-6 TArray of residuals based on modified force and moment equations, arranged like [RFx, RFy, RFz, RMx, RMy, RMz]
	* @see InitializeFromTrim()
	*/
	Eigen::Matrix<double, 6, 1> CalculateResidual(Eigen::Matrix<double, 6, 1> G, Eigen::Vector3d pqr);

	/** Calculates initial states/controls based on trim settings in JSON config file and Jacobian trim algorithm
	* Used in InitializeAircraftFromJSON() 
	* @see InitializeAircraftFromJSON() 
	*/
	void InitializeFromTrimJacobian();

	/** Calculates initial states/controls based on trim settings in JSON config file and fixed-point iteration trim algorithm
	* Used in InitializeAircraftFromJSON() 
	* @see InitializeAircraftFromJSON() 
	*/
	void InitializeFromTrimFixedPoint();

	/** Calculates initial states/controls based on state settings in JSON config file
	* Used in InitializeAircraftFromJSON()
	* @see InitializeAircraftFromJSON() 
	*/
	void InitializeFromState();

	/** Gets AircraftStates as floats for use in Blueprints
	* Converts u, v, w and xf, yf, zf to cm for use in UE
	* @return AircraftStates = (u, v, w,   p, q, r,   xf, yf, zf,   e0, ex, ey, ez)
	*/
	UFUNCTION(BlueprintCallable, Category = "FlightSim")
	void GetAircraftStatesUE(float &u, float &v, float &w, float &p, float &q, float &r, float &xf, float &yf, float &zf, float &e0, float &ex, float &ey, float &ez);

	/** Gets AircraftControls as floats for use in Blueprints
	* @return AircraftControls = (da, de, dr, tau)
	*/
	UFUNCTION(BlueprintCallable, Category = "FlightSim")
	void GetAircraftControls(float &da, float &de, float &dr, float &tau);

	/** Sets AircraftControls from floats for use in Blueprints
	* @param AircraftControls = (da, de, dr, tau)
	*/
	UFUNCTION(BlueprintCallable, Category = "FlightSim")
	void SetAircraftControls(float da, float de, float dr, float tau);

	/** Sets which aircraft is being used for dynamics from bool for use in Blueprints
	* @param IsBireAircraft = true for BIRE, false for F16
	*/
	UFUNCTION(BlueprintCallable, Category = "FlightSim")
	void SetIsBireAircraft(bool IsBireAircraft);

	/** Gets latitude and longitude [deg] as floats for use in Blueprints */
	UFUNCTION(BlueprintCallable, Category = "FlightSim")
	void GetLatitudeLongitudeDeg(float &Latitude, float &Longitude, float&alt);

	UFUNCTION(BlueprintCallable, Category = "FlightSim")
	void LLAtoECEF(double &x_trans, double &y_trans, double &z_trans, float lat, float lon, float alt);

	/** Gets ground-fixed orientation euler angles [deg] as floats for use in Blueprints */
	UFUNCTION(BlueprintCallable, Category = "FlightSim")
	void GetEulerAngles(float& BankAngle, float& ElevationAngle, float& AzimuthAngle);

	/** Gets ground-fixed velocity vector components as floats for use in Blueprints */
	UFUNCTION(BlueprintCallable, Category = "FlightSim")
	void GetGlobalVelocityVector(float& u, float& v, float& w);

	/** Gets mach number as float for use in Blueprints */
	UFUNCTION(BlueprintCallable, Category = "FlightSim")
	void GetMachNumber(float& MachNumber);

	/** Gets load factor (G-force) magnitude as float for use in Blueprints */
	/*UFUNCTION(BlueprintCallable, Category = "FlightSim")
	void GetLoadFactorMagnitude(float& LoadFactorMagnitude);*/

	/** Integrate aircraft states forward in time
	* Should be called each tick
	* @param DeltaTime = DeltaTime from Tick() function
	*/
	UFUNCTION(BlueprintCallable, Category = "FlightSim")
	void TickAircraftStates(float DeltaTime);

	/** Integrate aircraft latitude and longitude forward in time
	* Should be called each tick
	* @param dx, dy, dz = change in xf, yf, zf after a single tick's worth of integration
	* @param H1 = altitude before states integration
	*/
	void TickLatitudeLongitude(float dx, float dy, float dz, float H1);
	
	/** Checks if aircraft is stalled
	* @return True if aircraft is stalled
	*/
	UFUNCTION(BlueprintCallable, Category = "FlightSim")
	bool IsAircraftStalled();

	/** Calculates a sine-fit coefficient for BIRE aircraft
	* @param BireCoeffArray = array of sine wave parameters used to calculate specified BIRE aero coefficient
	* @param dB = BIRE deflection [rad]
	* @return BIRE coefficient value
	*/
	double FitBIRECoefficient(TArray<double> BireCoeffArray, double dB);
	
	/** Calculates CL,CS,CD,Cl,Cm,Cn for BIRE aircraft using BCL0, BCLa, etc coefficients
	* @param alpha = angle of attack [rad]
	* @param beta = sideslip angle [rad]
	* @param pbar, qbar, rbar = rotation rates [rad/s]
	* @param da = aileron deflection [rad]
	* @param de = elevator deflection [rad]
	* @param dB = BIRE deflection [rad]
	* @return Array of BIRE aero coefficients, [CL,CS,CD,Cl,Cm,Cn]
	*/
	TArray<double> CalculateBIRECoefficients(double alpha, double beta, double pbar, double qbar, double rbar, double da, double de, double dB);
	
	/** Calculates the compressibility-corrected version of the input aerodynamic coefficient, Coeff, using a modified Prandtl-Glauert correction
	* using a modified
	* @param Coeff = aero coefficient to be corrected for compressibility effects
	* @param Lambda = half-chord sweep angle of lifting surface related to Coeff
	* @param AspectRatio = aspect ratio of lifting surface related to Coeff
	* @param MachNum = current mach number
	* @return Compressibility-corrected version of Coeff for use in CalculateAeroForces()
	*/
	double CalculateCompressibilityCorrection(double Coeff, double Lambda, double AspectRatio, double MachNum);
	
	/** Calculates earth-fixed gust velocity and acceleration using randomized damped sinusoidal model
	* @param t = current simulation time
	* @return <Vgx, Vgy, Vgz, Vgxdot, Vgydot, Vgzdot> where Vg = gust velocity, Vgdot = gust acceleration
	*/
	void UpdateGustState(double t);

	/** Gets GustVelocity as floats for use in Blueprints
	* @param GustVelocity = (Vgx, Vgy, Vgz)
	*/
	UFUNCTION(BlueprintCallable, Category = "FlightSim")
	void GetGustVelocity(float& Vgx, float& Vgy, float& Vgz);
	
	/** Gets the calibrated airspeed for use in Blueprints
	* @param CAS = Calibrated Airspeed [ft/s]
	*/
	UFUNCTION(BlueprintCallable, Category = "FlightSim")
	void GetCalibratedAirspeed(float& CAS);

	/** Logs the input state variable. */
	void LogStates(Eigen::Matrix<double, 13, 1> state, FString name);

	/** Updates current BIRE inertia values, Ixx to Izz, based on current BIRE angle */
	void UpdateBIREInertia();
	// END FlightSim Functions

	
	// BEGIN Quaternion Functions
	/** Multiplies quaternions qA and qB together */
	TArray<double> QuatMult(TArray<double> qA, TArray<double> qB);

	/** Converts Euler = [roll, pitch, yaw] (rad) to a quaternion */
	TArray<double> EulerToQuat(TArray<double> Euler);

	/** Converts quaternion Quat to euler angles = [roll, pitch, yaw] (rad) */
	TArray<double> QuatToEuler(TArray<double> Quat);

	/** Converts size-3 body fixed vector, BodyFixed, to earth fixed vector using orientation quaterion, Quat */
	TArray<double> BodyToEarthFixed(TArray<double> BodyFixed, TArray<double> Quat);

	/** Converts size-3 earth fixed vector, EarthFixed, to body fixed vector using orientation quaterion, Quat */
	TArray<double> EarthToBodyFixed(TArray<double> EarthFixed, TArray<double> Quat);

	/** Normalizes quaternion, Quat, to have magnitude 1 */
	TArray<double> QuatNormalize(TArray<double> Quat);
	// END Quaternion Functions


	// BEGIN Stability Functions
	/** Calculates stability mode properties and saves them to file. 
	* For each eigenvalue, the following mode properties are saved to file (if applicable to mode): 
	*	eigenvector
	*	damping rate, sigma
	*	frequency, omega_d
	*	damping ratio, zeta
	*	undamped natural frequency, omega_n
	*	99% damping time, timeTo99Damp
	*	doubling time, timeToDouble
	*/
	UFUNCTION(BlueprintCallable, Category = "FlightSim")
	void OutputStabilityModeProperties();

	/** Solves the stability eigensystem and returns the EigenSolver class that holds the eigenvlaues and eigenvectors
	* @return eigensystemSolution = Eigen::EigenSolver<Eigen::Matrix(double, 12, 12)> class that holds the eigenvalues and eigenvectors
	*/
	Eigen::EigenSolver<Eigen::Matrix<double, 12, 12>> SolveStabilityEigensystem();

	/** Converts a body frame vector to a wind frame vector using the aircraft's current alpha and beta
	* @param bodyVector = vector in body frame
	* @return windVector = vector in wind frame
	*/
	Eigen::Vector3<double> BodyToWindFrame(Eigen::Vector3<double> bodyVector);

	/** Converts a wind frame vector to a body frame vector using the aircraft's current alpha and beta
	* @param windVector = vector in wind frame
	* @return bodyVector = vector in body frame
	*/
	Eigen::Vector3<double> WindToBodyFrame(Eigen::Vector3<double> windVector);

	/** Uses finite differencing to calculate dimensional force and moment coefficients necessary to solve stability eigensystem 
	* @param windDeltas = array of state deltas in wind frame used to approximate the force and moment derivatives. windDeltas = [du, dv, dw, dp, dq, dr]
	*/
	void CalculateForceMomentDerivatives(TArray<double> windDeltas);
	// END Stability Functions


	// BEGIN Landing Functions
	/** Calculates total landing gear forces and moments using a spring-damper system for each of 3 landing gear 
	@param dzFront = front landing gear spring compression (ft) based on blueprint line trace
	@param dzRight and dzLeft are the same as dzFront, but for the right and left landing gear, respectively
	@param dt = time step [sec] for rate calculations
	@param CollisionNormal = normal vector of ground plane from line trace
	*/
	UFUNCTION(BlueprintCallable, Category = "FlightSim")
	void UpdateLandingForces(float dzFront, float dzRight, float dzLeft, float dt, FVector CollisionNormal);
	// END Landing Functions



private:
	// BEGIN Json Variables
	// Json Variables = Aerodynamic variables prescribed in configuration .json file (ConfigFileName from InitializeAircraftFromJSON())
	// See InitializeAircraftFromJSON() for engineering abreviation definitions
	bool bConstantDensityAtmosphere;
	double windMagn;
	double windDir;
	Eigen::Vector3d windVect;
	// Mass/geometry properties
	double Sw;
	double bw;
	double W;
	double IxxInit;
	double IyyInit;
	double IzzInit;
	double IxyInit;
	double IxzInit;
	double IyzInit;
	double dIBIRE;
	double hx;
	double hy;
	double hz;
	Eigen::Vector3d CGShift;
	// Thrust properties
	Eigen::Vector3d ThrustLoc;
	Eigen::Vector3d ThrustDir;
	double ThrustT0;
	double ThrustT1;
	double ThrustT2;
	double ThrustA;
	// Landing properties
	Eigen::Vector3d frontGearLoc;
	double frontGearLength;
	Eigen::Vector3d frontGearVec;
	Eigen::Vector3d rightGearLoc;
	double rightGearLength;
	Eigen::Vector3d rightGearVec;
	Eigen::Vector3d leftGearLoc;
	double leftGearLength;
	Eigen::Vector3d leftGearVec;
	Eigen::Vector3d kArray; // spring constant of [front, right, left] landing gear
	Eigen::Vector3d cArray; // damping constant of [front, right, left] landing gear
	// Initial conditions
	double VInit;
	double altInit;
	double headingInit;
	double groundTrackInit;
	FString InitType;
	double elevInit;
	double bankInit;
	double alphaInit;
	double betaInit;
	double pInit;
	double qInit;
	double rInit;
	double daInit;
	double deInit;
	double drInit;
	double tauInit;
	FString TrimType;
	double climbInit;
	double sideslipInit;
	double trimStepSize;
	double trimRelaxFactor;
	double trimTolerance;
	// Aerodynamic coefficients
	double gustMagnitude;
	Eigen::Vector3d gustScales;
	bool bUseStallModel;
	double StallAlphaB;
	double StallM;
	double CL0;
	double CLa;
	double CLqbar;
	double CLde;
	double CSb;
	double CSpbar;
	double CSLpbar;
	double CSrbar;
	double CSda;
	double CSdr;
	double CD0;
	double CDL;
	double CDL2;
	double CDS2;
	double CDpbar;
	double CDSpbar;
	double CDqbar;
	double CDLqbar;
	double CDL2qbar;
	double CDrbar;
	double CDSrbar;
	double CDde;
	double CDLde;
	double CDde2;
	double CDda;
	double CDSda;
	double CDdr;
	double CDSdr;
	double Clb;
	double Clpbar;
	double Clrbar;
	double ClLrbar;
	double Clda;
	double Cldr;
	double Cm0;
	double Cma;
	double Cmqbar;
	double Cmde;
	double Cnb;
	double Cnpbar;
	double CnLpbar;
	double Cnrbar;
	double Cnda;
	double CnLda;
	double Cndr;
	// BIRE coefficient arrays = [A, w, phi, z, delta, multiplier] for each BIRE aero coefficient
	// 	   i.e. BCL0 = [BIRE CL0's amplitude (A), BIRE CL0's frequency (w), etc.]
	TArray<double> BCL0;
	TArray<double> BCLa;
	TArray<double> BCLb;
	TArray<double> BCLpbar;
	TArray<double> BCLqbar;
	TArray<double> BCLrbar;
	TArray<double> BCLda;
	TArray<double> BCLde;
	
	TArray<double> BCS0;
	TArray<double> BCSa;
	TArray<double> BCSb;
	TArray<double> BCSpbar;
	TArray<double> BCSLpbar;
	TArray<double> BCSqbar;
	TArray<double> BCSrbar;
	TArray<double> BCSda;
	TArray<double> BCSde;

	TArray<double> BCD0;
	TArray<double> BCDL;
	TArray<double> BCDL2;
	TArray<double> BCDS;
	TArray<double> BCDS2;
	TArray<double> BCDpbar;
	TArray<double> BCDSpbar;
	TArray<double> BCDqbar;
	TArray<double> BCDLqbar;
	TArray<double> BCDL2qbar;
	TArray<double> BCDrbar;
	TArray<double> BCDSrbar;
	TArray<double> BCDda;
	TArray<double> BCDSda;
	TArray<double> BCDde;
	TArray<double> BCDLde;
	TArray<double> BCDde2;

	TArray<double> BCl0;
	TArray<double> BCla;
	TArray<double> BClb;
	TArray<double> BClpbar;
	TArray<double> BClqbar;
	TArray<double> BClrbar;
	TArray<double> BClLrbar;
	TArray<double> BClda;
	TArray<double> BClde;

	TArray<double> BCm0;
	TArray<double> BCma;
	TArray<double> BCmb;
	TArray<double> BCmpbar;
	TArray<double> BCmqbar;
	TArray<double> BCmrbar;
	TArray<double> BCmda;
	TArray<double> BCmde;

	TArray<double> BCn0;
	TArray<double> BCna;
	TArray<double> BCnb;
	TArray<double> BCnpbar;
	TArray<double> BCnLpbar;
	TArray<double> BCnqbar;
	TArray<double> BCnrbar;
	TArray<double> BCnda;
	TArray<double> BCnLda;
	TArray<double> BCnde;
	// END Json Variables
	

	// BEGIN Precomputed Variables
	// Precomputed Variables = Global variables precomputed in InitializeAircraftFromJSON() for computation efficiency
	// cw = Mean wing chord
	double cw;
	// rho0 = Initial air density
	double rho0;
	// WInv = 1 / W
	double WInv;
	// gInit = g @ altInit
	double gInit;
	// Wg = W/g
	double Wg;
	// Ixx to Izz = Current inertia values
	double Ixx;
	double Iyy;
	double Izz;
	double Ixy;
	double Ixz;
	double Iyz;
	// IInv = 3x3 Matrix inverse of aircraft inertia tensor (constructed from Ixx, Ixy, Iyy, etc.)
	Eigen::Matrix3d IInv;
	// hArray = 3x3 Angular momentum tensor (constructed from hx, hy, hz)
	Eigen::Matrix3d hArray;
	// Whether elevation angle is supplied in JSON config
	bool bElevProvided;
	// Whether bank angle is supplied in JSON config
	bool bBankProvided;
	// s/c = sin()/cos(), b = beta, phi = bank angle, theta = elevation angle, gamma = climb angle (rad)
	double sbInit;
	double cbInit;
	double sphiInit;
	double cphiInit;
	double sthetaInit;
	double cthetaInit;
	double sgammaInit;
	double cgammaInit;
	// END Precomputed Variables


	// BEGIN Stability Variables
	// Fxyz_uvw = velocity force derivatives = [[Fxu, Fxv, Fxw],
	//											[Fyu, Fyv, Fyw],
	//											[Fzu, Fzv, Fzw]]
	Eigen::Matrix3d Fxyz_uvw;
	// Mxyz_uvw = velocity moment derivatives = [[Mxu, Mxv, Mxw],
	//											 [Myu, Myv, Myw],
	//											 [Mzu, Mzv, Mzw]]
	Eigen::Matrix3d Mxyz_uvw;
	// Fxyz_pqr = rate force derivatives = [[Fxp, Fxq, Fxr],
	//										[Fyp, Fyq, Fyr],
	//										[Fzp, Fzq, Fzr]]
	Eigen::Matrix3d Fxyz_pqr;
	// Mxyz_pqr = rate moment derivatives = [[Mxp, Mxq, Mxr],
	//										 [Myp, Myq, Myr],
	//										 [Mzp, Mzq, Mzr]]
	Eigen::Matrix3d Mxyz_pqr;
	double Fz_wdot;
	double My_wdot;
	// END Stability Variables


	// BEGIN Runtime Variables
	// Runtime Variables = Global variables storing aircraft states/conditions during runtime
	// AircraftStates = [u, v, w,   p, q, r,   xf, yf, zf,   e0, ex, ey, ez]
	//		u, v, w = x, y, z velocity components in body fixed coordinate frame (x-forward, y-right, z-down) (ft/s)
	//		p, q, r = rotation rates about x, y, z axis (body fixed frame) (rad/s)
	//		xf, yf, zf = global position (body fixed frame) (ft)
	//		e0, ex, ey, ez = quaternion orientation
	Eigen::Matrix<double, 13, 1> AircraftStates;
	// AircraftControls = [da, de, dr, tau]
	// 	   da = aileron deflection (rad)
	// 	   de = elevator deflection (rad)
	// 	   dr = rudder deflection (rad)
	// 	   tau = throttle setting (0 < tau < 1)
	Eigen::Vector4d AircraftControls;
	// Latitude/Longitude variables (spherical-earth approximation)
	// These are updated during runtime but are initialized from JSON
	double latitude;
	double longitude;
	//double x_trans;
	//double y_trans;
	//double z_trans;
	// Landing forces and moments
	// LandingForces = [Fx, Fy, Fz, Mx, My, Mz] in body fixed coordinates (lbf, lbf-ft)
	TArray<double> LandingForces;
	// Current and previous gear deflections are used to calculate rate for landing gear damper force
	Eigen::Vector3d CurrentGearDeflections;
	Eigen::Vector3d PreviousGearDeflections;
	// Gust model global state variables
	Eigen::Vector3d GustVelocity;
	Eigen::Vector3d GustAmplitude;
	Eigen::Vector3d GustOmega;
	Eigen::Vector3d GustLambda;
	Eigen::Vector3d GustDelay;
	Eigen::Vector3d GustPreviousStartTimes;
	Eigen::Vector3d GustStartTimes;
	Eigen::Vector3d GustAcceleration;
	// END Runtime Variables

};
