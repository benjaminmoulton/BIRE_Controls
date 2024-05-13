// Fill out your copyright notice in the Description page of Project Settings.


#include "Aircraft.h"
#include "Json.h"
#include <algorithm>
#include "Math/UnrealMathUtility.h"
#include <fstream>

#define PRINTSTATE(state) (UE_LOG(LogTemp, Warning, TEXT("{%f\t %f\t %f\t %f\t %f\t %f\t %f\t %f\t %f\t %f\t %f\t %f\t %f\t}"), state(0), state(1), state(2), state(3), state(4), state(5), state(6), state(7), state(8), state(9), state(10), state(11), state(12)))

// Sets default values
AAircraft::AAircraft()
{
 	// Set this pawn to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

}

void AAircraft::InitializeAircraftFromJSON(FString ConfigFileName)
{
	FString ConfigJsonFilePath = FPaths::ProjectContentDir() + "AircraftConfig/" + ConfigFileName;
	
	// Store config JSON in an FString
	FString ConfigJsonString;
	FFileHelper::LoadFileToString(ConfigJsonString, *ConfigJsonFilePath);

	UE_LOG(LogTemp, Warning, TEXT("Config File Path: %s"), *ConfigJsonFilePath);
	//UE_LOG(LogTemp, Warning, TEXT("Config Json String: %s"), *ConfigJsonString);

	TSharedPtr<FJsonObject> ConfigJsonObject = MakeShareable(new FJsonObject());
	TSharedRef<TJsonReader<>> ConfigJsonReader = TJsonReaderFactory<>::Create(ConfigJsonString);

	// Store config JSON values in respective member variables
	if (FJsonSerializer::Deserialize(ConfigJsonReader, ConfigJsonObject) && ConfigJsonObject.IsValid())
	{
		// 'simulation' category
		TSharedPtr<FJsonObject> SimulationJsonObject = ConfigJsonObject->GetObjectField("simulation");
		bConstantDensityAtmosphere = SimulationJsonObject->GetBoolField("constant_density");
		windMagn = SimulationJsonObject->GetNumberField("wind_magnitude[ft/s]");
		windDir = Pi/180.*SimulationJsonObject->GetNumberField("wind_direction[deg]");
		windVect = Eigen::Vector3d( 
			windMagn*cos(windDir),
			windMagn*sin(windDir),
			0.0 
		);

		// 'aircraft' category
		TSharedPtr<FJsonObject> AircraftJsonObject = ConfigJsonObject->GetObjectField("aircraft");
		// Mass/geometry properties
		Sw = AircraftJsonObject->GetNumberField("wing_area[ft^2]");
		bw = AircraftJsonObject->GetNumberField("wing_span[ft]");
		W = AircraftJsonObject->GetNumberField("weight[lbf]");
		Ixx = AircraftJsonObject->GetNumberField("Ixx[slug-ft^2]");
		Iyy = AircraftJsonObject->GetNumberField("Iyy[slug-ft^2]");
		Izz = AircraftJsonObject->GetNumberField("Izz[slug-ft^2]");
		Ixy = AircraftJsonObject->GetNumberField("Ixy[slug-ft^2]");
		Ixz = AircraftJsonObject->GetNumberField("Ixz[slug-ft^2]");
		Iyz = AircraftJsonObject->GetNumberField("Iyz[slug-ft^2]");
		hx = AircraftJsonObject->GetNumberField("hx[slug-ft^2/s]");
		hy = AircraftJsonObject->GetNumberField("hy[slug-ft^2/s]");
		hz = AircraftJsonObject->GetNumberField("hz[slug-ft^2/s]");
		TArray<TSharedPtr<FJsonValue>> CGShiftJsonArray = AircraftJsonObject->GetArrayField("CG_shift[ft]");
		CGShift = Eigen::Vector3d(
			CGShiftJsonArray[0]->AsNumber(), 
			CGShiftJsonArray[1]->AsNumber(), 
			CGShiftJsonArray[2]->AsNumber()
		);
		// 'thrust' category of 'aircraft'
		TSharedPtr<FJsonObject> ThrustJsonObject = AircraftJsonObject->GetObjectField("thrust");
		TArray<TSharedPtr<FJsonValue>> ThrustLocJsonArray = ThrustJsonObject->GetArrayField("location[ft]");
		ThrustLoc = Eigen::Vector3d(
			ThrustLocJsonArray[0]->AsNumber(),
			ThrustLocJsonArray[1]->AsNumber(),
			ThrustLocJsonArray[2]->AsNumber()
		);
		TArray<TSharedPtr<FJsonValue>> ThrustDirJsonArray = ThrustJsonObject->GetArrayField("direction");
		ThrustDir = Eigen::Vector3d(
			ThrustDirJsonArray[0]->AsNumber(),
			ThrustDirJsonArray[1]->AsNumber(),
			ThrustDirJsonArray[2]->AsNumber()
		);
		ThrustT0 = ThrustJsonObject->GetNumberField("T0[lbf]");
		ThrustT1 = ThrustJsonObject->GetNumberField("T1[lbf-s/ft]");
		ThrustT2 = ThrustJsonObject->GetNumberField("T2[lbf-s^2/ft^2]");
		ThrustA = ThrustJsonObject->GetNumberField("a");

		// 'landing' category of 'aircraft'
		TSharedPtr<FJsonObject> LandingJsonObject = AircraftJsonObject->GetObjectField("landing");
		TArray<TSharedPtr<FJsonValue>> FrontGearLocJsonArray = LandingJsonObject->GetArrayField("front_wheel_location[ft]");
		frontGearLoc = Eigen::Vector3d(
			FrontGearLocJsonArray[0]->AsNumber(),
			FrontGearLocJsonArray[1]->AsNumber(),
			FrontGearLocJsonArray[2]->AsNumber()
		);
		frontGearLength = LandingJsonObject->GetNumberField("front_gear_length[ft]");
		TArray<TSharedPtr<FJsonValue>> RightGearLocJsonArray = LandingJsonObject->GetArrayField("right_wheel_location[ft]");
		rightGearLoc = Eigen::Vector3d(
			RightGearLocJsonArray[0]->AsNumber(),
			RightGearLocJsonArray[1]->AsNumber(),
			RightGearLocJsonArray[2]->AsNumber()
		);
		rightGearLength = LandingJsonObject->GetNumberField("right_gear_length[ft]");
		TArray<TSharedPtr<FJsonValue>> LeftGearLocJsonArray = LandingJsonObject->GetArrayField("left_wheel_location[ft]");
		leftGearLoc = Eigen::Vector3d(
			LeftGearLocJsonArray[0]->AsNumber(),
			LeftGearLocJsonArray[1]->AsNumber(),
			LeftGearLocJsonArray[2]->AsNumber()
		);
		leftGearLength = LandingJsonObject->GetNumberField("left_gear_length[ft]");
		TArray<TSharedPtr<FJsonValue>> kArrayJsonArray = LandingJsonObject->GetArrayField("k_array[lbf/ft]");
		kArray = Eigen::Vector3d(
			kArrayJsonArray[0]->AsNumber(),
			kArrayJsonArray[1]->AsNumber(),
			kArrayJsonArray[2]->AsNumber()
		);
		TArray<TSharedPtr<FJsonValue>> cArrayJsonArray = LandingJsonObject->GetArrayField("c_array[lbf-s/ft]");
		kArray = Eigen::Vector3d(
			cArrayJsonArray[0]->AsNumber(),
			cArrayJsonArray[1]->AsNumber(),
			cArrayJsonArray[2]->AsNumber()
		);
		
		// 'initial' category
		TSharedPtr<FJsonObject> InitialJsonObject = ConfigJsonObject->GetObjectField("initial");
		VInit = InitialJsonObject->GetNumberField("airspeed[ft/s]");
		altInit = InitialJsonObject->GetNumberField("altitude[ft]");
		headingInit = Pi/180.*InitialJsonObject->GetNumberField("heading[deg]");
		groundTrackInit = Pi/180.*InitialJsonObject->GetNumberField("ground_track[deg]");
		latitude = Pi/180.*InitialJsonObject->GetNumberField("latitude[deg]");
		longitude = Pi/180.*InitialJsonObject->GetNumberField("longitude[deg]");
		InitType = InitialJsonObject->GetStringField("type");
		if (InitType == "state")
		{
			// 'state' category of 'initial'
			TSharedPtr<FJsonObject> StateJsonObject = InitialJsonObject->GetObjectField("state");
			elevInit = Pi/180.*StateJsonObject->GetNumberField("elevation_angle[deg]");
			bankInit = Pi/180.*StateJsonObject->GetNumberField("bank_angle[deg]");
			alphaInit = Pi/180.*StateJsonObject->GetNumberField("alpha[deg]");
			betaInit = Pi/180.*StateJsonObject->GetNumberField("beta[deg]");
			pInit = Pi/180.*StateJsonObject->GetNumberField("p[deg/s]");
			qInit = Pi/180.*StateJsonObject->GetNumberField("q[deg/s]");
			rInit = Pi/180.*StateJsonObject->GetNumberField("r[deg/s]");
			daInit = Pi/180.*StateJsonObject->GetNumberField("aileron[deg]");
			deInit = Pi/180.*StateJsonObject->GetNumberField("elevator[deg]");
			drInit = Pi/180.*StateJsonObject->GetNumberField("rudder[deg]");
			tauInit = StateJsonObject->GetNumberField("throttle");
		}
		else if (InitType == "trim")
		{
			// 'trim' category of 'initial'
			TSharedPtr<FJsonObject> TrimJsonObject = InitialJsonObject->GetObjectField("trim");
			TrimType = TrimJsonObject->GetStringField("type");
			// Requires elevation or climb angle to trim aircraft. If specified, elevation angle overrides.
			bElevProvided = TrimJsonObject->HasField("elevation_angle[deg]");
			// Using elevation angle
			if (bElevProvided)
			{
				elevInit = Pi/180.*TrimJsonObject->GetNumberField("elevation_angle[deg]");
				sthetaInit = sin(elevInit);
				cthetaInit = cos(elevInit);
			}
			// Using climb angle
			else if (TrimJsonObject->HasField("climb_angle[deg]"))
			{
				climbInit = Pi/180.*TrimJsonObject->GetNumberField("climb_angle[deg]");
				sgammaInit = sin(climbInit);
				cgammaInit = cos(climbInit);
			}
			// ERROR: Trim initialization requires bank or climb angle!
			else
			{
				UE_LOG(LogTemp, Error, TEXT("Missing elevation or climb angle for trim initialization. JSON 'initial/trim/elevation_angle[deg]' or 'initial/trim/climb_angle[deg]' should have a value."));
			}
			// Requires bank angle to trim aircraft for steady coordinated turn.
			// Requires bank or sideslip angle to trim aircraft for steady heading sideslip. If specified, bank angle overrides.
			bBankProvided = TrimJsonObject->HasField("bank_angle[deg]");
			// Either trim type, using bank angle
			if (bBankProvided)
			{
				bankInit = Pi/180.*TrimJsonObject->GetNumberField("bank_angle[deg]");
				sphiInit = sin(bankInit);
				cphiInit = cos(bankInit);
			}
			// ERROR: Steady coordinated turn initialization requires bank angle!
			else if ((TrimType == "sct") && !bBankProvided)
			{
				UE_LOG(LogTemp, Error, TEXT("Missing bank angle for steady coordinated turn trim initialization. JSON 'initial/trim/bank_angle[deg]' should have a value."));
			}
			// Steady heading sideslip, using sideslip angle
			else if (TrimJsonObject->HasField("sideslip_angle[deg]"))
			{
				sideslipInit = Pi/180.*TrimJsonObject->GetNumberField("sideslip_angle[deg]");
				sbInit = sin(sideslipInit);
				cbInit = cos(sideslipInit);
				betaInit = sideslipInit;
			}
			// ERROR: Steady heading sideslip initialization requires bank or sideslip angle!
			else
			{
				UE_LOG(LogTemp, Error, TEXT("Missing bank or sideslip angle for steady heading sideslip trim initialization. JSON 'initial/trim/bank_angle[deg]' or 'initial/trim/sideslip_angle[deg]' should have a value."));
			}
			// 'solver' category of 'trim'
			TSharedPtr<FJsonObject> SolverJsonObject = TrimJsonObject->GetObjectField("solver");
			trimStepSize = SolverJsonObject->GetNumberField("finite_difference_step_size");
			trimRelaxFactor = SolverJsonObject->GetNumberField("relaxation_factor");
			trimTolerance = SolverJsonObject->GetNumberField("tolerance");
		}
		else
		{
			UE_LOG(LogTemp, Error, TEXT("Incorrect aircraft initialization type. JSON 'initial/trim/type' should have value 'trim' or 'state'."));
		}

		// 'aerodynamics' category
		TSharedPtr<FJsonObject> AerodynamicsJsonObject = ConfigJsonObject->GetObjectField("aerodynamics");
		// 'gust' values of 'aerodynamics' for use in gust model
		gustMagnitude = AerodynamicsJsonObject->GetNumberField("gust_magnitude[ft/s]");
		TArray<TSharedPtr<FJsonValue>> GustScalesJsonArray = AerodynamicsJsonObject->GetArrayField("gust_scales");
		gustScales = Eigen::Vector3d(
			GustScalesJsonArray[0]->AsNumber(),
			GustScalesJsonArray[1]->AsNumber(),
			GustScalesJsonArray[2]->AsNumber()
		);
		// 'stall' category of 'aerodynamics' 
		TSharedPtr<FJsonObject> StallJsonObject = AerodynamicsJsonObject->GetObjectField("stall");
		bUseStallModel = StallJsonObject->GetBoolField("use_stall_model");
		StallAlphaB = Pi/180.*StallJsonObject->GetNumberField("alpha_blend[deg]");
		StallM = StallJsonObject->GetNumberField("blending_factor");
		
		// bIsBireAircraft = true if aircraft is BIRE
		//bIsBireAircraft = AerodynamicsJsonObject->GetBoolField("BIRE");
		UE_LOG(LogTemp, Warning, TEXT("BIRE: %s"), ( bIsBireAircraft ? TEXT("true") : TEXT("false") ));
		if (bIsBireAircraft)
		{
			// Override inertia with BIRE version
			W = AircraftJsonObject->GetNumberField("weight_BIRE[lbf]");
			IxxInit = AircraftJsonObject->GetNumberField("Ixx_BIRE[slug-ft^2]");
			IyyInit = AircraftJsonObject->GetNumberField("Iyy_BIRE[slug-ft^2]");
			IzzInit = AircraftJsonObject->GetNumberField("Izz_BIRE[slug-ft^2]");
			IxyInit = AircraftJsonObject->GetNumberField("Ixy_BIRE[slug-ft^2]");
			IxzInit = AircraftJsonObject->GetNumberField("Ixz_BIRE[slug-ft^2]");
			IyzInit = AircraftJsonObject->GetNumberField("Iyz_BIRE[slug-ft^2]");
			dIBIRE = AircraftJsonObject->GetNumberField("dI_BIRE[slug-ft^2]");
			Ixx = IxxInit;
			Iyy = IyyInit;
			Izz = IzzInit;
			Ixy = IxyInit;
			Ixz = IxzInit;
			Iyz = IyzInit;
			
			// Initialize BIRE aero coefficients
			FString BireModelJsonFilePath = FPaths::ProjectContentDir() + "AircraftConfig/" + "BIREModel.json";
	
			// Store config JSON in an FString
			FString BireModelJsonString;
			FFileHelper::LoadFileToString(BireModelJsonString, *BireModelJsonFilePath);

			UE_LOG(LogTemp, Warning, TEXT("BIRE Model File Path: %s"), *BireModelJsonFilePath);
			//UE_LOG(LogTemp, Warning, TEXT("BIRE Model Json String: %s"), *BireModelJsonString);

			TSharedPtr<FJsonObject> BireModelJsonObject = MakeShareable(new FJsonObject());
			TSharedRef<TJsonReader<>> BireModelJsonReader = TJsonReaderFactory<>::Create(BireModelJsonString);

			// Store config JSON values in respective member variables
			if (FJsonSerializer::Deserialize(BireModelJsonReader, BireModelJsonObject) && BireModelJsonObject.IsValid())
			{
				// 'CL' category
				TSharedPtr<FJsonObject> BCLJsonObject = BireModelJsonObject->GetObjectField("CL");
				TSharedPtr<FJsonObject> BCL0JsonObject = BCLJsonObject->GetObjectField("CL_0");
				BCL0 = TArray<double>({
					BCL0JsonObject->GetNumberField("A"),
					BCL0JsonObject->GetNumberField("w"),
					BCL0JsonObject->GetNumberField("phi"),
					BCL0JsonObject->GetNumberField("z"),
					BCL0JsonObject->GetNumberField("delta"),
					BCL0JsonObject->GetNumberField("multiplier"),
				});
				TSharedPtr<FJsonObject> BCLaJsonObject = BCLJsonObject->GetObjectField("CL_alpha");
				BCLa = TArray<double>({
					BCLaJsonObject->GetNumberField("A"),
					BCLaJsonObject->GetNumberField("w"),
					BCLaJsonObject->GetNumberField("phi"),
					BCLaJsonObject->GetNumberField("z"),
					BCLaJsonObject->GetNumberField("delta"),
					BCLaJsonObject->GetNumberField("multiplier"),
				});
				TSharedPtr<FJsonObject> BCLbJsonObject = BCLJsonObject->GetObjectField("CL_beta");
				BCLb = TArray<double>({
					BCLbJsonObject->GetNumberField("A"),
					BCLbJsonObject->GetNumberField("w"),
					BCLbJsonObject->GetNumberField("phi"),
					BCLbJsonObject->GetNumberField("z"),
					BCLbJsonObject->GetNumberField("delta"),
					BCLbJsonObject->GetNumberField("multiplier"),
				});
				TSharedPtr<FJsonObject> BCLpbarJsonObject = BCLJsonObject->GetObjectField("CL_pbar");
				BCLpbar = TArray<double>({
					BCLpbarJsonObject->GetNumberField("A"),
					BCLpbarJsonObject->GetNumberField("w"),
					BCLpbarJsonObject->GetNumberField("phi"),
					BCLpbarJsonObject->GetNumberField("z"),
					BCLpbarJsonObject->GetNumberField("delta"),
					BCLpbarJsonObject->GetNumberField("multiplier"),
				});
				TSharedPtr<FJsonObject> BCLqbarJsonObject = BCLJsonObject->GetObjectField("CL_qbar");
				BCLqbar = TArray<double>({
					BCLqbarJsonObject->GetNumberField("A"),
					BCLqbarJsonObject->GetNumberField("w"),
					BCLqbarJsonObject->GetNumberField("phi"),
					BCLqbarJsonObject->GetNumberField("z"),
					BCLqbarJsonObject->GetNumberField("delta"),
					BCLqbarJsonObject->GetNumberField("multiplier"),
				});
				TSharedPtr<FJsonObject> BCLrbarJsonObject = BCLJsonObject->GetObjectField("CL_rbar");
				BCLrbar = TArray<double>({
					BCLrbarJsonObject->GetNumberField("A"),
					BCLrbarJsonObject->GetNumberField("w"),
					BCLrbarJsonObject->GetNumberField("phi"),
					BCLrbarJsonObject->GetNumberField("z"),
					BCLrbarJsonObject->GetNumberField("delta"),
					BCLrbarJsonObject->GetNumberField("multiplier"),
				});
				TSharedPtr<FJsonObject> BCLdaJsonObject = BCLJsonObject->GetObjectField("CL_da");
				BCLda = TArray<double>({
					BCLdaJsonObject->GetNumberField("A"),
					BCLdaJsonObject->GetNumberField("w"),
					BCLdaJsonObject->GetNumberField("phi"),
					BCLdaJsonObject->GetNumberField("z"),
					BCLdaJsonObject->GetNumberField("delta"),
					BCLdaJsonObject->GetNumberField("multiplier"),
				});
				TSharedPtr<FJsonObject> BCLdeJsonObject = BCLJsonObject->GetObjectField("CL_de");
				BCLde = TArray<double>({
					BCLdeJsonObject->GetNumberField("A"),
					BCLdeJsonObject->GetNumberField("w"),
					BCLdeJsonObject->GetNumberField("phi"),
					BCLdeJsonObject->GetNumberField("z"),
					BCLdeJsonObject->GetNumberField("delta"),
					BCLdeJsonObject->GetNumberField("multiplier"),
				});
				
				// 'CS' category
				TSharedPtr<FJsonObject> BCSJsonObject = BireModelJsonObject->GetObjectField("CS");
				TSharedPtr<FJsonObject> BCS0JsonObject = BCSJsonObject->GetObjectField("CS_0");
				BCS0 = TArray<double>({
					BCS0JsonObject->GetNumberField("A"),
					BCS0JsonObject->GetNumberField("w"),
					BCS0JsonObject->GetNumberField("phi"),
					BCS0JsonObject->GetNumberField("z"),
					BCS0JsonObject->GetNumberField("delta"),
					BCS0JsonObject->GetNumberField("multiplier"),
				});
				TSharedPtr<FJsonObject> BCSaJsonObject = BCSJsonObject->GetObjectField("CS_alpha");
				BCSa = TArray<double>({
					BCSaJsonObject->GetNumberField("A"),
					BCSaJsonObject->GetNumberField("w"),
					BCSaJsonObject->GetNumberField("phi"),
					BCSaJsonObject->GetNumberField("z"),
					BCSaJsonObject->GetNumberField("delta"),
					BCSaJsonObject->GetNumberField("multiplier"),
				});
				TSharedPtr<FJsonObject> BCSbJsonObject = BCSJsonObject->GetObjectField("CS_beta");
				BCSb = TArray<double>({
					BCSbJsonObject->GetNumberField("A"),
					BCSbJsonObject->GetNumberField("w"),
					BCSbJsonObject->GetNumberField("phi"),
					BCSbJsonObject->GetNumberField("z"),
					BCSbJsonObject->GetNumberField("delta"),
					BCSbJsonObject->GetNumberField("multiplier"),
				});
				TSharedPtr<FJsonObject> BCSpbarJsonObject = BCSJsonObject->GetObjectField("CS_pbar");
				BCSpbar = TArray<double>({
					BCSpbarJsonObject->GetNumberField("A"),
					BCSpbarJsonObject->GetNumberField("w"),
					BCSpbarJsonObject->GetNumberField("phi"),
					BCSpbarJsonObject->GetNumberField("z"),
					BCSpbarJsonObject->GetNumberField("delta"),
					BCSpbarJsonObject->GetNumberField("multiplier"),
				});
				TSharedPtr<FJsonObject> BCSLpbarJsonObject = BCSJsonObject->GetObjectField("CS_Lpbar");
				BCSLpbar = TArray<double>({
					BCSLpbarJsonObject->GetNumberField("A"),
					BCSLpbarJsonObject->GetNumberField("w"),
					BCSLpbarJsonObject->GetNumberField("phi"),
					BCSLpbarJsonObject->GetNumberField("z"),
					BCSLpbarJsonObject->GetNumberField("delta"),
					BCSLpbarJsonObject->GetNumberField("multiplier"),
				});
				TSharedPtr<FJsonObject> BCSqbarJsonObject = BCSJsonObject->GetObjectField("CS_qbar");
				BCSqbar = TArray<double>({
					BCSqbarJsonObject->GetNumberField("A"),
					BCSqbarJsonObject->GetNumberField("w"),
					BCSqbarJsonObject->GetNumberField("phi"),
					BCSqbarJsonObject->GetNumberField("z"),
					BCSqbarJsonObject->GetNumberField("delta"),
					BCSqbarJsonObject->GetNumberField("multiplier"),
				});
				TSharedPtr<FJsonObject> BCSrbarJsonObject = BCSJsonObject->GetObjectField("CS_rbar");
				BCSrbar = TArray<double>({
					BCSrbarJsonObject->GetNumberField("A"),
					BCSrbarJsonObject->GetNumberField("w"),
					BCSrbarJsonObject->GetNumberField("phi"),
					BCSrbarJsonObject->GetNumberField("z"),
					BCSrbarJsonObject->GetNumberField("delta"),
					BCSrbarJsonObject->GetNumberField("multiplier"),
				});
				TSharedPtr<FJsonObject> BCSdaJsonObject = BCSJsonObject->GetObjectField("CS_da");
				BCSda = TArray<double>({
					BCSdaJsonObject->GetNumberField("A"),
					BCSdaJsonObject->GetNumberField("w"),
					BCSdaJsonObject->GetNumberField("phi"),
					BCSdaJsonObject->GetNumberField("z"),
					BCSdaJsonObject->GetNumberField("delta"),
					BCSdaJsonObject->GetNumberField("multiplier"),
				});
				TSharedPtr<FJsonObject> BCSdeJsonObject = BCSJsonObject->GetObjectField("CS_de");
				BCSde = TArray<double>({
					BCSdeJsonObject->GetNumberField("A"),
					BCSdeJsonObject->GetNumberField("w"),
					BCSdeJsonObject->GetNumberField("phi"),
					BCSdeJsonObject->GetNumberField("z"),
					BCSdeJsonObject->GetNumberField("delta"),
					BCSdeJsonObject->GetNumberField("multiplier"),
				});
				
				// 'CD' category
				TSharedPtr<FJsonObject> BCDJsonObject = BireModelJsonObject->GetObjectField("CD");
				TSharedPtr<FJsonObject> BCD0JsonObject = BCDJsonObject->GetObjectField("CD_0");
				BCD0 = TArray<double>({
					BCD0JsonObject->GetNumberField("A"),
					BCD0JsonObject->GetNumberField("w"),
					BCD0JsonObject->GetNumberField("phi"),
					BCD0JsonObject->GetNumberField("z"),
					BCD0JsonObject->GetNumberField("delta"),
					BCD0JsonObject->GetNumberField("multiplier"),
				});
				TSharedPtr<FJsonObject> BCDLJsonObject = BCDJsonObject->GetObjectField("CD_L");
				BCDL = TArray<double>({
					BCDLJsonObject->GetNumberField("A"),
					BCDLJsonObject->GetNumberField("w"),
					BCDLJsonObject->GetNumberField("phi"),
					BCDLJsonObject->GetNumberField("z"),
					BCDLJsonObject->GetNumberField("delta"),
					BCDLJsonObject->GetNumberField("multiplier"),
				});
				TSharedPtr<FJsonObject> BCDL2JsonObject = BCDJsonObject->GetObjectField("CD_L2");
				BCDL2 = TArray<double>({
					BCDL2JsonObject->GetNumberField("A"),
					BCDL2JsonObject->GetNumberField("w"),
					BCDL2JsonObject->GetNumberField("phi"),
					BCDL2JsonObject->GetNumberField("z"),
					BCDL2JsonObject->GetNumberField("delta"),
					BCDL2JsonObject->GetNumberField("multiplier"),
				});
				TSharedPtr<FJsonObject> BCDSJsonObject = BCDJsonObject->GetObjectField("CD_S");
				BCDS = TArray<double>({
					BCDSJsonObject->GetNumberField("A"),
					BCDSJsonObject->GetNumberField("w"),
					BCDSJsonObject->GetNumberField("phi"),
					BCDSJsonObject->GetNumberField("z"),
					BCDSJsonObject->GetNumberField("delta"),
					BCDSJsonObject->GetNumberField("multiplier"),
				});
				TSharedPtr<FJsonObject> BCDS2JsonObject = BCDJsonObject->GetObjectField("CD_S2");
				BCDS2 = TArray<double>({
					BCDS2JsonObject->GetNumberField("A"),
					BCDS2JsonObject->GetNumberField("w"),
					BCDS2JsonObject->GetNumberField("phi"),
					BCDS2JsonObject->GetNumberField("z"),
					BCDS2JsonObject->GetNumberField("delta"),
					BCDS2JsonObject->GetNumberField("multiplier"),
				});
				TSharedPtr<FJsonObject> BCDpbarJsonObject = BCDJsonObject->GetObjectField("CD_pbar");
				BCDpbar = TArray<double>({
					BCDpbarJsonObject->GetNumberField("A"),
					BCDpbarJsonObject->GetNumberField("w"),
					BCDpbarJsonObject->GetNumberField("phi"),
					BCDpbarJsonObject->GetNumberField("z"),
					BCDpbarJsonObject->GetNumberField("delta"),
					BCDpbarJsonObject->GetNumberField("multiplier"),
				});
				TSharedPtr<FJsonObject> BCDSpbarJsonObject = BCDJsonObject->GetObjectField("CD_Spbar");
				BCDSpbar = TArray<double>({
					BCDSpbarJsonObject->GetNumberField("A"),
					BCDSpbarJsonObject->GetNumberField("w"),
					BCDSpbarJsonObject->GetNumberField("phi"),
					BCDSpbarJsonObject->GetNumberField("z"),
					BCDSpbarJsonObject->GetNumberField("delta"),
					BCDSpbarJsonObject->GetNumberField("multiplier"),
				});
				TSharedPtr<FJsonObject> BCDqbarJsonObject = BCDJsonObject->GetObjectField("CD_qbar");
				BCDqbar = TArray<double>({
					BCDqbarJsonObject->GetNumberField("A"),
					BCDqbarJsonObject->GetNumberField("w"),
					BCDqbarJsonObject->GetNumberField("phi"),
					BCDqbarJsonObject->GetNumberField("z"),
					BCDqbarJsonObject->GetNumberField("delta"),
					BCDqbarJsonObject->GetNumberField("multiplier"),
				});
				TSharedPtr<FJsonObject> BCDLqbarJsonObject = BCDJsonObject->GetObjectField("CD_Lqbar");
				BCDLqbar = TArray<double>({
					BCDLqbarJsonObject->GetNumberField("A"),
					BCDLqbarJsonObject->GetNumberField("w"),
					BCDLqbarJsonObject->GetNumberField("phi"),
					BCDLqbarJsonObject->GetNumberField("z"),
					BCDLqbarJsonObject->GetNumberField("delta"),
					BCDLqbarJsonObject->GetNumberField("multiplier"),
				});
				TSharedPtr<FJsonObject> BCDL2qbarJsonObject = BCDJsonObject->GetObjectField("CD_L2qbar");
				BCDL2qbar = TArray<double>({
					BCDL2qbarJsonObject->GetNumberField("A"),
					BCDL2qbarJsonObject->GetNumberField("w"),
					BCDL2qbarJsonObject->GetNumberField("phi"),
					BCDL2qbarJsonObject->GetNumberField("z"),
					BCDL2qbarJsonObject->GetNumberField("delta"),
					BCDL2qbarJsonObject->GetNumberField("multiplier"),
				});
				TSharedPtr<FJsonObject> BCDrbarJsonObject = BCDJsonObject->GetObjectField("CD_rbar");
				BCDrbar = TArray<double>({
					BCDrbarJsonObject->GetNumberField("A"),
					BCDrbarJsonObject->GetNumberField("w"),
					BCDrbarJsonObject->GetNumberField("phi"),
					BCDrbarJsonObject->GetNumberField("z"),
					BCDrbarJsonObject->GetNumberField("delta"),
					BCDrbarJsonObject->GetNumberField("multiplier"),
				});
				TSharedPtr<FJsonObject> BCDSrbarJsonObject = BCDJsonObject->GetObjectField("CD_Srbar");
				BCDSrbar = TArray<double>({
					BCDSrbarJsonObject->GetNumberField("A"),
					BCDSrbarJsonObject->GetNumberField("w"),
					BCDSrbarJsonObject->GetNumberField("phi"),
					BCDSrbarJsonObject->GetNumberField("z"),
					BCDSrbarJsonObject->GetNumberField("delta"),
					BCDSrbarJsonObject->GetNumberField("multiplier"),
				});
				TSharedPtr<FJsonObject> BCDdaJsonObject = BCDJsonObject->GetObjectField("CD_da");
				BCDda = TArray<double>({
					BCDdaJsonObject->GetNumberField("A"),
					BCDdaJsonObject->GetNumberField("w"),
					BCDdaJsonObject->GetNumberField("phi"),
					BCDdaJsonObject->GetNumberField("z"),
					BCDdaJsonObject->GetNumberField("delta"),
					BCDdaJsonObject->GetNumberField("multiplier"),
				});
				TSharedPtr<FJsonObject> BCDSdaJsonObject = BCDJsonObject->GetObjectField("CD_Sda");
				BCDSda = TArray<double>({
					BCDSdaJsonObject->GetNumberField("A"),
					BCDSdaJsonObject->GetNumberField("w"),
					BCDSdaJsonObject->GetNumberField("phi"),
					BCDSdaJsonObject->GetNumberField("z"),
					BCDSdaJsonObject->GetNumberField("delta"),
					BCDSdaJsonObject->GetNumberField("multiplier"),
				});
				TSharedPtr<FJsonObject> BCDdeJsonObject = BCDJsonObject->GetObjectField("CD_de");
				BCDde = TArray<double>({
					BCDdeJsonObject->GetNumberField("A"),
					BCDdeJsonObject->GetNumberField("w"),
					BCDdeJsonObject->GetNumberField("phi"),
					BCDdeJsonObject->GetNumberField("z"),
					BCDdeJsonObject->GetNumberField("delta"),
					BCDdeJsonObject->GetNumberField("multiplier"),
				});
				TSharedPtr<FJsonObject> BCDLdeJsonObject = BCDJsonObject->GetObjectField("CD_Lde");
				BCDLde = TArray<double>({
					BCDLdeJsonObject->GetNumberField("A"),
					BCDLdeJsonObject->GetNumberField("w"),
					BCDLdeJsonObject->GetNumberField("phi"),
					BCDLdeJsonObject->GetNumberField("z"),
					BCDLdeJsonObject->GetNumberField("delta"),
					BCDLdeJsonObject->GetNumberField("multiplier"),
				});
				TSharedPtr<FJsonObject> BCDde2JsonObject = BCDJsonObject->GetObjectField("CD_de2");
				BCDde2 = TArray<double>({
					BCDde2JsonObject->GetNumberField("A"),
					BCDde2JsonObject->GetNumberField("w"),
					BCDde2JsonObject->GetNumberField("phi"),
					BCDde2JsonObject->GetNumberField("z"),
					BCDde2JsonObject->GetNumberField("delta"),
					BCDde2JsonObject->GetNumberField("multiplier"),
				});
				
				// 'C_l' category
				TSharedPtr<FJsonObject> BClJsonObject = BireModelJsonObject->GetObjectField("C_l");
				TSharedPtr<FJsonObject> BCl0JsonObject = BClJsonObject->GetObjectField("Cl_0");
				BCl0 = TArray<double>({
					BCl0JsonObject->GetNumberField("A"),
					BCl0JsonObject->GetNumberField("w"),
					BCl0JsonObject->GetNumberField("phi"),
					BCl0JsonObject->GetNumberField("z"),
					BCl0JsonObject->GetNumberField("delta"),
					BCl0JsonObject->GetNumberField("multiplier"),
				});
				TSharedPtr<FJsonObject> BClaJsonObject = BClJsonObject->GetObjectField("Cl_alpha");
				BCla = TArray<double>({
					BClaJsonObject->GetNumberField("A"),
					BClaJsonObject->GetNumberField("w"),
					BClaJsonObject->GetNumberField("phi"),
					BClaJsonObject->GetNumberField("z"),
					BClaJsonObject->GetNumberField("delta"),
					BClaJsonObject->GetNumberField("multiplier"),
				});
				TSharedPtr<FJsonObject> BClbJsonObject = BClJsonObject->GetObjectField("Cl_beta");
				BClb = TArray<double>({
					BClbJsonObject->GetNumberField("A"),
					BClbJsonObject->GetNumberField("w"),
					BClbJsonObject->GetNumberField("phi"),
					BClbJsonObject->GetNumberField("z"),
					BClbJsonObject->GetNumberField("delta"),
					BClbJsonObject->GetNumberField("multiplier"),
				});
				TSharedPtr<FJsonObject> BClpbarJsonObject = BClJsonObject->GetObjectField("Cl_pbar");
				BClpbar = TArray<double>({
					BClpbarJsonObject->GetNumberField("A"),
					BClpbarJsonObject->GetNumberField("w"),
					BClpbarJsonObject->GetNumberField("phi"),
					BClpbarJsonObject->GetNumberField("z"),
					BClpbarJsonObject->GetNumberField("delta"),
					BClpbarJsonObject->GetNumberField("multiplier"),
				});
				TSharedPtr<FJsonObject> BClqbarJsonObject = BClJsonObject->GetObjectField("Cl_qbar");
				BClqbar = TArray<double>({
					BClqbarJsonObject->GetNumberField("A"),
					BClqbarJsonObject->GetNumberField("w"),
					BClqbarJsonObject->GetNumberField("phi"),
					BClqbarJsonObject->GetNumberField("z"),
					BClqbarJsonObject->GetNumberField("delta"),
					BClqbarJsonObject->GetNumberField("multiplier"),
				});
				TSharedPtr<FJsonObject> BClrbarJsonObject = BClJsonObject->GetObjectField("Cl_rbar");
				BClrbar = TArray<double>({
					BClrbarJsonObject->GetNumberField("A"),
					BClrbarJsonObject->GetNumberField("w"),
					BClrbarJsonObject->GetNumberField("phi"),
					BClrbarJsonObject->GetNumberField("z"),
					BClrbarJsonObject->GetNumberField("delta"),
					BClrbarJsonObject->GetNumberField("multiplier"),
				});
				TSharedPtr<FJsonObject> BClLrbarJsonObject = BClJsonObject->GetObjectField("Cl_Lrbar");
				BClLrbar = TArray<double>({
					BClLrbarJsonObject->GetNumberField("A"),
					BClLrbarJsonObject->GetNumberField("w"),
					BClLrbarJsonObject->GetNumberField("phi"),
					BClLrbarJsonObject->GetNumberField("z"),
					BClLrbarJsonObject->GetNumberField("delta"),
					BClLrbarJsonObject->GetNumberField("multiplier"),
				});
				TSharedPtr<FJsonObject> BCldaJsonObject = BClJsonObject->GetObjectField("Cl_da");
				BClda = TArray<double>({
					BCldaJsonObject->GetNumberField("A"),
					BCldaJsonObject->GetNumberField("w"),
					BCldaJsonObject->GetNumberField("phi"),
					BCldaJsonObject->GetNumberField("z"),
					BCldaJsonObject->GetNumberField("delta"),
					BCldaJsonObject->GetNumberField("multiplier"),
				});
				TSharedPtr<FJsonObject> BCldeJsonObject = BClJsonObject->GetObjectField("Cl_de");
				BClde = TArray<double>({
					BCldeJsonObject->GetNumberField("A"),
					BCldeJsonObject->GetNumberField("w"),
					BCldeJsonObject->GetNumberField("phi"),
					BCldeJsonObject->GetNumberField("z"),
					BCldeJsonObject->GetNumberField("delta"),
					BCldeJsonObject->GetNumberField("multiplier"),
				});

				// 'Cm' category
				TSharedPtr<FJsonObject> BCmJsonObject = BireModelJsonObject->GetObjectField("Cm");
				TSharedPtr<FJsonObject> BCm0JsonObject = BCmJsonObject->GetObjectField("Cm_0");
				BCm0 = TArray<double>({
					BCm0JsonObject->GetNumberField("A"),
					BCm0JsonObject->GetNumberField("w"),
					BCm0JsonObject->GetNumberField("phi"),
					BCm0JsonObject->GetNumberField("z"),
					BCm0JsonObject->GetNumberField("delta"),
					BCm0JsonObject->GetNumberField("multiplier"),
				});
				TSharedPtr<FJsonObject> BCmaJsonObject = BCmJsonObject->GetObjectField("Cm_alpha");
				BCma = TArray<double>({
					BCmaJsonObject->GetNumberField("A"),
					BCmaJsonObject->GetNumberField("w"),
					BCmaJsonObject->GetNumberField("phi"),
					BCmaJsonObject->GetNumberField("z"),
					BCmaJsonObject->GetNumberField("delta"),
					BCmaJsonObject->GetNumberField("multiplier"),
				});
				TSharedPtr<FJsonObject> BCmbJsonObject = BCmJsonObject->GetObjectField("Cm_beta");
				BCmb = TArray<double>({
					BCmbJsonObject->GetNumberField("A"),
					BCmbJsonObject->GetNumberField("w"),
					BCmbJsonObject->GetNumberField("phi"),
					BCmbJsonObject->GetNumberField("z"),
					BCmbJsonObject->GetNumberField("delta"),
					BCmbJsonObject->GetNumberField("multiplier"),
				});
				TSharedPtr<FJsonObject> BCmpbarJsonObject = BCmJsonObject->GetObjectField("Cm_pbar");
				BCmpbar = TArray<double>({
					BCmpbarJsonObject->GetNumberField("A"),
					BCmpbarJsonObject->GetNumberField("w"),
					BCmpbarJsonObject->GetNumberField("phi"),
					BCmpbarJsonObject->GetNumberField("z"),
					BCmpbarJsonObject->GetNumberField("delta"),
					BCmpbarJsonObject->GetNumberField("multiplier"),
				});
				TSharedPtr<FJsonObject> BCmqbarJsonObject = BCmJsonObject->GetObjectField("Cm_qbar");
				BCmqbar = TArray<double>({
					BCmqbarJsonObject->GetNumberField("A"),
					BCmqbarJsonObject->GetNumberField("w"),
					BCmqbarJsonObject->GetNumberField("phi"),
					BCmqbarJsonObject->GetNumberField("z"),
					BCmqbarJsonObject->GetNumberField("delta"),
					BCmqbarJsonObject->GetNumberField("multiplier"),
				});
				TSharedPtr<FJsonObject> BCmrbarJsonObject = BCmJsonObject->GetObjectField("Cm_rbar");
				BCmrbar = TArray<double>({
					BCmrbarJsonObject->GetNumberField("A"),
					BCmrbarJsonObject->GetNumberField("w"),
					BCmrbarJsonObject->GetNumberField("phi"),
					BCmrbarJsonObject->GetNumberField("z"),
					BCmrbarJsonObject->GetNumberField("delta"),
					BCmrbarJsonObject->GetNumberField("multiplier"),
				});
				TSharedPtr<FJsonObject> BCmdaJsonObject = BCmJsonObject->GetObjectField("Cm_da");
				BCmda = TArray<double>({
					BCmdaJsonObject->GetNumberField("A"),
					BCmdaJsonObject->GetNumberField("w"),
					BCmdaJsonObject->GetNumberField("phi"),
					BCmdaJsonObject->GetNumberField("z"),
					BCmdaJsonObject->GetNumberField("delta"),
					BCmdaJsonObject->GetNumberField("multiplier"),
				});
				TSharedPtr<FJsonObject> BCmdeJsonObject = BCmJsonObject->GetObjectField("Cm_de");
				BCmde = TArray<double>({
					BCmdeJsonObject->GetNumberField("A"),
					BCmdeJsonObject->GetNumberField("w"),
					BCmdeJsonObject->GetNumberField("phi"),
					BCmdeJsonObject->GetNumberField("z"),
					BCmdeJsonObject->GetNumberField("delta"),
					BCmdeJsonObject->GetNumberField("multiplier"),
				});
				
				// 'Cn' category
				TSharedPtr<FJsonObject> BCnJsonObject = BireModelJsonObject->GetObjectField("Cn");
				TSharedPtr<FJsonObject> BCn0JsonObject = BCnJsonObject->GetObjectField("Cn_0");
				BCn0 = TArray<double>({
					BCn0JsonObject->GetNumberField("A"),
					BCn0JsonObject->GetNumberField("w"),
					BCn0JsonObject->GetNumberField("phi"),
					BCn0JsonObject->GetNumberField("z"),
					BCn0JsonObject->GetNumberField("delta"),
					BCn0JsonObject->GetNumberField("multiplier"),
				});
				TSharedPtr<FJsonObject> BCnaJsonObject = BCnJsonObject->GetObjectField("Cn_alpha");
				BCna = TArray<double>({
					BCnaJsonObject->GetNumberField("A"),
					BCnaJsonObject->GetNumberField("w"),
					BCnaJsonObject->GetNumberField("phi"),
					BCnaJsonObject->GetNumberField("z"),
					BCnaJsonObject->GetNumberField("delta"),
					BCnaJsonObject->GetNumberField("multiplier"),
				});
				TSharedPtr<FJsonObject> BCnbJsonObject = BCnJsonObject->GetObjectField("Cn_beta");
				BCnb = TArray<double>({
					BCnbJsonObject->GetNumberField("A"),
					BCnbJsonObject->GetNumberField("w"),
					BCnbJsonObject->GetNumberField("phi"),
					BCnbJsonObject->GetNumberField("z"),
					BCnbJsonObject->GetNumberField("delta"),
					BCnbJsonObject->GetNumberField("multiplier"),
				});
				TSharedPtr<FJsonObject> BCnpbarJsonObject = BCnJsonObject->GetObjectField("Cn_pbar");
				BCnpbar = TArray<double>({
					BCnpbarJsonObject->GetNumberField("A"),
					BCnpbarJsonObject->GetNumberField("w"),
					BCnpbarJsonObject->GetNumberField("phi"),
					BCnpbarJsonObject->GetNumberField("z"),
					BCnpbarJsonObject->GetNumberField("delta"),
					BCnpbarJsonObject->GetNumberField("multiplier"),
				});
				TSharedPtr<FJsonObject> BCnLpbarJsonObject = BCnJsonObject->GetObjectField("Cn_Lpbar");
				BCnLpbar = TArray<double>({
					BCnLpbarJsonObject->GetNumberField("A"),
					BCnLpbarJsonObject->GetNumberField("w"),
					BCnLpbarJsonObject->GetNumberField("phi"),
					BCnLpbarJsonObject->GetNumberField("z"),
					BCnLpbarJsonObject->GetNumberField("delta"),
					BCnLpbarJsonObject->GetNumberField("multiplier"),
				});
				TSharedPtr<FJsonObject> BCnqbarJsonObject = BCnJsonObject->GetObjectField("Cn_qbar");
				BCnqbar = TArray<double>({
					BCnqbarJsonObject->GetNumberField("A"),
					BCnqbarJsonObject->GetNumberField("w"),
					BCnqbarJsonObject->GetNumberField("phi"),
					BCnqbarJsonObject->GetNumberField("z"),
					BCnqbarJsonObject->GetNumberField("delta"),
					BCnqbarJsonObject->GetNumberField("multiplier"),
				});
				TSharedPtr<FJsonObject> BCnrbarJsonObject = BCnJsonObject->GetObjectField("Cn_rbar");
				BCnrbar = TArray<double>({
					BCnrbarJsonObject->GetNumberField("A"),
					BCnrbarJsonObject->GetNumberField("w"),
					BCnrbarJsonObject->GetNumberField("phi"),
					BCnrbarJsonObject->GetNumberField("z"),
					BCnrbarJsonObject->GetNumberField("delta"),
					BCnrbarJsonObject->GetNumberField("multiplier"),
				});
				TSharedPtr<FJsonObject> BCndaJsonObject = BCnJsonObject->GetObjectField("Cn_da");
				BCnda = TArray<double>({
					BCndaJsonObject->GetNumberField("A"),
					BCndaJsonObject->GetNumberField("w"),
					BCndaJsonObject->GetNumberField("phi"),
					BCndaJsonObject->GetNumberField("z"),
					BCndaJsonObject->GetNumberField("delta"),
					BCndaJsonObject->GetNumberField("multiplier"),
				});
				TSharedPtr<FJsonObject> BCnLdaJsonObject = BCnJsonObject->GetObjectField("Cn_Lda");
				BCnLda = TArray<double>({
					BCnLdaJsonObject->GetNumberField("A"),
					BCnLdaJsonObject->GetNumberField("w"),
					BCnLdaJsonObject->GetNumberField("phi"),
					BCnLdaJsonObject->GetNumberField("z"),
					BCnLdaJsonObject->GetNumberField("delta"),
					BCnLdaJsonObject->GetNumberField("multiplier"),
				});
				TSharedPtr<FJsonObject> BCndeJsonObject = BCnJsonObject->GetObjectField("Cn_de");
				BCnde = TArray<double>({
					BCndeJsonObject->GetNumberField("A"),
					BCndeJsonObject->GetNumberField("w"),
					BCndeJsonObject->GetNumberField("phi"),
					BCndeJsonObject->GetNumberField("z"),
					BCndeJsonObject->GetNumberField("delta"),
					BCndeJsonObject->GetNumberField("multiplier"),
				});
				
			}
			else
			{
				UE_LOG(LogTemp, Error, TEXT("Couldn't deserialize BIRE model JSON"));
			}

		}
		else
		{
			// Initialize traditional aircraft aero coefficients
			// 'CL' category of 'aerodynamics'
			TSharedPtr<FJsonObject> CLJsonObject = AerodynamicsJsonObject->GetObjectField("CL");
			CL0 = CLJsonObject->GetNumberField("CL_0");
			CLa = CLJsonObject->GetNumberField("CL_alpha");
			CLqbar = CLJsonObject->GetNumberField("CL_qbar");
			CLde = CLJsonObject->GetNumberField("CL_de");
			// 'CS' category of 'aerodynamics'
			TSharedPtr<FJsonObject> CSJsonObject = AerodynamicsJsonObject->GetObjectField("CS");
			CSb = CSJsonObject->GetNumberField("CS_beta");
			CSpbar = CSJsonObject->GetNumberField("CS_pbar");
			CSLpbar = CSJsonObject->GetNumberField("CS_Lpbar");
			CSrbar = CSJsonObject->GetNumberField("CS_rbar");
			CSda = CSJsonObject->GetNumberField("CS_da");
			CSdr = CSJsonObject->GetNumberField("CS_dr");
			// 'CD' category of 'aerodynamics'
			TSharedPtr<FJsonObject> CDJsonObject = AerodynamicsJsonObject->GetObjectField("CD");
			CD0 = CDJsonObject->GetNumberField("CD_0");
			CDL = CDJsonObject->GetNumberField("CD_L");
			CDL2 = CDJsonObject->GetNumberField("CD_L2");
			CDS2 = CDJsonObject->GetNumberField("CD_S2");
			CDpbar = CDJsonObject->GetNumberField("CD_pbar");
			CDSpbar = CDJsonObject->GetNumberField("CD_Spbar");
			CDqbar = CDJsonObject->GetNumberField("CD_qbar");
			CDLqbar = CDJsonObject->GetNumberField("CD_Lqbar");
			CDL2qbar = CDJsonObject->GetNumberField("CD_L2qbar");
			CDrbar = CDJsonObject->GetNumberField("CD_rbar");
			CDSrbar = CDJsonObject->GetNumberField("CD_Srbar");
			CDde = CDJsonObject->GetNumberField("CD_de");
			CDLde = CDJsonObject->GetNumberField("CD_Lde");
			CDde2 = CDJsonObject->GetNumberField("CD_de2");
			CDda = CDJsonObject->GetNumberField("CD_da");
			CDSda = CDJsonObject->GetNumberField("CD_Sda");
			CDdr = CDJsonObject->GetNumberField("CD_dr");
			CDSdr = CDJsonObject->GetNumberField("CD_Sdr");
			// 'Cl' category of 'aerodynamics'
			TSharedPtr<FJsonObject> ClJsonObject = AerodynamicsJsonObject->GetObjectField("C_l");
			Clb = ClJsonObject->GetNumberField("Cl_beta");
			Clpbar = ClJsonObject->GetNumberField("Cl_pbar");
			Clrbar = ClJsonObject->GetNumberField("Cl_rbar");
			ClLrbar = ClJsonObject->GetNumberField("Cl_Lrbar");
			Clda = ClJsonObject->GetNumberField("Cl_da");
			Cldr = ClJsonObject->GetNumberField("Cl_dr");
			// 'Cm' category of 'aerodynamics'
			TSharedPtr<FJsonObject> CmJsonObject = AerodynamicsJsonObject->GetObjectField("Cm");
			Cm0 = CmJsonObject->GetNumberField("Cm_0");
			Cma = CmJsonObject->GetNumberField("Cm_alpha");
			Cmqbar = CmJsonObject->GetNumberField("Cm_qbar");
			Cmde = CmJsonObject->GetNumberField("Cm_de");
			// 'Cm' category of 'aerodynamics'
			TSharedPtr<FJsonObject> CnJsonObject = AerodynamicsJsonObject->GetObjectField("Cn");
			Cnb = CnJsonObject->GetNumberField("Cn_beta");
			Cnpbar = CnJsonObject->GetNumberField("Cn_pbar");
			CnLpbar = CnJsonObject->GetNumberField("Cn_Lpbar");
			Cnrbar = CnJsonObject->GetNumberField("Cn_rbar");
			Cnda = CnJsonObject->GetNumberField("Cn_da");
			CnLda = CnJsonObject->GetNumberField("Cn_Lda");
			Cndr = CnJsonObject->GetNumberField("Cn_dr");
		}
		
	}
	else
	{
		UE_LOG(LogTemp, Error, TEXT("Couldn't deserialize config JSON"));
	}

	// Precompute global values for other functions to use
	cw = Sw/bw;
	rho0 = CalculateStdAtmProperties_English(0.)[3];
	WInv = 1./W;
	/*double gFractionInit = (20855531.49606297/(20855531.49606297 + altInit));
	gInit = 32.174032152230936*gFractionInit*gFractionInit;*/
	double gFractionInit = (6356766./(6356766. + altInit*0.3048));
	gInit = 9.806645*gFractionInit*gFractionInit/0.3048;
	Wg = W/gInit;
	Eigen::Matrix3d I;
	I << Ixx, -Ixy, -Ixz,
		-Ixy, Iyy, -Iyz,
		-Ixz, -Iyz, Izz;
	IInv = I.inverse();
	hArray << 0., -hz, hy,
		hz, 0., -hx,
		-hy, hx, 0.;


	// Initialize landing gear information
	// Each gear location is relative to the COM (ft), so must be shifted
	frontGearLoc -= CGShift;
	rightGearLoc -= CGShift;
	leftGearLoc -= CGShift;
	CurrentGearDeflections = Eigen::Vector3d(0.0, 0.0, 0.0);
	PreviousGearDeflections = Eigen::Vector3d(0.0, 0.0, 0.0);

	// Initialize gust values
	GustVelocity = Eigen::Vector3d(0.0, 0.0, 0.0);
	GustAcceleration = Eigen::Vector3d(0.0, 0.0, 0.0);
	GustStartTimes = Eigen::Vector3d(
		double(FMath::RandRange(1.0f, 10.0f)),
		double(FMath::RandRange(1.0f, 10.0f)),
		double(FMath::RandRange(1.0f, 10.0f))
	);

	// Initialize AircraftStates and AircraftControls using JSON prescribed states (InitializeFromState) or trimming the aircraft (InitializeFromTrim)
	if (InitType == "state")
	{
		InitializeFromState();
	}
	else if (InitType == "trim")
	{
		InitializeFromTrimJacobian();
	}
	else
	{
		UE_LOG(LogTemp, Error, TEXT("Incorrect aircraft initialization type. JSON 'initial/trim/type' should have value 'trim' or 'state'."));
	}

	// Update current inertia based on trim results
	if (bIsBireAircraft)
	{
		UpdateBIREInertia();
	}

}

TArray<double> AAircraft::CalculateStdAtmProperties_SI(double H)
{
	double Z = 6356766. * H / (6356766. + H);
	// Values hardcoded for code speed
	if (Z < 11000.)
	{
		double T = 288.15 - 0.0065 * Z;
		double p = 1.013250000000000E+05 * pow(T / 288.15, -0.03416320969521983 / -0.0065);
		double rho = p / (287.0528 * T);
		double a = sqrt(401.87392 * T);
		
		return TArray<double>({ Z, T, p, rho, a });
	}
	else if (Z < 20000.)
	{
		double T = 216.65;
		double p = 2.263204911899440E+04 * exp(-0.03416320969521983 * (Z - 11000.) / 216.65);
		double rho = p / (287.0528 * T);
		double a = sqrt(401.87392 * T);
		
		return TArray<double>({ Z, T, p, rho, a });
	}
	else if (Z < 32000.)
	{
		double T = 216.65 + 0.001 * (Z - 20000.);
		double p = 5.474881674065110E+03 * pow(T / 216.65, -0.03416320969521983 / 0.001);
		double rho = p / (287.0528 * T);
		double a = sqrt(401.87392 * T);
		
		return TArray<double>({ Z, T, p, rho, a });
	}
	else if (Z < 47000.)
	{
		double T = 228.65 + 0.0028 * (Z - 32000.);
		double p = 8.680168756424330E+02 * pow(T / 228.65, -0.03416320969521983 / 0.0028);
		double rho = p / (287.0528 * T);
		double a = sqrt(401.87392 * T);
		
		return TArray<double>({ Z, T, p, rho, a });
	}
	else if (Z < 52000.)
	{
		double T = 270.65;
		double p = 1.109059744878880E+02 * exp(-0.03416320969521983 * (Z - 47000.) / 270.65);
		double rho = p / (287.0528 * T);
		double a = sqrt(401.87392 * T);
		
		return TArray<double>({ Z, T, p, rho, a });

	}
	else if (Z < 61000.)
	{
		double T = 270.65 - 0.002 * (Z - 52000.);
		double p = 5.900074834561620E+01 * pow(T / 270.65, -0.03416320969521983 / -0.002);
		double rho = p / (287.0528 * T);
		double a = sqrt(401.87392 * T);
		
		return TArray<double>({ Z, T, p, rho, a });
	}
	else if (Z < 79000.)
	{
		double T = 252.65 - 0.004 * (Z - 61000.);
		double p = 1.821000497566030E+01 * pow(T / 252.65, -0.03416320969521983 / -0.004);
		double rho = p / (287.0528 * T);
		double a = sqrt(401.87392 * T);
		
		return TArray<double>({ Z, T, p, rho, a });
	}
	else if (Z < 90000.)
	{
		double T = 180.65;
		double p = 1.037706533552830E+00 * exp(-0.03416320969521983 * (Z - 79000.) / 180.65);
		double rho = p / (287.0528 * T);
		double a = sqrt(401.87392 * T);
		
		return TArray<double>({ Z, T, p, rho, a });
	}
	else
	{
		double T = 180.65;
		double p = 0.;
		double rho = 0.;
		double a = sqrt(401.87392 * T);
		
		return TArray<double>({ Z, T, p, rho, a });
	}
}

TArray<double> AAircraft::CalculateStdAtmProperties_English(double H)
{
	TArray<double> StdAtmPropertiesSI = CalculateStdAtmProperties_SI(H * 0.3048);
	
	double Z = StdAtmPropertiesSI[0] * 3.28083989501312;
	double T = StdAtmPropertiesSI[1] * 1.8;
	double p = StdAtmPropertiesSI[2] * 0.020885434304801722;
	double rho = StdAtmPropertiesSI[3] * 0.00194032032363104;
	double a = StdAtmPropertiesSI[4] * 3.28083989501312;
	
	return TArray<double>({ Z, T, p, rho, a });
}

Eigen::Matrix<double, 13, 1> AAircraft::CalculateStatesChange(double t, Eigen::Matrix<double, 13, 1> States, Eigen::Vector4d Controls)
{
	double u = States(0);
	double v = States(1);
	double w = States(2);
	double p = States(3);
	double q = States(4);
	double r = States(5);
	double e0 = States(9);
	double ex = States(10);
	double ey = States(11);
	double ez = States(12);
	double gFraction = 20855531.49606297 / (20855531.49606297 - States(8));
	double g = 32.174032152230936 * gFraction * gFraction;
	
	TArray<double> AeroForces = CalculateAeroForces(States, Controls);
	double Fx = AeroForces[0] + LandingForces[0];
	double Fy = AeroForces[1] + LandingForces[1];
	double Fz = AeroForces[2] + LandingForces[2];
	double Mx = AeroForces[3] + LandingForces[3];
	double My = AeroForces[4] + LandingForces[4];
	double Mz = AeroForces[5] + LandingForces[5];

	UpdateGustState(t);
	TArray<double> bodyGustAcceleration = EarthToBodyFixed(
		TArray<double>({ GustAcceleration(0), GustAcceleration(1), GustAcceleration(2) }),
		TArray<double>({ e0, ex, ey, ez })
	);
	
	Eigen::Vector3d hpqr = hArray * (Eigen::Vector3d(p, q, r));
	Eigen::Vector3d Mpqr = Eigen::Vector3d( 
		Mx + (Iyy - Izz)*q*r +  Iyz*(q*q - r*r) + Ixz*p*q - Ixy*p*r,
		My + (Izz - Ixx)*p*r + Ixz*(r*r - p*p) + Ixy*q*r - Iyz*p*q,
		Mz + (Ixx - Iyy)*p*q + Ixy * (p*p - q*q) + Iyz*p*r - Ixz*q*r 
	);
	Eigen::Vector3d pqrdot = IInv * (hpqr + Mpqr);
	double pdot = pqrdot(0);
	double qdot = pqrdot(1);
	double rdot = pqrdot(2);

	TArray<double> xyzdot = BodyToEarthFixed(TArray<double>({ u, v, w }), TArray<double>({ e0, ex, ey, ez }));
	double xdot = xyzdot[0] + windVect(0) + GustVelocity(0);
	double ydot = xyzdot[1] + windVect(1) + GustVelocity(1);
	double zdot = xyzdot[2] + windVect(2) + GustVelocity(2);

	return Eigen::Matrix<double, 13, 1>(
		g*Fx*WInv + g*2.*(ex*ez - ey*e0) + r*v - q*w + bodyGustAcceleration[0],
		g*Fy*WInv + g*2.*(ey*ez + ex*e0) + p*w - r*u + bodyGustAcceleration[1],
        g*Fz*WInv + g*(ez*ez + e0*e0 - ex*ex - ey*ey) + q*u - p*v + bodyGustAcceleration[2],
        pdot,
        qdot,
        rdot,
        xdot,
        ydot,
        zdot,
        0.5*(-ex*p - ey*q - ez*r),
        0.5*(e0*p - ez*q + ey*r),
        0.5*(ez*p + e0*q - ex*r),
        0.5*(-ey*p + ex*q + e0*r)
	);
}

Eigen::Matrix<double, 13, 1> AAircraft::IntegrateStates_RK4(double t0, Eigen::Matrix<double, 13, 1> States, Eigen::Vector4d Controls, double dt)
{
	Eigen::Matrix<double, 13, 1> k1 = CalculateStatesChange( t0, States, Controls);
	Eigen::Matrix<double, 13, 1> k2 = CalculateStatesChange( t0 + 0.5*dt, States + 0.5*dt*k1, Controls);
	Eigen::Matrix<double, 13, 1> k3 = CalculateStatesChange( t0 + 0.5*dt, States + 0.5*dt*k2, Controls);
	Eigen::Matrix<double, 13, 1> k4 = CalculateStatesChange( t0 + dt, States + dt*k3, Controls);

	return States + 0.166666666666666667*dt*(k1 + 2.*k2 + 2.*k3 + k4);
}

TArray<double> AAircraft::CalculateAeroForces(Eigen::Matrix<double, 13, 1> States, Eigen::Vector4d Controls)
{
	double u = States(0);
	double v = States(1);
	double w = States(2);
	double p = States(3);
	double q = States(4);
	double r = States(5);
	double H = -States(8);
	double da = Controls(0);
	double de = Controls(1);
	double dr = Controls(2);
	double tau = Controls(3);

	// Intermediate state-based values
	double V = sqrt(u*u + v*v + w*w);
	double halfVinv = 0.5/V;
    double pbar = p*bw*halfVinv;
    double qbar = q*cw*halfVinv;
    double rbar = r*bw*halfVinv;
    double alpha = atan2(w, u); // Need to gradually go to zero as speed decreases, BUT DOESN'T FIX PROBLEM
    double sa = sin(alpha);
    double ca = cos(alpha);
    double beta = asin(v/V);
    double sb = sin(beta);
    double cb = cos(beta);

	// Sigmoid function for blending stalled CLplate with below stall CLmodel
	double expMinus = exp(-StallM*(alpha - StallAlphaB));
	double expPlus = exp(StallM*(alpha + StallAlphaB));
	double sigmoid = (1 + expMinus + expPlus) / ((1 + expMinus)*(1 + expPlus));

	// Aero coefficients
	double CLmodel;
	double CS;
	double CDmodel;
	double Cl;
	double CmModel;
	double Cn;
	// Compressibility parameters
	double Lambda_w = 0.40142572795;
	double Lambda_h = 0.40142572795;
	double Lambda_v = 0.45378560551;
	double AR_w = 3.0;
	double AR_h = 2.11;
	double AR_v = 1.29;
	double speedSound = CalculateStdAtmProperties_English(H)[4];
	double machNum = V/speedSound;
	//UE_LOG(LogTemp, Warning, TEXT("machNum = %f"), float(machNum));
	if (bIsBireAircraft)
	{
		// BIRE aircraft store dB in Controls[2] = dr for traditional aircraft
		TArray<double> BireCoefficients = CalculateBIRECoefficients(alpha, beta, pbar, qbar, rbar, da, de, dr);
		CLmodel = BireCoefficients[0];
		CS = BireCoefficients[1];
		CDmodel = BireCoefficients[2];
		Cl = BireCoefficients[3];
		CmModel = BireCoefficients[4];
		Cn = BireCoefficients[5];
		// Apply compressibility correction for BIRE *NOTE: commented out for instability above machNum = 1.08
		CLmodel = CalculateCompressibilityCorrection(CLmodel, Lambda_w, AR_w, machNum);
		CS = CalculateCompressibilityCorrection(CS, Lambda_h, AR_h, machNum);
		CDmodel = CDmodel*(1.0 + 3.0*pow(machNum, 30.0));
		Cl = CalculateCompressibilityCorrection(Cl, Lambda_w, AR_w, machNum);
		CmModel = CalculateCompressibilityCorrection(CmModel, Lambda_w, AR_w, machNum);
		Cn = CalculateCompressibilityCorrection(Cn, Lambda_h, AR_h, machNum);
	}
	else
	{
		double CL1 = CL0 + CLa*alpha;
		double CS1 = CSb*beta;
		CLmodel = CL0 + CLa*alpha + CLqbar*qbar + CLde*de;
		CS = CS1 + (CSLpbar*CL1 + CSpbar)*pbar + CSrbar*rbar + CSda*da + CSdr*dr;
		CDmodel = CD0 + CDL*CL1 + CDL2*CL1*CL1 + (CDL2qbar*CL1*CL1 + CDLqbar*CL1 + CDqbar)*qbar + (CDLde*CL1 + CDde)*de + CDde2*de*de + 
			CDS2*CS1*CS1 + (CDSpbar*CS1 + CDpbar)*pbar + (CDSrbar*CS1 + CDrbar)*rbar + (CDSda*CS1 + CDda)*da + (CDSdr*CS1 + CDdr)*dr;
		Cl = Clb*beta + Clpbar*pbar + (ClLrbar*CL1 + Clrbar)*rbar + Clda*da + Cldr*dr;
		CmModel = Cm0 + Cma*alpha + Cmqbar*qbar + Cmde*de;
		Cn = Cnb*beta + (CnLpbar*CL1 + Cnpbar)*pbar + Cnrbar*rbar + (CnLda*CL1 + Cnda)*da + Cndr*dr;
		// Apply compressibility correction for baseline F16 *NOTE: commented out for instability above machNum = 1.08
		CLmodel = CalculateCompressibilityCorrection(CLmodel, Lambda_w, AR_w, machNum);
		CS = CalculateCompressibilityCorrection(CS, Lambda_v, AR_v, machNum);
		CDmodel = CDmodel*(1.0 + 3.0*pow(machNum, 30.0));
		Cl = CalculateCompressibilityCorrection(Cl, Lambda_v, AR_v, machNum);
		CmModel = CalculateCompressibilityCorrection(CmModel, Lambda_w, AR_w, machNum);
		Cn = CalculateCompressibilityCorrection(Cn, Lambda_v, AR_v, machNum);
	}
	// Apply flat plate stall model to CL,CD,Cm
	double CLplate = 2*copysign(1.0, alpha)*sa*sa*ca;
	double CL = (1 - sigmoid)*CLmodel + sigmoid*CLplate;
    double CDplate = 2*pow(sin(abs(alpha)), 1.5);
	double CD = (1 - sigmoid)*CDmodel + sigmoid*CDplate;
	double CmPlate = -0.8*sa;
	double Cm = (1 - sigmoid)*CmModel + sigmoid*CmPlate;

	// Calculate aero forces
	double rho = CalculateStdAtmProperties_English(H)[3];
	double rhoV2Sw = 0.5*rho*V*V*Sw;
	// FP = propulsive forces
	Eigen::Vector3d FP = Eigen::Vector3d(tau*pow((rho/rho0), ThrustA)*ThrustT0, 0., 0.);
	// MP = propulsive moments
	Eigen::Vector3d MP = ThrustLoc.cross(FP);
	// Fb = body fixed aero forces
	Eigen::Vector3d Fb = Eigen::Vector3d(
		FP(0) + rhoV2Sw*(CL*sa - CS*ca*sb - CD*ca*cb),
		FP(1) + rhoV2Sw*(CS*cb - CD*sb),
		FP(2) + rhoV2Sw*(-CL*ca - CS*sa*sb - CD*sa*cb)
	);
	// M1 = moment modification due to CG shift
	Eigen::Vector3d M1 = CGShift.cross(Fb);
	
	return TArray<double>({
		Fb(0),
		Fb(1),
		Fb(2),
		MP(0) + rhoV2Sw*bw*Cl - M1(0),
		MP(1) + rhoV2Sw*cw*Cm - M1(1),
		MP(2) + rhoV2Sw*bw*Cn - M1(2),
	});
}

Eigen::Matrix<double, 6, 1> AAircraft::CalculateResidual(Eigen::Matrix<double, 6, 1> G, Eigen::Vector3d pqr)
{
	double alpha = G(0);
	double da = G(2);
	double de = G(3);
	double dr = G(4);
	double tau = G(5);
	
	if (bBankProvided)
	{
		double beta = G(1);
		sbInit = sin(beta);
		cbInit = cos(beta);
	}
	else
	{
		double phi = G(1);
		sphiInit = sin(phi);
		cphiInit = cos(phi);
	}
	double saInit = sin(alpha);
	double caInit = cos(alpha);
	double u = VInit*caInit*cbInit;
	double v = VInit*sbInit;
	double w = VInit*saInit*cbInit;

	double p = pqr(0);
	double q = pqr(1);
	double r = pqr(2);

	Eigen::Matrix<double, 13, 1> StatesG = Eigen::Matrix<double, 13, 1>(u, v, w, p, q, r, 0., 0., -altInit, 1., 0., 0., 0.);
	Eigen::Vector4d ControlsG = Eigen::Vector4d(da, de, dr, tau);
	TArray<double> AeroForces = CalculateAeroForces(StatesG, ControlsG);
	double Fx = AeroForces[0];
	double Fy = AeroForces[1];
	double Fz = AeroForces[2];
	double Mx = AeroForces[3];
	double My = AeroForces[4];
	double Mz = AeroForces[5];

	return Eigen::Matrix<double, 6, 1>(
		Fx - W*sthetaInit + Wg*(r*v - q*w),
        Fy + W*sphiInit*cthetaInit + Wg*(p*w - r*u),
        Fz + W*cphiInit*cthetaInit + Wg*(q*u - p*v),
        Mx - hz*q + hy*r + (Iyy - Izz)*q*r + Iyz*(q*q - r*r) + Ixz*p*q - Ixy*p*r,
        My + hz*p - hx*r + (Izz - Ixx)*p*r + Ixz*(r*r - p*p) + Ixy*q*r - Iyz*p*q,
        Mz - hy*p + hx*q + (Ixx - Iyy)*p*q + Ixy*(p*p - q*q) + Iyz*p*r - Ixz*q*r
	);
}

void AAircraft::InitializeFromTrimJacobian()
{
	double stepInv = 0.5/trimStepSize;

	// G = [alpha, beta, da, de, dr, tau] if bBankProvided = True
	// G = [alpha, phi, da, de, dr, tau] if bBankProvided = False
	Eigen::Matrix<double, 6, 1> G = Eigen::Matrix<double, 6, 1>(0., 0., 0., 0., 0., 0.);
	Eigen::Vector3d pqr = Eigen::Vector3d(0., 0., 0.);
	
	int iterCount = 0;

	double RError = 1.;
	while (RError > trimTolerance)
	{
		iterCount += 1;
		
		double alpha = G(0);
		double beta;
		double phi;
		double da = G(2);
		double de = G(3);
		double dr = G(4);
		double tau = G(5);
	
		if (bBankProvided)
		{
			beta = G(1);
			sbInit = sin(beta);
			cbInit = cos(beta);
		}
		else
		{
			phi = G(1);
			sphiInit = sin(phi);
			cphiInit = cos(phi);
		}
		double saInit = sin(alpha);
		double caInit = cos(alpha);
		double u = VInit*caInit*cbInit;
		double v = VInit*sbInit;
		double w = VInit*saInit*cbInit;

		// Use climb angle to determine elevation angle if elevation angle is not provided
		double theta;
		if (!bElevProvided)
		{
			double vswc = v*sphiInit + w*cphiInit;
			double thetaPlus = asin( (u*VInit*sgammaInit + vswc*sqrt(u*u + vswc*vswc - pow(VInit*sgammaInit, 2.))) / (u*u + vswc*vswc) );
			double thetaMinus = asin( (u*VInit*sgammaInit - vswc*sqrt(u*u + vswc*vswc - pow(VInit*sgammaInit, 2.))) / (u*u + vswc*vswc) );
			// Determine which theta (thetaPlus or thetaMinus) is the root that satisfies Eq. 16.11
			double comparison = (u*sin(thetaPlus) - vswc*cos(thetaPlus) - VInit*sgammaInit);
			if ((-1.0e-8 < comparison) && (comparison < 1.0e-8))
			{
				theta = thetaPlus;
			}
			else
			{
				theta = thetaMinus;
			}
			elevInit = theta;
			sthetaInit = sin(theta);
			cthetaInit = cos(theta);
		}

		// Compute rotation rates for steady coordinated turn, or set rotation rates to 0 for steady heading sideslip
		// Steady coordinated turn
		if (TrimType == "sct")
		{
			double pqrFactor = gInit*sphiInit*cthetaInit / (u*cthetaInit*cphiInit + w*sthetaInit);
			pqr = Eigen::Vector3d(
				-pqrFactor*sthetaInit,
				pqrFactor*sphiInit*cthetaInit,
				pqrFactor*cphiInit*cthetaInit
			);
		}
		// Steady heading sideslip
		else
		{
			pqr = Eigen::Vector3d(0., 0., 0.);
		}

		// Calculate 6x6 jacobian matrix of partial derivatives of residuals with respect to G components (dR1/dG1, dR2/dG1, dR2/dG2, etc.)
		Eigen::Matrix<double, 6, 6> J;
		for (int i = 0; i < 6; i++)
		{
			Eigen::Matrix<double, 6, 1> Gi = G;
			Gi(i) += trimStepSize;
			Eigen::Matrix<double, 6, 1> fiPlus = CalculateResidual(Gi, pqr);
			Gi(i) -= 2.*trimStepSize;
			Eigen::Matrix<double, 6, 1> fiMinus = CalculateResidual(Gi, pqr);
			for (int j = 0; j < 6; j++)
			{
				J(j,i) = (fiPlus(j) - fiMinus(j))*stepInv;
			}
		}

		// Use J to estimate new G
		Eigen::Matrix<double, 6, 1> R = CalculateResidual(G, pqr);
		Eigen::Matrix<double, 6, 1> dG = -1.*(J.inverse()*R);
		G += trimRelaxFactor*dG;
		Eigen::Matrix<double, 6, 1> RNew = CalculateResidual(G, pqr);
		RError = std::max(RNew.maxCoeff(), -RNew.minCoeff());

		// Update BIRE inertia for next iteration
		if (bIsBireAircraft)
		{
			AircraftControls(2) = G(4);
			UpdateBIREInertia();
		} 

	}

	// Initialize states and controls from final G iteration
	double alphaFinal = G(0);
	double betaFinal;
	double phiFinal;
	double daFinal = G(2);
	double deFinal = G(3);
	double drFinal = G(4);
	double tauFinal = G(5);
	double bankFinal;
	if (bBankProvided)
	{
		betaFinal = G(1);
		bankFinal = bankInit;
	}
	else
	{
		phiFinal = G(1);
		betaFinal = sideslipInit;
		bankFinal = phiFinal;
	}
	
	double elevFinal = elevInit;
	double uInit = VInit*cos(alphaFinal)*cos(betaFinal);
	double vInit = VInit*sin(betaFinal);
	double wInit = VInit*sin(alphaFinal)*cos(betaFinal);
	pInit = pqr(0);
	qInit = pqr(1);
	rInit = pqr(2);

	// Calculate heading based on ground track (GT) and set initial rotation
	double cBankFinal = cos(bankFinal);
	double sBankFinal = sin(bankFinal);
	double sElevFinal = sin(elevFinal);
	double aGT = cBankFinal*uInit + sBankFinal*sElevFinal*vInit + cBankFinal*sElevFinal*wInit;
	double bGT = sBankFinal*wInit - cBankFinal*vInit;
	double tGroundTrack = tan(groundTrackInit);
	double K1GT = bGT + aGT*tGroundTrack;
	double K2GT = aGT - bGT*tGroundTrack;
	double K3GT = windVect(1) - windVect(0)*tGroundTrack;

	UE_LOG(LogTemp, Warning, TEXT("\nK1 = %e\n"), K1GT);
	UE_LOG(LogTemp, Warning, TEXT("\nK2 = %e\n"), K2GT);
	UE_LOG(LogTemp, Warning, TEXT("\nK3 = %e\n"), K3GT);

	double headingPlus = asin((-K2GT*K3GT + K1GT*sqrt(K1GT*K1GT + K2GT*K2GT - K3GT*K3GT)) / (K1GT*K1GT + K2GT*K2GT));
	double headingMinus = asin((-K2GT*K3GT - K1GT*sqrt(K1GT*K1GT + K2GT*K2GT - K3GT*K3GT)) / (K1GT*K1GT + K2GT*K2GT));
	// Determine which heading (headingPlus or headingMinus) is the root that satisfies Eq. 16.17
	double headingComparison = K2GT*sin(headingPlus) + K3GT - K1GT*cos(headingPlus);
	if ((-1.0e-8 < headingComparison) && (headingComparison < 1.0e-8))
	{
		headingInit = headingPlus;
	}
	else
	{
		headingInit = headingMinus;
	}
	TArray<double> quatInit = QuatNormalize(EulerToQuat(TArray<double>({bankInit, elevInit, headingInit})));

	UE_LOG(LogTemp, Warning, TEXT("\nheadingPlus = %e\n"), 180./PI*headingPlus);
	UE_LOG(LogTemp, Warning, TEXT("\nheadingMinus = %e\n"), 180./PI*headingMinus);
	UE_LOG(LogTemp, Warning, TEXT("\nheading = %e\n"), 180./PI*headingInit);
	UE_LOG(LogTemp, Warning, TEXT("\ntrimCount = %d\n"), iterCount);
	
	UE_LOG(LogTemp, Warning, TEXT("Final trim solution: \nelevation_angle[deg] = %E\nbank_angle[deg] = %E\nalpha[deg] = %E\nbeta[deg] = %E\np[deg/s] = %E\nq[deg/s] = %E\nr[deg/s] = %E\naileron[deg] = %E\nelevator[deg] = %E\nrudder[deg] = %E\nthrottle = %E"), 
		(180./PI*elevFinal),
		(180./PI*bankFinal),
		(180./PI*alphaFinal),
		(180./PI*betaFinal),
		(180./PI*pInit),
		(180./PI*qInit),
		(180./PI*rInit),
		(180./PI*daFinal),
		(180./PI*deFinal),
		(180./PI*drFinal),
		(tauFinal)
	);

	// AircraftStates = (u, v, w,   p, q, r,   xf, yf, zf,   e0, ex, ey, ez)
	AircraftStates = Eigen::Matrix<double, 13, 1>(
		uInit, vInit, wInit,
		pInit, qInit, rInit,
		0., 0., -altInit,
		quatInit[0], quatInit[1], quatInit[2], quatInit[3]
	);

	// AircraftControls = (da, de, dr, tau)
	AircraftControls = Eigen::Vector4d(daFinal, deFinal, drFinal, tauFinal);

	UE_LOG(LogTemp, Warning, TEXT("Final trim states: \nu = %f\nv = %f\nw = %f\np = %f\nq = %f\nr = %f\nx = %f\ny = %f\nz = %f\ne0 = %f\nex = %f\ney = %f\nez = %f"), 
		float(uInit), float(vInit), float(wInit), 
		float(pInit), float(qInit), float(rInit), 
		0.f, 0.f, float(-altInit), 
		float(quatInit[0]), float(quatInit[1]), float(quatInit[2]), float(quatInit[3])
	);

}

void AAircraft::InitializeFromTrimFixedPoint()
{
	// G = [alpha, beta, da, de, dr, tau] because bBankProvided = True if using fixed-point iteration
	
	if (bBankProvided)
	{
		Eigen::Matrix<double, 6, 1> G = Eigen::Matrix<double, 6, 1>(0., 0., 0., 0., 0., 0.);
		double p;
		double q;
		double r;
		double rho = CalculateStdAtmProperties_English(altInit)[3];

		int count = 0;
			
		double trimError = 1.;
		while (trimError > trimTolerance)
		{
			double alpha = G(0);
			double beta = G(1);
			double da = G(2);
			double de = G(3);
			double dr = G(4);
			double tau = G(5);
		
			sbInit = sin(beta);
			cbInit = cos(beta);
			double saInit = sin(alpha);
			double caInit = cos(alpha);
			double u = VInit*caInit*cbInit;
			double v = VInit*sbInit;
			double w = VInit*saInit*cbInit;

			// Use climb angle to determine elevation angle if elevation angle is not provided
			double theta;
			if (!bElevProvided)
			{
				double vswc = v*sphiInit + w*cphiInit;
				double thetaPlus = asin( (u*VInit*sgammaInit + vswc*sqrt(u*u + vswc*vswc - pow(VInit*sgammaInit, 2.))) / (u*u + vswc*vswc) );
				double thetaMinus = asin( (u*VInit*sgammaInit - vswc*sqrt(u*u + vswc*vswc - pow(VInit*sgammaInit, 2.))) / (u*u + vswc*vswc) );
				// Determine which theta (thetaPlus or thetaMinus) is the root that satisfies Eq. 16.11
				double comparison = (u*sin(thetaPlus) - vswc*cos(thetaPlus) - VInit*sgammaInit);
				if ((-1.0e-8 < comparison) && (comparison < 1.0e-8))
				{
					theta = thetaPlus;
				}
				else
				{
					theta = thetaMinus;
				}
				elevInit = theta;
				sthetaInit = sin(theta);
				cthetaInit = cos(theta);
			}

			// Compute rotation rates for steady coordinated turn, or set rotation rates to 0 for steady heading sideslip
			// Steady coordinated turn
			if (TrimType == "sct")
			{
				double pqrFactor = gInit*sphiInit*cthetaInit / (u*cthetaInit*cphiInit + w*sthetaInit);
				p = -pqrFactor*sthetaInit;
				q = pqrFactor*sphiInit*cthetaInit;
				r = pqrFactor*cphiInit*cthetaInit;
			}
			// Steady heading sideslip
			else
			{
				p = 0.;
				q = 0.;
				r = 0.;
			}

			// Use fixed-point iteration to calculate new G
			Eigen::Matrix<double, 13, 1> StatesG = Eigen::Matrix<double, 13, 1>(u, v, w, p, q, r, 0., 0., -altInit, 1., 0., 0., 0.);
			Eigen::Vector4d ControlsG = Eigen::Vector4d(da, de, dr, tau);
			TArray<double> AeroForcesG = CalculateAeroForces(StatesG, ControlsG);
			double Fx = AeroForcesG[0];
			double Fy = AeroForcesG[1];
			double Fz = AeroForcesG[2];
			double Mx = AeroForcesG[3];
			double My = AeroForcesG[4];
			double Mz = AeroForcesG[5];

			double V = sqrt(u*u + v*v + w*w);
			double rhoV2Sw = 0.5*rho*V*V*Sw;

			// dG = [dAlpha, dBeta, dDa, dDe, dDr, dTau]
			Eigen::Matrix<double, 6, 1> dG = Eigen::Matrix<double, 6, 1>(
				-(Fz + W*cphiInit*cthetaInit + (q*u - p*v)*Wg) / (rhoV2Sw*CLa*caInit),
				(Fy + W*sphiInit*cthetaInit + (p*w - r*u)*Wg) / (rhoV2Sw*CSb*cbInit),
				(Mx - hz*q + hy*r + (Iyy - Izz)*q*r + Iyz*(q*q - r*r) + Ixz*p*q - Ixy*p*r) / (rhoV2Sw*bw*Clda),
				(My + hz*p - hx*r + (Izz - Ixx)*p*r + Ixz*(r*r - p*p) + Ixy*q*r - Iyz*p*q) / (rhoV2Sw*cw*Cmde),
				(Mz - hy*p + hx*q + (Ixx - Iyy)*p*q + Ixy*(p*p - q*q) + Iyz*p*r - Ixz*q*r) / (rhoV2Sw*bw*Cndr),
				(Fx - W*sthetaInit + (r*v - q*w)*Wg) / (pow((rho/rho0), ThrustA)*(ThrustT0 + ThrustT1*V + ThrustT2*V*V))
				);
			// Calculate new G based on previous iteration
			G = G - trimRelaxFactor*dG;
			// Find current maximum error to determine if algorithm is converged
			trimError = std::max(dG.maxCoeff(), -dG.minCoeff());
			count += 1;
			if (count > 100000)
			{
				trimError = 0.;
			}
		
		}

		// Initialize states and controls from final G iteration
		double alphaFinal = G(0);
		double betaFinal = G(1);
		double daFinal = G(2);
		double deFinal = G(3);
		double drFinal = G(4);
		double tauFinal = G(5);
		double bankFinal = bankInit;
		double elevFinal = elevInit;
		double uInit = VInit*cos(alphaFinal)*cos(betaFinal);
		double vInit = VInit*sin(betaFinal);
		double wInit = VInit*sin(alphaFinal)*cos(betaFinal);
		pInit = p;
		qInit = q;
		rInit = r;

		// Calculate heading based on ground track (GT) and set initial rotation
		double cBankFinal = cos(bankFinal);
		double sBankFinal = sin(bankFinal);
		double sElevFinal = sin(elevFinal);
		double aGT = cBankFinal*uInit + sBankFinal*sElevFinal*vInit + cBankFinal*sElevFinal*wInit;
		double bGT = sBankFinal*wInit - cBankFinal*vInit;
		double tGroundTrack = tan(groundTrackInit);
		double K1GT = bGT + aGT*tGroundTrack;
		double K2GT = aGT - bGT*tGroundTrack;
		double K3GT = windVect(1) - windVect(0)*tGroundTrack;
		double headingPlus = asin((-K2GT*K3GT + K1GT*sqrt(K1GT*K1GT + K2GT*K2GT - K3GT*K3GT)) / (K1GT*K1GT + K2GT*K2GT));
		double headingMinus = asin((-K2GT*K3GT - K1GT*sqrt(K1GT*K1GT + K2GT*K2GT - K3GT*K3GT)) / (K1GT*K1GT + K2GT*K2GT));
		// Determine which heading (headingPlus or headingMinus) is the root that satisfies Eq. 16.17
		double headingComparison = K2GT*sin(headingPlus) + K3GT - K1GT*sin(headingPlus);
		if ((-1.0e-8 < headingComparison) && (headingComparison < 1.0e-8))
		{
			headingInit = headingPlus;
		}
		else
		{
			headingInit = headingMinus;
		}
		TArray<double> quatInit = QuatNormalize(EulerToQuat(TArray<double>({bankInit, elevInit, headingInit})));

		UE_LOG(LogTemp, Warning, TEXT("Final trim solution: \nelevation_angle[deg] = %E\nbank_angle[deg] = %E\nalpha[deg] = %E\nbeta[deg] = %E\np[deg/s] = %E\nq[deg/s] = %E\nr[deg/s] = %E\naileron[deg] = %E\nelevator[deg] = %E\nrudder[deg] = %E\nthrottle = %E"), 
			(180./PI*elevFinal),
			(180./PI*bankFinal),
			(180./PI*alphaFinal),
			(180./PI*betaFinal),
			(180./PI*pInit),
			(180./PI*qInit),
			(180./PI*rInit),
			(180./PI*daFinal),
			(180./PI*deFinal),
			(180./PI*drFinal),
			(tauFinal)
		);

		// AircraftStates = (u, v, w,   p, q, r,   xf, yf, zf,   e0, ex, ey, ez)
		AircraftStates = Eigen::Matrix<double, 13, 1>(
			uInit, vInit, wInit,
			pInit, qInit, rInit,
			0., 0., -altInit,
			quatInit[0], quatInit[1], quatInit[2], quatInit[3]
		);

		// AircraftControls = (da, de, dr, tau)
		AircraftControls = Eigen::Vector4d(daFinal, deFinal, drFinal, tauFinal);

		UE_LOG(LogTemp, Warning, TEXT("Final trim states: \nu = %f\nv = %f\nw = %f\np = %f\nq = %f\nr = %f\nx = %f\ny = %f\nz = %f\ne0 = %f\nex = %f\ney = %f\nez = %f"), 
			float(uInit), float(vInit), float(wInit), 
			float(pInit), float(qInit), float(rInit), 
			0.f, 0.f, float(-altInit), 
			float(quatInit[0]), float(quatInit[1]), float(quatInit[2]), float(quatInit[3])
		);
	}
	else
	{
		UE_LOG(LogTemp, Error, TEXT("Could not trim aircraft using fixed-point iteration. Must prescribe climb_angle and bank_angle in config .json"))
	}
	
}

void AAircraft::InitializeFromState()
{
	TArray<double> quatInit = QuatNormalize(EulerToQuat(TArray<double>({bankInit, elevInit, headingInit})));
	double uInit = VInit*cos(alphaInit)*cos(betaInit);
	double vInit = VInit*sin(betaInit);
	double wInit = VInit*sin(alphaInit)*cos(betaInit);

	// AircraftStates = (u, v, w,   p, q, r,   xf, yf, zf,   e0, ex, ey, ez)
	AircraftStates = Eigen::Matrix<double, 13, 1>(
		uInit, vInit, wInit,
		pInit, qInit, rInit,
		0., 0., -altInit,
		quatInit[0], quatInit[1], quatInit[2], quatInit[3]
	);

	// AircraftControls = (da, de, dr, tau)
	AircraftControls = Eigen::Vector4d(daInit, deInit, drInit, tauInit);

}

void AAircraft::GetAircraftStatesUE(float &u, float &v, float &w, float &p, float &q, float &r, float &xf, float &yf, float &zf, float &e0, float &ex, float &ey, float &ez)
{
	// AircraftStates = (u, v, w,   p, q, r,   xf, yf, zf,   e0, ex, ey, ez)
	u = 30.48*float(AircraftStates(0));
	v = 30.48*float(AircraftStates(1));
	w = 30.48*float(AircraftStates(2));
	p = float(AircraftStates(3));
	q = float(AircraftStates(4));
	r = float(AircraftStates(5));
	xf = 30.48*float(AircraftStates(6));
	yf = 30.48*float(AircraftStates(7));
	zf = 30.48*float(AircraftStates(8));
	e0 = float(AircraftStates(9));
	ex = float(AircraftStates(10));
	ey = float(AircraftStates(11));
	ez = float(AircraftStates(12));

}

void AAircraft::GetAircraftControls(float &da, float &de, float &dr, float &tau)
{
	// AircraftControls = (da, de, dr, tau)
	da = float(AircraftControls(0));
	de = float(AircraftControls(1));
	dr = float(AircraftControls(2));
	tau = float(AircraftControls(3));
}

void AAircraft::SetAircraftControls(float da, float de, float dr, float tau)
{
	// AircraftControls = (da, de, dr, tau)
	AircraftControls(0) = double(da);
	AircraftControls(1) = double(de);
	AircraftControls(2) = double(dr);
	AircraftControls(3) = double(tau);
}

void AAircraft::SetIsBireAircraft(bool IsBireAircraft)
{
	bIsBireAircraft = IsBireAircraft;
}

void AAircraft::GetLatitudeLongitudeDeg(float &Latitude, float &Longitude, float& alt)
{
	Latitude = float(180./Pi*latitude);
	Longitude = float(180./Pi*longitude);
	alt = altInit;
}

void AAircraft::LLAtoECEF(double &x, double &y, double &z, float lat, float lon, float alt)
{
	double f = 1 / 128.257223563;
	double R = 6378137;

	double h = altInit / 3.281;

	double Lat = lat * Pi / 180;
	double Lon = lon * Pi / 180;

	alt = alt / 3.281;

	double lambdaS = atan(pow((1 - f), 2) * tan(Lat));
	double SinlambdaS = sin(lambdaS);
	double rs = sqrt(pow(R, 2) / (1 + (1 / pow((1 - f), 2) - 1) * pow(SinlambdaS, 2)));

	x = (rs * cos(lambdaS) * cos(Lon) + alt * cos(Lat) * cos(Lon)) * 100;
	y = (rs * cos(lambdaS) * sin(Lon) + alt * cos(Lat) * sin(Lon)) * 100;
	z = (rs * sin(lambdaS) + alt * sin(Lat)) * 100;
}

void AAircraft::GetEulerAngles(float& BankAngle, float& ElevationAngle, float& AzimuthAngle)
{
	TArray<double> orientationQuat = TArray<double>({
		AircraftStates(9),
		AircraftStates(10),
		AircraftStates(11),
		AircraftStates(12)
	});
	TArray<double> orientationEuler = QuatToEuler(orientationQuat);
	
	BankAngle = orientationEuler[0];
	ElevationAngle = orientationEuler[1];
	AzimuthAngle = orientationEuler[2];
}

void AAircraft::GetGlobalVelocityVector(float& u, float& v, float& w)
{
	TArray<double> uvwBody = TArray<double>({
		AircraftStates(0),
		AircraftStates(1),
		AircraftStates(2)
	});
	TArray<double> AircraftQuat = TArray<double>({
		AircraftStates(9),
		AircraftStates(10),
		AircraftStates(11),
		AircraftStates(12)
	});
	TArray<double> uvwEarth = BodyToEarthFixed(uvwBody, AircraftQuat);
	
	u = uvwEarth[0];
	v = uvwEarth[1];
	w = uvwEarth[2];
}

void AAircraft::GetMachNumber(float& MachNumber)
{
	double u = AircraftStates(0);
	double v = AircraftStates(1);
	double w = AircraftStates(2);
	double speedSound = CalculateStdAtmProperties_English(-AircraftStates(8))[4];
	double V = sqrt(u * u + v * v + w * w);
	double machNum = V / speedSound;
	
	MachNumber = float(machNum);
}

void AAircraft::TickAircraftStates(float DeltaTime)
{
	// Store previous xf, yf, zf for TickLatitudeLongitude()'s dx, dy, dz
	double xf0 = AircraftStates(6);
	double yf0 = AircraftStates(7);
	double zf0 = AircraftStates(8);

	// Update current inertia based on current BIRE angle (for BIRE only)
	if (bIsBireAircraft)
	{
		UpdateBIREInertia();
	}
	
	// Update AircraftStates by integrating the aircraft dynamics forward DeltaTime seconds
	double t0 = GetGameTimeSinceCreation();
	AircraftStates = IntegrateStates_RK4(t0, AircraftStates, AircraftControls, DeltaTime);

	// Update latitude and longitude
	double dfx = AircraftStates(6) - xf0;
	double dfy = AircraftStates(7) - yf0;
	double dfz = AircraftStates(8) - zf0;
	TickLatitudeLongitude(dfx, dfy, dfz, -zf0);

	// Renormalize aircraft rotation quaternion to reduce integration error
	TArray<double> AircraftQuat = TArray<double>({
		AircraftStates(9),
		AircraftStates(10),
		AircraftStates(11),
		AircraftStates(12)
	});
	TArray<double> NormalizedAircraftQuat = QuatNormalize(AircraftQuat);
	AircraftStates(9) = NormalizedAircraftQuat[0];
	AircraftStates(10) = NormalizedAircraftQuat[1];
	AircraftStates(11) = NormalizedAircraftQuat[2];
	AircraftStates(12) = NormalizedAircraftQuat[3];
}

bool AAircraft::IsAircraftStalled()
{
	double u = AircraftStates(0);
	double w = AircraftStates(2);
	double alpha = atan2(w, u);
	
	return (alpha > StallAlphaB);
}

double AAircraft::FitBIRECoefficient(TArray<double> BireCoeffArray, double dB)
{
	double A = BireCoeffArray[0];
	double w = BireCoeffArray[1];
	double phi = BireCoeffArray[2];
	double z = BireCoeffArray[3];
	double delta = BireCoeffArray[4];
	double multiplier = BireCoeffArray[5];
	
	return multiplier*(A*sin(w*dB + phi) + z + delta);
}

TArray<double> AAircraft::CalculateBIRECoefficients(double alpha, double beta, double pbar, double qbar, double rbar, double da, double de, double dB)
{
	double BireCL0 = FitBIRECoefficient(BCL0, dB);
	double BireCLa = FitBIRECoefficient(BCLa, dB);
	double BireCLb = FitBIRECoefficient(BCLb, dB);
	double BireCLpbar = FitBIRECoefficient(BCLpbar, dB);
	double BireCLqbar = FitBIRECoefficient(BCLqbar, dB);
	double BireCLrbar = FitBIRECoefficient(BCLrbar, dB);
	double BireCLda = FitBIRECoefficient(BCLda, dB);
	double BireCLde = FitBIRECoefficient(BCLde, dB);
	
	double BireCL1 = BireCL0 + BireCLa*alpha;
	double BCL = BireCL1 + BireCLb*beta + BireCLpbar*pbar + BireCLqbar*qbar + BireCLrbar*rbar + BireCLda*da + BireCLde*de;

	double BireCS0 = FitBIRECoefficient(BCS0, dB);
	double BireCSa = FitBIRECoefficient(BCSa, dB);
	double BireCSb = FitBIRECoefficient(BCSb, dB);
	double BireCSpbar = FitBIRECoefficient(BCSpbar, dB);
	double BireCSLpbar = FitBIRECoefficient(BCSLpbar, dB);
	double BireCSqbar = FitBIRECoefficient(BCSqbar, dB);
	double BireCSrbar = FitBIRECoefficient(BCSrbar, dB);
	double BireCSda = FitBIRECoefficient(BCSda, dB);
	double BireCSde = FitBIRECoefficient(BCSde, dB);

	double BireCS1 = BireCS0 + BireCSb*beta;
	double BCS = BireCS0 + BireCSa*alpha + BireCSb*beta + (BireCSpbar + BireCSLpbar*BireCL1)*pbar + BireCSqbar*qbar + BireCSrbar*rbar + BireCSda*da + BireCSde*de;
	
	double BireCD0 = FitBIRECoefficient(BCD0, dB);
	double BireCDL = FitBIRECoefficient(BCDL, dB);
	double BireCDL2 = FitBIRECoefficient(BCDL2, dB);
	double BireCDS = FitBIRECoefficient(BCDS, dB);
	double BireCDS2 = FitBIRECoefficient(BCDS2, dB);
	double BireCDpbar = FitBIRECoefficient(BCDpbar, dB);
	double BireCDSpbar = FitBIRECoefficient(BCDSpbar, dB);
	double BireCDqbar = FitBIRECoefficient(BCDqbar, dB);
	double BireCDLqbar = FitBIRECoefficient(BCDLqbar, dB);
	double BireCDL2qbar = FitBIRECoefficient(BCDL2qbar, dB);
	double BireCDrbar = FitBIRECoefficient(BCDrbar, dB);
	double BireCDSrbar = FitBIRECoefficient(BCDSrbar, dB);
	double BireCDda = FitBIRECoefficient(BCDda, dB);
	double BireCDSda = FitBIRECoefficient(BCDSda, dB);
	double BireCDde = FitBIRECoefficient(BCDde, dB);
	double BireCDLde = FitBIRECoefficient(BCDLde, dB);
	double BireCDde2 = FitBIRECoefficient(BCDde2, dB);

	double BCD = BireCD0 + BireCDL*BireCL1 + BireCDL2*BireCL1*BireCL1 + (BireCDqbar + BireCDLqbar*BireCL1 + BireCDL2qbar*BireCL1*BireCL1)*qbar + (BireCDde + BireCDLde*BireCL1)*de + BireCDde2*de*de +
		BireCDS*BireCS1 + BireCDS2*BireCS1*BireCS1 + (BireCDpbar + BireCDSpbar*BireCS1)*pbar + (BireCDSrbar*BireCS1 + BireCDrbar)*rbar + (BireCDSda*BireCS1 + BireCDda)*da;

	double BireCl0 = FitBIRECoefficient(BCl0, dB);
	double BireCla = FitBIRECoefficient(BCla, dB);
	double BireClb = FitBIRECoefficient(BClb, dB);
	double BireClpbar = FitBIRECoefficient(BClpbar, dB);
	double BireClqbar = FitBIRECoefficient(BClqbar, dB);
	double BireClrbar = FitBIRECoefficient(BClrbar, dB);
	double BireClLrbar = FitBIRECoefficient(BClLrbar, dB);
	double BireClda = FitBIRECoefficient(BClda, dB);
	double BireClde = FitBIRECoefficient(BClde, dB);

	double BCl = BireCl0 + BireCla*alpha + BireClb*beta + BireClpbar*pbar + BireClqbar*qbar + (BireClrbar + BireClLrbar*BireCL1)*rbar + BireClda*da + BireClde*de;

	double BireCm0 = FitBIRECoefficient(BCm0, dB);
	double BireCma = FitBIRECoefficient(BCma, dB);
	double BireCmb = FitBIRECoefficient(BCmb, dB);
	double BireCmpbar = FitBIRECoefficient(BCmpbar, dB);
	double BireCmqbar = FitBIRECoefficient(BCmqbar, dB);
	double BireCmrbar = FitBIRECoefficient(BCmrbar, dB);
	double BireCmda = FitBIRECoefficient(BCmda, dB);
	double BireCmde = FitBIRECoefficient(BCmde, dB);

	double BCm = BireCm0 + BireCma*alpha + BireCmb*beta + BireCmpbar*pbar + BireCmqbar*qbar + BireCmrbar*rbar + BireCmda*da + BireCmde*de;

	double BireCn0 = FitBIRECoefficient(BCn0, dB);
	double BireCna = FitBIRECoefficient(BCna, dB);
	double BireCnb = FitBIRECoefficient(BCnb, dB);
	double BireCnpbar = FitBIRECoefficient(BCnpbar, dB);
	double BireCnLpbar = FitBIRECoefficient(BCnLpbar, dB);
	double BireCnqbar = FitBIRECoefficient(BCnqbar, dB);
	double BireCnrbar = FitBIRECoefficient(BCnrbar, dB);
	double BireCnda = FitBIRECoefficient(BCnda, dB);
	double BireCnLda = FitBIRECoefficient(BCnLda, dB);
	double BireCnde = FitBIRECoefficient(BCnde, dB);

	double BCn = BireCn0 + BireCna*alpha + BireCnb*beta + (BireCnpbar + BireCnLpbar*BireCL1)*pbar + BireCnqbar*qbar + BireCnrbar*rbar + 
		(BireCnda + BireCnLda*BireCL1)*da + BireCnde*de;
	
	return TArray<double>({BCL, BCS, BCD, BCl, BCm, BCn});
}

void AAircraft::TickLatitudeLongitude(float dx, float dy, float dz, float H1)
{
	double d = sqrt(dx*dx + dy*dy);
	if (d > 3e-16)
	{
		double Phi1 = latitude;
		double sPhi1 = sin(Phi1);
		double cPhi1 = cos(Phi1);

		double Psi1 = longitude;

		double Theta = d / (RE_ft + H1 - dz/2.);
		double sTheta = sin(Theta);
		double cTheta = cos(Theta);

		double psiG1 = atan2(dy, dx);
		double spsiG1 = sin(psiG1);
		double cpsiG1 = cos(psiG1);

		double xHat = cPhi1*cTheta - sPhi1*sTheta*cpsiG1;
		double yHat = sTheta*spsiG1;
		double zHat = sPhi1*cTheta + cPhi1*sTheta*cpsiG1;
		double rHat = sqrt(xHat*xHat + yHat*yHat);

		latitude = atan2(zHat, rHat);
		longitude = Psi1 + atan2(yHat, xHat);
	}
}

double AAircraft::CalculateCompressibilityCorrection(double Coeff, double Lambda, double AspectRatio, double MachNum)
{
	double numerator = Coeff*cos(Lambda);
	double numPiAR = numerator/(PI*AspectRatio);
	double denominator = sqrt(1.0 - pow(MachNum*cos(Lambda), 2) + numPiAR*numPiAR) + numPiAR;

	return numerator/denominator;
}

void AAircraft::UpdateGustState(double t)
{	
	for (int i = 0; i < 3; i++)
	{
		if (t > GustStartTimes(i))
		{
			GustAmplitude(i) = double(FMath::RandRange(float(-gustMagnitude), float(gustMagnitude))) * gustScales(i);
			GustOmega(i) = 2.0 * PI / double(FMath::RandRange(1.0f, 5.0f));
			GustLambda(i) = log(2.0) / double(FMath::RandRange(0.05f, 0.5f));
			GustDelay(i) = double(FMath::RandRange(1.0f, 3.0f));
			double gustDuration = -log(0.01) / GustLambda(i);
			GustPreviousStartTimes(i) = GustStartTimes(i);
			GustStartTimes(i) = t + gustDuration + GustDelay(i);
		}
		double timeSinceStart = t - GustPreviousStartTimes(i);
		double gustConst = GustAmplitude(i) * exp(-GustLambda(i) * timeSinceStart);
		GustVelocity(i) = gustConst * sin(GustOmega(i) * timeSinceStart);
		GustAcceleration(i) = gustConst * (GustOmega(i) * cos(GustOmega(i) * timeSinceStart) - GustLambda(i) * sin(GustOmega(i) * timeSinceStart));
	}
}

void AAircraft::GetGustVelocity(float& Vgx, float& Vgy, float& Vgz)
{
	Vgx = float(GustVelocity(0));
	Vgy = float(GustVelocity(1));
	Vgz = float(GustVelocity(2));
}

void AAircraft::GetCalibratedAirspeed(float& CAS)
{
	TArray<double> seaLevelAtmProperties = CalculateStdAtmProperties_English(0.0);
	TArray<double> currentAtmProperties = CalculateStdAtmProperties_English(-AircraftStates(8));

	double a0 = seaLevelAtmProperties[4];
	double P0 = seaLevelAtmProperties[2];
	double a = currentAtmProperties[4];
	double P = currentAtmProperties[2];
	
	double u = AircraftStates(0);
	double v = AircraftStates(1);
	double w = AircraftStates(2);
	double V = sqrt(u*u + v*v + w*w);

	double M = V / a;
	double qc = P * (pow((1 + 0.2 * M * M), 3.5) - 1.0);
	CAS = a0 * sqrt(5 * (pow((qc / P0 + 1.0), (2.0 / 7.0)) - 1.0));

}

void AAircraft::LogStates(Eigen::Matrix<double, 13, 1> state, FString name)
{
	UE_LOG(LogTemp, Warning, TEXT("%s = {%f\t %f\t %f\t %f\t %f\t %f\t %f\t %f\t %f\t %f\t %f\t %f\t %f\t}"), *name, state(0), state(1), state(2), state(3), state(4), state(5), state(6), state(7), state(8), state(9), state(10), state(11), state(12))
}

void AAircraft::UpdateBIREInertia()
{
	double dB = AircraftControls(2);
	Iyy = IyyInit - dIBIRE * cos(2. * dB);
	Izz = IzzInit + dIBIRE * cos(2. * dB);
	Iyz = IyzInit - dIBIRE * abs(sin(2. * dB));

	Eigen::Matrix3d I;
	I << Ixx, -Ixy, -Ixz,
		-Ixy, Iyy, -Iyz,
		-Ixz, -Iyz, Izz;
	IInv = I.inverse();
	
}


TArray<double> AAircraft::QuatMult(TArray<double> qA, TArray<double> qB)
{
	double A0 = qA[0];
	double Ax = qA[1];
	double Ay = qA[2];
	double Az = qA[3];
	
	double B0 = qB[0];
	double Bx = qB[1];
	double By = qB[2];
	double Bz = qB[3];
	
	return TArray<double>({
		A0*B0 - Ax*Bx - Ay*By - Az*Bz,
        A0*Bx + Ax*B0 + Ay*Bz - Az*By,
        A0*By - Ax*Bz + Ay*B0 + Az*Bx,
        A0*Bz + Ax*By - Ay*Bx + Az*B0
	});
}

TArray<double> AAircraft::EulerToQuat(TArray<double> Euler)
{
	double halfPhi = Euler[0] * 0.5;
    double halfTheta = Euler[1] * 0.5;
    double halfPsi = Euler[2] * 0.5;
    double SPhi = sin(halfPhi);
    double CPhi = cos(halfPhi);
    double STheta = sin(halfTheta);
    double CTheta = cos(halfTheta);
    double SPsi = sin(halfPsi);
    double CPsi = cos(halfPsi);
    double CCPhiTheta = CPhi*CTheta;
    double CSPhiTheta = CPhi*STheta;
    double SCPhiTheta = SPhi*CTheta;
    double SSPhiTheta = SPhi*STheta;
	
	return TArray<double>({
		CCPhiTheta*CPsi + SSPhiTheta*SPsi,
		SCPhiTheta*CPsi - CSPhiTheta*SPsi,
		CSPhiTheta*CPsi + SCPhiTheta*SPsi,
		CCPhiTheta*SPsi - SSPhiTheta*CPsi 
	});
}

TArray<double> AAircraft::QuatToEuler(TArray<double> Quat)
{
	double e0 = Quat[0];
	double ex = Quat[1];
	double ey = Quat[2];
	double ez = Quat[3];

	// Check to determine gimbal lock
	double check = e0*ey - ex*ez;
	if (abs(check) != 0.5)
	{
		return TArray<double>({
			atan2(2.*(e0*ex + ey*ez), (e0*e0 - ex*ex - ey*ey + ez*ez)),
            asin(2.*(e0*ey - ex*ez)),
            atan2(2.*(e0*ez + ex*ey), (e0*e0 + ex*ex - ey*ey - ez*ez))
		});
	}
	else if (check == 0.5)
	{
		return TArray<double>({
			2.*asin(ex/cos(0.25*Pi)),
            0.5*Pi,
            0.0
		});
	}
	else
	{
		return TArray<double>({
			2.*asin(ex/cos(0.25*Pi)),
            -0.5*Pi,
            0.0
		});
	}
}

TArray<double> AAircraft::BodyToEarthFixed(TArray<double> BodyFixed, TArray<double> Quat)
{
	double vbx = BodyFixed[0];
	double vby = BodyFixed[1];
	double vbz = BodyFixed[2];

	double e0 = Quat[0];
	double ex = Quat[1];
	double ey = Quat[2];
	double ez = Quat[3];

	TArray<double> quatProduct = QuatMult(Quat, QuatMult(TArray<double>({0., vbx, vby, vbz}), TArray<double>({e0, -ex, -ey, -ez})));
	
	return TArray<double>({
		quatProduct[1], 
		quatProduct[2], 
		quatProduct[3]
	});
}

TArray<double> AAircraft::EarthToBodyFixed(TArray<double> EarthFixed, TArray<double> Quat)
{
	double vex = EarthFixed[0];
	double vey = EarthFixed[1];
	double vez = EarthFixed[2];
	
	double e0 = Quat[0];
	double ex = Quat[1];
	double ey = Quat[2];
	double ez = Quat[3];
	
	TArray<double> quatProduct = QuatMult(TArray<double>({e0, -ex, -ey, -ez}), QuatMult(TArray<double>({0., vex, vey, vez}), Quat));
	
	return TArray<double>({
		quatProduct[1], 
		quatProduct[2], 
		quatProduct[3]
	});
}

TArray<double> AAircraft::QuatNormalize(TArray<double> Quat)
{
	double e0 = Quat[0];
	double ex = Quat[1];
	double ey = Quat[2];
	double ez = Quat[3];
	
	double magnInv = pow((e0*e0 + ex*ex + ey*ey + ez*ez), -0.5);

	return TArray<double>({
		e0*magnInv,
		ex*magnInv,
		ey*magnInv,
		ez*magnInv
	});
}

void AAircraft::OutputStabilityModeProperties()
{
	// Disable physics simulation and retrim aircraft
	AAircraft::SetActorTickEnabled(false);
	AAircraft::InitializeAircraftFromJSON("F16_MUx_Adjusted.json");

	// Approximate and save force and moment derivatives
	double du = 1.0;
	double dv = 0.5 * du;
	double dw = du;
	double dp = 0.06;
	double dq = 0.5 * dp;
	double dr = 0.5 * dp;
	TArray<double> deltas = TArray<double>({ du, dv, dw, dp, dq, dr });
	CalculateForceMomentDerivatives(deltas);

	// Solve eigensystem and save eigenvalues to file
	Eigen::EigenSolver<Eigen::Matrix<double, 12, 12>> stabilityEigensystemSolution = SolveStabilityEigensystem();

	// For each eigenvalue, save the corresponding mode properties
	FString aircraftName = "_F16";
	if (bIsBireAircraft)
	{
		aircraftName = "_BIRE";
	}
	FString bankCGStr = FString::Printf(TEXT("_b%i_cg%i"), int(180./Pi*bankInit), int(100.0 * CGShift(0)));
	FString outputCaseFilename = "modeProp" + aircraftName + bankCGStr;
	FString outputPathFString = FPaths::ProjectContentDir() + "StabilityOutput/" + outputCaseFilename + ".txt";
	UE_LOG(LogTemp, Warning, TEXT("%s"), *outputPathFString)
	const char* outputPathCharPtr = StringCast<ANSICHAR>(*outputPathFString).Get();
	std::ofstream modeOutputFile;
	modeOutputFile.open(outputPathCharPtr);
	modeOutputFile << "Eigensystem Solution Mode Properties\n";
	modeOutputFile << StringCast<ANSICHAR>(*outputCaseFilename).Get();
	
	for (int i = 0; i < 12; i++)
	{
		modeOutputFile << "\n\n---------------------------------------------------------------------";
		
		std::complex<double> eigenvalue = stabilityEigensystemSolution.eigenvalues()[i];
		modeOutputFile << "\nEigenvalue [1/s] = " << eigenvalue;

		// Determine which state is dominant for each group: uvw, pqr, xyz, phithetapsi
		Eigen::VectorXcd eigenvector = stabilityEigensystemSolution.eigenvectors().col(i);
		TArray<int> dominantIndexArray({ 0, 3, 6, 9 });
		for (int j = 0; j < 4; j++)
		{
			int index1 = j * 3;
			double evecReal1 = abs(eigenvector[index1].real());
			double evecReal2 = abs(eigenvector[index1 + 1].real());
			double evecReal3 = abs(eigenvector[index1 + 2].real());
			if ((evecReal1 > evecReal2) && (evecReal1 > evecReal3))
			{
				dominantIndexArray[j] = index1;
			}
			else if ((evecReal2 > evecReal1) && (evecReal2 > evecReal3))
			{
				dominantIndexArray[j] = index1 + 1;
			}
			else
			{
				dominantIndexArray[j] = index1 + 2;
			}
		}
		const char* stateNameArray[12] = { "u", "v", "w", "p", "q", "r", "x", "y", "z", "phi", "theta", "psi"};
		modeOutputFile << "\nDominant eigenvector states = ";
		for (int j = 0; j < 4; j++)
		{
			modeOutputFile << stateNameArray[dominantIndexArray[j]] << "  ";
		}
		
		modeOutputFile << "\nEigenvector = \n" << eigenvector;

		double sigma = -eigenvalue.real();
		modeOutputFile << "\nDamping rate, sigma [1/s] =                    " << sigma;
		
		if (eigenvalue.imag() != 0.0)
		{
			// Oscillatory properties for oscillatory modes (eigenvalue = complex conjugate)
			double omega_d = abs(eigenvalue.imag());
			modeOutputFile << "\nDamped natural frequency, omega_d [rad/s] =    " << omega_d;

			double period = 2. * Pi / omega_d;
			modeOutputFile << "\nDamped period [s] =                            " << period;

			std::complex<double> lambda1 = eigenvalue;
			std::complex<double> lambda2 = std::complex<double>(eigenvalue.real(), -1.*eigenvalue.imag());
			double omega_n = std::real(sqrt(lambda1 * lambda2));
			modeOutputFile << "\nUndamped natural frequency, omega_n [rad/s] =  " << omega_n;

			double zeta = std::real( -0.5 * (lambda1 + lambda2) / (sqrt(lambda1 * lambda2)) );
			modeOutputFile << "\nDamping ratio, zeta =                          " << zeta;
		
		}

		if (eigenvalue.real() > 0.)
		{
			// Doubling time for divergent modes
			double doublingTime = -log(2.) / sigma;
			modeOutputFile << "\nDoubling time [s] =                            " << doublingTime;
		}

		if (eigenvalue.real() < 0.)
		{
			// 99% damping time for convergent modes
			double dampingTime99 = -log(0.01) / sigma;
			modeOutputFile << "\n99% Damping Time [s] =                         " << dampingTime99;
		}

	}
	modeOutputFile.close();
	
	UE_LOG(LogTemp, Warning, TEXT("Stability output finished."));
		
}

Eigen::EigenSolver<Eigen::Matrix<double, 12, 12>> AAircraft::SolveStabilityEigensystem()
{
	// Precompute repeated values
	// Sine, Cosine, Tangent of trim phi and theta
	TArray<double> EulerAngles0 = QuatToEuler(TArray<double>({ AircraftStates(9), AircraftStates(10), AircraftStates(11), AircraftStates(12) }));
	UE_LOG(LogTemp, Warning, TEXT("Euler angles (phi, theta, psi) = %f\t%f\t%f"), float(EulerAngles0[0]), float(EulerAngles0[1]), float(EulerAngles0[2]));
	double phi0 = EulerAngles0[0];
	double Sphi = sin(phi0);
	double Cphi = cos(phi0);
	double Tphi = Sphi / Cphi;
	double theta0 = EulerAngles0[1];
	double Stheta = sin(theta0);
	double Ctheta = cos(theta0);
	double Ttheta = Stheta / Ctheta;
	// Force and moment derivatives
	// dForce / dVelocity
	double Fxu = Fxyz_uvw(0, 0);
	double Fxv = Fxyz_uvw(0, 1);
	double Fxw = Fxyz_uvw(0, 2);
	double Fyu = Fxyz_uvw(1, 0);
	double Fyv = Fxyz_uvw(1, 1);
	double Fyw = Fxyz_uvw(1, 2);
	double Fzu = Fxyz_uvw(2, 0);
	double Fzv = Fxyz_uvw(2, 1);
	double Fzw = Fxyz_uvw(2, 2);
	UE_LOG(LogTemp, Warning, TEXT("F,uvw = %f\t%f\t%f"), float(Fxu), float(Fxv), float(Fxw));
	UE_LOG(LogTemp, Warning, TEXT("        %f\t%f\t%f"), float(Fyu), float(Fyv), float(Fyw));
	UE_LOG(LogTemp, Warning, TEXT("        %f\t%f\t%f"), float(Fzu), float(Fzv), float(Fzw));
	// dMoment / dVelocity
	double Mxu = Mxyz_uvw(0, 0);
	double Mxv = Mxyz_uvw(0, 1);
	double Mxw = Mxyz_uvw(0, 2);
	double Myu = Mxyz_uvw(1, 0);
	double Myv = Mxyz_uvw(1, 1);
	double Myw = Mxyz_uvw(1, 2);
	double Mzu = Mxyz_uvw(2, 0);
	double Mzv = Mxyz_uvw(2, 1);
	double Mzw = Mxyz_uvw(2, 2);
	UE_LOG(LogTemp, Warning, TEXT("M,uvw = %f\t%f\t%f"), float(Mxu), float(Mxv), float(Mxw));
	UE_LOG(LogTemp, Warning, TEXT("        %f\t%f\t%f"), float(Myu), float(Myv), float(Myw));
	UE_LOG(LogTemp, Warning, TEXT("        %f\t%f\t%f"), float(Mzu), float(Mzv), float(Mzw));
	// dForce / dRotationRate
	double Fxp = Fxyz_pqr(0, 0);
	double Fxq = Fxyz_pqr(0, 1);
	double Fxr = Fxyz_pqr(0, 2);
	double Fyp = Fxyz_pqr(1, 0);
	double Fyq = Fxyz_pqr(1, 1);
	double Fyr = Fxyz_pqr(1, 2);
	double Fzp = Fxyz_pqr(2, 0);
	double Fzq = Fxyz_pqr(2, 1);
	double Fzr = Fxyz_pqr(2, 2);
	UE_LOG(LogTemp, Warning, TEXT("F,pqr = %f\t%f\t%f"), float(Fxp), float(Fxq), float(Fxr));
	UE_LOG(LogTemp, Warning, TEXT("        %f\t%f\t%f"), float(Fyp), float(Fyq), float(Fyr));
	UE_LOG(LogTemp, Warning, TEXT("        %f\t%f\t%f"), float(Fzp), float(Fzq), float(Fzr));
	// dMoment / dRotationRate
	double Mxp = Mxyz_pqr(0, 0);
	double Mxq = Mxyz_pqr(0, 1);
	double Mxr = Mxyz_pqr(0, 2);
	double Myp = Mxyz_pqr(1, 0);
	double Myq = Mxyz_pqr(1, 1);
	double Myr = Mxyz_pqr(1, 2);
	double Mzp = Mxyz_pqr(2, 0);
	double Mzq = Mxyz_pqr(2, 1);
	double Mzr = Mxyz_pqr(2, 2);
	UE_LOG(LogTemp, Warning, TEXT("M,pqr = %f\t%f\t%f"), float(Mxp), float(Mxq), float(Mxr));
	UE_LOG(LogTemp, Warning, TEXT("        %f\t%f\t%f"), float(Myp), float(Myq), float(Myr));
	UE_LOG(LogTemp, Warning, TEXT("        %f\t%f\t%f"), float(Mzp), float(Mzq), float(Mzr));
	// Initial velocity
	double V0 = VInit;
	double WV0 = W / V0;
	double gV0 = gInit / V0;
	// Moment rate derivative components
	double AMxp = Mxp + gV0*(Ixz*Tphi*Sphi*Ctheta - Ixy*Sphi*Ctheta);
	double AMxq = Mxq - hz + gV0*((Iyy-Izz)*Sphi*Ctheta + 2.*Iyz*Tphi*Sphi*Ctheta - Ixz*Tphi*Stheta);
	double AMxr = Mxr + hy + gV0*((Iyy-Izz)*Tphi*Sphi*Ctheta - 2.*Iyz*Sphi*Ctheta + Ixy*Tphi*Stheta);
	double AMyp = Myp + hz + gV0*((Izz-Ixx)*Sphi*Ctheta + 2.*Ixz*Tphi*Stheta - Iyz*Tphi*Sphi*Ctheta);
	double AMyq = Myq + gV0*(Ixy*Sphi*Ctheta + Iyz*Tphi*Stheta);
	double AMyr = Myr - hx + gV0*(-(Izz-Ixx)*Tphi*Stheta + 2.*Ixz*Sphi*Ctheta + Ixy*Tphi*Sphi*Ctheta);
	double AMzp = Mzp - hy + gV0*((Ixx-Iyy)*Tphi*Sphi*Ctheta - 2.*Ixy*Tphi*Stheta + Iyz*Sphi*Ctheta);
	double AMzq = Mzq + hx + gV0*(-(Ixx-Iyy)*Tphi*Stheta - 2.*Ixy*Tphi*Sphi*Ctheta - Ixz*Sphi*Ctheta);
	double AMzr = Mzr + gV0*(-Iyz*Tphi*Stheta - Ixz*Tphi*Sphi*Ctheta);
	// Equilibrium turning rate, Omega
	double psi0 = 0.;
	double Omega = gV0 * tan(psi0);
	double t = 0;
	double OmegaT = Omega * t;
	double SOt = sin(OmegaT);
	double COt = cos(OmegaT);
	
	// Populate A and B matrices using linearized dimensional coupled eigensystem
	Eigen::Matrix<double, 12, 12> AMatrix{
		{(Fxu),		  (Fxv + WV0*Sphi*Ctheta),	(Fxw - WV0*Tphi*Sphi*Ctheta),		Fxp, Fxq,			Fxr,			0., 0., 0.,		0.,				-W*Ctheta,		0.		},
		{(Fyu - WV0*Sphi*Ctheta),		(Fyv),		 (Fyw - WV0*Tphi*Stheta),		Fyp, Fyq, (Fyr - Wg*V0),			0., 0., 0.,		W*Cphi*Ctheta,  -W*Sphi*Stheta, 0.		},
		{(Fzu + WV0*Tphi*Sphi*Ctheta),	(Fzv + WV0*Tphi*Stheta),	   (Fzw),		Fzp, (Fzq + Wg*V0), Fzr,			0., 0., 0.,		-W*Sphi*Ctheta, -W*Cphi*Stheta, 0.		},
		{Mxu, Mxv, Mxw,																AMxp, AMxq, AMxr,					0., 0., 0.,		0., 0., 0.								},
		{Myu, Myv, Myw,																AMyp, AMyq, AMyr,					0., 0., 0.,		0., 0., 0.								},
		{Mzu, Mzv, Mzw,																AMzp, AMzq, AMzr,					0., 0., 0.,		0., 0., 0.								},
		{(Ctheta*COt), (Sphi*Stheta*COt - Cphi*SOt), (Cphi*Stheta*COt + Sphi*SOt),	0., 0., 0.,							0., 0., 0.,		0., (-Stheta*COt*V0), (-Ctheta*SOt*V0)	},
		{(Ctheta*SOt), (Sphi*Stheta*SOt - Cphi*COt), (Cphi*Stheta*SOt + Sphi*COt),	0., 0., 0.,							0., 0., 0.,		0., (-Stheta*SOt*V0), (Ctheta*COt*V0)	},
		{(-Stheta), (Sphi*Ctheta), (Cphi*Ctheta),									0., 0., 0.,							0., 0., 0.,		0., (-Ctheta*V0), 0.					},
		{0., 0., 0.,																1., (Sphi*Ttheta), (Cphi*Ttheta),	0., 0., 0.,		0., (gV0*Tphi/Ctheta), 0.				},
		{0., 0., 0.,																0., (Cphi), (-Sphi),				0., 0., 0.,		(-gV0*Tphi*Ctheta), 0., 0.				},
		{0., 0., 0.,																0., (Sphi/Ctheta), (Cphi/Ctheta),	0., 0., 0.,		0., (gV0*Tphi*Ttheta), 0.				}
	};

	UE_LOG(LogTemp, Warning, TEXT("\nA:"));
	for (int i = 0; i < 12; i++)
	{
		UE_LOG(LogTemp, Warning, TEXT("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f"), float(AMatrix(i, 0)), float(AMatrix(i, 1)), float(AMatrix(i, 2)), float(AMatrix(i, 3)), float(AMatrix(i, 4)), float(AMatrix(i, 5)), float(AMatrix(i, 6)), float(AMatrix(i, 7)), float(AMatrix(i, 8)), float(AMatrix(i, 9)), float(AMatrix(i, 10)), float(AMatrix(i, 11)));
	}

	Eigen::Matrix<double, 12, 12> BMatrix{
		{Wg, 0., 0.,				0., 0., 0.,			0., 0., 0.,		0., 0., 0.},
		{0., Wg, 0.,				0., 0., 0.,			0., 0., 0.,		0., 0., 0.},
		{0., 0., (Wg - Fz_wdot),	0., 0., 0.,			0., 0., 0.,		0., 0., 0.},
		{0., 0., 0.,				Ixx, -Ixy, -Ixz,	0., 0., 0.,		0., 0., 0.},
		{0., 0., -My_wdot,			-Ixy, Iyy, -Iyz,	0., 0., 0.,		0., 0., 0.},
		{0., 0., 0.,				-Ixz, -Iyz, Izz,	0., 0., 0.,		0., 0., 0.},
		{0., 0., 0.,				0., 0., 0.,			1, 0., 0.,		0., 0., 0.},
		{0., 0., 0.,				0., 0., 0.,			0., 1., 0.,		0., 0., 0.},
		{0., 0., 0.,				0., 0., 0.,			0., 0., 1.,		0., 0., 0.},
		{0., 0., 0.,				0., 0., 0.,			0., 0., 0.,		1., 0., 0.},
		{0., 0., 0.,				0., 0., 0.,			0., 0., 0.,		0., 1., 0.},
		{0., 0., 0.,				0., 0., 0.,			0., 0., 0.,		0., 0., 1.}
	};

	UE_LOG(LogTemp, Warning, TEXT("\nB:"));
	for (int i = 0; i < 12; i++)
	{
		UE_LOG(LogTemp, Warning, TEXT("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f"), float(BMatrix(i, 0)), float(BMatrix(i, 1)), float(BMatrix(i, 2)), float(BMatrix(i, 3)), float(BMatrix(i, 4)), float(BMatrix(i, 5)), float(BMatrix(i, 6)), float(BMatrix(i, 7)), float(BMatrix(i, 8)), float(BMatrix(i, 9)), float(BMatrix(i, 10)), float(BMatrix(i, 11)) );
	}

	// Solve Bx = A for x = B^(-1)A = BinvAMatrix
	Eigen::Matrix<double, 12, 12> BinvAMatrix = BMatrix.colPivHouseholderQr().solve(AMatrix);
	
	// Save the EigenSolver class that contains the eigenvalues and eigenvectors of BinvAMatrix
	Eigen::EigenSolver<Eigen::Matrix<double, 12, 12>> eigensystemSolution(BinvAMatrix);

	return eigensystemSolution;
}

Eigen::Vector3<double> AAircraft::BodyToWindFrame(Eigen::Vector3<double> bodyVector)
{
	// Calculate alpha and beta and their cosines and sines
	double u = AircraftStates(0);
	double v = AircraftStates(1);
	double w = AircraftStates(2);
	double V = sqrt(u * u + v * v + w * w);
	double alpha = atan2(w, u); 
	double sa = sin(alpha);
	double ca = cos(alpha);
	double beta = asin(v / V);
	double sb = sin(beta);
	double cb = cos(beta);
	
	// Transform body to wind frame
	Eigen::Matrix3<double> transformMatrix = Eigen::Matrix3<double>({
		{ca * cb,	sb,		sa * cb	},
		{-ca * sb,	cb,		-sa * sb},
		{-sa,		0.,		ca		}
	});
	Eigen::Vector3<double> windVector = transformMatrix * bodyVector;
	
	return windVector;
}

Eigen::Vector3<double> AAircraft::WindToBodyFrame(Eigen::Vector3<double> windVector)
{
	// Calculate alpha and beta and their cosines and sines
	double u = AircraftStates(0);
	double v = AircraftStates(1);
	double w = AircraftStates(2);
	double V = sqrt(u * u + v * v + w * w);
	double alpha = atan2(w, u);
	double sa = sin(alpha);
	double ca = cos(alpha);
	double beta = asin(v / V);
	double sb = sin(beta);
	double cb = cos(beta);

	// Transform body to wind frame
	Eigen::Matrix3<double> transformMatrix = Eigen::Matrix3<double>({
		{ca * cb,	-ca * sb,	-sa	},
		{sb,		cb,			0.},
		{sa * cb,	-sa * sb,	ca}
		});
	Eigen::Vector3<double> bodyVector = transformMatrix * windVector;

	return bodyVector;
}

void AAircraft::CalculateForceMomentDerivatives(TArray<double> windDeltas)
{
	// Create array of deltaState's to be added to the equilibrium state in the central difference approximation loop below
	// Create vector for each wind delta
	Eigen::Vector3<double> windDu({ windDeltas[0], 0., 0. });
	Eigen::Vector3<double> windDv({ 0., windDeltas[1], 0. });
	Eigen::Vector3<double> windDw({ 0., 0., windDeltas[2] });
	Eigen::Vector3<double> windDp({ windDeltas[3], 0., 0. });
	Eigen::Vector3<double> windDq({ 0., windDeltas[4], 0. });
	Eigen::Vector3<double> windDr({ 0., 0., windDeltas[5] });
	// Convert each wind delta to the body frame
	Eigen::Vector3<double> bodyDu = WindToBodyFrame(windDu);
	Eigen::Vector3<double> bodyDv = WindToBodyFrame(windDv);
	Eigen::Vector3<double> bodyDw = WindToBodyFrame(windDw);
	Eigen::Vector3<double> bodyDp = WindToBodyFrame(windDp);
	Eigen::Vector3<double> bodyDq = WindToBodyFrame(windDq);
	Eigen::Vector3<double> bodyDr = WindToBodyFrame(windDr);
	// Populate matrix with deltaState rows based on body deltas
	Eigen::Matrix<double, 6, 13> deltaStates({
		{bodyDu(0), bodyDu(1), bodyDu(2),	0., 0., 0.,							0., 0., 0.,		0., 0., 0., 0.},
		{bodyDv(0), bodyDv(1), bodyDv(2),	0., 0., 0.,							0., 0., 0.,		0., 0., 0., 0.},
		{bodyDw(0), bodyDw(1), bodyDw(2),	0., 0., 0.,							0., 0., 0.,		0., 0., 0., 0.},
		{0., 0., 0.,						bodyDp(0), bodyDp(1), bodyDp(2),	0., 0., 0.,		0., 0., 0., 0.},
		{0., 0., 0.,						bodyDq(0), bodyDq(1), bodyDq(2),	0., 0., 0.,		0., 0., 0., 0.},
		{0., 0., 0.,						bodyDr(0), bodyDr(1), bodyDr(2),	0., 0., 0.,		0., 0., 0., 0.}
	});
	// Save equilibrium trimmed states
	Eigen::Matrix<double, 13, 1> equilibriumStates = AircraftStates;

	// Loop through perterbations
	// NOTE: i = (0, 1, 2, 3, 4, 5) => (du, dv, dw, dp, dq, dr) in wind frame
	for (int i = 0; i < 6; i++)
	{
		LogStates(equilibriumStates, FString(TEXT("--- Aircraft States")));
		
		Eigen::Matrix<double, 1, 13> deltaState = deltaStates.row(i);
		LogStates(deltaState, FString(TEXT("deltaState")));
		
		// Calculate forces and moments at each perturbation
		Eigen::Matrix<double, 13, 1> positivePerturbedStates = equilibriumStates;
		positivePerturbedStates += deltaState;
		LogStates(positivePerturbedStates, FString(TEXT("states_i+1")));
		TArray<double> perturbedAeroForces_iPlus1 = CalculateAeroForces(positivePerturbedStates, AircraftControls);
		positivePerturbedStates += deltaState;
		LogStates(positivePerturbedStates, FString(TEXT("states_i+2")));
		TArray<double> perturbedAeroForces_iPlus2 = CalculateAeroForces(positivePerturbedStates, AircraftControls);

		Eigen::Matrix<double, 13, 1> negativePerturbedStates = equilibriumStates;
		negativePerturbedStates -= deltaState;
		LogStates(negativePerturbedStates, FString(TEXT("states_i-1")));
		TArray<double> perturbedAeroForces_iMinus1 = CalculateAeroForces(negativePerturbedStates, AircraftControls);
		negativePerturbedStates -= deltaState;
		LogStates(negativePerturbedStates, FString(TEXT("states_i-2")));
		TArray<double> perturbedAeroForces_iMinus2 = CalculateAeroForces(negativePerturbedStates, AircraftControls);

		// Convert force TArrays into eigen arrays so we can approximate derivatives of each force and moment simultaneously using coefficient-wise operations
		Eigen::Array<double, 6, 1> forceMatrix_iPlus1 {
			perturbedAeroForces_iPlus1[0], perturbedAeroForces_iPlus1[1], perturbedAeroForces_iPlus1[2],
			perturbedAeroForces_iPlus1[3], perturbedAeroForces_iPlus1[4], perturbedAeroForces_iPlus1[5]
		};
		Eigen::Array<double, 6, 1> forceMatrix_iPlus2 {
			perturbedAeroForces_iPlus2[0], perturbedAeroForces_iPlus2[1], perturbedAeroForces_iPlus2[2],
			perturbedAeroForces_iPlus2[3], perturbedAeroForces_iPlus2[4], perturbedAeroForces_iPlus2[5]
		};
		Eigen::Array<double, 6, 1> forceMatrix_iMinus1 {
			perturbedAeroForces_iMinus1[0], perturbedAeroForces_iMinus1[1], perturbedAeroForces_iMinus1[2],
			perturbedAeroForces_iMinus1[3], perturbedAeroForces_iMinus1[4], perturbedAeroForces_iMinus1[5]
		};
		Eigen::Array<double, 6, 1> forceMatrix_iMinus2 {
			perturbedAeroForces_iMinus2[0], perturbedAeroForces_iMinus2[1], perturbedAeroForces_iMinus2[2],
			perturbedAeroForces_iMinus2[3], perturbedAeroForces_iMinus2[4], perturbedAeroForces_iMinus2[5]
		};

		// Approximate derivatives using central difference with 2nd order taylor series term (error of O(h^4))
		// NOTE: Higher order central difference equation: f'(x_i) = (-f(x_i+2) + 8*f(x_i+1) - 8*f(x_i-1) + f(x_i-2)) / (12*h)
		UE_LOG(LogTemp, Warning, TEXT("h = %f"), windDeltas[i]);
		UE_LOG(LogTemp, Warning, TEXT("forceMatrix_iPlus1 = {%f\t %f\t %f\t %f\t %f\t %f}"), forceMatrix_iPlus1(0), forceMatrix_iPlus1(1), forceMatrix_iPlus1(2), forceMatrix_iPlus1(3), forceMatrix_iPlus1(4), forceMatrix_iPlus1(5));
		UE_LOG(LogTemp, Warning, TEXT("forceMatrix_iPlus2 = {%f\t %f\t %f\t %f\t %f\t %f}"), forceMatrix_iPlus2(0), forceMatrix_iPlus2(1), forceMatrix_iPlus2(2), forceMatrix_iPlus2(3), forceMatrix_iPlus2(4), forceMatrix_iPlus2(5));
		UE_LOG(LogTemp, Warning, TEXT("forceMatrix_iMinus1 = {%f\t %f\t %f\t %f\t %f\t %f}"), forceMatrix_iMinus1(0), forceMatrix_iMinus1(1), forceMatrix_iMinus1(2), forceMatrix_iMinus1(3), forceMatrix_iMinus1(4), forceMatrix_iMinus1(5));
		UE_LOG(LogTemp, Warning, TEXT("forceMatrix_iMinus2 = {%f\t %f\t %f\t %f\t %f\t %f}"), forceMatrix_iMinus2(0), forceMatrix_iMinus2(1), forceMatrix_iMinus2(2), forceMatrix_iMinus2(3), forceMatrix_iMinus2(4), forceMatrix_iMinus2(5));

		Eigen::Array<double, 6, 1> forceDerivativeMatrix = (-forceMatrix_iPlus2 + 8 * forceMatrix_iPlus1 - 8 * forceMatrix_iMinus1 + forceMatrix_iMinus2) * (1.0 / (12.0 * windDeltas[i]));
		UE_LOG(LogTemp, Warning, TEXT("forceDerivativeMatrix = {%f\t %f\t %f\t %f\t %f\t %f}"), forceDerivativeMatrix(0), forceDerivativeMatrix(1), forceDerivativeMatrix(2), forceDerivativeMatrix(3), forceDerivativeMatrix(4), forceDerivativeMatrix(5));

		// Save forces and moments in derivative member variables
		if (i < 3)
		{
			// For i < 3, deltas = du, dv, dw
			Fxyz_uvw(0, i) = forceDerivativeMatrix(0);
			Fxyz_uvw(1, i) = forceDerivativeMatrix(1);
			Fxyz_uvw(2, i) = forceDerivativeMatrix(2);
			Mxyz_uvw(0, i) = forceDerivativeMatrix(3);
			Mxyz_uvw(1, i) = forceDerivativeMatrix(4);
			Mxyz_uvw(2, i) = forceDerivativeMatrix(5);
		}
		else
		{
			UE_LOG(LogTemp, Warning, TEXT("Count for 2nd loop."));
			// For 3 <= i < 6, deltas = dp, dq, dr and 0 <= j < 3
			int j = i - 3;
			Fxyz_pqr(0, j) = forceDerivativeMatrix(0);
			Fxyz_pqr(1, j) = forceDerivativeMatrix(1);
			Fxyz_pqr(2, j) = forceDerivativeMatrix(2);
			Mxyz_pqr(0, j) = forceDerivativeMatrix(3);
			Mxyz_pqr(1, j) = forceDerivativeMatrix(4);
			Mxyz_pqr(2, j) = forceDerivativeMatrix(5);
		}
	}
		
	// Approximate Fz_wdot and My_wdot using 7.6.66 (Phillips p. 785)
	double rho = CalculateStdAtmProperties_English(-AircraftStates(8))[3];
	double Sh = 63.675;
	double xbwt = -7.358;
	double xbh = -13.13;
	double lwt = 1.1 * (xbwt - xbh);
	double CLaw = 3.3775691217788646;
	double CLah = 1.3657050471586294;
	if (bIsBireAircraft)
	{
		// Override F16 CLah with BIRE CLah
		double bireAngle = AircraftControls(2);
		CLah = 1.3858047943592773 * abs(cos(bireAngle));
	}

	Fz_wdot = -(rho * Sw * Sh * lwt * CLaw * CLah) / (PI * bw * bw);
	My_wdot = (rho * Sw * Sh * lwt * xbh * CLaw * CLah) / (PI * bw * bw);

}

void AAircraft::UpdateLandingForces(float dzFront, float dzRight, float dzLeft, float dt, FVector CollisionNormal)
{
	// Count number of wheels in contact with ground and exit function if none are touching
	int numWheelsOnGround = 0;
	float GearDeflectionsLoopArray[] = {dzFront, dzLeft, dzRight};
	for (float dz : GearDeflectionsLoopArray) {
		if (dz != 0.0f)
		{
			numWheelsOnGround += 1;
		}
	}
	
	// Caulculate and set gear rates
	CurrentGearDeflections = Eigen::Vector3d(dzFront, dzRight, dzLeft);
	Eigen::Vector3d gearRate = (CurrentGearDeflections - PreviousGearDeflections) / dt;
	PreviousGearDeflections = CurrentGearDeflections;
	
	
	// Calculate earth fixed force due to spring-damper system for each gear
	TArray<double> ForceMagn = TArray<double>({
		-double(dzFront)*kArray(0) + gearRate(0) * cArray(0),
		-double(dzRight)*kArray(1) + gearRate(1) * cArray(1),
		-double(dzLeft)*kArray(2) + gearRate(2) * cArray(2)});

	TArray<double> FbFront = TArray<double>({ForceMagn[0]*0.194466, ForceMagn[0]*0.0, ForceMagn[0]*0.980909});
	TArray<double> FbRight = TArray<double>({ForceMagn[1]*-0.03907, ForceMagn[1]*0.48482, ForceMagn[1]*0.873741});
	TArray<double> FbLeft = TArray<double>({ForceMagn[2]*-0.03907, ForceMagn[2]*-0.48482, ForceMagn[2]*0.873741});
	
	// Calculate friction force for each gear in direction opposite forward velocity and projected onto collision plane
	// Find direction opposite forward velocity (BackwardVelocityVec)
	Eigen::Vector3d ForwardVelocityVec(AircraftStates(0), AircraftStates(1), AircraftStates(2));
	ForwardVelocityVec.normalize();
	Eigen::Vector3d BackwardVelocityVec = -1.0 * ForwardVelocityVec;
	// Find collision surface normal vector
	TArray<double> orientationQuat = TArray<double>({
		AircraftStates(9),
		AircraftStates(10),
		AircraftStates(11),
		AircraftStates(12)
	});
	TArray<double> CollisionNormalBodyFixed = EarthToBodyFixed(TArray<double>({CollisionNormal.X, CollisionNormal.Y, -1.0*CollisionNormal.Z}), orientationQuat);
	Eigen::Vector3d CollisionNormalVec(CollisionNormalBodyFixed[0], CollisionNormalBodyFixed[1], CollisionNormalBodyFixed[2]);
	// Project BackwardVelocityVec onto collision plane defined by normal vector, CollisionNormalBodyFixed, to get friction force direction vector
	Eigen::Vector3d FrictionDirVec = BackwardVelocityVec - BackwardVelocityVec.dot(CollisionNormalVec)*CollisionNormalVec;
	FrictionDirVec.normalize();
	// Determine normal force magnitude
	TArray<double> AeroForces = CalculateAeroForces(AircraftStates, AircraftControls);
	TArray<double> GravForce = EarthToBodyFixed(TArray<double>({0.0, 0.0, W}), orientationQuat);
	Eigen::Vector3d AeroGravForcesVec(
		AeroForces[0] + GravForce[0], 
		AeroForces[1] + GravForce[1],
		AeroForces[2] + GravForce[2]
	);
	double NormalForceMagn = AeroGravForcesVec.dot(-1.0*CollisionNormalVec);
	// Calculate friction force vector
	Eigen::Vector3d FrictionForceVec = FrictionDirVec*0.1*NormalForceMagn;
	
	// Calculate body fixed force vector for each gear
	// Start with only spring-damper gear forces
	Eigen::Vector3d FbFrontVec(FbFront[0], FbFront[1], FbFront[2]);
	Eigen::Vector3d FbRightVec(FbRight[0], FbRight[1], FbRight[2]);
	Eigen::Vector3d FbLeftVec(FbLeft[0], FbLeft[1], FbLeft[2]);
	// Add friction for gears touching ground, dividing weight among touching gears
	if (numWheelsOnGround > 0)
	{
		Eigen::Vector3d DividedFrictionForceVec = FrictionForceVec / double(numWheelsOnGround);
		FbFrontVec = (dzFront > 0.0001f) ? FbFrontVec + DividedFrictionForceVec : FbFrontVec;
		FbRightVec = (dzRight > 0.0001f) ? FbRightVec + DividedFrictionForceVec : FbRightVec;
		FbLeftVec = (dzLeft > 0.0001f) ? FbLeftVec + DividedFrictionForceVec : FbLeftVec;
	}
	
	// Calculate moment due to each body-fixed gear force
	Eigen::Vector3d MbFront = frontGearLoc.cross(FbFrontVec);
	Eigen::Vector3d MbRight = rightGearLoc.cross(FbRightVec);
	Eigen::Vector3d MbLeft = leftGearLoc.cross(FbLeftVec);

	// Sum all gear forces and moments
	Eigen::Vector3d FbTotal = FbFrontVec + FbRightVec + FbLeftVec;
	Eigen::Vector3d MbTotal = MbFront + MbRight + MbLeft;

	LandingForces = TArray<double>({
		FbTotal(0),
		FbTotal(1),
		FbTotal(2),
		MbTotal(0),
		MbTotal(1),
		MbTotal(2)
	});
	
}

// Called when the game starts or when spawned
void AAircraft::BeginPlay()
{
	Super::BeginPlay();
	
}

// Called every frame
void AAircraft::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

}

// Called to bind functionality to input
void AAircraft::SetupPlayerInputComponent(UInputComponent* PlayerInputComponent)
{
	Super::SetupPlayerInputComponent(PlayerInputComponent);

}

