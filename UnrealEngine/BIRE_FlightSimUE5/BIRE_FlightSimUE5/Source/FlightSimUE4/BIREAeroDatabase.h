// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "BIREAeroDatabase.generated.h"

/**
 * 
 */
UCLASS()
class FLIGHTSIMUE4_API UBIREAeroDatabase : public UObject
{
	GENERATED_BODY()

public:
	UBIREAeroDatabase();
	
protected:
	double CL0(double dB);
	double CLa(double dB);
	double CLb(double dB);
	double CLpbar(double dB);
	double CLqbar(double dB);
	double CLrbar(double dB);
	double CLda(double dB);
	double CLde(double dB);
	double CS0(double dB);
	double CSa(double dB);
	double CSb(double dB);
	double CSpbar(double dB);
	double CSLpbar(double dB);
	double CSqbar(double dB);
	double CSrbar(double dB);
	double CSde(double dB);
	double CD0(double dB);
	double CDL(double dB);
	double CDL2(double dB);
	double CDS2(double dB);
	double CDpbar(double dB);
	double CDqbar(double dB);
	double CDLqbar(double dB);
	double CDrbar(double dB);
	double CDda(double dB);
	double CDde(double dB);
	double CDLde(double dB);
	double CDde2(double dB);
	double Cl0(double dB);
	double Cla(double dB);
	double Clb(double dB);
	double Clpbar(double dB);
	double Clqbar(double dB);
	double Clrbar(double dB);
	double ClLrbar(double dB);
	double Clda(double dB);
	double Clde(double dB);
	double Cm0(double dB);
	double Cma(double dB);
	double Cmb(double dB);
	double Cmpbar(double dB);
	double Cmqbar(double dB);
	double Cmrbar(double dB);
	double Cmda(double dB);
	double Cmde(double dB);
	double Cn0(double dB);
	double Cna(double dB);
	double Cnb(double dB);
	double Cnpbar(double dB);
	double CnLpbar(double dB);
	double Cnqbar(double dB);
	double Cnrbar(double dB);
	double Cnda(double dB);
	double CnLda(double dB);
	double Cnde(double dB);
	double CL(double dB);
	double CS(double dB);
	double CD(double dB);
	double Cl(double dB);
	double Cm(double dB);
	double Cn(double dB);
};
