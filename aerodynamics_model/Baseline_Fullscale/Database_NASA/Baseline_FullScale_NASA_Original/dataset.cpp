#include "dataset.h"

void Dataset::create_from_file(const string& filename) {
    mFilename = filename;

    vector<vector<string>> dataString;
    dmatrix fileData;

    read_file(filename,dataString);

    mNumIndependentVars = stoi(dataString[0][2]);
    int totalColumns = dataString[2].size();

    // cout<<"Number of independent variables = "<<mNumIndependentVars<<endl;
    // cout<<"Number of dependent variables = "<<totalColumns - mNumIndependentVars<<endl;

    // cout<<"Independent Variable Names = "<<endl;
    for (int i=0; i<mNumIndependentVars; i++){
        // cout<<dataString[1][i]<<endl;
        mIndependentVarNames.push_back(dataString[1][i]);
    }
    // cout<<endl;

    // cout<<"Dependent Variable Names = "<<endl;
    for (int i=mNumIndependentVars; i<totalColumns; i++){
        // cout<<dataString[1][i]<<endl;
        mDependentVarNames.push_back(dataString[1][i]);
    }
    // cout<<endl;

    // cout<<"Checking and Converting Data..."<<endl;
    for(int i=2; i<dataString.size(); i++){
        darray row;
        if (dataString[i].size() != totalColumns) {
            // cout<<"Row "<<i+1<<" has an error:"<<endl;
            throw runtime_error("Invalid number of columns in row");
        }
        for(int j=0; j<totalColumns; j++){
            row.push_back(stod(dataString[i][j]));
        }
        fileData.push_back(row);
    }


    // cout<<"Sorting Data by the Independent Variables"<<endl;
    // Sort the data by the independent variables (first columns)
    sort(fileData.begin(), fileData.end(), [this](const darray& a, const darray& b) {
        for (int i = 0; i < mNumIndependentVars; i++) {
            if (a[i] != b[i]) {
                return a[i] < b[i];
            }
        }
        return false; // No change if all independent variables are equal
    });

    create_from_data(fileData,mNumIndependentVars,mIndependentVarNames,mDependentVarNames);
    // cout<<"Dataset Successfully Loaded."<<endl<<endl;
}

void Dataset::create_from_data(dmatrix inputData, int numIndependentVars, vector<string> independentVarNames, vector<string> dependentVarNames){

    mNumIndependentVars = numIndependentVars;
    mIndependentVarNames = independentVarNames;
    mDependentVarNames = dependentVarNames;

    int totalColumns = inputData[0].size();

    // Create subsets if needed
    if(mNumIndependentVars == 1){
        mHasSubset = false;
        for(int i=0; i< inputData.size(); i++) {
            mX.push_back(inputData[i][0]); // Store the independent variable
            darray row;
            for(int j=1; j<totalColumns; j++){
                row.push_back(inputData[i][j]);
            }
            mY.push_back(row); // Store the dependent variables
        }
    }
    else{
        mHasSubset = true;
        set<double> uniqueSet;
        // Find number of unique values and store them in mX
        for(int i=0; i<inputData.size(); i++){
            if(uniqueSet.insert(inputData[i][0]).second){
                mX.push_back(inputData[i][0]);
            }
        }

        // Create a subset of data for each independent variable
        for(int k=0; k<mX.size(); k++){
            double uniqueVal = mX[k];
            dmatrix subset;
            for(int i=0; i<inputData.size(); i++){
                darray row;
                if(inputData[i][0] == uniqueVal){
                    for(int j=1; j<totalColumns; j++){
                        row.push_back(inputData[i][j]);
                    }
                    subset.push_back(row); // Store the dependent variables
                }
            }
            // Store the independent variable names of the subset
            vector<string> tempNames;
            for(int i=1; i<numIndependentVars; i++)
            {
                tempNames.push_back(mIndependentVarNames[i]);
            }
            Dataset* newDataset = new Dataset();
            newDataset->create_from_data(subset,mNumIndependentVars-1,tempNames,dependentVarNames);
            mSubset.push_back(newDataset);
        }
    }
}

void Dataset::print_data(int spacer){
    if(!mHasSubset)
    {
        print_hspace(spacer);
        // cout<<mIndependentVarNames[0]<<" ";
        sarray_print(mDependentVarNames);
    }
    for (int i=0; i< mX.size(); i++) {
        print_hspace(spacer);
        // cout<<mIndependentVarNames[0]<<" = "<<mX[i]<<" : ";
        if(mHasSubset){
            // cout<<endl;
            mSubset[i]->print_data(spacer+3);
        }
        else{
            for (int j=0; j<mY[i].size(); j++){
                // cout << mY[i][j] << " ";
            }
            // cout << endl;
        }
    }
}


darray Dataset::linear_interpolate(darray independentValues, int verbose){
    // verbose is an integer. If zero, it will not print. If >0, it will print with indented spaces of n=verbose.
    // Always interpolating relative to the first value listed in independentValues
    double value = independentValues[0];
    string iVarName = mIndependentVarNames[0];
    int lower = -1;
    int subVerbose = 0;
    darray vals1;
    darray vals2;
    darray ans;
    darray independentSubset;

    if(verbose>0){
        print_hspace(verbose); cout<<"Interpolating for "<<iVarName<<" = "<<value<<endl;
        subVerbose = verbose+3;
    }

    // Create subset of values to be interpolated if needed
    if(mHasSubset) {
        for (int i=1; i<independentValues.size(); i++) {
            independentSubset.push_back(independentValues[i]);
        }
    }

    // Find index of lower bound
    for (int i=0; i< mX.size(); i++){
        if(value > mX[i]){
            lower = i;
        }
    }


    if((lower <= -1) || (lower >= mX.size()-1)){ // Case when value is out of bounds of data
        if(lower == -1) { // set lower to 0 to represent lower bound value
            lower = 0;
        }
        if(mHasSubset) {
            vals1 = mSubset[lower]->linear_interpolate(independentSubset, subVerbose);
        }
        else{
            vals1 = mY[lower];
        }
        ans = vals1;
    }
    else {
        if(mHasSubset) {
            if(verbose>0){print_hspace(verbose); cout<<iVarName<<" lower index is "<<lower<<" with value of "<<iVarName<<" = "<<mX[lower]<<endl;}
            vals1 = mSubset[lower]->linear_interpolate(independentSubset, subVerbose);

            if(verbose>0){print_hspace(verbose); cout<<iVarName<<" upper index is "<<lower+1<<" with value of "<<iVarName<<" = "<<mX[lower+1]<<endl;}
            vals2 = mSubset[lower+1]->linear_interpolate(independentSubset, subVerbose);
        }
        else{
            if(verbose>0){print_hspace(verbose); cout<<iVarName<<" lower index is "<<lower<<" with value of "<<iVarName<<" = "<<mX[lower]<<endl;}
            vals1 = mY[lower];

            if(verbose>0){print_hspace(verbose); cout<<iVarName<<" upper index is "<<lower+1<<" with value of "<<iVarName<<" = "<<mX[lower+1]<<endl;}
            vals2 = mY[lower+1];
        }

        // Perform the Interpolation
        double ratio = (value - mX[lower])/(mX[lower+1] - mX[lower]);
        for (int i=0; i< vals1.size(); i++){
            ans.push_back(vals1[i] + ratio*(vals2[i] - vals1[i]));
        }
        if(verbose>0){
            print_hspace(verbose); cout<<"Solution for "<<iVarName<<" = "<<value<<endl;
            print_hspace(verbose); cout<<"   Dependent Variables = "; sarray_print(mDependentVarNames);
            print_hspace(verbose); cout<<"Lower dependent values = "; darray_print(vals1);
            print_hspace(verbose); cout<<"Upper dependent values = "; darray_print(vals2);
            print_hspace(verbose); cout<<"                 Ratio = "<<ratio<<endl;
            print_hspace(verbose); cout<<"                Answer = "; darray_print(ans);
            print_hspace(verbose); cout<<endl;
        }
    }

    return ans;
}