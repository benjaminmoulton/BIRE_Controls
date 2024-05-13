#ifndef HELPER_H
#define HELPER_H

#include <iostream>
#include <fstream>
#include <cmath>
#include "json.hpp"
// These are needed, but are included in json.hpp
// #include <functional>
// #include <vector>

using namespace std;
using json = nlohmann::json;

#define PRECISION 1.0e-12
#define PI 3.1415926535897932384626433832795
#define GRAVITY_SI 9.80665

using darray = std::vector<double>;
using dmatrix = std::vector<vector<double>>;
using sarray = std::vector<string>;


// struct State {
//     darray uvw;
//     darray pqr;
//     darray xyz;
//     darray quat;
//     darray controls; // gets initialized another location

//     State() :   uvw(3, 0.0),
//                 pqr(3, 0.0),
//                 xyz(3, 0.0),
//                 quat(4, 0.0) {}
// };

// JSON functions ------------------------------------------------------------------------------------------------------------------
static bool key_exists(json j_object, string key)
{
    return (j_object.find(key) != j_object.end());
}


static darray json_get_array(json j_object)
{
    darray A(j_object.size(),0.0);
    for (int i = 0; i < A.size(); i++) {
        A[i] = j_object[i];
    }
    return A;
}

static json json_get_dictionary_from_file(string filename)
{
    ifstream f(filename);
    json m_input = json::parse(f);
    return m_input;
}

static json json_get_dictionary(json parentDict, string key)
{
    if(key_exists(parentDict[key],"filepath")){
        string fn = parentDict[key]["filepath"];
        // string key = parentDict.key();
        printf("Reading %s information from file: %s\n",key.c_str(),fn.c_str());
        return json_get_dictionary_from_file(fn);
    } else {
        return parentDict[key];
    }
}

static void json_print_dictionary(json dictionary){
    cout << getw << dictionary << '\n';
}

// Misc Functions ------------------------------------------------------------------------------------------------------------------
static const char* bool_cast(const bool b)
{
    return b ? "true" : "false";
}

static double to_radians(double a)
{
    return a*PI/180.0;
}

static double gravity_si(double altitude)
{
    return GRAVITY_SI*pow((6356766./(6356766. + altitude)),2);
}

static double gravity_english(double altitude)
{
    // return 32.174;
    return GRAVITY_SI*pow((6356766./(6356766. + altitude*0.3048)),2)/0.3048;
}

static double get_sign(double a)
{
    if(signbit(a)) return -1.0;
    return 1.0;
}

static void print_hspace(int spacer)
{
    for (int i = 0; i < spacer; i++)
    {
        cout<<" ";
    }

}

// Kinematic Functions ------------------------------------------------------------------------------------------------------------------

static darray quat_mult(darray& A,darray& B)
{
    darray C(3,0.0);
    C[0] = A[0]*B[0] - A[1]*B[1] - A[2]*B[2] - A[3]*B[3];
    C[1] = A[0]*B[1] + A[1]*B[0] + A[2]*B[3] - A[3]*B[2];
    C[2] = A[0]*B[2] - A[1]*B[3] + A[2]*B[0] + A[3]*B[1];
    C[3] = A[0]*B[3] + A[1]*B[2] - A[2]*B[1] + A[3]*B[0];
    return C;
}

static void quat_norm(darray& quat)
{
    double mag = sqrt(quat[0]*quat[0] + quat[1]*quat[1] + quat[2]*quat[2] + quat[3]*quat[3]);
    
    quat[0] /= mag;
    quat[1] /= mag;
    quat[2] /= mag;
    quat[3] /= mag;
}

static darray euler_to_quat(darray& eul)
{
    darray quat(4,0.0);
    double E0 = 0.5*eul[0];
    double E1 = 0.5*eul[1];
    double E2 = 0.5*eul[2];

    double Cp = cos(E0);
    double Ct = cos(E1);
    double Cs = cos(E2);
    double Sp = sin(E0);
    double St = sin(E1);
    double Ss = sin(E2);

    double CtCs = Ct*Cs;
    double StSs = St*Ss;
    double StCs = St*Cs;
    double CtSs = Ct*Ss;

    quat[0] = Cp*CtCs + Sp*StSs;
    quat[1] = Sp*CtCs - Cp*StSs;
    quat[2] = Cp*StCs + Sp*CtSs;
    quat[3] = Cp*CtSs - Sp*StCs;
    return quat;
}

static darray quat_to_euler_axis(darray& quat)
{
    darray eul(4,0.0); // Euler axis has 4 components
    double theta = 2.0*acos(quat[0]);
    double denom = sin(0.5*theta);
    eul[0] = theta;
    eul[1] = quat[1]/denom;
    eul[2] = quat[2]/denom;
    eul[3] = quat[3]/denom;
    return eul;
}

static darray quat_to_euler(darray& quat)
{
    darray eul(3,0.0);
    double e0 = quat[0];
    double e1 = quat[1];
    double e2 = quat[2];
    double e3 = quat[3];

    double condition = e0*e2 - e1*e3;

    if(abs(condition) == 0.5){
        if (condition == 0.5){
            eul[0] =  2.0*asin(e1*1.414213562373095);
            eul[1] = 1.57079632679489;
            eul[2] = 0.0;
        }
        else{
            eul[0] = 2.0*asin(e1*1.4142135623730950);
            eul[1] = -1.57079632679489;
            eul[2] = 0.0;
        }
    }
    else{
        double e02 = e0*e0;
        double e12 = e1*e1;
        double e22 = e2*e2;
        double e32 = e3*e3;
        eul[0] = atan2(2.0*(e0*e1 + e2*e3),(e02 + e32 - e22 - e12));
        eul[1] = asin(2.0*condition);
        eul[2] = atan2(2.0*(e0*e3 + e1*e2),(e02 + e12 - e22 - e32));
    }
    return eul;
}


static darray quat_trans_2_1(darray& vec, darray& quat)
{
    darray ans(3,0.0);
    double x = vec[0];
    double y = vec[1];
    double z = vec[2];

    double e0 = quat[0];
    double ex = quat[1];
    double ey = quat[2];
    double ez = quat[3];

    double T0 = x*ex + y*ey + z*ez;
    double T1 = x*e0 - y*ez + z*ey;
    double T2 = x*ez + y*e0 - z*ex;
    double T3 = y*ex - x*ey + z*e0;

    ans[0] = e0*T1 + ex*T0 + ey*T3 - ez*T2;
    ans[1] = e0*T2 - ex*T3 + ey*T0 + ez*T1;
    ans[2] = e0*T3 + ex*T2 - ey*T1 + ez*T0;

    return ans;
}


static darray quat_trans_1_2(darray& vec, darray& quat)
{
    darray ans(3,0.0);
    double x = vec[0];
    double y = vec[1];
    double z = vec[2];

    double e0 = quat[0];
    double ex = quat[1];
    double ey = quat[2];
    double ez = quat[3];

    double T0 = -x*ex - y*ey - z*ez;
    double T1 =  x*e0 + y*ez - z*ey;
    double T2 =  y*e0 - x*ez + z*ex;
    double T3 =  x*ey - y*ex + z*e0;

    ans[0] = e0*T1 - ex*T0 - ey*T3 + ez*T2;
    ans[1] = e0*T2 + ex*T3 - ey*T0 - ez*T1;
    ans[2] = e0*T3 - ex*T2 + ey*T1 - ez*T0;

    return ans;
}

static dmatrix get_rotation_matrix(darray& quat)
{
    double e0 = quat[0];
    double ex = quat[1];
    double ey = quat[2];
    double ez = quat[3];

    dmatrix R(3, darray(3,0.0));
    R[0][0] = ex*ex + e0*e0 - ey*ey - ez*ez;
    R[0][1] = 2.0*(ex*ey - ez*e0);
    R[0][2] = 2.0*(ex*ez + ey*e0);
    R[1][0] = 2.0*(ex*ey + ez*e0);
    R[1][1] = ey*ey + e0*e0 - ex*ex - ez*ez;
    R[1][2] = 2.0*(ey*ez - ex*e0);
    R[2][0] = 2.0*(ex*ez - ey*e0);
    R[2][1] = 2.0*(ey*ez + ex*e0);
    R[2][2] = ez*ez + e0*e0 - ex*ex - ey*ey;
    return R;
}

// Array functions ------------------------------------------------------------------------------------------------------------------
// Arrays are 1-D Vectors

static void darray_print(darray& A)
{
    for(int i=0; i<A.size(); i++)
    {
        printf("%20.12e",A[i]);
    }
    printf("\n");
}

static darray darray_copy(darray& A)
{
    darray B(A.size(),0.0);
    for (int i = 0; i < A.size(); i++) {
        B[i] = A[i];
    }
    return B;
}

static darray darray_absolute(darray& A)
{
    darray B(A.size(),0.0);
    for (int i = 0; i < A.size(); i++)
    {
        B[i] = abs(A[i]);
    }
    return B;
}

static darray darray_scalar_mult(darray& A, double value)
{
    darray B(A.size(),0.0);
    for (int i = 0; i < A.size(); i++) {
        B[i] = value*A[i];
    }
    return B;
}

static darray darray_add(darray& A, darray& B)
{
    darray C(A.size(),0.0);
    for (int i = 0; i < A.size(); i++) {
        C[i] = A[i] + B[i];
    }
    return C;
}

static darray darray_combine(darray& A, darray& B)
{
    darray C = darray_copy(A);
    for (int i = 0; i < B.size(); i++) {
        C.push_back(B[i]);
    }
    return C;
}

static void darray_fill_zeros(darray& A)
{
    for (int i = 0; i < A.size(); i++) {
        A[i] = 0.0;
    }
}

static double darray_dot_product(darray& A, darray& B)
{
    double ans = 0.0;
    for (int i = 0; i < A.size(); i++) {
        ans += A[i]*B[i];
    }
    return ans;
}

static double darray_mag(darray& A)
{
    double mag = 0.0;
    for (int i = 0; i < A.size(); i++) {
        mag += A[i]*A[i];
    }
    return sqrt(mag);
}

static darray darray_cross_3(darray& A, darray& B)
{
    darray C(A.size(),0.0);
    C[0] = A[1]*B[2] - A[2]*B[1];
    C[1] = A[2]*B[0] - A[0]*B[2];
    C[2] = A[0]*B[1] - A[1]*B[0];
    return C;
}

static dmatrix darray_col_row_mult(darray& A, darray& B)
{
    dmatrix C(A.size(), darray(B.size(),0.0));
    for(int i = 0; i < A.size(); i++)
    {
        for(int j = 0; j < B.size(); j++)
        {
            C[i][j] = A[i]*B[j];
        }
    }
    return C;
}

static void sarray_print(sarray values)
{
    for(int i = 0; i < values.size(); i++)
    {
        cout<<values[i]<<" ";
    }
    cout<<endl;
}

// Matrix Functions ------------------------------------------------------------------------------------------------------------------
// Matrices are 2-D vectors

static void dmatrix_print(dmatrix& A)
{
    // Access A and print its contents
    for (const auto& row : A) {
        for (const auto& elem : row) {
            printf("%20.12e",elem);
        }
        cout << endl;
    }
}

static dmatrix dmatrix_add(dmatrix& A, dmatrix& B)
{
    dmatrix C(A.size(), darray(A[0].size(),0.0));
	for (int i = 0; i < A.size(); i++) {
		for (int j = 0; j < A[i].size(); j++) {
			C[i][j] = A[i][j] + B[i][j];
		}
	}
    return C;
}

static dmatrix dmatrix_copy(dmatrix& A)
{
    dmatrix B(A.size(), darray(A[0].size(),0.0));
	for (int i = 0; i < A.size(); i++) {
		for (int j = 0; j < A[i].size(); j++) {
			B[i][j] = A[i][j];
		}
	}
    return B;
}

static dmatrix dmatrix_scalar_mult(dmatrix& A, double value)
{
    dmatrix B(A.size(), darray(A[0].size(),0.0));
	for (int i = 0; i < A.size(); i++) {
		for (int j = 0; j < A[i].size(); j++) {
			B[i][j] = value*A[i][j];
		}
	}
    return B;
}

static dmatrix dmatrix_transpose(dmatrix& A)
{
    dmatrix B(A.size(), darray(A[0].size(),0.0));
	for (int i = 0; i < A.size(); i++) {
		for (int j = 0; j < A[i].size(); j++) {
			B[j][i] = A[i][j];
		}
	}
    return B;
}

static darray dmatrix_array_mult(dmatrix& A, darray& X)
{
    darray B(X.size(),0.0);
	for (int i = 0; i < X.size(); i++) {
		for (int j = 0; j < A[i].size(); j++) {
			B[i] += A[i][j]*X[j];
		}
	}
    return B;
}

static void dmatrix_fill_zeros(dmatrix& A)
{
    for (auto& row : A) {
        for (auto& elem : row) {
            elem = 0.0;
        }
    }
}

static dmatrix dmatrix_identity(int size)
{
    dmatrix A(size, darray(size, 0.0));
    for(int i=0; i<size; i++)
    {
        A[i][i] = 1.0;
    }
    return A;
}

static dmatrix dmatrix_multiply(dmatrix& A, dmatrix& B)
{
    // Check if the matrices can be multiplied
    if (A[0].size() != B.size()) {
        throw invalid_argument("Matrix dimensions are not compatible for multiplication");
    }

    dmatrix C(A.size(), darray(A[0].size(),0.0));

    // Perform dmatrix multiplication
    for (int i = 0; i < A.size(); i++) {
        for (int j = 0; j < B[0].size(); j++) {
            for (int k = 0; k < A[0].size(); k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

static dmatrix dmatrix_invert_2x2(dmatrix& A) {
    // Function accepts a 2x2 dmatrix as input (A) and stores the inverse in (B)
    double det = A[0][0]*A[1][1] - A[0][1]*A[1][0];

    if (det == 0) {
        // The dmatrix is singular, cannot be inverted
        throw invalid_argument("Matrix is singular and cannot be inverted.");
    }

    dmatrix B(A.size(), darray(A[0].size(),0.0));

    B[0][0] =  A[1][1] / det;
    B[0][1] = -A[0][1] / det;
    B[1][0] = -A[1][0] / det;
    B[1][1] =  A[0][0] / det;

    return B;
}

static dmatrix dmatrix_invert_3x3(dmatrix& A) {
    // Function accepts a 3x3 dmatrix as input (A) and stores the inverse in (B)
    double det =   A[0][0]*(A[1][1]*A[2][2] - A[1][2]*A[2][1])
                 - A[0][1]*(A[1][0]*A[2][2] - A[1][2]*A[2][0])
                 + A[0][2]*(A[1][0]*A[2][1] - A[1][1]*A[2][0]);

    if (det == 0) {
        // The dmatrix is singular, cannot be inverted
        throw invalid_argument("Matrix is singular and cannot be inverted.");
    }

    dmatrix B(A.size(), darray(A[0].size(),0.0));

    B[0][0] = (A[1][1]*A[2][2] - A[1][2]*A[2][1]) / det;
    B[0][1] = (A[0][2]*A[2][1] - A[0][1]*A[2][2]) / det;
    B[0][2] = (A[0][1]*A[1][2] - A[0][2]*A[1][1]) / det;
    B[1][0] = (A[1][2]*A[2][0] - A[1][0]*A[2][2]) / det;
    B[1][1] = (A[0][0]*A[2][2] - A[0][2]*A[2][0]) / det;
    B[1][2] = (A[0][0]*A[1][2] - A[0][2]*A[1][0]) / det;
    B[2][0] = (A[1][0]*A[2][1] - A[1][1]*A[2][0]) / det;
    B[2][1] = (A[0][1]*A[2][0] - A[0][0]*A[2][1]) / det;
    B[2][2] = (A[0][0]*A[1][1] - A[0][1]*A[1][0]) / det;

    return B;
}


static dmatrix parallel_axis_theorem(dmatrix& I1, double mass, darray s)
{
    dmatrix I2;
    dmatrix E = dmatrix_identity(3);
    dmatrix temp = darray_col_row_mult(s,s);
    dmatrix temp2 = dmatrix_scalar_mult(E, darray_dot_product(s,s));

    temp = dmatrix_scalar_mult(temp,-1.0);
    temp = dmatrix_add(temp,temp2);
    temp = dmatrix_scalar_mult(temp,mass);
    I2 = dmatrix_add(temp,I1);
    return I2;
}

// File Reading

static int tokenize(const string& str, vector<string>& tokens, const string& delimiters)
{
  // skip delimiters at beginning.
  string::size_type lastPos = str.find_first_not_of(delimiters, 0);

  // find first "non-delimiter".
  string::size_type pos   = str.find_first_of(delimiters, lastPos);

  while (string::npos != pos || string::npos != lastPos)
  {
    // found a token, add it to the vector.
    tokens.push_back(str.substr(lastPos, pos - lastPos));

    // skip delimiters. Note the "not_of"
    lastPos = str.find_first_not_of(delimiters, pos);

    // find next "non-delimiter"
    pos = str.find_first_of(delimiters, lastPos);
  }
  return 0;
}

static int read_file(string filename, vector<vector<string>> &data)
{
	string line;
	string delimiters(", ;\t");

    //// cout<<"Reading File "<<filename<<endl;
	ifstream file_in(filename.c_str());
	if(!file_in)
	{
        cout<<"Error: File "<<filename<<" not in directory."<<endl;
		return 1;
	}

	while(getline(file_in,line,'\n'))
	{
        vector<string> tokens;
		tokenize(line, tokens, delimiters);
        data.push_back(tokens);
	}
	file_in.close();
	return 0;
}

#endif // End of HELPER_H