#include <iostream>
#include <chrono>
#include <Eigen/Dense>
#include <Eigen/QR>
#include <cmath>
#include <vector>

using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Dynamic;
using Eigen::ArithmeticSequence;
using Eigen::seq;
using Eigen::seqN;
using namespace std;

VectorXd to_VXd(const vector<double> &Vd){
	VectorXd VXd(Vd.size());
	for (int i=0; i<Vd.size(); i++){
		VXd[i] = Vd[i];
	}
	return VXd;
	}

vector<double> to_vectdouble(const VectorXd &VXd)
	{
	vector<double> vectdouble(VXd.size());
	//VectorXd::Map(&vectdouble[0], v1.size()) = VXd
	for (int i=0; i < VXd.size(); i++)
	{ 
		vectdouble[i] = VXd[i];
	}
	return vectdouble;
	}

vector<double> sw_matrix_optim(const vector<double> &mic_ntde_raw, const int &nmics, const double &c=343.0){
	/*
	TODO: THE PSEUDOINVERSE CALCULATION NEEDS TO BE FIXED!! IT IS QUICK AND DIRTY RIGHT NOW!!!!!!
	IT'S REAAAALLLY DIRTY -- SOMETIMES ~A FEW METERS DIFFERENCE BETWEEN NUMPY AND EIGEN RESULTS!!!
	
	mic_ntde is 1D vector<double> with nmics*3 + Nmics-1 entries. 
	The entries are organised so: m0_x.1, m0_y.1, m0_z.1,..mNmics_x.1,mNmics_y.1, mNmics_z.1, tde10...tdeNmics0
	*/
	VectorXd mic_ntde = to_VXd(mic_ntde_raw);
	VectorXd solutions_vx(6);
	vector<double> solutions(6);
	double a1,a2,a3; 
    double a_quad, b_quad, c_quad;
    double t_soln1, t_soln2;
    VectorXd b(nmics-1);
    VectorXd f(nmics-1);
    VectorXd g(nmics-1);
    VectorXd tau(nmics-1);
	VectorXd s1(3),s2(3);
    int position_inds = nmics*3;
	VectorXd mic0 = mic_ntde.head(3);
	tau = mic_ntde.tail(nmics-1)/c;
	MatrixXd R(nmics-1,3);
	MatrixXd R_inv(3, nmics-1);
	ArithmeticSequence< long int, long int, long int > starts = seq(3, position_inds-3, 3);
	ArithmeticSequence< long int, long int, long int > stops = seq(5, position_inds-1, 3);
	for (int i=0; i<starts.size(); i++){
		mic_ntde(seq(starts[i],stops[i])) +=  -mic0;
		}
	R = mic_ntde(seq(3,position_inds-1)).reshaped(3,nmics-1).transpose();
	R_inv = R.completeOrthogonalDecomposition().pseudoInverse();

	for (int i=0; i < nmics-1; i++){
	b(i) = pow(R.row(i).norm(),2) - pow(c*tau(i),2);
	f(i) = (c*c)*tau(i);
	g(i) = 0.5*(c*c-c*c);  
  	}
    a1 = (R_inv*b).transpose()*(R_inv*b);
    a2 = (R_inv*b).transpose()*(R_inv*f);
    a3 = (R_inv*f).transpose()*(R_inv*f);

    a_quad = a3 - pow(c,2);
    b_quad = -1*a2;
    c_quad = a1/4.0;		

    t_soln1 = (-b_quad + sqrt(pow(b_quad,2) - 4*a_quad*c_quad))/(2*a_quad);
    t_soln2 = (-b_quad - sqrt(pow(b_quad,2) - 4*a_quad*c_quad))/(2*a_quad);	
	

    solutions_vx(seq(0,2)) = R_inv*b*0.5 - (R_inv*f)*t_soln1;
	solutions_vx(seq(0,2)) += mic0;
    solutions_vx(seq(3,5)) = R_inv*b*0.5 - (R_inv*f)*t_soln2;
	solutions_vx(seq(3,5)) += mic0;
	solutions = to_vectdouble(solutions_vx);
	return solutions;
}


int main(){
	std::vector<double> qq {0.1, 0.1, 0.1,
			3.61, 54.1, 51.1,
			68.1, 7.1,  8.1,
			9.1,  158.1, 117.1,
			18.1, 99.1, 123.1,
			12.1, 13.1, 14.1, 19.1};
	//VectorXd mictde = to_VXd(qq);
	
	int n_mics = 5;
	vector<double> output;
	output = sw_matrix_optim(qq, n_mics);
	for (auto ii : output){
		std::cout << ii  << std::endl;
	}
	
	return 0;	
}
