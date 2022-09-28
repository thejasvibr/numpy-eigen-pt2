#include <omp.h>
#include <iostream>
#include <chrono>
#include <unistd.h>
#include <Eigen/Core>
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
	
	MatrixXd Eye(R.rows(),R.rows());
    Eye = MatrixXd::Zero(R.rows(), R.rows());
    Eye.diagonal() = VectorXd::Ones(R.rows());
	
    //R_inv = R.colPivHouseholderQr().solve(Eye);
	R_inv = R.fullPivHouseholderQr().solve(Eye);
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

// for some reason cppyy throws an error if const double &c=343.0!
vector<vector<double>> blockwise_sw_optim(const vector<vector<double>> &block_inputs, const vector<int> &block_nmics, const double c=343.0){
	/*
	Accepts a 'block' of mic-posns+TDEs as a vector<double>
	*/
	vector<vector<double>> block_solutions(block_inputs.size());
	for (int i=0; i<block_inputs.size(); i++){
		block_solutions[i] = sw_matrix_optim(block_inputs[i], block_nmics[i], c);
	}
	return block_solutions;
	}



std::vector<std::vector<int>> split(const std::vector<int>& v, int Nsplit) {
	// thanks to @Damien : https://stackoverflow.com/a/66173799/4955732
    int n = v.size();
    int size_max = n / Nsplit + (n % Nsplit != 0);
    std::vector<std::vector<int>> split;
    for (int ibegin = 0; ibegin < n; ibegin += size_max) {
        int iend = ibegin + size_max;
        if (iend > n) iend = n;
        split.emplace_back (std::vector<int>(v.begin() + ibegin, v.begin() + iend));
    }
    return split;
}

vector<vector<double>> pll_sw_optim(vector<vector<double>> all_inputs, vector<int> &all_nmics, int num_cores, double c=343.0){
	Eigen::initParallel();
	vector<vector<double>> flattened_output;
	vector<vector<vector<double>>> all_block_outputs(num_cores);
	
	vector<vector<vector<double>>> blockwise_inputs(num_cores);
	vector<vector<int>> blockwise_nummics(num_cores);
	
	// split up all_inputs and all_nmics according to num_cores
	vector<int> all_indices(all_inputs.size());
			
	for (int i=0; i<all_inputs.size(); i++){
		all_indices[i] = i;
		}
	
	vector<vector<int>> blockwise_indices = split(all_indices, num_cores);
	int inner_k = 0;
	for (int block = 0; block < num_cores; block++) {
		blockwise_inputs[block] = {};
		blockwise_nummics[block] = {};
		for (auto index : blockwise_indices[block]){
			blockwise_inputs[block].push_back(all_inputs[index]);
			blockwise_nummics[block].push_back(all_nmics[index]);
			}
		}

	// Now run the parallelisable code
	#pragma omp parallel for
	for (int block = 0; block < num_cores; block++){
		all_block_outputs[block] = blockwise_sw_optim(blockwise_inputs[block], blockwise_nummics[block], c);
		
		}
	
	flattened_output = {};
	for (int block=0; block<num_cores; block++){
		
		for (auto solution : all_block_outputs[block]){
			flattened_output.push_back(solution);
		}
	}
	
	return flattened_output;
					}

int main(){
	
	std::vector<double> qq {0.1, 0.1, 0.1,
			3.61, 54.1, 51.1,
			68.1, 7.1,  8.1,
			9.1,  158.1, 117.1,
			18.1, 99.1, 123.1,
			0.001, -.001, 0.002, 0.005};
	//VectorXd mictde = to_VXd(qq);
	
	int n_mics = 5;
	vector<double> output;
	auto start = chrono::steady_clock::now();
	output = sw_matrix_optim(qq, n_mics);
	auto stop = chrono::steady_clock::now();
	double durn1 = chrono::duration_cast<chrono::microseconds>(stop - start).count();
	
	for (auto ii : output){
		std::cout << ii  << std::endl;
	}
	
	// Now run the parallelised version 
	int nruns = 500000;
	vector<vector<double>> block_in(nruns);
	vector<vector<double>> pll_out;
	vector<int> block_nmics(block_in.size());
	
	std::cout << block_in.size() << std::endl;
	for (int i=0; i < block_in.size(); i++){
		block_in[i] = qq;
		block_nmics[i] = n_mics;
	}
	// run the whole code without parallelism
	std::cout << "Serial run starting... " << std::endl;
	auto start1 = chrono::steady_clock::now(); 
	vector<vector<double>> serial_out(nruns);
	for (int i=0; i<nruns; i++){
		serial_out[i] = sw_matrix_optim(qq, n_mics);
	}
	auto stop1 = chrono::steady_clock::now();
	durn1 = chrono::duration_cast<chrono::microseconds>(stop1-start1).count();
	std::cout << durn1 << " Serial s"<< std::endl;

	// Now finally try to run the actual pll function
	std::cout << "Parallel run starting... " << std::endl;
	auto start2 = chrono::steady_clock::now();
	pll_out = pll_sw_optim(block_in, block_nmics, 8, 343.0);
	auto stop2 = chrono::steady_clock::now();
	auto durn2 = chrono::duration_cast<chrono::microseconds>(stop2 - start2).count();
	std::cout << durn2 << " FN pll s"<< std::endl;
	
	std::cout << "Obtained speedup: " << durn1/durn2 << std::endl;
	
	return 0;	
}
