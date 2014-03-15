#include <cusp/io/matrix_market.h>
#include <cusp/coo_matrix.h>
#include <cusp/dia_matrix.h>
#include <cusp/verify.h>
#include <cusp/print.h>
#include <cusp/multiply.h>
#include <cusp/transpose.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <iomanip>

#include <thrust/functional.h>
#include <thrust/system_error.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>


using namespace thrust;
using namespace cusp;
using namespace std;


//#define CPU	0


#ifdef CPU

#define CAST(X)	X ## _CPU
#define MEM	cusp::host_memory

#else

#define CAST(X)	X ## _GPU
#define MEM	cusp::device_memory

#endif


typedef coo_matrix<int, double, host_memory> COO_CPU;
typedef array1d <double, host_memory> ARR1D_CPU;

typedef coo_matrix<int, double, device_memory> COO_GPU;
typedef array1d <double, device_memory> ARR1D_GPU;

void read_matrix(CAST(COO) &temp, const char *fname) {
	// Reads web matrix from market file
	io::read_matrix_market_file(temp, fname);
	//print(temp);
	fprintf(stderr, "Read the matrix\n");

	// Link Matrix (Transpose of Web Matrix)
	//COO_CPU temp1 = temp;
	//transpose(temp, temp1);
	//temp = temp1;
}

/*
		Creating a struct to define the inversion operation. 
*/
template <typename T>
struct inversion_op{
__host__ __device__ T operator()(const T& x)const{
		return 1 / x;
}
};


void normalize(CAST(COO) &adj, CAST(ARR1D) &inv_sum) {

	device_vector<double> in_keys(adj.values.size()); //cols
	device_vector<double> in_values(adj.values.size());	//val
	device_vector<double> out_keys(adj.num_rows);	//cols
	device_vector<double> out_values(adj.num_rows, 1); //sum


	/*
		in_keys are row indices corresponding to nnz entries in the matrix
		in_values are the nnz values in the matrix
		in_keys: 	0 0 1 1 2
		in_values: 	1 1 1 1 1	
	*/
	thrust::copy(adj.row_indices.begin(), adj.row_indices.end(), in_keys.begin());
	thrust::copy(adj.values.begin(), adj.values.end(), in_values.begin());

	thrust::equal_to<double> binary_pred;
	thrust::plus<double> binary_op;

	/*
	reduces in_values using given operator (plus) compares like row numbers using (equal_to) and dumps result into out_values indexed by out_keys
	From the above example:
	out_keys:	0	1	2
	out_values:	2	2	1
	*/
	reduce_by_key(in_keys.begin(), in_keys.end(), in_values.begin(), out_keys.begin(), out_values.begin(), binary_pred, binary_op);

	fprintf(stderr, "Row sum calculated\n");
	/*cout << "INKEY\tINVAL =====================\n";
	for(int i = 0; i < in_keys.size(); i++)
			cout << in_keys[i] << "\t" << in_values[i] << endl;

	cout << "OUTKEY\tOUTVAL =====================\n";
	for(int i = 0; i < out_keys.size(); i++)
			cout << out_keys[i] << "\t" << out_values[i] << endl;
	*/
	/*
		instantiated an inversion_op (invert) for use in transform
	*/
	inversion_op<double> invert = inversion_op<double>();
	thrust::transform(out_values.begin(), out_values.end(), out_values.begin(), invert);

	thrust::copy(out_values.begin(), out_values.end(), inv_sum.begin());

	//cout << "INVERSE SUM" << endl;
	//print(inv_sum);
	
	CAST(COO) link_mat = adj;
	transpose(adj, link_mat);
	adj = link_mat;
	fprintf(stderr, "Transpose calculated.\n");

	/*
	create diagonal matrix (num_rows x num_rows) and assign corresponding inverse sum values to its principal diagonal.
	diagonal_offset => the offset from the principal diagonal (the one starting from element [0][0])
	dia.values(x, y) => the xth element in the yth diagonal

	For the rows in adj which has non_zero sum(ie at least one entry) set the corresponding row_num'th dia elmt to the inverse of its sum else set to 1
	e.g.
		index in inv_sum:		0 	2
		value in inv_sum:		1/2	1/3

		set (0,0) ---> 1/2, (1,1) ----> 1, (2,2) ----> 1/3
	*/
	cusp::dia_matrix<int, double, MEM> dia(adj.num_rows, adj.num_rows, adj.num_rows, 1);
	dia.diagonal_offsets[0] = 0;
	for(int i = 0; i < out_keys.size(); i++)
			dia.values(i, 0) = 1;
	for(int i = 0; i < out_keys.size(); i++) {
			//cout << i << "\t" << out_keys[i] << "\t" << inv_sum[i] << endl;
			dia.values(out_keys[i], 0) = inv_sum[i];
	}

	cout << "DIA ==========\n";
	print(dia);

	/*
		For some reason, the 0th entry in the diagonal is not being set in the above for loop. Therefore, this hack manually sets the first entry in the diagonal.
	*/
	//dia.values(0, 0) = inv_sum[0];

	fprintf(stderr, "Formed dia_mat.\n");


	if(is_valid_matrix(adj)) {
		multiply(adj, dia, link_mat);	// link_mat = adj * dia
		adj = link_mat;
	} else {
			cout << "Invalid format!" << endl;
			exit(1);
	}

	/*
	CAST(ARR1D) sum(adj.num_cols, 0);
	for(int i = 0; i < adj.values.size(); i++)
			sum[adj.column_indices[i]] += adj.values[i];
	print(sum);
	*/
	fprintf(stderr, "Normalized\n");
}


void pagerank(CAST(COO) &link, double beta, CAST(ARR1D) &rank) {
	int V = link.num_rows;
	double one_minus_beta = (1 - beta) / V;

	CAST(ARR1D) teleport(V, one_minus_beta);
	CAST(ARR1D) temp(V);

	blas::fill(rank, 1 / (double) V);
	if(!is_valid_matrix(link)) {
		cout << "Link: Invalid format!" << endl;
		return;
	}

	for(int i = 0; i < 30; i++) {
		multiply(link, rank, temp);	// temp = link * rank
		blas::axpby(temp, teleport, rank, beta, 1);	// rank = temp * beta + 1 * teleport
		#ifndef CPU
				cudaThreadSynchronize();
		#endif
		//cout << "==============" << i << "================" << endl;
		//print(rank);
	}
}

void print_array(CAST(ARR1D) rank) {

	for (int i = 0; i < rank.size(); i++)
		//printf ("%.10lf\n", rank[i]);
		cout << setprecision(10) << rank[i] << endl;
}

void check_normalized(CAST(COO) adj) {
	double sum[350045];
	int nodes = 350045;
	int i;

	cout << "CHECK NORMALIZED" << endl;
	for(i = 0; i < nodes; i++) sum[i] = 0.0;
	
	for(i = 0; i < adj.num_entries; i++)
		sum[adj.column_indices[i]] += adj.values[i];

	for(i = 0; i < nodes; i++)
		cout << sum[i] << endl;


	/*
	for(int i = 0; i < adj.num_rows; i++) {
		vec.row_indices[i] = 0;
		vec.column_indices[i] = i;
		vec.values[i] = 1;
	}

	multiply(vec, adj, sum);
	print(sum);*/
}


int main(int argc, char **argv) {
	CAST(COO) adj;

	read_matrix(adj, argv[1]);
	CAST(ARR1D) rank(adj.num_rows);

	CAST(ARR1D) inv_sum(adj.num_rows);
	normalize(adj, inv_sum);

	//cout << "NORMALIZED ADJ===============" << endl;
	//print(adj);

	//check_normalized(adj);

	//print(adj);
	//cout << "INVERSE ===============" << endl;
	//print(inv_sum);

	//pagerank(adj, atof(argv[2]), rank);

	//print(rank);
	//print_array(rank);
	return 0;
}
