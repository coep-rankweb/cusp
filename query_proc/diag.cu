#include <cusp/detail/timer.h>
#include <cusp/io/matrix_market.h>
#include <cusp/coo_matrix.h>
#include <cusp/dia_matrix.h>
#include <cusp/verify.h>
#include <cusp/print.h>
#include <cusp/multiply.h>
#include <cusp/transpose.h>

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <iostream>
#include <iomanip>

#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>


using namespace thrust;
using namespace cusp;
using namespace std;

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

/*
	Creating a struct to define the inversion operation. 
*/
template <typename T>
struct inversion_op{
__host__ __device__ T operator()(const T& x)const{
		return x > 0 ? 1 / x : 0;
}
};

/*
	Creating a struct to define an operation to detect dangling nodes 
*/
template <typename T>
struct dangling_op{
__host__ __device__ T operator()(const T& x)const{
		return x > 0 ? 0 : 1;
}
};

void read_matrix(CAST(COO) &temp, const char *fname) {
	io::read_matrix_market_file(temp, fname);
	cerr << "Read the matrix\n";
}

void normalize(CAST(COO) &adj, CAST(ARR1D) &dangling) {
	CAST(ARR1D) ones(adj.num_rows, 1);
	CAST(ARR1D) sum = ones;
	cusp::detail::timer t;

	multiply(adj, ones, sum);
	cerr << "Row sum calculated.\n";

	/*
		instantiated an inversion_op (invert) for use in transform
	*/
	t.start();
	inversion_op<double> invert = inversion_op<double>();
	thrust::transform(sum.begin(), sum.end(), sum.begin(), invert);
	cerr << "Inversion done.\n";

	dangling_op<double> dangle = dangling_op<double>();
	thrust::transform(sum.begin(), sum.end(), dangling.begin(), dangle);
	cerr << "Dangling nodes found.\n";

	CAST(COO) link_mat = adj;
	transpose(adj, link_mat);
	adj = link_mat;
	cerr << "Transpose calculated.\n";

	CAST(COO) dia(adj.num_rows, adj.num_rows, adj.num_rows);
	thrust::sequence(dia.row_indices.begin(), dia.row_indices.end());
	thrust::sequence(dia.column_indices.begin(), dia.column_indices.end());
	thrust::copy(sum.begin(), sum.end(), dia.values.begin());
	cerr << "Diagonal Matrix Formed.\n";

	if(is_valid_matrix(adj)) {
		multiply(adj, dia, link_mat);	// link_mat = adj * dia
		adj = link_mat;
	} else {
			cout << "Invalid format!" << endl;
			exit(1);
	}
	cerr << "Normalized\n";
	cerr << "TIME:NORMAL: " << t.milliseconds_elapsed() << endl;
}


void pagerank(CAST(COO) &link, double beta, CAST(ARR1D) &rank, CAST(ARR1D) &dangling) {
	int V = link.num_rows;
	double beta_V = beta / V;
	double one_minus_beta = (1 - beta) / V;
	cusp::detail::timer t;

	CAST(ARR1D) teleport(V, one_minus_beta);
	CAST(ARR1D) temp(V);

	t.start();
	blas::fill(rank, 1 / (double) V);
	if(!is_valid_matrix(link)) {
		cout << "Link: Invalid format!" << endl;
		return;
	}

	for(int i = 0; i < 30; i++) {
		multiply(link, rank, temp);	// temp = link * rank
		blas::axpbypcz(temp, dangling, teleport, rank, beta, beta_V, 1);	// rank = temp * beta + dangling * beta_V + teleport * 1
		#ifndef CPU
				cudaThreadSynchronize();
		#endif
	}
	cerr << "TIME:PR: " << t.milliseconds_elapsed() << endl;
}

void print_array(CAST(ARR1D) rank) {

	for (int i = 0; i < rank.size(); i++)
		//printf ("%.10lf\n", rank[i]);
		cout << setprecision(10) << rank[i] << endl;
}

int main(int argc, char **argv) {
	CAST(COO) adj;
	read_matrix(adj, argv[1]);

	struct timeval start, end;

	CAST(ARR1D) rank(adj.num_rows);
	CAST(ARR1D) dangling(adj.num_rows);

	gettimeofday(&start, NULL);
	normalize(adj, dangling);

	pagerank(adj, atof(argv[2]), rank, dangling);
	gettimeofday(&end, NULL);

	cerr << "TIME: " << end.tv_sec - start.tv_sec << "." << end.tv_usec - start.tv_usec << endl;
	//print(rank);
	print_array(rank);
	return 0;
}

//#define SIZE	5

/*
int main() {
	cusp::dia_matrix<int, double, cusp::host_memory> dia(SIZE, SIZE, SIZE, 1);
	dia.diagonal_offsets[0] = 0;

	int indices[3] = {0, 3, 4};
	int values[3] = {2983, 1232, 554};
	for(int i = 0; i < SIZE; i++)
		dia.values(i, 0) = 1;
	for(int i = 0; i < 3; i++) {
		dia.values(indices[i], 0) = values[i];
	}
	cerr << "Formed dia_mat.\n";

	print(dia);

}*/
