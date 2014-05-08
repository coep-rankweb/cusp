#include "page_rank.h"

#include <stdlib.h>
#include <iostream>
#include <iomanip>

#include <cusp/format.h>
#include <cusp/exception.h>

#include <thrust/sort.h>
#include <thrust/count.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>

#include <sstream>

using namespace std;


void read_matrix(CAST(COO) &temp, const char *fname) {
	cusp::io::read_matrix_market_file(temp, fname);
	// cerr << "Read the matrix\n";
}

void normalize(CAST(COO) &adj, CAST(ARR1D) &dangling) {
	CAST(ARR1D) ones(adj.num_rows, 1);
	CAST(ARR1D) sum = ones;
	cusp::detail::timer t;

	cusp::multiply(adj, ones, sum);
	// cerr << "Row sum calculated.\n";

	/*
		instantiated an inversion_op (invert) for use in transform
	*/
	t.start();
	inversion_op<double> invert = inversion_op<double>();
	thrust::transform(sum.begin(), sum.end(), sum.begin(), invert);
	// cerr << "Inversion done.\n";

	dangling_op<double> dangle = dangling_op<double>();
	thrust::transform(sum.begin(), sum.end(), dangling.begin(), dangle);
	// cerr << "Dangling nodes found.\n";

	CAST(COO) link_mat = adj;
	//cerr << "ADJ-row: " << adj.row_indices.size() << endl;
	//cerr << "ADJ-col: " << adj.column_indices.size() << endl;

	//cerr << "link-row: " << link_mat.row_indices.size() << endl;
	//cerr << "link-col: " << link_mat.column_indices.size() << endl;
	//is_valid_matrix(adj);

	cusp::transpose(adj, link_mat);
	//cerr << "LINK_MAT=======" <<  is_valid(link_mat) << endl;
	adj = link_mat;
	// cerr << "Transpose calculated.\n";

	//cerr << "L_ROW: " << adj.num_rows << endl;
	//cerr << "L_COL: " << adj.num_cols << endl;
	//cerr << "L_NNZ: " << adj.num_entries << endl;

	CAST(COO) dia(adj.num_rows, adj.num_rows, adj.num_rows);
	thrust::sequence(dia.row_indices.begin(), dia.row_indices.end());
	thrust::sequence(dia.column_indices.begin(), dia.column_indices.end());
	thrust::copy(sum.begin(), sum.end(), dia.values.begin());
	// cerr << "Diagonal Matrix Formed.\n";

	if(cusp::is_valid_matrix(adj)) {
		cusp::multiply(adj, dia, link_mat);	// link_mat = adj * dia
		adj = link_mat;
	} else {
			cout << "Invalid format!" << endl;
			exit(1);
	}
	// cerr << "Normalized\n";
	// cerr << "TIME:NORMAL: " << t.milliseconds_elapsed() << endl;
}

void print_array(CAST(ARR1D) rank) {
	for (int i = 0; i < rank.size(); i++)
		cout << setprecision(10) << rank[i] << endl;
}

void pagerank(CAST(COO) &link, double beta, CAST(ARR1D) &rank, CAST(ARR1D) &dangling) {
	int V = link.num_rows;
	double beta_V = beta / V;
	double one_minus_beta = (1 - beta) / V;
	cusp::detail::timer t;

	CAST(ARR1D) teleport(V, one_minus_beta);
	CAST(ARR1D) temp(V), prev_rank(V, 0);

	t.start();
	cusp::blas::fill(rank, 1 / (double) V);
	if(!is_valid_matrix(link)) {
		cout << "Link: Invalid format!" << endl;
		return;
	}

	do {
		// temp = link * rank
		multiply(link, rank, temp);

		// rank = temp * beta + dangling * beta_V + teleport * 1
		cusp::blas::axpbypcz(temp, dangling, teleport, rank, beta, beta_V, 1);
		#ifndef CPU
				cudaThreadSynchronize();
		#endif

		// tolerance check
		cusp::blas::axpy(rank, prev_rank, -1);
		check_above_threshold<double> check = check_above_threshold<double>();
		if(thrust::count_if(prev_rank.begin(), prev_rank.end(), check) == 0)
			break;
		prev_rank = rank;

	} while(1);
	// cerr << "TIME:PR: " << t.milliseconds_elapsed() << endl;
}
