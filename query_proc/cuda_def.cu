#include "cuda_def.h"

#include <iomanip>

using namespace std;

void read_matrix(CAST(COO) &temp, const char *fname) {
	cusp::io::read_matrix_market_file(temp, fname);
	cerr << "Read the matrix\n";
}

void normalize(CAST(COO) &adj, CAST(ARR1D) &dangling) {
	CAST(ARR1D) ones(adj.num_rows, 1);
	CAST(ARR1D) sum = ones;
	cusp::detail::timer t;

	cusp::multiply(adj, ones, sum);
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
	cusp::transpose(adj, link_mat);
	adj = link_mat;
	cerr << "Transpose calculated.\n";

	CAST(COO) dia(adj.num_rows, adj.num_rows, adj.num_rows);
	thrust::sequence(dia.row_indices.begin(), dia.row_indices.end());
	thrust::sequence(dia.column_indices.begin(), dia.column_indices.end());
	thrust::copy(sum.begin(), sum.end(), dia.values.begin());
	cerr << "Diagonal Matrix Formed.\n";

	if(cusp::is_valid_matrix(adj)) {
		cusp::multiply(adj, dia, link_mat);	// link_mat = adj * dia
		adj = link_mat;
	} else {
			cout << "Invalid format!" << endl;
			exit(1);
	}
	cerr << "Normalized\n";
	cerr << "TIME:NORMAL: " << t.milliseconds_elapsed() << endl;
}

void print_array(CAST(ARR1D) rank) {
	for (int i = 0; i < rank.size(); i++)
		cout << setprecision(10) << rank[i] << endl;
}
