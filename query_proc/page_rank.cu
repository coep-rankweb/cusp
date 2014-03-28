#include "cuda_def.h"

#include <stdlib.h>
#include <iostream>

using namespace std;

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
	cerr << "TIME:PR: " << t.milliseconds_elapsed() << endl;
}
/*
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
	print_array(rank);
	return 0;
}*/
