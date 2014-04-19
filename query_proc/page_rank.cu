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

/*
template <typename IndexVector>
thrust::pair<typename IndexVector::value_type, typename IndexVector::value_type>
index_range(const IndexVector& indices);

template <typename MatrixType>
bool is_valid(const MatrixType &A);
*/

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
	//cerr << "ADJ-row: " << adj.row_indices.size() << endl;
	//cerr << "ADJ-col: " << adj.column_indices.size() << endl;

	//cerr << "link-row: " << link_mat.row_indices.size() << endl;
	//cerr << "link-col: " << link_mat.column_indices.size() << endl;
	//is_valid_matrix(adj);

	cusp::transpose(adj, link_mat);
	//cerr << "LINK_MAT=======" <<  is_valid(link_mat) << endl;
	adj = link_mat;
	cerr << "Transpose calculated.\n";

	//cerr << "L_ROW: " << adj.num_rows << endl;
	//cerr << "L_COL: " << adj.num_cols << endl;
	//cerr << "L_NNZ: " << adj.num_entries << endl;

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
template <typename IndexVector>
thrust::pair<typename IndexVector::value_type, typename IndexVector::value_type>
index_range(const IndexVector& indices)
{
//    // return a pair<> containing the min and max value in a range
//    thrust::pair<typename IndexVector::const_iterator, typename IndexVector::const_iterator> iter = thrust::minmax_element(indices.begin(), indices.end());
//    return thrust::make_pair(*iter.first, *iter.second);
   
    // WAR lack of const_iterator in array1d_view
	cout << "INDICES: " << indices.size() << endl;
	for (int i = 0; i < indices.size(); i++) {
		cout << i;
		cout << ": " << indices[i] << endl;
	}
	cout << endl;
	cerr << "GOT A\n";
	int b = *thrust::max_element(indices.begin(), indices.end());
	cerr << "GOT b\n";
	int a = *thrust::min_element(indices.begin(), indices.end());
	cerr << "GOT a\n";
	
	return thrust::make_pair(a, b);
}

template <typename MatrixType>
bool is_valid(const MatrixType &A)
{
    typedef typename MatrixType::index_type IndexType;

    // we could relax some of these conditions if necessary
    if (A.row_indices.size() != A.num_entries)
    {
        cerr << "size of row_indices (" << A.row_indices.size() << ") "
                << "should be equal to num_entries (" << A.num_entries << ")";
        return false;
    }

	cerr << "Hello 1\n";
    
    if (A.column_indices.size() != A.num_entries)
    {
        cerr << "size of column_indices (" << A.column_indices.size() << ") "
                << "should be equal to num_entries (" << A.num_entries << ")";
        return false;
    }
	cerr << "Hello 2\n";
    
    if (A.values.size() != A.num_entries)
    {
        cerr << "size of values (" << A.column_indices.size() << ") "
                << "should be equal to num_entries (" << A.num_entries << ")";
        return false;
    }
	cerr << "Hello 3\n";
   
    if (A.num_entries > 0)
    {
        // check that row indices are within [0, num_rows)
        thrust::pair<IndexType,IndexType> min_max_row = index_range(A.row_indices);
        if (min_max_row.first < 0)
        {
            cerr << "row indices should be non-negative";
            return false;
        }
		cerr << "Hello 4\n";
        if (static_cast<size_t>(min_max_row.second) >= A.num_rows)
        {
            cerr << "row indices should be less than num_row (" << A.num_rows << ")";
            return false;
        }
		cerr << "Hello 5\n";
        
        // check that row_indices is a non-decreasing sequence
        if (!thrust::is_sorted(A.row_indices.begin(), A.row_indices.end()))
        {
            cerr << "row indices should form a non-decreasing sequence";
            return false;
        }
		cerr << "Hello 6\n";

        // check that column indices are within [0, num_cols)
        thrust::pair<IndexType,IndexType> min_max_col = index_range(A.column_indices);
        if (min_max_col.first < 0)
        {
            cerr << "column indices should be non-negative";
            return false;
        }
		cerr << "Hello 7\n";
        if (static_cast<size_t>(min_max_col.second) >= A.num_cols)
        {
            cerr << "column indices should be less than num_cols (" << A.num_cols << ")";
            return false;
        }
		cerr << "Hello 8\n";
    }

    return true;
}
*/
