#include "defines.h"

#include <cusp/io/matrix_market.h>
#include <cusp/print.h>
#include <cusp/coo_matrix.h>
#include <cusp/array1d.h>
#include <cusp/multiply.h>

#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>

#ifdef CPU
	#define CAST(X)	X ## _CPU
	#define MEM	cusp::host_memory
#else
	#define CAST(X)	X ## _GPU
	#define MEM	cusp::device_memory
#endif


using namespace thrust;
using namespace cusp;

typedef coo_matrix<int, double, host_memory> COO_CPU;
typedef array1d <double, host_memory> ARR1D_CPU;

typedef coo_matrix<int, double, device_memory> COO_GPU;
typedef array1d <double, device_memory> ARR1D_GPU;


template <typename T> struct inversion_op {
__host__ __device__ T operator()(const T& x) const {
	return x ? (1 / x) : 1;
}
};

void read_matrix(COO_CPU &temp, const char *fname) {
	// Reads web matrix from market file
	io::read_matrix_market_file(temp, fname);
	fprintf(stderr, "Read the matrix\n");
}

void write_matrix(COO_CPU &mat, const char *fname) {
	io::write_matrix_market_file(mat, fname);
	fprintf(stderr, "Matrix file dumped\n");
}

void print_vector(vector<int> v) {
	for(int i = 0; i < v.size(); i++)
		cout << v[i] << endl;
}

void cosine_distance(COO_CPU &kwd, COO_CPU &query_vec, COO_CPU &relevance_vec) {
	// cosine dist between normalized query and each document
	multiply(kwd, query_vec, relevance_vec);
}

void create_doc_kwd_matrix(vector<int> &urls, COO_CPU &kwd) {
	vector<int> doc_vec;
	int ctr = 0;

	for(int i = 0; i < urls.size(); i++) {
		get_doc_vec(urls[i], doc_vec);
		for(int j = 0; j < doc_vec.size(); j++) {
			kwd.row_indices[ctr] = i;
			kwd.column_indices[ctr] = doc_vec[j];
			kwd.values[ctr++] = 1 / std::sqrt(doc_vec.size());
		}
	}
}


void build_query(COO_CPU &query_vec, vector<int> &tokens) {
	//builds a normalized query vector from tokens
	for(int i = 0; i < tokens.size(); i++) {
		//word ids start from 0
		query_vec.row_indices[i] = tokens[i];
		query_vec.column_indices[i] = 0;
		query_vec.values[i] = 1 / std::sqrt(tokens.size());
	}
}

void get_subgraph(COO_CPU &subgraph_matrix, vector<int> &tokens, vector<int> &induced_urls) {
	vector<int> base_urls;

	get_base_url_set(tokens, base_urls, BASE_URL_SIZE);
	print_vector(base_urls);
	return;

	get_induced_url_set(base_urls, induced_urls);

	create_doc_kwd_matrix(induced_urls, subgraph_matrix);
}

void get_relevance_matrix(COO_CPU &subgraph_matrix, COO_CPU &relevance_vec, COO_CPU &relevance_mat) {
	multiply(subgraph_matrix, relevance_vec, relevance_mat);
}

void normalize(COO_CPU &relevance_mat) {
	ARR1D_CPU ones(relevance_mat.num_cols, 1);
	ARR1D_CPU sums(relevance_mat.num_cols);

	multiply(relevance_mat, ones, sums);
	inversion_op<double> invert = inversion_op<double>();
	thrust::transform(sums.begin(), sums.end(), sums.begin(), invert);

	COO_CPU dia(relevance_mat.num_rows, relevance_mat.num_cols, sums.size());
	thrust::sequence(dia.row_indices.begin(), dia.row_indices.end());
	thrust::sequence(dia.column_indices.begin(), dia.column_indices.end());
	thrust::copy(sums.begin(), sums.end(), dia.values.begin());

	COO_CPU temp;
	multiply(relevance_mat, dia, temp);
	relevance_mat = temp;
}

int main(int argc, char *argv[]) {
	string query;
	vector<int> tokens;
	vector<int> induced_urls;
	COO_CPU adj;
	char web_graph_file[] = "/home/nvdia/kernel_panic/core/spyder/data/web.mtx";

	// initializes redis context
	initDatabase();
	int word_count = get_word_count();
	// subgraph_matrix = subgraph size x word count
	read_matrix(adj, web_graph_file);

	// gets indexed words from the query. Indexed words => words which have been assigned ids as per the crawl
	std::getline(cin, query);
	get_query_words(tokens, query);
	COO_CPU query_vec(word_count, 1, tokens.size());
	build_query(query_vec, tokens);

	//print(query_vec);

	COO_CPU subgraph_matrix(adj.num_rows, adj.num_cols, adj.num_entries);
	get_subgraph(subgraph_matrix, tokens, induced_urls);
	return 0;

	COO_CPU relevance_vec(word_count, 1, tokens.size());
	cosine_distance(subgraph_matrix, query_vec, relevance_vec);

	COO_CPU relevance_mat;
	get_relevance_matrix(subgraph_matrix, relevance_vec, relevance_mat);
	normalize(relevance_mat);

	return 0;
}
