#include "mongo_def.h"
#include "cuda_def.h"

#include <thrust/uninitialized_fill.h>
#include <thrust/uninitialized_copy.h>
#include <thrust/iterator/transform_iterator.h>

using namespace std;

struct take_id : public thrust::unary_function<pair<const int, double>&, int>{
	__host__ __device__
		int operator()(pair<const int, double> &x) const
		{
			return x.first;
		}
};

struct take_score : public thrust::unary_function<pair<const int, double> &, int>
{
	__host__ __device__
		int operator()(pair<const int, double> &x) const
		{
			return int(x.second);
		}
};

/*
int take_id(const Pair<int, int> &entry) {
	return entry.first;
}

int take_freq(const Pair<int, int> &entry) {
	return entry.second;
}
*/

void get_kwd_vector(vector<int> &words, CAST(COO) &word_vec) {
	thrust::copy(words.begin(), words.end(), word_vec.row_indices.begin());
	thrust::fill(word_vec.column_indices.begin(),
				 word_vec.column_indices.begin() + words.size(),
				 0);
	thrust::fill(word_vec.values.begin(),
				 word_vec.values.begin() + words.size(),
				 1);
}

void get_doc_kwd_matrix(MongoDB &mongo, set<int> &urls, CAST(COO) &kwd) {
	int prev = 0;

	for(set<int>::iterator it = urls.begin(); it != urls.end(); ++it) {
		map<int, double> word_freq;

		mongo.get_word_vec(*it, word_freq);

		thrust::uninitialized_fill(	kwd.row_indices.begin() + prev,
									kwd.row_indices.begin() + prev + word_freq.size(),
									*it);

		thrust::uninitialized_copy( thrust::make_transform_iterator(word_freq.begin(), take_id()),
									thrust::make_transform_iterator(word_freq.end(), take_id()),
									kwd.column_indices.begin());

		thrust::uninitialized_copy( thrust::make_transform_iterator(word_freq.begin(), take_score()),
									thrust::make_transform_iterator(word_freq.end(), take_score()),
									kwd.values.begin());

		prev += word_freq.size();
	}

	// Not sure
	kwd.resize(kwd.num_rows, kwd.num_cols, prev);
}

void get_relevance_vector(CAST(COO) &kwd, CAST(COO) &word_vec, CAST(COO) &rel_vec) {
	// cosine distance
	cusp::multiply(kwd, word_vec, rel_vec);
}

void get_subgraph_matrix(CAST(COO) &adj, CAST(COO) &sub, set<int> &induced) {
	// Here we should have some qualifier for setting if the url has the word in title or ...
	CAST(COO) node_vec(adj.num_rows, 1, induced.size());


	// Not necessary but for reuse
	{
		vector<int> temp(induced.begin(), induced.end());
		get_kwd_vector(temp, node_vec);
	}
	cusp::multiply(adj, node_vec, sub);
}

void get_link_matrix(CAST(COO) &sub, CAST(COO) &rel_vec, CAST(COO) &link) {
	cusp::multiply(sub, rel_vec, link);
}

void get_intelligent_mat(MongoDB &mongo, CAST(COO) &adj, CAST(COO) &link, string &query, set<int> &urls, CAST(ARR1D) &dangling) {
	/*
		urls = base_urls U induced_urls
	*/
	vector<int> words;
	{
		vector<Rank_Tuple> url_rank;

		mongo.get_query_words(words, query);
		mongo.get_urls_from_words(words, url_rank);
		mongo.get_final_ranks(url_rank);
		mongo.get_outlinks_from_urls(url_rank, urls);
	} 

	CAST(COO) rel_vec(mongo.get_url_count(), 1, urls.size());

	{

		CAST(COO) word_vec(mongo.get_word_count(), 1, words.size());
		get_kwd_vector(words, word_vec);

		// nnz are not correct
		CAST(COO) kwd(mongo.get_url_count(), mongo.get_word_count(), urls.size() * words.size());
		get_doc_kwd_matrix(mongo, urls, kwd);

		get_relevance_vector(kwd, word_vec, rel_vec);
	}

	CAST(COO) sub;
	get_subgraph_matrix(adj, sub, urls);
	get_link_matrix(sub, rel_vec, link);
	normalize(link, dangling);
}

int main(int argc, char **argv) {
	CAST(COO) adj;
	MongoDB mongo;
	double beta;

	// argv[1] is web matrix filename
	read_matrix(adj, argv[1]);

	beta = atof(argv[2]);

	{
		string query;
		CAST(COO) link;
		CAST(ARR1D) dangling, rank;
		set<int> urls;

		getline(cin, query);
		get_intelligent_mat(mongo, adj, link, query, urls, dangling);
		pagerank(link, beta, rank, dangling);

		ARR1D_CPU cpu_rank = rank;
		vector<Rank_Tuple> url_rank;
		for(set<int>::iterator it = urls.begin(); it != urls.end(); ++it)
			url_rank.push_back(Rank_Tuple(*it, cpu_rank[*it]));

		sort(url_rank.begin(), url_rank.end(), pair_compare);

		vector<pair<string, string> > result;
		mongo.get_url_names_from_ids(url_rank, result, 0, 10);

		for(int i = 0; i < result.size(); i++)
			cout << result[i].first << endl << result[i].second << endl << endl;
	}
}
