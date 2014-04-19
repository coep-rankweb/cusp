#include "mongo_def.h"
#include "page_rank.h"

#include <fstream>

#define QUERY_PIPE	"/home/nvidia/query_pipe"
#define RESULT_PIPE	"/home/nvidia/result_pipe"

#include <thrust/find.h>
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


void get_kwd_vector(vector<int> &words, CAST(ARR1D) &word_vec) {
	for(int i = 0; i < words.size(); i++)
		word_vec[words[i]] = 1;

	/*
	   thrust::copy(words.begin(), words.end(), word_vec.row_indices.begin());
	   thrust::fill(word_vec.column_indices.begin(),
	   word_vec.column_indices.begin() + words.size(),
	   0);
	   thrust::fill(word_vec.values.begin(),
	   word_vec.values.begin() + words.size(),
	   1);
	 */
}

void get_doc_kwd_matrix(MongoDB &mongo, set<int> &urls, CAST(COO) &kwd) {
	int prev = 0;

	for(set<int>::iterator it = urls.begin(); it != urls.end(); ++it) {
		map<int, double> word_freq;

		if (mongo.get_word_vec(*it, word_freq) == -1)
			continue;

		//cout << "MAP: " << *it << endl;
		//for (map<int, double>::iterator mapit = word_freq.begin(); mapit != word_freq.end(); ++mapit)
		//	cout << mapit->first << ": " << mapit->second << endl;

		//cout << "AFTER MAPS\n";
		thrust::uninitialized_fill(	kwd.row_indices.begin() + prev,
				kwd.row_indices.begin() + prev + word_freq.size(),
				*it);

		thrust::uninitialized_copy( thrust::make_transform_iterator(word_freq.begin(), take_id()),
				thrust::make_transform_iterator(word_freq.end(), take_id()),
				kwd.column_indices.begin() + prev);

		thrust::uninitialized_copy( thrust::make_transform_iterator(word_freq.begin(), take_score()),
				thrust::make_transform_iterator(word_freq.end(), take_score()),
				kwd.values.begin() + prev);

		prev += word_freq.size();
	}

	cerr << "NNZ(prev): " << prev << endl;

	// Not sure
	//cout << "INV\n";
	kwd.resize(kwd.num_rows, kwd.num_cols, prev);
	if(!is_valid_matrix(kwd, cerr)) {
		cerr << "Invalid Matrix\n";
	}
}

void get_relevance_vector(CAST(COO) &kwd, CAST(ARR1D) &word_vec, CAST(ARR1D) &rel_vec) {
	// cosine distance
	cusp::multiply(kwd, word_vec, rel_vec);
}

void get_subgraph_matrix(CAST(COO) &adj, CAST(COO) &sub, set<int> &induced) {
	// Here we should have some qualifier for setting if the url has the word in title or ...
	CAST(COO) dia(adj.num_rows, adj.num_rows, induced.size());

	thrust::copy(induced.begin(), induced.end(), dia.row_indices.begin());
	thrust::copy(induced.begin(), induced.end(), dia.column_indices.begin());
	thrust::fill(dia.values.begin(), dia.values.begin() + induced.size(), 1);

	cusp::multiply(adj, dia, sub);
}

void get_link_matrix(CAST(COO) &sub, CAST(ARR1D) &rel_vec, CAST(COO) &link) {
	CAST(COO) dia(sub.num_rows, sub.num_rows, rel_vec.size());

	//thrust::copy(rel_vec.row_indices.begin(), rel_vec.row_indices.end(), dia.row_indices.begin());
	//thrust::copy(rel_vec.row_indices.begin(), rel_vec.row_indices.end(), dia.column_indices.begin());
	thrust::sequence(dia.row_indices.begin(), dia.row_indices.end());
	thrust::sequence(dia.column_indices.begin(), dia.column_indices.end());
	thrust::copy(rel_vec.begin(), rel_vec.end(), dia.values.begin());

	cusp::multiply(sub, dia, link);
	//cerr << "L_ROW: " << link.num_rows << endl;
	//cerr << "L_COL: " << link.num_cols << endl;
	//cerr << "L_NNZ: " << link.num_entries << endl << endl;

	//cerr << "L_values: " << link.values.size() << endl;
}

int get_intelligent_mat(MongoDB &mongo, CAST(COO) &adj, CAST(COO) &link, string &query, set<int> &urls, CAST(ARR1D) &dangling) {
	/*
	   urls = base_urls U induced_urls
	 */
	vector<int> words;
	{
		int retval;
		vector<Rank_Tuple> url_rank;

		retval = mongo.get_query_words(words, query);
		if (retval == -1)
			return -1;

		mongo.get_urls_from_words(words, url_rank);
		cerr << "Got the base urls: " << url_rank.size() << endl;
		mongo.get_final_ranks(url_rank);
		mongo.get_outlinks_from_urls(url_rank, urls);
		cerr << "Got the induced urls: " << urls.size() << endl;
	}

	CAST(ARR1D) rel_vec(mongo.get_url_count());

	{
		CAST(ARR1D) word_vec(mongo.get_word_count(), 0);
		//CAST(COO) word_vec(mongo.get_word_count(), 1, words.size());
		get_kwd_vector(words, word_vec);
		cerr << "Got Query Vector\n";

		// nnz are not correct
		CAST(COO) kwd(mongo.get_url_count(), mongo.get_word_count(), 1000000);
		get_doc_kwd_matrix(mongo, urls, kwd);
		cerr << "Got doc_kwd matrix\n";
		//cusp::print(kwd);

		// TODO: check rel_vec dimension
		get_relevance_vector(kwd, word_vec, rel_vec);
		cerr << "Got cosine distances\n";
	}

	//cusp::print(rel_vec);

	CAST(COO) sub;
	get_subgraph_matrix(adj, sub, urls);
	get_link_matrix(sub, rel_vec, link);

	normalize(link, dangling);
	return 0;
}

int main(int argc, char **argv) {
	CAST(COO) adj;
	MongoDB mongo;
	int retval;
	double beta;

	if(argc != 3) {
		cerr << "Usage: " << argv[0] << " <web_graph> <beta>" << endl;
		return 1;
	}

	// argv[1] is web matrix filename
	read_matrix(adj, argv[1]);

	beta = atof(argv[2]);

	cout << "Node: " << mongo.get_url_count() << endl;
	cout << "Word: " << mongo.get_word_count() << endl;

	while(1)
	{
		string query;
		CAST(COO) link(adj.num_rows, adj.num_cols, 1000000);
		CAST(ARR1D) dangling(adj.num_rows), rank(adj.num_rows);
		set<int> urls;

		ifstream query_fifo;
		ofstream result_fifo;
		query_fifo.open(QUERY_PIPE, ifstream::in);
		result_fifo.open(RESULT_PIPE, ofstream::out);

		result_fifo.rdbuf()->pubsetbuf(0, 0);

		cerr << "Query: ";
		getline(query_fifo, query);
		if(!query_fifo.good())
			break;

		/*
		if (cin.eof()) {
			cout << endl << "EXIT\n";
			break;
		}*/


		retval = get_intelligent_mat(mongo, adj, link, query, urls, dangling);
		if (retval == -1) {
			query_fifo.close();
			result_fifo.close();
			continue;
		}
		pagerank(link, beta, rank, dangling);

		ARR1D_CPU cpu_rank = rank;
		vector<Rank_Tuple> url_rank;
		for(set<int>::iterator it = urls.begin(); it != urls.end(); ++it)
			url_rank.push_back(pair<int, double> (*it, cpu_rank[*it]));

		sort(url_rank.begin(), url_rank.end(), pair_compare);

		vector<pair<string, string> > result;
		mongo.get_url_names_from_ids(url_rank, result, 0, 99);

		for(int i = 0; i < result.size(); i++) {
			//result_fifo << "hello" << endl << "goodbye" << endl << endl;
			result_fifo << result[i].first << "<@@@>" << result[i].second << "<###>";
		}
		cerr << "Done writing to FIFO" << endl;

		query_fifo.close();
		result_fifo.close();
	}
	
}
