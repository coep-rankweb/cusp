#include "page_rank.h"
#include "mongo_def.h"

using namespace std;

extern void print_vector(vector<pair<string, string> > &);

int main(int argc, char *argv[]) {
	MongoDB mongo;
	vector<Rank_Tuple> url_rank;
	vector<int> tokens;
	vector<pair<string, string> > url_names;
	string query;

	url_rank.clear();
	url_names.clear();
	tokens.clear();
	query = argv[1];
	if(query.compare("q") == 0) return 1;
	mongo.get_query_words(tokens, query);

	mongo.get_urls_from_words(tokens, url_rank);
	mongo.get_final_ranks(url_rank);
	mongo.get_url_names_from_ids(url_rank, url_names, 0, 10);

	print_vector(url_names);
	return 0;
}
