#include "mongo_def.h"

using namespace std;

void inplace_intersect(map<int, double> &map_1, map<int, double> &map_2) {
	// set_1 = set_1 ^ set2

	map<int, double>::iterator it1 = map_1.begin();
	map<int, double>::iterator it2 = map_2.begin();

	while ( (it1 != map_1.end()) && (it2 != map_2.end()) ) {
		if (it1->first < it2->first) {
				map_1.erase(it1++);
		} else if (it2->first < it1->first) {
				++it2;
		} else {
				++it1;
				++it2;
		}
	}
	map_1.erase(it1, map_1.end());
}

void clean(string &s) {
	transform(s.begin(), s.end(), s.begin(), ::tolower);
}

void print_vector(vector<string> v) {
	for(int i = 0; i < v.size(); i++)
		cout << v[i] << endl;
}

void print_vector(vector<int> v) {
	for(int i = 0; i < v.size(); i++)
		cout << v[i] << endl;
}

void print_vector(vector<Rank_Tuple> &v) {
	for(int i = 0; i < v.size(); i++)
		cout << v[i].first << "\t" << v[i].second << endl;
}

void print_vector(vector<pair<string, string> > &v) {
	for(int i = 0; i < v.size(); i++)
		cout << v[i].first << "\t" << v[i].second << "\r\n";
}

void MyBSON::getArray(set<int> &urls, char *pos) {
	bson_iterator i, sub;

	bson_find(&i, &obj, pos);
    bson_iterator_subiterator(&i, &sub);

	while(bson_iterator_more(&sub))
		if(bson_iterator_next(&sub) != BSON_EOO)
			urls.insert(bson_iterator_int(&sub));
}

void MyBSON::getDictArray(map<int, double> &word_freq, char *pos, char *field1, char *field2) {
    bson_iterator i, sub, temp_i;
	int type;

	bson_find(&i, &obj, pos);
	bson_iterator_subiterator(&i, &sub);
	while(bson_iterator_more(&sub))
		if((type = bson_iterator_next(&sub)) != BSON_EOO) {
			bson temp;
			int first, second;

			bson_iterator_subobject_init(&sub, &temp, 0);
			MyBSON sub_object(temp);

			if(sub_object.getValue(first, field1) == -1)
				continue;
			sub_object.getValue(second, field2);
			word_freq[first] = double(second);

			/*
			bson_iterator_init(&temp_i, &temp);
			while(bson_iterator_more(&temp_i))
				if(bson_iterator_next(&temp_i) != BSON_EOO)
					word_freq[atoi(bson_iterator_key(&temp_i))] = bson_iterator_int(&temp_i);
			*/
		}

}

int MyBSON::getValue(string &url, char *key) {
	bson_iterator i;

	if(bson_find(&i, &obj, key) == BSON_EOO)
		return -1;
	url = bson_iterator_string(&i);
	return 0;
}

int MyBSON::getValue(double &rank, char *key) {
	bson_iterator i;

	if(bson_find(&i, &obj, key) == BSON_EOO)
		return -1;
	rank = bson_iterator_double(&i);
	return 0;
}

int MyBSON::getValue(int &id, char *key) {
	bson_iterator i;

	if(bson_find(&i, &obj, key) == BSON_EOO)
		return -1;
	id = bson_iterator_int(&i);
	return 0;
}

void MongoDB::find_one(MyBSON &query, MyBSON &field, MyBSON &res, char *col) {
	mongo_find_one(&conn, col, &query.obj, &field.obj, &res.obj);
}

int MongoDB::get_wordid(string &token) {
	MyBSON query(DOIT), field(DOIT), res;
	int val;

	field.bson_append("_id", 1);
	field.finish();
	query.bson_append("word", token);
	query.finish();
	find_one(query, field, res, "SPIDER_DB.PROC_WORD_DATA");

	if(res.getValue(val, "_id") == -1)
		val = -1;
	return val - 1;	// return base 0 id
}

int MongoDB::get_query_words(vector<int> &tokens, string &query) {
	std::istringstream iss(query);
	string token;
	int stat;

	while(getline(iss, token, ' ')) {
		clean(token);
		stat = get_wordid(token);
		if(stat != -1)
			tokens.push_back(stat);
	}
	return 0;
}


void MongoDB::get_urls_from_words(vector<int> &words, vector<Rank_Tuple> &url_rank) {
	MyBSON field(DOIT);
	map<int, double> final;

	field.bson_append("present_in", 1);
	field.finish();
	for(int i = 0; i < words.size(); i++) {
		MyBSON res, query(DOIT);
		map<int, double> temp;

		query.bson_append("_id", words[i] + 1);	// words contain base 0 ids
		query.finish();
		find_one(query, field, res, "SPIDER_DB.PROC_WORD_DATA");
		
		// queries return base 1 ids
		res.getDictArray(temp, "present_in", "url", "score");
		if(i)
			inplace_intersect(final, temp);
		else
			final = temp;
	}

	map<int, double>::iterator it = final.begin();
	while(it != final.end()) {
		url_rank.push_back(Rank_Tuple(it->first - 1, it->second));
		++it;
	}
}

void MongoDB::get_final_ranks(vector<Rank_Tuple> &url_rank) {
	MyBSON field(DOIT);

	field.bson_append("rank", 1);
	field.finish();
	for(vector<Rank_Tuple>::iterator it = url_rank.begin(); it != url_rank.end(); ++it) {
		MyBSON res, query(DOIT);
		double rank;

		query.bson_append("_id", it->first + 1); //get base 1 id
		query.finish();
		find_one(query, field, res, "SPIDER_DB.PROC_URL_DATA");
		if(res.getValue(rank, "rank") == -1)
			rank = DBL_MIN;

		it->second += rank;
	}

	sort(url_rank.begin(), url_rank.end(), pair_compare);
}

void MongoDB::get_outlinks_from_urls(vector<Rank_Tuple> &urls, set<int> &induced) {
	MyBSON query(DOIT), field(DOIT);

	field.bson_append("out_links", 1);
	field.finish();
	for(int i = 0; i < urls.size() && i < LIM; i++) {
		MyBSON res;

		query.bson_append("_id", urls[i].first + 1);	// get base 1 id
		query.finish();
		find_one(query, field, res, "SPIDER_DB.PROC_URL_DATA");
		res.getArray(induced, "out_links");
	}
}

void MongoDB::get_word_vec(int url_id, map<int, double> &word_vec) {
	MyBSON field(DOIT), query(DOIT), res;

	field.bson_append("word_vec", 1);
	field.finish();
	query.bson_append("_id", url_id + 1); // get base 1 id
	query.finish();

	find_one(query, field, res, "SPIDER_DB.PROC_URL_DATA");
	res.getDictArray(word_vec, "word_vec", "word", "score");
}

void MongoDB::get_url_names_from_ids(vector<Rank_Tuple> &url_id, vector<pair<string, string> > &url_names, int offset = 0, int limit = 10) {
	MyBSON field(DOIT);
	int top_limit = offset + limit;

	field.bson_append("url", 1);
	field.bson_append("title", 1);
	field.finish();
	for(int i = offset; i < url_id.size() && i < top_limit ; ++i) {
		MyBSON res, query(DOIT);
		string temp;
		pair<string, string> item;

		query.bson_append("_id", url_id[i].first + 1);	// get base 1 id
		query.finish();
		find_one(query, field, res, "SPIDER_DB.PROC_URL_DATA");
		if(res.getValue(temp, "url") == -1)
			item.first = "";
		else
			item.first = temp;
		if(res.getValue(temp, "title") == -1)
			 item.second = "";
		else
			item.second = temp;
		url_names.push_back(item);
	}
}


/*
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
	mongo.get_url_names_from_ids(url_rank, url_names);

	print_vector(url_names);
	return 0;
}*/
