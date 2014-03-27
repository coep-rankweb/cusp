#include <iostream>
#include <algorithm>
#include <map>
#include <set>
#include <tuple>
#include <utility>	//pair
#include <vector>
#include <string>
#include <sstream>
#include <cfloat>

#include <bson.h>
#include <mongo.h>

#define DOIT	1

#define URL_WT 		(1 / 1.0)
#define TITLE_WT 	(1 / 2.0)
#define BODY_WT 	(1 / 5.0)

using namespace std;

typedef pair<int, double> Rank_Tuple;

void inplace_intersect(map<int, int> &map_1, map<int, int> &map_2) {
	// set_1 = set_1 ^ set2

	map<int, int>::iterator it1 = map_1.begin();
	map<int, int>::iterator it2 = map_2.begin();

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

bool pair_compare(const Rank_Tuple &i, const Rank_Tuple &j) {
	return i.second > j.second;
}

class MongoDB;
class MyBSON {
	bson obj;
	friend class MongoDB;

public:
	MyBSON() {
	}

	MyBSON(int doit) {
		bson_init(&obj);
	}

	MyBSON(bson &user_obj) : obj(user_obj) {
	}

	~MyBSON() {
		bson_destroy(&obj);
	}

	void bson_append(char *key, int val) {
		bson_append_int(&obj, key, val);
	}

	void bson_append(char *key, string &val) {
		bson_append_string(&obj, key, val.c_str());
	}

	void finish() {
		bson_finish(&obj);
	}

	/*
		get the values of the type: "pos" : [...]
	*/
	void getArray(set<int> &, char *pos);

	/*
		get the values of the type: "pos" : [
										{"field1": val,
										 "field2": val
										}, ...
									]
	*/
	void getDictArray(map<int, int> &, char *pos, char *field1, char *field2);

	/*
		get the value of tthe type: {"key1":val, "key2":val, ..., "key":val, ...}
	*/
	int getValue(string &url, char *key);
	int getValue(double &rank, char *key);
	int getValue(int &rank, char *key);
};

void MyBSON::getArray(set<int> &urls, char *pos) {
	bson_iterator i, sub;

	bson_find(&i, &obj, pos);
    bson_iterator_subiterator(&i, &sub);

	while(bson_iterator_more(&sub))
		if(bson_iterator_next(&sub) != BSON_EOO)
			urls.insert(bson_iterator_int(&sub));
}

void MyBSON::getDictArray(map<int, int> &word_freq, char *pos, char *field1, char *field2) {
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
			word_freq[first] = second;

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

class MongoDB {
	mongo conn;
	int url_count, word_count;

	void find_one(MyBSON &query, MyBSON &field, MyBSON &res, char *col);
public:
	MongoDB(char *host = "127.0.0.1", int port = 27017) {
		int status = mongo_client(&conn, host, port);

		if(status != MONGO_OK) {
			switch (conn.err) {
				case MONGO_CONN_NO_SOCKET:  printf("no socket\n"); exit(1);
				case MONGO_CONN_FAIL:       printf("connection failed\n"); exit(1);
				case MONGO_CONN_NOT_MASTER: printf("not master\n"); exit(1);
			}
		}

		url_count = int(mongo_count(&conn, "SPIDER_DB", "PROC_URL_DATA", NULL));
		word_count = int(mongo_count(&conn, "SPIDER_DB", "PROC_WORD_DATA", NULL));
	}
	~MongoDB() {
		mongo_destroy(&conn);
	}

	int get_word_count() {
		return word_count;
	}

	int get_url_count() {
		return url_count;
	}

	// Get sorted (url_id, rank) pair from a vecotr of words in the query
	void get_urls_from_words(vector<int> &words, vector<Rank_Tuple> &url_rank, char *pos);

	// Get the url_names from the sorted vector of (id, rank) pairs in the given range
	void get_url_names_from_ids(vector<Rank_Tuple> &url_id, vector<pair<string, string> > &url_names, int offset, int limit);

	// Get set of out_links of the url_ids in the vector
	void get_outlinks_from_urls(vector<Rank_Tuple> &urls, set<int> &induced);

	// Get the word_ids corresponding to the words in the query
	int get_query_words(vector<int> &words, string query);

	// Get the dictionary of (word_id, freq) corresponding to url_id
	void get_word_vec(int url_id, map<int, int> &word_vec);

	// Get word_id corresponding to given word
	int get_wordid(string &token);

	void get_urls_from_ids(vector<Rank_Tuple> &url_id, vector<string> &url_names, int offset, int limit);
	void get_urls_from_words(vector<int> &words, vector<Rank_Tuple> &url_rank);
	void get_final_ranks(vector<Rank_Tuple> &url_rank);
};

void MongoDB::find_one(MyBSON &query, MyBSON &field, MyBSON &res, char *col) {
	mongo_find_one(&conn, col, &query.obj, &field.obj, &res.obj);
}

void MongoDB::get_urls_from_words(vector<int> &words, vector<Rank_Tuple> &url_rank, char *pos) {
	MyBSON field1(DOIT);
	map<int, int> urls; //url_freq

	field1.bson_append(pos, 1);
	field1.finish();
	for(int i = 0; i < words.size(); i++) {
		MyBSON res, query(DOIT);
		map<int, int> temp;

		query.bson_append("_id", words[i]);
		query.finish();
		find_one(query, field1, res, "SPIDER_DB.PROC_WORD_DATA");
		res.getDictArray(temp, pos, "url", "freq");

		if(i == 0)
			urls = temp;
		else
			inplace_intersect(urls, temp);
	}

	MyBSON field(DOIT);

	field.bson_append("rank", 1);
	field.finish();
	for(map<int, int>::iterator it = urls.begin(); it != urls.end(); ++it) {
		MyBSON res, query(DOIT);
		double rank;
		Rank_Tuple temp;

		query.bson_append("_id", it->first);
		query.finish();
		find_one(query, field, res, "SPIDER_DB.PROC_URL_DATA");
		if(res.getValue(rank, "rank") == -1)
			rank = DBL_MIN;

		temp.first = it->first;
		temp.second = rank;
		url_rank.push_back(temp);
	}

	sort(url_rank.begin(), url_rank.end(), pair_compare);
}

void upsert(tuple<int, int, int> &temp, map<int, tuple<int, int, int> > &url_rank, int id) {
	map<int, tuple<int, int, int> >::iterator it;

	it = url_rank.find(id);
	if(it == url_rank.end()) {
		url_rank[id] = temp;
	}
	else {
		tuple<int, int, int> temp1 = it->second;
		url_rank[id] = tuple<int, int, int>(get<0>(temp) + get<0>(temp1),
											get<1>(temp) + get<1>(temp1),
											get<2>(temp) + get<2>(temp1));
	}
}

void three_way_merge(map<int, int> &first, map<int, int> &second, map<int, int> &third, map<int, tuple<int, int, int> > &url_rank) {

	map<int, int>::iterator it1 = first.begin();
	map<int, int>::iterator it2 = second.begin();
	map<int, int>::iterator it3 = third.begin();

	while ((it1 != first.end()) && (it2 != second.end())) {
		tuple<int, int, int> temp;

		if(it1->first == it2->first) {
			temp = make_tuple(it1->second, it2->second, 0);
			upsert(temp, url_rank, it1->first);
			it1++;
			it2++;
		}
		else if(it1->first < it2->first) {
			temp = make_tuple(it1->second, 0, 0);
			upsert(temp, url_rank, it1->first);
			it1++;
		}
		else {
			temp = make_tuple(0, it2->second, 0);
			upsert(temp, url_rank, it2->first);
			it2++;
		}
	}

	if(it1 == first.end()) {
		while(it2 != second.end()) {
			tuple<int, int, int> temp;
			temp = make_tuple(0, it2->second, 0);
			upsert(temp, url_rank, it2->first);
			it2++;
		}
	}
	else {
		while(it1 != first.end()) {
			tuple<int, int, int> temp;
			temp = make_tuple(it1->second, 0, 0);
			upsert(temp, url_rank, it1->first);
			it1++;
		}
	}

	while(it3 != third.end()) {
		tuple<int, int, int> temp;
		temp = make_tuple(0, 0, it3->second);
		upsert(temp, url_rank, it3->first);
		it3++;
	}
}

double calculate_tf_rank(tuple<int, int, int> &item) {
	return (URL_WT * get<0>(item)) + (TITLE_WT * get<1>(item)) + (BODY_WT * get<2>(item));
}

void MongoDB::get_urls_from_words(vector<int> &words, vector<Rank_Tuple> &url_rank) {
	MyBSON field(DOIT);
	map<int, tuple<int, int, int> > url_wtf;

	field.bson_append("in_url", 1);
	field.bson_append("in_title", 1);
	field.bson_append("in_body", 1);
	field.finish();
	for(int i = 0; i < words.size(); i++) {
		MyBSON res, query(DOIT);
		map<int, int> temp1, temp2, temp3;

		query.bson_append("_id", words[i]);
		query.finish();
		find_one(query, field, res, "SPIDER_DB.PROC_WORD_DATA");

		res.getDictArray(temp1, "in_url", "url", "freq");
		res.getDictArray(temp2, "in_title", "url", "freq");
		res.getDictArray(temp3, "in_body", "url", "freq");

		three_way_merge(temp1, temp2, temp3, url_wtf);
	}

	map<int, tuple<int, int, int> >::iterator it = url_wtf.begin();
	while(it != url_wtf.end()) {
		Rank_Tuple temp;

		temp.first = it->first;
		temp.second = calculate_tf_rank(it->second);
		url_rank.push_back(temp);
		++it;
	}
}

void MongoDB::get_outlinks_from_urls(vector<Rank_Tuple> &urls, set<int> &induced) {
	MyBSON query(DOIT), field(DOIT);

	field.bson_append("out_links", 1);
	field.finish();
	for(int i = 0; i < urls.size(); i++) {
		MyBSON res;

		query.bson_append("_id", urls[i].first);
		query.finish();
		find_one(query, field, res, "SPIDER_DB.PROC_URL_DATA");
		res.getArray(induced, "out_links");
	}
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

		query.bson_append("_id", url_id[i].first);
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

void MongoDB::get_urls_from_ids(vector<Rank_Tuple> &url_id, vector<string> &url_names, int offset = 0, int limit = 10) {
	MyBSON field(DOIT);
	int top_limit = offset + limit;

	field.bson_append("url", 1);
	field.finish();
	for(int i = offset; i < url_id.size() && i < top_limit ; ++i) {
		MyBSON res, query(DOIT);
		string temp;

		query.bson_append("_id", url_id[i].first);
		query.finish();
		find_one(query, field, res, "SPIDER_DB.PROC_URL_DATA");
		if(res.getValue(temp, "url") == -1)
			temp = "";

		url_names.push_back(temp);
	}
}

void MongoDB::get_final_ranks(vector<Rank_Tuple> &url_rank) {
	MyBSON field(DOIT);

	field.bson_append("rank", 1);
	field.finish();
	for(vector<Rank_Tuple>::iterator it = url_rank.begin(); it != url_rank.end(); ++it) {
		MyBSON res, query(DOIT);
		double rank;

		query.bson_append("_id", it->first);
		query.finish();
		find_one(query, field, res, "SPIDER_DB.PROC_URL_DATA");
		if(res.getValue(rank, "rank") == -1)
			rank = DBL_MIN;

		it->second += rank;
		//url_rank[it->first] = it->second * rank;
	}

	sort(url_rank.begin(), url_rank.end(), pair_compare);
}

void MongoDB::get_word_vec(int url_id, map<int, int> &word_vec) {
	MyBSON field(DOIT), query(DOIT), res;

	field.bson_append("word_vec", 1);
	field.finish();
	query.bson_append("_id", url_id);
	query.finish();

	find_one(query, field, res, "SPIDER_DB.PROC_URL_DATA");
	res.getDictArray(word_vec, "word_vec", "word", "freq");
}

void clean(string &s) {
	transform(s.begin(), s.end(), s.begin(), ::tolower);
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
	return val;
}

int MongoDB::get_query_words(vector<int> &tokens, string query) {
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
}


/*
int main(int argc, char *argv[]) {
	MongoDB mongo;
	vector<Rank_Tuple> url_rank;
	vector<int> tokens;
	vector<pair<string, string> > url_names;
	string query;

	while(1) {
		url_rank.clear();
		url_names.clear();
		tokens.clear();
		std::getline(cin, query);
		if(query.compare("q") == 0) break;
		mongo.get_query_words(tokens, query);

		//print_vector(tokens);

		mongo.get_urls_from_words(tokens, url_rank);
		mongo.get_final_ranks(url_rank);
		mongo.get_url_names_from_ids(url_rank, url_names);

		//print_vector(url_rank);

		cout << "url size: " << url_rank.size() << endl;
		print_vector(url_names);
		cout << "\n==========================\n\n";
	}
	return 0;
}
*/
