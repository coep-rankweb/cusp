#include <stdio.h>
#include <stdlib.h>
#include <bson.h>
#include <mongo.h>

#include <iostream>
#include <algorithm>
#include <map>
#include <set>
#include <utility>
#include <vector>
#include <string>
#include <sstream>
#include <cfloat>

#define DOIT	1

#define URL_WT 		(1 / 1.0)
#define TITLE_WT 	(1 / 2.0)
#define BODY_WT 	(1 / 5.0)

#define LIM	200

using namespace std;

typedef pair<int, double> Rank_Tuple;

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
	void getDictArray(map<int, double> &, char *pos, char *field1, char *field2);

	/*
		get the value of tthe type: {"key1":val, "key2":val, ..., "key":val, ...}
	*/
	int getValue(string &url, char *key);
	int getValue(double &rank, char *key);
	int getValue(int &rank, char *key);
};

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

	// Get word_id corresponding to given word
	int get_wordid(string &token);

	// Get the word_ids corresponding to the words in the query
	int get_query_words(vector<int> &words, string &query);

	// Get sorted (url_id, tf_rank) pair from a vecotr of words in the query
	void get_urls_from_words(vector<int> &words, vector<Rank_Tuple> &url_rank);

	// Get (tf_rank + static page_rank). CHANGES url_rank
	void get_final_ranks(vector<Rank_Tuple> &url_rank);

	// Get set of out_links of the url_ids in the vector
	void get_outlinks_from_urls(vector<Rank_Tuple> &urls, set<int> &induced);

	// Get the dictionary of (word_id, freq) corresponding to url_id
	void get_word_vec(int url_id, map<int, double> &word_vec);

	// Get the url_names from the sorted vector of (id, rank) pairs in the given range
	void get_url_names_from_ids(vector<Rank_Tuple> &url_id, vector<pair<string, string> > &url_names, int offset, int limit);
};

inline bool pair_compare(const Rank_Tuple &i, const Rank_Tuple &j) {
	return i.second > j.second;
}
