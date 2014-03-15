#include <mongo/client/dbclient.h>
#include "defines.h"

#define DB					"SPIDER_DB"
#define CRAWLER_DATA		DB##".CRAWLER_DATA"
#define URL_DATA			DB##".URL_DATA"
#define WORD_DATA			DB##".WORD_DATA"
#define COLLOCATION_DATA	DB##".COLLOCATION_DATA"

using namespace std;
using namespace mongo;

DBClientConnection c;

extern int stem(char *, int, int);

int initDatabase(void) {
	try {
		c.connect("localhost");
		return 0;
	}
	catch(const DBException &e) {
		return 1;
	}
}

static void clean(string &s) {
	int i;
	char *stemmed = new char [s.length() + 1];

	std::transform(s.begin(), s.end(), s.begin(), ::tolower);
	strcpy(stemmed, s.c_str());
	i = stem(stemmed, 0, s.length() - 1);
	stemmed[i + 1] = '\0';

	s.clear();
	s.assign(stemmed);

	delete[] stemmed;
}

/*
int get_feature_col(string token) {

	sprintf(command, "GET FEATURE_COL:%s", token.c_str());
	reply = (redisReply *)redisCommand(c, command);

	if(!reply || reply->type == REDIS_REPLY_ERROR || reply->type == REDIS_REPLY_NIL)
		return -1;
	return atoi(reply->str);
}
*/

/*
void get_query_features(std::vector<int> &tokens, string query) {
	std::istringstream iss(query);
	string token;
	int stat;

	while(getline(iss, token, ' ')) {
		clean(token);
		stat = get_feature_col(token);
		if(stat != -1)
			tokens.push_back(stat);
		else
			cerr << token << " not found!\n";
	}
}
*/

int get_word_col(string token) {
	BSONObj b = BSONObjBuilder().append("word", token.c_str()).obj();
	auto_ptr<DBClientCursor> cursor = c.query(WORD_DATA, b);

	return atoi(cursor->next()["_id"].toString())
}

void get_query_words(std::vector<int> &tokens, string query) {
	std::istringstream iss(query);
	string token;
	int stat;

	while(getline(iss, token, ' ')) {
		clean(token);
		stat = get_word_col(token);
		if(stat != -1)
			tokens.push_back(stat);
		else
			cerr << token << " not found!\n";
	}
}

int get_word_count(void) {
	return c.count(WORD_DATA, BSONObj());
}

int get_base_url_set(vector<int> &tokens, vector<int> &urls, int num_results) {
	// Assume we have a set of urls sorted by the static ranks corresponding to each word. The function fills the urls vector by top <num_result> urls by rank.

	char temp_dest[] = "dest";
	char temp_set[32];

	set::set<int> ret, temp;

	sprintf(command, "ZUNIONSTORE %s %d ", temp_dest, tokens.size());
	for(int i = 0; i < tokens.size(); i++) {
		sprintf(temp_set, "SORTED_WORD_IN:%d ", tokens[i]);
		strcat(command, temp_set);
	}
	strcat(command, "AGGREGATE MAX");
	if(process_command(command))
		return -1;

	sprintf(command, "ZREVRANGE %s 0 %d", temp_dest, num_results - 1);
	if(process_command(command))
		return -1;

	for(int i = 0; i < reply->elements; i++)
		urls.push_back(atoi(reply->element[i]->str));

	sprintf(command, "DELETE %s", temp_dest);
	if(process_command(command))
		return -1;

	return 0;
}

int get_induced_url_set(vector<int> &base_urls, vector<int> &induced_urls) {
	char temp_dest[] = "dest";
	char temp_set[32];

	sprintf(command, "SINTERSTORE %s %d ", temp_dest, 2 * base_urls.size());
	for(int i = 0; i < base_urls.size(); i++) {
		sprintf(temp_set, "IN_LINKS:%d ", base_urls[i]);
		strcat(command, temp_set);
		sprintf(temp_set, "OUT_LINKS:%d ", base_urls[i]);
		strcat(command, temp_set);
	}
	if(process_command(command))
		return -1;

	for(int i = 0; i < reply->elements; i++)
		induced_urls.push_back(atoi(reply->element[i]->str));

	return 0;
}

int get_doc_vec(int doc_id, vector<int> &doc_vec) {
	sprintf(command, "SMEMBERS DOC_VECTOR:%d", doc_id); 
	if(process_command(command))
		return -1;
	
	for(int k = 0; k < reply->elements; k++)
		doc_vec.push_back(atoi(reply->element[k]->str));

	return 0;
}
