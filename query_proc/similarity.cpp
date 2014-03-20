#include <hiredis/hiredis.h>
#include "defines.h"

#define LOCAL	0
#define REMOTE	1


using namespace std;

redisContext *local_c, *remote_c;
redisReply *reply;
char command[256];

extern int stem(char *, int, int);

int initDatabase(void) {
	const char *hostname = "127.0.0.1";
	int port = 6379;
	struct timeval timeout = {1, 500000};

	local_c = redisConnectWithTimeout(hostname, port, timeout);
	if(!local_c) {
		perror("redisContext");
		return 1;
	} else if(local_c->err) {
		perror("connection");
		redisFree(local_c);
		return 1;
	}

	hostname = "10.1.99.15";
	port = 6379;
	timeout = {1, 500000};

	remote_c = redisConnectWithTimeout(hostname, port, timeout);
	if(!remote_c) {
		perror("redisContext");
		return 1;
	} else if(remote_c->err) {
		perror("connection");
		redisFree(remote_c);
		return 1;
	}
	return 0;
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
//WILL ONLY BE USED IF WE USE THE CLASSIFIER
int get_feature_col(string token) {

	sprintf(command, "GET FEATURE_COL:%s", token.c_str());
	reply = (redisReply *)redisCommand(c, command);

	if(!reply || reply->type == REDIS_REPLY_ERROR || reply->type == REDIS_REPLY_NIL)
		return -1;
	return atoi(reply->str);
}

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

	sprintf(command, "GET CWORD2ID:%s", token.c_str());
	reply = (redisReply *)redisCommand(remote_c, command);

	if(!reply || reply->type == REDIS_REPLY_ERROR || reply->type == REDIS_REPLY_NIL)
		return -1;
	return atoi(reply->str);
}

void get_query_words(std::vector<unsigned int> &tokens, string query) {
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
	sprintf(command, "GET CWORD_CTR");
	reply = (redisReply *)redisCommand(remote_c, command);

	if(!reply || reply->type == REDIS_REPLY_ERROR || reply->type == REDIS_REPLY_NIL)
		return -1;
	else
		return atoi(reply->str);
}

int process_command(const char *cmd, int where) {
	redisContext *c = where == LOCAL ? local_c : remote_c;
	reply = (redisReply *)redisCommand(c, command);
	if(!reply || reply->type == REDIS_REPLY_ERROR || reply->type == REDIS_REPLY_NIL)
		return -1;
	return 0;
}


int migrate_if_required(vector<unsigned int> &tokens, vector<unsigned int> &migrated) {
	int exists_on_remote;
	for(int i = 0; i < tokens.size(); i++) {
		sprintf(command, "EXISTS SORTED_WORD_IN:%u", tokens[i]);
		process_command(command, REMOTE);
		exists_on_remote = reply->integer;
		if(exists_on_remote) {
			sprintf(command, "MIGRATE 10.1.99.197 6379 SORTED_WORD_IN:%u 0 20000", tokens[i]);
			if(process_command(command, REMOTE)) return 1;
			migrated.push_back(tokens[i]);
		}
	}
	return 0;
}

int return_migrated(vector<unsigned int> &tokens) {
	for(int i = 0; i < tokens.size(); i++) {
		sprintf(command, "MIGRATE 10.1.99.15 6379 SORTED_WORD_IN:%u 0 20000", tokens[i]);
		if(process_command(command, LOCAL)) return 1;
	}
	return 0;
}

int get_base_url_set(vector<unsigned int> &tokens, vector<unsigned int> &urls, int num_results) {
	// Assume we have a set of urls sorted by the static ranks corresponding to each word. The function fills the urls vector by top <num_result> urls by rank.

	// FIXME:manually setting to local without rhyme or reason

	vector<unsigned int> migrated;
	migrate_if_required(tokens, migrated);

	cout << "CACHE MISS: " << migrated.size() << endl;

	char temp_dest[] = "dest";
	char temp_set[32];

	sprintf(command, "ZUNIONSTORE %s %d ", temp_dest, tokens.size());
	for(int i = 0; i < tokens.size(); i++) {
		sprintf(temp_set, "SORTED_WORD_IN:%u ", tokens[i]);
		strcat(command, temp_set);
	}
	strcat(command, "AGGREGATE MAX");
	if(process_command(command, LOCAL))
		return -1;

	sprintf(command, "ZREVRANGE %s 0 %d", temp_dest, num_results - 1);
	if(process_command(command, LOCAL))
		return -1;

	for(int i = 0; i < reply->elements; i++)
		urls.push_back(atoi(reply->element[i]->str));

	sprintf(command, "DELETE %s", temp_dest);
	if(process_command(command, LOCAL))
		return -1;

	return_migrated(migrated);

	return 0;
}

int get_induced_url_set(vector<unsigned int> &base_urls, vector<unsigned int> &induced_urls) {
	char temp_dest[] = "dest";
	char temp_set[32];

	sprintf(command, "SINTERSTORE %s %d ", temp_dest, 2 * base_urls.size());
	for(int i = 0; i < base_urls.size(); i++) {
		sprintf(temp_set, "IN_LINKS:%lu ", base_urls[i]);
		strcat(command, temp_set);
		sprintf(temp_set, "OUT_LINKS:%lu ", base_urls[i]);
		strcat(command, temp_set);
	}
	if(process_command(command, LOCAL))
		return -1;

	for(int i = 0; i < reply->elements; i++)
		induced_urls.push_back(atoi(reply->element[i]->str));

	return 0;
}

int get_doc_vec(int doc_id, vector<unsigned int> &doc_vec) {
	sprintf(command, "SMEMBERS URL_VECTOR:%d", doc_id); 
	if(process_command(command, LOCAL))
		return -1;
	
	for(int k = 0; k < reply->elements; k++)
		doc_vec.push_back(atoi(reply->element[k]->str));

	return 0;
}

void print_vector(vector<unsigned int> v) {
	for(int i = 0; i < v.size(); i++)
		cout << v[i] << endl;
}
void print_str_vector(vector<string> v) {
	cout << v.size() << endl;
	for(int i = 0; i < v.size(); i++)
		cout << v[i] << endl;
}

int get_url_names(vector<unsigned int> urls, vector<string> &names) {
	cout << urls.size() << endl;
	for(int i = 0; i < urls.size(); i++) {
		sprintf(command, "GET HASH2URL:%lu", urls[i]);
		if(process_command(command, LOCAL)) return 1;
		names.push_back(reply->str);
	}
	return 0;
}

int main() {
	initDatabase();
	vector<unsigned int> tokens, urls;
	vector<string> names;
	string query;
	while(1) {
		urls.clear();
		names.clear();
		tokens.clear();
		std::getline(cin, query);
		if(query.compare("q") == 0) break;
		get_query_words(tokens, query);
		get_base_url_set(tokens, urls, 20);
		cout << "url size: " << urls.size() << endl;
		cout << get_url_names(urls, names) << endl;
		print_str_vector(names);
		cout << "\n==========================\n\n";
	}
	return 0;
}
