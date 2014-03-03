#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <iostream>
#include <cstring>
#include <vector>
#include <set>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <algorithm>

//#include <thrust/functional.h>
//#include <thrust/system_error.h>
//#include <thrust/host_vector.h>
//#include <thrust/device_vector.h>
//#include <thrust/scatter.h>

#define BASE_URL_SIZE	200

using namespace std;

int initDatabase(void);
int get_word_count(void);

int get_feature_col(string token);
void get_query_features(vector<int> &, string);

int get_word_col(string token);
void get_query_words(std::vector<int> &tokens, string query);
int get_word_count(void);

int get_base_url_set(vector<int> &tokens, vector<int> &urls, int num_results);
int get_induced_url_set(vector<int> &base_urls, vector<int> &induced_urls);
int get_doc_vec(int doc_id, vector<int> &doc_vec);
