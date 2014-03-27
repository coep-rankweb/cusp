int &take_id(const pair<int, int> &entry) {
	return entry.first;
}

int &take_freq(const pair<int, int> &entry) {
	return entry.second;
}

void get_kwd_vector(vector<int> &words, COO_GPU &word_vec) {
	thrust::copy(words.begin(), words.end(), word_vec.row_indices.begin());
	thrust::fill(word_vec.column_indices.begin(),
				 word_vec.column_indices.begin() + words.size(),
				 0);
	thrust::fill(word_vec.values.begin(),
				 word_vec.values.begin() + words.size(),
				 1);
}

void get_doc_kwd_matrix(MongoDB &mongo, set<int> &urls, COO_GPU &kwd) {
	int prev = 0;

	for(int i = 0; i < urls.size(); i++) {
		map<int, int> word_freq;

		mongo.get_word_vec(urls[i], word_freq);

		thrust::uninitialized_fill(	kwd.row_indices.begin() + prev,
									kwd.row_indices.begin() + prev + word_freq.size(),
									urls[i]);

		thrust::uninitialized_copy( boost::make_transform_iterator(map.begin(), take_id<int, int>),
									boost::make_transform_iterator(map.end(), take_id<int, int>,
									kwd.column_indices.begin());

		thrust::uninitialized_copy( boost::make_transform_iterator(map.begin(), take_freq<int, int>),
									boost::make_transform_iterator(map.end(), take_freq<int, int>,
									kwd.values.begin());

		prev += word_freq.size();
	}

	// Not sure
	kwd.resize(kwd.num_rows, kwd.num_cols, prev);
}

void get_relevance_vector(COO_GPU &kwd, COO_GPU &word_vec, COO_GPU &rel_vec) {
	multiply(kwd, word_vec, rel_vec);
}

void get_subgraph_matrix(COO_GPU &adj, COO_GPU &sub, set<int> &induced) {
	// Here we should have some qualifier for setting if the url has the word in title or ...
	COO_GPU node_vec(adj.num_rows, 1, induced.size());


	// Not necessary but for reuse
	{
		vector<int> temp(induced.begin(), induced.end());
		get_kwd_vec(temp, node_vec);
	}
	multiply(adj, node_vec, sub);
}

void normalize(COO_CPU &mat) {
	ARR1D_CPU ones(mat.num_cols, 1);
	ARR1D_CPU sums(mat.num_cols);

	multiply(relevance_mat, ones, sums);
	inversion_op<double> invert = inversion_op<double>();
	thrust::transform(sums.begin(), sums.end(), sums.begin(), invert);

	COO_CPU dia(mat.num_rows, mat.num_cols, sums.size());
	thrust::sequence(dia.row_indices.begin(), dia.row_indices.end());
	thrust::sequence(dia.column_indices.begin(), dia.column_indices.end());
	thrust::copy(sums.begin(), sums.end(), dia.values.begin());

	COO_CPU temp;
	multiply(mat, dia, temp);
	mat = temp;
}

void get_link_matrix(COO_GPU &sub, COO_GPU &rel_vec, COO_GPU &link) {
	multiply(sub, rel_vec, link);
	normalize(link);
}

void get_intelligent_mat(Mongo &mongo, COO_GPU &adj, COO_GPU &link, set<int> &urls, vector<int> &words) {
	/*
		urls = base_urls U induced_urls
	*/
	COO_GPU rel_vec(mongo.get_url_count(), 1, urls.size());
	{
		COO_GPU word_vec(mongo.get_word_count(), 1, word.size());
		get_kwd_vec(words, word_vec);

		// nnz are not correct
		COO_GPU kwd(mongo.get_url_count(), mongo.get_word_count(), urls.size() * words.size());
		get_doc_kwd_matrix(mongo, urls, kwd);

		get_relevance_vector(kwd, word_vec, rel_vec);
	}

	COO_GPU sub;
	get_subgraph_matrix(adj, sub, urls);
	get_link_matrix(sub, rel_vec, link);
}
