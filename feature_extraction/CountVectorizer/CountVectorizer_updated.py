class CountVectorizer :

    def __init__(self, ngram_size) :
        self.ngram_size = ngram_size
        self.vocab = {}

    def fit(self, corpus) :
        self.vocab.clear()
        all_token_list = []
        for x in corpus :
            tokens_list = []
            while len(x) >= self.ngram_size :
                token = x[:self.ngram_size]
                tokens_list.append(token)
                all_token_list.append(token)
                x = x[1 :]
        all_token_list = sorted(list(set(all_token_list)))
        self.vocab = {key : idx for idx, key in enumerate(all_token_list)}

    def transform(self, corpus) :
        intermediate_corpus = {}
        transformed_corpus = []
        for x in corpus :
            initial_token = x
            tokens_list = []
            while len(x) >= self.ngram_size :
                token = x[:self.ngram_size]
                tokens_list.append(token)
                x = x[1 :]
            intermediate_corpus[initial_token] = tokens_list
        for string_element in corpus :
            transformed_element_list = []
            for dic_token in sorted(self.vocab.keys()) :
                transformed_element_list.append(intermediate_corpus[string_element].count(dic_token))
            transformed_corpus.append(transformed_element_list)
        return transformed_corpus

    def fit_transform(self, corpus) :
        self.fit(corpus)
        return self.transform(corpus)
