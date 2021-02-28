"""

Term frequency–inverse document frequency vectorized -
s a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus.
[1] It is often used as a weighting factor in searches of information retrieval, text mining, and user modeling.
The tf–idf value increases proportionally to the number of times a word appears in the document and is offset by the
 number of documents in the corpus that contain the word, which helps to adjust for the fact that some words appear more
 frequently in general


 For more details refer this - https://en.wikipedia.org/wiki/Tf%E2%80%93idf

 tf(term frequency) - In its raw frequency form, tf is just the frequency of the "word" for each document. In each document,
                      the word appears once.

 idf(Inverse document frequency) - The inverse document frequency is a measure of how much information the word provides,
                                   i.e., if it's common or rare across all documents.

 tf-idf = tf(term frequency) * idf(Inverse document frequency)


"""

# Built in modules
import math


class TfIdfVectorizer :

    def __init__(self, ngram_size) :
        self.ngram_size = ngram_size
        self.vocab = {}
        self.intermediate_corpus = {}

    def fit(self, corpus) :
        self.vocab.clear()
        all_token_list = []
        intermediate_corpus = {}

        ''' Generating tokens '''
        for x in corpus :
            initial_token = x
            tokens_list = []
            while len(x) >= self.ngram_size :
                token = x[:self.ngram_size]
                tokens_list.append(token)
                all_token_list.append(token)
                x = x[1 :]
                intermediate_corpus[initial_token] = tokens_list
        self.intermediate_corpus = intermediate_corpus
        all_token_list = sorted(list(set(all_token_list)))
        self.vocab = {key : idx for idx, key in enumerate(all_token_list)}

    def transform(self, corpus) :
        transformed_corpus = []
        for document in corpus :
            transformed_element_list = []
            for term in sorted(self.vocab.keys()) :

                '''Calculate tfidf for dic token'''
                tf = self.intermediate_corpus[document].count(term) / len(self.intermediate_corpus[document])
                no_of_document_containing_term = 0
                for intermediate_document in self.intermediate_corpus :
                    if term in intermediate_document :
                        no_of_document_containing_term += 1
                idf = math.log10(len(corpus) / no_of_document_containing_term)
                transformed_element_list.append(tf * idf)
                ''' End of calculation'''

            transformed_corpus.append(transformed_element_list)
        return transformed_corpus

    def fit_transform(self, corpus) :
        self.fit(corpus)
        return self.transform(corpus)


if __name__ == '__main__' :
    '''Text different feature extraction way'''
    '''Count Vectoriser'''
    corpus = [
        'AATACAT',  # 'AA', 'AT', 'TA', 'AC', 'CA', 'AT'
        'CTACCCT',  # 'CT', 'TA', 'AC', 'CC', 'CC', 'CT'
        'TACCTAC',  # 'TA', 'AC', 'CC', 'CT', 'TA', 'AC'
    ]

    '''Tf-Idf Vectoriser'''
    vectorizer = TfIdfVectorizer(2)
    vectorizer.fit(corpus)
    print(vectorizer.transform(corpus))
