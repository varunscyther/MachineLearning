from feature_extraction.CountVectorizer.CountVectorizer_updated import CountVectorizer
from feature_extraction.TfidfVectorizer.TfIdfVectorizer import TfIdfVectorizer


if __name__ == '__main__' :
    '''Text different feature extraction way'''
    '''Count Vectoriser'''
    corpus = [
        'AATACAT',  # 'AA', 'AT', 'TA', 'AC', 'CA', 'AT'
        'CTACCCT',  # 'CT', 'TA', 'AC', 'CC', 'CC', 'CT'
        'TACCTAC',  # 'TA', 'AC', 'CC', 'CT', 'TA', 'AC'
    ]

    correct_transformation = [
        [1, 1, 2, 1, 0, 0, 1],
        [0, 1, 0, 0, 2, 2, 1],
        [0, 2, 0, 0, 1, 1, 2],
    ]

    # case 1
    vectorizer = CountVectorizer(2)
    vectorizer.fit(corpus)
    vectorizer.transform(corpus) == correct_transformation

    # case 2
    vectorizer = CountVectorizer(2)
    print(vectorizer.fit_transform(corpus) == correct_transformation)

    # case 3
    corpus_2 = ['TCAATCAC', 'GGGGGGGGGGG', 'AAAA']
    vectorizer = CountVectorizer(2)
    vectorizer.fit(corpus)
    print(vectorizer.transform(corpus_2) == [
        [1, 1, 1, 2, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [3, 0, 0, 0, 0, 0, 0]
    ])

    '''Tf-Idf Vectoriser'''
    vectorizer = TfIdfVectorizer(2)
    vectorizer.fit(corpus)
    print(vectorizer.transform(corpus))