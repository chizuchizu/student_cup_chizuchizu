import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, hstack

BASE_PATH = '../data/'
train = pd.read_csv(BASE_PATH + "train.csv").drop(['id'], axis=1)["description"]
test = pd.read_csv(BASE_PATH + "test.csv")["description"]

sentences = pd.concat([train, test])

print(0)

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    analyzer="word",
    stop_words="english",
    max_features=5000,
)

word_vectorizer.fit(sentences)
# vec = pd.DataFrame(vec.toarray(), columns=word_vectorizer.get_feature_names())
print(0)

train_word_features = word_vectorizer.transform(train)
test_word_features = word_vectorizer.transform(test)

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    analyzer="char",
    stop_words="english",
    ngram_range=(2, 6),
    max_features=20000,
)
char_vectorizer.fit(sentences)

train_char_features = char_vectorizer.transform(train)
test_char_features = char_vectorizer.transform(test)

train_features = pd.DataFrame(hstack([train_char_features, train_word_features]).toarray())
test_features = pd.DataFrame(hstack([test_char_features, test_word_features]).toarray())

train_features.to_pickle("../data/train_tfidf.pkl")
test_features.to_pickle("../data/test_tfidf.pkl")