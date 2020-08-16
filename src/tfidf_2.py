import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, hstack
from keras.preprocessing.text import Tokenizer
# import lightgbm as lgb
from sklearn import metrics
import optuna.integration.lightgbm as lgb

BASE_PATH = '../data/'
train = pd.read_csv(BASE_PATH + "train.csv").drop(['id'], axis=1)  # ["description"]
test = pd.read_csv(BASE_PATH + "test.csv").drop(["id"], axis=1)  # ["description"]

sentences = pd.concat([train["description"], test["description"]])

tokenizer = Tokenizer(
    num_words=1000,
    lower=True,

)  # 出現頻度上位{num_words}だけを用いる
tokenizer.fit_on_texts(sentences)

train_X, test_X = np.split(tokenizer.texts_to_matrix(sentences, mode='binary'),
                           [len(train)], axis=0)

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    analyzer="char",
    stop_words="english",
    ngram_range=(2, 6),
    max_features=500,
)
word_vectorizer.fit(sentences)
print(train_X)
print((word_vectorizer.transform(train["description"])).toarray())

train_X = np.concatenate([train_X, (word_vectorizer.transform(train["description"])).toarray()], 1)
test_X = np.concatenate([test_X, (word_vectorizer.transform(test["description"])).toarray()], 1)

train_y = train['jobflag'].values - 1  # maps {1, 2, 3 ,4} -> {0, 1, 2, 3}

calc_f1 = lambda y, p: metrics.f1_score(y, p.argmax(axis=1), average='macro')


def macro_f1(pred: np.array, data: lgb.Dataset):
    y = data.get_label()
    pred = pred.reshape(-1, len(y)).T  # -> (N, num_class)

    f1 = calc_f1(y, pred)
    return 'macro_f1', f1, True  # True means "higher is better"


weight = 1 / pd.DataFrame(train_y).reset_index().groupby(0).count().values
weight = weight[train_y].ravel()
weight /= weight.sum()

print(train.shape, train_X.shape, train_y.shape, test_X.shape, test.shape)

dtrain = lgb.Dataset(train_X, train_y, weight=weight)

params = {
    'objective': 'multiclass',
    'metric': 'custom',
    'num_class': 4,
    'learning_rate': 0.01,
    'max_depth': 10,
    'num_leaves': 15,
    'max_bin': 31,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'nthread': -1,
    'bagging_freq': 1,
    'verbose': -1,
    'seed': 1,
}

# githubからインストールしないと`return_cvbooster`が使えないので注意
cvbooster = lgb.cv(params, dtrain, return_cvbooster=True, stratified=False,
                   num_boost_round=9999, verbose_eval=100, early_stopping_rounds=200,
                   feval=macro_f1)['cvbooster']

test_index = pd.read_csv(BASE_PATH + "test.csv")["id"]

pred = np.stack(cvbooster.predict(test_X)).mean(axis=0).argmax(axis=1) + 1
# pd.DataFrame(pred, index=test_index).to_csv('lgb_0.569.csv', header=False)
