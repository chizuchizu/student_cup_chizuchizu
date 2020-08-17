import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
import optuna.integration.lightgbm as lgb

BASE_PATH = '../data/'
train = pd.read_csv(BASE_PATH + "train.csv").drop(['id'], axis=1)  # ["description"]
test = pd.read_csv(BASE_PATH + "test.csv").drop(["id"], axis=1)  # ["description"]

sentences = pd.concat([train["description"], test["description"]])

cv = CountVectorizer()
cv.fit(sentences)
X_train = cv.transform(train["description"])
X_test = cv.transform(test["description"])

train_voc_df = pd.DataFrame(X_train.toarray(), columns=cv.get_feature_names())
train_X = pd.concat([train_voc_df, train], axis=1).drop(['description','jobflag'], axis=1)
train_y = train['jobflag'] -1
test_voc_df = pd.DataFrame(X_test.toarray(), columns=cv.get_feature_names())
test_X = pd.concat([test_voc_df, test], axis=1).drop(['description'], axis=1)

calc_f1 = lambda y, p: metrics.f1_score(y, p.argmax(axis=1), average='macro')


def macro_f1(pred: np.array, data: lgb.Dataset):
    y = data.get_label()
    pred = pred.reshape(-1, len(y)).T  # -> (N, num_class)

    f1 = calc_f1(y, pred)
    return 'macro_f1', f1, True  # True means "higher is better"

weight = 1 / pd.DataFrame(train_y.values).reset_index().groupby(0).count().values
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
# pd.DataFrame(pred, index=test_index).to_csv('count_vec.csv', header=False)