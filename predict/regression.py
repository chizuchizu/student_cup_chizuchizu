import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, hstack
from keras.preprocessing.text import Tokenizer
import lightgbm as lgb
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
import pickle


SEED = 2020
BASE_PATH = '../for_train_data/'
TEXT_COL = "description"
TARGET = "jobflag"
NUM_CLASS = 4
N_FOLDS = 4
# augmentation = True
memo = "non hack"
make_submit_file = False
LB_HACK = False

params = {
    'objective': 'regression',
    'metric': 'rmse',
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

def preprocess():
    train = pd.read_csv(BASE_PATH + "train.csv").drop(['id'], axis=1)  # ["description"]
    test = pd.read_csv(BASE_PATH + "test.csv").drop(["id"], axis=1)  # ["description"]

    sentences = pd.concat([train["description"], test["description"]])

    tokenizer = Tokenizer(
        num_words=2000,
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
        max_features=1000,
    )
    word_vectorizer.fit(sentences)
    # print(train_X)
    # print((word_vectorizer.transform(train["description"])).toarray())

    train_X = np.concatenate([train_X, (word_vectorizer.transform(train["description"])).toarray()], 1)
    test_X = np.concatenate([test_X, (word_vectorizer.transform(test["description"])).toarray()], 1)

    text_svd = TruncatedSVD(n_components=100, algorithm="arpack", random_state=1234)
    text_svd.fit(train_X)
    train_X = text_svd.transform(train_X)
    test_X = text_svd.transform(test_X)

    kmeans = KMeans(n_clusters=100, random_state=10).fit(np.concatenate([train_X, test_X]))
    train_X = np.concatenate([train_X, (kmeans.transform(train_X))], 1)
    test_X = np.concatenate([test_X, (kmeans.transform(test_X))], 1)


    train_y = train['jobflag'].values - 1  # maps {1, 2, 3 ,4} -> {0, 1, 2, 3}
    return train_X, train_y, test_X


train_X, train_y_, test_X = preprocess()
"""
0: DS
1: ML
2: ソフトウェアエンジニア
3: コンサルタント
"""
flag_dict = {
    0: 0,
    1: 1,
    2: 2,
    3: 3
}
flag_list = [{0: 0, 1: 1, 2: 2, 3: 3},
             {0: 0, 1: 1, 2: 3, 3: 2},
             {0: 0, 1: 2, 2: 1, 3: 3},
             {0: 0, 1: 2, 2: 3, 3: 1},
             {0: 0, 1: 3, 2: 1, 3: 2},
             {0: 0, 1: 3, 2: 2, 3: 1},

             {0: 1, 1: 0, 2: 2, 3: 3},
             {0: 1, 1: 0, 2: 3, 3: 2},
             {0: 1, 1: 2, 2: 0, 3: 3},
             {0: 1, 1: 2, 2: 3, 3: 0},
             {0: 1, 1: 3, 2: 0, 3: 2},
             {0: 1, 1: 3, 2: 2, 3: 0},

             {0: 2, 1: 0, 2: 1, 3: 3},
             {0: 2, 1: 0, 2: 3, 3: 1},
             {0: 2, 1: 1, 2: 0, 3: 3},
             {0: 2, 1: 1, 2: 3, 3: 0},
             {0: 2, 1: 3, 2: 0, 3: 1},
             {0: 2, 1: 3, 2: 1, 3: 0},

             {0: 3, 1: 0, 2: 1, 3: 2},
             {0: 3, 1: 0, 2: 2, 3: 1},
             {0: 3, 1: 1, 2: 0, 3: 2},
             {0: 3, 1: 1, 2: 2, 3: 0},
             {0: 3, 1: 2, 2: 0, 3: 1},
             {0: 3, 1: 2, 2: 1, 3: 0},
             ]
mean_score = 0
for i, flag_dict in enumerate(flag_list):
    train_y = pd.Series(train_y_).map(flag_dict).values.copy()

    kfold = StratifiedKFold(n_splits=N_FOLDS)
    pred = np.zeros(test_X.shape[0])
    oof = np.zeros(train_X.shape[0])
    f1_score = 0
    for fold, (train_idx, valid_idx) in enumerate(kfold.split(train_X, train_y)):
        X_train = train_X[train_idx]
        X_valid = train_X[valid_idx]
        y_train = train_y[train_idx]
        y_valid = train_y[valid_idx]

        weight = 1 / pd.DataFrame(y_train).reset_index().groupby(0).count().values
        train_weight = weight[y_train].ravel()
        # train_weight /= train_weight.sum()
        val_weight = weight[y_valid].ravel()

        d_train = lgb.Dataset(X_train, label=y_train, weight=train_weight)
        d_valid = lgb.Dataset(X_valid, label=y_valid, weight=val_weight)

        print(fold + 1, "done")
        file = f"../models/regression/{i}_{fold}.pkl"
        estimator = pickle.load(open(file, "rb"))
        # pickle.dump(estimator, open(file, "wb"))
        y_pred = estimator.predict(test_X)
        # print(y_pred)
        pred += y_pred / N_FOLDS

        oof[valid_idx] = estimator.predict(X_valid)
        # f1_score += estimator.best_score["valid_1"]["macro_f1"] / N_FOLDS

        # lgb.plot_importance(estimator, importance_type="gain", max_num_features=25)
        # plt.show()
        f1_score += estimator.best_score["valid_1"]["rmse"] / N_FOLDS
    print("rmse", f1_score)

    pd.Series(pred).to_csv(f"{BASE_PATH}languages/test_lgbm_{i}.csv", index=False)
    pd.Series(oof).to_csv(f"{BASE_PATH}languages/train_lgbm_{i}.csv", index=False)
    mean_score += f1_score / 24
print(mean_score)

# print(pd.Series(np.round(pred.mean(axis=1)).astype(int)).value_counts())
# print(pd.Series((pred.mean(axis=1)).astype(int)).value_counts())


# pred = stats.mode(pred, axis=1)[0].flatten().astype(int)
