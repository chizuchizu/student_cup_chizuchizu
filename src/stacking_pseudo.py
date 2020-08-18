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

from src.bert_model import hack

SEED = 2020
BASE_PATH = '../data/'
TEXT_COL = "description"
TARGET = "jobflag"
NUM_CLASS = 4
N_FOLDS = 4
# augmentation = True
memo = "non hack"
make_submit_file = False
LB_HACK = False

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
languages = ["ja", "fr", "de", "default"]
models = ["bert-base-uncased"]  # defaultはbert
calc_f1 = lambda y, p: metrics.f1_score(y, p.argmax(axis=1), average='macro')


def macro_f1(pred: np.array, data: lgb.Dataset):
    y = data.get_label()
    pred = pred.reshape(-1, len(y)).T  # -> (N, num_class)

    f1 = calc_f1(y, pred)
    return 'macro_f1', f1, True  # True means "higher is better"


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

    for model in models:
        for language in languages:
            lang_train = pd.read_csv(f"{BASE_PATH}languages/train_{language}_{model}_False.csv").iloc[:, 1:]
            lang_test = pd.read_csv(f"{BASE_PATH}languages/test_{language}_{model}_False.csv").iloc[:, 1:]
            lang_train[f"{language}_pred"] += + 1
            lang_test[f"{language}_pred"] += + 1

            columns = lang_train.columns
            columns = [model + column for column in columns]

            lang_train.columns = columns.copy()
            lang_test.columns = columns.copy()

            train_X = pd.concat([lang_train, pd.DataFrame(train_X)], axis=1)
            test_X = pd.concat([lang_test, pd.DataFrame(test_X)], axis=1)

        # train_X = np.concatenate([train_X, lang_train.values], 1)
        # test_X = np.concatenate([test_X, lang_test.values], 1)

    train_y = train['jobflag'].values - 1  # maps {1, 2, 3 ,4} -> {0, 1, 2, 3}
    return train_X, train_y, test_X


train_X, train_y, test_X = preprocess()

kfold = StratifiedKFold(n_splits=N_FOLDS)
pred = np.zeros((test_X.shape[0], N_FOLDS))
f1_score = 0
for fold, (train_idx, valid_idx) in enumerate(kfold.split(train_X, train_y)):
    X_train = train_X.loc[train_idx]
    X_valid = train_X.loc[valid_idx]
    y_train = train_y[train_idx]
    y_valid = train_y[valid_idx]

    weight = 1 / pd.DataFrame(y_train).reset_index().groupby(0).count().values
    train_weight = weight[y_train].ravel()
    # train_weight /= train_weight.sum()
    val_weight = weight[y_valid].ravel()

    d_train = lgb.Dataset(X_train, label=y_train, weight=train_weight)
    d_valid = lgb.Dataset(X_valid, label=y_valid, weight=val_weight)

    estimator = lgb.train(
        params=params,
        train_set=d_train,
        num_boost_round=1000,
        valid_sets=[d_train, d_valid],
        feval=macro_f1,
        verbose_eval=100,
        early_stopping_rounds=100,
    )
    print(fold + 1, "done")
    y_pred = estimator.predict(test_X)
    # print(y_pred)
    pred[:, fold] = hack(y_pred, LB_HACK)
    f1_score += estimator.best_score["valid_1"]["macro_f1"] / N_FOLDS

    lgb.plot_importance(estimator, importance_type="gain", max_num_features=25)
    plt.show()

pred = stats.mode(pred, axis=1)[0].flatten().astype(int)


def make_submit_file(pred, f1_score):
    test_id = pd.read_csv(BASE_PATH + "test.csv")["id"]
    submit = pd.DataFrame({'index': test_id, 'pred': pred + 1})
    # aug = "using_aug" if augmentation else "non_aug"
    submit.to_csv(f"../outputs/submit_stacking_{round(f1_score, 4)}_{memo}.csv", index=False, header=False)


if make_submit_file:
    make_submit_file(pred, f1_score)

print("f1_score: " + str(f1_score))
print("DONE")
