import pandas as pd
import numpy as np
import pulp
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from scipy import stats

BASE_PATH = './data/'
NUM_CLASS = 4
N_FOLDS = 4
# augmentation = True
# memo = "first submit"
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

def hack(prob):
    N_CLASSES = [404, 320, 345, 674]
    
    scaler = MinMaxScaler()
    prob = scaler.fit_transform(prob)
    print(prob)

    # prob = np.where(prob < 0, 0, prob)
    logp = np.log(prob + 1e-8)
    N = prob.shape[0]
    K = prob.shape[1]

    m = pulp.LpProblem('Problem', pulp.LpMaximize)  # 最大化問題

    # 最適化する変数(= 提出ラベル)
    x = pulp.LpVariable.dicts('x', [(i, j) for i in range(N) for j in range(K)], 0, 1, pulp.LpBinary)

    # log likelihood(目的関数)
    log_likelihood = pulp.lpSum([x[(i, j)] * logp[i, j] for i in range(N) for j in range(K)])
    m += log_likelihood

    # 各データについて，1クラスだけを予測ラベルとする制約
    for i in range(N):
        m += pulp.lpSum([x[(i, k)] for k in range(K)]) == 1  # i.e., SOS1

    # 各クラスについて，推定個数の合計に関する制約
    for k in range(K):
        m += pulp.lpSum([x[(i, k)] for i in range(N)]) == N_CLASSES[k]

    m.solve()  # 解く

    assert m.status == 1  # assert 最適 <=>（実行可能解が見つからないとエラー）

    x_ast = np.array([[int(x[(i, j)].value()) for j in range(K)] for i in range(N)])  # 結果の取得
    
    return x_ast.argmax(axis=1)  # 結果をonehotから -> {0, 1, 2, 3}のラベルに変換

weight = 1 / pd.DataFrame(train_y.values).reset_index().groupby(0).count().values
weight = weight[train_y].ravel()
weight /= weight.sum()

print(train.shape, train_X.shape, train_y.shape, test_X.shape, test.shape)

kfold = StratifiedKFold(n_splits=N_FOLDS)
pred = np.zeros((test_X.shape[0], N_FOLDS))
f1_score = 0

for fold, (train_idx, valid_idx) in enumerate(kfold.split(train_X, train_y)):
    X_train = train_X.loc[train_idx]
    X_valid = train_X.loc[valid_idx]
    y_train = train_y[train_idx]
    y_valid = train_y[valid_idx]

    weight = 1 / pd.DataFrame(y_train.values).reset_index().groupby(0).count().values
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
    pred[:, fold] = hack(y_pred)
    f1_score += estimator.best_score["valid_1"]["macro_f1"] / N_FOLDS

pred = stats.mode(pred, axis=1)[0].flatten().astype(int)