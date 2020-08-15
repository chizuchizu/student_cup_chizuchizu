import pandas as pd
import numpy as np
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
# 3asef
NUM_CLASS = 4
TARGET = "label"

train = pd.read_pickle("../data/train_tfidf.pkl")
test = pd.read_pickle("../data/test_tfidf.pkl")
features = train.columns

train[TARGET] = pd.read_csv("../data/train.csv")["jobflag"] - 1


# test[TARGET] = pd.read_csv("../data/test.csv")["jobflag"] - 1

# for class_name in range(NUM_CLASS):
#     print(class_name)
#     train_target = train[TARGET]
#
#     model = LogisticRegression(solver="sag", n_jobs=-1)
#     sfm = SelectFromModel(model, threshold=0.2)
#     train_sparse_matrix = sfm.fit_transform(train[features], train[TARGET])
#
#     train_sparse_matrix, valid_sparse_matrix, y_train, y_valid = train_test_split(train_sparse_matrix, train_target,
#                                                                                   test_size=0.1, random_state=123)
#     test_sparse_matrix = sfm.transform(test[features])
#
#     d_train = lgb.Dataset(train_sparse_matrix, label=y_train)
#     d_valid = lgb.Dataset(valid_sparse_matrix, label=y_valid)
#

def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = y_hat.reshape(len(y_hat) // NUM_CLASS, NUM_CLASS).argmax(axis=1)
    y_hat = np.round(y_hat)  # scikits f1 doesn't like probabilities
    return 'f1', f1_score(y_true, y_hat, average="macro"), True


params = {
    "learning_rate": 0.1,
    "objective": "multiclass",
    "num_class": NUM_CLASS,
    "num_leaves": 70,
    "max_depth": 7,
    "subsample": 0.6,
    "seed": 123,
    # "metric": "f1",
    "verbosity": -1
}

train_x, valid_x, train_y, valid_y = train_test_split(train[features], train[TARGET], test_size=0.1, random_state=123)
d_train = lgb.Dataset(train_x, label=train_y)
d_valid = lgb.Dataset(valid_x, label=valid_y)

estimator = lgb.train(
    params,
    d_train,
    num_boost_round=2000,
    valid_sets=[d_train, d_valid],
    early_stopping_rounds=100,
    feval=lgb_f1_score,
    # verbose_eval=200
)

print(0)

pred = estimator.predict(test).argmax(axis=1)

test_id = pd.read_csv("../data/test.csv")["id"]

submit = pd.DataFrame({'index': test_id, 'pred': pred + 1})
submit.to_csv(f"../outputs/submit_model2_bert_{123}.csv", index=False, header=False)
