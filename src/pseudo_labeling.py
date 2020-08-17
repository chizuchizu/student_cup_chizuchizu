import pandas as pd

test = pd.read_csv("../data/test.csv")
train = pd.read_csv("../data/train.csv")
test_pred = pd.read_csv("../data/languages/test_default.csv").iloc[:, 1:]

"""
CEが2.5以上のものをを利用
"""

test["jobflag"] = test_pred["default_pred"].astype(int) + 1

test = test[test_pred.iloc[:, 1:].max(axis=1) > 2.5]

train = pd.concat([train, test])

train.to_csv("../data/pseudo_train.csv", index=False)
