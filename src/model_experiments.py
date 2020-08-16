import pandas as pd
import mlflow
import mlflow.pytorch

from src import bert_model

"""
実験結果を確認する方法

cd src
mlflow ui
"""

# mlflow.set_tracking_uri("")
mlflow.set_experiment("モデルの比較")

DATA_PATH = "../data/"
SEED = 2020
N_FOLDS = 4
# MODEL_TYPE = "bert"
# MODEL_NAME = "bert-base-uncased"
MODEL_TYPE = "albert"
MODEL_NAME = "albert-base-v2"  # xxlarge-v2
params = {
    # "output_dir": "outputs/",
    "max_seq_length": 64,
    "train_batch_size": 64,
    "eval_batch_size": 64,
    "num_train_epochs": 5,
    "learning_rate": 1e-4,
    "reprocess_input_data": True,
    "do_lower_case": True,
    "manual_seed": SEED,
    "verbose": False,
    "save_eval_checkpoints": False,
    "overwrite_output_dir": True,
}

bert_model.seed_everything(SEED)

test = pd.read_csv(f"{DATA_PATH}test.csv").drop(['id'], axis=1)


def preprocess(data, augment=True):
    if augment:
        data = data.drop(columns=['id', "description"])
        data = data.rename(columns={"transrated": 'text', "jobflag": 'label'})
    else:
        data = data.drop(columns=["id"])
        data = data.rename(columns={"description": "text", "jobflag": "label"})
    # data = data.rename(columns={"description": "text", "jobflag": "label"})
    data['label'] -= 1
    # 順序が違うとだめらしいので
    data = data.reindex(columns=["text", "label"])

    return data


train = preprocess(pd.read_csv(f"{DATA_PATH}train.csv"), augment=False)
with mlflow.start_run():
    """log params"""
    mlflow.log_param("model_type", MODEL_TYPE)
    mlflow.log_param("model_name", MODEL_NAME)
    for k, v in params.items():
        mlflow.log_param(k, v)

    test_pred, f1_score, oof_pred = bert_model.model(train, test, params, N_FOLDS, MODEL_NAME, MODEL_TYPE)
    print("model training have done")

    mlflow.log_metric("f1_score", f1_score)
