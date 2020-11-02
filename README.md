# MLflow Tutorial

[Quickstart — MLflow 1.11.0 documentation](https://www.mlflow.org/docs/latest/quickstart.html)

```
pip install mlflow
```

## 学習

[サンプルコード](https://github.com/mlflow/mlflow/blob/master/examples/sklearn_logistic_regression/train.py)を実行します。

```py
import numpy as np
from sklearn.linear_model import LogisticRegression

import mlflow
import mlflow.sklearn

if __name__ == "__main__":
    X = np.array([-2, -1, 0, 1, 2, 1]).reshape(-1, 1)
    y = np.array([0, 0, 1, 1, 1, 0])
    lr = LogisticRegression()
    lr.fit(X, y)
    score = lr.score(X, y)
    print("Score: %s" % score)
    mlflow.log_metric("score", score)
    mlflow.sklearn.log_model(lr, "model")
    print("Model saved in run %s" % mlflow.active_run().info.run_uuid)
```

```
python3 examples/sklearn_logistic_regression/train.py
# 実行後、mlruns というディレクトリが作成される
```

## 学習結果の確認

```
mlflow ui
```

`http://127.0.0.1:5000` にアクセスすると学習結果の画面が表示される。

- モデルの作成時刻、ハイパーパラメータ、評価などを確認できる
- モデルごとに比較が可能

## MLflow プロジェクトの実行

```
mlflow run examples/sklearn_logistic_regression
```
