import numpy as np
from sklearn.metrics import roc_auc_score


def labelwise_auc(y_true, y_score):
    scores = []
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    for idx in range(y_true.shape[1]):
        values = y_true[:, idx]
        if values.min() == values.max():
            continue
        scores.append(roc_auc_score(values, y_score[:, idx]))
    if not scores:
        return float("nan")
    return float(np.mean(scores))

