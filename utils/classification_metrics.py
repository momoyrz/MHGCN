import warnings
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix


warnings.filterwarnings("ignore")

# 计算分类模型的各种指标，包括：acc, pre, sen, f1, spec, kappa，auc,Quadratic Weighted Kappa
def _all_metrics(y_true, y_pred, y_score):
    """
    :param y_true: 真实值
    :param y_pred: 预测值
    :param y_score: 预测属于某类的概率
    :return: 返回acc, pre, sen, f1, spec, kappa，auc,Quadratic Weighted Kappa
    """
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, average="macro")
    sen = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")
    spec = _calculate_average_specificity_sklearn(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    my_auc = roc_auc_score(y_true, y_score, multi_class='ovr')
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    return acc, pre, sen, f1, spec, kappa, my_auc, qwk


def all_metrics(y_true, y_score):
    y_pred = np.argmax(y_score, axis=1)

    return _all_metrics(y_true, y_pred, y_score)  # 返回acc, pre, sen, f1, spec, kappa，auc,Quadratic Weighted Kappa


def _calculate_average_specificity_sklearn(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    specificity_list = []
    for i in range(len(cm)):
        specificity = _calculate_specificity_sklearn(cm, i)
        specificity_list.append(specificity)
    return sum(specificity_list) / len(specificity_list)


def _calculate_specificity_sklearn(cm, class_index):
    tp = cm[class_index, class_index]
    fn = np.sum(cm[class_index, :]) - tp
    fp = np.sum(cm[:, class_index]) - tp
    tn = np.sum(cm) - tp - fn - fp
    specificity = tn / (tn + fp)
    return specificity
