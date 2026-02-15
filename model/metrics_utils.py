from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

def compute_classification_metrics(y_true, y_pred, y_proba=None):
    """
    Compute required classification metrics.
    Works for binary & multiclass classification.
    """

    is_multiclass = len(set(y_true)) > 2

    accuracy = accuracy_score(y_true, y_pred)

    if is_multiclass:
        precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        auc = None
        if y_proba is not None:
            auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")

    else:
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        auc = None
        if y_proba is not None:
            auc = roc_auc_score(y_true, y_proba[:, 1])

    mcc = matthews_corrcoef(y_true, y_pred)

    return {
        "Accuracy": accuracy,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "MCC": mcc
    }


def get_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)


def get_classification_report(y_true, y_pred):
    return classification_report(y_true, y_pred)
