
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def evaluate_model(y_true, y_pred):

    results = {

        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),

        "precision": float(
            precision_score(
                y_true, y_pred,
                pos_label=1,
                zero_division=0
            )
        ),

        "recall": float(
            recall_score(
                y_true, y_pred,
                pos_label=1,
                zero_division=0
            )
        ),

        "f1_score": float(
            f1_score(
                y_true, y_pred,
                pos_label=1,
                zero_division=0
            )
        ),

        "confusion_matrix": confusion_matrix(
            y_true, y_pred,
            labels=[0, 1]
        ).tolist(),

        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["genuine", "fake"],
            output_dict=True,
            zero_division=0
        )
    }

    return results
