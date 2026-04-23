"""
Evaluation metrics for ordinal classification tasks.
"""
import numpy as np
from sklearn.metrics import confusion_matrix


def quadratic_weighted_kappa(y_true, y_pred):
    """
    Compute the quadratic weighted kappa coefficient.
    
    The quadratic weighted kappa is a metric used for evaluating ordinal
    classification performance, where disagreements are weighted quadratically
    based on their distance from the diagonal.
    
    Args:
        y_true: True labels (1D array).
        y_pred: Predicted labels (1D array).
    
    Returns:
        qwk: Quadratic weighted kappa score (float between -1 and 1).
    """
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Get the number of classes
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)
    
    # Build confusion matrix
    c = confusion_matrix(y_true, y_pred, labels=classes)
    
    # Normalize by total number of samples
    c = c.astype(float) / c.sum()
    
    # Compute observed disagreement
    disagreements = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            disagreements[i, j] = (i - j) ** 2
    
    observed = np.sum(c * disagreements)
    
    # Compute expected disagreement (under random assignment)
    marginal_true = np.sum(c, axis=1)
    marginal_pred = np.sum(c, axis=0)
    
    expected = 0.0
    for i in range(n_classes):
        for j in range(n_classes):
            expected += marginal_true[i] * marginal_pred[j] * disagreements[i, j]
    
    # Compute QWK
    if expected == 0:
        return 1.0 if observed == 0 else 0.0
    
    qwk = 1.0 - (observed / expected)
    return qwk
