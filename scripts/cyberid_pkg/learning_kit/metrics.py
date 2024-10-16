from torch import Tensor
from sklearn.metrics import confusion_matrix as confusion_matrix_
from torch.nn import BCELoss


loss_function = BCELoss()



def confusion_matrix(outputs: Tensor, targets: Tensor, threshold=0.5):
    predictions = outputs >= threshold
    tp = ((predictions == 1) & (targets == 1)).sum().item()
    tn = ((predictions == 0) & (targets == 0)).sum().item()
    fp = ((predictions == 1) & (targets == 0)).sum().item()
    fn = ((predictions == 0) & (targets == 1)).sum().item()
    return tp, tn, fp, fn



def get_metrics(tp, tn, fp, fn, loss):
        
    metrics_dict = {
        'Accuracy': (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0,
        'Precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'Recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'F-measure': (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
        'FPR': fp / (tn + fp) if (tn + fp) > 0 else 0,
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'BCELoss': loss,
    }
    return metrics_dict