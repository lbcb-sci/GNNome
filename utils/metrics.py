import torch

from sklearn.metrics import precision_recall_curve, average_precision_score


def calculate_tfpn(edge_predictions, edge_labels):
    edge_predictions = torch.round(torch.sigmoid(edge_predictions))
    TP = torch.sum(torch.logical_and(edge_predictions==1, edge_labels==1)).item()
    TN = torch.sum(torch.logical_and(edge_predictions==0, edge_labels==0)).item()
    FP = torch.sum(torch.logical_and(edge_predictions==1, edge_labels==0)).item()
    FN = torch.sum(torch.logical_and(edge_predictions==0, edge_labels==1)).item()
    return TP, TN, FP, FN


def calculate_metrics(TP, TN, FP, FN):
    try: 
        precision = TP / (TP + FP)
    except ZeroDivisionError:
        precision = 0
    try:
        recall = TP / (TP + FN)
    except ZeroDivisionError:
        recall = 0
    try:
        f1 = TP / (TP + 0.5 * (FP + FN) )
    except ZeroDivisionError:
        f1 = 0
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return accuracy, precision, recall, f1


def calculate_metrics_inverse(TP, TN, FP, FN): 
    TP, TN = TN, TP
    FP, FN = FN, FP
    try: 
        precision = TP / (TP + FP)
    except ZeroDivisionError:
        precision = 0
    try:
        recall = TP / (TP + FN)
    except ZeroDivisionError:
        recall = 0
    try:
        f1 = TP / (TP + 0.5 * (FP + FN) )
    except ZeroDivisionError:
        f1 = 0
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return accuracy, precision, recall, f1


def get_precision_recall_curve(preds, labels):
    preds = torch.sigmoid(preds).cpu().detach().numpy()
    labels = labels.cpu().numpy()
    precision, recall, thresholds = precision_recall_curve(labels, preds)
    return precision, recall, thresholds


def get_precision_recall_curve_inverse(preds, labels):
    preds = torch.sigmoid(preds).cpu().detach().numpy()
    preds = 1 - preds
    labels = labels.cpu().numpy()
    precision, recall, thresholds = precision_recall_curve(labels, preds, pos_label=0)
    return precision, recall, thresholds


# Actually computes average_precision_score instead of AUC-PC
def get_aps(preds, labels):
    preds = torch.sigmoid(preds).cpu().detach().numpy()
    labels = labels.cpu().numpy()
    auc_pc = average_precision_score(labels, preds)
    return auc_pc


# Actually computes average_precision_score instead of AUC-PC
def get_aps_inverse(preds, labels):
    preds = torch.sigmoid(preds).cpu().detach().numpy()
    preds = 1 - preds
    labels = labels.cpu().numpy()
    auc_pc = average_precision_score(labels, preds, pos_label=0)
    return auc_pc
