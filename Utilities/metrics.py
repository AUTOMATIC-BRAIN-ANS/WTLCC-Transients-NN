import torch
import numpy as np
import scipy.stats as st
import wandb
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score

def ImputationMetrics(outputs, labels, info=None):
    return {"imputation_MAE": torch.mean(outputs["imputation_MAE"]), "reconstruction_MAE": torch.mean(outputs["reconstruction_MAE"])}

def accuracy(outputs, labels, info=None):
    outputs = (outputs > 0.5).float()
    return {"accuracy": accuracy_score(labels, outputs)}

def confidence_interval(outputs, labels, info=None):
    output_dict = {}
    squared_errors = (outputs - labels) ** 2
    squared_errors = squared_errors.reshape(-1).numpy()
    intervals = np.sqrt(st.t.interval(0.95, len(squared_errors) - 1,
                         loc=squared_errors.mean(),
                         scale=st.sem(squared_errors)))
    output_dict["Confidence_Interval"] = {
        "lower": intervals[0],
        "upper": intervals[1],
        "mean": intervals[1] - intervals[0]
    }
    return output_dict
    
def per_class_recall_precission(outputs, labels, info=None):
    output_dict = {}
    _, predicted = torch.max(outputs.data, 1)
    unique_labels = np.unique(labels)
    
    for lab in unique_labels:
        tp = (predicted[labels == lab] == labels[labels == lab]).sum().item()
        fp = (predicted[labels != lab] == lab).sum().item()
        fn = (predicted[labels == lab] != lab).sum().item()
        if tp == 0 and (fp==0 or fn==0):
            if fp == 0:   
                output_dict["Precision_cls_{}".format(lab)] = 0
            if fn == 0:
                output_dict["Recall_cls_{}".format(lab)] = 0
        else:
            output_dict["Precision_cls_{}".format(lab)] = 100 * tp / (tp + fp)
            output_dict["Recall_cls_{}".format(lab)] = 100 * tp / (tp + fn)
        
    return output_dict

def rmse(outputs, labels, info=None):
    return {"rmse": torch.sqrt(torch.mean((outputs - labels) ** 2))}

def mse(outputs, labels, info=None):
    return {"mse": torch.mean((outputs - labels) ** 2)}

def mae(outputs, labels, info=None):
    return {"mae": torch.mean(torch.abs(outputs - labels))}

def mape(outputs, labels, info=None):
    return {"mape": torch.mean(torch.abs(outputs - labels) / torch.abs(labels))}

def data_histograms(outputs, labels, info=None):
    output_dict = {}
    hist_outputs = np.histogram(outputs.data.numpy().reshape(-1), bins=64, density=True)
    hist_labels = np.histogram(labels.data.numpy().reshape(-1), bins=64, density=True)
    output_dict["Network_Output_Histogram"] = wandb.Histogram(np_histogram = hist_outputs)
    output_dict["Label_Histogram"] = wandb.Histogram(np_histogram = hist_labels)
    return output_dict

def F1score(outputs, labels, info=None):
    outputs = (outputs > 0.5).float()
    return {"F1": f1_score(labels, outputs)}

def precision(outputs, labels, info=None):
    outputs = (outputs > 0.5).float()
    return {"Precision": precision_score(labels, outputs)}

def recall(outputs, labels, info=None):
    outputs = (outputs > 0.5).float()
    return {"Recall":  recall_score(labels, outputs)}

def BCELoss(outputs, labels, info=None):
    return {"BCELoss": torch.nn.BCELoss()(outputs, labels)}

def roc_auc(outputs, labels, info=None):
    return {"ROC_AUC": roc_auc_score(labels, outputs)}

#FOR NOW ONLY WORKS ON TEST DATASET (UNSHUFFLED)
def per_patient_scores(outputs, labels, info=None):
    patient_list = info["patient_list"]
    outputs = (outputs > 0.5).float().reshape(-1)
    labels = labels.reshape(-1)

    patients_predictions = {}
    patients_labels = {}
    for patient, prediction, label in zip(patient_list, outputs, labels):
        if patient not in patients_predictions:
            patients_predictions[patient] = []
            patients_labels[patient] = label
        patients_predictions[patient].append(prediction)

    for patient in patients_predictions:
        patients_predictions[patient] = int(np.mean(patients_predictions[patient]) > 0.5)

    #Predict
    predictions = [patients_predictions[patient] for patient in patients_predictions.keys()]
    labels = [patients_labels[patient] for patient in patients_predictions.keys()]
    return {
        "Per_patient_F1": f1_score(labels, predictions),
        "Per_patient_Precision": precision_score(labels, predictions),
        "Per_patient_Recall": recall_score(labels, predictions),
        "Per_patient_ROC_AUC": roc_auc_score(labels, predictions),
        "Per_patient_Accuracy": accuracy_score(labels, predictions)
    }


AVAILABLE_METRICS = {
    "BCELoss": BCELoss,
    "F1": F1score,
    "Precision": precision,
    "Recall": recall,
    "ROC_AUC": roc_auc,
    "Accuracy": accuracy,
    "Per_patient_scores": per_patient_scores,
}