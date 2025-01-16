from sklearn.metrics import roc_curve, confusion_matrix
from sklearn import metrics
import numpy as np
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,roc_auc_score,recall_score,precision_score
from sklearn.metrics import roc_curve
import pandas as pd

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
import matplotlib.pyplot as plt


def find_Optimal_Cutoff(target, predicted): 
    fpr, tpr, thresholds = roc_curve(target, predicted)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    # print('the optimal threshold:', optimal_threshold)
    return optimal_threshold
# data['pred'] = data['pred_proba'].map(lambda x: 1 if x > threshold else 0)
# confusion_matrix(data['admit'], data['pred'])

def calculate_metric(gt, pred, threshold=None): 
    fpr, tpr, thres = roc_curve(gt, pred)
    if threshold is None: 
        optimal_idx = np.argmax(tpr - fpr)
        threshold = thres[optimal_idx]

    pred[pred>threshold]=1
    pred[pred<1]=0 
    auc = metrics.auc(fpr, tpr) 

    tn, fp, fn, tp = confusion_matrix(gt,pred).ravel()
    Sen = tp / float(tp+fn)
    Spe = tn / float(tn+fp)  

    # confusion = confusion_matrix(gt,pred)
    # TP = confusion[1, 1]
    # TN = confusion[0, 0]
    # FP = confusion[0, 1]
    # FN = confusion[1, 0]
    # Sen = TP / float(TP+FN)
    # Spe = TN / float(TN+FP)  
    return auc, Sen, Spe, threshold

def dice_score(pred, target, epsilon=1e-6):
    """
    Compute the Dice score.
    pred: tensor with predictions (0 or 1).
    target: tensor with ground truth labels (0 or 1).
    epsilon: small constant to avoid division by zero.
    """
    # Convert probabilities to binary predictions (if necessary)
    pred = (pred >= 0.5).float()
    
    # Flatten label and prediction tensors
    pred = pred.view(-1)
    target = target.view(-1)
    
    # Intersection and union
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    # Dice score
    dice = (2. * intersection + epsilon) / (union + epsilon)
    
    return dice