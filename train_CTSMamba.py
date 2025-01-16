

import torch 
# import sys
# sys.path.append(r"./mamba")
# sys.path.append(r"./mamba/mamba_ssm")
from model_segmamba.model_build import CTSMamba, CTSMamba_v2, TSMamba
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from loss import cox_loss, DiceLoss, concordance_index, c_index 
from metric import calculate_metric, find_Optimal_Cutoff, dice_score
import os 
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
import lifelines 
import skimage
from make_dataloader import imBalanced_MES, prepare_data_v3, make_dataloader_v3
import argparse
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.utils import resample
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, confusion_matrix
import pandas as pd
from sklearn.metrics import precision_recall_curve
import gc

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
seed=5
np.random.seed(seed)
torch.manual_seed(seed)

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor 

criterion = nn.BCELoss(reduction='mean') 
criterion_dice = DiceLoss()
criterion_surv = cox_loss() 

def lr_scheduler(epoch):
    if epoch < 20:
        lr = 5e-5
    elif epoch < 40:
        lr = 1e-5
    else:
        lr = 1e-6
    return lr

# prepare losses
from loss import Loglike_loss, L2_Regu_loss 
Losses = [Loglike_loss, L2_Regu_loss]
Weights = [1.0, 1.0]

def Get_survival_time(Survival_pred):

    # breaks = np.array([0,300,600,900,1100,1300,1500,2100,2700,3500,6000])
    breaks = np.array([0,10,20,30,40,50,60,70,80,120,160])

    intervals = breaks[1:] - breaks[:-1]
    n_intervals = len(intervals)
    
    Survival_time = 0
    for i in range(n_intervals):
        cumulative_prob = np.prod(Survival_pred[0:i+1])
        Survival_time = Survival_time + cumulative_prob * intervals[i]
    
    return Survival_time

def hr_values(pred_surv_list, labels_time_list, labels_status_list, best_cutoff_value=None ): 
    if best_cutoff_value is None: 
        ## method 1. median_cutoff_value
        best_cutoff_value = np.median( pred_surv_list ) 

    duration_col = 'OS' 
    event_col = 'Died' 
    data_ = pd.DataFrame()
    data_['pred'] = pred_surv_list
    data_[duration_col] = labels_time_list
    data_[event_col] = labels_status_list
    data_['risk_group'] = (data_['pred'] > best_cutoff_value).astype(int)
    group1 = data_[data_['pred'] <= best_cutoff_value]
    group2 = data_[data_['pred'] > best_cutoff_value] 
    result = logrank_test(group1[duration_col], group2[duration_col], event_observed_A=group1[event_col], event_observed_B=group2[event_col]) 
    cph = CoxPHFitter()
    cph.fit(data_[[duration_col, event_col, 'risk_group']], duration_col=duration_col, event_col=event_col)
    # Printing the Cox model results
    hr = cph.summary.loc['risk_group', 'exp(coef)']
    lower_ci = cph.summary.loc['risk_group', 'exp(coef) lower 95%']
    upper_ci = cph.summary.loc['risk_group', 'exp(coef) upper 95%'] 
    if result.p_value<0.05: 
        print(">>>>>>>>>>>>cutoff_value", best_cutoff_value, result.p_value) 
        print(f"Hazard Ratio (HR): {hr:.4f}, 95% CI: ({lower_ci:.4f}, {upper_ci:.4f})")
    return best_cutoff_value 

def test_model(model, test_loader, best_cutoff_value=None ):
    model.eval() 
    labels_N_list = [] 
    labels_time_list = [] 
    labels_status_list = [] 
    pred_N_list = [] 
    pred_surv_list = [] 
    Survival_time = [] 
    with torch.no_grad(): 
        for x, labels in test_loader: 
            labels_N = labels[0].unsqueeze_(1).type(Tensor) 
            labels_status = labels[1].unsqueeze_(1).type(Tensor) 
            labels_time = labels[2].unsqueeze_(1).type(Tensor) 
            # labels_surv = torch.transpose(torch.stack(labels[1], dim=0), 0,1 ).type(Tensor) 

            pred_N, pred_surv = model( (x[0]).type(Tensor), (x[1]).type(Tensor) ) 
            # # calculate validation metrics
            # Survival_pred = pred_surv[0].detach().cpu().numpy().squeeze()
            # Survival_time.append(Get_survival_time(Survival_pred))

            for v in pred_N.cpu().detach().squeeze([1]).numpy(): pred_N_list.append(v) 
            for v in labels_N.cpu().detach().squeeze([1]).numpy(): labels_N_list.append(v) 
            for v in pred_surv.cpu().detach().squeeze([1]).numpy(): pred_surv_list.append(v)             
            for v in labels_time.cpu().detach().squeeze([1]).numpy(): labels_time_list.append(v) 
            for v in labels_status.cpu().detach().squeeze([1]).numpy(): labels_status_list.append(v) 

        pred_N_list = np.array(pred_N_list)
        labels_N_list = np.array(labels_N_list)
        pred_surv_list = np.array(pred_surv_list)
        labels_time_list = np.array(labels_time_list) 
        labels_status_list = np.array(labels_status_list)
        auc, Sen, Spe  = calculate_metric(labels_N_list, pred_N_list ) 
        # print('-'*80)
        print(f'auc = {auc:.4f}, Sen = {Sen:.4f}, Spe = {Spe:.4f} ') 

        mask = (labels_status_list==1) | (labels_status_list==0) 
        labels_time_list = labels_time_list[mask] 
        labels_status_list = labels_status_list[mask] 
        pred_surv_list = pred_surv_list[mask] 
        pred_N_list = pred_N_list[mask]
        surv_ci = concordance_index(torch.from_numpy(np.array([ labels_time_list, labels_status_list]).transpose()), 
                                    torch.from_numpy(np.array(pred_surv_list)) ) 
        print(f'surv_ci = {surv_ci:.4f}') 
        c_index2 = c_index(torch.from_numpy(np.array(pred_surv_list)), torch.from_numpy(np.array(labels_time_list)), torch.from_numpy(np.array(labels_status_list)))
        print(f'concordance_index = {c_index2:.4f}') 
        best_cutoff_value = hr_values(pred_surv_list, labels_time_list, labels_status_list, best_cutoff_value)
        # surv_ci = concordance_index(torch.from_numpy(np.array([ labels_time_list, labels_status_list]).transpose()), 
        #                             torch.from_numpy(np.array(pred_N_list)) )
        # print(f'surv_ci = {surv_ci:.4f}') 
        return best_cutoff_value

import SimpleITK as sitk
def read_itk_files(img_path, label_path): 
    image_sitk = sitk.ReadImage( img_path ) 
    x = sitk.GetArrayFromImage(image_sitk) 
    originalimg_spacing = image_sitk.GetSpacing()

    label_sitk = sitk.ReadImage( label_path ) 
    y = sitk.GetArrayFromImage(label_sitk) 
    return x, y

def test_model_v2(model, test_loader, df_features, best_cutoff_value=None ):
    labels_N_list = [] 
    labels_time_list = [] 
    labels_status_list = [] 
    pred_N_list = [] 
    Survival_time = [] 
    labels_N_list = df_features['target_'].to_numpy()
    labels_status_list, labels_time_list = df_features['是否死亡'].to_numpy(), df_features['OS时间（月份）'].to_numpy()
    model.eval() 
    with torch.no_grad(): 
        for x, y, labels in test_loader: 
            labels_N = labels[0].unsqueeze_(1).type(Tensor) 
            labels_surv = torch.transpose(torch.stack(labels[1], dim=0), 0,1 ).type(Tensor) 

            pred_N, pred_surv = model( (x[0]).type(Tensor), (x[1]).type(Tensor) ) 
            # # calculate validation metrics
            Survival_pred = pred_surv[0].detach().cpu().numpy().squeeze()
            Survival_time.append(Get_survival_time(Survival_pred))
            for v in pred_N.cpu().detach().squeeze([1]).numpy(): pred_N_list.append(v) 

    Survival_time = np.array(Survival_time)
    pred_N_list = np.array(pred_N_list)
    labels_N_list = np.array(labels_N_list)
    labels_time_list = np.array(labels_time_list) 
    labels_status_list = np.array(labels_status_list)
    auc, Sen, Spe  = calculate_metric(labels_N_list, pred_N_list ) 
    # print('-'*80)
    print(f'auc = {auc:.4f}, Sen = {Sen:.4f}, Spe = {Spe:.4f} ') 

    valid_cindex = lifelines.utils.concordance_index(labels_time_list, Survival_time, labels_status_list)
    valid_cindex_info = 'Valid C-index: %.4f' % valid_cindex
    print(valid_cindex_info) 

    best_cutoff_value = hr_values(Survival_time, labels_time_list, labels_status_list, best_cutoff_value)

    return best_cutoff_value

# Function to calculate Negative Predictive Value (NPV)
def cal_NPV(y_true, y_pred): 
    ### calculate negative predictive value (NPV) 
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    npv = tn / (tn + fn) 
    return npv

# Function to calculate various metrics and find the optimal threshold
def calculate_metrics(y_true, y_pred_proba, optimal_threshold=None):
    auc = metrics.roc_auc_score(y_true, y_pred_proba) 
    y_pred = y_pred_proba.copy()
    fpr, tpr, thres = roc_curve(y_true, y_pred_proba) 
    if optimal_threshold: 
        y_pred[y_pred>optimal_threshold]=1 
        y_pred[y_pred<1]=0 
    else: 
        ##1 Youden’s Index
        optimal_idx = np.argmax(tpr - fpr) 
        optimal_threshold = thres[optimal_idx] 
        print('find the optimal threshold: ', optimal_threshold) 
        # ##2 Equal Error Rate 找到使假阳性率和假阴性率相等的阈值
        # fnr = 1 - tpr  # 假阴性率 
        # optimal_threshold = thres[np.nanargmin(np.absolute((fnr - fpr)))] 
        # print("Equal Error Rate threshold:", optimal_threshold) 
        ##3 最大化F1分数  使用精确度-召回率曲线来找到最大F1分数的阈值
        # precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        # fscore = (2 * precision * recall) / (precision + recall)
        # fscore = np.nan_to_num(fscore)  # 处理NaN值
        # optimal_idx = np.argmax(fscore)
        # optimal_threshold = thresholds[optimal_idx]
        # print("Optimal threshold by F1 Score:", optimal_threshold) 

        y_pred[y_pred>optimal_threshold]=1 
        y_pred[y_pred<1]=0 

    accuracy = metrics.accuracy_score(y_true, y_pred)
    sensitivity = metrics.recall_score(y_true, y_pred)
    specificity = metrics.recall_score(y_true, y_pred, pos_label=0) 
    f1_score = metrics.f1_score(y_true, y_pred)  
    precision = metrics.precision_score(y_true, y_pred)  ## precision, positive predictive value (PPV) 
    npv = cal_NPV(y_true, y_pred) 
    results = { "AUC": auc, "Accuracy": accuracy, "Sensitivity": sensitivity, "Specificity": specificity, 
               "F1 Score": f1_score, "precision": precision, "npv": npv, }
    # print(results) 
    return results, optimal_threshold 

# Function to calculate metrics, save confusion matrix, and find the optimal threshold
def calculate_metrics_save_ConfusionMatrix(y_true, y_pred_proba, n_iterations=1000, alpha=0.05, save_fig=None, optimal_threshold=None):
    auc = metrics.roc_auc_score(y_true, y_pred_proba) 
    y_pred = y_pred_proba.copy()
    fpr, tpr, thres = roc_curve(y_true, y_pred_proba) 
    if optimal_threshold: 
        y_pred[y_pred>optimal_threshold]=1
        y_pred[y_pred<1]=0 
    else: 
        
        ##1 Youden’s Index
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thres[optimal_idx]
        print('find the optimal threshold: ', optimal_threshold)
        # auc = metrics.auc(fpr, tpr) 
        y_pred[y_pred>optimal_threshold]=1
        y_pred[y_pred<1]=0 
        # y_pred = np.round(y_pred_proba)  # Convert probabilities to binary predictions
        # ##2 Equal Error Rate 找到使假阳性率和假阴性率相等的阈值
        # fnr = 1 - tpr  # 假阴性率
        # eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
        # print("Equal Error Rate threshold:", eer_threshold)
        # ##3 最大化F1分数  使用精确度-召回率曲线来找到最大F1分数的阈值
        # precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        # fscore = (2 * precision * recall) / (precision + recall)
        # fscore = np.nan_to_num(fscore)  # 处理NaN值
        # optimal_idx = np.argmax(fscore)
        # optimal_threshold = thresholds[optimal_idx]
        # # print("Optimal threshold by F1 Score:", optimal_threshold) 
        
    if save_fig: 
        disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=['DS-TB', 'DR-TB'], cmap=plt.cm.Blues) ##plt.cm.Blues RdBu_r
        disp.im_.colorbar.remove()
        plt.xticks(rotation=0, ha='center')
        plt.yticks(rotation=90, va='center') 
        plt.savefig(save_fig, bbox_inches='tight')
        # plt.show() 
    
    accuracy = metrics.accuracy_score(y_true, y_pred)
    sensitivity = metrics.recall_score(y_true, y_pred)
    specificity = metrics.recall_score(y_true, y_pred, pos_label=0)
    f1_score = metrics.f1_score(y_true, y_pred) 
    precision = metrics.precision_score(y_true, y_pred)  ## precision, positive predictive value (PPV) 
    npv = cal_NPV(y_true, y_pred)
    # Calculate 95% confidence intervals using bootstrapping
    auc_scores = []
    accuracy_scores = []
    sensitivity_scores = []
    specificity_scores = []
    f1_scores = []
    precision_scores = [] 
    npv_scores = []
    for _ in range(n_iterations):
        # Bootstrap resampling
        y_true_resampled, y_pred_proba_resampled, y_pred_resampled = resample(y_true, y_pred_proba, y_pred)
        # y_pred_proba_resampled = resample(y_pred_proba)
        # y_pred_resampled = resample(y_pred)
        # Calculate performance metrics
        auc_scores.append(metrics.roc_auc_score(y_true_resampled, y_pred_proba_resampled))
        accuracy_scores.append(metrics.accuracy_score(y_true_resampled, y_pred_resampled))
        sensitivity_scores.append(metrics.recall_score(y_true_resampled, y_pred_resampled))
        specificity_scores.append(metrics.recall_score(y_true_resampled, y_pred_resampled, pos_label=0))
        f1_scores.append(metrics.f1_score(y_true_resampled, y_pred_resampled))
        precision_scores.append(metrics.precision_score(y_true_resampled, y_pred_resampled) )
        npv_scores.append(cal_NPV(y_true_resampled, y_pred_resampled) )
    # Calculate 95% confidence intervals
    auc_ci = np.percentile(auc_scores, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    accuracy_ci = np.percentile(accuracy_scores, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    sensitivity_ci = np.percentile(sensitivity_scores, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    specificity_ci = np.percentile(specificity_scores, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    f1_ci = np.percentile(f1_scores, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    precision_ci = np.percentile(precision_scores, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    npv_ci = np.percentile(npv_scores, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # Return the results as a dictionary
    results = { 
                "AUC": f"{auc:.3f}({auc_ci[0]:.3f}-{auc_ci[1]:.3f})", 
                "Accuracy": f"{accuracy:.3f}({accuracy_ci[0]:.3f}-{accuracy_ci[1]:.3f})", 
                "Sensitivity": f"{sensitivity:.3f}({sensitivity_ci[0]:.3f}-{sensitivity_ci[1]:.3f})",
                "Specificity":  f"{specificity:.3f}({specificity_ci[0]:.3f}-{specificity_ci[1]:.3f})", 
                "F1 Score": f"{f1_score:.3f}({f1_ci[0]:.3f}-{f1_ci[1]:.3f})", 
                "precision": f"{precision:.3f}({precision_ci[0]:.3f}-{precision_ci[1]:.3f})", 
                "npv": f"{npv:.3f}({npv_ci[0]:.3f}-{npv_ci[1]:.3f})", 
                 "optimal_threshold": optimal_threshold  
                } 
    return results, optimal_threshold

def bootstrap_cindex(labels_time_list, Survival_time, labels_status_list, n_bootstraps=1000, random_seed=42): 
    bootstrapped_scores = []
    for i in range(n_bootstraps):
        indices = np.random.randint(0, len(labels_time_list), len(labels_time_list))
        if len(np.unique(labels_status_list[indices])) < 2:
            continue
        score = lifelines.utils.concordance_index(labels_time_list[indices], Survival_time[indices], labels_status_list[indices])
        bootstrapped_scores.append(score)
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    ci_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    ci_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    return ci_lower, ci_upper

def calculate_surv(Survival_time, labels_time_list, labels_status_list, best_cutoff_value=None): 
    def _hr_values(pred_surv_list, labels_time_list, labels_status_list, best_cutoff_value=None ): 
        if best_cutoff_value is None: 
            ## method 1. median_cutoff_value
            best_cutoff_value = np.median( pred_surv_list ) 
            # ## method 2. Maximally Selected Rank Statistics
            # best_cutoff_result, best_cutoff_value = None, None
            # for cutoff_value in np.unique( pred_surv_list ): 
            #     group1_mask = pred_surv_list <= cutoff_value 
            #     group2_mask = pred_surv_list > cutoff_value
            #     result = logrank_test(labels_time_list[group1_mask], labels_time_list[group2_mask], 
            #                           event_observed_A=labels_status_list[group1_mask], 
            #                           event_observed_B=labels_status_list[group2_mask])
            #     if best_cutoff_result is None or result.test_statistic > best_cutoff_result.test_statistic:
            #         best_cutoff_result = result 
            #         best_cutoff_value = cutoff_value 
        duration_col = 'OS' 
        event_col = 'Died' 
        data_ = pd.DataFrame()
        data_['pred'] = pred_surv_list
        data_[duration_col] = labels_time_list
        data_[event_col] = labels_status_list
        data_['risk_group'] = (data_['pred'] > best_cutoff_value).astype(int)
        group1 = data_[data_['pred'] <= best_cutoff_value]
        group2 = data_[data_['pred'] > best_cutoff_value] 
        result = logrank_test(group1[duration_col], group2[duration_col], event_observed_A=group1[event_col], event_observed_B=group2[event_col]) 
        cph = CoxPHFitter()
        cph.fit(data_[[duration_col, event_col, 'risk_group']], duration_col=duration_col, event_col=event_col)
        # Printing the Cox model results
        hr = cph.summary.loc['risk_group', 'exp(coef)']
        lower_ci = cph.summary.loc['risk_group', 'exp(coef) lower 95%']
        upper_ci = cph.summary.loc['risk_group', 'exp(coef) upper 95%'] 
        # if result.p_value<0.05: 
        #     print(">>>>>>>>>>>>cutoff_value", best_cutoff_value, result.p_value) 
        #     print(f"Hazard Ratio (HR): {hr:.4f}, 95% CI: ({lower_ci:.4f}, {upper_ci:.4f})")
        return best_cutoff_value, result.p_value, hr, lower_ci, upper_ci
    valid_cindex = lifelines.utils.concordance_index(labels_time_list, Survival_time, labels_status_list)
    cindex_ci_lower, cindex_ci_upper = bootstrap_cindex(labels_time_list, Survival_time, labels_status_list)
    valid_cindex_info = f"Valid C-index: {valid_cindex:.4f} ({cindex_ci_lower:.4f}-{cindex_ci_upper:.4f})"
    best_cutoff_value, p_value, hr, lower_ci, upper_ci = _hr_values(Survival_time, labels_time_list, labels_status_list, best_cutoff_value) 
    # results = {'cindex':f'{valid_cindex:.3f}({cindex_ci_lower:.3f}-{cindex_ci_upper:.3f})', 'pvalue': p_value, 'hr': hr, 'lower_ci': lower_ci, 'upper_ci': upper_ci, 'best_cutoff_value': best_cutoff_value} 
    results = {'cindex':f'{valid_cindex:.3f}({cindex_ci_lower:.3f}-{cindex_ci_upper:.3f})', 'pvalue': p_value, 'hr': f'{hr:.3f}({lower_ci:.3f}-{upper_ci:.3f})', 'best_cutoff_value': best_cutoff_value} 
                                                                                            
    return results, best_cutoff_value 


def evaluation(df_preds_train, df_preds_test, df_preds_evc1, df_preds_evc2, save_fname='metrics_table.csv'):
    metrics_DL_1, optimal_threshold = calculate_metrics_save_ConfusionMatrix(df_preds_train['labels_N'], df_preds_train['pred_N'], optimal_threshold=None ) 
    metrics_DL_2, optimal_threshold = calculate_metrics_save_ConfusionMatrix(df_preds_test['labels_N'], df_preds_test['pred_N'], optimal_threshold=optimal_threshold )  
    metrics_DL_3, optimal_threshold = calculate_metrics_save_ConfusionMatrix(df_preds_evc1['labels_N'], df_preds_evc1['pred_N'], optimal_threshold=optimal_threshold ) 
    metrics_DL_4, optimal_threshold = calculate_metrics_save_ConfusionMatrix(df_preds_evc2['labels_N'],  df_preds_evc2['pred_N'], optimal_threshold=optimal_threshold ) 
    
    surv_1, best_cutoff_value = calculate_surv(df_preds_train['pred_surv_risk'].to_numpy(), df_preds_train['labels_time'].to_numpy(), df_preds_train['labels_status'].to_numpy(), best_cutoff_value=None)
    surv_2, best_cutoff_value = calculate_surv(df_preds_test['pred_surv_risk'].to_numpy(), df_preds_test['labels_time'].to_numpy(), df_preds_test['labels_status'].to_numpy(), best_cutoff_value=best_cutoff_value)
    surv_4, best_cutoff_value = calculate_surv(df_preds_evc2['pred_surv_risk'].to_numpy(), df_preds_evc2['labels_time'].to_numpy(), df_preds_evc2['labels_status'].to_numpy(), best_cutoff_value=best_cutoff_value)

    data_metrics = [metrics_DL_1, metrics_DL_2, metrics_DL_3, metrics_DL_4, surv_1, surv_2, surv_4]   
    df = pd.DataFrame(data_metrics, index=[['DL',]*7, 
                                           ['train', 'interValid', 'evc1', 'evc2', 
                                            'train_surv', 'interValid_surv', 'evc2_surv']] ) 
    df.to_csv(save_fname, encoding='utf-8-sig') 

def evaluation_v2(df_preds_train, save_fname='metrics_table.csv' ): 
    metrics_DL_1, optimal_threshold = calculate_metrics_save_ConfusionMatrix(df_preds_train['labels_N'], df_preds_train['pred_N'], optimal_threshold=None ) 
    surv_1, best_cutoff_value = calculate_surv(df_preds_train['pred_surv_risk'].to_numpy(), df_preds_train['labels_time'].to_numpy(), df_preds_train['labels_status'].to_numpy(), best_cutoff_value=None)
    return metrics_DL_1, optimal_threshold, surv_1, best_cutoff_value

def test_model_v3(model, X_train_path, df_features, best_cutoff_value=None, ifcal_surv=True, save_path=None, auc_thres=None ):
    labels_N_list = [] 
    labels_time_list = [] 
    labels_status_list = [] 
    pred_N_list = [] 
    Survival_time = [] 
    windowCenterWidth=(40, 400)
    imgMinMax = [ windowCenterWidth[0] - windowCenterWidth[1]/2.0, windowCenterWidth[0] + windowCenterWidth[1]/2.0 ]
    labels_N_list = df_features['target_'].to_numpy()
    labels_status_list, labels_time_list = df_features['是否死亡'].to_numpy(), df_features['OS时间（月份）'].to_numpy() 
    model.eval() 
    with torch.no_grad(): 
        for i in range(len(X_train_path[0])): 
            x_1_patch_list=[] 
            y_1_patch_list=[] 
            for j in range(2): 
                paths = X_train_path[0][i] 
                img_path, clu_path = paths[j][0], paths[j][1] 
                x_1_patch_, y_1_patch_ = read_itk_files(img_path, clu_path) 
                # print(x_1_patch_.shape, y_1_patch_.shape)
                x_1_patch_ = np.clip(x_1_patch_, a_min=imgMinMax[0], a_max=imgMinMax[1] ) 
                x_1_patch_ = (x_1_patch_ - imgMinMax[0] ) / (imgMinMax[1] - imgMinMax[0]) 
                x_1_patch = x_1_patch_ 
                y_1_patch = (y_1_patch_ >0.5)*1 
                x_1_patch_ = skimage.transform.resize(x_1_patch, [96, 96, 96], order=1, preserve_range=True, anti_aliasing=False)
                y_1_patch_ = skimage.transform.resize(y_1_patch, [96, 96, 96], order=0, preserve_range=True, anti_aliasing=False)
                x_1_patch_ =  np.expand_dims(x_1_patch_, axis=0 ) 
                y_1_patch_ =  np.expand_dims(y_1_patch_, axis=0 ) 
                x_1_patch_list.append(x_1_patch_) 
                y_1_patch_list.append(y_1_patch_) 
            x0, x1 = x_1_patch_list[0]*y_1_patch_list[0], x_1_patch_list[1]*y_1_patch_list[1]
            x0, x1 = torch.from_numpy(x0 ).type(Tensor), torch.from_numpy(x1 ).type(Tensor) 
            x0, x1 = torch.unsqueeze(x0, 0), torch.unsqueeze(x1, 0)
            pred_N, pred_surv = model( x0.type(Tensor), x1.type(Tensor) ) 

            # # calculate validation metrics
            Survival_pred = pred_surv[0].detach().cpu().numpy().squeeze()
            Survival_time.append(Get_survival_time(Survival_pred))
            for v in pred_N.cpu().detach().squeeze([1]).numpy(): pred_N_list.append(v) 

    Survival_time = np.array(Survival_time) 
    pred_N_list = np.array(pred_N_list) 
    labels_N_list = np.array(labels_N_list) 
    labels_time_list = np.array(labels_time_list) 
    labels_status_list = np.array(labels_status_list) 
    auc, Sen, Spe, auc_thres  = calculate_metric(labels_N_list, pred_N_list.copy(), auc_thres ) 
    # print('-'*80)
    print(f'threshold = {auc_thres:.4f}, auc = {auc:.4f}, Sen = {Sen:.4f}, Spe = {Spe:.4f} ') 
    
    if ifcal_surv: 
        valid_cindex = lifelines.utils.concordance_index(labels_time_list, Survival_time, labels_status_list)
        valid_cindex_info = 'Valid C-index: %.4f' % valid_cindex 
        print(valid_cindex_info) 
        # if Survival_time.min() == Survival_time.max(): Survival_time[:50] = Survival_time[:50] + 50 
        if Survival_time.min() == Survival_time.max(): 
            print("Get the problem:: Survival_time.min() == Survival_time.max()") 
            best_cutoff_value = Survival_time[0]
        else: 
            best_cutoff_value = hr_values(Survival_time, labels_time_list, labels_status_list, best_cutoff_value)

    # data_ = pd.DataFrame() 
    if save_path: 
        df_features['labels_N'] = labels_N_list 
        df_features['labels_time'] = labels_time_list 
        df_features['labels_status'] = labels_status_list 
        df_features['pred_N'] = pred_N_list 
        df_features['pred_surv_risk'] = Survival_time 
        df_features.to_excel(save_path) 
    return best_cutoff_value, auc_thres

def train_featsEncoding(model, trainloader, ifsavemodel=True, optimizer=None):  
    for epoch in range(1, 3 ):  # loop over the dataset multiple times
        model.train() 
        ii=0
        for i, data in enumerate(trainloader, 0): 
            ####train the 1st time CT
            optimizer.zero_grad()
            x, y, labels = data
            pred, feats = model((x[0]).type(Tensor)) 
            loss_bce = criterion(torch.sigmoid(pred), y[0].type(Tensor)) 
            loss_dice = criterion_dice(torch.sigmoid(pred), y[0].type(Tensor)) 
            alpha=0.5
            loss = alpha*loss_bce + (1-alpha)*loss_dice
            loss.backward() 
            optimizer.step() 
            ii+=1
            print(f'[epoch: {epoch} iteration: {ii}] pred loss: {loss.item():.3f}, {loss_bce.item():.3f}, {loss_dice.item():.3f}') 

            ####train the 2nd time CT
            optimizer.zero_grad()
            pred, feats = model((x[1]).type(Tensor)) 
            loss_bce = criterion(torch.sigmoid(pred), y[1].type(Tensor)) 
            loss_dice = criterion_dice(torch.sigmoid(pred), y[1].type(Tensor)) 
            alpha=0.5
            loss = alpha*loss_bce + (1-alpha)*loss_dice
            loss.backward() 
            optimizer.step() 
            ii+=1
            print(f'[epoch: {epoch} iteration: {ii}] pred loss: {loss.item():.3f}, {loss_bce.item():.3f}, {loss_dice.item():.3f}') 
        if ifsavemodel: 
            if epoch%2==0: 
                torch.save(model.state_dict(), f"./pth/segmamba-featsEncoding-{epoch}.pth") 
            # torch.save(model.state_dict(), f"./pth/segmamba.pth") 
        gc.collect() 

def train_CTSMamba(model, trainloader, ifsavemodel=True, optimizer=None): 
    # writer = SummaryWriter('experiment_1')
    for epoch in range(1, 11 ):  # loop over the dataset multiple times
        model.train()
        train_total_loss = []
        for i, data in enumerate(trainloader, 0): 
            x, y, labels = data
            labels_N = labels[0].unsqueeze_(1).type(Tensor) 
            labels_surv = torch.transpose(torch.stack(labels[1], dim=0), 0,1 ).type(Tensor) 
            # pred_N, pred_surv = model((x[0]*y[0]).type(Tensor),(x[1]*y[1]).type(Tensor)) 
            pred, pred_surv = model( x[0].type(Tensor), x[1].type(Tensor) ) 
            loss_N = criterion(pred, labels_N.type(Tensor)) 
            loss_surv = 0
            loss_list = []
            for i, Loss in enumerate(Losses):
                curr_loss = Loss(labels_surv, pred_surv[i]) * Weights[i] 
                loss_list.append(curr_loss.item()) 
                loss_surv += curr_loss 
            alpha=0.5
            loss = alpha*loss_N + (1-alpha)*loss_surv 
            optimizer.zero_grad()
            loss.backward() 
            optimizer.step() 
            train_total_loss.append(loss.item())

        print('-'*80)
        print(f'[{epoch}-{i}] pred loss: {loss.item():.3f}, {loss_N.item():.3f}, {loss_surv.item():.3f}') 
        print('Train loss: %.4f' % (np.mean(train_total_loss)) )
        # best_cutoff_value = test_model_v2(model, testloader0, X_train)
        # test_model_v2(model, testloader01, X_test, best_cutoff_value)
        if epoch%5==0: 
            torch.save(model.state_dict(), f"./pth/ctsmamba-{epoch}.pth") 
        gc.collect() 

if __name__=="__main__": 
    # Define the argument parser
    parser = argparse.ArgumentParser(description='Train and evaluate a model for lymph node metastasis prediction.')
    parser.add_argument('--training_set', type=str, default='./BDataset/Training_dataset_example.xlsx',
                        help='Path to the training dataset Excel file.')
    parser.add_argument('--validation_set', type=str, default='./BDataset/Training_dataset_example.xlsx',
                        help='Path to the validation dataset Excel file.')
    parser.add_argument('--ifDealWithImbalance', action='store_false',
                        help='Whether to handle class imbalance.')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training and testing.')
    parser.add_argument('--save_path_train', type=str, default='./BDataset/pred_train.xlsx',
                        help='Path to save the training predictions Excel file.')
    parser.add_argument('--save_path_test', type=str, default='./BDataset/pred_test.xlsx',
                        help='Path to save the testing predictions Excel file.')
    parser.add_argument('--metrics_path', type=str, default='./BDataset/metrics_table.csv',
                        help='Path to save the evaluation metrics CSV file.')
    # Parse the arguments
    args = parser.parse_args()

    # Use the parsed arguments in your script
    training_set1 = args.training_set
    interValid_set1 = args.validation_set
    ifDealWithImbalance = args.ifDealWithImbalance
    batch_size = args.batch_size
    save_path_train = args.save_path_train
    save_path_test = args.save_path_test
    metrics_path = args.metrics_path

    # Load the model
    
    model_featsEncoding = TSMamba(in_chans=1, out_chans=1, depths=[2,2,2,2], feat_size=[48, 96, 192, 384]) 
    if cuda: model_featsEncoding = model_featsEncoding.cuda()

    ### Step 1: Read the Excel files containing the training and validation datasets
    # training_set1 = r'./BDataset/Training_dataset_example.xlsx'  
    # interValid_set1 = r'./BDataset/Training_dataset_example.xlsx'  
    X_train = pd.read_excel(training_set1 ) 
    X_test = pd.read_excel(interValid_set1 ) 

    # Extract target labels for lymph node metastasis
    X_train['target_'] = X_train['淋巴结转移（0为N0,1为N+）']
    X_test['target_'] = X_test['淋巴结转移（0为N0,1为N+）']
    # Prepare data for training and testing
    image_mask_files_times_train, label_DrugRest_ptLever_train = prepare_data_v3(X_train) 
    image_mask_files_times_test, label_DrugRest_ptLever_test = prepare_data_v3(X_test)

    # Handle class imbalance if specified
    ifDealWithImbalance=False 
    if ifDealWithImbalance: 
        X_train_path, y_train, X_test_path, y_test = imBalanced_MES(image_mask_files_times_train, label_DrugRest_ptLever_train, num_perClass=100 ) 
    else: 
        X_train_path, y_train = [ image_mask_files_times_train ], [ label_DrugRest_ptLever_train ] 
        X_train_path_test, y_train_test = [ image_mask_files_times_test ], [ label_DrugRest_ptLever_test ] 

    # Create data loaders for training and testing
    ## make the train dataloader
    trainloader = make_dataloader_v3(X_train_path, y_train, bs=2, ifshuffle=False)  ##, iftransform=True 
    ## make the test dataloader
    testloader0 = make_dataloader_v3(X_train_path, y_train, bs=1, ifshuffle=False ) 
    testloader01 = make_dataloader_v3(X_train_path_test, y_train_test, bs=1, ifshuffle=False ) 

    ### Step 3: define and train CTSMamba model  
    model_CTSMamba = CTSMamba_v2() 
    if cuda: model_CTSMamba = model_CTSMamba.cuda() 
    ifloadFeatsEncoder = True
    if ifloadFeatsEncoder: 
        modelFeatsEncoder = r'./pth/segmamba-featsEncoding-2.pth'
        model_CTSMamba.featsEncoder.load_state_dict(torch.load(modelFeatsEncoder))

    ifloadmodelCTSMamba = False
    if ifloadmodelCTSMamba: 
        print('load the model of ', f'./pth/best_model.pth') 
        model_CTSMamba.load_state_dict(torch.load(f'./pth/best_model.pth', map_location=lambda storage, loc: storage)) 

    optimizer = torch.optim.Adam(model_CTSMamba.parameters(), lr=1e-4)
    # optimizer = torch.optim.SGD(model_CTSMamba.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-5) 
    train_CTSMamba(model_CTSMamba, trainloader, ifsavemodel=True, optimizer=optimizer)
    torch.save(model_CTSMamba.state_dict(), f"./pth/ctsmamba.pth") 

    ### Step 4: Evaluate the model and save predictions to Excel files 
    model_CTSMamba.load_state_dict(torch.load(f"./pth/ctsmamba.pth", map_location=lambda storage, loc: storage))
    best_cutoff_value, auc_thres = test_model_v3(model_CTSMamba, X_train_path, X_train, save_path=save_path_train) 
    test_model_v3(model_CTSMamba, X_train_path_test, X_test, best_cutoff_value, save_path=save_path_test, auc_thres=auc_thres) 

