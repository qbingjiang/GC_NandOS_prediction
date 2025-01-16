import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency
from lifelines import CoxPHFitter

path = r'.'
training_shanxi_set1 = path + r'/pred_train.xlsx' 
interValid_shanxi_set1 = path + r'/pred_test.xlsx' 
evc1_set1 = path + r'/pred_evc1.xlsx' 
evc2_set1 = path + r'/pred_evc2.xlsx' 

df_features_shanxi_1 = pd.read_excel(training_shanxi_set1 ) 
df_features_shanxi_2 = pd.read_excel(interValid_shanxi_set1 ) 
df_features_evc1 = pd.read_excel(evc1_set1 ) 
df_features_evc2 = pd.read_excel(evc2_set1 ) 

print('number of cases for pre NAC and post NAC: ', df_features_shanxi_1.shape, df_features_shanxi_2.shape, df_features_evc1.shape, df_features_evc2.shape)
df_features_shanxi_1['target_'] = df_features_shanxi_1['淋巴结转移（0为N0,1为N+）']
df_features_shanxi_2['target_'] = df_features_shanxi_2['淋巴结转移（0为N0,1为N+）']
df_features_evc1['target_'] = df_features_evc1['淋巴结转移（0为N0,1为N+）']
df_features_evc2['target_'] = df_features_evc2['淋巴结转移（0为N0,1为N+）']

X_train = df_features_shanxi_1 
X_test = df_features_shanxi_2 
df_features_evc1_1 = df_features_evc1
df_features_evc2_1 = df_features_evc2

column_sel_e = ['Age', 'BMI', 'Sex', 
                'T stage', 'N stage',
                'Locations', 'Differentiation', 
                'Borrmann', 'Lauren', 
                'Pre-NACT CEA', 'Pre-NACT CA199', 
                'Regimens', 'Course' 
                ]

X_train  = X_train[column_sel_e + ['labels_time', 'labels_status'] + ['pred_surv_risk'] + ['labels_N', 'pred_N',]]
X_test  = X_test[column_sel_e + ['labels_time', 'labels_status'] + ['pred_surv_risk'] + ['labels_N', 'pred_N',]] 
df_features_evc1_1  = df_features_evc1_1[column_sel_e + ['labels_time', 'labels_status'] + ['pred_surv_risk'] + ['labels_N', 'pred_N',]]
df_features_evc2_1  = df_features_evc2_1[column_sel_e + ['labels_time', 'labels_status'] + ['pred_surv_risk'] + ['labels_N', 'pred_N',]]

X_train = X_train.rename(columns= {'pred_surv_risk': 'DLS', 'labels_time': 'time', 'labels_status': 'event'} ) 
X_test = X_test.rename(columns= {'pred_surv_risk': 'DLS', 'labels_time': 'time', 'labels_status': 'event'} ) 
df_features_evc1_1 = df_features_evc1_1.rename(columns= {'pred_surv_risk': 'DLS', 'labels_time': 'time', 'labels_status': 'event'} ) 
df_features_evc2_1 = df_features_evc2_1.rename(columns= {'pred_surv_risk': 'DLS', 'labels_time': 'time', 'labels_status': 'event'} ) 

def prepare_data(clini_pred_training): 
    clini_pred_training['Age'] = np.where(clini_pred_training['Age']>=60, '≥60', '<60')  
    clini_pred_training['BMI'] = np.where(clini_pred_training['BMI']>24, '>24', '≤24')
    clini_pred_training['Target'] = np.where(clini_pred_training['labels_N']>0, 'N+', 'N0') 

    conditions = [ clini_pred_training['Borrmann']==1, clini_pred_training['Borrmann']==2, clini_pred_training['Borrmann']==3, clini_pred_training['Borrmann']==4 ]
    choices     = [ "type I+II", "type I+II", "type III+IV", "type III+IV" ] 
    clini_pred_training["Borrmann"] = np.select(conditions, choices, )  ## default=np.nan

    conditions = [ clini_pred_training['Lauren']==1, clini_pred_training['Lauren']==2, clini_pred_training['Lauren']==3 ]
    choices     = [ "intestinal", "diffuse", "mixed" ] 
    clini_pred_training["Lauren"] = np.select(conditions, choices, )  ## default=np.nan

    conditions = [ clini_pred_training['Differentiation']==1, clini_pred_training['Differentiation']==2, clini_pred_training['Differentiation']==3 ]
    choices     = [ "well+moderate","well+moderate", "poor" ] 
    clini_pred_training["Differentiation"] = np.select(conditions, choices, )  ## default=np.nan

    conditions = [ clini_pred_training['Locations']==0, clini_pred_training['Locations']==1, clini_pred_training['Locations']==2, clini_pred_training['Locations']==3 ]
    choices     = [ "cardia", "body+antrum", "body+antrum","multiple+whole" ] 
    clini_pred_training["Location"] = np.select(conditions, choices, )  ## default=np.nan

    conditions = [ clini_pred_training['Pre-NACT CEA']==0, clini_pred_training['Pre-NACT CEA']==1, ]
    choices     = [ "normal", "elevated"  ] 
    clini_pred_training["CEA"] = np.select(conditions, choices, )  ## default=np.nan

    conditions = [ clini_pred_training['Pre-NACT CA199']==0, clini_pred_training['Pre-NACT CA199']==1, ]
    choices     = [ "normal", "elevated" ] 
    clini_pred_training["CA199"] = np.select(conditions, choices, )  ## default=np.nan

    conditions = [ clini_pred_training['Regimens']==2, clini_pred_training['Regimens']==3, ]
    choices     = [ "doublet-drug", "triplet-drug" ] 
    clini_pred_training["Regimens"] = np.select(conditions, choices, )  ## default=np.nan

    conditions = [ clini_pred_training['Course']==0, clini_pred_training['Course']==1, ]
    choices     = [ "≤3", ">3" ] 
    clini_pred_training["Cycles"] = np.select(conditions, choices, )  ## default=np.nan

    conditions = [ clini_pred_training['Sex']==0, clini_pred_training['Sex']==1, ]
    choices     = [ "female", "male" ] 
    clini_pred_training["Sex"] = np.select(conditions, choices, )  ## default=np.nan

    conditions = [ clini_pred_training['N stage']==0, clini_pred_training['N stage']==1, 
                clini_pred_training['N stage']==2, clini_pred_training['N stage']==3, 
                ]
    choices     = [ "N0", "N1+N2+N3", "N1+N2+N3", "N1+N2+N3" ] 
    clini_pred_training["cN stage"] = np.select(conditions, choices, )  ## default=np.nan

    conditions = [ clini_pred_training['T stage']==1, clini_pred_training['T stage']==2, 
                clini_pred_training['T stage']==3, clini_pred_training['T stage']==4, 
                ]
    choices     = [ "T1+T2", "T1+T2", "T3+T4", "T3+T4" ] 
    clini_pred_training["cT stage"] = np.select(conditions, choices, )  ## default=np.nan

    order_dict = {} 
    order_dict['Age'] = [ '≥60', '<60', ] 
    order_dict['BMI'] = [ '>24', '≤24', ] 
    order_dict['Borrmann'] = [ "type I+II", "type III+IV"]
    order_dict['Lauren'] = [ "intestinal", "diffuse", "mixed" ] 
    order_dict['Differentiation'] = [ "well+moderate", "poor" ] 
    order_dict['Location'] = [ "cardia", "body+antrum", "multiple+whole" ]
    order_dict['CEA'] = [ "normal", "elevated" ] 
    order_dict['CA199'] = [ "normal", "elevated" ] 
    order_dict['Regimens'] = [ "doublet-drug", "triplet-drug" ] 
    order_dict['Cycles'] = [ "≤3", ">3" ] 
    order_dict['Sex'] = [ "female", "male" ] 
    order_dict['cN stage'] = [ "N0", "N1+N2+N3" ] 
    order_dict['cT stage'] = [ "T1+T2", "T3+T4" ] ##"cT1", 

    return clini_pred_training, order_dict 

def cal_NPV(y_true, y_pred): 
    ### calculate negative predictive value (NPV) 
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    npv = tn / (tn + fn) 
    return npv
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
    return results, optimal_threshold 


import numpy as np
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
        if np.sum(y_true_resampled)==0 or np.sum(y_true_resampled)==len(y_true_resampled): 
            continue
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
                 "optimal_threshold": optimal_threshold } 
    return results, optimal_threshold
def calculate_metrics_subgroup(df_train, df_test1, df_test2, df_test3, category, value_list ): 
    def pred_dl(df, category, v): 
        y_true = df.loc[df[category]==v, 'labels_N'].to_numpy() 
        y_proba = df.loc[df[category]==v, 'pred_N'].to_numpy() 
        return y_true, y_proba 
    res = {} 
    for v in value_list: 
        y_true_1, y_proba_1 = pred_dl(df_train, category, v)
        y_true_2, y_proba_2 = pred_dl(df_test1, category, v)
        y_true_3, y_proba_3 = pred_dl(df_test2, category, v)
        y_true_4, y_proba_4 = pred_dl(df_test3, category, v) 
        metrics_DL_1, optimal_threshold_1 = calculate_metrics_save_ConfusionMatrix(y_true_1, y_proba_1, optimal_threshold=None, n_iterations=100 )  ##save_fig='ConfusionMatrix DLHC 1.svg', 
        metrics_DL_2, optimal_threshold_2 = calculate_metrics_save_ConfusionMatrix(y_true_2, y_proba_2, optimal_threshold=optimal_threshold_1, n_iterations=100 )  ##save_fig='ConfusionMatrix DLHC 1.svg', 
        metrics_DL_3, optimal_threshold_3 = calculate_metrics_save_ConfusionMatrix(y_true_3, y_proba_3, optimal_threshold=optimal_threshold_1, n_iterations=100 )  ##save_fig='ConfusionMatrix DLHC 1.svg', 
        metrics_DL_4, optimal_threshold_4 = calculate_metrics_save_ConfusionMatrix(y_true_4, y_proba_4, optimal_threshold=optimal_threshold_1, n_iterations=100 )  ##save_fig='ConfusionMatrix DLHC 1.svg', 
        # res[v] = [[metrics_DL_1, optimal_threshold_1], [metrics_DL_2, optimal_threshold_2], [metrics_DL_3, optimal_threshold_3], [metrics_DL_4, optimal_threshold_4], ] 
        res[v] = [metrics_DL_1, metrics_DL_2, metrics_DL_3, metrics_DL_4 ] 
    return res 
def subgroup_evaluation(df_train, df_test1, df_test2, df_test3, categories, categories_num=[3, 4]): 
    results = {} 
    for i in range(len(categories)): 
        category = categories[i]
        nums = categories_num[i]
        results[category] = {} 
        res = calculate_metrics_subgroup(df_train, df_test1, df_test2, df_test3, category, nums ) 
        results[category] = res 
        res_reset = []
        res_values = list(res.values() ) 
        for i_d in range(len(res_values[0])):
            for i_sg in range(len(res_values)): 
                res_reset.append(res_values[i_sg][i_d]) 
        nums = [str(n) for n in nums]
        nums_sub = [category+'_'+n for n in nums ]
        df = pd.DataFrame(res_reset, 
                    index= [['train']*len(res_values) + ['interValid']*len(res_values) + ['evc1']*len(res_values) + ['evc2']*len(res_values), 
                            nums_sub*4 ] ) 
        nums_ =  '_'.join(nums) 
        df = df.transpose() 
        df.to_csv(f'./Blogistic and cox analysis/metrics_table_subgroup_{category}_{nums_}.csv', encoding='utf-8-sig') 

    return results 

import matplotlib
matplotlib.rcParams.update({'font.size': 15})
import matplotlib.pyplot as plt

datas = [X_train, X_test, df_features_evc1_1, df_features_evc2_1] 
fnames = ['train', 'interValid', 'evc1', 'evc2' ] 

########################### Sub-group analysis LNM prediction ####################################
data_all = pd.concat(datas, axis=0) 
data_all_cate, order_dict = prepare_data(data_all) 

keys = list( order_dict.keys() ) 
keys = keys[:11]
results, index = [], [] 

for idx, key in enumerate(keys): 
    # if idx>4: 
    # for value in order_dict[key]: 
    results = subgroup_evaluation(data_all_cate[:278], data_all_cate[278:278+120], 
                                data_all_cate[278+120:278+120+335], data_all_cate[278+120+335:278+120+335+288], [key], [order_dict[key]] )


########################## Radar maps of subgroup analysis #################################
# 定义数据
categories = ['AUC', 'Accuracy', 'Sensitivity', 'Specificity', 'F1 Score', 'Precision', 'NPV']
from math import pi 
def radar_plot(categories, values_list, fname=None): 
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    plt.xticks(angles[:-1], categories)
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=7)
    plt.ylim(0, 1)
    for model, values in values_list.items():
        values += values[:1]  # 使图闭合
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model)
        # ax.fill(angles, values, alpha=0.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1)) 
    plt.tight_layout()  # 为标题留出空间
    if fname: 
        plt.savefig(fname, bbox_inches='tight')

def radar_plot_combine(categories, values_list, fname=None, ax=None, title=None, show_legend=False ): 
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    # Initialize radar chart
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    # Draw one axe per variable + add labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    # Draw ylabels
    ax.set_rlabel_position(0)
    ax.yaxis.set_tick_params(labelsize=7)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=7)
    ax.set_ylim(0, 1)
    # Plot data
    for model, values in values_list.items():
        values += values[:1]  # Close the loop
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model)
    if show_legend: 
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    if title:
        # plt.suptitle(title, fontsize=20, y=0.95, ha='center')  # 调整y参数使标题更接近图形
        ax.set_title(title, fontsize=20,  ha='center') 

import re 
def extract_number(value):
    match = re.match(r"(\d+\.\d+)", value)
    return float(match.group(1)) if match else value

#### subgroup radar plot
for idx, key in enumerate(keys): 
    results = [key], [order_dict[key]] 
    category=key 
    nums_ =  '_'.join(order_dict[key]) 
    filename = f'./Blogistic and cox analysis/metrics_table_subgroup_{category}_{nums_}.csv' 
    metrics_ = pd.read_csv(filename, header=[0,1], index_col=0) 
    metrics_ = metrics_.rename(index={'precision': 'Precision', 'npv': 'NPV'}) 
    nums_sub = [category+'_'+n for n in order_dict[key] ] 
    values_list={}
    for sub_colname in nums_sub: 
        mat = metrics_.loc[categories, [('train', sub_colname), ('interValid', sub_colname), 
                                        ('evc1', sub_colname), ('evc2', sub_colname)] ].applymap(extract_number) 
        values_list['train'] = list(mat.loc[categories, [('train', sub_colname), ] ].to_numpy()[:,0] ) 
        values_list['interValid'] = list(mat.loc[categories, [('interValid', sub_colname), ] ].to_numpy()[:,0] )
        values_list['evc1'] = list(mat.loc[categories, [('evc1', sub_colname), ] ].to_numpy()[:,0] )
        values_list['evc2'] = list(mat.loc[categories, [('evc2', sub_colname), ] ].to_numpy()[:,0] )
        fname=f'./Blogistic and cox analysis/subgroup radarPlot_{sub_colname}.svg'
        radar_plot(categories, values_list, fname=fname) 
plt.close()

fig, axs = plt.subplots(6, 4, figsize=(25, 35), subplot_kw=dict(polar=True))
axs = axs.flatten()
i=0
for idx, key in enumerate(keys): 
    results = [key], [order_dict[key]] 
    category=key 
    nums_ =  '_'.join(order_dict[key]) 
    filename = f'./Blogistic and cox analysis/metrics_table_subgroup_{category}_{nums_}.csv' 
    metrics_ = pd.read_csv(filename, header=[0,1], index_col=0) 
    metrics_ = metrics_.rename(index={'precision': 'Precision', 'npv': 'NPV'}) 
    nums_sub = [category+'_'+n for n in order_dict[key] ] 
    values_list={}
    for i_sub in range(len(nums_sub)): 
        sub_colname = nums_sub[i_sub] 
        mat = metrics_.loc[categories, [('train', sub_colname), ('interValid', sub_colname), 
                                        ('evc1', sub_colname), ('evc2', sub_colname)] ].applymap(extract_number) 
        values_list['train'] = list(mat.loc[categories, [('train', sub_colname), ] ].to_numpy()[:,0] ) 
        values_list['interValid'] = list(mat.loc[categories, [('interValid', sub_colname), ] ].to_numpy()[:,0] )
        values_list['evc1'] = list(mat.loc[categories, [('evc1', sub_colname), ] ].to_numpy()[:,0] )
        values_list['evc2'] = list(mat.loc[categories, [('evc2', sub_colname), ] ].to_numpy()[:,0] )
        fname=f'./Blogistic and cox analysis/subgroup radarPlot_{sub_colname}.svg'
        radar_plot_combine(categories, values_list, ax=axs[i], show_legend=(i == 0), title=f'{key}: {order_dict[key][i_sub]}') 
        i = i+1


plt.tight_layout(rect=[0, 0, 1, 0.95])  # 为整体标题留出空间
plt.savefig('./Blogistic and cox analysis/subgroup radarPlot_all.svg', bbox_inches='tight')

# plt.show()
print()
