import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency
from lifelines import CoxPHFitter

path = r'.'
training_shanxi_set1 = path + r'/pred_train_add_postClini.xlsx' 
interValid_shanxi_set1 = path + r'/pred_test_add_postClini.xlsx' 
evc1_set1 = path + r'/pred_evc1_add_postClini.xlsx' 
evc2_set1 = path + r'/pred_evc2_add_postClini.xlsx' 

df_features_shanxi_1 = pd.read_excel(training_shanxi_set1 ) 
df_features_shanxi_2 = pd.read_excel(interValid_shanxi_set1 ) 
df_features_evc1 = pd.read_excel(evc1_set1 ) 
df_features_evc2 = pd.read_excel(evc2_set1 )  ##
# df_features_total = pd.concat([df_features_shanxi_1, df_features_shanxi_2, df_features_shanxi_3, df_features_shanxi_4], axis=0)  ###KASHI feike hospital 

print('number of cases for pre NAC and post NAC: ', df_features_shanxi_1.shape, df_features_shanxi_2.shape, df_features_evc1.shape, df_features_evc2.shape)
df_features_shanxi_1['target_'] = df_features_shanxi_1['淋巴结转移（0为N0,1为N+）']
df_features_shanxi_2['target_'] = df_features_shanxi_2['淋巴结转移（0为N0,1为N+）']
df_features_evc1['target_'] = df_features_evc1['淋巴结转移（0为N0,1为N+）']
df_features_evc2['target_'] = df_features_evc2['淋巴结转移（0为N0,1为N+）']

X_train = df_features_shanxi_1 
X_test = df_features_shanxi_2 
df_features_evc1_1 = df_features_evc1
df_features_evc2_1 = df_features_evc2

X_test[X_test['Lauren']==0] = 1 
# X_test.loc[X_test['Lauren'] == 0, 'Lauren'] = np.nan
X_train = X_train.rename(columns= {'pCR': 'Pathological response', 'ypT': 'ypT stage', 'ypN': 'ypN stage', 
                                   'ypTNM': 'ypTNM stage', 'numofLN': 'node numbers', 'treatmentpostNAC': 'Postoperative treatment', } ) 
X_test = X_test.rename(columns= {'pCR': 'Pathological response', 'ypT': 'ypT stage', 'ypN': 'ypN stage', 
                                   'ypTNM': 'ypTNM stage', 'numofLN': 'node numbers', 'treatmentpostNAC': 'Postoperative treatment', } ) 
df_features_evc1_1 = df_features_evc1_1.rename(columns= {'pCR': 'Pathological response', 'ypT': 'ypT stage', 'ypN': 'ypN stage', 
                                   'ypTNM': 'ypTNM stage', 'numofLN': 'node numbers', 'treatmentpostNAC': 'Postoperative treatment', } ) 
df_features_evc2_1 = df_features_evc2_1.rename(columns= {'pCR': 'Pathological response', 'ypT': 'ypT stage', 'ypN': 'ypN stage', 
                                   'ypTNM': 'ypTNM stage', 'numofLN': 'node numbers', 'treatmentpostNAC': 'Postoperative treatment', } ) 

column_sel_e = ['Age', 'BMI', 'Sex', 
                'T stage', 'N stage',
                'Locations', 'Differentiation', 
                'Borrmann', 'Lauren', 
                'Pre-NACT CEA', 'Pre-NACT CA199', 
                'Regimens', 'Course',  
                'Pathological response', 'ypT stage', 'ypN stage', 'ypTNM stage', ####postClini
                'node numbers', 'Postoperative treatment', #####postClini
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

    conditions = [ clini_pred_training['ypT stage']==0, clini_pred_training['ypT stage']==1, clini_pred_training['ypT stage']==2, 
                  clini_pred_training['ypT stage']==3, clini_pred_training['ypT stage']==4,  ]
    choices     = [ "ypT0", "ypT1", "ypT2", "ypT3", "ypT4", ] 
    clini_pred_training["ypT stage"] = np.select(conditions, choices, )  ## default=np.nan

    conditions = [ clini_pred_training['ypN stage']==0, clini_pred_training['ypN stage']==1, 
                  clini_pred_training['ypN stage']==2, clini_pred_training['ypN stage']==3, ]
    choices     = [ "ypN0", "ypN1", "ypN2", "ypN3", ] 
    clini_pred_training["ypN stage"] = np.select(conditions, choices, )  ## default=np.nan

    conditions = [ clini_pred_training['ypTNM stage']==0, clini_pred_training['ypTNM stage']==1, 
                  clini_pred_training['ypTNM stage']==2, clini_pred_training['ypTNM stage']==3, clini_pred_training['ypTNM stage']==4, ]
    choices     = [ "Stage 0", "Stage I", "Stage II", "Stage III", "Stage IV", ] 
    clini_pred_training["ypTNM stage"] = np.select(conditions, choices, )  ## default=np.nan
    clini_pred_training["Postoperative treatment"] = np.where(clini_pred_training['Postoperative treatment']>0, 'adjuvant chemotherapy', 'none')  

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
    ######################## postClini #########################
    order_dict['ypT stage'] = [ "ypT0", "ypT1", "ypT2", "ypT3", "ypT4" ] 
    order_dict['ypN stage'] = [ "ypN0", "ypN1", "ypN2", "ypN3" ] 
    order_dict['ypTNM stage'] = [ "Stage 0", "Stage I", "Stage II", "Stage III", "Stage IV" ] 
    order_dict['Postoperative treatment'] = ["none", "adjuvant chemotherapy",]
    return clini_pred_training, order_dict 



import statsmodels.api as sm
import pandas as pd

from lifelines import CoxPHFitter
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.plotting import plot_lifetimes
from matplotlib.table import Table
from lifelines.plotting import add_at_risk_counts
import matplotlib
matplotlib.rcParams.update({'font.size': 15})
import matplotlib.pyplot as plt

color_sns_list=[(138/255, 173/255, 216/255), (251/255, 195/255, 167/255), (0.12156862745098039, 0.4666666666666667, 0.7058823529411765), (1.0, 0.4980392156862745, 0.054901960784313725), 
 (0.17254901960784313, 0.6274509803921569, 0.17254901960784313), (0.8392156862745098, 0.15294117647058825, 0.1568627450980392), 
 (0.5803921568627451, 0.403921568627451, 0.7411764705882353), (0.5490196078431373, 0.33725490196078434, 0.29411764705882354), 
 (0.8901960784313725, 0.4666666666666667, 0.7607843137254902), (0.4980392156862745, 0.4980392156862745, 0.4980392156862745), 
 (0.7372549019607844, 0.7411764705882353, 0.13333333333333333), (0.09019607843137255, 0.7450980392156863, 0.8117647058823529)] 

color_sns_list=[(227/255, 141/255, 140/255), (154/255, 184/255, 212/255), ]

def plot_KMcurves_survRate_at_times(data, fname='shanxi', duration_col = 'time', event_col = 'event', best_cutoff_value = None, title=None): 
    # Create KM curve with the best cutoff
    data_ = data[column_sel_e + ['DLS', duration_col, event_col ] ] 
    # data_.loc[data_[duration_col]<duration_time, 'Died' ] = 1
    if best_cutoff_value is None: 
        # ## method 2. Median 
        best_cutoff_value = np.median( data_['DLS'].to_numpy() ) 
        # ## method 3. Mean 
        # best_cutoff_value = np.mean( data_['DLS'].to_numpy() ) 
        # best_cutoff_value = np.percentile(data_['DLS'].to_numpy(), 75)
        
    group1 = data_[data_['DLS'] <= best_cutoff_value]
    group2 = data_[data_['DLS'] > best_cutoff_value] 
    result = logrank_test(group1[duration_col], group2[duration_col], event_observed_A=group1[event_col], event_observed_B=group2[event_col]) 
    print("best_cutoff_value", best_cutoff_value, result.p_value) 

    data_['risk_group'] = (data_['DLS'] > best_cutoff_value).astype(int)
    cph = CoxPHFitter()
    cph.fit(data_[[duration_col, event_col, 'risk_group']], duration_col=duration_col, event_col=event_col)
    # Printing the Cox model results 
    hr = cph.summary.loc['risk_group', 'exp(coef)'] 
    lower_ci = cph.summary.loc['risk_group', 'exp(coef) lower 95%']
    upper_ci = cph.summary.loc['risk_group', 'exp(coef) upper 95%']
    print(f"Hazard Ratio (HR): {hr:.4f}, 95% CI: ({lower_ci:.4f}, {upper_ci:.4f})")

    # fig, ax = plt.subplots(figsize=(8, 6))
    # fig, (ax, table_ax) = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [4, 1]})

    fig, ax = plt.subplots(figsize=(10, 8))

    km_curve_group1 = data_[data_['DLS'] <= best_cutoff_value]  ##'pred'
    km_curve_group2 = data_[data_['DLS'] > best_cutoff_value]  ##'pred'

    kmf_group1 = KaplanMeierFitter() 
    kmf_group2 = KaplanMeierFitter() 
    kmf_group1.fit(durations=km_curve_group1[duration_col], event_observed=km_curve_group1[event_col], label='high risk')  ##f'Group <= {best_cutoff_value}'
    kmf_group2.fit(durations=km_curve_group2[duration_col], event_observed=km_curve_group2[event_col], label='low risk')  ##f'Group > {best_cutoff_value}'

    # kmf_group1.plot_survival_function(ax=ax, color='blue')
    # kmf_group2.plot_survival_function(ax=ax, color='orange')
    kmf_group1.plot(ax=ax, color=color_sns_list[0], label='high risk', ci_alpha=0.1, linewidth=4, ci_show=False)  ##f'Group <= {best_cutoff_value}'
    kmf_group2.plot(ax=ax, color=color_sns_list[1], label='low risk',  ci_alpha=0.1, linewidth=4, ci_show=False)  ##f'Group > {best_cutoff_value}'
    # Get survival probabilities for censored points
    survival_prob_group1 = kmf_group1.survival_function_.loc[km_curve_group1[duration_col]].values.flatten()
    survival_prob_group2 = kmf_group2.survival_function_.loc[km_curve_group2[duration_col]].values.flatten()

    ## 添加 'number at risk' 表格
    add_at_risk_counts(kmf_group1, kmf_group2, ax=ax, rows_to_show=['At risk'], fontsize=18)  ##  fontsize=18
    ax.legend(prop={'size': 18})  ##, 'weight': 'bold'  prop={'size': 18}

    if result.p_value<0.0001: 
        ax.text(0.05, 0.05, f'p<0.0001', transform=ax.transAxes, fontsize=15, verticalalignment='bottom')  ##, horizontalalignment='left'
    else: 
        ax.text(0.05, 0.05, f'p={result.p_value:.4f}', transform=ax.transAxes, fontsize=15, verticalalignment='bottom')  ##, horizontalalignment='left'
    ax.set_xlabel('Time of months', fontsize=17)
    ax.set_ylabel('Survival probability', fontsize=17)
    # ax.grid(True, linestyle='--', linewidth=0.5, color='gray')  # lighter grid lines
    ax.tick_params(which='major', labelsize=18)  ##axis='both', 
    if title:
        plt.suptitle(title, fontsize=20, y=0.95, ha='center')  # 调整y参数使标题更接近图形
    plt.tight_layout()  # 为标题留出空间
    # plt.show()
    plt.savefig(f'./BEvaluation/subgroup KM_curves_{fname}_{duration_col}.svg', bbox_inches='tight')

    # return best_cutoff_value 
    ################################ survival probability at time of [12, 24, 36, 48, 60]################################
    results_rate = {}
    for month in [12, 24, 36, 48, 60]:
        survival_group1_at_time = kmf_group1.survival_function_at_times(month).values[0]
        survival_group2_at_time = kmf_group2.survival_function_at_times(month).values[0]
        # print(f"Survival rate at {month} months - High risk: {survival_group1_at_time:.3f}, Low risk: {survival_group2_at_time:.3f}")
        results_rate[f'{month}_month_os_rate_H_L'] = f'{survival_group1_at_time:.3f} vs {survival_group2_at_time:.3f}' 
    return best_cutoff_value, results_rate

def plot_KMcurves_survRate_at_times_combine(data, fname='shanxi', duration_col = 'time', event_col = 'event', best_cutoff_value = None, 
                                            title=None, ax=None, show_legend=True): 
    # Create KM curve with the best cutoff
    data_ = data[column_sel_e + ['DLS', duration_col, event_col ] ] 
    # data_.loc[data_[duration_col]<duration_time, 'Died' ] = 1
    if best_cutoff_value is None: 
        # ## method 2. Median 
        best_cutoff_value = np.median( data_['DLS'].to_numpy() ) 
    group1 = data_[data_['DLS'] <= best_cutoff_value]
    group2 = data_[data_['DLS'] > best_cutoff_value] 
    result = logrank_test(group1[duration_col], group2[duration_col], event_observed_A=group1[event_col], event_observed_B=group2[event_col]) 
    print("best_cutoff_value", best_cutoff_value, result.p_value) 

    data_['risk_group'] = (data_['DLS'] > best_cutoff_value).astype(int)
    cph = CoxPHFitter()
    cph.fit(data_[[duration_col, event_col, 'risk_group']], duration_col=duration_col, event_col=event_col)
    # Printing the Cox model results 
    hr = cph.summary.loc['risk_group', 'exp(coef)'] 
    lower_ci = cph.summary.loc['risk_group', 'exp(coef) lower 95%']
    upper_ci = cph.summary.loc['risk_group', 'exp(coef) upper 95%']
    print(f"Hazard Ratio (HR): {hr:.4f}, 95% CI: ({lower_ci:.4f}, {upper_ci:.4f})")

    if ax is None: 
        fig, ax = plt.subplots(figsize=(10, 8))

    km_curve_group1 = data_[data_['DLS'] <= best_cutoff_value]  ##'pred'
    km_curve_group2 = data_[data_['DLS'] > best_cutoff_value]  ##'pred'

    kmf_group1 = KaplanMeierFitter() 
    kmf_group2 = KaplanMeierFitter() 
    kmf_group1.fit(durations=km_curve_group1[duration_col], event_observed=km_curve_group1[event_col], label='high risk')  ##f'Group <= {best_cutoff_value}'
    kmf_group2.fit(durations=km_curve_group2[duration_col], event_observed=km_curve_group2[event_col], label='low risk')  ##f'Group > {best_cutoff_value}'

    if show_legend: 
        kmf_group1.plot(ax=ax, color=color_sns_list[0], label='high risk', ci_alpha=0.1, linewidth=4, ci_show=False)  ##f'Group <= {best_cutoff_value}'
        kmf_group2.plot(ax=ax, color=color_sns_list[1], label='low risk',  ci_alpha=0.1, linewidth=4, ci_show=False)  ##f'Group > {best_cutoff_value}'
    else: 
        kmf_group1.plot(ax=ax, color=color_sns_list[0],                    ci_alpha=0.1, linewidth=4, ci_show=False)  ##f'Group <= {best_cutoff_value}'
        kmf_group2.plot(ax=ax, color=color_sns_list[1],                    ci_alpha=0.1, linewidth=4, ci_show=False)  ##f'Group > {best_cutoff_value}'

    ax.set_ylim(0.15, 1.04) 
    yticks = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ax.set_yticks(yticks)  
    ax.set_xlim(-5, 85)  # Set x-axis range to slightly more than 100
    xticks = range(0, 91, 20)
    ax.set_xticks(xticks)  # Ensure the xticks remain consistent after adding at risk counts
    add_at_risk_counts(kmf_group1, kmf_group2, ax=ax, rows_to_show=['At risk'], fontsize=16, xticks=xticks)  ##  fontsize=18
    ax.set_xticks(xticks)  # Ensure the xticks remain consistent after adding at risk counts
    ax.get_legend().remove() ##移除所有组的图例

    if result.p_value<0.0001: 
        ax.text(0.05, 0.05, f'p<0.0001\nHR: {hr:.3f}({lower_ci:.3f}-{upper_ci:.3f})'.replace('.', '·'), transform=ax.transAxes, fontsize=15, verticalalignment='bottom')  ##, horizontalalignment='left'
    else: 
        ax.text(0.05, 0.05, f'p={result.p_value:.4f}\nHR: {hr:.3f}({lower_ci:.3f}-{upper_ci:.3f})'.replace('.', '·'), transform=ax.transAxes, fontsize=15, verticalalignment='bottom')  ##, horizontalalignment='left'


    # if title=='Age: ≥60': 
    if show_legend: 
        ax.legend(prop={'size': 16})  ##, 'weight': 'bold'  prop={'size': 18}
        if title in [ 'ypT stage: ypT0' ]:   ##'Pathological response: pCR'
            ax.legend(loc='center left') 

    ax.set_xlabel('Time of months', fontsize=16)
    ax.set_ylabel('Survival probability', fontsize=17)
    ax.tick_params(which='major', labelsize=18)  ##axis='both', 
    plt.yticks([tick for tick in plt.yticks()[0]], [
            f'{tick:.1f}'.replace('.', '·') for tick in plt.yticks()[0]])
    if title:
        ax.set_title(title, fontsize=20,  ha='center') 
    plt.tight_layout() 

    ################################ survival probability at time of [12, 24, 36, 48, 60]################################
    results_rate = {}
    for month in [12, 24, 36, 48, 60]:
        survival_group1_at_time = kmf_group1.survival_function_at_times(month).values[0]
        survival_group2_at_time = kmf_group2.survival_function_at_times(month).values[0]
        # print(f"Survival rate at {month} months - High risk: {survival_group1_at_time:.3f}, Low risk: {survival_group2_at_time:.3f}")
        results_rate[f'{month}_month_os_rate_H_L'] = f'{survival_group1_at_time:.3f} vs {survival_group2_at_time:.3f}' 
    return best_cutoff_value, results_rate 



datas = [X_train, X_test, df_features_evc2_1] 
fnames = ['train', 'interValid', 'evc2' ] 

########################### Sub-group analysis KM curves ####################################
########################### Sub-group analysis KM curves postClini ####################################
data_all = pd.concat(datas, axis=0)
data_all_cate, order_dict = prepare_data(data_all) 

# ########## 检查是否正确 #################################
# for idx, key in enumerate(order_dict.keys()): 
#     print(key, '::--', data_all_cate[key].unique() )
#     print(data_all_cate[key].value_counts() ) 

########## only keep postClini #################################
keys_to_keep = [ 'ypT stage', 'ypN stage', 'ypTNM stage', 'node numbers', 'Postoperative treatment' ]  ##'Pathological response',
order_dict = {key: order_dict[key] for key in keys_to_keep if key in order_dict}

keys = list( order_dict.keys() ) 
results, index = [], [] 
# # fig, axs = plt.subplots(6, 5, figsize=(25, 30))
# fig, axs = plt.subplots(7, 4, figsize=(25, 40))
fig, axs = plt.subplots(6, 4, figsize=(25, 35))
axs = axs.flatten() 
i_fig = 0 
for idx, key in enumerate(keys): 
    for value in order_dict[key]: 
        data = data_all_cate[data_all_cate[key]==value] 
        ax = axs[i_fig]
        
        # best_cutoff_value, results_rate = plot_KMcurves_survRate_at_times(data, fname=f'data_all_subgroup_{key}_{value}', best_cutoff_value=None, title=f'{key}: {value}') 
        best_cutoff_value, results_rate = plot_KMcurves_survRate_at_times_combine(data, fname=f'data_all_subgroup_{key}_{value}', best_cutoff_value=None, 
                                                                                  title=f'{key}: {value}', ax=ax, show_legend= (i_fig==0) ) 
        results.append(results_rate ) 
        index.append(f'{key}_{value}') 
        i_fig += 1 
# 删除多余的子图 
for j in range(i_fig, len(axs)): 
    fig.delaxes(axs[j]) 
plt.tight_layout(rect=[0, 0, 1, 0.95])  # 为整体标题留出空间
plt.savefig('./BEvaluation/all_subgroup_KM_curves_add_postClini.svg', bbox_inches='tight')
plt.savefig('./BEvaluation/all_subgroup_KM_curves_add_postClini.png', bbox_inches='tight', dpi=400)

df = pd.DataFrame(results, index=index ) 
df.to_excel(f'./BEvaluation/subgroup KM_curves_OS rate table_add_postClini.xlsx', encoding='utf-8-sig') 

print()

for i in range(len(column_sel_e)): 
    best_cutoff_value, results_rate = plot_KMcurves_survRate_at_times(data_all_cate, fname='data_all', best_cutoff_value=None) 
