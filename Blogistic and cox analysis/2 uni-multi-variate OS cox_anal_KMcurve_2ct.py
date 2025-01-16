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
df_features_evc2 = pd.read_excel(evc2_set1 )  ##

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
                # 'T stage', 'N stage',
                'Locations', 'Differentiation', 
                'Borrmann', 'Lauren', 
                'Pre-NACT CEA', 'Pre-NACT CA199', 
                'Regimens', 'Course' 
                ] 

X_train  = X_train[column_sel_e + ['labels_time', 'labels_status'] + ['pred_surv_risk'] ]
X_test  = X_test[column_sel_e + ['labels_time', 'labels_status'] + ['pred_surv_risk'] ] 

df_features_evc1_1  = df_features_evc1_1[column_sel_e + ['labels_time', 'labels_status'] + ['pred_surv_risk'] ]
df_features_evc2_1  = df_features_evc2_1[column_sel_e + ['labels_time', 'labels_status'] + ['pred_surv_risk'] ]

X_train = X_train.rename(columns= {'pred_surv_risk': 'DLS', 'labels_time': 'time', 'labels_status': 'event'} ) 
X_test = X_test.rename(columns= {'pred_surv_risk': 'DLS', 'labels_time': 'time', 'labels_status': 'event'} ) 
df_features_evc1_1 = df_features_evc1_1.rename(columns= {'pred_surv_risk': 'DLS', 'labels_time': 'time', 'labels_status': 'event'} ) 
df_features_evc2_1 = df_features_evc2_1.rename(columns= {'pred_surv_risk': 'DLS', 'labels_time': 'time', 'labels_status': 'event'} ) 

def cox_analysis(data, fname='shanxi'): 

    column_sel = column_sel_e 
    for col in column_sel[2:]: 
        data[col] = data[col].astype('category')
    # dummy_variables = pd.get_dummies(data[column_sel[2:]], prefix=column_sel[2:], drop_first=True)  ## 
    # data = pd.concat([data, dummy_variables], axis=1)
    data['intercept'] = 1 
    column_sel = ['DLS']+column_sel

    # Fit univariate Cox regression model
    univariate_results = [] 
    for col in column_sel: 
        if col in column_sel[3:]: 
            dummy_vars = pd.get_dummies(data[col], drop_first=True, prefix=col) 
            data_modified = pd.concat([data[['time', 'event']], dummy_vars], axis=1) 
            dummy_cols = dummy_vars.columns.to_list()
        else: 
            data_modified = data[[col] + ['time', 'event'] ] 
            dummy_cols = [col]
        
        cph = CoxPHFitter()
        cph.fit(data_modified, duration_col='time', event_col='event')
        summary = cph.summary
        summary['Variable'] = col
        univariate_results.append(summary)
        
    univariate_results_df = pd.concat(univariate_results)

    univariate_results_df.to_csv(f'./Blogistic and cox analysis/{fname} univariate_cox_analysis.csv')

    feats_selected_from_UnivariateAnal = univariate_results_df[univariate_results_df['p'] < 0.05].index.to_list()

    print(univariate_results_df)
    print('Features with p-value < 0.05:')
    print(feats_selected_from_UnivariateAnal)

    # Select significant features from univariate analysis
    feats_selected_from_UniAnal_list = []
    for i in range(len(feats_selected_from_UnivariateAnal)): 
        feats_selected_from_UniAnal_list.append(feats_selected_from_UnivariateAnal[i].split('_')[0]) 
    feats_selected_from_UniAnal_list = list(set(feats_selected_from_UniAnal_list)) 
    column_sel_from_UniAnal = [feat for feat in feats_selected_from_UniAnal_list if feat in column_sel]
    column_sel_from_UniAnal = list( set(column_sel_from_UniAnal)-set(['DLS']) )
    dummy_vars = pd.get_dummies(data[column_sel_from_UniAnal], drop_first=True, prefix=column_sel_from_UniAnal)
    dummy_vars = pd.concat([data[['DLS']], dummy_vars], axis=1)
    data_modified = pd.concat([data[['time', 'event']], dummy_vars], axis=1)
    dummy_cols = dummy_vars.columns.to_list()

    # Fit multivariate Cox regression model
    cph_multivariate = CoxPHFitter()
    cph_multivariate.fit(data_modified, duration_col='time', event_col='event')
    multivariate_summary = cph_multivariate.summary 

    multivariate_summary.to_csv(f'./Blogistic and cox analysis/{fname} multivariate_cox_analysis.csv')

    feats_selected_from_multivariate = multivariate_summary[multivariate_summary['p'] < 0.05].index.to_list()

    print(multivariate_summary)
    print('Features with p-value < 0.05:')
    print(feats_selected_from_multivariate)
    # print(result_multivariate.summary())
    # print(result_multivariate.pvalues) 

    blank_df = pd.DataFrame(columns=univariate_results_df.columns, index=[0])

    combined_results = pd.concat([univariate_results_df, blank_df, blank_df, multivariate_summary], ignore_index=False) 

    combined_results = combined_results[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p', 'Variable']]
    combined_results['HR (95% CI)'] = combined_results.apply(
        lambda row: f"{row['exp(coef)']:.3f} ({row['exp(coef) lower 95%']:.3f}-{row['exp(coef) upper 95%']:.3f})", axis=1 )
    combined_results = combined_results[['HR (95% CI)', 'p']]
    combined_results.to_excel(f'./Blogistic and cox analysis/{fname} uni-multi-variate_cox_analysis.xlsx')

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

def plot_KMcurves_survRate_at_times(data, fname='shanxi', duration_col = 'time', event_col = 'event', best_cutoff_value = None, 
                                    title=None, ax=None, show_km_fig=True): 
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
    # fig, ax = plt.subplots(figsize=(10, 8))
    km_curve_group1 = data_[data_['DLS'] <= best_cutoff_value]  ##'pred'
    km_curve_group2 = data_[data_['DLS'] > best_cutoff_value]  ##'pred'

    kmf_group1 = KaplanMeierFitter() 
    kmf_group2 = KaplanMeierFitter() 
    kmf_group1.fit(durations=km_curve_group1[duration_col], event_observed=km_curve_group1[event_col], label='high risk')  ##f'Group <= {best_cutoff_value}'
    kmf_group2.fit(durations=km_curve_group2[duration_col], event_observed=km_curve_group2[event_col], label='low risk')  ##f'Group > {best_cutoff_value}'

    kmf_group1.plot(ax=ax, color=color_sns_list[0], label='high risk', ci_alpha=0.1, linewidth=4, ci_show=False)  ##f'Group <= {best_cutoff_value}'
    kmf_group2.plot(ax=ax, color=color_sns_list[1], label='low risk',  ci_alpha=0.1, linewidth=4, ci_show=False)  ##f'Group > {best_cutoff_value}'
    # Get survival probabilities for censored points
    survival_prob_group1 = kmf_group1.survival_function_.loc[km_curve_group1[duration_col]].values.flatten()
    survival_prob_group2 = kmf_group2.survival_function_.loc[km_curve_group2[duration_col]].values.flatten()

    ## 添加 'number at risk' 表格
    ax.set_ylim(0.15, 1.04) 
    yticks = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ax.set_yticks(yticks)  
    ax.set_xlim(-5, 85)  # Set x-axis range to slightly more than 100
    xticks = range(0, 91, 20)
    ax.set_xticks(xticks)  # Ensure the xticks remain consistent after adding at risk counts
    add_at_risk_counts(kmf_group1, kmf_group2, ax=ax, rows_to_show=['At risk'], fontsize=16, xticks=xticks)  ##  fontsize=18
    ax.set_xticks(xticks)  # Ensure the xticks remain consistent after adding at risk counts
    if result.p_value<0.0001: 
        ax.text(0.05, 0.05, f'p<0.0001\nHR: {hr:.3f}({lower_ci:.3f}-{upper_ci:.3f})'.replace('.', '·'), transform=ax.transAxes, fontsize=15, verticalalignment='bottom')  ##, horizontalalignment='left'
    else: 
        ax.text(0.05, 0.05, f'p={result.p_value:.4f}\nHR: {hr:.3f}({lower_ci:.3f}-{upper_ci:.3f})'.replace('.', '·'), transform=ax.transAxes, fontsize=15, verticalalignment='bottom')  ##, horizontalalignment='left'
    

    ax.legend(prop={'size': 18})  ##, 'weight': 'bold'
    ax.set_xlabel('Time of months', fontsize=17)
    ax.set_ylabel('Survival probability', fontsize=17)
    ax.tick_params(which='major', labelsize=18)  ##axis='both', 
    plt.yticks([tick for tick in plt.yticks()[0]], [
            f'{tick:.1f}'.replace('.', '·') for tick in plt.yticks()[0]])
    if title:
        ax.set_title(title, fontsize=20,  ha='center') 
    plt.tight_layout()
    if show_km_fig: 
        plt.savefig(f'./Blogistic and cox analysis/KM_curves_{fname}_{duration_col}.svg', bbox_inches='tight')
        plt.savefig(f'./Blogistic and cox analysis/KM_curves_{fname}_{duration_col}.png', bbox_inches='tight', dpi=600)

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

########################### uni-multi-variate_cox_analysis #####################################
for i in range(len(datas)): 
    cox_analysis(datas[i], fname=fnames[i]) 


########################### KM curves ####################################
results = [] 
for i in range(len(datas)): 
    if i==0: 
        best_cutoff_value, results_rate = plot_KMcurves_survRate_at_times(datas[i], fname=fnames[i], best_cutoff_value=None) 
    else: 
        best_cutoff_value, results_rate = plot_KMcurves_survRate_at_times(datas[i], fname=fnames[i], best_cutoff_value=best_cutoff_value) 
    results.append(results_rate ) 
# results = [x for xs in results for x in xs] 
df = pd.DataFrame(results, index=['train_surv' , 'interValid_surv', 'evc2_surv'] ) 
df = df.transpose() 
df.to_excel(f'./Blogistic and cox analysis/KM_curves_OS rate table.xlsx', encoding='utf-8-sig') 
print() 

###########################

########################### KM curves make three fig in one ####################################
fig, axs = plt.subplots(1, 3, figsize=(20, 7))
axs = axs.flatten() 
i_fig = 0 
for i in range(len(datas)): 
    ax = axs[i]
    if i==0: 
        best_cutoff_value, results_rate = plot_KMcurves_survRate_at_times(datas[i], fname=fnames[i], best_cutoff_value=None, title=f'{fnames[i]}', ax=ax, show_km_fig=False) 
    else: 
        best_cutoff_value, results_rate = plot_KMcurves_survRate_at_times(datas[i], fname=fnames[i], best_cutoff_value=best_cutoff_value, title=f'{fnames[i]}', ax=ax, show_km_fig=False) 

plt.tight_layout(rect=[0, 0, 1, 0.95])  # 为整体标题留出空间
# plt.suptitle('Kaplan-Meier Survival Curves', fontsize=20, y=0.99)
plt.savefig('./Blogistic and cox analysis/all_KM_curves.svg', bbox_inches='tight') 


