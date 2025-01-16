import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored

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

def rm_nan(df , col='patient_image2'): 
    ## remove the row if its col is None
    return df[df[col].notna()] 

df_features_shanxi_1 = rm_nan(df_features_shanxi_1, col='patient_image1' ) 
df_features_shanxi_2 = rm_nan(df_features_shanxi_2, col='patient_image1' ) 
df_features_evc1 = rm_nan(df_features_evc1, col='patient_image1' ) 
df_features_evc2 = rm_nan(df_features_evc2, col='patient_image1' ) 
print('number of cases for pre NAC: ', df_features_shanxi_1.shape, df_features_shanxi_2.shape, df_features_evc1.shape, df_features_evc2.shape)

df_features_shanxi_1 = rm_nan(df_features_shanxi_1, col='patient_image2' ) 
df_features_shanxi_2 = rm_nan(df_features_shanxi_2, col='patient_image2' ) 
df_features_evc1 = rm_nan(df_features_evc1, col='patient_image2' ) 
# # df_features_evc1 = rm_nan(df_features_evc1, col='OS时间（月份）' ) 
# df_features_evc1['OS时间（月份）'] = df_features_evc1['OS时间（月份）'][3]
# df_features_evc1['是否死亡'] = df_features_evc1['OS时间（月份）'][3] 
df_features_evc2 = rm_nan(df_features_evc2, col='patient_image2' ) 
df_features_evc2 = rm_nan(df_features_evc2, col='是否死亡' ) 

print('number of cases for pre NAC and post NAC: ', df_features_shanxi_1.shape, df_features_shanxi_2.shape, df_features_evc1.shape, df_features_evc2.shape)
df_features_shanxi_1['target_'] = df_features_shanxi_1['淋巴结转移（0为N0,1为N+）']
df_features_shanxi_2['target_'] = df_features_shanxi_2['淋巴结转移（0为N0,1为N+）']
df_features_evc1['target_'] = df_features_evc1['淋巴结转移（0为N0,1为N+）']
df_features_evc2['target_'] = df_features_evc2['淋巴结转移（0为N0,1为N+）']

X_train = df_features_shanxi_1 
X_test = df_features_shanxi_2 
df_features_evc1_1 = df_features_evc1
df_features_evc2_1 = df_features_evc2

# X_test[X_test['Lauren']==0] = 1 ###这个是错误的
X_test.loc[X_test['Lauren'] == 0, 'Lauren'] = 1

column_sel_e_0 = ['patient_image1', 'patient_mask1', 'patient_image2', 'patient_mask2', ]
column_sel_e_01 = ['新编号', '住院号_x', '姓名', 
            'patient_image1', 'patient_mask1', 'patient_image2', 'patient_mask2', 
            'patient_shape1', 'patient_spacing1', 'patient_shape_tumor1', 'patient_volumns1',
            'patient_shape2', 'patient_spacing2', 'patient_shape_tumor2', 'patient_volumns2', ] 
column_sel_e_02 = ['影像号', '姓名', 
            'patient_image1', 'patient_mask1', 'patient_image2', 'patient_mask2', 
            'patient_shape1', 'patient_spacing1', 'patient_shape_tumor1', 'patient_volumns1',
            'patient_shape2', 'patient_spacing2', 'patient_shape_tumor2', 'patient_volumns2', ] 
# column_sel = ['年龄', 'BMI', '性别（1男0女）', '影像cT分期', '影像cN分期', 
#               '病灶位置（0=贲门胃底，1=胃体，2=胃窦幽门，3=多个部位或全胃）', 'NAC分化程度（1=高分化，2=中分化，3=低分化或未分化）', 
#               'Borrman分型（1=结节隆起型，2=局部溃疡性，3=浸润溃疡型，4=弥漫浸润型）', 'Lauren分型（1=肠型，2=弥漫型，3=混合型）', 
#               'NAC前CEA小于5ng/ml=0，大于5ng/ml=1', 'NAC前CA199 小于27U/ml=0，大于 27U/ml=1', 
#               '新辅助化疗方案类型（2：双药联合化疗；3:3药联合化疗及以上）', '术前化疗疗程（≤3为0，＞3为1）'
#               ] 
column_sel_e = ['Age', 'BMI', 'Sex', 
                'T stage', 'N stage',
                'Locations', 'Differentiation', 
                'Borrmann', 'Lauren', 
                'Pre-NACT CEA', 'Pre-NACT CA199', 
                'Regimens', 'Course', 
                'pCR', 'ypT', 'ypN', 'ypTNM', 'numofLN', 'treatmentpostNAC'
                ] 
# rename_dict = dict(map(lambda i,j : (i,j) , column_sel, column_sel_e))
# X_train = X_train.rename(columns=rename_dict ) 
# X_test = X_test.rename(columns=rename_dict ) 
X_train  = X_train[column_sel_e_01 + column_sel_e + ['labels_time', 'labels_status'] + ['pred_surv_risk'] + ['labels_N', 'pred_N',]]
X_test  = X_test[column_sel_e_01 + column_sel_e + ['labels_time', 'labels_status'] + ['pred_surv_risk'] + ['labels_N', 'pred_N',]] 
# df_features_evc1_1 = df_features_evc1_1.rename(columns=rename_dict ) 
# df_features_evc2_1 = df_features_evc2_1.rename(columns=rename_dict ) 
# df_features_evc1_1  = df_features_evc1_1[column_sel_e_0 + column_sel_e + ['labels_time', 'labels_status'] + ['pred_surv_risk'] + ['labels_N', 'pred_N',]]
df_features_evc2_1  = df_features_evc2_1[column_sel_e_02 + column_sel_e + ['labels_time', 'labels_status'] + ['pred_surv_risk'] + ['labels_N', 'pred_N',]]

X_train = X_train.rename(columns= {'pred_surv_risk': 'DLS', 'labels_time': 'time', 'labels_status': 'event'} ) 
X_test = X_test.rename(columns= {'pred_surv_risk': 'DLS', 'labels_time': 'time', 'labels_status': 'event'} ) 
# df_features_evc1_1 = df_features_evc1_1.rename(columns= {'pred_surv_risk': 'DLS', 'labels_time': 'time', 'labels_status': 'event'} ) 
df_features_evc2_1 = df_features_evc2_1.rename(columns= {'pred_surv_risk': 'DLS', 'labels_time': 'time', 'labels_status': 'event'} ) 

###-----------------imputer the miss data using sklearn.impute 
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer 
print(X_train[column_sel_e].isnull().sum() )
imputer = IterativeImputer(max_iter=10, random_state=0, initial_strategy='most_frequent',)
# imputer = SimpleImputer(strategy='most_frequent')
imputer.fit(X_train[column_sel_e]) 
print(X_train[column_sel_e].isnull().sum() )
X_train[column_sel_e] = imputer.fit_transform(X_train[column_sel_e])
X_test[column_sel_e] = imputer.fit_transform(X_test[column_sel_e])
# df_features_evc1_1[column_sel_e] = imputer.fit_transform(df_features_evc1_1[column_sel_e])
df_features_evc2_1[column_sel_e] = imputer.fit_transform(df_features_evc2_1[column_sel_e])
def feats_stardand(df, featsName=None): 
    df[featsName] = ( df[featsName] + 0.5 ).astype(int) 
    return df 
X_train = feats_stardand(X_train, column_sel_e[:1]+column_sel_e[2:])
X_test = feats_stardand(X_test, column_sel_e[:1]+column_sel_e[2:])
# df_features_evc1_1 = feats_stardand(df_features_evc1_1, column_sel_e[:1]+column_sel_e[2:])
df_features_evc2_1 = feats_stardand(df_features_evc2_1, column_sel_e[:1]+column_sel_e[2:])

def data_prepare(X_train): 
    X_train_uni_multi = X_train[[ 'DLS', 'ypTNM', 'Locations', 'ypN' ] ] 
    y_train_uni_multi = X_train[['event', 'time' ]] 
    # Convert target to structured array
    y_train_uni_multi = np.array([(e, t) for e, t in zip(y_train_uni_multi['event'], y_train_uni_multi['time'])],
                          dtype=[('event', bool), ('time', float)])
    return X_train_uni_multi, y_train_uni_multi



X_train_uni_multi, y_train_uni_multi = data_prepare(X_train) 
X_test_uni_multi, y_test_uni_multi = data_prepare(X_test) 
X_evc2_uni_multi, y_evc2_uni_multi = data_prepare(df_features_evc2_1) 

# Train Cox Proportional Hazards model 
coxph = CoxPHSurvivalAnalysis() 
coxph.fit(X_train_uni_multi, y_train_uni_multi) 

# Evaluate the model using Concordance Index
c_index_train = concordance_index_censored(y_train_uni_multi['event'], y_train_uni_multi['time'], coxph.predict(X_train_uni_multi))
c_index_test = concordance_index_censored(y_test_uni_multi['event'], y_test_uni_multi['time'], coxph.predict(X_test_uni_multi))
c_index_evc2 = concordance_index_censored(y_evc2_uni_multi['event'], y_evc2_uni_multi['time'], coxph.predict(X_evc2_uni_multi))

print(f'Training Concordance Index: {c_index_train[0]:.4f}') 
print(f'Test Concordance Index: {c_index_test[0]:.4f}') 
print(f'Test Concordance Index: {c_index_evc2[0]:.4f}') 

def get_dummies(data, feats_from_Uni_Multi):
    for col in feats_from_Uni_Multi: 
        data[col] = data[col].astype('category')
    dummy_vars = pd.get_dummies(data[feats_from_Uni_Multi], drop_first=True, prefix=feats_from_Uni_Multi)
    dummy_vars = pd.concat([data[['DLS']], dummy_vars], axis=1)
    data_modified = pd.concat([data[['time', 'event']], dummy_vars], axis=1)
    return data_modified

train_uni_multi_ori = X_train[[ 'DLS', 'ypTNM', 'Locations', 'ypN', 'event', 'time' ] ] 
test_uni_multi_ori = X_test[[ 'DLS', 'ypTNM', 'Locations', 'ypN', 'event', 'time' ] ] 
evc2_uni_multi_ori = df_features_evc2_1[[ 'DLS', 'ypTNM', 'Locations', 'ypN', 'event', 'time' ] ] 

# train_uni_multi       = get_dummies(train_uni_multi_ori, [ 'ypTNM', 'Locations', 'ypN', ])
# test_uni_multi        = get_dummies(test_uni_multi_ori, [ 'ypTNM', 'Locations', 'ypN', ])
# evc2_uni_multi   = get_dummies(evc2_uni_multi_ori, [ 'ypTNM', 'Locations', 'ypN', ])

train_uni_multi = train_uni_multi_ori
test_uni_multi = test_uni_multi_ori
evc2_uni_multi = evc2_uni_multi_ori



from sklearn.utils import resample
from lifelines.utils import concordance_index
# Function to calculate the bootstrap confidence interval
def bootstrap_concordance_index(model, df, duration_col, event_col, n_bootstrap=1000, random_seed=None):
    ci_values = []
    for _ in range(n_bootstrap):
        bootstrap_sample = resample(df)
        bootstrap_sample = bootstrap_sample.reset_index(drop=True)
        ci = concordance_index(
            bootstrap_sample[duration_col],
            -model.predict_partial_hazard(bootstrap_sample),
            bootstrap_sample[event_col]
        )
        ci_values.append(ci)
    ci_lower = np.percentile(ci_values, 2.5)
    ci_upper = np.percentile(ci_values, 97.5)
    return np.mean(ci_values), ci_lower, ci_upper

def get_hazard_risk(cph, train_uni_multi, train_uni_multi_ori, fname=None): 

    # Predict partial hazard (risk scores)
    train_predictions = cph.predict_partial_hazard(train_uni_multi)
    # Output the predictions
    train_uni_multi_ori['predicted_risk'] = train_predictions.values
    train_uni_multi['predicted_risk'] = train_predictions.values
    # Predict survival function
    train_survival_function = cph.predict_survival_function(train_uni_multi)
    train_survival_function = cph.predict_survival_function(train_uni_multi)
    # Save predictions to CSV files
    if fname: 
        train_uni_multi_ori.to_excel(fname, index=False) 
    return train_uni_multi_ori





######################### clincial 'ypTNM' OS model #########################
## Train Cox Proportional Hazards model using lifelines
## range(0,10000) ##[1400]: [143, 160, 321, 351]
train_list = [] 
train_CI_lower_list = [] 
train_CI_upper_list = [] 
test_list = [] 
test_CI_lower_list = [] 
test_CI_upper_list = [] 
evc2_list = [] 
evc2_CI_lower_list = [] 
evc2_CI_upper_list = [] 
time_points = [12, 36, 60]
time_points = [12, 24, 36, 48, 60]

cph = CoxPHFitter()
cph.fit(train_uni_multi.drop(labels=['ypTNM', 'Locations', 'ypN'], axis=1), duration_col='time', event_col='event', )  ##formula = 'DLS + ypTNM + Locations + ypN'
for time_point in time_points: 
    train_uni_multi_filtered = train_uni_multi.copy() 
    test_uni_multi_filtered = test_uni_multi.copy() 
    evc2_uni_multi_filtered = evc2_uni_multi.copy() 

    train_uni_multi_filtered.loc[train_uni_multi_filtered['time'] > time_point, 'event'] = 0 
    test_uni_multi_filtered.loc[test_uni_multi_filtered['time'] > time_point, 'event'] = 0 
    evc2_uni_multi_filtered.loc[evc2_uni_multi_filtered['time'] > time_point, 'event'] = 0 

    # Concordance Index with bootstrap 95% CI
    ci_train, lower_ci_train, upper_ci_train = bootstrap_concordance_index(cph, train_uni_multi_filtered.drop(labels=['ypTNM', 'Locations', 'ypN'], axis=1), 
                                                                        'time', 'event', n_bootstrap=100)
    ci_test, lower_ci_test, upper_ci_test = bootstrap_concordance_index(cph, test_uni_multi_filtered.drop(labels=['ypTNM', 'Locations', 'ypN'], axis=1), 
                                                                        'time', 'event', n_bootstrap=100)
    ci_evc2, lower_ci_evc2, upper_ci_evc2 = bootstrap_concordance_index(cph, evc2_uni_multi_filtered.drop(labels=['ypTNM', 'Locations', 'ypN'], axis=1), 
                                                                                    'time', 'event', n_bootstrap=100)

    print(f'clinical model: -{i}-: time_point: -{time_point}-, Training Concordance Index: \
            {ci_train:.4f} (95% CI: {lower_ci_train:.4f}, {upper_ci_train:.4f})')
    print(f'clinical model: -{i}-: time_point: -{time_point}-, Test Concordance Index: \
            {ci_test:.4f} (95% CI: {lower_ci_test:.4f}, {upper_ci_test:.4f})')
    print(f'clinical model: -{i}-: time_point: -{time_point}-, evc2 Concordance Index: \
            {ci_evc2:.4f} (95% CI: {lower_ci_evc2:.4f}, {upper_ci_evc2:.4f})')

    train_list.append(ci_train) 
    train_CI_lower_list.append(lower_ci_train) 
    train_CI_upper_list.append(upper_ci_train)
    test_list.append(ci_test)
    test_CI_lower_list.append(lower_ci_test)
    test_CI_upper_list.append(upper_ci_test)
    evc2_list.append(ci_evc2)
    evc2_CI_lower_list.append(lower_ci_evc2)
    evc2_CI_upper_list.append(upper_ci_evc2)
print()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
import seaborn as sns

# plt.plot(sorted(df['time'].unique()), c_indices, marker='o')
# plt.xlabel('Time (years)')
# plt.ylabel('Concordance index')
# plt.title('Concordance index over time')
# plt.show() 

data = {
    "Time (months)": time_points,
    "train": train_list,
    "train_CI_lower": train_CI_lower_list,
    "train_CI_upper": train_CI_upper_list,
    "test": test_list,
    "test_CI_lower": test_CI_lower_list,
    "test_CI_upper": test_CI_upper_list,
    "evc2": evc2_list, 
    "evc2_CI_lower": evc2_CI_lower_list,
    "evc2_CI_upper": evc2_CI_upper_list, 
}

df = pd.DataFrame(data)

plt.figure(figsize=(10, 6))
# Plotting the external data with confidence intervals
g = sns.lineplot(x="Time (months)", y="train", data=df, label="train (95% CI)", color="brown", marker="o", markersize=10)
plt.fill_between(df["Time (months)"], df["train_CI_lower"], df["train_CI_upper"], alpha=0.2, color="brown") 
# g.set(xticks=([12, 36, 60]))
g.set(xticks=(time_points))
plt.xlabel("Time (months)")
plt.ylabel("Concordance index")
plt.ylim(0.2, 0.9)
# Set the x and y ticks to the specified ranges with replaced decimal points
plt.yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], [f'{tick:.1f}'.replace('.', '·') for tick in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]])
plt.legend()
plt.savefig(f'./Blogistic and cox analysis/time_cindex_train.svg', dpi = 300, bbox_inches='tight') 
plt.close()
# plt.show()


plt.figure(figsize=(10, 6)) 
# Plotting the internal data with confidence intervals
g = sns.lineplot(x="Time (months)", y="test", data=df, label="test (95% CI)", color="brown", marker="o", markersize=10)  ## "purple"
plt.fill_between(df["Time (months)"], df["test_CI_lower"], df["test_CI_upper"], alpha=0.2, color="brown")  
# g.set(xticks=([12, 36, 60]))
g.set(xticks=(time_points))
plt.xlabel("Time (months)")
plt.ylabel("Concordance index")
plt.ylim(0.2, 0.9)
# Set the x and y ticks to the specified ranges with replaced decimal points
plt.yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], [f'{tick:.1f}'.replace('.', '·') for tick in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]])
plt.legend()
plt.savefig(f'./Blogistic and cox analysis/time_cindex_test.svg', dpi = 300, bbox_inches='tight')
# plt.show()

plt.figure(figsize=(10, 6))
# Plotting the internal data with confidence intervals
g = sns.lineplot(x="Time (months)", y="evc2", data=df, label="evc2 (95% CI)", color="brown", marker="o", markersize=10)  ## "green"
plt.fill_between(df["Time (months)"], df["evc2_CI_lower"], df["evc2_CI_upper"], alpha=0.2, color="brown")
# g.set(xticks=([12, 36, 60]))
g.set(xticks=(time_points))
plt.xlabel("Time (months)")
plt.ylabel("Concordance index")
plt.ylim(0.2, 0.9)
# Set the x and y ticks to the specified ranges with replaced decimal points
plt.yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], [f'{tick:.1f}'.replace('.', '·') for tick in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]])
plt.legend()
plt.savefig(f'./Blogistic and cox analysis/time_cindex_evc2.svg', dpi = 300, bbox_inches='tight')
# plt.show()

# # Add horizontal line at y=0.78
# plt.axhline(y=0.78, color='gray', linestyle='--')
# plt.text(0.5, 0.78, 'Better', horizontalalignment='right', verticalalignment='bottom', color='gray')



print() 