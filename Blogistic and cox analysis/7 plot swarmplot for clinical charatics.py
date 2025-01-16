import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import matplotlib
matplotlib.rcParams.update({'font.size': 15})


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
    clini_pred_training['Score'] = np.log( clini_pred_training['pred_N'] / ( 1 - clini_pred_training['pred_N'] ) )  ## change proba to score 
    clini_pred_training['Age'] = np.where(clini_pred_training['Age']>=60, '≥60', '<60')  
    clini_pred_training['BMI'] = np.where(clini_pred_training['BMI']>24, '>24', '≤24')
    clini_pred_training['Target'] = np.where(clini_pred_training['labels_N']>0, 'N+', 'N0') 
    conditions = [ clini_pred_training['Borrmann']==1, clini_pred_training['Borrmann']==2, clini_pred_training['Borrmann']==3, clini_pred_training['Borrmann']==4 ]
    choices     = [ "type I", "type II", "type III", "type IV", ] 
    clini_pred_training["Borrmann"] = np.select(conditions, choices, )  ## default=np.nan
    conditions = [ clini_pred_training['Lauren']==1, clini_pred_training['Lauren']==2, clini_pred_training['Lauren']==3 ]
    choices     = [ "intestinal", "diffuse", "mixed" ] 
    clini_pred_training["Lauren"] = np.select(conditions, choices, )  ## default=np.nan
    conditions = [ clini_pred_training['Differentiation']==1, clini_pred_training['Differentiation']==2, clini_pred_training['Differentiation']==3 ]
    choices     = [ "well", "moderate", "poor" ] 
    clini_pred_training["Differentiation"] = np.select(conditions, choices, )  ## default=np.nan
    conditions = [ clini_pred_training['Locations']==0, clini_pred_training['Locations']==1, clini_pred_training['Locations']==2, clini_pred_training['Locations']==3 ]
    choices     = [ "cardia", "body", "antrum", "multiple or whole" ] 
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
    choices     = [ "cN0", "cN1", "cN2", "cN3",  ] 
    clini_pred_training["cN stage"] = np.select(conditions, choices, )  ## default=np.nan
    conditions = [ clini_pred_training['T stage']==1, clini_pred_training['T stage']==2, 
                clini_pred_training['T stage']==3, clini_pred_training['T stage']==4, 
                ]
    choices     = [ "cT1", "cT2", "cT3", "cT4",  ] 
    clini_pred_training["cT stage"] = np.select(conditions, choices, )  ## default=np.nan

    order_dict = {} 
    order_dict['Age'] = [ '≥60', '<60', ] 
    order_dict['BMI'] = [ '>24', '≤24', ] 
    order_dict['Borrmann'] = [ "type I", "type II", "type III", "type IV", ] 
    order_dict['Lauren'] = [ "intestinal", "diffuse", "mixed" ] 
    order_dict['Differentiation'] = [ "well", "moderate", "poor" ] 
    order_dict['Location'] = [ "cardia", "body", "antrum", "multiple or whole" ]
    order_dict['CEA'] = [ "normal", "elevated" ] 
    order_dict['CA199'] = [ "normal", "elevated" ] 
    order_dict['Regimens'] = [ "doublet-drug", "triplet-drug" ] 
    order_dict['Cycles'] = [ "≤3", ">3" ] 
    order_dict['Sex'] = [ "female", "male" ] 
    order_dict['cN stage'] = [ "cN0", "cN1", "cN2", "cN3",  ] 
    order_dict['cT stage'] = [ "cT2", "cT3", "cT4",  ] ##"cT1", 
    return clini_pred_training, order_dict 

# 扩展的调色板
nature_palette = ['#003c67', '#007f7f', '#55a630', '#9e2a2b', '#2a9d8f', '#264653', '#e76f51', '#f4a261', '#e9c46a', '#a8dadc' ]
lancet_palette = ['#004c97', '#4c4c4c', '#e60000', '#ffcc00', '#00509e', '#ff6f61', '#ffa500', '#d4d4d4', '#8a2be2', '#e9967a' ]
jama_palette = ['#0077b6', '#023e8a', '#00b4d8', '#90e0ef', '#03045e', '#caf0f8', '#48cae4', '#ade8f4', '#ffb703', '#fb8500' ]

def plot_swarmplot(clini_pred_training, x="Target", y="Score", hue="Age", palette=None, order_dict=None, ax=None): 
    palette = jama_palette 
    box = sns.boxplot(x="Target", y="Score", hue=hue, hue_order=order_dict[hue], data=clini_pred_training,
                        dodge=True, fliersize=0, linewidth=2, palette="pastel", showfliers=False)
    handles, labels = box.get_legend_handles_labels()
    if box.get_legend() is not None:
        box.get_legend().remove()
    for patch in box.artists:
        patch.set_facecolor('none')
    swarm = sns.swarmplot(x="Target", y="Score", hue=hue, hue_order=order_dict[hue], 
                            data=clini_pred_training, palette=palette, dodge=True, size=3.5) 
    handles, labels = swarm.get_legend_handles_labels()
    plt.legend(handles[len(order_dict[hue]):], labels[len(order_dict[hue]):], title=hue, loc='upper left', fontsize=12)
    plt.savefig(f'./Blogistic and cox analysis/violin_{hue}.svg', dpi = 300, bbox_inches='tight') 
    plt.close() 

datas = [X_train, X_test, df_features_evc1_1, df_features_evc2_1] 
fnames = ['train', 'interValid', 'evc1', 'evc2' ] 

########################### Sub-group analysis LNM prediction ####################################
data_all = pd.concat(datas, axis=0) 
df_dl_preds_, order_dict = prepare_data(data_all) 
clini_pred_training = df_dl_preds_
keys = list( order_dict.keys() ) 
results, index = [], [] 
new_feats=keys
for feat in new_feats: 
    plot_swarmplot(clini_pred_training, hue=feat, order_dict=order_dict) ##, palette=["blue", "green"]
print()
