import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
from itertools import cycle
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

import matplotlib
matplotlib.rcParams.update({'font.size': 15})

data_path1 = './BEvaluation/5preds_all_models_2ct.xlsx'
df_preds_all_models = pd.read_excel(data_path1) 
y_true = df_preds_all_models['target'].to_numpy()
y_proba_model1 = df_preds_all_models['DlPreds'].to_numpy()
y_proba_model2 = df_preds_all_models['ClinicalPreds'].to_numpy()
y_proba_model3 = df_preds_all_models['DlClinicalPreds'].to_numpy()

# Calculate ROC curve and AUC for each classifier
fpr_model1_d1, tpr_model1_d1, thresholds_model1_d1 = roc_curve(y_true[:278],                        y_proba_model1[:278] )
roc_auc_model1_d1 = auc(fpr_model1_d1, tpr_model1_d1) 
fpr_model1_d2, tpr_model1_d2, thresholds_model1_d2 = roc_curve(y_true[278:278+120],                 y_proba_model1[278:278+120] )
roc_auc_model1_d2 = auc(fpr_model1_d2, tpr_model1_d2) 
fpr_model1_d3, tpr_model1_d3, thresholds_model1_d3 = roc_curve(y_true[278+120:278+120+335],         y_proba_model1[278+120:278+120+335] )
roc_auc_model1_d3 = auc(fpr_model1_d3, tpr_model1_d3) 
fpr_model1_d4, tpr_model1_d4, thresholds_model1_d4 = roc_curve(y_true[278+120+335:278+120+335+288], y_proba_model1[278+120+335:278+120+335+288] )
roc_auc_model1_d4 = auc(fpr_model1_d4, tpr_model1_d4) 

fpr_model2_d1, tpr_model2_d1, thresholds_model2_d1 = roc_curve(y_true[:278],                        y_proba_model2[:278] )
roc_auc_model2_d1 = auc(fpr_model2_d1, tpr_model2_d1) 
fpr_model2_d2, tpr_model2_d2, thresholds_model2_d2 = roc_curve(y_true[278:278+120],                 y_proba_model2[278:278+120] )
roc_auc_model2_d2 = auc(fpr_model2_d2, tpr_model2_d2) 
fpr_model2_d3, tpr_model2_d3, thresholds_model2_d3 = roc_curve(y_true[278+120:278+120+335],         y_proba_model2[278+120:278+120+335] )
roc_auc_model2_d3 = auc(fpr_model2_d3, tpr_model2_d3) 
fpr_model2_d4, tpr_model2_d4, thresholds_model2_d4 = roc_curve(y_true[278+120+335:278+120+335+288], y_proba_model2[278+120+335:278+120+335+288] )
roc_auc_model2_d4 = auc(fpr_model2_d4, tpr_model2_d4) 

fpr_model3_d1, tpr_model3_d1, thresholds_model3_d1 = roc_curve(y_true[:278],                        y_proba_model3[:278] )
roc_auc_model3_d1 = auc(fpr_model3_d1, tpr_model3_d1) 
fpr_model3_d2, tpr_model3_d2, thresholds_model3_d2 = roc_curve(y_true[278:278+120],                 y_proba_model3[278:278+120] )
roc_auc_model3_d2 = auc(fpr_model3_d2, tpr_model3_d2) 
fpr_model3_d3, tpr_model3_d3, thresholds_model3_d3 = roc_curve(y_true[278+120:278+120+335],         y_proba_model3[278+120:278+120+335] )
roc_auc_model3_d3 = auc(fpr_model3_d3, tpr_model3_d3) 
fpr_model3_d4, tpr_model3_d4, thresholds_model3_d4 = roc_curve(y_true[278+120+335:278+120+335+288], y_proba_model3[278+120+335:278+120+335+288] )
roc_auc_model3_d4 = auc(fpr_model3_d4, tpr_model3_d4) 

net_benefit_model1_d1 = tpr_model1_d1 - fpr_model1_d1
net_benefit_model1_d2 = tpr_model1_d2 - fpr_model1_d2
net_benefit_model1_d3 = tpr_model1_d3 - fpr_model1_d3
net_benefit_model1_d4 = tpr_model1_d4 - fpr_model1_d4

net_benefit_model2_d1 = tpr_model2_d1 - fpr_model2_d1
net_benefit_model2_d2 = tpr_model2_d2 - fpr_model2_d2
net_benefit_model2_d3 = tpr_model2_d3 - fpr_model2_d3
net_benefit_model2_d4 = tpr_model2_d4 - fpr_model2_d4

net_benefit_model3_d1 = tpr_model3_d1 - fpr_model3_d1
net_benefit_model3_d2 = tpr_model3_d2 - fpr_model3_d2
net_benefit_model3_d3 = tpr_model3_d3 - fpr_model3_d3
net_benefit_model3_d4 = tpr_model3_d4 - fpr_model3_d4


################################delong test########################
import sys
sys.path.append(r"/home/bj/Documents/code_workspace/70 Mamba/SegMamba-main")
from delong_test import DelongTest
p_delong_model1_model2_d1 = DelongTest(y_proba_model1[:278] ,                          y_proba_model2[:278],                       y_true[:278])
p_delong_model1_model2_d2 = DelongTest(y_proba_model1[278:278+120] ,                   y_proba_model2[278:278+120],                y_true[278:278+120] )
p_delong_model1_model2_d3 = DelongTest(y_proba_model1[278+120:278+120+335] ,           y_proba_model2[278+120:278+120+335],        y_true[278+120:278+120+335] )
p_delong_model1_model2_d4 = DelongTest(y_proba_model1[278+120+335:278+120+335+288] ,   y_proba_model2[278+120+335:278+120+335+288], y_true[278+120+335:278+120+335+288] )
# print(p_delong_model1_model2_d1, p_delong_model1_model2_d2, p_delong_model1_model2_d3, p_delong_model1_model2_d4 )

p_delong_model1_model3_d1 = DelongTest(y_proba_model1[:278] ,                          y_proba_model3[:278],                       y_true[:278])
p_delong_model1_model3_d2 = DelongTest(y_proba_model1[278:278+120] ,                   y_proba_model3[278:278+120],                y_true[278:278+120] )
p_delong_model1_model3_d3 = DelongTest(y_proba_model1[278+120:278+120+335] ,           y_proba_model3[278+120:278+120+335],        y_true[278+120:278+120+335] )
p_delong_model1_model3_d4 = DelongTest(y_proba_model1[278+120+335:278+120+335+288] ,   y_proba_model3[278+120+335:278+120+335+288], y_true[278+120+335:278+120+335+288] )
# print(p_delong_model1_model3_d1, p_delong_model1_model3_d2, p_delong_model1_model3_d3, p_delong_model1_model3_d4 )

d = {'p_delong_DL_Clinical': [  p_delong_model1_model2_d1._compute_z_p()[1], p_delong_model1_model2_d2._compute_z_p()[1], 
                                p_delong_model1_model2_d3._compute_z_p()[1], p_delong_model1_model2_d4._compute_z_p()[1]], 
    'p_delong_DL_DLClinical': [ p_delong_model1_model3_d1._compute_z_p()[1], p_delong_model1_model3_d2._compute_z_p()[1], 
                                p_delong_model1_model3_d3._compute_z_p()[1], p_delong_model1_model3_d4._compute_z_p()[1]]}
df_delong = pd.DataFrame(data=d) 
df_delong.to_csv('p_delong.csv', encoding='utf-8-sig') 


from sklearn.metrics import confusion_matrix
def calculate_net_benefit_model(thresh_group, y_pred_score, y_label):
    net_benefit_model = np.array([])
    for thresh in thresh_group:
        y_pred_label = y_pred_score > thresh
        tn, fp, fn, tp = confusion_matrix(y_label, y_pred_label).ravel()
        n = len(y_label)
        net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
        net_benefit_model = np.append(net_benefit_model, net_benefit)
    return net_benefit_model

def calculate_net_benefit_all(thresh_group, y_label):
    net_benefit_all = np.array([])
    tn, fp, fn, tp = confusion_matrix(y_label, y_label).ravel()
    total = tp + tn
    for thresh in thresh_group:
        net_benefit = (tp / total) - (tn / total) * (thresh / (1 - thresh))
        net_benefit_all = np.append(net_benefit_all, net_benefit)
    return net_benefit_all

def plot_DCA(ax, thresh_group, net_benefit_model, net_benefit_all, labelname='Model'):
    #Plot
    ax.plot(thresh_group, net_benefit_model, label = labelname) ##, color = 'crimson'
    # ax.plot(thresh_group, net_benefit_all, color = 'black',label = 'Treat all')
    # ax.plot((0, 1), (0, 0), color = 'black', linestyle = ':', label = 'Treat none')

    #Fill，显示出模型较于treat all和treat none好的部分
    y2 = np.maximum(net_benefit_all, 0)
    y1 = np.maximum(net_benefit_model, y2)
    # ax.fill_between(thresh_group, y1, y2, alpha = 0.2)  ##, color = 'crimson'

    #Figure Configuration， 美化一下细节
    ax.set_xlim(0,1)
    ax.set_ylim(net_benefit_model.min() - 0.15, net_benefit_model.max() + 0.15)#adjustify the y axis limitation
    ax.set_xlabel(
        xlabel = 'Threshold Probability', 
        # fontdict= {'family': 'Times New Roman', 'fontsize': 15}
        )
    ax.set_ylabel(
        ylabel = 'Net Benefit', 
        # fontdict= {'family': 'Times New Roman', 'fontsize': 15}
        )
    ax.grid('major')
    ax.spines['right'].set_color((0.8, 0.8, 0.8))
    ax.spines['top'].set_color((0.8, 0.8, 0.8))
    ax.legend(loc = 'upper right')

    return ax

################################## Plot DCA curves#################################
############################# DL, Clinical, DL+Clinical#######################
#构造一个分类效果不是很好的模型
y_proba_model1_d1 = y_proba_model1[:278] 
y_proba_model2_d1 = y_proba_model2[:278] 
y_proba_model3_d1 = y_proba_model3[:278] 
y_label_d1        = y_true[:278] 
thresh_group = np.arange(0,1,0.01)
net_benefit_model1 = calculate_net_benefit_model(thresh_group, y_proba_model1_d1, y_label_d1)
net_benefit_all = calculate_net_benefit_all(thresh_group, y_label_d1)
fig, ax = plt.subplots()
ax = plot_DCA(ax, thresh_group, net_benefit_model1, net_benefit_all, labelname='CTSMamba')
net_benefit_model2 = calculate_net_benefit_model(thresh_group, y_proba_model2_d1, y_label_d1)
ax = plot_DCA(ax, thresh_group, net_benefit_model2, net_benefit_all, labelname='Clinical')
net_benefit_model3 = calculate_net_benefit_model(thresh_group, y_proba_model3_d1, y_label_d1)
ax = plot_DCA(ax, thresh_group, net_benefit_model3, net_benefit_all, labelname='Nomogram')
ax.plot(thresh_group, net_benefit_all, color = 'black', label = 'Treat all')
ax.plot((0, 1), (0, 0), color = 'black', linestyle = ':', label = 'Treat none')
fig.savefig('DCA_2ct.svg', dpi = 300, bbox_inches='tight')
# plt.show()

################################## Plot DCA curves#################################
################################# DL, Clinical (train)#######################
#构造一个分类效果不是很好的模型
y_proba_model1_d1 = y_proba_model1[:278] 
y_proba_model2_d1 = y_proba_model2[:278] 
y_proba_model3_d1 = y_proba_model3[:278] 
y_label_d1        = y_true[:278] 
thresh_group = np.arange(0,1,0.01)
net_benefit_model1 = calculate_net_benefit_model(thresh_group, y_proba_model1_d1, y_label_d1)
net_benefit_all = calculate_net_benefit_all(thresh_group, y_label_d1)
fig, ax = plt.subplots()
ax = plot_DCA(ax, thresh_group, net_benefit_model1, net_benefit_all, labelname='CTSMamba')
net_benefit_model2 = calculate_net_benefit_model(thresh_group, y_proba_model2_d1, y_label_d1)
ax = plot_DCA(ax, thresh_group, net_benefit_model2, net_benefit_all, labelname='Clinical')
# net_benefit_model3 = calculate_net_benefit_model(thresh_group, y_proba_model3_d1, y_label_d1)
# ax = plot_DCA(ax, thresh_group, net_benefit_model3, net_benefit_all, labelname='DL+Clinical')
ax.plot(thresh_group, net_benefit_all, color = 'black', label = 'Treat all')
ax.plot((0, 1), (0, 0), color = 'black', linestyle = ':', label = 'Treat none')
###lancet journal format #### need to replace . to ·   把小数点.改成·  ##lancet digital medicine format 
plt.xticks([tick for tick in plt.xticks()[0]], [
        f'{tick:.1f}'.replace('.', '·') for tick in plt.xticks()[0]])
plt.yticks([tick for tick in plt.yticks()[0]], [
        f'{tick:.1f}'.replace('.', '·') for tick in plt.yticks()[0]])
fig.savefig('DCA_2ct_DL_Clinical_train.svg', dpi = 300, bbox_inches='tight')
# plt.show()

################################## Plot DCA curves#################################
################################# DL, Clinical, (test)#######################
#构造一个分类效果不是很好的模型
y_proba_model1_d1 = y_proba_model1[278:278+120] 
y_proba_model2_d1 = y_proba_model2[278:278+120] 
y_proba_model3_d1 = y_proba_model3[278:278+120] 
y_label_d1        = y_true[278:278+120] 
thresh_group = np.arange(0,1,0.01)
net_benefit_model1 = calculate_net_benefit_model(thresh_group, y_proba_model1_d1, y_label_d1)
net_benefit_all = calculate_net_benefit_all(thresh_group, y_label_d1)
fig, ax = plt.subplots()
ax = plot_DCA(ax, thresh_group, net_benefit_model1, net_benefit_all, labelname='CTSMamba')
net_benefit_model2 = calculate_net_benefit_model(thresh_group, y_proba_model2_d1, y_label_d1)
ax = plot_DCA(ax, thresh_group, net_benefit_model2, net_benefit_all, labelname='Clinical')
# net_benefit_model3 = calculate_net_benefit_model(thresh_group, y_proba_model3_d1, y_label_d1)
# ax = plot_DCA(ax, thresh_group, net_benefit_model3, net_benefit_all, labelname='DL+Clinical')
ax.plot(thresh_group, net_benefit_all, color = 'black', label = 'Treat all')
ax.plot((0, 1), (0, 0), color = 'black', linestyle = ':', label = 'Treat none')
###lancet journal format #### need to replace . to ·   把小数点.改成·  ##lancet digital medicine format 
plt.xticks([tick for tick in plt.xticks()[0]], [
        f'{tick:.1f}'.replace('.', '·') for tick in plt.xticks()[0]])
plt.yticks([tick for tick in plt.yticks()[0]], [
        f'{tick:.1f}'.replace('.', '·') for tick in plt.yticks()[0]])
fig.savefig('DCA_2ct_DL_Clinical_test.svg', dpi = 300, bbox_inches='tight')
# plt.show()

################################## Plot DCA curves#################################
################################# DL, Clinical, (evc1)#######################
#构造一个分类效果不是很好的模型
y_proba_model1_d1 = y_proba_model1[278+120:278+120+335] 
y_proba_model2_d1 = y_proba_model2[278+120:278+120+335] 
y_proba_model3_d1 = y_proba_model3[278+120:278+120+335] 
y_label_d1        = y_true[278+120:278+120+335] 
thresh_group = np.arange(0,1,0.01)
net_benefit_model1 = calculate_net_benefit_model(thresh_group, y_proba_model1_d1, y_label_d1)
net_benefit_all = calculate_net_benefit_all(thresh_group, y_label_d1)
fig, ax = plt.subplots()
ax = plot_DCA(ax, thresh_group, net_benefit_model1, net_benefit_all, labelname='CTSMamba')
net_benefit_model2 = calculate_net_benefit_model(thresh_group, y_proba_model2_d1, y_label_d1)
ax = plot_DCA(ax, thresh_group, net_benefit_model2, net_benefit_all, labelname='Clinical')
# net_benefit_model3 = calculate_net_benefit_model(thresh_group, y_proba_model3_d1, y_label_d1)
# ax = plot_DCA(ax, thresh_group, net_benefit_model3, net_benefit_all, labelname='DL+Clinical')
ax.plot(thresh_group, net_benefit_all, color = 'black', label = 'Treat all')
ax.plot((0, 1), (0, 0), color = 'black', linestyle = ':', label = 'Treat none')
###lancet journal format #### need to replace . to ·   把小数点.改成·  ##lancet digital medicine format 
plt.xticks([tick for tick in plt.xticks()[0]], [
        f'{tick:.1f}'.replace('.', '·') for tick in plt.xticks()[0]])
plt.yticks([tick for tick in plt.yticks()[0]], [
        f'{tick:.1f}'.replace('.', '·') for tick in plt.yticks()[0]])
fig.savefig('DCA_2ct_DL_Clinical_evc1.svg', dpi = 300, bbox_inches='tight')
# plt.show()

################################## Plot DCA curves#################################
################################# DL, Clinical, (evc2)#######################
#构造一个分类效果不是很好的模型
y_proba_model1_d1 = y_proba_model1[278+120+335:278+120+335+288] 
y_proba_model2_d1 = y_proba_model2[278+120+335:278+120+335+288] 
y_proba_model3_d1 = y_proba_model3[278+120+335:278+120+335+288] 
y_label_d1        = y_true[278+120+335:278+120+335+288] 
thresh_group = np.arange(0,1,0.01)
net_benefit_model1 = calculate_net_benefit_model(thresh_group, y_proba_model1_d1, y_label_d1)
net_benefit_all = calculate_net_benefit_all(thresh_group, y_label_d1)
fig, ax = plt.subplots()
ax = plot_DCA(ax, thresh_group, net_benefit_model1, net_benefit_all, labelname='CTSMamba')
net_benefit_model2 = calculate_net_benefit_model(thresh_group, y_proba_model2_d1, y_label_d1)
ax = plot_DCA(ax, thresh_group, net_benefit_model2, net_benefit_all, labelname='Clinical')
# net_benefit_model3 = calculate_net_benefit_model(thresh_group, y_proba_model3_d1, y_label_d1)
# ax = plot_DCA(ax, thresh_group, net_benefit_model3, net_benefit_all, labelname='DL+Clinical')
ax.plot(thresh_group, net_benefit_all, color = 'black', label = 'Treat all')
ax.plot((0, 1), (0, 0), color = 'black', linestyle = ':', label = 'Treat none')
###lancet journal format #### need to replace . to ·   把小数点.改成·  ##lancet digital medicine format 
plt.xticks([tick for tick in plt.xticks()[0]], [
        f'{tick:.1f}'.replace('.', '·') for tick in plt.xticks()[0]])
plt.yticks([tick for tick in plt.yticks()[0]], [
        f'{tick:.1f}'.replace('.', '·') for tick in plt.yticks()[0]])
fig.savefig('DCA_2ct_DL_Clinical_evc2.svg', dpi = 300, bbox_inches='tight')


######################################### Plot ROC curves ########################################
######################################## ROC Curves of the DL model  ###################################
plt.figure()
plt.plot(fpr_model1_d1, tpr_model1_d1, label='Train DL (AUC = %0.3f)' % roc_auc_model1_d1)  ##, color='r', linestyle='-'
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.axis("square")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right', fontsize=12)
plt.savefig('ROC train_2ct.svg', bbox_inches='tight') 

plt.figure()
plt.plot(fpr_model1_d2, tpr_model1_d2, label='InterValid DL (AUC = %0.3f)' % roc_auc_model1_d2)  ##, color='r', linestyle='-'
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.axis("square")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right', fontsize=12)
plt.savefig('ROC InterValid_2ct.svg', bbox_inches='tight') 

plt.figure()
plt.plot(fpr_model1_d3, tpr_model1_d3, label='ExterValid1 DL (AUC = %0.3f)' % roc_auc_model1_d3)  ##, color='r', linestyle='-'
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.axis("square")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right', fontsize=12)
plt.savefig('ROC ExterValid1_2ct.svg', bbox_inches='tight') 

plt.figure()
plt.plot(fpr_model1_d4, tpr_model1_d4, label='ExterValid2 DL (AUC = %0.3f)' % roc_auc_model1_d4)  ##, color='r', linestyle='-'
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.axis("square")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right', fontsize=12)
plt.savefig('ROC ExterValid2_2ct.svg', bbox_inches='tight') 


######################################### ROC Curves comparison between DL and Clinical ##########################################################
plt.figure()
plt.plot(fpr_model1_d1, tpr_model1_d1, label=f'CTSMamba (AUC = {roc_auc_model1_d1:.3f})'.replace('.', '·') )  ##, color='r', linestyle='-'
plt.plot(fpr_model2_d1, tpr_model2_d1, label=f'Clinical (AUC = {roc_auc_model2_d1:.3f})'.replace('.', '·') )  ## color='r', linestyle=':', 
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
###lancet journal format #### need to replace . to ·   把小数点.改成·  ##lancet digital medicine format 
plt.xticks([tick for tick in plt.xticks()[0]], [
        f'{tick:.1f}'.replace('.', '·') for tick in plt.xticks()[0]])
plt.yticks([tick for tick in plt.yticks()[0]], [
        f'{tick:.1f}'.replace('.', '·') for tick in plt.yticks()[0]])
plt.axis("square")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right', fontsize=12)
plt.savefig('ROC d1_2ct_DL_Clinical.svg', bbox_inches='tight')
# plt.show()

plt.figure()
plt.plot(fpr_model1_d2, tpr_model1_d2, label=f'CTSMamba (AUC = {roc_auc_model1_d2:.3f})'.replace('.', '·') )  ##, color='r', linestyle='-'
plt.plot(fpr_model2_d2, tpr_model2_d2, label=f'Clinical (AUC = {roc_auc_model2_d2:.3f})'.replace('.', '·') )  ## color='r', linestyle=':', 
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
###lancet journal format #### need to replace . to ·   把小数点.改成·  ##lancet digital medicine format 
plt.xticks([tick for tick in plt.xticks()[0]], [
        f'{tick:.1f}'.replace('.', '·') for tick in plt.xticks()[0]])
plt.yticks([tick for tick in plt.yticks()[0]], [
        f'{tick:.1f}'.replace('.', '·') for tick in plt.yticks()[0]])
plt.axis("square")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right', fontsize=12)
plt.savefig('ROC d2_2ct_DL_Clinical.svg', bbox_inches='tight')
# plt.show()

plt.figure()
plt.plot(fpr_model1_d3, tpr_model1_d3, label=f'CTSMamba (AUC = {roc_auc_model1_d3:.3f})'.replace('.', '·')  )  ##, color='r', linestyle='-'
plt.plot(fpr_model2_d3, tpr_model2_d3, label=f'Clinical (AUC = {roc_auc_model2_d3:.3f})'.replace('.', '·') )  ## color='r', linestyle=':', 
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
###lancet journal format #### need to replace . to ·   把小数点.改成·  ##lancet digital medicine format 
plt.xticks([tick for tick in plt.xticks()[0]], [
        f'{tick:.1f}'.replace('.', '·') for tick in plt.xticks()[0]])
plt.yticks([tick for tick in plt.yticks()[0]], [
        f'{tick:.1f}'.replace('.', '·') for tick in plt.yticks()[0]])
plt.axis("square")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right', fontsize=12)
plt.savefig('ROC d3_2ct_DL_Clinical.svg', bbox_inches='tight')
# plt.show()

plt.figure()
plt.plot(fpr_model1_d4, tpr_model1_d4, label=f'CTSMamba (AUC = {roc_auc_model1_d4:.3f})'.replace('.', '·') )  ##, color='r', linestyle='-'
plt.plot(fpr_model2_d4, tpr_model2_d4, label=f'Clinical (AUC = {roc_auc_model2_d4:.3f})'.replace('.', '·') )  ## color='r', linestyle=':', 
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
###lancet journal format #### need to replace . to ·  把小数点.改成·  ##lancet digital medicine format 
plt.xticks([tick for tick in plt.xticks()[0]], [
        f'{tick:.1f}'.replace('.', '·') for tick in plt.xticks()[0]])
plt.yticks([tick for tick in plt.yticks()[0]], [
        f'{tick:.1f}'.replace('.', '·') for tick in plt.yticks()[0]])
plt.axis("square")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right', fontsize=12)
plt.savefig('ROC d4_2ct_DL_Clinical.svg', bbox_inches='tight')
# plt.show()


################################### plot_calibration_curve ####################################
n_bins=5
#### train cohort 
from sklearn.calibration import calibration_curve
plt.figure()
fraction_of_positives, mean_predicted_value = calibration_curve(y_true[:278], y_proba_model1[:278], n_bins=n_bins) 
plt.plot(mean_predicted_value, fraction_of_positives, "o-")  ##label='Training'
plt.plot([0, 1], [0, 1], "k--")
# Set the x and y ticks to the specified ranges with replaced decimal points  把小数点.改成·  ##lancet digital medicine format 
plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [f'{tick:.1f}'.replace('.', '·') for tick in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]])
plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [f'{tick:.1f}'.replace('.', '·') for tick in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]])
plt.xlabel("Mean Predicted Value")
plt.ylabel("Fraction of Positives")
# plt.legend(loc='lower right', fontsize=12)
plt.savefig('calibration_curve DL_2ct train.svg', bbox_inches='tight') 
# plt.show()
plt.close()

#### test cohort 
plt.figure()
fraction_of_positives, mean_predicted_value = calibration_curve(y_true[278:278+120], y_proba_model1[278:278+120], n_bins=n_bins) 
plt.plot(mean_predicted_value, fraction_of_positives, "o-")  ##label='Internal Validation'
plt.plot([0, 1], [0, 1], "k--")
# Set the x and y ticks to the specified ranges with replaced decimal points 把小数点.改成·  ##lancet digital medicine format 
plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [f'{tick:.1f}'.replace('.', '·') for tick in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]])
plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [f'{tick:.1f}'.replace('.', '·') for tick in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]])
plt.xlabel("Mean Predicted Value")
plt.ylabel("Fraction of Positives")
# plt.legend(loc='lower right', fontsize=12)
plt.savefig('calibration_curve DL_2ct test.svg', bbox_inches='tight') 
# plt.show()
plt.close()

#### evc1 cohort 
plt.figure()
fraction_of_positives, mean_predicted_value = calibration_curve(y_true[278+120:278+120+335], y_proba_model1[278+120:278+120+335], n_bins=n_bins) 
plt.plot(mean_predicted_value, fraction_of_positives, "o-")  ## label='External Validation 1'
plt.plot([0, 1], [0, 1], "k--")
# Set the x and y ticks to the specified ranges with replaced decimal points  把小数点.改成·  ##lancet digital medicine format 
plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [f'{tick:.1f}'.replace('.', '·') for tick in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]])
plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [f'{tick:.1f}'.replace('.', '·') for tick in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]])
plt.xlabel("Mean Predicted Value")
plt.ylabel("Fraction of Positives")
# plt.legend(loc='lower right', fontsize=12)
plt.savefig('calibration_curve DL_2ct evc1.svg', bbox_inches='tight') 
# plt.show()
plt.close()

#### evc2 cohort 
plt.figure()
fraction_of_positives, mean_predicted_value = calibration_curve(y_true[278+120+335:278+120+335+288], y_proba_model1[278+120+335:278+120+335+288], n_bins=n_bins) 
plt.plot(mean_predicted_value, fraction_of_positives, "o-")  ##label='External Validation 2'
plt.plot([0, 1], [0, 1], "k--")
# Set the x and y ticks to the specified ranges with replaced decimal points  把小数点.改成·  ##lancet digital medicine format 
plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [f'{tick:.1f}'.replace('.', '·') for tick in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]])
plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [f'{tick:.1f}'.replace('.', '·') for tick in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]])
plt.xlabel("Mean Predicted Value")
plt.ylabel("Fraction of Positives")
# plt.legend(loc='lower right', fontsize=12)
plt.savefig('calibration_curve DL_2ct evc2.svg', bbox_inches='tight') 
# plt.show()
plt.close()

print()
