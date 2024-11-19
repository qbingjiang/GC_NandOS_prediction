

import torch 
# import sys
# sys.path.append(r"./mamba")
# sys.path.append(r"./mamba/mamba_ssm")
from model_build import CTSMamba
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

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
seed=5
np.random.seed(seed)
torch.manual_seed(seed)

training_set1 = r'/path2dataset_train.xlsx'  
interValid_set1 = r'/path2dataset_test.xlsx'
X_train = pd.read_excel(training_set1 )
X_test = pd.read_excel(interValid_set1 )

from make_dataloader import imBalanced_MES, prepare_data_v3, make_dataloader_v3
image_mask_files_times_train, label_DrugRest_ptLever_train          = prepare_data_v3(X_train)
image_mask_files_times_test, label_DrugRest_ptLever_test            = prepare_data_v3(X_test)

ifDealWithImbalance=False 
if ifDealWithImbalance: 
    X_train_path, y_train, X_test_path, y_test = imBalanced_MES(image_mask_files_times_train, label_DrugRest_ptLever_train, num_perClass=100 ) 
else: 
    X_train_path = [ image_mask_files_times_train ] 
    y_train      = [ label_DrugRest_ptLever_train ] 

image_mask_files_times_all = image_mask_files_times_train + image_mask_files_times_test 

label_DrugRest_ptLever_all = [
                            label_DrugRest_ptLever_train[0] + label_DrugRest_ptLever_test[0] , 
                            label_DrugRest_ptLever_train[1] + label_DrugRest_ptLever_test[1] , 
                            ] 
    
## make the train dataloader
trainloader = make_dataloader_v3(X_train_path, y_train, bs=8, ifshuffle=False)  ##, iftransform=True 
## make the test dataloader
testloader0 = make_dataloader_v3(X_train_path, y_train, bs=1, ifshuffle=False ) 
X_train_path_test, y_train_test = [ image_mask_files_times_test ], [ label_DrugRest_ptLever_test ] 
testloader01 = make_dataloader_v3(X_train_path_test, y_train_test, bs=1, ifshuffle=False ) 


model = CTSMamba().cuda()

cuda = True if torch.cuda.is_available() else False
if cuda:
    model.cuda()
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor 

criterion = nn.BCELoss(reduction='mean') 
criterion_dice = DiceLoss()
criterion_surv = cox_loss() 

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-5) 
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
        print(f'auc = {auc:.5f}, Sen = {Sen:.5f}, Spe = {Spe:.5f} ') 

        mask = (labels_status_list==1) | (labels_status_list==0) 
        labels_time_list = labels_time_list[mask] 
        labels_status_list = labels_status_list[mask] 
        pred_surv_list = pred_surv_list[mask] 
        pred_N_list = pred_N_list[mask]
        surv_ci = concordance_index(torch.from_numpy(np.array([ labels_time_list, labels_status_list]).transpose()), 
                                    torch.from_numpy(np.array(pred_surv_list)) ) 
        print(f'surv_ci = {surv_ci:.5f}') 
        c_index2 = c_index(torch.from_numpy(np.array(pred_surv_list)), torch.from_numpy(np.array(labels_time_list)), torch.from_numpy(np.array(labels_status_list)))
        print(f'concordance_index = {c_index2:.5f}') 
        best_cutoff_value = hr_values(pred_surv_list, labels_time_list, labels_status_list, best_cutoff_value)
        # surv_ci = concordance_index(torch.from_numpy(np.array([ labels_time_list, labels_status_list]).transpose()), 
        #                             torch.from_numpy(np.array(pred_N_list)) )
        # print(f'surv_ci = {surv_ci:.5f}') 
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
    print(f'auc = {auc:.5f}, Sen = {Sen:.5f}, Spe = {Spe:.5f} ') 

    valid_cindex = lifelines.utils.concordance_index(labels_time_list, Survival_time, labels_status_list)
    valid_cindex_info = 'Valid C-index: %.4f' % valid_cindex
    print(valid_cindex_info) 

    best_cutoff_value = hr_values(Survival_time, labels_time_list, labels_status_list, best_cutoff_value)

    return best_cutoff_value

# from torch.utils.tensorboard import SummaryWriter

def train(model, trainloader):  
    import gc
    # writer = SummaryWriter('experiment_1')
    for epoch in range(1, 11 ):  # loop over the dataset multiple times
        model.train()
        train_total_loss = []
        for i, data in enumerate(trainloader, 0): 
            x, y, labels = data
            labels_N = labels[0].unsqueeze_(1).type(Tensor) 
            labels_surv = torch.transpose(torch.stack(labels[1], dim=0), 0,1 ).type(Tensor) 
            # pred_N, pred_surv = model((x[0]*y[0]).type(Tensor),(x[1]*y[1]).type(Tensor)) 
            pred, pred_surv = model( (x[0]).type(Tensor), (x[1]).type(Tensor) ) 
            loss_N = criterion(pred, labels_N.type(Tensor)) 
            loss_surv = 0
            loss_list = []
            for i, Loss in enumerate(Losses):
                curr_loss = Loss(labels_surv, pred_surv[i]) * Weights[i] 
                loss_list.append(curr_loss.item()) 
                loss_surv += curr_loss 
            alpha=0.5
            loss = alpha*loss_N + (1-alpha)*loss_surv 

            # backpropagate and optimize
            optimizer.zero_grad()
            loss.backward() 
            optimizer.step() 
            train_total_loss.append(loss.item())

        print('-'*80)
        print(f'[{epoch}-{i}] pred loss: {loss.item():.3f}, {loss_N.item():.3f}, {loss_surv.item():.3f}') 
        print('Train loss: %.4f' % (np.mean(train_total_loss)) )

        best_cutoff_value = test_model_v2(model, testloader0, X_train)
        test_model_v2(model, testloader01, X_test, best_cutoff_value)

        if epoch%5==0: 
            torch.save(model.state_dict(), f"./pth/ctsmamba-{epoch}.pth") 
        gc.collect() 

### train the model
train(model, trainloader) 


