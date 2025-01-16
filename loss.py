import torch
import torch.nn as nn

# def cox_loss(y_true, y_pred):

#     time_value = torch.squeeze(torch.index_select(y_true, 1, torch.tensor([0]).cuda()))
#     event = torch.squeeze(torch.index_select(y_true, 1, torch.tensor([1]).cuda())).bool()
#     score = torch.squeeze(y_pred, dim=1)

#     ix = torch.where(event)

#     sel_mat = (time_value[ix[0]].unsqueeze(-1) <= time_value).float()

#     p_lik = torch.gather(score, 0, ix) - torch.log(torch.sum(sel_mat * torch.exp(score.t()), dim=-1))

#     loss = -torch.mean(p_lik)

#     return loss

class DiceLoss(nn.Module):
    def __init__(self):  #weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, logits, targets):
        num = targets.size(0)
        m1 = logits.view(num, -1)
        m2  = targets.view(num,-1)
        intersection = (m1 * m2)
        score = (2.0 * intersection.sum(1)+1.0) / (m1.sum(1) + m2.sum(1)+1.0)
        score = 1- score.sum()/num
        return score

class cox_loss(nn.Module):
    def __init__(self):  #weight=None, size_average=True):
        super(cox_loss, self).__init__()

    def forward(self, y_true, y_pred):
    # def cox_loss(y_true, y_pred): 
        time_value = y_true[:, 0]
        event = y_true[:, 1].bool()
        score = y_pred.squeeze()
        nan_mask = torch.isnan(time_value)
        event[nan_mask] = False

        if not torch.any(event):
            return torch.tensor(1e-8, requires_grad=True)

        ix = torch.where(event)
        sel_mat = torch.gather(time_value, 0, ix[0]).unsqueeze(0).t() <= time_value
        p_lik = score - torch.log(torch.sum((sel_mat * torch.exp(score))+ 1e-8, dim=0))
        loss = -torch.mean(p_lik )
        return loss
    
        # risk = y_pred.squeeze()
        # time, event = y_true[:, 0], y_true[:, 1]
        # risk_sorted, indices = torch.sort(risk, descending=True)
        # time_sorted = time[indices]
        # event_sorted = event[indices]

        # hazard_ratio = torch.exp(risk_sorted)
        # log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0))
        # uncensored_likelihood = risk_sorted - log_risk
        # censored_likelihood = uncensored_likelihood * event_sorted
        # neg_likelihood = -torch.sum(censored_likelihood)
        # return neg_likelihood / torch.sum(event)

import numpy as np
class cox_loss_v2(nn.Module):
    def __init__(self):  #weight=None, size_average=True):
        super(cox_loss_v2, self).__init__()

    def forward(self, y_true, hazard_pred):
    # def cox_loss(y_true, y_pred): 
        # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
        # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
        survtime = y_true[:, 0]
        censor = y_true[:, 1].bool()
        hazard_pred = hazard_pred.squeeze()

        current_batch_len = len(survtime)
        R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
        for i in range(current_batch_len):
            for j in range(current_batch_len):
                R_mat[i, j] = survtime[j] >= survtime[i]

        R_mat = torch.FloatTensor(R_mat)
        theta = hazard_pred.reshape(-1)
        exp_theta = torch.exp(theta)
        loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta * R_mat, dim=1))) * censor)
        return loss_cox


def R_set(x):
	'''Create an indicator matrix of risk sets, where T_j >= T_i.
	Note that the input data have been sorted in descending order.
	Input:
		x: a PyTorch tensor that the number of rows is equal to the number of samples.
	Output:
		indicator_matrix: an indicator matrix (which is a lower traiangular portions of matrix).
	'''
	n_sample = x.size(0)
	matrix_ones = torch.ones(n_sample, n_sample)
	indicator_matrix = torch.tril(matrix_ones)
	return(indicator_matrix)

def neg_par_log_likelihood(pred, ytime, yevent):#event=0,censored
    #ytime should be sorted with increasing order
	'''Calculate the average Cox negative partial log-likelihood.
	Input:
		pred: linear predictors from trained model.
		ytime: true survival time from load_data().
		yevent: true censoring status from load_data().
	Output:
		cost: the cost that is to be minimized.
	'''
	###exclude nan data from evc1 os data 
	nan_mask = torch.isnan(ytime).squeeze()
	yevent = yevent[~nan_mask,:] 
	pred = pred[~nan_mask,:] 
	ytime = ytime[~nan_mask,:] 
      
	n_observed = yevent.sum(0)
	ytime_indicator = R_set(ytime)
	if torch.cuda.is_available():
		ytime_indicator = ytime_indicator.cuda()
	risk_set_sum = ytime_indicator.mm(torch.exp(pred)) 
	diff = pred - torch.log(risk_set_sum)
	sum_diff_in_observed = torch.transpose(diff, 0, 1).mm(yevent)
	cost = (- (sum_diff_in_observed / n_observed)).reshape((-1,))
	return(cost)


def concordance_index(y_true, y_pred):
    time_value = y_true[:, 0]
    event = y_true[:, 1].bool()

    ix = torch.where((torch.unsqueeze(time_value, dim=-1) < time_value) & torch.unsqueeze(event, dim=-1))

    s1 = torch.index_select(y_pred, 0, ix[0])
    s2 = torch.index_select(y_pred, 0, ix[1])
    ci = torch.mean((s1 < s2).float())

    return ci


def c_index(pred, ytime, yevent):
	'''Calculate concordance index to evaluate models.
	Input:
		pred: linear predictors from trained model.
		ytime: true survival time from load_data().
		yevent: true censoring status from load_data().
	Output:
		concordance_index: c-index (between 0 and 1).
	'''
	###exclude nan data from evc1 os data 
	nan_mask = torch.isnan(ytime).squeeze()
	yevent = yevent[~nan_mask] 
	pred = pred[~nan_mask] 
	ytime = ytime[~nan_mask] 

	n_sample = len(ytime)
	ytime_indicator = R_set(ytime)
	ytime_matrix = ytime_indicator - torch.diag(torch.diag(ytime_indicator))
	###T_i is uncensored
	censor_idx = (yevent == 0).nonzero()
	zeros = torch.zeros(n_sample)
	ytime_matrix[censor_idx, :] = zeros
	###1 if pred_i < pred_j; 0.5 if pred_i = pred_j
	pred_matrix = torch.zeros_like(ytime_matrix)
	for j in range(n_sample):
		for i in range(n_sample):
			if pred[i] < pred[j]:
				pred_matrix[j, i]  = 1
			elif pred[i] == pred[j]: 
				pred_matrix[j, i] = 0.5
	
	concord_matrix = pred_matrix.mul(ytime_matrix)
	###numerator
	concord = torch.sum(concord_matrix)
	###denominator
	epsilon = torch.sum(ytime_matrix)
	###c-index = numerator/denominator
	concordance_index = torch.div(concord, epsilon)
	###if gpu is being used
	if torch.cuda.is_available():
		concordance_index = concordance_index.cuda()
	###
	return(concordance_index)

def Loglike_loss(y_true, y_pred, n_intervals=10):
    '''
    y_true: Tensor.
        First half: 1 if individual survived that interval, 0 if not.
        Second half: 1 for time interval before which failure has occured, 0 for other intervals.
    y_pred: Tensor.
        Predicted survival probability (1-hazard probability) for each time interval.
    '''
    
    cens_uncens = torch.clamp(1.0 + y_true[:,0:n_intervals]*(y_pred-1.0), min=1e-5)
    uncens = torch.clamp(1.0 - y_true[:,n_intervals:2*n_intervals]*y_pred, min=1e-5)
    loss = -torch.mean(torch.log(cens_uncens) + torch.log(uncens))
        
    return loss


def L2_Regu_loss(_, weights, alpha=0.1):
    '''
    Loss for L2 Regularization on weights
    '''
    
    loss = 0
    for weight in weights:
        loss += torch.square(weight).sum()

    return alpha * loss





