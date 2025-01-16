
import pandas as pd
import torch
from torch.autograd import Variable
import torch.nn as nn
import random 
from torch.utils.data import Dataset
import skimage
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset 
import numpy as np 

def prepare_data(df_features_shanxi_rm_nan): 
    image_mask_files_times = []
    for i in range(len(df_features_shanxi_rm_nan)): 
        image_mask_files_times.append([
                                    [df_features_shanxi_rm_nan['patient_image1'].tolist()[i], df_features_shanxi_rm_nan['patient_mask1'].tolist()[i]], 
                                    [df_features_shanxi_rm_nan['patient_image2'].tolist()[i], df_features_shanxi_rm_nan['patient_mask2'].tolist()[i]]
                                    ] )
    label_DrugRest_ptLever = [df_features_shanxi_rm_nan['target_'].tolist(), df_features_shanxi_rm_nan['是否死亡'].tolist(), df_features_shanxi_rm_nan['OS时间（月份）'].tolist()] 
    return image_mask_files_times, label_DrugRest_ptLever


def make_surv_array(time, event):
    '''
    Transforms censored survival data into vector format that can be used in Keras.
    Arguments
        time: Array of failure/censoring times.
        event: Array of censoring indicator. 1 if failed, 0 if censored.
        breaks: Locations of breaks between time intervals for discrete-time survival model (always includes 0)
    Returns
        surv_array: Dimensions with (number of samples, number of time intervals*2)
    '''
    breaks = np.array([0,10,20,30,40,50,60,70,80,120,160])
    
    n_samples=time.shape[0]
    n_intervals=len(breaks)-1
    timegap = breaks[1:] - breaks[:-1]
    breaks_midpoint = breaks[:-1] + 0.5*timegap
    
    surv_array = np.zeros((n_samples, n_intervals*2))
    for i in range(n_samples):
        if event[i] == 1:
            surv_array[i,0:n_intervals] = 1.0*(time[i]>=breaks[1:]) 
            if time[i]<breaks[-1]:
                surv_array[i,n_intervals+np.where(time[i]<breaks[1:])[0][0]]=1
        else: # event[i] == 0
            surv_array[i,0:n_intervals] = 1.0*(time[i]>=breaks_midpoint)
    
    return surv_array


def prepare_data_v2(df_features_shanxi_rm_nan): 
    image_mask_files_times = []
    for i in range(len(df_features_shanxi_rm_nan)): 
        image_mask_files_times.append([
                                    df_features_shanxi_rm_nan['patient_image1'].tolist()[i], df_features_shanxi_rm_nan['patient_image2'].tolist()[i]
                                    ] )
    surv_array = make_surv_array( time=df_features_shanxi_rm_nan['OS时间（月份）'].to_numpy(), event=df_features_shanxi_rm_nan['是否死亡'].to_numpy() )
    label_DrugRest_ptLever = [df_features_shanxi_rm_nan['target_'].tolist(), surv_array.tolist() ] 
    return image_mask_files_times, label_DrugRest_ptLever

def prepare_data_v3(df_features_shanxi_rm_nan): 
    image_mask_files_times = []
    for i in range(len(df_features_shanxi_rm_nan)): 
        image_mask_files_times.append([
                                    [df_features_shanxi_rm_nan['patient_image1'].tolist()[i], df_features_shanxi_rm_nan['patient_mask1'].tolist()[i]], 
                                    [df_features_shanxi_rm_nan['patient_image2'].tolist()[i], df_features_shanxi_rm_nan['patient_mask2'].tolist()[i]]
                                    ] )
    surv_array = make_surv_array( time=df_features_shanxi_rm_nan['OS时间（月份）'].to_numpy(), event=df_features_shanxi_rm_nan['是否死亡'].to_numpy() )
    label_DrugRest_ptLever = [df_features_shanxi_rm_nan['target_'].tolist(), surv_array.tolist() ] 
    return image_mask_files_times, label_DrugRest_ptLever

def imBalanced_MES(Xpath, y, num_perClass=100 ): 
    '''Multiexpert systems (MES)  
    considers the imbalanced classes distribution'''
    y = np.array(y)
    percent_1 = np.sum(y==1) / y.shape[0]  
    ind_1 = np.where(y==1)
    ind_0 = np.where(y==0)
    Xpath_1 = [Xpath[i] for i in ind_1[0]]
    Xpath_0 = [Xpath[i] for i in ind_0[0]] 
    y_1, y_0 = list(y[ind_1[0]]), list(y[ind_0[0]]) 
    ind_min = len(ind_1[0]) 
    ind_max = len(ind_0[0]) 

    X_train_1 = Xpath_1[:num_perClass]
    y_train_1 = y_1[:num_perClass] 
    proportion = num_perClass / ind_min 
    num_train_0 = int(ind_max * proportion)

    split_points = list(range(0, num_train_0, num_perClass) ) 
    if ( num_train_0 - split_points[-1] ) / num_perClass < 0.5: 
        split_points = split_points[:-1] 
    
    # for i in range(len(split_points)): 
    #     a = random.randint(num_perClass, num_train_0-num_perClass)
    
    X_train_list = [] 
    y_train_list = [] 
    for i in range(len(split_points)): 
        X_train = X_train_1 + Xpath_0[split_points[i]:split_points[i]+num_perClass]
        y_train = y_train_1 + y_0[    split_points[i]:split_points[i]+num_perClass]
        X_train_list.append(X_train)
        y_train_list.append(y_train) 
    
    X_test  = Xpath_1[num_perClass: ] + Xpath_0[num_train_0: ]
    y_test  = y_1[num_perClass: ] + y_0[num_train_0: ]

    return X_train_list, y_train_list, X_test, y_test 

def imBalanced_MES_v2(Xpath, y, num_perClass=100 ): 
    '''Multiexpert systems (MES)  
    considers the imbalanced classes distribution'''
    y = np.array(y)
    percent_1 = np.sum(y==1) / y.shape[0]  
    ind_1 = np.where(y==1)
    ind_0 = np.where(y==0)
    Xpath_1 = [Xpath[i] for i in ind_1[0]]
    Xpath_0 = [Xpath[i] for i in ind_0[0]] 
    y_1, y_0 = list(y[ind_1[0]]), list(y[ind_0[0]]) 
    ind_min = len(ind_1[0]) 
    ind_max = len(ind_0[0]) 

    X_train_0 = Xpath_0[:num_perClass] 
    y_train_0 = y_0[:num_perClass] 
    # combined_tensor = torch.cat(X_train_0, dim=0)
    X_train_0 = torch.stack(X_train_0, dim=0) 
    stacked_X = torch.cat([ Xpath, X_train_0 ], dim=0) 
    stacked_y = np.concatenate([y, np.array( y_train_0 )] ) 
    stacked_y = torch.tensor(stacked_y) 
    return stacked_X, stacked_y  


import nibabel as nib
import pickle
import skimage
from skimage.transform import resize
import SimpleITK as sitk
import json
import os
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
import torchio as tio

import imgaug as ia
from imgaug import augmenters as iaa
import cv2

def Data_augmentation_seg(PET, CT, Seg_Tumor, Seg_Node):
    
    # define augmentation sequence
    aug_seq = iaa.Sequential([
        # translate/move them and rotate them.
        iaa.Affine(translate_percent={"x": [-0.1, 0.1], "y": [-0.1, 0.1]},
                   scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                   shear=(-20, 20),
                   rotate=(-20, 20)),
        # iaa.CropToFixedSize(width=112, height=None)
        iaa.Fliplr(0.4),  # horizontally flip 50% of the images
        iaa.Flipud(0.2),  # vertically flip 20% of the images
        iaa.GaussianBlur(sigma=(0, 0.5)),  # apply Gaussian blur with a sigma of 0 to 0.5
        iaa.LinearContrast((0.75, 1.25)),  # improve or reduce the contrast
        iaa.Multiply((0.8, 1.2)),  # change brightness, multiplying the pixel values by 0.8 to 1.2
        ], 
        random_order=True)

    # flip/translate in x axls, rotate along z axls
    images = np.concatenate((PET,CT,Seg_Tumor,Seg_Node), -1)
    
    images_aug = np.array(aug_seq(images=images))
    
    PET = images_aug[..., 0:int(images_aug.shape[3]/4)]
    CT = images_aug[..., int(images_aug.shape[3]/4):int(images_aug.shape[3]/4*2)]
    Seg_Tumor = images_aug[..., int(images_aug.shape[3]/4*2):int(images_aug.shape[3]/4*3)]
    Seg_Node = images_aug[..., int(images_aug.shape[3]/4*3):int(images_aug.shape[3])]
    
    # translate in z axls, rotate along y axls
    PET = np.transpose(PET,(0,3,1,2))
    CT = np.transpose(CT,(0,3,1,2))
    Seg_Tumor = np.transpose(Seg_Tumor,(0,3,1,2))
    Seg_Node = np.transpose(Seg_Node,(0,3,1,2))
    images = np.concatenate((PET,CT,Seg_Tumor,Seg_Node), -1)
    
    images_aug = np.array(aug_seq(images=images))
    
    PET = images_aug[..., 0:int(images_aug.shape[3]/4)]
    CT = images_aug[..., int(images_aug.shape[3]/4):int(images_aug.shape[3]/4*2)]
    Seg_Tumor = images_aug[..., int(images_aug.shape[3]/4*2):int(images_aug.shape[3]/4*3)]
    Seg_Node = images_aug[..., int(images_aug.shape[3]/4*3):int(images_aug.shape[3])]
    
    # translate in y axls, rotate along x axls
    PET = np.transpose(PET,(0,3,1,2))
    CT = np.transpose(CT,(0,3,1,2))
    Seg_Tumor = np.transpose(Seg_Tumor,(0,3,1,2))
    Seg_Node = np.transpose(Seg_Node,(0,3,1,2))
    images = np.concatenate((PET,CT,Seg_Tumor,Seg_Node), -1)
    
    images_aug = np.array(aug_seq(images=images))
    
    PET = images_aug[..., 0:int(images_aug.shape[3]/4)]
    CT = images_aug[..., int(images_aug.shape[3]/4):int(images_aug.shape[3]/4*2)]
    Seg_Tumor = images_aug[..., int(images_aug.shape[3]/4*2):int(images_aug.shape[3]/4*3)]
    Seg_Node = images_aug[..., int(images_aug.shape[3]/4*3):int(images_aug.shape[3])]
    
    # recover axls
    PET = np.transpose(PET,(0,3,1,2))
    CT = np.transpose(CT,(0,3,1,2))
    Seg_Tumor = np.transpose(Seg_Tumor,(0,3,1,2))
    Seg_Node = np.transpose(Seg_Node,(0,3,1,2))
    
    # reset Seg mask to 1/0
    for i in range(Seg_Tumor.shape[0]):
        _, Seg_Tumor[i] = cv2.threshold(Seg_Tumor[i],0.4,1,cv2.THRESH_BINARY)
        _, Seg_Node[i] = cv2.threshold( Seg_Node[i], 0.4,1,cv2.THRESH_BINARY)
    
    # # post-process data shape
    # PET = PET[..., np.newaxis].transpose((0,4,1,2,3))
    # CT = CT[..., np.newaxis].transpose((0,4,1,2,3))
    # Seg_Tumor = Seg_Tumor[..., np.newaxis].transpose((0,4,1,2,3))
    # Seg_Node = Seg_Node[..., np.newaxis].transpose((0,4,1,2,3))
    
    return PET, CT, Seg_Tumor, Seg_Node

class data_set(Dataset): 
    def __init__(self, image_paths, label_trg, ifSaveDatasetTemp=False, ifReadDatasetTemp=False, windowCenterWidth=(40, 400), iftransform=False):
        ##
        self.image_paths = image_paths
        self.label_trg = label_trg
        self.ifSaveDatasetTemp = ifSaveDatasetTemp
        self.ifReadDatasetTemp = ifReadDatasetTemp
        # self.windowCenterWidth = windowCenterWidth
        self.imgMinMax = [ windowCenterWidth[0] - windowCenterWidth[1]/2.0, windowCenterWidth[0] + windowCenterWidth[1]/2.0 ]
        self.iftransform = iftransform 
    def __len__(self):
        return len(self.image_paths) 

    def _read_itk_files(self, img_path, label_path): 
        image_sitk = sitk.ReadImage( img_path ) 
        x = sitk.GetArrayFromImage(image_sitk) 
        originalimg_spacing = image_sitk.GetSpacing()

        label_sitk = sitk.ReadImage( label_path ) 
        y = sitk.GetArrayFromImage(label_sitk) 
        return x, y, originalimg_spacing

    def __getitem__(self, index): 

        paths = self.image_paths[index]
        x_1_patch_list=[]
        y_1_patch_list=[] 
        if len(paths)==1: 
            paths = [paths, paths]
        for i in range( 2 ):  ##len(paths)  ###for multi-time scan
            img_path, clu_path = paths[i][0], paths[i][1] 
            x_1_patch_, y_1_patch_, _ = self._read_itk_files(img_path, clu_path) 
            # print(x_1_patch_.shape, y_1_patch_.shape)
            x_1_patch_ = np.clip(x_1_patch_, a_min=self.imgMinMax[0], a_max=self.imgMinMax[1] ) 
            x_1_patch_ = (x_1_patch_ - self.imgMinMax[0] ) / (self.imgMinMax[1] - self.imgMinMax[0])
            y_1_t = (y_1_patch_>0.5)*1 
            x_1_patch = x_1_patch_
            y_1_patch = (y_1_patch_ >0.5)*1 
            x_1_patch_ = skimage.transform.resize(x_1_patch, [96, 96, 96], order=1, preserve_range=True, anti_aliasing=False)
            y_1_patch_ = skimage.transform.resize(y_1_patch, [96, 96, 96], order=0, preserve_range=True, anti_aliasing=False)
            x_1_patch_ =  np.expand_dims(x_1_patch_, axis=0 ) 
            y_1_patch_ =  np.expand_dims(y_1_patch_, axis=0 ) 
            x_1_patch_list.append(x_1_patch_) 
            y_1_patch_list.append(y_1_patch_) 
        if self.iftransform: 
            x_1_patch_list[0], x_1_patch_list[1], y_1_patch_list[0], y_1_patch_list[1] = \
                Data_augmentation_seg(np.float32(x_1_patch_list[0]), np.float32(x_1_patch_list[1]), 
                                    np.float32(y_1_patch_list[0]), np.float32(y_1_patch_list[1]) )
        x_1_patch_list = [torch.from_numpy(x_1_patch_ ).type(torch.FloatTensor) for x_1_patch_ in x_1_patch_list ]
        y_1_patch_list = [torch.from_numpy(y_1_patch_ ).type(torch.FloatTensor) for y_1_patch_ in y_1_patch_list ]

        return x_1_patch_list, y_1_patch_list, [self.label_trg[0][index], self.label_trg[1][index], self.label_trg[2][index]]  

    def _transforms(self, image_3d_list, mask_3d_list=None, ): 
        ifAffine     = random.random()>0.3
        ifFlip       = random.random()>0.3
        ifGamma      = False # random.random()>0.6
        ifNoise      = False # random.random()>0.6
        ifBiasField  = False # random.random()>0.6
        ifBlur       = False # random.random()>0.6 
        deg     = random.randint(-30, 30) 
        tran    = random.randint(-5, 5) 
        sca     = random.randint(6,9)/10.0 
        axes_randomFlip = int(np.random.choice([0,1,2]))

        image_3d_list_transformed, mask_3d_list_transformed = [], []
        for i in range(len(image_3d_list)): 
            image_3d = image_3d_list[i]
            subject = tio.Subject( 
                                    image=tio.ScalarImage(tensor=image_3d),)
            # 将 numpy 数组转换为 TorchIO 图像
            # image = tio.Image(image_3d, type=tio.INTENSITY) 
            if mask_3d_list is not None: 
                mask_3d = mask_3d_list[i]
                # mask = tio.Image(mask_3d, type=tio.LABEL)
                subject = tio.Subject(
                                        image=tio.ScalarImage(tensor=image_3d),
                                        mask=tio.LabelMap(tensor=mask_3d),
                                    )
            if ifAffine: 
                transform = tio.RandomAffine(degrees=(deg, deg ),
                                            # translation=(tran, tran, tran),
                                            # scale=(sca, sca, sca),
                                            # shear=(10, 10, 10),
                                            isotropic=False,
                                            default_pad_value='otsu',
                                            image_interpolation='linear',
                                            label_interpolation='nearest',
                                            p=1, 
                                            ) 
                if mask_3d is not None: 
                    subject = transform(subject ) 
                    image = subject['image'] 
                    mask = subject['mask']
                else: 
                    subject = transform(subject ) 
                    image = subject['image'] 
            elif ifFlip: 
                transform  = tio.RandomFlip(axes=axes_randomFlip, 
                                            flip_probability=1
                                            ) 
                if mask_3d is not None: 
                    subject = transform(subject ) 
                    image = subject['image'] 
                    mask = subject['mask']
                else: 
                    subject = transform(subject ) 
                    image = subject['image'] 
            elif ifGamma: 
                transform = tio.RandomGamma(log_gamma=(0.2, 0.2), p=1, include='image')
                if mask_3d is not None: 
                    subject = transform(subject ) 
                    image = subject['image'] 
                    mask = subject['mask']
                else: 
                    subject = transform(subject ) 
                    image = subject['image'] 
            elif ifNoise: 
                transform = tio.RandomNoise(include='image', p=1)
                if mask_3d is not None: 
                    subject = transform(subject) 
                    image = subject['image'] 
                    mask = subject['mask']
                else: 
                    subject = transform(subject) 
                    image = subject['image'] 
            elif ifBiasField: 
                transform = tio.RandomBiasField(include='image', p=1)
                if mask_3d is not None: 
                    subject = transform(subject ) 
                    image = subject['image'] 
                    mask = subject['mask']
                else: 
                    subject = transform(subject ) 
                    image = subject['image']  
            elif ifBlur: 
                transform = tio.RandomBlur(include='image', p=1)
                if mask_3d is not None: 
                    subject = transform(subject ) 
                    image = subject['image'] 
                    mask = subject['mask']
                else: 
                    subject = transform(subject ) 
                    image = subject['image'] 
            else: 
                if mask_3d is not None: 
                    image = subject['image'] 
                    mask = subject['mask']
                else: 
                    image = subject['image'] 
            image_3d_list_transformed.append(image[tio.DATA])
            if mask_3d_list is not None: 
                mask_3d_list_transformed.append(mask[tio.DATA]) 
        if mask_3d_list is not None: 
            return image_3d_list_transformed, mask_3d_list_transformed
        else: 
            return image_3d_list_transformed


def make_dataloader(X_train_path, y_train, bs=10, ifshuffle=True, iftransform=False): 
    if len(X_train_path)>1: 
        dataset_list = [data_set(X_train_path[i], y_train[i] ) for i in range(len(X_train_path))]
        dataset = ConcatDataset(dataset_list)
        dataloader = DataLoader(dataset=dataset, batch_size=bs, num_workers=10, shuffle=ifshuffle, pin_memory=True, prefetch_factor=2) 
    else: 
        X_train_path = X_train_path[0] 
        y_train = y_train[0] 
        dataset = data_set(X_train_path, y_train, iftransform=iftransform ) 
        dataloader = DataLoader(dataset=dataset, batch_size=bs, num_workers=10, shuffle=ifshuffle, pin_memory=True, prefetch_factor=2) 
    return dataloader 

def Data_augmentation_surv(PET, CT):
    
    # define augmentation sequence
    aug_seq = iaa.Sequential([
        iaa.SomeOf((0, 5), [
                            iaa.Affine(
                                    translate_percent={"x": [-0.1, 0.1], "y": [-0.1, 0.1]},
                                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                                    shear=(-20, 20), # shear by -10 to +10 degrees
                                    rotate=(-45, 45), # rotate by -20 to +20 degrees
                                    ),
                            iaa.Fliplr(0.5),  # horizontally flip 50% of the images
                            iaa.Flipud(0.5),  # vertically flip 20% of the images
                            iaa.GaussianBlur(sigma=(0, 0.5)),  # apply Gaussian blur with a sigma of 0 to 0.5
                            iaa.LinearContrast((0.75, 1.2)),  # improve or reduce the contrast
                            iaa.Multiply((0.8, 1.2)),  # change brightness, multiplying the pixel values by 0.8 to 1.2
        ]), 
        ], random_order=True)
    # flip/translate in x axls, rotate along z axls
    images = np.concatenate((PET,CT), -1)
    images_aug = np.array(aug_seq(images=images))
    PET = images_aug[..., 0:int(images_aug.shape[3]/2)]
    CT = images_aug[..., int(images_aug.shape[3]/2):int(images_aug.shape[3])]
    
    # flip/translate in z axls, rotate along y axls
    PET = np.transpose(PET,(0,3,1,2))
    CT = np.transpose(CT,(0,3,1,2))
    
    images = np.concatenate((PET,CT), -1)
    images_aug = np.array(aug_seq(images=images))
    PET = images_aug[..., 0:int(images_aug.shape[3]/2)]
    CT = images_aug[..., int(images_aug.shape[3]/2):int(images_aug.shape[3])]
    
    # flip/translate in y axls, rotate along x axls
    PET = np.transpose(PET,(0,3,1,2))
    CT = np.transpose(CT,(0,3,1,2))
    
    images = np.concatenate((PET,CT), -1)
    images_aug = np.array(aug_seq(images=images))
    PET = images_aug[..., 0:int(images_aug.shape[3]/2)]
    CT = images_aug[..., int(images_aug.shape[3]/2):int(images_aug.shape[3])]
    
    # recover axls
    PET = np.transpose(PET,(0,3,1,2))
    CT = np.transpose(CT,(0,3,1,2))
    return PET, CT

class data_set_v2(Dataset): 
    def __init__(self, image_paths, label_trg, ifSaveDatasetTemp=False, ifReadDatasetTemp=False, windowCenterWidth=(40, 400), iftransform=False):
        ##
        self.image_paths = image_paths
        self.label_trg = label_trg
        self.ifSaveDatasetTemp = ifSaveDatasetTemp
        self.ifReadDatasetTemp = ifReadDatasetTemp
        # self.windowCenterWidth = windowCenterWidth
        self.imgMinMax = [ windowCenterWidth[0] - windowCenterWidth[1]/2.0, windowCenterWidth[0] + windowCenterWidth[1]/2.0 ]
        self.iftransform = iftransform 
    def __len__(self):
        return len(self.image_paths) 

    def __getitem__(self, index): 

        paths = self.image_paths[index]
        x_1_patch_list=[]
        y_1_patch_list=[] 
        x0 = np.load(paths[0]).squeeze(0) 
        x1 = np.load(paths[1]).squeeze(0) 
        if self.iftransform: 
            x0, x1 = Data_augmentation_surv(x0, x1)
        x_1_patch_list = [x0, x1] 
        x_1_patch_list = [torch.from_numpy(x_1_patch_ ).type(torch.FloatTensor) for x_1_patch_ in x_1_patch_list ] 
        return x_1_patch_list, [self.label_trg[0][index], self.label_trg[1][index] ]  

def make_dataloader_v2(X_train_path, y_train, bs=10, ifshuffle=True, iftransform=False): 
    X_train_path = X_train_path[0] 
    y_train = y_train[0] 
    dataset = data_set_v2(X_train_path, y_train, iftransform=iftransform ) 
    dataloader = DataLoader(dataset=dataset, batch_size=bs, num_workers=10, shuffle=ifshuffle, pin_memory=True, prefetch_factor=2) 
    return dataloader 

class data_set_v3(Dataset): 
    def __init__(self, image_paths, label_trg, ifSaveDatasetTemp=False, ifReadDatasetTemp=False, windowCenterWidth=(40, 400), iftransform=False):
        ##
        self.image_paths = image_paths
        self.label_trg = label_trg
        self.ifSaveDatasetTemp = ifSaveDatasetTemp
        self.ifReadDatasetTemp = ifReadDatasetTemp
        # self.windowCenterWidth = windowCenterWidth
        self.imgMinMax = [ windowCenterWidth[0] - windowCenterWidth[1]/2.0, windowCenterWidth[0] + windowCenterWidth[1]/2.0 ]
        self.iftransform = iftransform 
    def __len__(self):
        return len(self.image_paths) 

    def _read_itk_files(self, img_path, label_path): 
        image_sitk = sitk.ReadImage( img_path ) 
        x = sitk.GetArrayFromImage(image_sitk) 
        originalimg_spacing = image_sitk.GetSpacing()

        label_sitk = sitk.ReadImage( label_path ) 
        y = sitk.GetArrayFromImage(label_sitk) 
        return x, y, originalimg_spacing

    def __getitem__(self, index): 

        paths = self.image_paths[index]
        x_1_patch_list=[]
        y_1_patch_list=[] 
        if len(paths)==1: 
            paths = [paths, paths]
        for i in range( 2 ):  ##len(paths)  ###for multi-time scan
            img_path, clu_path = paths[i][0], paths[i][1] 
            x_1_patch_, y_1_patch_, _ = self._read_itk_files(img_path, clu_path) 
            x_1_patch_ = np.clip(x_1_patch_, a_min=self.imgMinMax[0], a_max=self.imgMinMax[1] ) 
            x_1_patch_ = (x_1_patch_ - self.imgMinMax[0] ) / (self.imgMinMax[1] - self.imgMinMax[0])
            y_1_t = (y_1_patch_>0.5)*1 
            x_1_patch = x_1_patch_
            y_1_patch = (y_1_patch_ >0.5)*1 
            x_1_patch_ = skimage.transform.resize(x_1_patch, [96, 96, 96], order=1, preserve_range=True, anti_aliasing=False)
            y_1_patch_ = skimage.transform.resize(y_1_patch, [96, 96, 96], order=0, preserve_range=True, anti_aliasing=False)
            x_1_patch_ =  np.expand_dims(x_1_patch_, axis=0 ) 
            y_1_patch_ =  np.expand_dims(y_1_patch_, axis=0 ) 
            x_1_patch_list.append(x_1_patch_) 
            y_1_patch_list.append(y_1_patch_) 
        if self.iftransform: 
            x_1_patch_list[0], x_1_patch_list[1], y_1_patch_list[0], y_1_patch_list[1] = \
                Data_augmentation_seg(np.float32(x_1_patch_list[0]), np.float32(x_1_patch_list[1]), 
                                    np.float32(y_1_patch_list[0]), np.float32(y_1_patch_list[1]) )

        x_1_patch_list = [torch.from_numpy(x_1_patch_ ).type(torch.FloatTensor) for x_1_patch_ in x_1_patch_list ]
        y_1_patch_list = [torch.from_numpy(y_1_patch_ ).type(torch.FloatTensor) for y_1_patch_ in y_1_patch_list ]
        return x_1_patch_list, y_1_patch_list, [self.label_trg[0][index], self.label_trg[1][index] ]  


def make_dataloader_v3(X_train_path, y_train, bs=10, ifshuffle=True, iftransform=False): 
    X_train_path = X_train_path[0] 
    y_train = y_train[0] 
    dataset = data_set_v3(X_train_path, y_train, iftransform=iftransform ) 
    dataloader = DataLoader(dataset=dataset, batch_size=bs, num_workers=10, shuffle=ifshuffle, pin_memory=True, prefetch_factor=2) 
    return dataloader 