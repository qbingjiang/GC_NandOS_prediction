# Model Overview
This repository contains the code for CTSMamba, a model designed to predict lymph node metastasis (LNM) and overall survival (OS) in patients with locally advanced gastric cancer (LAGC) who underwent neo-adjuvant chemotherapy (NAC). The model leverages two pre-operative imaging time points (one before NAC and one after) to improve predictive performance.
The model was developed and internally validated on a cohort from a single institution (training set = 278, internal validation = 120) and further externally validated on two additional independent cohorts (n = 335 and n = 288) from other institutions. The results demonstrate robust performance in both tasks (LNM and OS prediction), with impressive generalization across external cohorts. 
The results are thoroughly analyzed across various clinical factors, highlighting the model's robustness and consistent performance across diverse patient subgroups. Overall, the findings are well-presented and demonstrate the potential of CTSMamba as a valuable tool for clinical decision-making in LAGC patients undergoing NAC.

### Requirements
- **Operating System:** Ubuntu 22.04 LTS
- **Python Version:** Python 3.8
- **CUDA:** Required for GPU computation

### Installation

   ```bash
   git clone https://github.com/qbingjiang/GC_NandOS_prediction.git
   cd GC_NandOS_prediction
```

### Installing Dependencies
Dependencies can be installed using:
``` bash
pip install -r requirements.txt
```

### Training

A CTSMamba network with standard hyper-parameters for the task of LNM and OS prediction can be defined as follows:

``` bash
featsEncoder = TSMamba(in_chans=1,
                            out_chans=1,
                            depths=[2,2,2,2],
                            feat_size=[48, 96, 192, 384]) 
coattNet = CoAttNet(out_chans_class=1, out_chans_surv=10) 

```

The above CTSMamba model is used for CT images (1-channel input) and for LNM and OS prediction outputs. 
The network expects resampled input images with size of ```(96, 96, 96)```.

Using the default values for hyper-parameters, the following command can be used to initiate training using PyTorch package:
``` bash
python train_featsEncoder.py --training_set './BDataset/training_set.xlsx' --validation_set './BDataset/validation_set.xlsx' --batch_size 2

```

Note that you need to provide the location of your dataset directories by using ```--training_set``` and ```--validation_set```.

After trained ```featsEncoder``` model, then we can train the CTSMamba model using: 
``` bash
python train_CTSMamba.py --training_set './BDataset/training_set.xlsx' --validation_set './BDataset/validation_set.xlsx' --batch_size 2
```

### Testing
You can use the best checkpoint of CTSMamba to test it on your own data.

The following command runs inference using the provided checkpoint: 
``` bash
python test_CTSMamba.py --training_set './BDataset/training_set.xlsx' --validation_set './BDataset/validation_set.xlsx' --save_path_train './Bresults/pred_train.xlsx' --save_path_test './Bresults/pred_test.xlsx' --metrics_path './Bresults/metrics_table.csv'
```


## References and Acknowledgement 
Many thanks for these works and repos for their great contribution!

[1] Meng, Mingyuan, et al. "AdaMSS: Adaptive Multi-Modality Segmentation-to-Survival Learning for Survival Outcome Prediction from PET/CT Images." arXiv preprint arXiv:2305.09946 (2023). https://github.com/MungoMeng/Survival-AdaMSS.

[2] Xing, Zhaohu, et al. "Segmamba: Long-range sequential modeling mamba for 3d medical image segmentation." International Conference on Medical Image Computing and Computer-Assisted Intervention. Cham: Springer Nature Switzerland, 2024. https://github.com/ge-xing/SegMamba.

[3] https://github.com/bowang-lab/U-Mamba