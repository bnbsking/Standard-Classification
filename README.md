# Preface
This is a template repository for **N-class classification problem** based on Pytorch. The example dataset is from [here](https://www.kaggle.com/datasets/tongpython/cat-and-dog).

# Prerequisites
### Software
+ Install python 3.9.12. [reference](https://docs.conda.io/projects/miniconda/en/latest/)
+ Install packages
```
pip install -r requirements.txt 
```
+ Install pytorch alongside gpu
```
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```
### Hardware
+ ResNet50 (25M parameters), GPU momory >= 4GB is guaranteed work

# File structure
+ main.py # modes include train, valid, infer
+ utils.py # common utilities e.g. dataset, model, plot, etc.
+ training_set/training_set/ # dataset
    + cats/*.jpg
    + dogs/*.jpg
+ test_set/test_set/ # dataset
    + cats/*.jpg
    + dogs/*.jpg  
+ results/ # auto-generated by main.py
    + exp1/
        + *_args.json # arguments
        + *_pred.csv # prediction results
        + *.jpg # result curve
        + *.pt # trained model weights
        + history.json # training history

# Quick start
### Train / Valid / Infer  
```
python main.py [--options]
```
+ \--mode: be ''train" or "valid" or "infer"
+ see more in main.py

# Pipeline
The code will be executed in the following steps
+ Global setting
	+ set random seed
	+ make result folder
	+ read args then save at the folder
	+ check GPU and set device
+ Dataset
	+ Customized part for preparing:
		+ train_path: list[str]. path of 1 training data
		+ valid_path: list[str]. path of 1 validation data
		+ train_label: list[int]. class index of each data in train_path
		+ valid_label: list[int]. class index of each data in valid_path
	+ generate loaders according to the above format
+ Setting weights for loss function
	+ [Train] Count counts of each class and get weights from harmonic mean
+ Model
	+ Get model (Backbone + Linear head).
	+ Resume checkpoint
	+ To GPU
+ Loss function
	+ [Train] Loss reduced by mean in a batch
	+ [Valid/Infer] Loss not reduced
+ Optimizer
	+ get optmizer and lr_scheduler
+ Core
	+ [Train] grad loop
		+ standard
		+ collect all prediction as shape (N, classes)
		+ compute F1, APs, mAP, cls_report
	+ no-grad loop
		+ standard
		+ [Valid/Infer] collect all loss as shape (N,) 
		+ collect all prediction as shape (N, classes)
		+ compute F1, APs, mAP, cls_report
	+ more
		+ [Train] save history and save checkpoint if reach best mAP
		+ [Valid] save history and compute AUC & specificity & confusion matrix with exporting top-N losses
		+ [Infer] export top-N unconfidence
+ Save prediction results
+ Plotting
	+ [Train] History of loss, f1, mAP of train and valid
	+ [Valid] PRF-T curve, P-R curve, ROC curve

# Features
+ As precise and comprehensive as possible
+ SOTA backbone, lr scheduler
+ Complete metrics across industry and medical 
+ Valid mode: Export worst images from confusion matrix
+ Infer mode: Export the most unconfident images
+ Feel free to contact me if you have any question. Thanks.
