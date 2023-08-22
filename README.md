# Prerequistes
### Software
+ Python 3.9.12
+ Pytorch 1.9.0+cu111
+ opencv ("pip install opencv-python-headless" is recommended)
+ numpy
+ matplotlib
+ pandas
### Hardware
+ Since using ResNet50 (25M parameters) as Model, GPU momory >= 4GB is available

# File structure
+ train.py
+ test.py
+ mymodule.py # contains dataset and model which can be shared for train.py and test.py
+ training_set/training_set/ # dataset
    + cats/*.jpg
    + dogs/*.jpg
+ test_set/test_set/ # dataset
    + cats/*.jpg
    + dogs/*.jpg
+ output/ # (auto-generated while training and testing)
    + exp1/
        + *.pt # trained model
        + *.jpg # result curve
        + *.json # history result
        + *.csv # inference result

Please download dataset from [here](https://www.kaggle.com/datasets/tongpython/cat-and-dog) and extract to the folder as above
For testing only please download the weights from [here](https://drive.google.com/file/d/1SibZx_Pad8YbYrUjE7dOvrZIc099zZwP/view?usp=drive_link) to "output/exp1/"

# Quick start
### train
```
python train.py [--options]
```

more arguments:
+ batch_size: int, default=16
+ epochs: int, default=30
+ lr-scheduler: str, default="linear"
+ exp-name: str, default="exp1" # output directory under ./output

### test
```
python test.py [--options]
```

more arguments:
+ batch_size: int, default=16
+ exp-name: str, default="exp1" # output directory under ./output
+ model-path: str, default="exp1/model_ep22.pt"

testing can share exp-name with training because the files of testing result will have prefix "testing".

# More
+ Coding style
    + as precise as possible
    + as comprehensive as possible
+ Improvement than usual:
    + learning rate scheduler which is widely use in SOTA CV
    + more compelte metrics
        + At training step, compute metrics independent from threshold.
        + At testing step, threshold optimization can be applied and plotting AUROC & AUPRC curves are needed.
    + output top-10 worst predicted images by ranking of loss -> improve image quality in real scenario. 
+ Use CNN for the toy model is a trade-off between accuracy and efficiency. (SVM is weak but ViT is large)
+ Output prediction results for further post-analysis if needed.
+ Feel free to contact me if you have any question. Thanks.
