import random # build-in package
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchvision import transforms
from torchvision.models import resnet50

class MyDataset(Dataset):
    def __init__(self, pathL:list, classL:list): # imgs belongs to the name of the folder 
        self.pathL = pathL
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), # RGB mean & std based on imagenet
        ]) # can applied on (3,w,h) or (B,3,w,h)
        self.labelDict = { category:i for i,category in enumerate(classL) }

    def __len__(self):
        return len(self.pathL)

    def __getitem__(self, index):
        y = self.labelDict[self.pathL[index].split("/")[-2]]
        y = torch.Tensor([y]).type(torch.long) # Tensor([0]) or Tensor([1])
        x = cv2.imread( self.pathL[index] )/255. # read & normalize
        x = torch.Tensor(x).permute(2,0,1) # convert to Tensor & channel first
        x = self.transform(x)
        return x,y # (3,224,224), (1,)

    def getY(self): # for easily computing classification report
        return np.array([
            self.labelDict[self.pathL[index].split("/")[-2]] for index in range(len(self)) ]).astype(int)

def getData(pathL:list, shuffle:bool, batch_size:int, classL:list):
    random.Random(7).shuffle(pathL) if shuffle else None # set seed for reproducibility
    dataset = MyDataset(pathL, classL)
    Y = dataset.getY() # for easily getting classifiction report
    dataLoader = DataLoader(dataset, batch_size=batch_size) # ceil(8007/16)=501 | ceil(2025/16)=127
    return dataset, Y, dataLoader

class MyModel(nn.Module):
    def __init__(self, classes:int):
        super().__init__()
        self.resnet50 = resnet50() # (B,3,224,224) -> (B,1000)
        self.dense1  = nn.Linear(1000, 64)
        self.dropout = nn.Dropout(0.1)
        self.dense2  = nn.Linear(64,classes)
        print("parameters=", sum( parameter.numel() for parameter in self.parameters() ) ) # 25M
        
    def forward(self, x): # (B,3,224,224) -> (B,2)
        x = self.resnet50(x)
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return x