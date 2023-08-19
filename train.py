import argparse, os, glob, math, json # build-in package
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import classification_report, average_precision_score
from mymodule import getData, MyModel # self-defined module contains dataset and model

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--class-info", type=str, default="cats,dogs")
parser.add_argument("--lr-scheduler", type=str, default="none") # none, linear, sine
parser.add_argument("--exp-name", type=str, default="exp1")
args = parser.parse_args()
print(args)

# check GPU
print(torch.cuda.is_available(), torch.backends.cudnn.is_available(), torch.cuda.get_device_name(0))
device = torch.device('cuda')

# global setting
os.makedirs(f"./output/{args.exp_name}", exist_ok=True)

# prepare dataset
classL, classes = args.class_info.split(","), args.class_info.count(",")+1
trainPathL = sorted(glob.glob("training_set/training_set/*/*.jpg")) # len=8007
validPathL = sorted(glob.glob("test_set/test_set/*/*.jpg")) # len=2025
trainDataset, trainY, trainDataLoader = getData(trainPathL, True, args.batch_size, classL)
validDataset, validY, validDataLoader = getData(validPathL, False, args.batch_size, classL)

# prepare model
model = MyModel(classes).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
lr_schedulerD = {\
    "none": lambda x:x,
    "linear": lambda x: (1 - x / (args.epochs - 1)) * (1.0 - 0.1) + 0.1,
    "sine": lambda x: 1 - 0.9 * math.sin(0.5*math.pi*x/args.epochs)
}
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedulerD[args.lr_scheduler])

loss_func = torch.nn.CrossEntropyLoss()

# training
class History: # collecting reports for every epoch # Can plot finally 
    def __init__(self, classes:int):
        self.D = {
            "loss": [],
            **{ str(cls):{"precision":[], "recall":[], "f1-score":[], "AP":[]} for cls in range(classes) },
            "accuracy":[], "mAP":[],
            **{ view:{"precision":[], "recall":[], "f1-score":[]} for view in ('macro avg','weighted avg') },
        }
    
    def update(self, loss:float, classAP:list, report:dict):
        self.D["loss"].append( round(loss,5) )
        self.D["mAP"].append( round(sum(classAP)/len(classAP),4) ) # macro
        for cls in range(len(classAP)):
            self.D[str(cls)]["AP"].append( round(classAP[cls],4) )
        for key in report.keys():
            if key=="accuracy":
                self.D[key].append( round(report[key],4) )
            else:
                for subkey in report[key]:
                    if subkey!="support":
                        self.D[key][subkey].append( round(report[key][subkey],4) )

historyTrain = History(classes)
historyValid = History(classes)
bestValidAP = 0
for ep in range(1,args.epochs+1):
    print(f"Epoch: {ep}/{args.epochs}")
    
    if 1: # training loop # for pretty aligning with validation loop
        lossAll, predCls, predPro = 0, [], []
        for i,(X,y) in enumerate(trainDataLoader):
            X, y = X.to(device), y.to(device).squeeze()
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_func(pred,y) # ypred:torch.float32(B,Prob), ytrue:torch.int8(B,)
            loss.backward()
            optimizer.step()
            lossAll += loss.item() / len(trainDataset)
            print(f"\rbatch={i+1}/{len(trainDataLoader)}, trainLoss={lossAll:.5f}", end="")
            predCls += pred.argmax(axis=1).cpu().detach().numpy().tolist()
            predPro.append( torch.nn.functional.softmax(pred,dim=1).cpu().detach().numpy() )
        scheduler.step()
        print("\n", classification_report(trainY, predCls) )
        predPro = np.concatenate(predPro, axis=0)
        classAP = [ average_precision_score(trainY==cls, predPro[:,cls]) for cls in range(classes) ]
        historyTrain.update( lossAll, classAP, classification_report(trainY, predCls, output_dict=True) )

    with torch.no_grad(): # validation loop
        lossAll, predCls, predPro = 0, [], []
        for i,(X,y) in enumerate(validDataLoader):
            X, y = X.to(device), y.to(device).squeeze()
            pred = model(X)
            loss = loss_func(pred,y)
            lossAll += loss.item() / len(validDataset)
            print(f"\rbatch={i+1}/{len(validDataLoader)}, validLoss={lossAll:.5f}", end="")
            predCls += pred.argmax(axis=1).cpu().detach().numpy().tolist()
            predPro.append( torch.nn.functional.softmax(pred,dim=1).cpu().detach().numpy() )
        print("\n", classification_report(validY, predCls) )
        predPro = np.concatenate(predPro, axis=0)
        classAP = [ average_precision_score(validY==cls, predPro[:,cls]) for cls in range(classes) ]
        historyValid.update( lossAll, classAP, classification_report(validY, predCls, output_dict=True) )

        if historyValid.D["mAP"][-1]>bestValidAP:
            bestValidAP = historyValid.D["mAP"][-1]
            print(f"Get best Valid AP: {bestValidAP}")
            torch.save(model.state_dict(), f"./output/{args.exp_name}/model_ep{ep}.pt")

with open(f"./output/{args.exp_name}/history_train.json","w") as f:
    json.dump(historyTrain.D, f)
with open(f"./output/{args.exp_name}/history_valid.json","w") as f:
    json.dump(historyValid.D, f)

# plot result
def rowPlot(keys:list, keyPrefix:str, trainD:dict, validD:dict, saveName:str):
    plt.figure(figsize=(6*len(keys),4))
    for i in range(len(keys)):
        plt.subplot(1,len(keys),i+1)
        plt.plot(trainD[keys[i]])
        plt.plot(validD[keys[i]])
        plt.legend(labels=["train","valid"], fontsize=14)
        plt.xlabel("epochs", fontsize=14)
        plt.ylabel(keyPrefix+keys[i], fontsize=14)
    plt.savefig(f"./output/{args.exp_name}/{saveName}")
    #plt.show()

rowPlot(["loss", "accuracy", "mAP"], "global_", historyTrain.D, historyValid.D, "global1.jpg")
rowPlot(["precision", "recall", "f1-score"], "macro_", historyTrain.D["macro avg"], historyValid.D["macro avg"], "global2.jpg")
rowPlot(["precision", "recall", "f1-score"], "weighted_", historyTrain.D["weighted avg"], historyValid.D["weighted avg"], "global3.jpg")
for cls in range(classes):
    rowPlot(["precision", "recall", "f1-score", "AP"], f"class_{cls}_", historyTrain.D[str(cls)], historyValid.D[str(cls)], f"class{cls}.jpg")
print("successfully finished :D")