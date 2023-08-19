import argparse, glob # build-in package
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics import precision_recall_curve, roc_curve
import pandas as pd
import cv2
from mymodule import getData, MyModel # self-defined module contains dataset and model

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--class-info", type=str, default="cats,dogs")
parser.add_argument("--exp-name", type=str, default="exp1")
parser.add_argument("--model-path", type=str, default="exp1/model_ep22.pt")
args = parser.parse_args()
print(args)

# check GPU
print(torch.cuda.is_available(), torch.backends.cudnn.is_available(), torch.cuda.get_device_name(0))
device = torch.device('cuda')

# prepare dataset
classL, classes = args.class_info.split(","), args.class_info.count(",")+1
validPathL = sorted(glob.glob("test_set/test_set/*/*.jpg")) # len=2025
validDataset, validY, validDataLoader = getData(validPathL, False, args.batch_size, classL)

#prepare model
model = MyModel(classes).to(device)
model.load_state_dict(torch.load(f"output/{args.model_path}"))
model.eval()
loss_func = torch.nn.CrossEntropyLoss(reduction="none")

# inference
lossL, predCls, predPro = [], [], []
with torch.no_grad():
    for i,(X,y) in enumerate(validDataLoader):
        X, y = X.to(device), y.to(device).squeeze()
        pred = model(X)
        loss = loss_func(pred,y)
        print(f"\rbatch={i+1}/{len(validDataLoader)}", end="")
        lossL += loss.cpu().detach().numpy().tolist()
        predCls += pred.argmax(axis=1).cpu().detach().numpy().tolist()
        predPro.append( torch.nn.functional.softmax(pred,dim=1).cpu().detach().numpy() )
print()
predPro = np.concatenate(predPro, axis=0)
predSco = predPro.max(axis=1)

# output result
df = pd.DataFrame({"path":validDataset.pathL, "gt":validY, "pdCls":predCls, \
    "pdPro":[tuple(row) for row in predPro], "loss":lossL})
df.to_csv(f"./output/{args.exp_name}/test_prediction.csv", index=False)

# Visualize worst image
df_sort = df[ df["pdCls"] != df["gt"] ].sort_values(by=["loss"])
plt.figure(figsize=(24,12))
for i in range(10):
    plt.subplot(2,5,i+1)
    path, gt, pdCls, pdPro, loss = df_sort.iloc[i]
    img = cv2.imread(path)
    pdPro = tuple( str(round(ele,3)) for ele in pdPro )
    plt.title(f"gt,pd,pdPro={gt,pdCls,pdPro}", fontsize=14)
    plt.imshow(img[:,:,::-1])
    plt.xlabel(path.split('/')[-1], fontsize=14)
plt.savefig(f"output/{args.exp_name}/worst_imgs.jpg")
#plt.show()

# improvement: Get best threshold by macro-f1 # It's useful espscially has "backgorund class"
weights = [ (validY==cls).sum() for cls in range(classes) ]
precisionL, recallL, f1L = np.zeros((101,classes)), np.zeros((101,classes)), np.zeros((101,classes))
bestmf1, bestThreshold, bestC = 0, 0.5, np.zeros((classes,classes))
for i in range(101): # confusion matrix in sklearn: gt-i, pd-j
    C = np.zeros((classes,classes))
    for gt,pdcls,pdsco in zip(validY, predCls, predSco):
        C[gt][pdcls] += int(pdsco>i*0.01)
    for cls in range(classes):
        p = C[cls][cls] / C[:,cls].sum() if C[:,cls].sum() else 0
        r = C[cls][cls] / C[cls,:].sum() if C[cls,:].sum() else 0
        f = 2*p*r/(p+r) if p+r else 0
        precisionL[i][cls], recallL[i][cls], f1L[i][cls] = p, r, f
    mf1 = f1L[i].mean()
    if mf1>bestmf1:
        bestmf1, best_threshold, bestC = mf1, i*0.01, C.copy()
best_i = round(best_threshold/0.01)
classAP = np.array([ average_precision_score(validY==cls,predPro[:,cls]) for cls in range(classes) ])
classAU = np.array([ roc_auc_score(validY==cls, predPro[:,cls]) for cls in range(classes) ])

def printStats(metric:str, row:np.array, weights=weights):
    print(f"class_{metric}={row}, macro={row.mean()}, micro={np.average(row,weights=weights)}")

print("best threshold=", best_threshold)
printStats("precision", precisionL[best_i])
printStats("recall", recallL[best_i])
printStats("f1", f1L[best_i])
printStats("AP", classAP)
printStats("AUC", classAU)
print("confusion matrix (i,j)=(gt,pd):\n", bestC)

# plot results
fprL, tprL = [], [] # (classes, points)
for cls in range(classes):
    fpr, tpr, _ = roc_curve(validY==cls, predPro[:,cls])
    fprL.append(fpr), tprL.append(tpr)

ref_recallL, ref_precisionL = [], [] # (classes, points) # refined
for cls in range(classes):
    precision, recall, _ = precision_recall_curve(validY==cls, predPro[:,cls])
    ref_recallL.append(recall), ref_precisionL.append(precision)

def plotPRF(title, rL, pL, fL):
    R = list(range(101))
    plt.plot(rL, color="r")
    plt.plot(pL, color="b")
    plt.plot(fL, color="g")
    plt.legend(labels=["recall", "precision","f1"])
    plt.scatter(R, rL, color="r")
    plt.scatter(R, pL, color="b")
    plt.scatter(R, fL, color="g")
    plt.title(title, fontsize=14)
    plt.xlabel("threshold", fontsize=14)
    plt.ylabel(f"score", fontsize=14)

def plotCur(title, X, Y, xlabel, ylabel): # For PRC and ROC
    plt.plot(X, Y)
    plt.scatter(X, Y)
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)

for cls in range(classes):
    plt.figure(figsize=(6*3,4))
    plt.subplot(1,3,1)
    plotPRF(f"class_{cls}", recallL[:,cls], precisionL[:,cls], f1L[:,cls])
    plt.subplot(1,3,2)
    plotCur(f"class_{cls}", ref_recallL[cls], ref_precisionL[cls], "recall", "precision")
    plt.subplot(1,3,3)
    plotCur(f"class_{cls}", fprL[cls], tprL[cls], "fpr", "tpr")
    plt.savefig(f"./output/{args.exp_name}/test_class{cls}.jpg")
    plt.show()
print("successfully finished :D")