"""
h: history, v: single
        loss, f1s, f1, aps, map, cls_report, aucs, auc, spec, specs, confusion, export_wst, prt, pr
train:  h     v    h   v    h    v
valid:  v     v    v   v    v    v           v     v    v     v      v          v           v    v
infer:                                                                          v

+ background class
"""
import argparse, os, glob, json, shutil

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, average_precision_score, classification_report
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve

import utils

# arguments
parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default='train', choices=['train', 'valid', 'infer'])
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--output-dim", type=int, default=2, help="binary classification can be 1 or 2 affects metrics")
parser.add_argument("--resume", type=str, default='', help="checkpoint path")
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--save-results", type=str, default="./save_results/exp1", help="results folder")
parser.add_argument("--save-models", type=str, default="./save_models/exp1", help="models folder")
args = parser.parse_args()
args.backbone = 'resnet50'
args.pretrained = True
args.optim_algo = 'adam'
args.lr_scheduler = 'none'
args.lr = 0.0003
args.loss = 'CE' if args.output_dim>1 else 'BCE'
args.loss_weight = None
print(args)

# global setting
os.makedirs(args.save_results, exist_ok=True)
os.makedirs(args.save_models, exist_ok=True)
json.dump(vars(args), open(f"{args.save_results}/{args.mode}_args.json", "w"))
classes = max(2, args.output_dim)

# check GPU
print(torch.cuda.is_available(), torch.backends.cudnn.is_available(), torch.cuda.get_device_name(0))
device = torch.device('cuda')

# prepare dataset
if 1: # customize part
    train_c0 = sorted(glob.glob("./training_set/training_set/cats/*.jpg"))
    train_c1 = sorted(glob.glob("./training_set/training_set/dogs/*.jpg"))
    train_path  = train_c0 + train_c1
    train_label = [0]*len(train_c0) + [1]*len(train_c1)
    valid_c0 = sorted(glob.glob("./test_set/test_set/cats/*.jpg"))
    valid_c1 = sorted(glob.glob("./test_set/test_set/dogs/*.jpg"))
    valid_path = infer_path = valid_c0 + valid_c1
    valid_label = [0]*len(valid_c0) + [1]*len(valid_c1)
if args.mode == 'train':
    train_loader = utils.get_loader(train_path, train_label, 'train', args.batch_size)
if args.mode in ('train', 'valid'):
    valid_loader = utils.get_loader(valid_path, valid_label, 'valid', args.batch_size)
if args.mode == 'infer':
    valid_loader = utils.get_loader(infer_path, [0]*len(infer_path), 'infer', args.batch_size)

# prepare model
model = utils.MyModel(args.backbone, args.pretrained, args.output_dim)
if args.resume:
    model.load_state_dict(torch.load(args.resume))
model.to(device)

# optimizer
optimizer, scheduler = utils.get_optimizer(model, args.optim_algo, args.lr, args.lr_scheduler)

# loss
loss_func = utils.get_loss(args.loss, args.loss_weight, 'mean' if args.mode=='train' else 'none')

# training
history = utils.History(classes, args.save_results)
if args.mode=='train':
    train_label = train_loader.dataset.label_list
if args.mode in ('valid', 'infer'):
    model.eval()
    loss_all = []
valid_label = valid_loader.dataset.label_list
for ep in range(args.epochs):
    print(f"Epoch: {ep+1}/{args.epochs}") if args.mode=='train' else None
    
    # training loop
    if args.mode=='train':
        pred_cls_all, pred_probs_all = [], []
        history.train_loss.append(0)
        for i, (x, y) in enumerate(train_loader):
            print(f"\rbatch={i+1}/{len(train_loader)}, train_loss={history.train_loss[-1]:.5f}", end="")
            
            # basic
            x, y = x.to(device), y.to(device)
            if args.loss=='CE':
                y = y.reshape(-1)
            else:
                y = y.type(torch.float32)
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_func(pred, y) # CE:(B,2),(B,)int; BCE:(B,1),(B,1)float
            loss.backward()
            optimizer.step()

            # history loss
            history.train_loss[-1] += loss.item() / len(train_label)
            
            # pred collect
            if args.loss=='CE':
                pred_cls, pred_probs = pred.argmax(axis=1), torch.nn.functional.softmax(pred, dim=1)
            else:
                pred_cls, pred_probs = (pred>0).type(torch.long), torch.sigmoid(pred)
                pred_cls, pred_probs = pred_cls.reshape(-1), torch.cat((pred_probs,1-pred_probs), axis=1)
            pred_cls_all += pred_cls.cpu().detach().numpy().tolist()
            pred_probs_all += pred_probs.cpu().detach().numpy().tolist()

        # pred numpy
        scheduler.step()
        pred_cls_all = np.array(pred_cls_all)
        pred_probs_all = np.array(pred_probs_all)
        
        # history f1 & aps & map
        history.train_f1.append( f1_score(train_label, pred_cls_all, average='macro') )
        for i in range(classes):
            ap = average_precision_score(train_label==i, pred_probs_all[:,i])
            history.train_aps[i].append(ap)
        history.train_map.append( sum(history.train_aps[i][-1] for i in range(classes))/classes )
        print("\n", history, "\n", classification_report(train_label, pred_cls_all) )

    # validation loop
    with torch.no_grad():
        pred_cls_all, pred_probs_all = [], []
        history.valid_loss.append(0)
        for i, (x, y) in enumerate(valid_loader):
            print(f"\rbatch={i+1}/{len(valid_loader)}, valid_loss={history.valid_loss[-1]:.5f}", end="")
            
            # basic
            x, y = x.to(device), y.to(device)
            if args.loss=='CE':
                y = y.reshape(-1)
            else:
                y = y.type(torch.float32)
            pred = model(x)
            loss = loss_func(pred, y)

            # history loss
            if args.mode != 'train':
                loss_all += loss.cpu().detach().numpy().tolist()
                loss = loss.mean(axis=0)
            history.valid_loss[-1] += loss.item() / len(valid_loader.dataset)
            
            # pred collect
            if args.loss=='CE':
                pred_cls, pred_probs = pred.argmax(axis=1), torch.nn.functional.softmax(pred, dim=1)
            else:
                pred_cls, pred_probs = (pred>0).type(torch.long), torch.sigmoid(pred)
                pred_cls, pred_probs = pred_cls.reshape(-1), torch.cat((pred_probs,1-pred_probs), axis=1)
            pred_cls_all += pred_cls.cpu().detach().numpy().tolist()
            pred_probs_all += pred_probs.cpu().detach().numpy().tolist()
        
        # pred numpy
        pred_cls_all = np.array(pred_cls_all)
        pred_probs_all = np.array(pred_probs_all)

        # history f1 & aps & map
        history.valid_f1.append( f1_score(valid_label, pred_cls_all, average='macro') )
        for i in range(classes):
            history.valid_aps[i].append( average_precision_score(valid_label==i, pred_probs_all[:,i]) )
        history.valid_map.append( sum(history.valid_aps[i][-1] for i in range(classes))/classes )
        print("\n", history, "\n", classification_report(valid_label, pred_cls_all) )

        if args.mode=='train':
            # checkpoint
            if ep==0 or history.valid_map[-1]>history.valid_map[-2]:
                torch.save(model.state_dict(), os.path.join(args.save_models, 'model.pt'))
            history.save(args.mode)
        
        elif args.mode=='valid':
            # auc & sensitivity
            auc_list, specificity_list = [], []
            for i in range(classes):
                auc_list.append( roc_auc_score(valid_label==i, pred_probs_all[:,i]) )
                fpr, tpr, thresholds = roc_curve(valid_label==i, pred_probs_all[:,i])
                specificity_list.append( 1-fpr.mean() )
            print(f"auc_list={auc_list}, auc_mean={sum(auc_list)/classes}")
            print(f"specificity_list={specificity_list}, specificity_mean={sum(specificity_list)/classes}")

            # confusion matrixby max prob + false_imgs_top-N_loss,
            confusion = [ [ [] for _ in range(classes) ] for _ in range(classes) ]
            for gt, pdc, path, loss in zip(valid_label, pred_cls_all, valid_loader.dataset.path_list, loss_all):
                confusion[gt][pdc].append( (loss,path) )
            print(f"confusion={[ [ len(confusion[i][j]) for j in range(classes) ] for i in range(classes) ] }")
        
            # export 
            for i in range(classes):
                for j in range(classes):
                    if i==j: continue
                    grid_path = f"{args.save_results}/confusion/gt_{i}_pd_{j}"
                    os.makedirs(grid_path, exist_ok=True)
                    for _, path in sorted(confusion[i][j])[:5]:
                        shutil.copy(path, grid_path)
            break
        
        else:
            # export worst
            prob_path_list = sorted(zip(pred_probs_all.max(axis=1), valid_loader.dataset.path_list))
            worst_path = f"{args.save_results}/worst_imgs"
            os.makedirs(worst_path, exist_ok=True)
            for prob, path in prob_path_list[:5]:
                shutil.copy(path, worst_path)
            break

# save prediction results
df = pd.DataFrame({
    "data": valid_loader.dataset.path_list,
    "label": valid_label, 
    "pred_probs_all": map(tuple,pred_probs_all),
    "pred_cls_all": pred_cls_all,
})
df.to_csv(os.path.join(args.save_results, f'{args.mode}_pred.csv'), index=False)

# plot results
if args.mode=='train':
    # loss, f1, map x ep
    data_sub1 = [history.train_loss, history.valid_loss]
    data_sub2 = [history.train_f1, history.valid_f1]
    data_sub3 = [history.train_map, history.valid_map]
    utils.row_plot_1d( [data_sub1, data_sub2, data_sub3], ['epoch']*3, ['loss','f1','mAP'], \
        [['train','valid'] for i in range(3)], os.path.join(args.save_results, "curve_loss_map.jpg") )

elif args.mode=='valid':
    # first plot: p/r/f curve per class # secod_plot: p-r curve per class
    data = [] # for first plot
    data_x, data_y = [], [] # for second plot
    for i in range(classes):
        precision, recall, thresholds = precision_recall_curve(valid_label==i, pred_probs_all[:,i])
        refine_precision = [precision[0]]
        for j in range(1,len(precision)):
            refine_precision.append( max(refine_precision[-1],precision[i]) )
        f1 = [ 2*p*r/(p+r) if p+r else 0 for p, r in zip(refine_precision, recall) ]
        
        # collect data
        data.append([refine_precision, recall, f1])
        data_x.append([recall])
        data_y.append([refine_precision])
    
    # plot
    utils.row_plot_1d(data, ['threshold']*classes, ['']*classes, [['precision','recall'] for _ in range(classes)], \
        os.path.join(args.save_results, "curve_prt.jpg") )
    utils.row_plot_2d(data_x, data_y, ['recall']*classes, ['precision']*classes, ['']*classes, \
        os.path.join(args.save_results, "curve_pr.jpg") )

print("successfully finished :D")