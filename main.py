"""
+ metrics
h: history, v: single
        loss, f1s, f1, aps, map, cls_report, aucs, auc, spec, specs, confusion, export_wst, prt, pr, roc
train:  h     v    h   v    h    v
valid:  v     v    v   v    v    v           v     v    v     v      v          v           v    v   v
infer:                                                                          v
+ in object detection: p~r~1 since bg>>fg -> not compute bg metrics, do threshold optimization
+ in cls with bgcls: compute metrics is important, can do threshold optimization
"""
import argparse, glob, json, os

import numpy as np
import pandas as pd
import torch

import utils

# arguments
parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default='train', choices=['train', 'valid', 'infer'])
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--output-dim", type=int, default=2, help="Binary cls can be 1 or 2")
parser.add_argument("--resume", type=str, default='', help="Checkpoint path")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--results", type=str, default="./results/exp1", help="Folder saves results and models")
parser.add_argument("--threshold-opt", type=str, default='', choices=['', 'f1'], help="Works when mode=valid and output_dim=1 only")
args = parser.parse_args()
args.backbone = 'resnet50'
args.pretrained = True
args.optim_algo = 'adam'
args.lr = 1e-4
args.lr_scheduler = 'linear'
print(args)

# global setting
classes = max(args.output_dim, 2)
loss_name = 'CE' if args.output_dim>1 else 'BCE'
torch.manual_seed(7)
os.makedirs(args.results, exist_ok=True)
json.dump(vars(args), open(f"{args.results}/args_{args.mode}.json", "w"))
print(torch.cuda.is_available(), torch.backends.cudnn.is_available(), torch.cuda.get_device_name(0))
device = torch.device('cuda')

# dataset
if 1: # customized part
    train_c0 = sorted(glob.glob("./_data/catdog_simple/training_set/training_set/cats/*.jpg"))
    train_c1 = sorted(glob.glob("./_data/catdog_simple/training_set/training_set/dogs/*.jpg"))
    #train_c2 = sorted(glob.glob("./_data/catdog_simple/training_set/training_set/noise/*.jpg"))
    train_path  = train_c0 + train_c1 #+ train_c2
    train_label = [0]*len(train_c0) + [1]*len(train_c1) #+ [2]*len(train_c2)
    valid_c0 = sorted(glob.glob("./_data/catdog_simple/test_set/test_set/cats/*.jpg"))
    valid_c1 = sorted(glob.glob("./_data/catdog_simple/test_set/test_set/dogs/*.jpg"))
    #valid_c2 = sorted(glob.glob("./_data/catdog_simple/test_set/test_set/noise/*.jpg"))
    valid_path = infer_path = valid_c0 + valid_c1 #+ valid_c2
    valid_label = [0]*len(valid_c0) + [1]*len(valid_c1) #+ [2]*len(valid_c2)
if args.mode == 'train':
    train_loader = utils.get_loader(train_path, train_label, 'train', args.batch_size)
    valid_loader = utils.get_loader(valid_path, valid_label, 'valid', args.batch_size)
    _, cnts = np.unique(train_loader.dataset.label_list, return_counts=True) # category, counts
    loss_weight = torch.tensor([cnts[0]/cnts[1]]) if loss_name=='BCE' else torch.tensor(1/cnts/(1/cnts).sum(), dtype=torch.float32)
elif args.mode == 'valid':
    valid_loader = utils.get_loader(valid_path, valid_label, 'valid', args.batch_size)
    loss_weight = None
else: # 'infer'
    valid_loader = utils.get_loader(infer_path, [0]*len(infer_path), 'infer', args.batch_size)
    loss_weight = None
print(f"loss_weight={loss_weight}")

# model
model = utils.MyModel(args.backbone, args.pretrained, args.output_dim)
if args.resume:
    model.load_state_dict(torch.load(args.resume))
else:
    assert args.mode=='train', f"{args.mode} needs pretrained weights"
model.to(device)

# loss
loss_func = utils.get_loss(loss_name, loss_weight, 'mean' if args.mode=='train' else 'none')
loss_func.to(device)

# optimizer
optimizer, scheduler = utils.get_optimizer(model, args.optim_algo, args.lr, args.epochs, args.lr_scheduler)

# training
history = utils.History(args.results)
if args.mode=='train':
    train_label = train_loader.dataset.label_list
if args.mode in ('valid', 'infer'):
    loss_all = []
valid_label = valid_loader.dataset.label_list
for ep in range(args.epochs):
    print(f"Epoch: {ep+1}/{args.epochs}") if args.mode=='train' else None
    
    # training loop
    model.train()
    if args.mode=='train':
        pred_probs_all = []
        history.train_loss.append(0)
        for i, (x, y) in enumerate(train_loader):
            print(f"\rbatch={i+1}/{len(train_loader)}, train_loss={history.train_loss[-1]:.5f}", end="")
            
            # basic
            x, y = x.to(device), y.to(device)
            y = y.reshape(-1) if loss_name=='CE' else y.type(torch.float32)
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_func(pred, y) # CE:(B,2),(B,)int; BCE:(B,1),(B,1)float
            loss.backward()
            optimizer.step()

            # history loss
            history.train_loss[-1] += loss.item() / len(train_label)
            
            # pred collect
            if loss_name=='CE':
                pred_probs = torch.nn.functional.softmax(pred, dim=1)
            else:
                pred_probs = torch.cat((1-torch.sigmoid(pred), torch.sigmoid(pred)), axis=1)
            pred_probs_all += pred_probs.cpu().detach().numpy().tolist()

        # pred numpy
        scheduler.step()
        pred_probs_all = np.array(pred_probs_all)
        
        # history f1 & aps & map
        metrics = utils.ComputeMetrics(train_label, pred_probs_all)
        history.train_f1.append(metrics.get_f1())
        history.train_aps.append(metrics.get_aps())
        history.train_map.append(sum(history.train_aps[-1])/classes)
        print("\n", history, "\n", metrics.get_cls_report())

    # validation loop
    model.eval()
    with torch.no_grad():
        pred_probs_all = []
        history.valid_loss.append(0)
        for i, (x, y) in enumerate(valid_loader):
            print(f"\rbatch={i+1}/{len(valid_loader)}, valid_loss={history.valid_loss[-1]:.5f}", end="")
            
            # basic
            x, y = x.to(device), y.to(device)
            y = y.reshape(-1) if loss_name=='CE' else y.type(torch.float32)
            pred = model(x)
            loss = loss_func(pred, y)

            # history loss
            if args.mode != 'train':
                loss_all += loss.cpu().detach().numpy().tolist()
                loss = loss.mean(axis=0)
            history.valid_loss[-1] += loss.item() / len(valid_loader.dataset)
            
            # pred collect
            if loss_name=='CE':
                pred_probs = torch.nn.functional.softmax(pred, dim=1)
            else:
                pred_probs = torch.cat((1-torch.sigmoid(pred), torch.sigmoid(pred)), axis=1)
            pred_probs_all += pred_probs.cpu().detach().numpy().tolist()
        
        # pred numpy
        pred_probs_all = np.array(pred_probs_all)

        # history f1 & aps & map
        metrics = utils.ComputeMetrics(valid_label, pred_probs_all, args.threshold_opt)
        history.valid_f1.append(metrics.get_f1())
        history.valid_aps.append(metrics.get_aps())
        history.valid_map.append(sum(history.valid_aps[-1])/classes)
        print("\n", history, "\n", metrics.get_cls_report() )

        if args.mode=='train':
            # checkpoint
            if ep==0 or history.valid_map[-1]>=max(history.valid_map):
                torch.save(model.state_dict(), os.path.join(args.results, 'model.pt'))
            
            # show ap
            print("valid_aps=", history.valid_aps[-1], "\n"+"-"*50)
            history.save()
        
        elif args.mode=='valid':
            # show ap
            print("valid_aps=", history.valid_aps[-1])
            history.save('valid')

            # auc & sensitivity
            aucs, specificities = metrics.get_aucs_specificities()
            print(f"aucs={aucs}, auc_mean={sum(aucs)/classes}")
            print(f"specificities={specificities}, specificity_mean={sum(specificities)/classes}")

            # confusion matrixby max prob + false_imgs_top-N_loss,
            confusion, confusion_cnt = metrics.get_confusion(valid_loader.dataset.path_list, loss_all)
            print(f"confusion_cnt={confusion_cnt}")
            metrics.export_confusion(confusion, args.results)
            break
        
        else:
            # export worst
            metrics.export_lowest_conf(valid_loader.dataset.path_list, args.results)
            break

# save prediction results
df = pd.DataFrame({
    "data": valid_loader.dataset.path_list,
    "label": valid_label, 
    "pred_probs_all": map(tuple,pred_probs_all),
    "pred_cls_all": pred_probs_all.max(axis=1),
})
df.to_csv(os.path.join(args.results, f'pred_{args.mode}.csv'), index=False)

# plot results
if args.mode=='train':
    # loss, f1, map x ep
    data_sub1 = [history.train_loss, history.valid_loss]
    data_sub2 = [history.train_f1, history.valid_f1]
    data_sub3 = [history.train_map, history.valid_map]
    utils.row_plot_1d( [data_sub1, data_sub2, data_sub3], ['epoch']*3, ['loss','f1','mAP'], \
        [['train','valid'] for i in range(3)], os.path.join(args.results, "curve_loss_f1_map.jpg") )

elif args.mode=='valid':
    precision_list, recall_list, f1_list, threshold_list = utils.get_prf_pr_data(valid_label, pred_probs_all) # p,r,f,t | cls
    # first plot: p/r/f curve per class
    prf_subplots_x = [ [threshold_list[i]]*3 for i in range(classes) ]
    prf_subplots_y = [ [precision_list[i], recall_list[i], f1_list[i]] for i in range(classes) ]    
    utils.row_plot_2d(prf_subplots_x, prf_subplots_y, ['threshold']*classes, ['']*classes, [['precision','recall','f1'] for _ in range(classes)], \
        os.path.join(args.results, "curve_prf.jpg") )
    # second_plot: p-r curve per class
    pr_subplots_x = [ [recall_list[i]] for i in range(classes) ]
    pr_subplots_y = [ [precision_list[i]] for i in range(classes) ]
    utils.row_plot_2d(pr_subplots_x, pr_subplots_y, ['recall']*classes, ['precision']*classes, [['.'] for _ in range(classes)], \
        os.path.join(args.results, "curve_pr.jpg") )

    # third plot: roc curve
    roc_subplots_x, roc_subplots_y = utils.get_roc_data(valid_label, pred_probs_all)
    utils.row_plot_2d(roc_subplots_x, roc_subplots_y, ['fpr']*classes, ['tpr']*classes, [['.'] for _ in range(classes)], \
        os.path.join(args.results, "curve_roc.jpg") )

print("successfully finished :D")