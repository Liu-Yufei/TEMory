import torch
from options import *
from model import *
import numpy as np
from dataset_loader import *
from sklearn.metrics import roc_curve,auc,precision_recall_curve
import warnings
warnings.filterwarnings("ignore")

def test(net, test_loader,device):
    with torch.no_grad():
        net.eval()
        net.flag = "Test"
        frame_gt = list(np.load("/home/lyf/code/File/gt-colon.npy"))
        frame_predict = torch.zeros(0).to(device)
        gt = {}
        pred = {}
        count = 0
        TP_num = 0
        TN_num = 0
        FP_num = 0
        FN_num = 0
        for i,(features, filename) in enumerate(test_loader):
            features = features.to(device)
            filename = filename[0].split('.npy')[0].split('/')[-1]
            filename = filename.split('.')[0]
            filename = filename.split('_')[0]
            gt[filename] = frame_gt[count:count + len(features.mean(0).cpu().numpy()) * 16]
            count = count + features.shape[1] * 16
            features = features.squeeze(2)
            pred_temp = torch.zeros(0).to(device)
            len_num_seg = features.shape[1]
            for j in range(features.shape[1]//32+1):
                start_idx = j * 32
                end_idx = (j + 1)*32
                input_tmp = features[:, start_idx:end_idx, :]
                if input_tmp.shape[1] < 32:
                    for last in range((32-input_tmp.shape[1])):
                        input_tmp = torch.cat((input_tmp, features[:, -1, :].unsqueeze(1)), dim=1)
                predict = net(input_tmp)
                logits = torch.mean(predict['frame'], 0)
                sig = logits
                pred_temp = torch.cat((pred_temp, sig))
            pred[filename] = np.repeat(pred_temp[:len_num_seg].cpu().detach().numpy(), 16)
            for j in range(len(pred[filename])):
                if pred[filename][j] > 0.5 and gt[filename][j] == 0:
                    FP_num += 1
                elif pred[filename][j] <= 0.5 and gt[filename][j] == 1:
                    FN_num += 1
                elif pred[filename][j] > 0.5 and gt[filename][j] == 1:
                    TP_num += 1
                elif pred[filename][j] <= 0.5 and gt[filename][j] == 0:
                    TN_num += 1
            frame_predict = torch.cat((frame_predict, pred_temp[:len_num_seg]))
        frame_predict = list(frame_predict.cpu().detach().numpy())
        frame_predict = np.repeat(np.array(frame_predict), 16)
        fpr, tpr, _ = roc_curve(list(frame_gt), frame_predict)
        rec_auc = auc(fpr, tpr) 
        precision, recall, _ = precision_recall_curve(list(frame_gt), frame_predict)
        pr_auc = auc(recall, precision)
        accuracy = (TP_num + TN_num) / (TP_num + TN_num + FP_num + FN_num)
        return rec_auc, pr_auc, accuracy