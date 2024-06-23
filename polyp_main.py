import pdb
import numpy as np
from torch.utils.data import DataLoader
import utils
from options import *
from config import *
from train import train
from train import loss
from polyp_test import *
from model import *
import os
from dataset_loader import *
from tqdm import tqdm
import matplotlib.pyplot as plt
if __name__ == "__main__":
    # Initialize
    test_info = {"step": [], "auc": [],"ap":[],"loss":[], "accuracy": []}
    best_AUC = 0
    best_AUC_loss = 0
    best_AUC_step = 0
    best_AP = 0
    best_AP_loss = 0
    best_AP_step = 0
    best_Accuracy = 0
    best_Accuracy_loss = 0
    best_Accuracy_step = 0
    test_info["step"].append(0)
    test_info["auc"].append(0)
    test_info["ap"].append(0)
    test_info["loss"].append(0)
    test_info["accuracy"].append(0)
    costs = [0]
    args = parse_args()
    if args.debug:
        pdb.set_trace()
    config = Config(args)
    worker_init_fn = None
    gpus = [0]
    torch.cuda.set_device('cuda:0')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if config.seed >= 0:
        utils.set_seed(config.seed)
        worker_init_fn = np.random.seed(config.seed) # reproducible data loading 
    save_path = os.path.join(config.output_path, config.exp_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Load Data
    ntrain_dataset =polyp_Dataset(
        root_dir = config.root_dir, 
        mode = "Train", 
        num_segments=config.num_segments,
        is_normal = True
    )
    normal_train_loader = DataLoader(
        ntrain_dataset,
        batch_size=config.batch_size, 
        shuffle=True,
        pin_memory=False, drop_last=True,
        worker_init_fn = worker_init_fn
    )

    atrain_dataset = polyp_Dataset(
        root_dir = config.root_dir, 
        mode = "Train", 
        num_segments = config.num_segments, 
        is_normal = False)
    abnormal_train_loader = DataLoader(
        atrain_dataset,
        batch_size=config.batch_size, 
        shuffle=True,
        pin_memory=False, drop_last=True,
        worker_init_fn = worker_init_fn
    )

    t_dataset = polyp_Dataset(
        root_dir = config.root_dir, 
        mode = "Test", 
        num_segments = config.num_segments, 
    )
    test_loader = DataLoader(
        t_dataset,
        batch_size=1, 
        shuffle=False,
        pin_memory=False,
        worker_init_fn = worker_init_fn
    )

    # Load Net
    net = TEMory(config.len_feature, flag = "Train")
    net = net.to(device)
    criterion = loss()
    optimizer = torch.optim.Adam(net.parameters(), lr = config.lr[0], betas = (0.9, 0.999), weight_decay = 0.00005)
    
    # Train and Test
    with tqdm(
            range(1, config.num_iters + 1),
            total = config.num_iters,
            dynamic_ncols = True # 进度条
        )as t:
        for step in t:
            if step > 1 and config.lr[step - 1] != config.lr[step - 2]:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = config.lr[step - 1]
            if (step - 1) % len(normal_train_loader) == 0:
                normal_loader_iter = iter(normal_train_loader)
            if (step - 1) % len(abnormal_train_loader) == 0:
                abnormal_loader_iter = iter(abnormal_train_loader)
            cost= train(net, normal_loader_iter,abnormal_loader_iter, optimizer, criterion) # 训练一个batch
            
            if step % 5 == 0 and step > 10:
                auc, ap, accuracy= test(net,test_loader,device)
                
                test_info["step"].append(step)
                test_info["auc"].append(auc)
                test_info["ap"].append(ap)
                test_info["loss"].append(cost.item())
                test_info["accuracy"].append(accuracy)
                costs.append(cost.item())
                if test_info["auc"][-1] > best_AUC:
                    best_AUC = test_info["auc"][-1]
                    best_AUC_loss = test_info["loss"][-1]
                    best_AUC_step = step
                    torch.save(net.state_dict(), os.path.join(save_path, f'{config.exp_name}_Best_AUC.pkl'))
                    utils.save_best_record(test_info, os.path.join(save_path,f'{config.exp_name}_Best_AUC_results.txt'))
                if test_info["ap"][-1] > best_AP:
                    best_AP = test_info["ap"][-1]
                    best_AP_loss = test_info["loss"][-1]
                    best_AP_step = step
                    utils.save_best_record(test_info, os.path.join(save_path, f'{config.exp_name}_Best_AP_results.txt'))
                    torch.save(net.state_dict(), os.path.join(save_path, f'{config.exp_name}_Best_AP.pkl'))
                if test_info["accuracy"][-1] > best_Accuracy:
                    best_Accuracy = test_info["accuracy"][-1]
                    best_Accuracy_loss = test_info["loss"][-1]
                    best_Accuracy_step = step
                    utils.save_best_record(test_info, os.path.join(save_path, f'{config.exp_name}_Best_Accuracy_results.txt'))
                    torch.save(net.state_dict(), os.path.join(save_path, f'{config.exp_name}_Best_Accuracy.pkl'))
            t.set_postfix({'loss':cost.item(),'ap':test_info["ap"][-1],'auc':test_info["auc"][-1],'accuracy':test_info["accuracy"][-1]}, refresh = True)
            t.update(1)
    
    # Save Test Result
    fo = open("/home/lyf/code/File/experiment/20240214.csv", "a")
    fo.write("{},".format(config.exp_name))
    fo.write("{:.4f},".format(best_AUC))
    fo.write("{:.4f},".format(best_AUC_loss))
    fo.write("{},".format(best_AUC_step))
    fo.write("{:.4f},".format(best_AP))
    fo.write("{:.4f},".format(best_AP_loss))
    fo.write("{}\n".format(best_AP_step))
    fo.write("{:.4f},".format(best_Accuracy))
    fo.write("{:.4f},".format(best_Accuracy_loss))
    fo.write("{}\n".format(best_Accuracy_step))
    fo.close()

    plt.plot(test_info["step"], costs,label = 'loss')
    plt.plot(test_info['step'],test_info['ap'],label = 'ap')
    plt.plot(test_info['step'],test_info['accuracy'],label = 'ap')
    plt.title('Example Plot')
    plt.xlabel('step')
    plt.ylabel('%')
    
    plt.savefig(os.path.join(save_path, f'{config.exp_name}_figure.png'))
