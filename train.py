import torch
import torch.nn as nn

class loss(nn.Module): 
    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCELoss() 
        
    def forward(self, result, _label):
        loss = {}

        _label = _label.float()

        # triplet = result["triplet_margin"] # loss6
        att = result['frame']
        t = att.size(1)      
        anomaly = torch.topk(att, t//16 + 1, dim=-1)[0].mean(-1)
        anomaly_loss = self.bce(anomaly, _label)
        cost = anomaly_loss
        loss['total_loss'] = cost
        return cost, loss

def train(net, normal_loader, abnormal_loader, optimizer, criterion):
    
    net.train() # 设置为训练模式
    net.flag = "Train"
    ninput, nlabel = next(normal_loader)
    ainput, alabel = next(abnormal_loader) 
    _data = torch.cat((ninput, ainput), 0) 
    _label = torch.cat((nlabel, alabel), 0)
    _data = _data.cuda()
    _label = _label.cuda()
    predict = net(_data)
    cost, loss = criterion(predict, _label)
    optimizer.zero_grad()
    cost.backward()
    
    optimizer.step()
    return cost