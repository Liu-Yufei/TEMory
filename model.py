import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from memory import Memory_Unit
from translayer import Transformer
class Temporal(Module):
    def __init__(self, input_size, out_size):
        super(Temporal, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=out_size, kernel_size=3,
                    stride=1, padding=1),
            nn.ReLU(),
        )
    def forward(self, x):  
        x = x.permute(0, 2, 1)
        x = self.conv_1(x)
        x = x.permute(0, 2, 1)
        return x

class CLS_head(Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim,128), nn.ReLU(), nn.Linear(128,out_dim), nn.Sigmoid())
    def forward(self, x):
        return self.mlp(x)

class TEMory(Module):
    def __init__(self, input_size, flag):
        super().__init__()
        self.flag = flag
        self.embedding = Temporal(input_size,512)
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=1,
                            bidirectional=True, batch_first=True)
        self.cls_head = CLS_head(1024, 1)
        self.Amemory = Memory_Unit(nums=512, dim=512)
        self.Nmemory = Memory_Unit(nums=512, dim=512)
        self.selfatt = Transformer(512, 2, 4, 128, 512, dropout = 0.5)
        self.encoder_mu = nn.Sequential(nn.Linear(512, 512))
        self.encoder_var = nn.Sequential(nn.Linear(512, 512))
        self.relu = nn.ReLU()
    def _reparameterize(self, mu, logvar):
        std = torch.exp(logvar).sqrt()
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def latent_loss(self, mu, var):
        kl_loss = torch.mean(-0.5 * torch.sum(1 + var - mu ** 2 - var.exp(), dim = 1))
        return kl_loss

    def forward(self, x):
        if len(x.size()) == 4:
            b, n, t, d = x.size()
            x = x.reshape(b * n, t, d)
        else:
            b, t, d = x.size()
            n = 1
        x = self.embedding(x)
        x,_ = self.lstm(x)
        x = self.selfatt(x)
        if self.flag == "Train":
            N_x = x[:b*n//2]
            A_x = x[b*n//2:]
            A_aug = self.Amemory(A_x,A_x)
            N_Aaug = self.Nmemory(A_x,A_x)
            A_Naug = self.Amemory(N_x,N_x)
            N_aug = self.Nmemory(N_x,N_x)
            N_aug_mu = self.encoder_mu(N_aug)
            N_aug_var = self.encoder_var(N_aug)
            N_aug_new = self._reparameterize(N_aug_mu, N_aug_var)
            A_aug_new = self.encoder_mu(A_aug)
            A_Naug = self.encoder_mu(A_Naug) 
            N_Aaug = self.encoder_mu(N_Aaug)
            x = torch.cat((x, torch.cat([N_aug_new + A_Naug, A_aug_new + N_Aaug], dim=0)), dim=-1)
            pre_att = self.cls_head(x).reshape((b, n, -1)).mean(1)
            return {
                    "frame": pre_att,
                }
        else:           
            A_aug = self.Amemory(x,x)
            N_aug = self.Nmemory(x,x)  

            A_aug = self.encoder_mu(A_aug)
            N_aug = self.encoder_mu(N_aug)

            x = torch.cat([x, A_aug + N_aug], dim=-1)
            pre_att = self.cls_head(x).reshape((b, n, -1)).mean(1)
            return {"frame": pre_att}
if __name__ == "__main__":
    m = TEMory(input_size = 1024, flag = "Train", a_nums = 60, n_nums = 60).cuda()
    src = torch.rand(100, 32, 1024).cuda()
    out = m(src)["frame"]
    
    print(out.size())
    
