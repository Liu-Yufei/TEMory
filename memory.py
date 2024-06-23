import math
import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from einops import rearrange

class Memory_Unit(Module):
    def __init__(self, nums, dim, heads = 8, dim_head = 64):
        super().__init__()
        self.dim = dim
        self.nums = nums
        self.memory_block = nn.Parameter(torch.empty(nums, dim))
        self.sig = nn.Sigmoid()
        self.reset_parameters()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head *  heads
        self.attend = nn.Softmax(dim = -1)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(dim, inner_dim, bias = False)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.memory_block.size(1))
        self.memory_block.data.uniform_(-stdv, stdv)
        if self.memory_block is not None:
            self.memory_block.data.uniform_(-stdv, stdv)
    
       
    def forward(self, x1,x2):  ####data size---> B,T,D       K,V size--->K,D
        q =rearrange( self.to_q(x1),'b n (h d) -> b h n d',h = self.heads)
        k = rearrange(self.to_k(x2),'b n (h d) -> b h n d',h = self.heads)
        b,h,n,d = q.size()
        v = rearrange( self.to_v(self.memory_block),'k (h n d) -> k h n d',h = self.heads,n = n)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        memory_bank_att = torch.matmul(attn, v)
        memory_bank_att =  torch.reshape(memory_bank_att, (memory_bank_att.size(0), -1))

        attention = self.sig(torch.einsum('btd,kd->btk', x1, memory_bank_att) / (self.dim**0.5))   #### Att---> B,T,K # 计算S
        augment = torch.einsum('btk,kd->btd', attention, self.memory_block)                   #### feature_aug B,T,D # 计算M_aug
        return augment