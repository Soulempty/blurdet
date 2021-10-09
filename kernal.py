import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def conv2d(x,k):
    ksize = k.size(-1)
    padding = ksize//2
    layer = nn.Conv2d(1, 1, ksize, 1, padding=padding,padding_mode='reflect',bias=False)
    layer.weight = torch.nn.Parameter(k)
    layer.weight.requires_grad=False
    out = layer(x)
    return out

def gaussion(k=3,sigma=5):
    ct = k//2
    kernal = np.zeros((k,k))
    for r in range(k):
        for c in range(k):
            kernal[r,c] = np.exp(((r-ct)**2+(c-ct)**2)/(-2*sigma**2))
    kernal /= kernal.sum()
    return torch.FloatTensor(kernal).view(1,1,k,k)

def laplacian(style=0):
    ops = [[[0,1,0],[1,-4,1],[0,1,0]],[[1,1,1],[1,-8,1],[1,1,1]]]
    op = ops[style]
    return torch.FloatTensor(op).view(1,1,3,3)

def togray(x):
    dist = x
    if len(x.size()) == 3:
        dist = (x[:,:,0]*0.1140 + x[:,:,1]*0.5870 + x[:,:,2]*0.2989) 
    return dist

def gridImg(img_tensor,h_num=11,w_num=7):
    assert len(img_tensor.size()) == 2
    num = h_num*w_num
    vblocks = torch.chunk(img_tensor, h_num, dim=0)
    hstrip = torch.cat(vblocks,-1)
    hblocks = torch.chunk(hstrip, num, dim=1)
    batch = torch.stack(hblocks,-1) # h x w x num
    data = batch.permute(2,0,1).unsqueeze(1) 
    return data



