import os
import sys
import os
import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.join("ChamferDistancePytorch"))

from chamfer3D import dist_chamfer_3D

# Use Chamfer Loss which will be slower but more consistant with pretrain.

class ChamferLoss(nn.Module):
    def __init__(self):
        super(ChamferLoss,self).__init__()

    def forward(self,state_x,target):
        s = state_x.shape
        cham_loss = dist_chamfer_3D.chamfer_3DDist()
        dist1, dist2, _, _ = cham_loss(target.view(1,s[0],s[1]), state_x.view(1,s[0],s[1]))
        loss = (torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)) / 2
        loss = loss.mean()*10
        return loss