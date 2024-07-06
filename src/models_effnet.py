import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import numpy as np
import math
from einops import rearrange
from torchvision.models import EfficientNet_V2_S_Weights
import timm

class EfficientNet_V2(nn.Module):
    def __init__(self, n_out):
        super(EfficientNet_V2, self).__init__()
        # Define model
	# ここの引数を変えるだけで別のモデルも使える！
        self.effnet = timm.create_model('efficientnetv2_s', pretrained=False, num_classes=n_out) 

    def forward(self, x):
        return self.effnet(x)