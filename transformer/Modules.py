import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import draw_atten_func
__author__ = "Yu-Hsiang Huang"

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        # starmen=torch.cuda.memory_allocated(device=0)

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        # endmen=torch.cuda.memory_allocated(device=0)
        # uesd=endmen-starmen
        # print(f"the used memory:{uesd/1024/1024}  MB")

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn



