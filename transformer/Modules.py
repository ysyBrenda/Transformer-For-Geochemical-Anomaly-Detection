import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import draw_atten_func
__author__ = "Yu-Hsiang Huang"

class ScaledDotProductAttention(nn.Module):   #todo: add 可视化
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature   #d^1/2=8
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        # starmen=torch.cuda.memory_allocated(device=0)

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        # endmen=torch.cuda.memory_allocated(device=0)
        # uesd=endmen-starmen
        # print(f"the used memory:{uesd/1024/1024}  MB")

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)  #todo  bug

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        # Save_atten_func(q,k,output,attn,v)
        # attn=v   # todo 方便调试，记得删除

        return output, attn




class tracer:   #装饰器 计数，为了知道是第几次调用=哪一层atten
    def __init__(self, func):
        self.calls = 0
        self.func = func
    def __call__(self, *args, **keys):
        self.calls += 1
        # print('call %s to %s' % (self.calls, self.func.__name__))
        self.func(*args, **keys)

@tracer
def Save_atten_func(q,k,output,attn,v):
    # if Save_atten_func.calls % 6==0 :
    # # print(attn.cpu().numpy(),attn.size())
    #     np.save('attentest/1/atten_point'+str( np.fix(Save_atten_func.calls /6)), attn.cpu().numpy())
    print('the atten 次数=',Save_atten_func.calls)
    if Save_atten_func.calls % 6 == 0:
        np.save('attentest/atten_point' + str(np.fix(Save_atten_func.calls / 6)), attn.cpu().numpy())
    elif Save_atten_func.calls % 6==5 :
        np.save('attentest/atten5_point'+str( np.fix(Save_atten_func.calls /6)), attn.cpu().numpy())
    elif Save_atten_func.calls % 6== 4:
        np.save('attentest/atten4_point' + str(np.fix(Save_atten_func.calls / 6)), attn.cpu().numpy())
    elif Save_atten_func.calls % 6 == 3:
        np.save('attentest/atten3_point' + str(np.fix(Save_atten_func.calls / 6)), attn.cpu().numpy())
    elif Save_atten_func.calls % 6 == 2:
        np.save('attentest/atten2_point' + str(np.fix(Save_atten_func.calls / 6)), attn.cpu().numpy())
    elif Save_atten_func.calls % 6 == 1:
        np.save('attentest/atten1_point' + str(np.fix(Save_atten_func.calls / 6)), attn.cpu().numpy())


