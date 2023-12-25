#=========画图，AUC计算=============
import numpy
import torch
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import pandas as pd
from scipy.interpolate import griddata
import random

def PlotAndAUC(Pred,Trg,epoch_i=0,is1585=1,head=''):
    # print('this is computAUC func')
    # pred tensor 1586,38     trg 1586,40
    if is1585 == 1:
        gold = Trg.numpy()   #goal data
        pred=Pred.numpy()    # pred data
    elif isinstance(Trg,torch.Tensor):  #判断数据类型
        gold = Trg.numpy()   #goal data
        pred=Pred.numpy()    # pred data
    else:
        gold = Trg # goal data
        pred = Pred # pred data
    gold=gold[1:,2:]
    pred=pred[1:,:]
      #---------计算anomaly score（Err)---------
    Err = np.linalg.norm((gold - pred), ord=2,axis=1)
    #save
    np.savetxt('./Err0331nomask.csv', Err, delimiter=",", fmt="%.8f")

    #------------read xy data-------------
    if is1585==1:
        data = pd.read_csv("data/pos_feature.csv", header=None)
    else:
        data = pd.read_csv("data/Interp_data_100.csv", header=None)
    data = data.values
    pred_x = data[:, 0]
    pred_y = data[:, 1]
    data = pd.read_csv("data/Au.csv", header=None)
    data = data.values
    Au_x = data[:, 2]
    Au_y = data[:, 3]

    LinX=np.linspace(min(pred_x),max(pred_x),200)
    LinY = np.linspace(min(pred_y), max(pred_y), 200)
    X_mg,Y_mg=np.meshgrid(LinX,LinY)

    # ------------AUC-------------
        # 1. 准备正样本：二维插值得到金矿点的anomaly score.
    positive_err=griddata((pred_x,pred_y),Err,(Au_x,Au_y),method='linear')   # positive:Au_Err
    # print(positive_err)

       # 2. 准备负样本：
    S=numpy.zeros((len(Au_x),len(pred_x)))
    for i in range(0,len(Au_x)):
        for j in range(0,len(pred_x)):
            S[i,j]=math.sqrt((Au_x[i]-pred_x[j])**2+(Au_y[i]-pred_y[j])**2)
      # 排除距离金矿点5千米以内的sample。
    L2=sum(S<5000,0)
    m=np.where(L2==0)
    negative_err = Err[m] # negative:sample


       # 3. 计算Auc :从负样本中挑选27个，计算AUC。循环100次
    # AUC=[]
    # for i in range(0,100):
    #     random.seed(i)
    #     n_err_i=random.sample(list(negative_err),size(positive_err))
    #     # print(n_err_i)
    #     auc =computeAUC(positive_err,n_err_i)
    #     AUC.append(auc)
    #     # print(auc)
    # AUC_mean=np.mean(np.array(AUC))
    # print('-----the AUC_MEAN is: {:.4f}'.format(AUC_mean))
    #

    # 3-1. 计算Auc并输出正负样本，用于画ROC :从负样本中挑选27个，计算AUC。循环100次
    AUC = []
    N=[]
    N = np.append(N, positive_err, axis=0)
    for i in range(0, 100):
        random.seed(i)
        n_err_i = random.sample(list(negative_err), size(positive_err))
        # print(n_err_i)
        auc = computeAUC(positive_err, n_err_i)
        AUC.append(auc)
        N=np.append(N,n_err_i,axis=0)
    AUC_mean = np.mean(np.array(AUC))
    print('-----the AUC_MEAN is: {:.4f}'.format(AUC_mean))
    # np.savetxt('./output/best_0887.csv', N, delimiter=",", fmt="%.8f")
    # np.savetxt('./output_adj/adj1000Sample0320P.csv', positive_err, delimiter=",", fmt="%.8f")


    # ------------draw-------------
    if 1:
    # if AUC_mean>0.80:
        plt.figure(figsize=(18,5))

        vq1=griddata((pred_x,pred_y),gold[:,3],(X_mg,Y_mg),method='linear')    #linear
        plt.subplot(1,3,1)
        pcolor(X_mg, Y_mg, vq1, cmap='jet')
        MAX,MIN=gold[:,3].max(),gold[:,3].min()
        clim(vmin=MIN, vmax=MAX)
        colorbar()
        scatter(Au_x,Au_y,s=15, c='m', marker='x',alpha=0.5,linewidths=1.0)
        title('goldAu')

        plt.subplot(1, 3, 2)
        vq1 = griddata((pred_x, pred_y), pred[:, 3], (X_mg, Y_mg), method='linear')  # linear
        pcolor(X_mg, Y_mg, vq1, cmap='jet')
        # MAX, MIN = pred[:, 3].max(), pred[:, 3].min()
        clim(vmin=MIN, vmax=MAX)
        colorbar()
        scatter(Au_x, Au_y, s=15,c='m', marker='x', alpha=0.5, linewidths=1.0)
        title('predAu')

        plt.subplot(1, 3, 3)
        vq3 = griddata((pred_x, pred_y), Err, (X_mg, Y_mg), method='linear')  # linear
        pcolor(X_mg, Y_mg, vq3, cmap='jet')
        MAX, MIN = Err.max(), Err.min()
        clim(vmin=MIN, vmax=MAX)
        colorbar()
        scatter(Au_x, Au_y,s=15 ,c='m', marker='x', linewidths=1.0)
        title('Err')
        #---save figure ----
        # plt.savefig('output/h1L2_hid8_d8/INTERP1epoch'+str(epoch_i)+'_'+str(round(AUC_mean,4))+'_1028.jpg')   #,dpi=600
        # plt.savefig('output/h1L1_hid4_d4/draw/epoch' + str(epoch_i) + '_' + str(round(AUC_mean, 4)) + '.jpg')  # ,dpi=600
        plt.savefig('output_remask/'+ head+'/draw_epoch' + str(epoch_i) + '_' + str(round(AUC_mean, 4)) + '.jpg')  # ,dpi=600
        # plt.show()
        plt.close()

    return AUC_mean


def computeAUC(x,y):
    # xi:positive sample
    # yi:negative sample
    q,p=len(x),len(y)
    sum=0
    for i in range(0,len(x)):
        sum_i=0
        for j in range(0,len(y)):
            if (x[i]-y[j])>10e-6:
                a=1
            elif abs(x[i]-y[j])<10e-6:
                a=0.5
            else:
                a=0
            sum_i=sum_i+a
        sum=sum+sum_i
    AUC=sum/(q*p)
    return AUC


def main():
    trg_file = './output/trg_1202_.csv'
    pred_file = './output/pred_1202_.csv'
    Trg=np.loadtxt(trg_file, delimiter=",")
    Pred=np.loadtxt(pred_file, delimiter=",")
    PlotAndAUC(Pred, Trg, epoch_i=0,is1585=0)
if __name__ == '__main__':
    main()
