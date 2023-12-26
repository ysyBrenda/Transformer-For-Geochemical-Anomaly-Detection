'''
calculate anomaly score,  PLot, and Calculate AUC (valuation)
author: ysy Brenda
'''
import numpy
import torch

from pylab import *
import pandas as pd
from scipy.interpolate import griddata
import random

def calculate(Pred,Trg,epoch_i=0,head='',raw_data='',Au_data='data/Au.csv'):

    if isinstance(Trg,torch.Tensor):
        gold = Trg.numpy()   #goal data
        pred=Pred.numpy()    # pred data
    else:
        gold = Trg # goal data
        pred = Pred # pred data
    gold=gold[1:,2:]
    pred=pred[1:,:]
      #---------calculate anomaly score（Err)---------
    Err = np.linalg.norm((gold - pred), ord=2,axis=1)
    #save
    np.savetxt('./output/Err.csv', Err, delimiter=",", fmt="%.8f")

    #------------read raw data-------------
    data = pd.read_csv(raw_data, header=None)
    data = data.values
    pred_x = data[:, 0]
    pred_y = data[:, 1]
    data = pd.read_csv(Au_data, header=None)
    data = data.values
    Au_x = data[:, 2]  #todo  the coordinates of known gold site
    Au_y = data[:, 3]  #todo

    LinX=np.linspace(min(pred_x),max(pred_x),200)
    LinY = np.linspace(min(pred_y), max(pred_y), 200)
    X_mg,Y_mg=np.meshgrid(LinX,LinY)

    # ------------AUC-------------
    # 1. Positive sample：interp to obtain mineralized sample's anomaly score.
    positive_err=griddata((pred_x,pred_y),Err,(Au_x,Au_y),method='linear')   # positive:Au_Err
    # print(positive_err)

    # 2. Negative sample : 5km away from the mineralized samples
    S=numpy.zeros((len(Au_x),len(pred_x)))
    for i in range(0,len(Au_x)):
        for j in range(0,len(pred_x)):
            S[i,j]=math.sqrt((Au_x[i]-pred_x[j])**2+(Au_y[i]-pred_y[j])**2)
    L2=sum(S<5000,0)
    m=np.where(L2==0)
    negative_err = Err[m]

    # 3. calculate Auc,save PN samples.
    # select negative samples(the same number as the positive sample), and loop 100 times
    AUC = []
    N=[]
    for i in range(0, 100):
        random.seed(i)
        n_err_i = random.sample(list(negative_err), size(positive_err))
        # print(n_err_i)
        auc = computeAUC(positive_err, n_err_i)
        AUC.append(auc)
        N=np.append(N,n_err_i,axis=0)
    AUC_mean = np.mean(np.array(AUC))

    np.savetxt('./output/negative_err.csv', N, delimiter=",", fmt="%.8f")
    np.savetxt('./output/positive_err.csv', positive_err, delimiter=",", fmt="%.8f")


    # ------------draw-------------
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
    plt.savefig('output/'+ head+'/draw_epoch' + str(epoch_i) + '_' + str(round(AUC_mean, 4)) + '.jpg')  # ,dpi=600
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
    trg_file = './output/trg.csv'
    pred_file = './output/pred.csv'
    Trg=np.loadtxt(trg_file, delimiter=",")
    Pred=np.loadtxt(pred_file, delimiter=",")
    calculate(Pred, Trg, epoch_i=0,raw_data='data/pos_feature.csv',Au_data='data/Au.csv')
if __name__ == '__main__':
    main()
