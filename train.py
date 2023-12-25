# ========= Yuhsuyan 0608 masked version ===========  #run in env:torch1.3.0   D=38.  add contrast loss
'''
This script handles the training process.
'''
isCustomData = 1
import draw_atten_func
#  1：输入我们的自定义数据  0"：原始demo
import argparse
import math
import time
import dill as pickle
import numpy
from tqdm import tqdm
import numpy as np
import random
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import OUTPUTeval
# from torchtext.datasets import TranslationDataset

import transformer.Constants as Constants
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim
import torch.utils.data as Data
# import scipy.io as io
from torch.utils.data import Dataset, DataLoader, TensorDataset

isAdjmask = 0
# isRandMask = 1
# isContrastLoss=1

__author__ = "Yu-Hsiang Huang"


# to use tensorboard，在terminal中输入：
# $ tensorboard --logdir=output --port 6006
# if【was already in use】：lsof -i:6006， kill -9 PID
def adj_mask3(x):
    import pandas as pd
    len = x.size(0)  # len=1585
    adj = pd.read_csv("./data/adj1585_R10000.csv", header=None)
    adj = adj.values
    # for i in range(len):
    #     index = np.where(adj[i, :] == 0)
    #     x[i, index, :] = 1  #pad=1 use 1 padding

    index = np.where(adj == 0)

    batch_ind = torch.arange(len).unsqueeze(-1)
    x = x[0:len, index]
    return x


def mask_data(data, device, unmask):
    # unmask=500
    b, len, d = data.size()
    # unmask=int(1585*0.5)
    unmask = random.randint(int(1585 * 0.8), 1585)  # default
    shuffle_indices = torch.rand(b, len, device=device).argsort()  # b,
    unmask_ind, mask_ind = shuffle_indices[:, :unmask], shuffle_indices[:, unmask:]
    # 对应 batch 维度的索引：(b,1)
    batch_ind = torch.arange(b, device=device).unsqueeze(-1)
    # 利用先前生成的索引对 patches 进行采样，分为 mask 和 unmasked 两组
    # mask_patches, unmask_patches = data[batch_ind, mask_ind], data[batch_ind, unmask_ind]
    # data = unmask_patches
    return data[batch_ind, unmask_ind]


def adj_mask(x):
    import pandas as pd
    len = x.size(0)  # len=1585
    adj = pd.read_csv("./data/adj1585_R50000.csv", header=None)
    adj = adj.values
    for i in range(len):
        index = np.where(adj[i, :] == 0)
        x[i, index, :] = 1  # pad=1 use 1 padding
    return x


def adj_mask2(data, device):
    b, len, d = data.size()
    index = torch.where(data.sum(dim=1) == 1)[0]
    batch_ind = torch.arange(b, device=device).unsqueeze(-1)
    data = data[batch_ind, data.sum(dim=2) != 0]
    #
    index = torch.where(data[:, :, 3] != 0)
    index = index.long()  # bool 转01=

    return data[batch_ind, index]
    # return data


def patch_src(src, pad_idx):
    src = src.transpose(0, 1)  # type(src)=<class 'torch.Tensor'>
    return src


def patch_trg(trg, pad_idx):
    trg = trg.transpose(0, 1)
    trg, gold = trg[:, :-1], trg[:, 1:].contiguous().view(-1)  # trg取最后1-（n-1），gold取trg的拉直向量
    return trg, gold


def train_epoch(model, training_data, optimizer, opt, device, smoothing):
    ''' Epoch operation in training phase'''
    model.train()
    total_loss = 0

    desc = '  - (Training)   '
    for batch in tqdm(training_data, mininterval=2, desc=desc, leave=False):  # todo
        # prepare data
        if opt.isContrastLoss:
            temp= batch[0].to(device)
            # b, len, d = temp.size()
            # a=torch.split(temp,1585,dim=1)
            a = torch.chunk(temp, 3, dim=1)
            src_seq=torch.cat([a[0],a[1],a[2]],0)
        else:
            src_seq = batch[0].to(device)  # batchsize在前

        gold = batch[1][:, 2:].unsqueeze(1)
        trg_seq, gold = map(lambda x: x.to(device), [batch[1].unsqueeze(1), gold.contiguous().view(-1)])  # 转置、拉直向量  # unsqueeze:将（b,40） 改为（b,1,40)
        if opt.isContrastLoss:
             trg_seq=torch.cat([trg_seq,trg_seq,trg_seq],0)

        # forward
        optimizer.zero_grad()
        pred, *_ = model(src_seq, trg_seq)  # （6912/9216,9516）   256,33   256,36xailkjib

        # =============  contrast loss==========================
        if opt.isContrastLoss:
            a = torch.chunk(pred, 3, dim=0)
            contras_loss = F.mse_loss(a[1].contiguous().view(-1), a[2].contiguous().view(-1), reduction='mean')
            loss = F.mse_loss(a[0].contiguous().view(-1), gold, reduction='mean') + opt.lambda_con * contras_loss
        else:
            loss = F.mse_loss(pred.contiguous().view(-1), gold, reduction='mean')  # reduction="mean"

        # backward and update parameters
        # =====loss==========
        # loss = F.mse_loss(pred.contiguous().view(-1), gold, reduction='mean')  # reduction="mean"
        loss.backward()
        optimizer.step_and_update_lr()
        # print(loss.item())
        # note keeping
        total_loss += loss.item()
        # tqdm.write(loss.item())

    print('total_train loss:{} '.format(total_loss))
    return loss


def eval_epoch(model, validation_data, device, opt):
    ''' Epoch operation in evaluation phase '''
    isContrastLoss1=0
    model.eval()
    total_loss = 0
    Trg_all = torch.zeros(1, 40)
    Pred_all = torch.zeros(1, 38)
    desc = '  - (Validation) '
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2, desc=desc, leave=False):

            if isContrastLoss1:
                temp = batch[0].to(device)
                a = torch.chunk(temp, 3, dim=1)
                src_seq = torch.cat([a[0], a[1], a[2]], 0)
            else:
                src_seq = batch[0].to(device)  # batchsize在前

            gold = batch[1][:, 2:].unsqueeze(1)
            trg_seq, gold = map(lambda x: x.to(device), [batch[1].unsqueeze(1), gold.contiguous().view(-1)])  # 转置、拉直向量  # unsqueeze:将（b,40） 改为（b,1,40)
            if isContrastLoss1:
                trg_seq = torch.cat([trg_seq, trg_seq, trg_seq], 0)

            # forward
            pred, *_ = model(src_seq, trg_seq)  # （6912/9216,9516）   256,33   256,36xailkjib

            # =============  contrast loss==========================
            if isContrastLoss1:
                a = torch.chunk(pred, 3, dim=0)
                contras_loss = F.mse_loss(a[1].contiguous().view(-1), a[2].contiguous().view(-1), reduction='mean')
                loss = F.mse_loss(a[0].contiguous().view(-1), gold, reduction='mean') + opt.lambda_con * contras_loss
            else:
                loss = F.mse_loss(pred.contiguous().view(-1), gold, reduction='mean')  # reduction="mean"

            if isContrastLoss1:
                Pred_all = torch.cat((Pred_all, a[0].cpu()), 0)
            else:
                Pred_all = torch.cat((Pred_all, pred.cpu()), 0)  # 0:按行拼接


            Trg_all = torch.cat((Trg_all, trg_seq.squeeze(1).cpu()), 0)
            # np.savetxt("Val_data.csv", np.column_stack((x, y, err_xy)), delimiter=",", header='gold,pred,err',
            #            fmt="%.8f")
            # note keeping
            total_loss += loss.item()
    print('total_val loss:{} '.format(total_loss))
    return loss, Pred_all, Trg_all


def train(model, training_data, validation_data, optimizer, device, opt):
    """ Start training """

    # Use tensorboard to plot curves, e.g. perplexity, accuracy, learning rate
    if opt.use_tb:  # O
        print("[Info] Use Tensorboard")
        from tensorboardX import SummaryWriter
        # from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=os.path.join(opt.output_dir, 'tensorboard' + opt.fileHead))

    log_train_file = os.path.join(opt.output_dir, opt.fileHead, 'train.log')
    log_valid_file = os.path.join(opt.output_dir, opt.fileHead, 'valid.log')

    print('[Info] Training performance will be written to file: {} and {}'.format(
        log_train_file, log_valid_file))

    with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
        log_tf.write('epoch,loss,lr\n')  # todo: add AUC
        log_vf.write('epoch,loss,lr,auc\n')

    def print_performances(header, loss, start_time, lr):
        print('  - {header:12} loss: {loss: 8.2e},  lr: {lr:8.2e}, ' \
              'elapse: {elapse:3.3f} min'.format(
            header=f"({header})", loss=loss,
            elapse=(time.time() - start_time) / 60, lr=lr))  # lr: {lr:8.5f}  5位小数

    Auces = []
    valid_losses = []
    bad_counter = 0
    best = 50000
    patience = 5  # 5
    for epoch_i in range(opt.epoch):  # 1个 epoch要算完所有数据。
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss = train_epoch(
            model, training_data, optimizer, opt, device, smoothing=opt.label_smoothing)  # todo
        # Current learning rate
        lr = optimizer._optimizer.param_groups[0]['lr']
        print_performances('Training', train_loss, start, lr)

        # start = time.time()
        valid_loss, Pred, Trg, *_ = eval_epoch(model, validation_data, device, opt)  # todo
        print_performances('Validation', valid_loss, start, lr)

        # =========画图，AUC计算============= #todo
        Auc_i = OUTPUTeval.PlotAndAUC(Pred, Trg, epoch_i,train_loss=train_loss,valid_loss=valid_loss,head=opt.fileHead)  # pred tensor 1586,38     trg 1586,40

        valid_losses += [valid_loss]
        Auces += [Auc_i]

        checkpoint = {'epoch': epoch_i, 'settings': opt, 'model': model.state_dict()}

        if opt.save_mode == 'all':
            # if epoch_i % 10 == 9:
            model_name = 'output/' + opt.fileHead + '/model/model_{epoch:d}_vloss_{vloss:.4f}_auc_{auc:.4f}.chkpt'.format(
                epoch=epoch_i, vloss=valid_loss, auc=Auc_i)
            torch.save(checkpoint, model_name)
        elif opt.save_mode == 'best':
            if Auc_i >= max(Auces) and Auc_i > 0.880:
                model_name = 'model_best_{epoch:d}_vloss_{vloss:.4f}_auc_{auc:.4f}.chkpt'.format(epoch=epoch_i,
                                                                                                 vloss=valid_loss,
                                                                                                 auc=Auc_i)
                # model_name = 'model_best.chkpt'  # 'model.chkpt'
                torch.save(checkpoint, os.path.join(opt.output_dir, opt.fileHead, model_name))
                # torch.save(checkpoint, model_name)
                # print(' - [Info] The checkpoint file has been updated.')

                # #draw atten
                # for l in range(0, len(dec_enc_attn_list)):  # n_Layer
                #     for i in range(0, 2):
                #         dec_enc_attn_list1 = dec_enc_attn_list[l].data
                #         draw_atten_func.draw(dec_enc_attn_list1[i].unsqueeze(0), save_file_L=str(l + 1), point=str(i),
                #                              head=opt.fileHead)

        with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
            log_tf.write('{epoch},{loss: 8.5f},{lr:8.2e},{auc:8.5f}\n'.format(
                epoch=epoch_i, loss=train_loss, lr=lr, auc=Auc_i))
            log_vf.write('{epoch},{loss: 8.9f},{lr:8.2e},{auc:8.5f}\n'.format(
                epoch=epoch_i, loss=valid_loss, lr=lr, auc=Auc_i))

        if opt.use_tb:
            tb_writer.add_scalars('loss', {'train': train_loss, 'val': valid_loss}, epoch_i)
            tb_writer.add_scalar('learning_rate', lr, epoch_i)
            tb_writer.add_scalar('auc', Auc_i, epoch_i)

        # print('【best AUC：】',max(Auces))
        print('   best AUC：{auc: 8.3f}, elapse: {elapse:3.3f} min'.format(auc=max(Auces),
                                                                          elapse=(time.time() - start) / 60, lr=lr))

        # auto break

        if valid_loss < best:
            best = valid_loss
            bad_counter = 0
        else:
            bad_counter += 1
        if bad_counter == patience:
            break

    log_opt_file = 'gridsearch_July.log'  # add
    with open(log_opt_file, 'a') as log_f:
        log_f.write(str(opt.fileHead) + '__{auc:8.5f}\n'.format(auc=max(Auces)))


def main():
    ''' 
    Usage:
    python train.py
                    -data_pkl ./data/pre_data.pkl -embs_share_weight -proj_share_weight -label_smoothing -output_dir output -warmup 128000 -epoch 150 -b 16 -use_tb -save_mode all
    '''

    parser = argparse.ArgumentParser()

    parser.add_argument('-data_pkl', default=None)  # all-in-1 data pickle or bpe field

    parser.add_argument('-train_path', default=None)  # bpe encoded data
    parser.add_argument('-val_path', default=None)  # bpe encoded data

    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=2048)

    parser.add_argument('-d_model', type=int, default=38)  # 38;8  #todo
    parser.add_argument('-d_inner_hid', type=int, default=2048)  # 64  #todo   )2048 但是太大?  256-1585-38 *  2048
    parser.add_argument('-d_k', type=int, default=38)  # 38;8       #64
    parser.add_argument('-d_v', type=int, default=38)  # 38;8     #64

    parser.add_argument('-n_head', type=int, default=1)
    parser.add_argument('-n_layers', type=int, default=1)  # 6
    parser.add_argument('-warmup', '--n_warmup_steps', type=int, default=4000)
    parser.add_argument('-lr_mul', type=float, default=2.0)  # 2.0  200比2效果好
    parser.add_argument('-seed', type=int, default=None)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')
    parser.add_argument('-scale_emb_or_prj', type=str, default='prj')

    parser.add_argument('-output_dir', type=str, default=None)
    parser.add_argument('-use_tb', action='store_true')
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-label_smoothing', action='store_true')
    parser.add_argument('-unmask', type=float, default=0.3) #add,  default input all  1500
    parser.add_argument('-l2', type=float, default=0.0)  # add,  weight_dacay
    parser.add_argument('-lambda_con', type=float, default=0.01)  # add,  contrast loss lambda
    parser.add_argument('-T', type=int, default=1)  # add, the times of mask
    parser.add_argument('-isContrastLoss', action='store_true')
    parser.add_argument('-isRandMask', action='store_true')

    opt = parser.parse_args()
    # # ++++++++++++++++
    opt.d_k = opt.d_model
    opt.d_v = opt.d_model
    # opt.seed=np.random.randint(1,10000,1)  #为了随机取seed加的
    # opt.seed=opt.seed[0]
    #  ++++++++++++++
    print(opt.seed)
    #
    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model  # 512

    # ------自动生成输出文件夹的名字----
    opt.fileHead = 'shuffle0723_T' + str(opt.T) + '_pad1_unmask' + str(opt.unmask) + '_h' + str(opt.n_head) + 'L' + str(
        opt.n_layers) + '_hid' + str(opt.d_inner_hid) + '_d' + str(opt.d_model) + '_b' + str(
        opt.batch_size) + '_warm' + str(opt.n_warmup_steps) + '_lrm' + str(opt.lr_mul) + '_seed' + \
                   str(opt.seed) + '_dr' + str(opt.dropout) +'_isCL'+str(opt.isContrastLoss)+ '_lamb'+str(opt.lambda_con) #+'_ismask'+str(opt.isRandMask)  # + '_l2'+str(opt.l2)
    if os.path.exists(os.path.join(opt.output_dir, opt.fileHead)):  # todo
        print('the output file is rewriting....', opt.fileHead)
    else:
        os.mkdir(os.path.join(opt.output_dir, opt.fileHead))
        # os.mkdir(os.patoin(opt.output_dir,opt.fileHead,'model'))
        print('The output filename is generated: ', opt.fileHead)

    # https://pytorch.org/docs/stable/notes/randomness.html
    # For reproducibility
    if opt.seed is not None:
        torch.manual_seed(opt.seed)
        torch.backends.cudnn.benchmark = False
        # torch.set_deterministic(True)
        np.random.seed(opt.seed)
        random.seed(opt.seed)

    if not opt.output_dir:
        print('No experiment result will be saved.')
        raise

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    if opt.batch_size < 2048 and opt.n_warmup_steps <= 4000:
        print('[Warning] The warmup steps may be not enough.\n' \
              '(sz_b, warmup) = (2048, 4000) is the official setting.\n' \
              'Using smaller batch w/o longer warmup may cause ' \
              'the warmup stage ends with only little data trained.')

    device = torch.device('cuda' if opt.cuda else 'cpu')

    # ========= Loading Dataset =========#

    if all((opt.train_path, opt.val_path)):
        raise
        # training_data, validation_data = prepare_dataloaders_from_bpe_files(opt, device)
    elif opt.data_pkl:
        training_data, validation_data = prepare_dataloaders(opt, device)  # ysy:load data
        print(training_data.__len__())  # 114  ==len(training_data)
        print(validation_data.__len__())
    else:
        raise

    print(opt)
    log_opt_file = os.path.join(opt.output_dir, opt.fileHead, 'opt.log')
    with open(log_opt_file, 'w') as log_f:
        log_f.write(str(opt))

    transformer = Transformer(
        opt.src_vocab_size,  # 9516 ==>1585?
        opt.trg_vocab_size,  # 9516
        src_pad_idx=opt.src_pad_idx,  # 1
        trg_pad_idx=opt.trg_pad_idx,  # 1
        trg_emb_prj_weight_sharing=opt.proj_share_weight,  # true
        emb_src_trg_weight_sharing=opt.embs_share_weight,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout,
        scale_emb_or_prj=opt.scale_emb_or_prj).to(device)

    optimizer = ScheduledOptim(
        optim.Adam(transformer.parameters(), betas=(0.9, 0.98), eps=1e-09),  # ,weight_decay=opt.l2
        opt.lr_mul, opt.d_model, opt.n_warmup_steps)

    train(transformer, training_data, validation_data, optimizer, device, opt)


def Rand_mask(x, seed,unmask):
    if 0:  #固定mask：
        len = x.size(0)  # len=1585
        random.seed(seed)
        torch.manual_seed(seed)
        # unmask=random.randint(int(len*0.8),len)  #个数
        unmask = int(len * 0.3)  # 先固定unmask个数
        shuffle_indices = torch.rand(len, len).argsort()  # b,
        unmask_ind, mask_ind = shuffle_indices[:, :unmask], shuffle_indices[:, unmask:]
        # 对应 batch 维度的索引：(b,1)
        batch_ind = torch.arange(len).unsqueeze(-1)

        x[batch_ind, unmask_ind] = 1  # pad=1 use 1 padding

    else: #随机mask
        len = x.size(0)  # len=1585
        unmask = random.randint(int(len * unmask), len)  #个数
        random.seed(seed)
        torch.manual_seed(seed)
        # unmask = int(len * 0.3)  # 先固定unmask个数
        shuffle_indices = torch.rand(len, len).argsort()  # b,
        unmask_ind, mask_ind = shuffle_indices[:, :unmask], shuffle_indices[:, unmask:]
        # 对应 batch 维度的索引：(b,1)
        batch_ind = torch.arange(len).unsqueeze(-1)

        x[batch_ind, unmask_ind] = 1  # pad=1 use 1 padding

    return x
    # return x[batch_ind, unmask_ind]


def prepare_dataloaders(opt, device):
    batch_size = opt.batch_size
    pkldata = pickle.load(open(opt.data_pkl, 'rb'))  # './data/pre_data.pkl'
    x = pkldata['x']
    y = pkldata['y']
    x = torch.FloatTensor(np.array(x))
    y = torch.FloatTensor(np.array(y))

    # if isAdjmask:
    #     x = adj_mask3(x)

    if opt.isRandMask:
        print("~~~RandMask~~~~!")
        random.seed(42)
        random_integers = [random.randint(0, 1000) for _ in range(10)]  #[654, 114, 25, 759, 281, 250, 228
        print(random_integers)
        for T in range(0, opt.T):
            x1 = Rand_mask(x, random_integers[T],opt.unmask)
            if T == 0:
                if opt.isContrastLoss:
                    x1 = torch.cat([x1,Rand_mask(x, random_integers[T]+3,opt.unmask),Rand_mask(x, random_integers[T]+6,opt.unmask)], 1)  #拼接增强的数据×3
                train_x = x1
                train_y = y
            else:
                if opt.isContrastLoss:
                    x1=torch.cat([x1,Rand_mask(x, random_integers[T]+3,opt.unmask),Rand_mask(x, random_integers[T]+6,opt.unmask)], 1)
                train_x = torch.cat([train_x, x1], 0)
                train_y = torch.cat([train_y, y], 0)

    else:
        print("~~~No RandMask~~~~!")
        train_x=x
        train_y=y
    train_torch_dataset = Data.TensorDataset(train_x, train_y)
    val_torch_dataset = Data.TensorDataset(x, y)
    # val_torch_dataset = Data.TensorDataset(train_x, train_y)  if val use mask+loss
    train_iterator = Data.DataLoader(
        dataset=train_torch_dataset,
        batch_size=batch_size,
        shuffle=True,  # $TrueFalse
        num_workers=2,  # pin_memory=True  放在GPU上
        drop_last=True
    )
    val_iterator = Data.DataLoader(
        dataset=val_torch_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    # for epoch in range(1):
    #     for step, batch in enumerate(loader):
    #         print("Epoch:", epoch, "|step:", step, "  |batch x:", batch[0].numpy(), "  |batch y:", batch[1].numpy())

    opt.src_pad_idx = 1  # todo
    opt.trg_pad_idx = 1

    opt.src_vocab_size = len(x)  # 1585
    opt.trg_vocab_size = len(y)  # 1585

    return train_iterator, val_iterator


if __name__ == '__main__':
    main()

# -data_pkl ./data/pre_data.pkl -proj_share_weight -label_smoothing -output_dir output_retry -n_head 1 -n_layer 2 -warmup 16000 -lr_mul 2.0 -epoch 25 -b 2 -seed 10 -save_mode best -use_tb


# -data_pkl ./data/pre_data.pkl -proj_share_weight -label_smoothing -output_dir output_retry -n_head 1 -n_layer 3 -warmup 128000 -lr_mul 200 -epoch 60 -b 15 -save_mode best -use_tb -lambda 0.01
# -data_pkl ./data/pre_data.pkl -proj_share_weight -label_smoothing -output_dir output_retry -n_head 1 -n_layer 2 -warmup 16000 -lr_mul 2.0 -epoch 35 -b 2 -seed 10 -save_mode best -use_tb

#
