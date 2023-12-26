'''
This script handles the training process.
author： ysyBrenda
run in env:torch1.3.0
'''
import argparse
import time
import dill as pickle
from tqdm import tqdm
import numpy as np
import random
import os
import torch
import torch.nn.functional as F
import torch.optim as optim

from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim
import torch.utils.data as Data

# to use tensorboard，input following in terminal：
# $ tensorboard --logdir=output --port 6006
# if【was already in use】：lsof -i:6006， kill -9 PID

def train_epoch(model, training_data, optimizer, opt, device):
    ''' Epoch operation in training'''
    model.train()
    total_loss = 0
    iter = 0
    desc = '  - (Training)   '
    for batch in tqdm(training_data, mininterval=2, desc=desc, leave=False):  # todo
        # prepare data
        if opt.isContrastLoss:
            temp= batch[0].to(device)
            a = torch.chunk(temp, 3, dim=1)
            src_seq=torch.cat([a[0],a[1],a[2]],0)
        else:
            src_seq = batch[0].to(device)

        gold = batch[1][:, 2:].unsqueeze(1)
        trg_seq, gold = map(lambda x: x.to(device), [batch[1].unsqueeze(1), gold.contiguous().view(-1)])  # transpose、unsqueeze vector
        if opt.isContrastLoss:
             trg_seq=torch.cat([trg_seq,trg_seq,trg_seq],0)

        # forward
        optimizer.zero_grad()
        pred, *_ = model(src_seq, trg_seq)

        # backward and update parameters
        if opt.isContrastLoss:
            a = torch.chunk(pred, 3, dim=0)
            contras_loss = F.l1_loss(a[1].contiguous().view(-1), a[2].contiguous().view(-1), reduction='mean')
            loss = F.l1_loss(a[0].contiguous().view(-1), gold, reduction='mean') + opt.lambda_con * contras_loss
        else:
            loss = F.l1_loss(pred.contiguous().view(-1), gold, reduction='mean')  # F.l1_loss,F_mse_loss

        loss.backward()
        optimizer.step_and_update_lr()
        total_loss += loss.item()
        iter += 1

    print('total_train loss: {:8.5f},iter:{},average_train loss:{:8.5f} '.format(total_loss,iter,total_loss/iter)) #optimizer.n_steps=iter
    return total_loss/iter

def eval_epoch(model, validation_data, device, opt):
    ''' Epoch operation in evaluation '''
    model.eval()
    total_loss = 0
    iter=0
    desc = '  - (Validation) '
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2, desc=desc, leave=False):
            if opt.isContrastLoss:
                temp = batch[0].to(device)
                a = torch.chunk(temp, 3, dim=1)
                src_seq = torch.cat([a[0], a[1], a[2]], 0)
            else:
                src_seq = batch[0].to(device)
            gold = batch[1][:, 2:].unsqueeze(1)
            trg_seq, gold = map(lambda x: x.to(device), [batch[1].unsqueeze(1), gold.contiguous().view(-1)])
            if opt.isContrastLoss:
                trg_seq = torch.cat([trg_seq, trg_seq, trg_seq], 0)

            # forward
            pred, *_ = model(src_seq, trg_seq)

            # =============  loss==========================
            if opt.isContrastLoss:
                a = torch.chunk(pred, 3, dim=0)
                contras_loss = F.l1_loss(a[1].contiguous().view(-1), a[2].contiguous().view(-1), reduction='mean')
                loss = F.l1_loss(a[0].contiguous().view(-1), gold, reduction='mean') + opt.lambda_con * contras_loss
            else:
                loss = F.l1_loss(pred.contiguous().view(-1), gold, reduction='mean')  # reduction="mean"

            total_loss += loss.item()
            iter +=1
    print('total_val loss:{:8.5f} ,iter:{},average_val loss:{:8.5f}'.format(total_loss,iter,total_loss/iter))
    return total_loss/iter

def train(model, training_data, validation_data, optimizer, device, opt):
    """ Start training """

    # Use tensorboard to plot curves, e.g. loss, learning rate
    if opt.use_tb:
        print("[Info] Use Tensorboard")
        from tensorboardX import SummaryWriter
        # from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=os.path.join(opt.output_dir, 'tensorboard' + opt.fileHead))

    log_train_file = os.path.join(opt.output_dir, opt.fileHead, 'train.log')
    log_valid_file = os.path.join(opt.output_dir, opt.fileHead, 'valid.log')

    print('[Info] Training performance will be written to file: {} and {}'.format(log_train_file, log_valid_file))

    with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
        log_tf.write('epoch,loss,lr\n')
        log_vf.write('epoch,loss,lr\n')

    def print_performances(header, loss, start_time, lr):
        print('  - {header:12} loss: {loss: 8.5f},  lr: {lr: 8.2e}, ' \
              'elapse: {elapse:3.3f} min'.format(
            header=f"({header})", loss=loss,
            elapse=(time.time() - start_time) / 60, lr=lr))  # lr: {lr:8.5f}  8.2e

    valid_losses = []
    bad_counter = 0
    best = 50000
    patience = 5  # 5
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss = train_epoch(
            model, training_data, optimizer, opt, device)  # todo
        # Current learning rate
        lr = optimizer._optimizer.param_groups[0]['lr']
        print_performances('Training', train_loss, start, lr)

        # start = time.time()
        valid_loss = eval_epoch(model, validation_data, device, opt)  # todo
        print_performances('Validation', valid_loss, start, lr)

        valid_losses += [valid_loss]

        checkpoint = {'epoch': epoch_i, 'settings': opt, 'model': model.state_dict()}

        if opt.save_mode == 'all':
            # if epoch_i % 10 == 9:
            model_name = 'model_{epoch:d}_vloss_{vloss:.4f}.chkpt'.format(epoch=epoch_i, vloss=valid_loss)
            torch.save(checkpoint, os.path.join(opt.output_dir, opt.fileHead, model_name))
        elif opt.save_mode == 'best':
            model_name = 'model_best.chkpt'
            torch.save(checkpoint, os.path.join(opt.output_dir, opt.fileHead, model_name))
            print(' - [Info] The checkpoint file has been updated.')

        with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
            log_tf.write('{epoch},{loss: 8.5f},{lr:8.2e}\n'.format(
                epoch=epoch_i, loss=train_loss, lr=lr))
            log_vf.write('{epoch},{loss: 8.5f},{lr:8.2e}\n'.format(
                epoch=epoch_i, loss=valid_loss, lr=lr))

        if opt.use_tb:
            tb_writer.add_scalars('loss', {'train': train_loss, 'val': valid_loss}, epoch_i)
            tb_writer.add_scalar('learning_rate', lr, epoch_i)

        # auto break
        if valid_loss < best:
            best = valid_loss
            bad_counter = 0
        else:
            bad_counter += 1
        if bad_counter == patience:
            break

    log_opt_file = 'opt_file_log.log'  # add
    with open(log_opt_file, 'a') as log_f:
        log_f.write(str(opt.fileHead) + '__loss__{:8.5f}\n'.format(valid_loss))


def main():
    ''' 
    Usage:
    python train.py
                    -data_pkl ./data/pre_data.pkl  -output_dir output -epoch 150 -b 16 -use_tb -save_mode all
    '''

    parser = argparse.ArgumentParser()

    parser.add_argument('-data_pkl', default=None)  # all-in-1 data pickle or bpe field

    parser.add_argument('-train_path', default=None)  # bpe encoded data
    parser.add_argument('-val_path', default=None)  # bpe encoded data

    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=2048)

    parser.add_argument('-d_model', type=int, default=38)  # 38;8  #todo
    parser.add_argument('-d_inner_hid', type=int, default=2048)  # 64  #todo
    parser.add_argument('-d_k', type=int, default=38)  
    parser.add_argument('-d_v', type=int, default=38)  

    parser.add_argument('-n_head', type=int, default=2)
    parser.add_argument('-n_layers', type=int, default=4)  # 6
    parser.add_argument('-warmup', '--n_warmup_steps', type=int, default=4000)
    parser.add_argument('-lr_mul', type=float, default=2.0)  # 2.0
    parser.add_argument('-seed', type=int, default=None)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-output_dir', type=str, default=None)
    parser.add_argument('-use_tb', action='store_true')
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-unmask', type=float, default=0.3)
    parser.add_argument('-l2', type=float, default=0.0)  #  weight_dacay
    parser.add_argument('-lambda_con', type=float, default=0.01)  # contrast loss lambda
    parser.add_argument('-T', type=int, default=1)  # the times of mask
    parser.add_argument('-isContrastLoss', action='store_true')
    parser.add_argument('-isRandMask', action='store_true')

    opt = parser.parse_args()
    # # ++++++++++++++++
    opt.d_k = opt.d_model
    opt.d_v = opt.d_model

    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model  # 512 ==>38

    # ------Output fileHead----
    opt.fileHead = 'T' + str(opt.T) + '_unmask' + str(opt.unmask) + '_h' + str(opt.n_head) + 'L' + str(
        opt.n_layers) + '_hid' + str(opt.d_inner_hid) + '_d' + str(opt.d_model) + '_b' + str(
        opt.batch_size) + '_warm' + str(opt.n_warmup_steps) + '_lrm' + str(opt.lr_mul) + '_seed' + \
                   str(opt.seed) + '_dr' + str(opt.dropout) +'_isCL'+str(opt.isContrastLoss)+ '_lamb'+str(opt.lambda_con) +'_ismask'+str(opt.isRandMask)
    if os.path.exists(os.path.join(opt.output_dir, opt.fileHead)):
        print('the output file is rewriting....', opt.fileHead)
    else:
        os.mkdir(os.path.join(opt.output_dir, opt.fileHead))
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
    training_data, validation_data = prepare_dataloaders(opt, device)
    print("training data size:{}, validation data size:{}".format(training_data.__len__(),validation_data.__len__()))

    print(opt)
    log_opt_file = os.path.join(opt.output_dir, opt.fileHead, 'opt.log')
    with open(log_opt_file, 'w') as log_f:
        log_f.write(str(opt))

    transformer = Transformer(
        src_pad_idx=opt.src_pad_idx,  # 1
        trg_pad_idx=opt.trg_pad_idx,  # 1
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout).to(device)

    optimizer = ScheduledOptim(
        optim.Adam(transformer.parameters(), betas=(0.9, 0.98), eps=1e-09),  # ,weight_decay=opt.l2
        opt.lr_mul, opt.d_model, opt.n_warmup_steps)

    train(transformer, training_data, validation_data, optimizer, device, opt)


def Rand_mask(x, seed,unmask):
    if 0:  #fixed masking number
        len = x.size(0)  # len=1585
        random.seed(seed)
        torch.manual_seed(seed)
        # unmask=random.randint(int(len*0.8),len)
        unmask = int(len * 0.3)  # fix mask number
        shuffle_indices = torch.rand(len, len).argsort()  # b,
        unmask_ind, mask_ind = shuffle_indices[:, :unmask], shuffle_indices[:, unmask:]
        batch_ind = torch.arange(len).unsqueeze(-1)
        x[batch_ind, unmask_ind] = 1  # padding=1

    else: #ramdomly masking number
        len = x.size(0)  # len=1585
        unmask = random.randint(int(len * unmask), len)  #number of mask
        random.seed(seed)
        torch.manual_seed(seed)

        shuffle_indices = torch.rand(len, len).argsort()
        unmask_ind, mask_ind = shuffle_indices[:, :unmask], shuffle_indices[:, unmask:]
        batch_ind = torch.arange(len).unsqueeze(-1)
        x[batch_ind, unmask_ind] = 1  # padding=1
    return x


def prepare_dataloaders(opt, device):
    batch_size = opt.batch_size
    pkldata = pickle.load(open(opt.data_pkl, 'rb'))
    x = pkldata['x']
    y = pkldata['y']
    x = torch.FloatTensor(np.array(x))
    y = torch.FloatTensor(np.array(y))


    if opt.isRandMask:
        print("~~~RandMask~~~~!")
        random.seed(42)  #todo
        random_integers = [random.randint(0, 1000) for _ in range(10)]
        print("random_integers: ",random_integers)
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

    dataset=Data.TensorDataset(train_x, train_y)
    #Random split dataset
    train_size=int(len(dataset)*0.8)
    val_size=len(dataset)-train_size
    torch.manual_seed(42)
    train_torch_dataset,val_torch_dataset=Data.random_split(dataset,[train_size,val_size])

    train_iterator = Data.DataLoader(
        dataset=train_torch_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,  # pin_memory=True
    )
    val_iterator = Data.DataLoader(
        dataset=val_torch_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    opt.src_pad_idx = 1
    opt.trg_pad_idx = 1

    return train_iterator, val_iterator


if __name__ == '__main__':
    main()


#-data_pkl ./data/pre_data.pkl -output_dir output -n_head 2 -n_layer 4 -warmup 128000 -lr_mul 200 -epoch 50 -b 8 -save_mode best -use_tb -seed 10 -unmask 0.3 -T 2 -isRandMask -isContrastLoss
#                                                                      -warmup 16000 -lr_mul 2.0
