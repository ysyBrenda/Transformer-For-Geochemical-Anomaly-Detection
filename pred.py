# 
''' Pred with trained model. '''

import torch
import argparse
import dill as pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
import draw_atten_func
import transformer.Constants as Constants
# from torchtext.data import Dataset

from torch.utils.data import Dataset, DataLoader,TensorDataset


from transformer.Models import Transformer
from transformer.Translator import Translator_my

print('【Info】 YSY customData.')

def load_model(opt, device):

    checkpoint = torch.load(opt.model, map_location=device)
    model_opt = checkpoint['settings']

    model = Transformer(
        model_opt.src_vocab_size,
        model_opt.trg_vocab_size,

        model_opt.src_pad_idx,
        model_opt.trg_pad_idx,

        trg_emb_prj_weight_sharing=model_opt.proj_share_weight,
        emb_src_trg_weight_sharing=model_opt.embs_share_weight,
        d_k=model_opt.d_k,
        d_v=model_opt.d_v,
        d_model=model_opt.d_model,
        d_word_vec=model_opt.d_word_vec,
        d_inner=model_opt.d_inner_hid,
        n_layers=model_opt.n_layers,
        n_head=model_opt.n_head,
        dropout=model_opt.dropout).to(device)

    model.load_state_dict(checkpoint['model'])
    print('[Info] Trained model state loaded————',opt.model)
    # print(model.encoder.position_enc.normalization.weight)
    # print(model.encoder.position_enc.normalization.bias)
    # print(model.encoder.position_enc.normalization.running_mean)
    return model 


def main():
    '''Main Function'''

    parser = argparse.ArgumentParser(description='translate.py')

    parser.add_argument('-model', required=False,
                        help='Path to model weight file')
    parser.add_argument('-data_pkl', required=True,
                        help='Pickle file with both instances and vocabulary.')
    parser.add_argument('-output', default='pred_ysy0329.txt',
                        help="""Path to output the predictions (each line will
                        be the decoded sequence""")
    parser.add_argument('-beam_size', type=int, default=5)
    parser.add_argument('-max_seq_len', type=int, default=100)
    parser.add_argument('-no_cuda', action='store_true')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda

    # readdata = pd.read_csv("./data/pos_feature.csv", header=None)
    # data = readdata.values
    opt.src_pad_idx = 1
    opt.trg_pad_idx = 1
    opt.trg_bos_idx = 2
    opt.trg_eos_idx = 3

    opt.model = 'model_best.chkpt'

    import torch.utils.data as Data
    pkldata = pickle.load(open(opt.data_pkl, 'rb'))  #'./data/pre_data.pkl'
    print('[Info] pre data is loaded————', opt.data_pkl)
    x = pkldata['x']
    y = pkldata['y']
    x = torch.FloatTensor(np.array(x))
    y = torch.FloatTensor(np.array(y))
    test_loader = Data.TensorDataset(x, y)

    
    device = torch.device('cuda' if opt.cuda else 'cpu')
    translator = Translator_my(
        model=load_model(opt, device),
        beam_size=opt.beam_size,
        max_seq_len=opt.max_seq_len,
        src_pad_idx=opt.src_pad_idx,
        trg_pad_idx=opt.trg_pad_idx,
        trg_bos_idx=opt.trg_bos_idx,
        trg_eos_idx=opt.trg_eos_idx).to(device)

    trg_file='./output/trg_0329_nomask.csv'
    pred_file ='./output/pred_0329_nomask.csv'

    Trg_all =torch.zeros(1,40)
    Pred_all = torch.zeros(1,38)
    import time
    start = time.time()
    for i,example in enumerate(tqdm(test_loader, mininterval=2, desc='  - (Test)', leave=False)):
    
        src_seq=example[0]
        trg_seq=example[1]
        pred_seq,dec_enc_attn=translator.translate_sentence(src_seq.to(device),trg_seq.to(device))
  
  
        Trg_all=torch.cat((Trg_all,trg_seq.unsqueeze(0)),0)  #0:按行拼接
        Pred_all=torch.cat((Pred_all,pred_seq.cpu()),0)
      
    # np.savetxt(trg_file, Trg_all, delimiter=",", fmt="%.8f")
    # np.savetxt(pred_file, Pred_all, delimiter=",", fmt="%.8f")
    print('[Info] Finished. {}s'.format(time.time()-start))

    import OUTPUTeval_ROC
    OUTPUTeval_ROC.PlotAndAUC(Pred_all, Trg_all,is1585=1)




if __name__ == "__main__":

    main()
         

 # Usage: python pred.py -data_pkl ./data/interp_data/interp0.pkl 