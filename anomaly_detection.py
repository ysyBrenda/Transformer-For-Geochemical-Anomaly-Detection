'''
geochemical anomaly detection
1，reconstruct geochemical data with trained model.
2，then, identify geochemical anomaly
Author: ysyBrenda
'''

import torch
import argparse
import dill as pickle
from tqdm import tqdm
import numpy as np

from transformer.Models import Transformer
from transformer.Translator import Translator
import calculate_anomalyscore

def load_model(opt, device):

    checkpoint = torch.load(opt.model, map_location=device)
    model_opt = checkpoint['settings']

    model = Transformer(
        model_opt.src_pad_idx,
        model_opt.trg_pad_idx,
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
    return model 


def main():
    '''Main Function'''

    parser = argparse.ArgumentParser(description='anomaly_detection.py')

    parser.add_argument('-model', required=False,
                        help='Path to model weight file')
    parser.add_argument('-data_pkl', required=True,
                        help='Pickle file.')
    parser.add_argument('-output', default='pred.txt',
                        help="Path to output the predictions")
    parser.add_argument('-no_cuda', action='store_true')

    parser.add_argument('-raw_data', default='data/pos_feature.csv',help="Path to raw_data(with coordinates)")
    parser.add_argument('-Au_data', default='data/Au.csv', help="Path to Au_data(with coordinates)")

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda

    opt.src_pad_idx = 1
    opt.trg_pad_idx = 1
    opt.trg_bos_idx = 2
    opt.trg_eos_idx = 3

    import torch.utils.data as Data
    pkldata = pickle.load(open(opt.data_pkl, 'rb'))
    print('[Info] pre data is loaded————', opt.data_pkl)
    x = pkldata['x']
    y = pkldata['y']
    x = torch.FloatTensor(np.array(x))
    y = torch.FloatTensor(np.array(y))
    test_loader = Data.TensorDataset(x, y)

    
    device = torch.device('cuda' if opt.cuda else 'cpu')
    translator = Translator(
        model=load_model(opt, device),
        src_pad_idx=opt.src_pad_idx).to(device)

    # ====== reconstruct geochemical data ======================
    Trg_all =torch.zeros(1,40)
    Pred_all = torch.zeros(1,38)
    import time
    start = time.time()
    for i,example in enumerate(tqdm(test_loader, mininterval=2, desc='  - (Test)', leave=False)):
    
        src_seq=example[0]
        trg_seq=example[1]
        pred_seq,dec_enc_attn=translator.translate_sentence(src_seq.to(device),trg_seq.to(device))

        Trg_all=torch.cat((Trg_all,trg_seq.unsqueeze(0)),0)
        Pred_all=torch.cat((Pred_all,pred_seq.cpu()),0)

    print('[Info] Reconstruction Finished. {}s'.format(time.time()-start))

    #====== calculate anomalyscore, and valuation (AUC)======================
    AUC_mean=calculate_anomalyscore.calculate(Pred_all, Trg_all,\
                                              raw_data=opt.raw_data,Au_data=opt.Au_data)
    print('[Info] Identify Geochamical Anomaly Finished.'.format(time.time() - start))
    print('----- the AUC_MEAN is: {:.4f}'.format(AUC_mean))



if __name__ == "__main__":
    '''
      Usage:
      python anomaly_detection.py -data_pkl ./data/interp_data/interp.pkl -model ./output/model_best.chkpt 
    '''
    main()
