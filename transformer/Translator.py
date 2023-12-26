''' This module will handle the text generation with beam search. '''

import torch
import torch.nn as nn
from transformer.Models import Transformer, get_pad_mask, get_subsequent_mask


def split_pos(data):
    #slite the position and data
    if data.size(0) == 1585 or (len(data.size()) ==2 and data.size(0)==1):
        return data[:,0:2],data[:,2:]
    elif (data.size(1) == 1585 or data.size(1)==1) and len(data.size()) == 3:
        return data[:,:,0:2].squeeze(0),data[:,:,2:]
    else:
        print("【 ！】data size Error")

def concat_pos(pos,data):
    # concate the position and data
    if data.size(0)==1585:
        data=torch.cat((pos, data), 1)
    elif (data.size(1)==1585 or data.size(1)==1) and len(data.size())==3:
        data=torch.cat((pos.unsqueeze(0), data),2)
    else:
        print("【 ！】data size Error")
    return data


class Translator(nn.Module):
    ''' Load a trained model and translate in beam search fashion. '''

    def __init__(
            self, model,src_pad_idx):
        

        super(Translator, self).__init__()

        self.src_pad_idx = src_pad_idx
        self.model = model
        self.model.eval()

    def _model_decode(self, trg_seq, enc_output, src_mask):
        trg_mask = get_subsequent_mask(trg_seq[:, :,0] )
        dec_output, dec_slf_attn,dec_enc_attn = self.model.decoder(trg_seq, trg_mask, enc_output, src_mask,return_attns=True)

        seq_logit=dec_output

        return seq_logit.view(-1, seq_logit.size(2)),dec_enc_attn


    def translate_sentence(self, src_seq,trg_seq):
        src_pad_idx= self.src_pad_idx

        with torch.no_grad():
            if len(src_seq.size())==2:
                src_seq=src_seq.unsqueeze(0)
                trg_seq=trg_seq.unsqueeze(0)
            src_mask = get_pad_mask(src_seq[:,:,0], src_pad_idx)
            enc_output, *_ = self.model.encoder(src_seq, src_mask)

            dec_output,dec_enc_attn = self._model_decode(trg_seq.unsqueeze(0), enc_output, src_mask)

        return dec_output,dec_enc_attn




