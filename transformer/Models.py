''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np

import draw_atten_func
from transformer.Layers import EncoderLayer, DecoderLayer
isCustomData=1
if isCustomData==1:
    ALL_posNUM=2000
else:
    ALL_posNUM=200



def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask

#我自己的positional encoding
class PositionalEncoding_my(nn.Module):

    def __init__(self, d_hid, n_position=2000):
        super(PositionalEncoding_my, self).__init__()

        # self.linear_p = nn.Sequential(nn.Linear(2, 2), nn.ReLU(inplace=True), nn.Linear(2, d_hid))
        # self.linear_p = nn.Sequential(nn.Linear(2, 2),nn.BatchNorm1d(2), nn.ReLU(inplace=True), nn.Linear(2, d_hid))  # BatchNorm1得转置输入
        self.linear_p=nn.Sequential(nn.Linear(2,d_hid),nn.ReLU(inplace=True),nn.Linear(d_hid,d_hid))  #nn.BatchNorm1d(2)  最后用这个版本
        # self.normalization=nn.BatchNorm1d(2)

    def forward(self, x):
        pos,x=x[:,:,0:2],x[:,:,2:]
        pos[:,:,0]=pos[:,:,0]/83000
        pos[:,:,1]=pos[:,:,1]/102000
        # pos_norm=(self.normalization(pos.transpose(1,2).contiguous())).transpose(1,2).contiguous()
        # draw_atten_func.draw_position(pos)
        self.pos_table = self.linear_p(pos)
        #--add
        p_e=self.pos_table.clone().detach()
        return x + self.pos_table[:, :x.size(1)].clone().detach(), p_e,x  #todo  x=256,25,512 selfp..=1,25,512
    #x=1,1585,512

class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=2000):
        super(PositionalEncoding, self).__init__()
       # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))   #todo   # n_position=1585

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)   #2000*512=n_position* d_hid

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()  #todo  x=256,25,512 selfp..=1,25,512
    #x=1,1585,512


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=0.1, n_position=ALL_posNUM,):

        super().__init__()

        self.position_enc = PositionalEncoding_my(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.d_model = d_model

        self.data_prj= nn.Linear(in_features=38,out_features=d_word_vec, bias=False)  #512， n_trg_vocab=1585

    def forward(self, src_seq, src_mask, return_attns=False):

        enc_slf_attn_list = []

        # -- Forward
        # enc_output = self.data_prj(src_seq)
        enc_output = src_seq


        POSE,pe,x=self.position_enc(enc_output)
        enc_output = self.dropout(POSE)
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, n_position=ALL_posNUM, dropout=0.1):

        super().__init__()

        self.position_enc = PositionalEncoding_my(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.d_model = d_model
        if isCustomData == 1:
            self.data_prj = nn.Linear(in_features=38, out_features=d_word_vec, bias=False)  # 512， n_trg_vocab=1585

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []


        dec_output=trg_seq

        # -- Forward
        # train时，标注数据和enc_output作为decoder输入，在测试时，上一时刻的decoder输出 和enc_output 作为下一时刻的decoder输入。
        POSE,pe,x=self.position_enc(dec_output)
        dec_output = self.dropout(POSE)
        dec_output = self.layer_norm(dec_output)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output


class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self,  src_pad_idx, trg_pad_idx,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=ALL_posNUM,
            scale_emb_or_prj='prj'):

        super().__init__()

        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx

        # In section 3.4 of paper "Attention Is All You Need", there is such detail:
        # "In our model, we share the same weight matrix between the two
        # embedding layers and the pre-softmax linear transformation...
        # In the embedding layers, we multiply those weights by \sqrt{d_model}".
        #
        # Options here:
        #   'emb': multiply \sqrt{d_model} to embedding output
        #   'prj': multiply (\sqrt{d_model} ^ -1) to linear projection output
        #   'none': no multiplication

        assert scale_emb_or_prj in ['emb', 'prj', 'none']
        self.scale_prj = (scale_emb_or_prj == 'prj')
        self.d_model = d_model

        self.encoder = Encoder(
            n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=src_pad_idx, dropout=dropout)

        self.decoder = Decoder(
            n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=trg_pad_idx, dropout=dropout)


        self.trg_out_prj  = nn.Linear(d_model, 38, bias=False)  #512， n_trg_vocab=1585
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'



    def forward(self, src_seq, trg_seq):


        if isCustomData == 0:
            src_mask = get_pad_mask(src_seq, self.src_pad_idx)
            trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)
        else:
            src_mask=get_pad_mask(src_seq[:,:,0], self.src_pad_idx) #256,1,40   #计算mask 要输入（b，len）   new:is 256 1 1585
            trg_mask=trg_seq[:, :,0]  #.unsqueeze(1)
            trg_mask = get_pad_mask(trg_mask, self.trg_pad_idx) & get_subsequent_mask(trg_mask) #256,40,40

        enc_output,enc_slf_attn_list = self.encoder(src_seq, src_mask,return_attns=True)  #todo bug  *_
        dec_output, dec_slf_attn_list, dec_enc_attn_list= self.decoder(trg_seq, trg_mask, enc_output, src_mask,return_attns=True)

        seq_logit=dec_output

        if self.scale_prj:  #1
            seq_logit *= self.d_model ** -0.5

        # return seq_logit.view(-1, seq_logit.size(2))  #按照 size（2），reshape成矩阵，  合并batch
        return seq_logit.view(-1, seq_logit.size(2)),enc_slf_attn_list,dec_enc_attn_list  #按照 size（2），reshape成矩阵，  合并batch


# Modify Log:  n_position=200 ==》2000
# Modify Log:  d直接写的38，没有用变量

