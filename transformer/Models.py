''' Define the Transformer model '''
import torch
import torch.nn as nn
from transformer.Layers import EncoderLayer, DecoderLayer


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class PositionalEncoding(nn.Module):
    # -------------------- Learnabel position encoding -----------------------------------
    def __init__(self, d_hid):
        super(PositionalEncoding, self).__init__()

        self.linear_p=nn.Sequential(nn.Linear(2,d_hid),nn.ReLU(inplace=True),nn.Linear(d_hid,d_hid))
        # self.normalization=nn.BatchNorm1d(2)

    def forward(self, x):
        pos,x=x[:,:,0:2],x[:,:,2:]

        self.pos_table = self.linear_p(pos)
        return x + self.pos_table[:, :x.size(1)].clone().detach(), self.pos_table.clone().detach()


# class PositionalEncoding(nn.Module):
# -------------------- Sinusoid position encoding -----------------------------------
#     def __init__(self, d_hid, n_position=2000):
#         super(PositionalEncoding, self).__init__()
#         self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))
#
#     def _get_sinusoid_encoding_table(self, n_position, d_hid):
#         ''' Sinusoid position encoding table '''
#
#         def get_position_angle_vec(position):
#             return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
#
#         sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
#         sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
#         sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
#
#         return torch.FloatTensor(sinusoid_table).unsqueeze(0)
#
#     def forward(self, x):
#         return x + self.pos_table[:, :x.size(1)].clone().detach()



class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=0.1, n_position=2000,):

        super().__init__()

        self.position_enc = PositionalEncoding(d_word_vec)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.d_model = d_model

    def forward(self, src_seq, src_mask, return_attns=False):

        enc_slf_attn_list = []

        # -- Forward

        enc_output = src_seq

        enc_output,pe=self.position_enc(enc_output)
        enc_output = self.dropout(enc_output)
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
            d_model, d_inner, pad_idx, n_position=2000, dropout=0.1):

        super().__init__()

        self.position_enc = PositionalEncoding(d_word_vec)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.d_model = d_model

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        dec_output=trg_seq

        # -- Forward
        dec_output,pe=self.position_enc(dec_output)
        dec_output = self.dropout(dec_output)
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
            d_word_vec=38, d_model=38, d_inner=2048,
            n_layers=6, n_head=8, d_k=38, d_v=38, dropout=0.1, n_position=2000,
                 ):

        super().__init__()

        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx

        self.scale_prj = False   #True
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

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

    def forward(self, src_seq, trg_seq):

        src_mask=get_pad_mask(src_seq[:,:,0], self.src_pad_idx)
        trg_mask=trg_seq[:, :,0]  #.unsqueeze(1)
        trg_mask = get_pad_mask(trg_mask, self.trg_pad_idx) & get_subsequent_mask(trg_mask)

        enc_output,enc_slf_attn_list = self.encoder(src_seq, src_mask,return_attns=True)
        dec_output, dec_slf_attn_list, dec_enc_attn_list= self.decoder(trg_seq, trg_mask, enc_output, src_mask,return_attns=True)

        seq_logit=dec_output

        return seq_logit.view(-1, seq_logit.size(2)),enc_slf_attn_list,dec_enc_attn_list


