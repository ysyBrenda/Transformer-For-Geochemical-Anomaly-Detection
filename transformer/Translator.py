''' This module will handle the text generation with beam search. '''

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer.Models import Transformer, get_pad_mask, get_subsequent_mask
from tqdm import tqdm



def split_pos(data):
    #将position与data分离
    # return data[:,0:2],data[:,2:] if data.size(0)==1585 else print("【 ！】data size Error")
    if data.size(0) == 1585 or (len(data.size()) ==2 and data.size(0)==1):
        return data[:,0:2],data[:,2:]
    elif (data.size(1) == 1585 or data.size(1)==1) and len(data.size()) == 3:
        return data[:,:,0:2].squeeze(0),data[:,:,2:]
    else:
        print("【 ！】data size Error")

def concat_pos(pos,data):
    # 将position与data合并
    if data.size(0)==1585:
        data=torch.cat((pos, data), 1)
    elif (data.size(1)==1585 or data.size(1)==1) and len(data.size())==3:
        data=torch.cat((pos.unsqueeze(0), data),2)
    else:
        print("【 ！】data size Error")
    return data


class Translator_my(nn.Module):
    ''' Load a trained model and translate in beam search fashion. '''

    def __init__(
            self, model, beam_size, max_seq_len,
            src_pad_idx, trg_pad_idx, trg_bos_idx, trg_eos_idx):
        

        super(Translator_my, self).__init__()

        self.alpha = 0.7
        self.beam_size = beam_size
        self.max_seq_len = max_seq_len
        self.src_pad_idx = src_pad_idx
        self.trg_bos_idx = trg_bos_idx
        self.trg_eos_idx = trg_eos_idx

        self.model = model
        self.model.eval()

        self.register_buffer('init_seq', torch.LongTensor([[trg_bos_idx]]))
        self.register_buffer(
            'blank_seqs', 
            torch.full((beam_size, max_seq_len), trg_pad_idx, dtype=torch.long))
        self.blank_seqs[:, 0] = self.trg_bos_idx
        self.register_buffer(
            'len_map', 
            torch.arange(1, max_seq_len + 1, dtype=torch.long).unsqueeze(0))


    def _model_decode(self, trg_seq, enc_output, src_mask):
        # trg_seq1 = trg_seq[:, 0].unsqueeze(0) if len(trg_seq.size())==2 else trg_seq
        trg_mask = get_subsequent_mask(trg_seq[:, :,0] )
        dec_output, dec_slf_attn,dec_enc_attn = self.model.decoder(trg_seq, trg_mask, enc_output, src_mask,return_attns=True)

                    #=========modify by ysy
        if self.model.d_model==38:
            seq_logit=dec_output
        else:
            seq_logit=self.model.trg_out_prj(dec_output)
        seq_logit *= self.model.d_model ** -0.5
        return seq_logit.view(-1, seq_logit.size(2)),dec_enc_attn
        # return F.softmax(self.model.trg_word_prj(dec_output), dim=-1)
    # def _get_init_state_my(self, src_seq, src_mask):
    #
    #     enc_output, attns = self.model.encoder(src_seq, src_mask,return_attns=True)
    #     dec_output = self._model_decode(self.init_seq, enc_output, src_mask)
    #
    #     return enc_output, dec_output

    def translate_sentence(self, src_seq,trg_seq):
        # Only accept batch size equals to 1 in this function.
        # ysy:add a parameter:trg_seq

        src_pad_idx, trg_eos_idx = self.src_pad_idx, self.trg_eos_idx 
        max_seq_len, beam_size, alpha = self.max_seq_len, self.beam_size, self.alpha 

        with torch.no_grad():
            # 每个sample单独放入计算（即每个词单独计算）

            # for i in tqdm(range(0, 1585), mininterval=2, leave=False):
            if len(src_seq.size())==2:  #一句话时。需要加1维，从1585,40==> 1,1585,40
                src_seq=src_seq.unsqueeze(0)
                trg_seq=trg_seq.unsqueeze(0)
            src_mask = get_pad_mask(src_seq[:,:,0], src_pad_idx)
            enc_output, *_ = self.model.encoder(src_seq, src_mask)   #enc output 一直不变！

            # trg_seq 这里使用传进来的目标句。
            dec_output,dec_enc_attn = self._model_decode(trg_seq.unsqueeze(0), enc_output, src_mask)
            # print(" -【trg_seq】", trg_seq.data.cpu().numpy()[:,0:5])
            # print(" -【dec_output】", dec_output.data.cpu().numpy()[:,0:3])

        return dec_output,dec_enc_attn




class Translator(nn.Module):
    ''' Load a trained model and translate in beam search fashion. '''

    def __init__(
            self, model, beam_size, max_seq_len,
            src_pad_idx, trg_pad_idx, trg_bos_idx, trg_eos_idx):

        super(Translator, self).__init__()

        self.alpha = 0.7
        self.beam_size = beam_size
        self.max_seq_len = max_seq_len
        self.src_pad_idx = src_pad_idx
        self.trg_bos_idx = trg_bos_idx
        self.trg_eos_idx = trg_eos_idx

        self.model = model
        self.model.eval()

        self.register_buffer('init_seq', torch.LongTensor([[trg_bos_idx]]))
        self.register_buffer(
            'blank_seqs',
            torch.full((beam_size, max_seq_len), trg_pad_idx, dtype=torch.long))
        self.blank_seqs[:, 0] = self.trg_bos_idx
        self.register_buffer(
            'len_map',
            torch.arange(1, max_seq_len + 1, dtype=torch.long).unsqueeze(0))

    def _model_decode(self, trg_seq, enc_output, src_mask):
        trg_mask = get_subsequent_mask(trg_seq)
        dec_output, *_ = self.model.decoder(trg_seq, trg_mask, enc_output, src_mask)
        return F.softmax(self.model.trg_word_prj(dec_output), dim=-1)

    def _get_init_state(self, src_seq, src_mask):
        beam_size = self.beam_size

        enc_output, *_ = self.model.encoder(src_seq, src_mask)
        dec_output = self._model_decode(self.init_seq, enc_output, src_mask)

        best_k_probs, best_k_idx = dec_output[:, -1, :].topk(beam_size)

        scores = torch.log(best_k_probs).view(beam_size)
        gen_seq = self.blank_seqs.clone().detach()
        gen_seq[:, 1] = best_k_idx[0]
        enc_output = enc_output.repeat(beam_size, 1, 1)
        return enc_output, gen_seq, scores

    def _get_the_best_score_and_idx(self, gen_seq, dec_output, scores, step):
        assert len(scores.size()) == 1

        beam_size = self.beam_size

        # Get k candidates for each beam, k^2 candidates in total.
        best_k2_probs, best_k2_idx = dec_output[:, -1, :].topk(beam_size)

        # Include the previous scores.
        scores = torch.log(best_k2_probs).view(beam_size, -1) + scores.view(beam_size, 1)

        # Get the best k candidates from k^2 candidates.
        scores, best_k_idx_in_k2 = scores.view(-1).topk(beam_size)

        # Get the corresponding positions of the best k candidiates.
        best_k_r_idxs, best_k_c_idxs = best_k_idx_in_k2 // beam_size, best_k_idx_in_k2 % beam_size
        best_k_idx = best_k2_idx[best_k_r_idxs, best_k_c_idxs]

        # Copy the corresponding previous tokens.
        gen_seq[:, :step] = gen_seq[best_k_r_idxs, :step]
        # Set the best tokens in this beam search step
        gen_seq[:, step] = best_k_idx

        return gen_seq, scores

    def translate_sentence(self, src_seq):
        # Only accept batch size equals to 1 in this function.
        # TODO: expand to batch operation.
        assert src_seq.size(0) == 1

        src_pad_idx, trg_eos_idx = self.src_pad_idx, self.trg_eos_idx
        max_seq_len, beam_size, alpha = self.max_seq_len, self.beam_size, self.alpha

        with torch.no_grad():
            src_mask = get_pad_mask(src_seq, src_pad_idx)  #src_seq=Tensor(1,11)
            enc_output, gen_seq, scores = self._get_init_state(src_seq, src_mask)

            ans_idx = 0  # default
            for step in range(2, max_seq_len):  # decode up to max length
                dec_output = self._model_decode(gen_seq[:, :step], enc_output, src_mask)  #gen..上一个decode结果的输出作为下一个decode输入
                gen_seq, scores = self._get_the_best_score_and_idx(gen_seq, dec_output, scores, step)

                # Check if all path finished
                # -- locate the eos in the generated sequences
                eos_locs = gen_seq == trg_eos_idx
                # -- replace the eos with its position for the length penalty use
                seq_lens, _ = self.len_map.masked_fill(~eos_locs, max_seq_len).min(1)
                # -- check if all beams contain eos
                if (eos_locs.sum(1) > 0).sum(0).item() == beam_size:
                    # TODO: Try different terminate conditions.
                    _, ans_idx = scores.div(seq_lens.float() ** alpha).max(0)
                    ans_idx = ans_idx.item()
                    break
        return gen_seq[ans_idx][:seq_lens[ans_idx]].tolist()
