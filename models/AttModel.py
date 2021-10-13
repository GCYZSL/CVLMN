# This file contains Att2in2, AdaAtt, AdaAttMO, TopDown model

# AdaAtt is from Knowing When to Look: Adaptive Attention via A Visual Sentinel for Image Captioning
# https://arxiv.org/abs/1612.01887
# AdaAttMO is a modified version with maxout lstm

# Att2in is from Self-critical Sequence Training for Image Captioning
# https://arxiv.org/abs/1612.00563
# In this file we only have Att2in2, which is a slightly different version of att2in,
# in which the img feature embedding and word embedding is the same as what in adaatt.

# TopDown is from Bottom-Up and Top-Down Attention for Image Captioning and VQA
# https://arxiv.org/abs/1707.07998
# However, it may not be identical to the author's architecture.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import misc.utils as utils
import os
import copy
import math
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence


from .CaptionModel import CaptionModel

def sort_pack_padded_sequence(input, lengths):
    sorted_lengths, indices = torch.sort(lengths, descending=True)
    tmp = pack_padded_sequence(input[indices], sorted_lengths, batch_first=True)
    inv_ix = indices.clone()
    inv_ix[indices] = torch.arange(0,len(indices)).type_as(inv_ix)
    return tmp, inv_ix

def pad_unsort_packed_sequence(input, inv_ix):
    tmp, _ = pad_packed_sequence(input, batch_first=True)
    tmp = tmp[inv_ix]
    return tmp

def pack_wrapper(module, att_feats, att_masks):
    if att_masks is not None:
        packed, inv_ix = sort_pack_padded_sequence(att_feats, att_masks.data.long().sum(1))
        return pad_unsort_packed_sequence(PackedSequence(module(packed[0]), packed[1]), inv_ix)
    else:
        return module(att_feats)

class AttModel(CaptionModel):
    def __init__(self, opt):
        super(AttModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.rnn_size*3 #[obj mean, attr mean, rela mean]
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size
        self.seq_per_img = opt.seq_per_img
        self.cont_ver = opt.cont_ver
        self.rela_mod_index = getattr(opt, 'rela_mod_index', 1)
        self.relu_mod = getattr(opt, 'relu_mod', 'relu')
        self.leaky_relu_value = getattr(opt, 'leaky_relu_value', 0.1)
        self.memory_cell_path = getattr(opt, 'memory_cell_path', '0')

        self.index_eval = getattr(opt, 'index_eval', 0)

        self.use_bn = getattr(opt, 'use_bn', 0)

        self.ss_prob = 0.0 # Schedule sampling probability

        if self.relu_mod == 'relu':
            self.embed = nn.Sequential(nn.Embedding(self.vocab_size + 1, self.input_encoding_size),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(self.drop_prob_lm))
            self.fc_embed = nn.Sequential(nn.Linear(self.fc_feat_size, self.rnn_size),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(self.drop_prob_lm))
            # obj, attr, rela and sen module network
            self.obj_embed = nn.Sequential(*(
                                        ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ())+
                                        (nn.Linear(self.att_feat_size, self.rnn_size),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(self.drop_prob_lm))+
                                        ((nn.BatchNorm1d(self.rnn_size),) if self.use_bn==2 else ())))

            self.attr_embed = nn.Sequential(*(
                    ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ()) +
                    (nn.Linear(self.att_feat_size, self.rnn_size),
                     nn.ReLU(inplace=True),
                     nn.Dropout(self.drop_prob_lm)) +
                    ((nn.BatchNorm1d(self.rnn_size),) if self.use_bn == 2 else ())))

            self.rela_embed = nn.Sequential(*(
                    ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ()) +
                    (nn.Linear(self.att_feat_size, self.rnn_size),
                     nn.ReLU(inplace=True),
                     nn.Dropout(self.drop_prob_lm)) +
                    ((nn.BatchNorm1d(self.rnn_size),) if self.use_bn == 2 else ())))

        elif self.relu_mod == 'leaky_relu':
            self.embed = nn.Sequential(nn.Embedding(self.vocab_size + 1, self.input_encoding_size),
                                       nn.LeakyReLU(self.leaky_relu_value, inplace=True),
                                       nn.Dropout(self.drop_prob_lm))
            self.fc_embed = nn.Sequential(nn.Linear(self.fc_feat_size, self.rnn_size),
                                          nn.LeakyReLU(self.leaky_relu_value, inplace=True),
                                          nn.Dropout(self.drop_prob_lm))
            # obj, attr, rela and sen module network
            self.obj_embed = nn.Sequential(*(
                    ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ()) +
                    (nn.Linear(self.att_feat_size, self.rnn_size),
                     nn.LeakyReLU(self.leaky_relu_value, inplace=True),
                     nn.Dropout(self.drop_prob_lm)) +
                    ((nn.BatchNorm1d(self.rnn_size),) if self.use_bn == 2 else ())))

            self.attr_embed = nn.Sequential(*(
                    ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ()) +
                    (nn.Linear(self.att_feat_size, self.rnn_size),
                     nn.LeakyReLU(self.leaky_relu_value, inplace=True),
                     nn.Dropout(self.drop_prob_lm)) +
                    ((nn.BatchNorm1d(self.rnn_size),) if self.use_bn == 2 else ())))

            self.rela_embed = nn.Sequential(*(
                    ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ()) +
                    (nn.Linear(self.att_feat_size, self.rnn_size),
                     nn.LeakyReLU(self.leaky_relu_value, inplace=True),
                     nn.Dropout(self.drop_prob_lm)) +
                    ((nn.BatchNorm1d(self.rnn_size),) if self.use_bn == 2 else ())))

        if self.rela_mod_index:
            self.rela_mod = Rela_Mod(opt)

        self.logit_layers = getattr(opt, 'logit_layers', 1)
        if self.logit_layers == 1:
            self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
        else:
            self.logit = [[nn.Linear(self.rnn_size, self.rnn_size), nn.ReLU(), nn.Dropout(0.5)] for _ in range(opt.logit_layers - 1)]
            self.logit = nn.Sequential(*(reduce(lambda x,y:x+y, self.logit) + [nn.Linear(self.rnn_size, self.vocab_size + 1)]))

        self.ctx2att_obj = nn.Linear(self.rnn_size, self.att_hid_size)
        self.ctx2att_rela = nn.Linear(self.rnn_size, self.att_hid_size)
        self.ctx2att_attr = nn.Linear(self.rnn_size, self.att_hid_size)

        if os.path.isfile(self.memory_cell_path):
            print('load memory_cell from {0}'.format(self.memory_cell_path))
            memory_init = np.load(self.memory_cell_path)['memory_cell'][()]
        else:
            print('create a new memory_cell')
            memory_init = np.random.rand(self.memory_size, self.rnn_size) / 100
        memory_init = np.float32(memory_init)
        self.memory_cell = torch.from_numpy(memory_init).cuda().requires_grad_()


    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, bsz, self.rnn_size),
                    weight.new_zeros(self.num_layers, bsz, self.rnn_size))

    def post_process(self, rs_data):
        """
        1.get fc feats
        """
        obj_feats = rs_data['obj_feats_new']    #N_img, N_obj_max, 2048
        attr_feats = rs_data['attr_feats_new']  #N_img, N_attr_max, 2048
        rela_feats = rs_data['rela_feats_new']  #N_img, N_obj_max, 2048

        obj_masks = rs_data['att_masks']     #N_img, N_obj_max
        attr_masks = rs_data['attr_masks']     #N_img, N_obj_max
        rela_masks = rs_data['rela_masks']     #N_img, N_obj_max

        obj_feats_size = obj_feats.size()
        N_att = obj_feats_size[0]
        if self.index_eval == 1:
            seq_per_img = 1
        else:
            seq_per_img = self.seq_per_img
        N_img = int(N_att/seq_per_img)

        fc_feats = torch.zeros([N_img*seq_per_img, self.fc_feat_size]).cuda()

        for img_id in range(N_img):
            N_obj = int(torch.sum(obj_masks[img_id*seq_per_img, :]))
            N_attr = int(torch.sum(attr_masks[img_id*seq_per_img, :]))
            N_rela = int(torch.sum(rela_masks[img_id*seq_per_img, :]))

            fc_feats[img_id*seq_per_img:(img_id+1)*seq_per_img, :] = \
                    torch.cat((torch.mean(obj_feats[img_id*seq_per_img,0:N_obj,:], 0),
                               torch.mean(attr_feats[img_id*seq_per_img,0:N_attr,:], 0),
                               torch.mean(rela_feats[img_id * seq_per_img, 0:N_rela, :], 0)) )

        rs_data['fc_feats'] = fc_feats

        return rs_data

    def prepare_feats(self, rs_data):
        rs_data['obj_fc1'] = rs_data['att_feats']
        rs_data['attr_fc1'] = rs_data['attr_feats']
        if self.rela_mod_index:
            rs_data = self.rela_mod(rs_data)
        else:
            rs_data['rela_fc1'] = rs_data['rela_feats']

        rs_data['obj_feats_new'] = self.obj_embed(rs_data['obj_fc1'])
        rs_data['attr_feats_new'] = self.attr_embed(rs_data['attr_fc1'])
        rs_data['rela_feats_new'] = self.rela_embed(rs_data['rela_fc1'])

        rs_data = self.post_process(rs_data)
        rs_data['p_obj_feats'] = self.ctx2att_obj(rs_data['obj_feats_new'])
        rs_data['p_attr_feats'] = self.ctx2att_attr(rs_data['attr_feats_new'])
        rs_data['p_rela_feats'] = self.ctx2att_rela(rs_data['rela_feats_new'])
        rs_data['fc_feats_new'] = self.fc_embed(rs_data['fc_feats'])
        rs_data['memory_cell'] = self.memory_cell
        return rs_data


    def _forward(self, rs_data, seq):
        batch_size = rs_data['att_feats'].size(0)
        state = self.init_hidden(batch_size)

        weight = next(self.parameters())
        rs_data['module_embeddings'] = weight.new_zeros(batch_size,1,self.input_encoding_size)

        outputs = rs_data['att_feats'].new_zeros(batch_size, seq.size(1) - 1, self.vocab_size+1)
        cont_weights = rs_data['att_feats'].new_zeros(batch_size, seq.size(1) - 1, 4)

        rs_data = self.prepare_feats(rs_data)

        for i in range(seq.size(1) - 1):
            if self.training and i >= 1 and self.ss_prob > 0.0: # otherwiste no need to sample
                sample_prob = rs_data['fc_feats_new'].new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq[:, i].data.clone()
                    prob_prev = torch.exp(outputs[:, i-1].detach()) # fetch prev distribution: shape Nx(M+1)
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
            else:
                it = seq[:, i].clone()
            if i >= 1 and seq[:, i].sum() == 0:
                break


            output, state, cont_weight = self.get_logprobs_state(it, rs_data, state)
            cont_weights[:, i] = cont_weight
            outputs[:, i] = output

        return outputs, cont_weights

    def get_logprobs_state(self, it, rs_data, state):
        # 'it' contains a word index
        xt = self.embed(it)

        output, state, cont_weight = self.core(xt, rs_data, state)

        logprobs = F.log_softmax(self.logit(output), dim=1)

        return logprobs, state, cont_weight


    def _sample_beam(self, rs_data, opt={}):
        beam_size = opt.get('beam_size', 10)

        batch_size = rs_data['att_feats'].size(0)

        rs_data = self.prepare_feats(rs_data)

        fc_feats = rs_data['fc_feats_new']
        obj_feats = rs_data['obj_feats_new']
        obj_masks = rs_data['att_masks']
        p_obj_feats = rs_data['p_obj_feats']

        attr_feats = rs_data['attr_feats_new']
        p_attr_feats = rs_data['p_attr_feats']
        attr_masks = rs_data['attr_masks']

        rela_feats = rs_data['rela_feats_new']
        p_rela_feats = rs_data['p_rela_feats']
        rela_masks = rs_data['rela_masks']

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        tags = torch.FloatTensor(self.seq_length, batch_size)
        seqtagprobs = torch.FloatTensor(self.seq_length, batch_size,4)

        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            weight = next(self.parameters())


            tmp_fc_feats = fc_feats[k:k+1].expand(beam_size, fc_feats.size(1))
            tmp_obj_feats = obj_feats[k:k+1].expand(*((beam_size,)+obj_feats.size()[1:])).contiguous()
            tmp_attr_feats = attr_feats[k:k+1].expand(*((beam_size,)+attr_feats.size()[1:])).contiguous()
            tmp_rela_feats = rela_feats[k:k+1].expand(*((beam_size,)+rela_feats.size()[1:])).contiguous()

            tmp_p_obj_feats = p_obj_feats[k:k+1].expand(*((beam_size,)+p_obj_feats.size()[1:])).contiguous()
            tmp_p_attr_feats = p_attr_feats[k:k+1].expand(*((beam_size,)+p_attr_feats.size()[1:])).contiguous()
            tmp_p_rela_feats = p_rela_feats[k:k+1].expand(*((beam_size,)+p_rela_feats.size()[1:])).contiguous()

            module_embeddings = weight.new_zeros(batch_size, 1, self.input_encoding_size)
            tmp_module_embeddings = module_embeddings[k:k + 1].expand(*((beam_size,)+module_embeddings.size()[1:])).contiguous()

            tmp_obj_masks = obj_masks[k:k+1].expand(*((beam_size,)+obj_masks.size()[1:])).contiguous() \
                if obj_masks is not None else None
            tmp_attr_masks = attr_masks[k:k + 1].expand(*((beam_size,) + attr_masks.size()[1:])).contiguous() \
                if attr_masks is not None else None
            tmp_rela_masks = rela_masks[k:k + 1].expand(*((beam_size,) + rela_masks.size()[1:])).contiguous() \
                if rela_masks is not None else None

            tmp_rs_data = {}
            tmp_rs_data['fc_feats_new'] = tmp_fc_feats
            tmp_rs_data['obj_feats_new'] = tmp_obj_feats
            tmp_rs_data['attr_feats_new'] = tmp_attr_feats
            tmp_rs_data['rela_feats_new'] = tmp_rela_feats
            tmp_rs_data['att_masks'] = tmp_obj_masks
            tmp_rs_data['attr_masks'] = tmp_attr_masks
            tmp_rs_data['rela_masks'] = tmp_rela_masks
            tmp_rs_data['p_obj_feats'] = tmp_p_obj_feats
            tmp_rs_data['p_attr_feats'] = tmp_p_attr_feats
            tmp_rs_data['p_rela_feats'] = tmp_p_rela_feats
            tmp_rs_data['memory_cell'] = self.memory_cell
            tmp_rs_data['module_embeddings'] = tmp_module_embeddings

            for t in range(1):
                if t == 0: # input <bos>
                    it = fc_feats.new_zeros([beam_size], dtype=torch.long)

                logprobs, state, taglogprobs  = self.get_logprobs_state(it, tmp_rs_data, state)

            tagprobs = torch.exp(taglogprobs)
            self.done_beams[k] = self.beam_search(state, logprobs, tagprobs, tmp_rs_data, opt=opt)
            seq[:, k] = self.done_beams[k][0]['seq']  # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
            tags[:, k] = self.done_beams[k][0]['tag']
            # seqtagprobs[:, k] = self.done_beams[k][0]['tag_p']


        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1), tags.transpose(0, 1)

    def _sample(self, rs_data, opt={}):

        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        decoding_constraint = opt.get('decoding_constraint', 0)
        if beam_size > 1:
            return self._sample_beam(rs_data, opt)

        batch_size = rs_data['att_feats'].size(0)
        state = self.init_hidden(batch_size)

        rs_data = self.prepare_feats(rs_data)
        weight = next(self.parameters())
        rs_data['module_embeddings'] = weight.new_zeros(batch_size, 1, self.input_encoding_size)

        seq = rs_data['fc_feats_new'].new_zeros((batch_size, self.seq_length), dtype=torch.long)
        seqLogprobs = rs_data['fc_feats_new'].new_zeros(batch_size, self.seq_length)
        cont_weights = rs_data['att_feats'].new_zeros(batch_size, self.seq_length+1, 4)

        for t in range(self.seq_length + 1):
            if t == 0: # input <bos>
                it = rs_data['fc_feats_new'].new_zeros(batch_size, dtype=torch.long)
            elif sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data) # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature))
                it = torch.multinomial(prob_prev, 1)
                sampleLogprobs = logprobs.gather(1, it) # gather the logprobs at sampled positions
                it = it.view(-1).long() # and flatten indices for downstream processing

            if t >= 1:
                # stop when all finished
                if t == 1:
                    unfinished = it > 0
                else:
                    unfinished = unfinished * (it > 0)
                if unfinished.sum() == 0:
                    break
                it = it * unfinished.type_as(it)
                seq[:,t-1] = it
                # seq.append(it) #seq[t] the input of t+2 time step

                # seqLogprobs.append(sampleLogprobs.view(-1))
                seqLogprobs[:,t-1] = sampleLogprobs.view(-1)

            logprobs, state, cont_weight = self.get_logprobs_state(it, rs_data, state)
            cont_weights[:, t] = cont_weight

            if decoding_constraint and t > 0:
                tmp = output.new_zeros(output.size(0), self.vocab_size + 1)
                tmp.scatter_(1, seq[:,t-1].data.unsqueeze(1), float('-inf'))
                logprobs = logprobs + tmp

        return seq, seqLogprobs, cont_weights
        #return seq, seqLogprobs

class TopDownCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(TopDownCore, self).__init__()
        self.opt = opt
        self.drop_prob_lm = opt.drop_prob_lm
        self.combine_att = opt.combine_att
        self.topdown_res = getattr(opt, 'topdown_res', 0)
        self.cont_ver = getattr(opt, 'cont_ver', 0)
        if self.combine_att == 'concat':
            self.lang_lstm = nn.LSTMCell(opt.rnn_size * 6, opt.rnn_size)
        elif self.combine_att == 'add':
            self.lang_lstm = nn.LSTMCell(opt.rnn_size * 3, opt.rnn_size)  # h^1_t, \hat v

        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size) # we, fc, h^2_t-1
        self.query_fc = nn.Linear(3*opt.input_encoding_size, opt.input_encoding_size)
        # four modules
        self.embed1 = nn.Parameter(torch.randn(1,1000))
        self.embed2 = nn.Parameter(torch.randn(1,1000))
        self.embed3 = nn.Parameter(torch.randn(1,1000))
        self.embed4 = nn.Parameter(torch.randn(1,1000))

        self.relu_mod = getattr(opt, 'relu_mod', 'relu')
        self.leaky_relu_value = getattr(opt, 'leaky_relu_value', 0.1)
        if self.relu_mod == 'relu':
            self.lang_embed = nn.Sequential(nn.Linear(opt.rnn_size, opt.rnn_size),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout(self.drop_prob_lm))
            self.mem_fc = nn.Sequential(nn.Linear(5 * opt.rnn_size, opt.rnn_size),
                                        nn.ReLU(self.leaky_relu_value, inplace=True),
                                        nn.Dropout(self.drop_prob_lm))

        elif self.relu_mod == 'leaky_relu':
            self.lang_embed = nn.Sequential(nn.Linear(opt.rnn_size, opt.rnn_size),
                                            nn.LeakyReLU(self.leaky_relu_value, inplace=True),
                                            nn.Dropout(self.drop_prob_lm))
            self.mem_fc = nn.Sequential(nn.Linear(5*opt.rnn_size, opt.rnn_size),
                                            nn.LeakyReLU(self.leaky_relu_value, inplace=True),
                                            nn.Dropout(self.drop_prob_lm))

        self.attention_obj = Attention(opt)
        self.attention_attr = Attention(opt)
        self.attention_rela = Attention(opt)
        self.ssg_mem = Memory_cell2(opt)

        if self.cont_ver == 1:
            self.controller = Mod_Cont(opt)
            #4 means the number of module networks, if more modules are used, this number should be changed
            self.W_cont = nn.Linear(opt.rnn_size, 4)

    def forward(self, xt, rs_data, state):
        prev_h = state[0][1]
        #prev_h = state[0][-1]
        att_lstm_input = torch.cat([prev_h, rs_data['fc_feats_new'], xt], 1)

        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))

        att_obj = self.attention_obj(h_att, rs_data['obj_feats_new'],
                                     rs_data['p_obj_feats'], rs_data['att_masks']) #batch * att_feat_size
        att_attr = self.attention_attr(h_att, rs_data['attr_feats_new'],
                                     rs_data['p_attr_feats'], rs_data['attr_masks'])
        att_rela = self.attention_rela(h_att, rs_data['rela_feats_new'],
                                     rs_data['p_rela_feats'], rs_data['rela_masks'])
        att_lang = self.lang_embed(state[1][1])

        if self.cont_ver == 1:
            rs_data['query'] = self.query_fc(att_lstm_input)

            if rs_data.get('beam_search',0) == 1:
                weight = next(self.parameters())
                module_embeddings = weight.new_zeros(1, 1, self.opt.input_encoding_size)
                rs_data['module_embeddings'] = module_embeddings.expand(
                    *((self.opt.beam_size,) + module_embeddings.size()[1:])).contiguous()
                for t in range(rs_data['beam_step']+1):
                    cont_weights = rs_data['beam_cont_weights'][t].cuda()
                    me_new = cont_weights[:, 0].unsqueeze(1).expand_as(att_obj) * self.embed1.expand_as(att_obj) \
                         + cont_weights[:, 1].unsqueeze(1).expand_as(att_obj) * self.embed2.expand_as(att_obj) \
                         + cont_weights[:, 2].unsqueeze(1).expand_as(att_obj) * self.embed3.expand_as(att_obj) \
                         + cont_weights[:, 3].unsqueeze(1).expand_as(att_obj) * self.embed4.expand_as(att_obj)

                    me_new = me_new.unsqueeze(1)
                    rs_data['module_embeddings'] = torch.cat([rs_data['module_embeddings'], me_new], 1)

            h_new = self.controller(rs_data)
            cont_weights = F.softmax(self.W_cont(h_new),dim=1)
            cw_logit = F.log_softmax(self.W_cont(h_new),dim=1)
            att_obj = att_obj * cont_weights[:, 1].unsqueeze(1).expand_as(att_obj)
            att_attr = att_attr * cont_weights[:, 3].unsqueeze(1).expand_as(att_attr)
            att_rela = att_rela * cont_weights[:, 2].unsqueeze(1).expand_as(att_rela)
            att_lang = att_lang * cont_weights[:, 0].unsqueeze(1).expand_as(att_lang)

            if rs_data.get('beam_search', 0) == 0:
                me_new = cont_weights[:, 0].unsqueeze(1).expand_as(att_obj) * self.embed1.expand_as(att_obj) \
                         + cont_weights[:, 1].unsqueeze(1).expand_as(att_obj) * self.embed2.expand_as(att_obj) \
                         + cont_weights[:, 2].unsqueeze(1).expand_as(att_obj) * self.embed3.expand_as(att_obj) \
                         + cont_weights[:, 3].unsqueeze(1).expand_as(att_obj) * self.embed4.expand_as(att_obj)

                me_new = me_new.unsqueeze(1)
                rs_data['module_embeddings']=torch.cat([rs_data['module_embeddings'], me_new],1)

        if self.combine_att == 'concat':
            att = torch.cat([att_obj, att_attr, att_rela, att_lang], 1)
        elif self.combine_att == 'add':
            att = (att_obj + att_attr +att_rela + att_lang)

        lang_lstm_input = torch.cat([att, h_att], 1)
        query_mem = self.mem_fc(lang_lstm_input)
        lang_mem = self.ssg_mem(query_mem, rs_data['memory_cell'])
        lang_lstm_input = torch.cat([lang_lstm_input, lang_mem], 1)

        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))
        if self.topdown_res:
            h_lang = h_lang + h_att
            c_lang = c_lang + c_att

        output = F.dropout(h_lang, self.drop_prob_lm, self.training)
        state = (torch.stack([h_att, h_lang, h_new]), torch.stack([c_att, c_lang, h_new]))

        return output, state, cw_logit

class Attention(nn.Module):
    def __init__(self, opt):
        super(Attention, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size

        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)

    def forward(self, h, att_feats, p_att_feats, att_masks=None):
        # The p_att_feats here is already projected
        att_size = att_feats.numel() // att_feats.size(0) // att_feats.size(-1)
        att = p_att_feats.view(-1, att_size, self.att_hid_size)
        
        att_h = self.h2att(h)                        # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)            # batch * att_size * att_hid_size
        dot = att + att_h                                   # batch * att_size * att_hid_size
        dot = F.tanh(dot)                                # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)               # (batch * att_size) * att_hid_size
        dot = self.alpha_net(dot)                           # (batch * att_size) * 1
        dot = dot.view(-1, att_size)                        # batch * att_size
        
        weight = F.softmax(dot, dim=1)                             # batch * att_size
        if att_masks is not None:
            weight = weight * att_masks.view(-1, att_size).float()
            weight = weight / weight.sum(1, keepdim=True) # normalize to 1
        # att_index = torch.argmax(weight,dim=1)
        # print(att_index)
        att_feats_ = att_feats.view(-1, att_size, att_feats.size(-1)) # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1) # batch * att_feat_size

        return att_res

class Cap_Reason3_mem_new(AttModel):
    def __init__(self, opt):
        super(Cap_Reason3_mem_new, self).__init__(opt)
        if opt.cont_ver!=0:
            self.num_layers = 3
        else:
            self.num_layers = 2
        self.core = TopDownCore(opt)


class Mod_Cont(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(Mod_Cont, self).__init__()
        c = copy.deepcopy
        h = 8
        N = 1
        dropout = 0.1
        d_model = 1000
        d_ff = 1000

        self.attn = MultiHeadedAttention(h, d_model)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)

    def forward(self, rs_data):
        h = self.attn(rs_data['query'], rs_data['module_embeddings'], rs_data['module_embeddings'])
        h = self.ff(h)
        return h.squeeze()

#########################################use self-attention to build rela module################
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

def compute_attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
            mask = mask.unsqueeze(2)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = compute_attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class Rela_Mod(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(Rela_Mod, self).__init__()
        c = copy.deepcopy
        h = 8
        N = getattr(opt, 'rela_mod_layer', 2)
        dropout = 0.1
        d_model = opt.att_feat_size
        d_ff = opt.att_feat_size

        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.model = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)

    def forward(self, rs_data):
        rs_data['rela_fc1'] = self.model(rs_data['rela_feats'], rs_data['rela_masks'])
        return rs_data

class Memory_cell2(nn.Module):
    def __init__(self, opt):
        """
        a_i = h^T*m_i
        a_i: 1*1

        h: R*1
        m_i: R*1
        M=[m_1,m_2,...,m_K]^T: K*R
        att = softmax(a): K*1
        h_out = M*att: N*R

        :param opt:
        """
        super(Memory_cell2, self).__init__()
        self.R = opt.rnn_size
        self.V = opt.att_hid_size

        self.W = nn.Linear(self.V, 1)

    def forward(self, h, M):
        M_size = M.size()  # K*R
        h_size = h.size()  # N*R
        h = h.view(-1, h_size[1]) # (N*T)*R
        att = torch.mm(h, torch.t(M)) #(N*T)*K
        att = F.softmax(att, dim=1) #(N*T)*K
        #att_sum = torch.sum(att, dim=1)
        att_max = torch.max(att,dim=1)
        max_index = torch.argmax(att,dim=1)
        att_res = torch.mm(att, M)  #(N*T)*K * K*R->(N*T)*R
        # att_res = att_res.view([h_size[0], h_size[1], h_size[2]])
        return att_res






