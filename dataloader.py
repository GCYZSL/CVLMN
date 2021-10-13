from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import h5py
import os
import numpy as np
import random
from models.ass_fun import *

import torch
import torch.utils.data as data

import multiprocessing


class DataLoader(data.Dataset):

    def reset_iterator(self, split):
        del self._prefetch_process[split]
        self._prefetch_process[split] = BlobFetcher(split, self, split == 'train')
        self.iterators[split] = 0

    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def __init__(self, opt):
        self.opt = opt
        self.batch_size = self.opt.batch_size
        self.seq_per_img = opt.seq_per_img

        # load the json file which contains additional information about the dataset
        print('DataLoader loading json file: ', opt.input_json)
        self.info = json.load(open(self.opt.input_json))
        self.ix_to_word = self.info['ix_to_word']
        self.vocab_size = len(self.ix_to_word)

        print('vocab size is ', self.vocab_size)

        # open the hdf5 file
        print('DataLoader loading h5 file: ', opt.input_fc_dir, opt.input_att_dir, opt.input_box_dir,
              opt.input_label_h5)
        self.h5_label_file = h5py.File(self.opt.input_label_h5, 'r', driver='core')

        self.input_fc_dir = self.opt.input_fc_dir

        self.input_att_dir = self.opt.input_att_dir         #obj feature
        self.input_attr_dir = getattr(self.opt, 'input_attr_dir', self.input_att_dir)         #attr feature
        self.input_rela_dir = getattr(self.opt, 'input_rela_dir', self.input_att_dir)         #rela feature

        self.use_box = self.opt.use_box
        if self.use_box:
            self.input_box_dir = self.opt.input_box_dir

        # load in the sequence data
        seq_size = self.h5_label_file['labels'].shape
        print("seq_size:{0}".format(seq_size))
        self.seq_length = seq_size[1]
        print('max sequence length in data is', self.seq_length)
        # load the pointers in full to RAM (should be small enough)
        self.label_start_ix = self.h5_label_file['label_start_ix'][:]
        self.label_end_ix = self.h5_label_file['label_end_ix'][:]

        self.num_images = self.label_start_ix.shape[0]
        print('read %d image features' % (self.num_images))

        # separate out indexes for each of the provided splits
        self.split_ix = {'train': [], 'val': [], 'test': [], 'train_sg': [], 'val_sg': [], 'test_sg': []}
        for ix in range(len(self.info['images'])):
            img = self.info['images'][ix]
            if img['split'] == 'train':
                self.split_ix['train'].append(ix)
            elif img['split'] == 'val':
                self.split_ix['val'].append(ix)
                self.split_ix['train'].append(ix)
            elif img['split'] == 'test':
                self.split_ix['test'].append(ix)
            elif opt.train_only == 0:  # restval
                self.split_ix['train'].append(ix)

        print('assigned %d images to split train' % len(self.split_ix['train']))
        print('assigned %d images to split val' % len(self.split_ix['val']))
        print('assigned %d images to split test' % len(self.split_ix['test']))

        self.iterators = {'train': 0, 'val': 0, 'test': 0}

        self._prefetch_process = {}  # The three prefetch process
        for split in self.iterators.keys():
            self._prefetch_process[split] = BlobFetcher(split, self, split == 'train')
            # Terminate the child process when the parent exists

        def cleanup():
            print('Terminating BlobFetcher')
            for split in self.iterators.keys():
                del self._prefetch_process[split]

        import atexit
        atexit.register(cleanup)

    def get_captions(self, ix, seq_per_img):
        # fetch the sequence labels
        ix1 = self.label_start_ix[ix] - 1  # label_start_ix starts from 1
        ix2 = self.label_end_ix[ix] - 1
        ncap = ix2 - ix1 + 1  # number of captions available for this image
        assert ncap > 0, 'an image does not have any label. this can be handled but right now isn\'t'

        if ncap < seq_per_img:
            # we need to subsample (with replacement)
            seq = np.zeros([seq_per_img, self.seq_length], dtype='int')
            mod = np.zeros([seq_per_img, self.seq_length], dtype='int')
            for q in range(seq_per_img):
                ixl = random.randint(ix1, ix2)
                seq[q, :] = self.h5_label_file['labels'][ixl, :self.seq_length]
                mod[q, :] = self.h5_label_file['tags'][ixl, :self.seq_length]
        else:
            ixl = random.randint(ix1, ix2 - seq_per_img + 1)
            seq = self.h5_label_file['labels'][ixl: ixl + seq_per_img, :self.seq_length]
            mod = self.h5_label_file['tags'][ixl: ixl + seq_per_img, :self.seq_length]
        cont_weights = np.eye(4)[mod]
        return seq, mod, cont_weights

    def get_batch(self, split, batch_size=None, seq_per_img=None):
        batch_size = batch_size or self.batch_size
        seq_per_img = seq_per_img or self.seq_per_img

        att_batch = []  # np.ndarray((batch_size * seq_per_img, 14, 14, self.opt.att_feat_size), dtype = 'float32')
        attr_batch = []
        rela_batch = []
        box_batch = []

        label_batch = np.zeros([batch_size * seq_per_img, self.seq_length + 2], dtype='int')
        mod_batch = np.zeros([batch_size * seq_per_img, self.seq_length + 2], dtype='int')
        mask_batch = np.zeros([batch_size * seq_per_img, self.seq_length + 2], dtype='float32')
        cont_weights_batch = np.zeros([batch_size * seq_per_img, self.seq_length + 2, 4], dtype='float32')

        wrapped = False

        infos = []
        gts = []

        for i in range(batch_size):
            # fetch image
            tmp_rs, ix, tmp_wrapped = self._prefetch_process[split].get()

            att_batch.append(tmp_rs['att_feat'])
            attr_batch.append(tmp_rs['attr_feat'])
            rela_batch.append(tmp_rs['rela_feat'])
            if self.use_box:
                box_batch.append(tmp_rs['box_feat'])

            label_batch[i * seq_per_img: (i + 1) * seq_per_img, 1: self.seq_length + 1], \
            mod_batch[i * seq_per_img: (i + 1) * seq_per_img, 1: self.seq_length + 1], \
            cont_weights_batch[i * seq_per_img: (i + 1) * seq_per_img, 1: self.seq_length + 1,:] = \
                self.get_captions(ix, seq_per_img)

            if tmp_wrapped:
                wrapped = True

            # Used for reward evaluation
            gts.append(self.h5_label_file['labels'][self.label_start_ix[ix] - 1: self.label_end_ix[ix]])

            # record associated info as well
            info_dict = {}
            info_dict['ix'] = ix
            info_dict['id'] = self.info['images'][ix]['id']
            info_dict['file_path'] = self.info['images'][ix]['file_path']
            infos.append(info_dict)


        data = {}
        max_att_len = max([_.shape[0] for _ in att_batch])
        if self.use_box:
            max_box_len = max([_.shape[0] for _ in box_batch])
        max_attr_len = max([_.shape[0] for _ in attr_batch])
        max_rela_len = max([_.shape[0] for _ in rela_batch])

        # merge att_feats
        data['att_feats'] = np.zeros([len(att_batch) * seq_per_img, max_att_len, att_batch[0].shape[1]],
                                     dtype='float32')
        for i in range(len(att_batch)):
            data['att_feats'][i * seq_per_img:(i + 1) * seq_per_img, :att_batch[i].shape[0]] = att_batch[i]
        data['att_masks'] = np.zeros(data['att_feats'].shape[:2], dtype='float32')
        for i in range(len(att_batch)):
            data['att_masks'][i * seq_per_img:(i + 1) * seq_per_img, :att_batch[i].shape[0]] = 1

        if self.use_box:
            if len(box_batch) == len(rela_batch):
                data['box_feats'] = np.zeros([len(box_batch) * seq_per_img, max_box_len, box_batch[0].shape[1]],
                                         dtype='float32')
            else:
                print('rela len is not equal to box len')
            for i in range(len(box_batch)):
                data['box_feats'][i * seq_per_img:(i + 1) * seq_per_img, :box_batch[i].shape[0]] = box_batch[i]

        data['attr_feats'] = np.zeros([len(attr_batch) * seq_per_img, max_attr_len, attr_batch[0].shape[1]],
                                     dtype='float32')
        for i in range(len(attr_batch)):
            data['attr_feats'][i * seq_per_img:(i + 1) * seq_per_img, :attr_batch[i].shape[0]] = attr_batch[i]
        data['attr_masks'] = np.zeros(data['attr_feats'].shape[:2], dtype='float32')
        for i in range(len(attr_batch)):
            data['attr_masks'][i * seq_per_img:(i + 1) * seq_per_img, :attr_batch[i].shape[0]] = 1

        data['rela_feats'] = np.zeros([len(rela_batch) * seq_per_img, max_rela_len, rela_batch[0].shape[1]],
                                     dtype='float32')
        for i in range(len(rela_batch)):
            data['rela_feats'][i * seq_per_img:(i + 1) * seq_per_img, :rela_batch[i].shape[0]] = rela_batch[i]
        data['rela_masks'] = np.zeros(data['rela_feats'].shape[:2], dtype='float32')
        for i in range(len(rela_batch)):
            data['rela_masks'][i * seq_per_img:(i + 1) * seq_per_img, :rela_batch[i].shape[0]] = 1

        data['labels'] = np.vstack(label_batch)
        data['mods'] = np.vstack(mod_batch)
        data['cont_weights'] = cont_weights_batch
        # generate mask
        nonzeros = np.array(list(map(lambda x: (x != 0).sum() + 2, data['labels'])))
        for ix, row in enumerate(mask_batch):
            row[:nonzeros[ix]] = 1
        data['masks'] = mask_batch

        data['gts'] = gts  # all ground truth captions of each images
        data['bounds'] = {'it_pos_now': self.iterators[split], 'it_max': len(self.split_ix[split]), 'wrapped': wrapped}
        data['infos'] = infos

        return data

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        ix = index  # self.split_ix[index]
        rs_data = {} # reason data blob

        att_feat = np.load(os.path.join(self.input_att_dir, str(self.info['images'][ix]['id']) + '.npz'))['feat']
        att_feat = att_feat.reshape(-1, att_feat.shape[-1])
        rs_data['att_feat'] = att_feat

        if self.input_attr_dir == self.input_att_dir:
            rs_data['attr_feat'] = att_feat
        else:
            attr_feat = np.load(os.path.join(self.input_attr_dir, str(self.info['images'][ix]['id']) + '.npz'))['feat']
            attr_feat = attr_feat.reshape(-1, attr_feat.shape[-1])
            rs_data['attr_feat'] = attr_feat

        if self.input_rela_dir == self.input_att_dir:
            rs_data['rela_feat'] = att_feat
        else:
            rela_feat = np.load(os.path.join(self.input_rela_dir, str(self.info['images'][ix]['id']) + '.npz'))['feat']
            rela_feat = rela_feat.reshape(-1, rela_feat.shape[-1])
            rs_data['rela_feat'] = rela_feat

        if self.use_box:
            box_feat = np.load(os.path.join(self.input_box_dir, str(self.info['images'][ix]['id']) + '.npy'))
            # devided by image width and height
            x1, y1, x2, y2 = np.hsplit(box_feat, 4)
            h, w = self.info['images'][ix]['height'], self.info['images'][ix]['width']
            box_feat = np.hstack((x1 / w, y1 / h, x2 / w, y2 / h, (x2 - x1) * (y2 - y1) / (w * h)))
            rs_data['box_feat'] = box_feat

        return (rs_data,
                ix)

    def __len__(self):
        return len(self.info['images'])


class SubsetSampler(torch.utils.data.sampler.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.
    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


class BlobFetcher():
    """Experimental class for prefetching blobs in a separate process."""

    def __init__(self, split, dataloader, if_shuffle=False):
        """
        db is a list of tuples containing: imcrop_name, caption, bbox_feat of gt box, imname
        """
        self.split = split
        self.dataloader = dataloader
        self.if_shuffle = if_shuffle

    # Add more in the queue
    def reset(self):
        """
        Two cases for this function to be triggered:
        1. not hasattr(self, 'split_loader'): Resume from previous training. Create the dataset given the saved split_ix and iterator
        2. wrapped: a new epoch, the split_ix and iterator have been updated in the get_minibatch_inds already.
        """
        # batch_size is 1, the merge is done in DataLoader class
        self.split_loader = iter(data.DataLoader(dataset=self.dataloader,
                                                 batch_size=1,
                                                 sampler=SubsetSampler(self.dataloader.split_ix[self.split][
                                                                       self.dataloader.iterators[self.split]:]),
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 num_workers=4,  # 4 is usually enough
                                                 collate_fn=lambda x: x[0]))

    def _get_next_minibatch_inds(self):
        max_index = len(self.dataloader.split_ix[self.split])
        wrapped = False

        ri = self.dataloader.iterators[self.split]
        ix = self.dataloader.split_ix[self.split][ri]

        ri_next = ri + 1
        if ri_next >= max_index:
            ri_next = 0
            if self.if_shuffle:
                random.shuffle(self.dataloader.split_ix[self.split])
            wrapped = True
        self.dataloader.iterators[self.split] = ri_next

        return ix, wrapped

    def get(self):
        if not hasattr(self, 'split_loader'):
            self.reset()

        ix, wrapped = self._get_next_minibatch_inds()
        tmp = self.split_loader.next()
        if wrapped:
            self.reset()

        assert tmp[1] == ix, "ix not equal"

        return tmp + [wrapped]