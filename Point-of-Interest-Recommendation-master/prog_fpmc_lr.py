#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from collections import OrderedDict     # Preserve insertion order when building dictionaries.
import time
import datetime
import numpy as np
import os
import random
from public.FPMC_LR import OboFpmc_lr
from public.Global_Best import GlobalBest
from public.Load_Data_fpmc_lr import load_data, fun_data_buys_masks
from public.Load_Data_fpmc_lr import fun_random_neg_masks_tes, fun_acquire_negs_tra
from public.Load_Data_fpmc_lr import fun_acquire_neighbors_for_each_poi
from public.Valuate import fun_predict_auc_recall_map_ndcg, fun_save_best_and_losses
__docformat__ = 'restructedtext en'

WHOLE = './poidata/'
PATH_f = os.path.join(WHOLE, 'Foursquare/sequence')
PATH_g = os.path.join(WHOLE, 'Gowalla/sequence')
PATH = PATH_g


def exe_time(func):
    def new_func(*args, **args2):
        name = func.__name__
        start = datetime.datetime.now()
        print("-- {%s} start: @ %ss" % (name, start))
        back = func(*args, **args2)
        end = datetime.datetime.now()
        print("-- {%s} start: @ %ss" % (name, start))
        print("-- {%s} end:   @ %ss" % (name, end))
        total = (end - start).total_seconds()
        print("-- {%s} total: @ %.3fs = %.3fh" % (name, total, total / 3600.0))
        return back
    return new_func


class Params(object):
    def __init__(self, p=None):
        """Build model parameters and load data."""
        # Initialize all parameters.
        if not p:
            t = 't'                       # Legacy implementation note.
            assert 't' == t or 'v' == t   # no other case
            p = OrderedDict(
                [
                    # ('dataset',             'Foursquare.txt'),
                    ('dataset',           'Gowalla.txt'),
                    ('mode',                'test' if 't' == t else 'valid'),

                    ('split',               -1 if 't' == t else -2),   # Legacy implementation note.
                    ('at_nums',             [5, 10, 15, 20]),
                    ('epochs',              200),

                    ('latent_size',         20),
                    ('alpha',               0.01),
                    ('lambda',              0.001),

                    ('UD',                  20),    # Distance truncation threshold.

                    ('mini_batch',          0),     # Legacy implementation note.

                    ('batch_size_train',    1),     #
                    ('batch_size_test',     32),   # Matrix operation note.
                ])
            for i in p.items():
                print(i)

        # Load data.
        # Matrix operation note.
        [(user_num, item_num), pois_cordis, (tra_pois, tes_pois), tra_last_poi] = \
            load_data(os.path.join(PATH, p['dataset']), p['mode'], p['split'])
        # Legacy implementation note.
        tes_pois_masks, tes_masks = fun_data_buys_masks(tes_pois, [item_num])
        tes_pois_neg_masks = fun_random_neg_masks_tes(item_num, tra_pois, tes_pois_masks)   # Used during prediction.
        # Legacy implementation note.
        # Distance truncation threshold.
        all_pois_neighbors = fun_acquire_neighbors_for_each_poi(pois_cordis, p['UD'])
        neis = [len(nei) for nei in all_pois_neighbors.values()]
        print(min(neis), sum(neis) / item_num)  # Legacy implementation note.
        # Legacy implementation note.
        # Legacy implementation note.
        tra_pois_negs = fun_acquire_negs_tra(tra_pois, all_pois_neighbors)

        # Create instance variables.
        self.p = p
        self.user_num, self.item_num = user_num, item_num
        self.pois_cordis = pois_cordis
        self.tra_pois = tra_pois
        self.tra_pois_negs = tra_pois_negs
        self.tra_last_poi = tra_last_poi
        self.tes_pois_masks, self.tes_masks = tes_pois_masks, tes_masks
        self.tes_pois_neg_masks = tes_pois_neg_masks

    def build_model_one_by_one(self):
        """Build the model object."""
        print('Building the model one_by_one ...')      # Legacy implementation note.
        p = self.p
        size = p['latent_size']
        model = OboFpmc_lr(
            train=[self.tra_pois, self.tra_pois_negs, self.tra_last_poi],
            test= [self.tes_pois_masks, self.tes_masks, self.tes_pois_neg_masks],
            alpha_lambda=[p['alpha'], p['lambda']],
            n_user=self.user_num,
            n_item=self.item_num,
            n_size=size)
        model_name = model.__class__.__name__
        print('\t the current Class name is: {val}'.format(val=model_name))
        return model, model_name

    def compute_start_end(self, flag):
        """Compute mini-batch start/end index ranges."""
        assert flag in ['train', 'test', 'test_auc']
        if 'train' == flag:
            size = self.p['batch_size_train']
        elif 'test' == flag:
            size = self.p['batch_size_test']        # test: top-k and acquire user vector
        else:
            size = self.p['batch_size_test'] * 10   # test: auc
        user_num = self.user_num
        rest = (user_num % size) > 0   # Legacy implementation note.
        n_batches = np.minimum(user_num // size + rest, user_num)
        batch_idxs = np.arange(n_batches, dtype=np.int32)
        starts_ends = []
        for bidx in batch_idxs:
            start = bidx * size
            end = np.minimum(start + size, user_num)   # Legacy implementation note.
            start_end = np.arange(start, end, dtype=np.int32)
            starts_ends.append(start_end)
        return batch_idxs, starts_ends


def train_valid_or_test():
    """Main program."""
    # Legacy implementation note.
    pas = Params()
    p = pas.p
    model, model_name = pas.build_model_one_by_one()
    best = GlobalBest(at_nums=p['at_nums'])   # Store best metric values.
    _, starts_ends_tes = pas.compute_start_end(flag='test')
    _, starts_ends_auc = pas.compute_start_end(flag='test_auc')

    # Unpack frequently used variables.
    user_num, item_num = pas.user_num, pas.item_num
    tra_pois, tra_pois_negs = pas.tra_pois, pas.tra_pois_negs
    tes_pois_masks, tes_masks, tes_pois_neg_masks = pas.tes_pois_masks, pas.tes_masks, pas.tes_pois_neg_masks
    del pas

    # Main training loop.
    losses = []
    times0, times1, times2, times3 = [], [], [], []
    for epoch in np.arange(p['epochs']):
        print("Epoch {val} ==================================".format(val=epoch))
        # Refresh negative samples and shuffle users each epoch.
        if epoch > 0:       # Initialization.
            tes_pois_neg_masks = fun_random_neg_masks_tes(item_num, tra_pois, tes_pois_masks)
            model.update_neg_masks(tes_pois_neg_masks)

        # ----------------------------------------------------------------------------------------------------------
        print("\tTraining ...")
        t0 = time.time()
        loss = 0.
        random.seed(str(123 + epoch))
        user_idxs_tra = np.arange(user_num, dtype=np.int32)
        random.shuffle(user_idxs_tra)       # Shuffle users each epoch.
        for uidx in user_idxs_tra:
            tra = tra_pois[uidx]            # list
            negs = tra_pois_negs[uidx]      # Legacy implementation note.
            for i in np.arange(len(tra)-1):     # Legacy implementation note.
                # Distance truncation threshold.
                loss += model.train(uidx, tra[i], tra[i+1], random.sample(negs[i+1], 1))
        rnn_l2_sqr = model.l2.eval()            # Legacy implementation note.
        print('\t\tsum_loss = {val} = {v1} - {v2}'.format(val=loss + rnn_l2_sqr, v1=loss, v2=rnn_l2_sqr))
        losses.append('{v1}'.format(v1=int(loss - rnn_l2_sqr)))
        t1 = time.time()
        times0.append(t1 - t0)

        # ----------------------------------------------------------------------------------------------------------
        print("\tPredicting ...")
        # Best metric values.
        fun_predict_auc_recall_map_ndcg(
            p, model, best, epoch, starts_ends_auc, starts_ends_tes, tes_pois_masks, tes_masks)
        best.fun_print_best(epoch)   # Legacy implementation note.
        t2 = time.time()
        times1.append(t2-t1)
        print('\tavg. time (train, test): %0.0fs,' % np.average(times0), '%0.0fs,' % np.average(times1),
              '| alpha, lam: {v1}'.format(v1=', '.join([str(lam) for lam in [p['alpha'], p['lambda']]])),
              '| model: {v1}'.format(v1=model_name))

        # ----------------------------------------------------------------------------------------------------------
        if epoch == p['epochs'] - 1:
            # Best metric values.
            print("\tBest and losses saving ...")
            path = os.path.join(os.path.split(__file__)[0], '..', 'Results_best_and_losses', PATH.split('/')[-2])
            fun_save_best_and_losses(path, model_name, epoch, p, best, losses)

    for i in p.items():
        print(i)
    print('\t the current Class name is: {val}'.format(val=model_name))


@exe_time
def main():
    train_valid_or_test()


if '__main__' == __name__:
    main()
