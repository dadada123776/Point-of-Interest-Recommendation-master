#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import time
import numpy as np
import random
from numpy.random import uniform
import theano
import theano.tensor as T
from theano.tensor.nnet import sigmoid
__docformat__ = 'restructedtext en'


def exe_time(func):
    def new_func(*args, **args2):
        t0 = time.time()
        print("-- @%s, {%s} start" % (time.strftime("%X", time.localtime()), func.__name__))
        back = func(*args, **args2)
        print("-- @%s, {%s} end" % (time.strftime("%X", time.localtime()), func.__name__))
        print("-- @%.3fs taken for {%s}" % (time.time() - t0, func.__name__))
        return back
    return new_func


# One-by-one training.
# ======================================================================================================================
class OboFpmc_lr(object):
    def __init__(self, train, test, alpha_lambda, n_user, n_item, n_size):
        """Build model parameters."""
        # train
        tra_buys, tra_buys_negs, tra_last_poi = train
        self.tra_buys = tra_buys
        self.tra_buys_negs = tra_buys_negs
        self.tra_last_poi = theano.shared(borrow=True, value=np.asarray(tra_last_poi, dtype='int32'))
        # test
        tes_buys_masks, tes_masks, tes_buys_neg_masks = test
        self.tes_buys_masks = theano.shared(borrow=True, value=np.asarray(tes_buys_masks, dtype='int32'))
        self.tes_masks = theano.shared(borrow=True, value=np.asarray(tes_masks, dtype='int32'))
        self.tes_buys_neg_masks = theano.shared(borrow=True, value=np.asarray(tes_buys_neg_masks, dtype='int32'))
        # Legacy implementation note.
        self.alpha_lambda = theano.shared(borrow=True, value=np.asarray(alpha_lambda, dtype=theano.config.floatX))
        # Initialization.
        rang = 0.5
        # Legacy implementation note.
        # au = uniform(-rang, rang, (n_item + 1, n_size))
        ui = uniform(-rang, rang, (n_user, n_size))
        iu = uniform(-rang, rang, (n_item + 1, n_size))
        ia = uniform(-rang, rang, (n_item + 1, n_size))
        ai = uniform(-rang, rang, (n_item + 1, n_size))
        # Matrix operation note.
        self.ui = theano.shared(borrow=True, value=ui.astype(theano.config.floatX))
        self.iu = theano.shared(borrow=True, value=iu.astype(theano.config.floatX))
        self.ai = theano.shared(borrow=True, value=ai.astype(theano.config.floatX))
        self.ia = theano.shared(borrow=True, value=ia.astype(theano.config.floatX))
        # Legacy implementation note.
        self.params = [self.ui, self.iu, self.ai, self.ia]
        self.l2_sqr = (
            T.sum([T.sum(param ** 2) for param in self.params]))
        self.l2 = (
            0.5 * self.alpha_lambda[1] * self.l2_sqr)
        self.__theano_train__(n_size)

    def update_neg_masks(self, tes_buys_neg_masks):
        # Shuffle users each epoch.
        self.tes_buys_neg_masks.set_value(np.asarray(tes_buys_neg_masks, dtype='int32'), borrow=True)

    def compute_sub_all_scores(self, start_end):    # Legacy implementation note.
        # Legacy implementation note.
        # Score computation note.
        # Score computation note.
        sub_all_scores = T.dot(self.ui[start_end], self.iu[:-1].T) + \
                         T.dot(self.ai[self.tra_last_poi[start_end]], self.ia[:-1].T)
        return sub_all_scores.eval()                # shape=(sub_n_user, n_item)

    def compute_sub_auc_preference(self, start_end):
        # Legacy implementation note.
        # Matrix operation note.
        # Legacy implementation note.
        users = self.ui[start_end]
        tes_usrs = self.iu[self.tes_buys_masks[start_end]]  # shape=(sub_n_user, len(tes_mask[0]), n_hidden)
        tes_usrs_neg = self.iu[self.tes_buys_neg_masks[start_end]]
        # Legacy implementation note.
        pois_pre1 = self.ai[self.tra_last_poi[start_end]]
        tes_items = self.ia[self.tes_buys_masks[start_end]]  # shape=(sub_n_user, len(tes_mask[0]), n_hidden)
        tes_items_neg = self.ia[self.tes_buys_neg_masks[start_end]]
        shp0, shp2 = users.shape        # shape=(sub_n_user, n_hidden)
        # Legacy implementation note.
        # Legacy implementation note.
        # Legacy implementation note.
        all_upqs = T.sum(users.reshape((shp0, 1, shp2)) * (tes_usrs - tes_usrs_neg), axis=2) + \
            T.sum(pois_pre1.reshape((shp0, 1, shp2)) * (tes_items - tes_items_neg), axis=2)
        all_upqs *= self.tes_masks[start_end]       # Legacy implementation note.
        return all_upqs.eval() > 0                  # Legacy implementation note.

    def __theano_train__(self, n_size):
        """Run one pass over the training sequence."""
        # self.alpha_lambda = ['alpha', 'lambda']
        uidx = T.iscalar()
        aidx = T.iscalar()
        tidxs = T.ivector()     # Legacy implementation note.

        ui = self.ui[uidx]
        ai = self.ai[aidx]
        tu = self.iu[tidxs]
        ta = self.ia[tidxs]
        """Matrix operation note."""
        upq = T.dot(tu[0]-tu[1:], ui) + T.dot(ta[0]-ta[1:], ai)   # shape=(le, )
        los = T.sum(T.log(sigmoid(upq)))

        # ----------------------------------------------------------------------------
        # cost, gradients, learning rate, l2 regularization
        lr, l2 = self.alpha_lambda[0], self.alpha_lambda[1]
        seq_l2_sq = T.sum([T.sum(par ** 2) for par in [ui, ai, tu, ta]])
        costs = (
            los -
            0.5 * l2 * seq_l2_sq)
        pars_subs = [(self.ui, ui), (self.ai, ai),
                     (self.iu, tu), (self.ia, ta)]
        updates = [(par, T.set_subtensor(sub, sub + lr * T.grad(costs, sub)))
                   for par, sub in pars_subs]
        # ----------------------------------------------------------------------------

        # Legacy implementation note.
        self.seq_train = theano.function(
            inputs=[uidx, aidx, tidxs],
            outputs=los,
            updates=updates)

    def train(self, uidx, aidx, iidx, jidxs):
        # consider the whole user sequence as a mini-batch and perform one update per sequence
        """Legacy implementation note."""
        return self.seq_train(uidx, aidx, [iidx] + jidxs)


@exe_time
def main():
    print('... construct the class: FPMC_LR')


if '__main__' == __name__:
    main()

