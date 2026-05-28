#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 17/09/2018 6:45 PM

@author: Tangrizzly
"""

from __future__ import print_function
import time
import numpy as np
from numpy.random import uniform
import theano
import theano.tensor as T
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh
from theano.tensor import exp
from theano.tensor.extra_ops import Unique
from GRU import GruBasic

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


def softmax(x):
    # Legacy implementation note.
    # Legacy implementation note.
    # Legacy implementation note.
    e_x = exp(x - x.max(axis=0, keepdims=True))
    out = e_x / e_x.sum(axis=0, keepdims=True)
    return out


# One-by-one training.
# ======================================================================================================================
class OboCARNN(GruBasic):
    def __init__(self, train, test, dist, alpha_lambda, n_user, n_item, n_dists, n_in, n_hidden, ulptai):
        super(OboCARNN, self).__init__(train, test, alpha_lambda, n_user, n_item, n_in, n_hidden)
        # Legacy implementation note.
        self.ulptai = ulptai
        tra_dist_masks, tes_dist_masks, tra_dist_neg_masks = dist
        self.tra_dist_masks = theano.shared(borrow=True, value=np.asarray(tra_dist_masks, dtype='int32'))
        self.tes_dist_masks = theano.shared(borrow=True, value=np.asarray(tes_dist_masks, dtype='int32'))
        self.tra_dist_neg_masks = theano.shared(borrow=True, value=np.asarray(tra_dist_neg_masks, dtype='int32'))
        rang = 0.5
        M = uniform(-rang, rang, (n_hidden, n_in))
        self.M = theano.shared(borrow=True, value=M.astype(theano.config.floatX))
        # params --------------------------------------------------------------------------
        # Distance interval.
        n_dist, dd = n_dists
        self.dd = dd

        wd = uniform(-rang, rang, (n_dist + 1, n_hidden, n_in))   # Legacy implementation note.
        self.wd = theano.shared(borrow=True, value=wd.astype(theano.config.floatX))

        # Legacy implementation note.
        # Distance interval.
        trained_dists = uniform(-rang, rang, (n_dist + 1, n_hidden, n_in))
        self.trained_dists = theano.shared(borrow=True, value=trained_dists.astype(theano.config.floatX))
        # params：-----------------------------------------------------------------
        self.params = [self.M]
        self.l2_sqr = (
            T.sum(self.lt ** 2) +   # Legacy implementation note.
            T.sum(self.wd ** 2) +
            T.sum([T.sum(param ** 2) for param in self.params]))
        self.l2 = (
            0.5 * self.alpha_lambda[1] * self.l2_sqr)
        self.__theano_train__(n_in, n_hidden)
        self.__theano_predict__(n_in, n_hidden)

    def s_update_neg_masks(self, tra_buys_neg_masks, tes_buys_neg_masks, tra_dist_neg_masks):
        # Shuffle users each epoch.
        self.tra_buys_neg_masks.set_value(np.asarray(tra_buys_neg_masks, dtype='int32'), borrow=True)
        self.tes_buys_neg_masks.set_value(np.asarray(tes_buys_neg_masks, dtype='int32'), borrow=True)
        self.tra_dist_neg_masks.set_value(np.asarray(tra_dist_neg_masks, dtype='int32'), borrow=True)

    def update_trained_dists(self):
        # Legacy implementation note.
        wd = self.wd.get_value(borrow=True)
        self.trained_dists.set_value(np.asarray(wd, dtype=theano.config.floatX), borrow=True)  # update

    def compute_sub_all_scores(self, start_end):    # Legacy implementation note.
        # Legacy implementation note.
        shp1, shp2 = self.trained_items[:-1].shape
        shp0 = len(start_end)
        # T.dot(T.dot(wdp_t1, h_t), T.dot(M, xp_t1).T)
        # batch_size x n_node x n_hidden x n_hidden, batch_size x n_hidden
        h_W = T.sum(self.trained_dists[self.ulptai[start_end]] + self.trained_users[start_end].reshape((shp0, 1, 1, shp2)), 3)
        r_M = T.dot(self.trained_items[:-1], self.M.T).reshape((1, shp1, shp2))
        sub_all_scores = - T.sum(h_W + r_M, 2)
        return sub_all_scores.eval()                # shape=(sub_n_user, n_item)

    def __theano_train__(self, n_in, n_hidden):
        """Run one pass over the training sequence."""
        M = self.M

        tra_mask = T.ivector()
        seq_length = T.sum(tra_mask)  # Legacy implementation note.

        h0 = self.h0

        xpidxs = T.ivector()
        xqidxs = T.ivector()
        dpidxs = T.ivector()
        dqidxs = T.ivector()
        xps = self.lt[xpidxs]    # shape=(seq_length, n_in)
        xqs = self.lt[xqidxs]
        wdps = self.wd[dpidxs]
        wdqs = self.wd[dqidxs]

        pqs = T.concatenate((xpidxs, xqidxs))         # Legacy implementation note.
        uiq_pqs = Unique(False, False, False)(pqs)  # Legacy implementation note.
        uiq_x = self.lt[uiq_pqs]                    # Legacy implementation note.

        dpqs = T.concatenate((dpidxs, dqidxs))         # Legacy implementation note.
        uiq_dpqs = Unique(False, False, False)(dpqs)  # Legacy implementation note.
        uiq_d = self.wd[uiq_dpqs]                    # Legacy implementation note.

        def recurrence(x_t, xp_t1, xq_t1, wd_t, wdp_t1, wdq_t1,
                       h_t_pre1):
            # Legacy implementation note.
            h_t = sigmoid(T.dot(M, x_t) + T.dot(wd_t, h_t_pre1))
            yp = T.dot(T.dot(wdp_t1, h_t), T.dot(M, xp_t1).T)
            yq = T.dot(T.dot(wdq_t1, h_t), T.dot(M, xq_t1).T)
            loss_t_bpr = T.log(sigmoid(yp - yq))

            return [h_t, loss_t_bpr]

        [h, loss_bpr], _ = theano.scan(
            fn=recurrence,
            sequences=[xps, xps[1:], xqs[1:], wdps, wdps[1:], wdqs[1:]],
            outputs_info=[h0, None],
            n_steps=seq_length-1)

        # ----------------------------------------------------------------------------
        # cost, gradients, learning rate, l2 regularization
        lr, l2 = self.alpha_lambda[0], self.alpha_lambda[1]
        seq_l2_sq = T.sum([T.sum(par ** 2) for par in [xps, xqs, M, wdps, wdqs]])
        los = - T.sum(loss_bpr)
        seq_costs = (
            los +
            0.5 * l2 * seq_l2_sq)
        seq_grads = T.grad(seq_costs, self.params)
        seq_updates = [(par, par - lr * gra) for par, gra in zip(self.params, seq_grads)]
        update_x = T.set_subtensor(uiq_x, uiq_x - lr * T.grad(seq_costs, self.lt)[uiq_pqs])
        update_d = T.set_subtensor(uiq_d, uiq_d - lr * T.grad(seq_costs, self.wd)[uiq_dpqs])
        seq_updates.append((self.lt, update_x))     # Legacy implementation note.
        seq_updates.append((self.wd, update_d))
        # ----------------------------------------------------------------------------

        # Legacy implementation note.
        uidx = T.iscalar()  # Legacy implementation note.
        self.seq_train = theano.function(
            inputs=[uidx],
            outputs=los,
            updates=seq_updates,
            givens={
                xpidxs: self.tra_buys_masks[uidx],  # Legacy implementation note.
                xqidxs: self.tra_buys_neg_masks[uidx],  # negtive poi
                dpidxs: self.tra_dist_masks[uidx],  # Legacy implementation note.
                dqidxs: self.tra_dist_neg_masks[uidx],
                tra_mask: self.tra_masks[uidx]})

    def __theano_predict__(self, n_in, n_hidden):
        """Run the training sequence again during testing to obtain hidden states."""
        M = self.M

        tra_mask = T.imatrix()
        actual_batch_size = tra_mask.shape[0]
        seq_length = T.max(T.sum(tra_mask, axis=1))  # Compute mini-batch start/end index ranges.

        h0 = T.alloc(self.h0, actual_batch_size, n_hidden)  # shape=(n, 20)

        # Legacy implementation note.
        pidxs = T.imatrix()
        didxs = T.imatrix()
        ps = self.trained_items[pidxs]      # shape((actual_batch_size, seq_length, n_hidden))
        ps = ps.dimshuffle(1, 0, 2)
        wds = self.trained_dists[didxs]
        wds = wds.dimshuffle(1, 0, 2, 3)

        def recurrence(p_t, wd_t, h_t_pre1):
            h_t = sigmoid(T.dot(p_t, M.T) + T.sum(wd_t + h_t_pre1.reshape((h_t_pre1.shape[0], 1, h_t_pre1.shape[1])), 2))
            # wd_t: batch_size x n_hidden x n_hidden
            # h_t_pre1: batch_size x n_hidden
            return h_t

        h, _ = theano.scan(  # h.shape=(157, n, 20)
            fn=recurrence,
            sequences=[ps, wds],
            outputs_info=h0,
            n_steps=seq_length)

        # Legacy implementation note.
        # Legacy implementation note.
        hs = h.dimshuffle(1, 0, 2)                      # shape=(batch_size, seq_length, n_hidden)
        hts = hs[                                       # shape=(n, n_hidden)
            T.arange(actual_batch_size),                # Legacy implementation note.
            T.sum(tra_mask, axis=1) - 1]                # Legacy implementation note.

        # Legacy implementation note.
        start_end = T.ivector()
        self.seq_predict = theano.function(
            inputs=[start_end],
            outputs=hts,
            givens={
                pidxs: self.tra_buys_masks[start_end],  # Legacy implementation note.
                didxs: self.tra_dist_masks[start_end],
                tra_mask: self.tra_masks[start_end]})

    def train(self, idx):
        # consider the whole user sequence as a mini-batch and perform one update per sequence
        return self.seq_train(idx)


@exe_time  # Legacy implementation note.
def main():
    print('... construct the class: GRU')


if '__main__' == __name__:
    main()
