#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import time
import numpy as np
from numpy.random import uniform
import theano
import theano.tensor as T
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh
from theano.tensor.shared_randomstreams import RandomStreams
# from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.extra_ops import Unique
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


# Legacy implementation note.
# Used during prediction.
# ======================================================================================================================
class GruBasic(object):
    def __init__(self, train, test, alpha_lambda, n_user, n_item, n_in, n_hidden):
        """Build model parameters."""
        # Legacy implementation note.
        rng = np.random.RandomState(123)
        self.thea_rng = RandomStreams(rng.randint(2 ** 30))     # Legacy implementation note.
        # Legacy implementation note.
        tra_buys_masks, tra_masks, tra_buys_neg_masks = train
        tes_buys_masks, tes_masks, tes_buys_neg_masks = test
        self.tra_buys_masks = theano.shared(borrow=True, value=np.asarray(tra_buys_masks, dtype='int32'))
        self.tes_buys_masks = theano.shared(borrow=True, value=np.asarray(tes_buys_masks, dtype='int32'))
        self.tra_masks = theano.shared(borrow=True, value=np.asarray(tra_masks, dtype='int32'))
        self.tes_masks = theano.shared(borrow=True, value=np.asarray(tes_masks, dtype='int32'))
        self.tra_buys_neg_masks = theano.shared(borrow=True, value=np.asarray(tra_buys_neg_masks, dtype='int32'))
        self.tes_buys_neg_masks = theano.shared(borrow=True, value=np.asarray(tes_buys_neg_masks, dtype='int32'))
        # Legacy implementation note.
        self.alpha_lambda = theano.shared(borrow=True, value=np.asarray(alpha_lambda, dtype=theano.config.floatX))
        # Initialization.
        rang = 0.5
        lt = uniform(-rang, rang, (n_item + 1, n_in))   # Legacy implementation note.
        ui = uniform(-rang, rang, (3, n_hidden, n_in))
        wh = uniform(-rang, rang, (3, n_hidden, n_hidden))
        h0 = np.zeros((n_hidden, ), dtype=theano.config.floatX)
        bi = np.zeros((3, n_hidden), dtype=theano.config.floatX)
        # Legacy implementation note.
        self.lt = theano.shared(borrow=True, value=lt.astype(theano.config.floatX))
        self.ui = theano.shared(borrow=True, value=ui.astype(theano.config.floatX))
        self.wh = theano.shared(borrow=True, value=wh.astype(theano.config.floatX))
        self.h0 = theano.shared(borrow=True, value=h0)
        self.bi = theano.shared(borrow=True, value=bi)
        # Legacy implementation note.
        trained_items = uniform(-rang, rang, (n_item + 1, n_hidden))
        trained_users = uniform(-rang, rang, (n_user, n_hidden))
        self.trained_items = theano.shared(borrow=True, value=trained_items.astype(theano.config.floatX))
        self.trained_users = theano.shared(borrow=True, value=trained_users.astype(theano.config.floatX))
        # Legacy implementation note.
        # self.__theano_predict__(n_in, n_hidden)

    def update_neg_masks(self, tra_buys_neg_masks, tes_buys_neg_masks):
        # Shuffle users each epoch.
        self.tra_buys_neg_masks.set_value(np.asarray(tra_buys_neg_masks, dtype='int32'), borrow=True)
        self.tes_buys_neg_masks.set_value(np.asarray(tes_buys_neg_masks, dtype='int32'), borrow=True)

    def update_trained_items(self):
        # Legacy implementation note.
        lt = self.lt.get_value(borrow=True)    # Legacy implementation note.
        self.trained_items.set_value(np.asarray(lt, dtype=theano.config.floatX), borrow=True)     # update

    def update_trained_users(self, all_hus):
        # Legacy implementation note.
        self.trained_users.set_value(np.asarray(all_hus, dtype=theano.config.floatX), borrow=True)  # update

    def compute_sub_all_scores(self, start_end):    # Legacy implementation note.
        # Legacy implementation note.
        sub_all_scores = T.dot(self.trained_users[start_end], self.trained_items[:-1].T)
        return sub_all_scores.eval()                # shape=(sub_n_user, n_item)

    def compute_sub_auc_preference(self, start_end):
        # Legacy implementation note.
        # Matrix operation note.
        tes_items = self.trained_items[self.tes_buys_masks[start_end]]  # shape=(sub_n_user, len(tes_mask[0]), n_hidden)
        tes_items_neg = self.trained_items[self.tes_buys_neg_masks[start_end]]
        users = self.trained_users[start_end]
        shp0, shp2 = users.shape        # shape=(sub_n_user, n_hidden)
        # Legacy implementation note.
        # Legacy implementation note.
        # Legacy implementation note.
        all_upqs = T.sum(users.reshape((shp0, 1, shp2)) * (tes_items - tes_items_neg), axis=2)
        all_upqs *= self.tes_masks[start_end]       # Legacy implementation note.
        return all_upqs.eval() > 0                  # Legacy implementation note.

    def get_corrupted_input_whole(self, inp, corruption_prob):
        # Matrix operation note.
        # Legacy implementation note.
        # Matrix operation note.
        # if corruption_prob < 0. or corruption_prob >= 1.:
        #     raise Exception('Drop prob must be in interval [0, 1)')
        retain_prob = 1. - corruption_prob
        randoms = self.thea_rng.binomial(
            size=(inp.shape[0], 1),     # shape=(num, 1)
            n=1,
            p=retain_prob,             # Legacy implementation note.
            dtype=theano.config.floatX)
        randoms = T.Rebroadcast((1, True))(randoms)
        return inp * randoms            # shape=(num, 1024)

    def get_corrupted_input_whole_minibatch(self, inp, corruption_prob):
        # Legacy implementation note.
        # Matrix operation note.
        retain_prob = 1. - corruption_prob
        randoms = self.thea_rng.binomial(
            size=(inp.shape[0], inp.shape[1], 1),     # shape=(seq_length, batch_size, 1)
            n=1,
            p=retain_prob,             # Legacy implementation note.
            dtype=theano.config.floatX)
        randoms = T.Rebroadcast((2, True))(randoms)
        return inp * randoms            # shape=(seq_length, batch_size, 1024)

    def dropout(self, inp, drop_prob):
        # Legacy implementation note.
        # Legacy implementation note.
        # Legacy implementation note.
        # if drop_prob < 0. or drop_prob >= 1.:
        #     raise Exception('Drop prob must be in interval [0, 1)')
        retain_prob = 1. - drop_prob      # Legacy implementation note.
        randoms = self.thea_rng.binomial(
            size=inp.shape,     # Vector dimension.
            n=1,                # Legacy implementation note.
            p=retain_prob)     # Legacy implementation note.
        inp *= randoms          # Legacy implementation note.
        inp /= retain_prob     # Legacy implementation note.
        return inp              # Legacy implementation note.

    def __theano_predict__(self, n_in, n_hidden):
        """Run the training sequence again during testing to obtain hidden states."""
        ui, wh = self.ui, self.wh

        tra_mask = T.imatrix()
        actual_batch_size = tra_mask.shape[0]
        seq_length = T.max(T.sum(tra_mask, axis=1))     # Compute mini-batch start/end index ranges.

        h0 = T.alloc(self.h0, actual_batch_size, n_hidden)      # shape=(n, 20)
        bi = T.alloc(self.bi, actual_batch_size, 3, n_hidden)   # Vector dimension.
        bi = bi.dimshuffle(1, 2, 0)                             # shape=(3, 20, n)

        # Legacy implementation note.
        pidxs = T.imatrix()
        ps = self.trained_items[pidxs]      # shape((actual_batch_size, seq_length, n_hidden))
        ps = ps.dimshuffle(1, 0, 2)         # shape=(seq_length, batch_size, n_hidden)=(157, n, 20)

        def recurrence(p_t, h_t_pre1):
            # Legacy implementation note.
            z_r = sigmoid(T.dot(ui[:2], p_t.T) +
                          T.dot(wh[:2], h_t_pre1.T) + bi[:2])
            z, r = z_r[0].T, z_r[1].T                           # shape=(n, 20)
            c = tanh(T.dot(ui[2], p_t.T) +
                     T.dot(wh[2], (r * h_t_pre1).T) + bi[2])    # shape=(20, n)
            h_t = (T.ones_like(z) - z) * h_t_pre1 + z * c.T     # shape=(n, 20)
            return h_t
        h, _ = theano.scan(         # h.shape=(157, n, 20)
            fn=recurrence,
            sequences=ps,
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
                pidxs: self.tra_buys_masks[start_end],       # Legacy implementation note.
                tra_mask: self.tra_masks[start_end]})

    def predict(self, idxs):
        return self.seq_predict(idxs)


# Legacy implementation note.
# Legacy implementation note.
# ======================================================================================================================
class GruBasic2Units(GruBasic):
    def __init__(self, train, test, alpha_lambda, n_user, n_item, n_in, n_hidden):
        super(GruBasic2Units, self).__init__(train, test, alpha_lambda, n_user, n_item, n_in, n_hidden)
        # Initialization.
        rang = 0.5
        lt = uniform(-rang, rang, (n_item + 1, n_in))   # Legacy implementation note.
        mm = uniform(-rang, rang, (n_item + 1, n_in))   # Legacy implementation note.
        uix = uniform(-rang, rang, (3, n_hidden, n_hidden))
        uim = uniform(-rang, rang, (3, n_hidden, n_hidden))
        whx = uniform(-rang, rang, (3, n_hidden, n_hidden))
        whm = uniform(-rang, rang, (3, n_hidden, n_hidden))
        h0x = np.zeros((n_hidden, ), dtype=theano.config.floatX)
        h0m = np.zeros((n_hidden, ), dtype=theano.config.floatX)
        bix = np.zeros((3, n_hidden), dtype=theano.config.floatX)
        bim = np.zeros((3, n_hidden), dtype=theano.config.floatX)
        # Legacy implementation note.
        self.lt = theano.shared(borrow=True, value=lt.astype(theano.config.floatX))
        self.mm = theano.shared(borrow=True, value=mm.astype(theano.config.floatX))
        self.uix = theano.shared(borrow=True, value=uix.astype(theano.config.floatX))
        self.uim = theano.shared(borrow=True, value=uim.astype(theano.config.floatX))
        self.whx = theano.shared(borrow=True, value=whx.astype(theano.config.floatX))
        self.whm = theano.shared(borrow=True, value=whm.astype(theano.config.floatX))
        self.h0x = theano.shared(borrow=True, value=h0x)
        self.h0m = theano.shared(borrow=True, value=h0m)
        self.bix = theano.shared(borrow=True, value=bix)
        self.bim = theano.shared(borrow=True, value=bim)
        # Legacy implementation note.
        # self.__theano_predict__(n_in, n_hidden)

    def __theano_predict__(self, n_in, n_hidden):
        """Run the training sequence again during testing to obtain hidden states."""
        uix, whx = self.uix, self.whx
        uim, whm = self.uim, self.whm

        tra_mask = T.imatrix()          # shape=(n, 157)
        actual_batch_size = tra_mask.shape[0]
        seq_length = T.max(T.sum(tra_mask, axis=1))     # Compute mini-batch start/end index ranges.

        h0x = T.alloc(self.h0x, actual_batch_size, n_hidden)
        h0m = T.alloc(self.h0m, actual_batch_size, n_hidden)
        bix = T.alloc(self.bix, actual_batch_size, 3, n_hidden)     # Vector dimension.
        bim = T.alloc(self.bim, actual_batch_size, 3, n_hidden)     # Vector dimension.
        bix = bix.dimshuffle(1, 2, 0)                               # shape=(3, 20, n)
        bim = bim.dimshuffle(1, 2, 0)                               # shape=(3, 20, n)

        # Legacy implementation note.
        pidxs = T.imatrix()
        xps, mps = self.lt[pidxs], self.mm[pidxs]                   # shape((actual_batch_size, seq_length, n_in))
        xps, mps = xps.dimshuffle(1, 0, 2), mps.dimshuffle(1, 0, 2) # shape=(seq_length, batch_size, n_in)

        def recurrence(xp_t, mp_t, hx_t_pre1, hm_t_pre1):
            # Legacy implementation note.
            # Legacy implementation note.
            z_rx = sigmoid(T.dot(uix[:2], xp_t.T) + T.dot(whx[:2], hx_t_pre1.T) + bix[:2])    # shape=(2, 20, n)
            z_rm = sigmoid(T.dot(uim[:2], mp_t.T) + T.dot(whm[:2], hm_t_pre1.T) + bim[:2])    # shape=(2, 20, n)
            zx, rx = z_rx[0].T, z_rx[1].T                   # shape=(n, 20)
            zm, rm = z_rm[0].T, z_rm[1].T                   # shape=(n, 20)
            cx = tanh(T.dot(uix[2], xp_t.T) + T.dot(whx[2], (rx * hx_t_pre1).T) + bix[2])    # shape=(20, n)
            cm = tanh(T.dot(uim[2], mp_t.T) + T.dot(whm[2], (rm * hm_t_pre1).T) + bim[2])    # shape=(20, n)
            hx_t = (T.ones_like(zx) - zx) * hx_t_pre1 + zx * cx.T        # shape=(n, 20)
            hm_t = (T.ones_like(zm) - zm) * hm_t_pre1 + zm * cm.T        # shape=(n, 20)
            return [hx_t, hm_t]
        [hx, hm], _ = theano.scan(              # h.shape=(157, n, 20)
            fn=recurrence,
            sequences=[xps, mps],
            outputs_info=[h0x, h0m],
            n_steps=seq_length)
        h = T.concatenate((hx, hm), axis=2)     # h.shape=(157, n, 40)

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
                pidxs: self.tra_buys_masks[start_end],       # Legacy implementation note.
                tra_mask: self.tra_masks[start_end]})


# One-by-one training.
# ======================================================================================================================
class OboGru(GruBasic):
    def __init__(self, train, test, alpha_lambda, n_user, n_item, n_in, n_hidden):
        super(OboGru, self).__init__(train, test, alpha_lambda, n_user, n_item, n_in, n_hidden)
        self.params = [self.ui, self.wh, self.bi]       # Legacy implementation note.
        self.l2_sqr = (
            T.sum(self.lt ** 2) +
            T.sum([T.sum(param ** 2) for param in self.params]))
        self.l2 = (
            0.5 * self.alpha_lambda[1] * self.l2_sqr)
        self.__theano_train__(n_in, n_hidden)
        self.__theano_predict__(n_in, n_hidden)

    def __theano_train__(self, n_in, n_hidden):
        """Run one pass over the training sequence."""
        # self.alpha_lambda = ['alpha', 'lambda']
        ui, wh = self.ui, self.wh

        tra_mask = T.ivector()
        seq_length = T.sum(tra_mask)                # Legacy implementation note.

        h0 = self.h0
        bi = self.bi

        pidxs, qidxs = T.ivector(), T.ivector()
        xps, xqs = self.lt[pidxs], self.lt[qidxs]   # shape((seq_length, n_in))

        pqs = T.concatenate((pidxs, qidxs))         # Legacy implementation note.
        uiq_pqs = Unique(False, False, False)(pqs)  # Legacy implementation note.
        uiq_x = self.lt[uiq_pqs]                    # Legacy implementation note.

        """Compute the current hidden state and loss at time step t."""
        def recurrence(xp_t, xq_t, h_t_pre1):
            z_r = sigmoid(T.dot(ui[:2], xp_t) +
                          T.dot(wh[:2], h_t_pre1) + bi[:2])
            z, r = z_r[0], z_r[1]
            c = tanh(T.dot(ui[2], xp_t) +
                     T.dot(wh[2], (r * h_t_pre1)) + bi[2])
            h_t = (T.ones_like(z) - z) * h_t_pre1 + z * c
            upq_t = T.dot(h_t_pre1, xp_t - xq_t)
            loss_t = T.log(sigmoid(upq_t))
            return [h_t, loss_t]
        [h, loss], _ = theano.scan(
            fn=recurrence,
            sequences=[xps, xqs],
            outputs_info=[h0, None],
            n_steps=seq_length,
            truncate_gradient=-1)

        # ----------------------------------------------------------------------------
        # cost, gradients, learning rate, l2 regularization
        lr, l2 = self.alpha_lambda[0], self.alpha_lambda[1]
        seq_l2_sq = T.sum([T.sum(par ** 2) for par in [xps, xqs, ui, wh, bi]])
        upq = T.sum(loss)
        seq_costs = (
            - upq +
            0.5 * l2 * seq_l2_sq)
        seq_grads = T.grad(seq_costs, self.params)
        seq_updates = [(par, par - lr * gra) for par, gra in zip(self.params, seq_grads)]
        update_x = T.set_subtensor(uiq_x, uiq_x - lr * T.grad(seq_costs, self.lt)[uiq_pqs])
        seq_updates.append((self.lt, update_x))     # Legacy implementation note.
        # ----------------------------------------------------------------------------

        # Legacy implementation note.
        uidx = T.iscalar()                              # Legacy implementation note.
        self.seq_train = theano.function(
            inputs=[uidx],
            outputs=-upq,
            updates=seq_updates,
            givens={
                pidxs: self.tra_buys_masks[uidx],       # Legacy implementation note.
                qidxs: self.tra_buys_neg_masks[uidx],
                tra_mask: self.tra_masks[uidx]})

    def train(self, idx):
        # consider the whole user sequence as a mini-batch and perform one update per sequence
        return self.seq_train(idx)


# Legacy implementation note.
# Legacy implementation note.
# ======================================================================================================================
class Gru(GruBasic):
    def __init__(self, train, test, alpha_lambda, n_user, n_item, n_in, n_hidden):
        super(Gru, self).__init__(train, test, alpha_lambda, n_user, n_item, n_in, n_hidden)
        self.params = [self.ui, self.wh, self.bi]       # Legacy implementation note.
        self.l2_sqr = (
            T.sum(self.lt ** 2) +
            T.sum([T.sum(param ** 2) for param in self.params]))
        self.l2 = (
            0.5 * self.alpha_lambda[1] * self.l2_sqr)
        self.__theano_train__(n_hidden)
        self.__theano_predict__(n_in, n_hidden)

    def __theano_train__(self, n_hidden):
        """Run one pass over the training sequence."""
        # self.alpha_lambda = ['alpha', 'lambda']
        ui, wh = self.ui, self.wh

        tra_mask = T.imatrix()                          # shape=(n, 157)
        actual_batch_size = tra_mask.shape[0]
        seq_length = T.max(T.sum(tra_mask, axis=1))     # Compute mini-batch start/end index ranges.
        mask = tra_mask.T                               # shape=(157, n)

        h0 = T.alloc(self.h0, actual_batch_size, n_hidden)      # shape=(n, 20)
        bi = T.alloc(self.bi, actual_batch_size, 3, n_hidden)   # Legacy implementation note.
        bi = bi.dimshuffle(1, 2, 0)                             # shape=(3, 20, n)

        pidxs, qidxs = T.imatrix(), T.imatrix()         # TensorType(int32, matrix)
        xps, xqs = self.lt[pidxs], self.lt[qidxs]       # shape((actual_batch_size, seq_length, n_in))
        xps, xqs = xps.dimshuffle(1, 0, 2), xqs.dimshuffle(1, 0, 2)     # shape=(seq_length, batch_size, n_in)

        pqs = T.concatenate((pidxs, qidxs))         # Legacy implementation note.
        uiq_pqs = Unique(False, False, False)(pqs)  # Legacy implementation note.
        uiq_x = self.lt[uiq_pqs]                    # Legacy implementation note.

        """Compute the current hidden state and loss at time step t."""
        def recurrence(xp_t, xq_t, mask_t, h_t_pre1):
            # Legacy implementation note.
            z_r = sigmoid(T.dot(ui[:2], xp_t.T) +
                          T.dot(wh[:2], h_t_pre1.T) + bi[:2])   # shape=(2, 20, n)
            z, r = z_r[0].T, z_r[1].T                           # shape=(n, 20)
            c = tanh(T.dot(ui[2], xp_t.T) +
                     T.dot(wh[2], (r * h_t_pre1).T) + bi[2])    # shape=(20, n)
            h_t = (T.ones_like(z) - z) * h_t_pre1 + z * c.T     # shape=(n, 20)
            # Legacy implementation note.
            upq_t = T.sum(h_t_pre1 * (xp_t - xq_t), axis=1)     # shape=(n, )
            loss_t = T.log(sigmoid(upq_t))                      # shape=(n, )
            loss_t *= mask_t                            # Legacy implementation note.
            return [h_t, loss_t]                        # shape=(n, 20), (n, )
        [h, loss], _ = theano.scan(
            fn=recurrence,
            sequences=[xps, xqs, mask],
            outputs_info=[h0, None],
            n_steps=seq_length)     # Legacy implementation note.

        # ----------------------------------------------------------------------------
        # cost, gradients, learning rate, l2 regularization
        lr, l2 = self.alpha_lambda[0], self.alpha_lambda[1]
        seq_l2_sq = (
            T.sum([T.sum(par ** 2) for par in [xps, xqs, ui, wh]]) +
            T.sum([T.sum(par ** 2) for par in [bi]]) / actual_batch_size)
        upq = T.sum(loss)
        seq_costs = (
            - upq / actual_batch_size +
            0.5 * l2 * seq_l2_sq)       # Legacy implementation note.
        seq_grads = T.grad(seq_costs, self.params)
        seq_updates = [(par, par - lr * gra) for par, gra in zip(self.params, seq_grads)]
        update_x = T.set_subtensor(uiq_x, uiq_x - lr * T.grad(seq_costs, self.lt)[uiq_pqs])
        seq_updates.append((self.lt, update_x))     # Legacy implementation note.
        # ----------------------------------------------------------------------------

        # Legacy implementation note.
        # Legacy implementation note.
        start_end = T.ivector()     # int32
        self.seq_train = theano.function(
            inputs=[start_end],
            outputs=-upq,
            updates=seq_updates,
            givens={
                pidxs: self.tra_buys_masks[start_end],       # Legacy implementation note.
                qidxs: self.tra_buys_neg_masks[start_end],   # Legacy implementation note.
                tra_mask: self.tra_masks[start_end]})

        # Legacy implementation note.
        self.normalize = theano.function(
            inputs=[],
            updates={
                self.lt: self.lt / T.sqrt(T.sum(self.lt ** 2, axis=1).dimshuffle(0, 'x'))
            })

    def train(self, idxs):
        return self.seq_train(idxs)


# ======================================================================================================================
class Lstm(GruBasic):
    def __init__(self, train, test, alpha_lambda, n_user, n_item, n_in, n_hidden):
        super(Lstm, self).__init__(train, test, alpha_lambda, n_user, n_item, n_in, n_hidden)
        # Initialization.
        rang = 0.5
        ui = uniform(-rang, rang, (4, n_hidden, n_hidden))
        wh = uniform(-rang, rang, (4, n_hidden, n_hidden))
        c0 = np.zeros((n_hidden, ), dtype=theano.config.floatX)
        bi = np.zeros((4, n_hidden), dtype=theano.config.floatX)
        # Legacy implementation note.
        self.ui = theano.shared(borrow=True, value=ui.astype(theano.config.floatX))
        self.wh = theano.shared(borrow=True, value=wh.astype(theano.config.floatX))
        self.c0 = theano.shared(borrow=True, value=c0)
        self.bi = theano.shared(borrow=True, value=bi)
        self.params = [self.ui, self.wh, self.bi]       # Legacy implementation note.
        self.l2_sqr = (
            T.sum(self.lt ** 2) +
            T.sum([T.sum(param ** 2) for param in self.params]))
        self.l2 = (
            0.5 * self.alpha_lambda[1] * self.l2_sqr)
        self.__theano_train__(n_hidden)
        self.__theano_predict__(n_in, n_hidden)

    def __theano_train__(self, n_hidden):
        """Run one pass over the training sequence."""
        # self.alpha_lambda = ['alpha', 'lambda']
        ui, wh = self.ui, self.wh

        tra_mask = T.imatrix()                          # shape=(n, 157)
        actual_batch_size = tra_mask.shape[0]
        seq_length = T.max(T.sum(tra_mask, axis=1))     # Compute mini-batch start/end index ranges.
        mask = tra_mask.T                               # shape=(157, n)

        h0 = T.alloc(self.h0, actual_batch_size, n_hidden)      # shape=(n, 20)
        c0 = T.alloc(self.c0, actual_batch_size, n_hidden)      # shape=(n, 20)
        bi = T.alloc(self.bi, actual_batch_size, 4, n_hidden)   # Legacy implementation note.
        bi = bi.dimshuffle(1, 2, 0)                             # shape=(3, 20, n)

        pidxs, qidxs = T.imatrix(), T.imatrix()         # TensorType(int32, matrix)
        xps, xqs = self.lt[pidxs], self.lt[qidxs]       # shape((actual_batch_size, seq_length, n_in))
        xps, xqs = xps.dimshuffle(1, 0, 2), xqs.dimshuffle(1, 0, 2)     # shape=(seq_length, batch_size, n_in)

        pqs = T.concatenate((pidxs, qidxs))         # Legacy implementation note.
        uiq_pqs = Unique(False, False, False)(pqs)  # Legacy implementation note.
        uiq_x = self.lt[uiq_pqs]                    # Legacy implementation note.

        """Compute the current hidden state and loss at time step t."""
        def recurrence(xp_t, xq_t, mask_t, c_t_pre1, h_t_pre1):
            # Legacy implementation note.
            gates = T.dot(ui, xp_t.T) + T.dot(wh, h_t_pre1.T) + bi  # shape=(4, 20, n)
            i, f, g, o = sigmoid(gates[0]).T, sigmoid(gates[1]).T, tanh(gates[2]).T, sigmoid(gates[3]).T
            c_t = f * c_t_pre1 + i * g
            h_t = o * tanh(c_t)   # shape=(n, 20)
            # Legacy implementation note.
            upq_t = T.sum(h_t_pre1 * (xp_t - xq_t), axis=1)     # shape=(n, )
            loss_t = T.log(sigmoid(upq_t))                      # shape=(n, )
            loss_t *= mask_t                            # Legacy implementation note.
            return [c_t, h_t, loss_t]                        # shape=(n, 20), (n, )
        [c, h, loss], _ = theano.scan(
            fn=recurrence,
            sequences=[xps, xqs, mask],
            outputs_info=[c0, h0, None],
            n_steps=seq_length)     # Legacy implementation note.

        # ----------------------------------------------------------------------------
        # cost, gradients, learning rate, l2 regularization
        lr, l2 = self.alpha_lambda[0], self.alpha_lambda[1]
        seq_l2_sq = (
            T.sum([T.sum(par ** 2) for par in [xps, xqs, ui, wh]]) +
            T.sum([T.sum(par ** 2) for par in [bi]]) / actual_batch_size)
        upq = T.sum(loss)
        seq_costs = (
            - upq / actual_batch_size +
            0.5 * l2 * seq_l2_sq)
        seq_grads = T.grad(seq_costs, self.params)
        seq_updates = [(par, par - lr * gra) for par, gra in zip(self.params, seq_grads)]
        update_x = T.set_subtensor(uiq_x, uiq_x - lr * T.grad(seq_costs, self.lt)[uiq_pqs])
        seq_updates.append((self.lt, update_x))     # Legacy implementation note.
        # ----------------------------------------------------------------------------

        # Legacy implementation note.
        # Legacy implementation note.
        start_end = T.ivector()     # int32
        self.seq_train = theano.function(
            inputs=[start_end],
            outputs=-upq,
            updates=seq_updates,
            givens={
                pidxs: self.tra_buys_masks[start_end],       # Legacy implementation note.
                qidxs: self.tra_buys_neg_masks[start_end],   # Legacy implementation note.
                tra_mask: self.tra_masks[start_end]})

    def train(self, idxs):
        return self.seq_train(idxs)

    def __theano_predict__(self, n_in, n_hidden):
        """Run the training sequence again during testing to obtain hidden states."""
        ui, wh = self.ui, self.wh

        tra_mask = T.imatrix()
        actual_batch_size = tra_mask.shape[0]
        seq_length = T.max(T.sum(tra_mask, axis=1))     # Compute mini-batch start/end index ranges.

        h0 = T.alloc(self.h0, actual_batch_size, n_hidden)      # shape=(n, 20)
        c0 = T.alloc(self.c0, actual_batch_size, n_hidden)      # shape=(n, 20)
        bi = T.alloc(self.bi, actual_batch_size, 4, n_hidden)   # Vector dimension.
        bi = bi.dimshuffle(1, 2, 0)                             # shape=(3, 20, n)

        # Legacy implementation note.
        pidxs = T.imatrix()
        ps = self.trained_items[pidxs]      # shape((actual_batch_size, seq_length, n_hidden))
        ps = ps.dimshuffle(1, 0, 2)         # shape=(seq_length, batch_size, n_hidden)=(157, n, 20)

        def recurrence(p_t, c_t_pre1, h_t_pre1):
            # Legacy implementation note.
            gates = T.dot(ui, p_t.T) + T.dot(wh, h_t_pre1.T) + bi  # shape=(4, 20, n)
            i, f, g, o = sigmoid(gates[0]).T, sigmoid(gates[1]).T, tanh(gates[2]).T, sigmoid(gates[3]).T
            c_t = f * c_t_pre1 + i * g
            h_t = o * tanh(c_t)   # shape=(n, 20)
            return [c_t, h_t]
        [c, h], _ = theano.scan(         # h.shape=(157, n, 20)
            fn=recurrence,
            sequences=ps,
            outputs_info=[c0, h0],
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
                pidxs: self.tra_buys_masks[start_end],       # Legacy implementation note.
                tra_mask: self.tra_masks[start_end]})


# ======================================================================================================================
class Rnn(GruBasic):
    def __init__(self, train, test, alpha_lambda, n_user, n_item, n_in, n_hidden):
        super(Rnn, self).__init__(train, test, alpha_lambda, n_user, n_item, n_in, n_hidden)
        # Initialization.
        rang = 0.5
        ui = uniform(-rang, rang, (n_hidden, n_hidden))
        wh = uniform(-rang, rang, (n_hidden, n_hidden))
        bi = np.zeros((n_hidden, ), dtype=theano.config.floatX)
        # Legacy implementation note.
        self.ui = theano.shared(borrow=True, value=ui.astype(theano.config.floatX))
        self.wh = theano.shared(borrow=True, value=wh.astype(theano.config.floatX))
        self.bi = theano.shared(borrow=True, value=bi)
        self.params = [self.ui, self.wh, self.bi]       # Legacy implementation note.
        self.l2_sqr = (
            T.sum(self.lt ** 2) +
            T.sum([T.sum(param ** 2) for param in self.params]))
        self.l2 = (
            0.5 * self.alpha_lambda[1] * self.l2_sqr)
        self.__theano_train__(n_hidden)
        self.__theano_predict__(n_in, n_hidden)

    def __theano_train__(self, n_hidden):
        """Run one pass over the training sequence."""
        # self.alpha_lambda = ['alpha', 'lambda']
        ui, wh = self.ui, self.wh

        tra_mask = T.imatrix()                          # shape=(n, 157)
        actual_batch_size = tra_mask.shape[0]
        seq_length = T.max(T.sum(tra_mask, axis=1))     # Compute mini-batch start/end index ranges.
        mask = tra_mask.T                               # shape=(157, n)

        h0 = T.alloc(self.h0, actual_batch_size, n_hidden)      # shape=(n, 20)
        bi = T.alloc(self.bi, actual_batch_size, n_hidden)   # Legacy implementation note.
        bi = bi.T   # shape=(20, n)

        pidxs, qidxs = T.imatrix(), T.imatrix()         # TensorType(int32, matrix)
        xps, xqs = self.lt[pidxs], self.lt[qidxs]       # shape((actual_batch_size, seq_length, n_in))
        xps, xqs = xps.dimshuffle(1, 0, 2), xqs.dimshuffle(1, 0, 2)     # shape=(seq_length, batch_size, n_in)

        pqs = T.concatenate((pidxs, qidxs))         # Legacy implementation note.
        uiq_pqs = Unique(False, False, False)(pqs)  # Legacy implementation note.
        uiq_x = self.lt[uiq_pqs]                    # Legacy implementation note.

        """Compute the current hidden state and loss at time step t."""
        def recurrence(xp_t, xq_t, mask_t, h_t_pre1):
            # Legacy implementation note.
            h_t = sigmoid(T.dot(ui, xp_t.T) +
                          T.dot(wh, h_t_pre1.T) + bi)     # Legacy implementation note.
            h_t = h_t.T
            # Legacy implementation note.
            upq_t = T.sum(h_t_pre1 * (xp_t - xq_t), axis=1)     # shape=(n, )
            loss_t = T.log(sigmoid(upq_t))                      # shape=(n, )
            loss_t *= mask_t                            # Legacy implementation note.
            return [h_t, loss_t]                        # shape=(n, 20), (n, )
        [h, loss], _ = theano.scan(
            fn=recurrence,
            sequences=[xps, xqs, mask],
            outputs_info=[h0, None],
            n_steps=seq_length)     # Legacy implementation note.

        # ----------------------------------------------------------------------------
        # cost, gradients, learning rate, l2 regularization
        lr, l2 = self.alpha_lambda[0], self.alpha_lambda[1]
        seq_l2_sq = (
            T.sum([T.sum(par ** 2) for par in [xps, xqs, ui, wh]]) +
            T.sum([T.sum(par ** 2) for par in [bi]]) / actual_batch_size)
        upq = T.sum(loss)
        seq_costs = (
            - upq / actual_batch_size +
            0.5 * l2 * seq_l2_sq)
        seq_grads = T.grad(seq_costs, self.params)
        seq_updates = [(par, par - lr * gra) for par, gra in zip(self.params, seq_grads)]
        update_x = T.set_subtensor(uiq_x, uiq_x - lr * T.grad(seq_costs, self.lt)[uiq_pqs])
        seq_updates.append((self.lt, update_x))     # Legacy implementation note.
        # ----------------------------------------------------------------------------

        # Legacy implementation note.
        # Legacy implementation note.
        start_end = T.ivector()     # int32
        self.seq_train = theano.function(
            inputs=[start_end],
            outputs=-upq,
            updates=seq_updates,
            givens={
                pidxs: self.tra_buys_masks[start_end],       # Legacy implementation note.
                qidxs: self.tra_buys_neg_masks[start_end],   # Legacy implementation note.
                tra_mask: self.tra_masks[start_end]})

    def train(self, idxs):
        return self.seq_train(idxs)

    def __theano_predict__(self, n_in, n_hidden):
        """Run the training sequence again during testing to obtain hidden states."""
        ui, wh = self.ui, self.wh

        tra_mask = T.imatrix()
        actual_batch_size = tra_mask.shape[0]
        seq_length = T.max(T.sum(tra_mask, axis=1))     # Compute mini-batch start/end index ranges.

        h0 = T.alloc(self.h0, actual_batch_size, n_hidden)      # shape=(n, 20)
        bi = T.alloc(self.bi, actual_batch_size, n_hidden)   # Vector dimension.
        bi = bi.T   # shape=(20, n)

        # Legacy implementation note.
        pidxs = T.imatrix()
        ps = self.trained_items[pidxs]      # shape((actual_batch_size, seq_length, n_hidden))
        ps = ps.dimshuffle(1, 0, 2)         # shape=(seq_length, batch_size, n_hidden)=(157, n, 20)

        def recurrence(p_t, h_t_pre1):
            h_t = sigmoid(T.dot(ui, p_t.T) +
                          T.dot(wh, h_t_pre1.T) + bi)   # shape=(20, n)
            h_t = h_t.T                                     # Legacy implementation note.
            return h_t
        h, _ = theano.scan(         # h.shape=(157, n, 20)
            fn=recurrence,
            sequences=ps,
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
                pidxs: self.tra_buys_masks[start_end],       # Legacy implementation note.
                tra_mask: self.tra_masks[start_end]})


@exe_time  # Legacy implementation note.
def main():
    print('... construct the class: GRU')


if '__main__' == __name__:
    main()

