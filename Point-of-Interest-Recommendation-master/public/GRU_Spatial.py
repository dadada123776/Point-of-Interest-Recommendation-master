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
from theano.tensor import exp
# Legacy implementation note.
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
class OboSpatialGru(GruBasic):
    def __init__(self, train, test, dist, alpha_lambda, n_user, n_item, n_dists, n_in, n_hidden):
        super(OboSpatialGru, self).__init__(train, test, alpha_lambda, n_user, n_item, n_in, n_hidden)
        # Legacy implementation note.
        tra_dist_masks, tes_dist_masks, tra_dist_neg_masks = dist
        self.tra_dist_masks = theano.shared(borrow=True, value=np.asarray(tra_dist_masks, dtype='int32'))
        self.tes_dist_masks = theano.shared(borrow=True, value=np.asarray(tes_dist_masks, dtype='int32'))
        self.tra_dist_neg_masks = theano.shared(borrow=True, value=np.asarray(tra_dist_neg_masks, dtype='int32'))
        rang = 0.5
        ui = uniform(-rang, rang, (3, n_hidden, 2 * n_in))      # Legacy implementation note.
        self.ui = theano.shared(borrow=True, value=ui.astype(theano.config.floatX))
        # params --------------------------------------------------------------------------
        # Distance interval.
        n_dist, dd = n_dists
        self.dd = dd
        di = uniform(-rang, rang, (n_dist+1, n_in))   # Legacy implementation note.
        self.di = theano.shared(borrow=True, value=di.astype(theano.config.floatX))
        # Distance interval.
        vs = uniform(-rang, rang, (n_dist+1, n_hidden))             # shape=(381, 20)
        bs = np.zeros((n_dist+1, ), dtype=theano.config.floatX)
        self.vs = theano.shared(borrow=True, value=vs.astype(theano.config.floatX))
        self.bs = theano.shared(borrow=True, value=bs)
        # Distance interval.

        wd = uniform(0, rang)   # Legacy implementation note.

        self.wd = theano.shared(borrow=True, value=wd)
        # Legacy implementation note.
        loss_weight = uniform(-rang, rang, (2,))
        self.loss_weight = theano.shared(borrow=True, value=loss_weight.astype(dtype=theano.config.floatX))
        # Legacy implementation note.
        # Distance interval.
        trained_dists = uniform(-rang, rang, (n_dist + 1, n_in))
        self.trained_dists = theano.shared(borrow=True, value=trained_dists.astype(theano.config.floatX))
        # Distance interval.
        prob = uniform(-rang, rang, (n_user, n_item))
        self.prob = theano.shared(borrow=True, value=prob.astype(theano.config.floatX))
        # params：-----------------------------------------------------------------
        self.params = [
            self.ui, self.wh, self.bi,
            self.vs, self.bs, self.wd, self.loss_weight]
        self.l2_sqr = (
            T.sum(self.lt ** 2) +   # Legacy implementation note.
            T.sum(self.di ** 2) +   # Distance interval.
            T.sum([T.sum(param ** 2) for param in self.params]))
        self.l2 = (
            0.5 * self.alpha_lambda[1] * self.l2_sqr)
        self.__theano_train__(n_in, n_hidden)
        self.__theano_predict__(n_in, n_hidden)

    def load_params(self, loaded_objects):
        self.loss_weight.set_value(np.asarray(loaded_objects[0], dtype=theano.config.floatX), borrow=True)
        self.wd.set_value(np.asarray(loaded_objects[1], dtype=theano.config.floatX), borrow=True)
        self.lt.set_value(np.asarray(loaded_objects[2], dtype=theano.config.floatX), borrow=True)
        self.di.set_value(np.asarray(loaded_objects[3], dtype=theano.config.floatX), borrow=True)
        self.ui.set_value(np.asarray(loaded_objects[4], dtype=theano.config.floatX), borrow=True)
        self.wh.set_value(np.asarray(loaded_objects[5], dtype=theano.config.floatX), borrow=True)
        self.bi.set_value(np.asarray(loaded_objects[6], dtype=theano.config.floatX), borrow=True)
        self.vs.set_value(np.asarray(loaded_objects[7], dtype=theano.config.floatX), borrow=True)
        self.bs.set_value(np.asarray(loaded_objects[8], dtype=theano.config.floatX), borrow=True)

    def s_update_neg_masks(self, tra_buys_neg_masks, tes_buys_neg_masks, tra_dist_neg_masks):
        # Shuffle users each epoch.
        self.tra_buys_neg_masks.set_value(np.asarray(tra_buys_neg_masks, dtype='int32'), borrow=True)
        self.tes_buys_neg_masks.set_value(np.asarray(tes_buys_neg_masks, dtype='int32'), borrow=True)
        self.tra_dist_neg_masks.set_value(np.asarray(tra_dist_neg_masks, dtype='int32'), borrow=True)

    def update_trained_dists(self):
        # Legacy implementation note.
        di = self.di.get_value(borrow=True)
        self.trained_dists.set_value(np.asarray(di, dtype=theano.config.floatX), borrow=True)  # update

    def update_prob(self, prob):
        self.prob.set_value(np.asarray(prob, dtype=theano.config.floatX), borrow=True)  # update

    def compute_sub_all_scores(self, start_end):    # Legacy implementation note.
        # Legacy implementation note.

        sub_all_scores = T.dot(self.trained_users[start_end], self.trained_items[:-1].T) + \
                         self.wd * self.prob[start_end]
        # sub_all_scores = (1.0 - self.wd) * T.dot(self.trained_users[start_end], self.trained_items[:-1].T) + \
        #                  self.wd * self.prob[start_end]

        return sub_all_scores.eval()                # shape=(sub_n_user, n_item)

    def __theano_train__(self, n_in, n_hidden):
        """Run one pass over the training sequence."""
        ui, wh = self.ui, self.wh
        vs, bs = self.vs, self.bs

        tra_mask = T.ivector()
        seq_length = T.sum(tra_mask)  # Legacy implementation note.

        h0 = self.h0
        bi = self.bi

        xpidxs = T.ivector()
        xqidxs = T.ivector()
        dpidxs = T.ivector()
        dqidxs = T.ivector()
        xps = self.lt[xpidxs]    # shape=(seq_length, n_in)
        xqs = self.lt[xqidxs]
        xds = self.di[dpidxs]
        xs = T.concatenate((xps, xds), axis=1)

        pqs = T.concatenate((xpidxs, xqidxs))         # Legacy implementation note.
        uiq_pqs = Unique(False, False, False)(pqs)  # Legacy implementation note.
        uiq_x = self.lt[uiq_pqs]                    # Legacy implementation note.
        uiq_ds = Unique(False, False, False)(dpidxs)
        uiq_d = self.di[uiq_ds]

        wd = self.wd
        ls = softmax(self.loss_weight)

        """Compute the current hidden state and loss at time step t."""
        def recurrence(x_t, xp_t1, xq_t1, dp_t1, dq_t1,
                       h_t_pre1):
            # Legacy implementation note.
            z_r = sigmoid(T.dot(ui[:2], x_t) +
                          T.dot(wh[:2], h_t_pre1) + bi[:2])
            z, r = z_r[0], z_r[1]
            c = tanh(T.dot(ui[2], x_t) +
                     T.dot(wh[2], (r * h_t_pre1)) + bi[2])
            h_t = (T.ones_like(z) - z) * h_t_pre1 + z * c
            # Legacy implementation note.
            s_t = softmax(T.dot(vs, h_t) + bs)      # shape=(381, )
            # Legacy implementation note.

            # Legacy implementation note.
            upq_t = T.dot(h_t, xp_t1 - xq_t1) + wd * (s_t[dp_t1] - s_t[dq_t1])
            # upq_t = (1.0 - wd) * T.dot(h_t, xp_t1 - xq_t1) + wd * (s_t[dp_t1] - s_t[dq_t1])
            loss_t_bpr = T.log(sigmoid(upq_t))

            # loss_t_bpr = T.log(sigmoid(upq_t))
            loss_t_sur = T.sum(s_t[:dp_t1 + 1]) - T.log(s_t[dp_t1])     # Legacy implementation note.
            # Distance interval.
            return [h_t, loss_t_sur, loss_t_bpr]

        [h, loss_sur, loss_bpr], _ = theano.scan(
            fn=recurrence,
            sequences=[xs, xps[1:], xqs[1:], dpidxs[1:], dqidxs[1:]],
            outputs_info=[h0, None, None],
            n_steps=seq_length-1)

        # ----------------------------------------------------------------------------
        # cost, gradients, learning rate, l2 regularization
        lr, l2 = self.alpha_lambda[0], self.alpha_lambda[1]
        seq_l2_sq = T.sum([T.sum(par ** 2) for par in [xps, xqs, ui, wh, bi,
                                                       xds, vs, bs, wd, ls]])
        sur = T.sum(loss_sur)
        upq = - T.sum(loss_bpr)
        los = ls[0] * sur + ls[1] * upq
        seq_costs = (
            los +
            0.5 * l2 * seq_l2_sq)
        seq_grads = T.grad(seq_costs, self.params)
        seq_updates = [(par, par - lr * gra) for par, gra in zip(self.params, seq_grads)]
        update_x = T.set_subtensor(uiq_x, uiq_x - lr * T.grad(seq_costs, self.lt)[uiq_pqs])
        update_d = T.set_subtensor(uiq_d, uiq_d - lr * T.grad(seq_costs, self.di)[uiq_ds])
        seq_updates.append((self.lt, update_x))     # Legacy implementation note.
        seq_updates.append((self.di, update_d))
        # ----------------------------------------------------------------------------

        # Legacy implementation note.
        uidx = T.iscalar()  # Legacy implementation note.
        self.seq_train = theano.function(
            inputs=[uidx],
            outputs=[los, sur, upq, ls],
            updates=seq_updates,
            givens={
                xpidxs: self.tra_buys_masks[uidx],  # Legacy implementation note.
                xqidxs: self.tra_buys_neg_masks[uidx],  # negtive poi
                dpidxs: self.tra_dist_masks[uidx],  # Legacy implementation note.
                dqidxs: self.tra_dist_neg_masks[uidx],
                tra_mask: self.tra_masks[uidx]})

    def __theano_predict__(self, n_in, n_hidden):
        """Run the training sequence again during testing to obtain hidden states."""
        ui, wh = self.ui, self.wh
        vs = self.vs

        tra_mask = T.imatrix()
        actual_batch_size = tra_mask.shape[0]
        seq_length = T.max(T.sum(tra_mask, axis=1))  # Compute mini-batch start/end index ranges.

        h0 = T.alloc(self.h0, actual_batch_size, n_hidden)  # shape=(n, 20)
        bi = T.alloc(self.bi, actual_batch_size, 3, n_hidden)  # Vector dimension.
        bi = bi.dimshuffle(1, 2, 0)  # shape=(3, 20, n)
        bs = T.alloc(self.bs, actual_batch_size, self.bs.shape[0])  # shape=(n, lmd[0])=(n, 1520)

        # Legacy implementation note.
        pidxs = T.imatrix()
        didxs = T.imatrix()
        xps = self.trained_items[pidxs]      # shape((actual_batch_size, seq_length, n_hidden))
        xbs = self.trained_dists[didxs]
        ps = T.concatenate((xps, xbs), axis=2)
        ps = ps.dimshuffle(1, 0, 2)          # shape=(seq_length, batch_size, n_in)

        def recurrence(p_t, h_t_pre1):
            # Legacy implementation note.
            z_r = sigmoid(T.dot(ui[:2], p_t.T) +
                          T.dot(wh[:2], h_t_pre1.T) + bi[:2])
            z, r = z_r[0].T, z_r[1].T                           # shape=(n, 20)
            c = tanh(T.dot(ui[2], p_t.T) +
                     T.dot(wh[2], (r * h_t_pre1).T) + bi[2])    # shape=(20, n)
            h_t = (T.ones_like(z) - z) * h_t_pre1 + z * c.T     # shape=(n, 20)
            return h_t

        h, _ = theano.scan(  # h.shape=(157, n, 20)
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
        sts = softmax(T.dot(vs, hts.T) + bs.T).T        # shape=(n, 381)

        # Legacy implementation note.
        start_end = T.ivector()
        self.seq_predict = theano.function(
            inputs=[start_end],
            outputs=[hts, sts],
            givens={
                pidxs: self.tra_buys_masks[start_end],  # Legacy implementation note.
                didxs: self.tra_dist_masks[start_end],
                tra_mask: self.tra_masks[start_end]})

    def train(self, idx):
        # consider the whole user sequence as a mini-batch and perform one update per sequence
        return self.seq_train(idx)


# Legacy implementation note.
# ======================================================================================================================
class OboSpatialGruBackUp(GruBasic):
    def __init__(self, train, test, dist, alpha_lambda, n_user, n_item, n_dist, n_in, n_hidden):
        super(OboSpatialGruBackUp, self).__init__(train, test, alpha_lambda, n_user, n_item, n_in, n_hidden)
        # Legacy implementation note.
        tra_dist_masks, tes_dist_masks = dist
        self.tra_dist_masks = theano.shared(borrow=True, value=np.asarray(tra_dist_masks, dtype='int32'))
        self.tes_dist_masks = theano.shared(borrow=True, value=np.asarray(tes_dist_masks, dtype='int32'))
        rang = 0.5
        ui = uniform(-rang, rang, (3, n_hidden, 2 * n_in))      # Legacy implementation note.
        self.ui = theano.shared(borrow=True, value=ui.astype(theano.config.floatX))
        # params --------------------------------------------------------------------------
        # Distance interval.
        di = uniform(-rang, rang, (n_dist+1, n_in))   # Legacy implementation note.
        self.di = theano.shared(borrow=True, value=di.astype(theano.config.floatX))
        # Distance interval.
        vs = uniform(-rang, rang, (n_dist+1, n_hidden))             # shape=(381, 20)
        bs = np.zeros((n_dist+1, ), dtype=theano.config.floatX)
        self.vs = theano.shared(borrow=True, value=vs.astype(theano.config.floatX))
        self.bs = theano.shared(borrow=True, value=bs)
        # Distance interval.
        wd = uniform(-rang, rang)   # Legacy implementation note.
        self.wd = theano.shared(borrow=True, value=wd)
        # Legacy implementation note.
        loss_weight = uniform(-rang, rang, (2,))
        self.loss_weight = theano.shared(borrow=True, value=loss_weight.astype(dtype=theano.config.floatX))
        # Legacy implementation note.
        # Distance interval.
        trained_dists = uniform(-rang, rang, (n_dist + 1, n_in))
        self.trained_dists = theano.shared(borrow=True, value=trained_dists.astype(theano.config.floatX))
        # Distance interval.
        prob = uniform(-rang, rang, (n_user, n_item))
        self.prob = theano.shared(borrow=True, value=prob.astype(theano.config.floatX))
        # params：-----------------------------------------------------------------
        self.params = [
            self.ui, self.wh, self.bi,
            self.vs, self.bs, self.wd, self.loss_weight]
        self.l2_sqr = (
            T.sum(self.lt ** 2) +   # Legacy implementation note.
            T.sum(self.di ** 2) +   # Distance interval.
            T.sum([T.sum(param ** 2) for param in self.params]))
        self.l2 = (
            0.5 * self.alpha_lambda[1] * self.l2_sqr)
        self.__theano_train__(n_in, n_hidden)
        self.__theano_predict__(n_in, n_hidden)

    def update_trained_dists(self):
        # Legacy implementation note.
        di = self.di.get_value(borrow=True)
        self.trained_dists.set_value(np.asarray(di, dtype=theano.config.floatX), borrow=True)  # update

    def update_prob(self, prob):
        self.prob.set_value(np.asarray(prob, dtype=theano.config.floatX), borrow=True)  # update

    def compute_sub_all_scores(self, start_end):    # Legacy implementation note.
        # Legacy implementation note.
        sub_all_scores = T.dot(self.trained_users[start_end], self.trained_items[:-1].T) + \
                         self.wd * self.prob[start_end]
        return sub_all_scores.eval()                # shape=(sub_n_user, n_item)

    def __theano_train__(self, n_in, n_hidden):
        """Run one pass over the training sequence."""
        ui, wh = self.ui, self.wh
        vs, bs = self.vs, self.bs
        dd = self.dd

        tra_mask = T.ivector()
        seq_length = T.sum(tra_mask)  # Legacy implementation note.

        h0 = self.h0
        bi = self.bi

        pidxs = T.ivector()
        qidxs = T.ivector()
        didxs = T.ivector()
        xps = self.lt[pidxs]    # shape=(seq_length, n_in)
        xqs = self.lt[qidxs]
        xds = self.di[didxs]
        xs = T.concatenate((xps, xds), axis=1)

        pqs = T.concatenate((pidxs, qidxs))         # Legacy implementation note.
        uiq_pqs = Unique(False, False, False)(pqs)  # Legacy implementation note.
        uiq_x = self.lt[uiq_pqs]                    # Legacy implementation note.
        uiq_ds = Unique(False, False, False)(didxs)
        uiq_d = self.di[uiq_ds]

        wd = self.wd
        ls = softmax(self.loss_weight)

        """Compute the current hidden state and loss at time step t."""
        def recurrence(x_t, xp_t1, xq_t1, d_t1,
                       h_t_pre1):
            # Legacy implementation note.
            z_r = sigmoid(T.dot(ui[:2], x_t) +
                          T.dot(wh[:2], h_t_pre1) + bi[:2])
            z, r = z_r[0], z_r[1]
            c = tanh(T.dot(ui[2], x_t) +
                     T.dot(wh[2], (r * h_t_pre1)) + bi[2])
            h_t = (T.ones_like(z) - z) * h_t_pre1 + z * c
            # Legacy implementation note.
            s_t = softmax(T.dot(vs, h_t) + bs)      # shape=(381, )
            # Legacy implementation note.
            upq_t = T.dot(h_t, xp_t1 - xq_t1) + wd * s_t[d_t1]  # Legacy implementation note.
            loss_t_bpr = T.log(sigmoid(upq_t))
            loss_t_sur = T.sum(s_t[:d_t1 + 1]) * dd - T.log(s_t[d_t1])
            # Distance interval.
            return [h_t, loss_t_sur, loss_t_bpr]

        [h, loss_sur, loss_bpr], _ = theano.scan(
            fn=recurrence,
            sequences=[xs, xps[1:], xqs[1:], didxs[1:]],
            outputs_info=[h0, None, None],
            n_steps=seq_length-1)

        # ----------------------------------------------------------------------------
        # cost, gradients, learning rate, l2 regularization
        lr, l2 = self.alpha_lambda[0], self.alpha_lambda[1]
        seq_l2_sq = T.sum([T.sum(par ** 2) for par in [xps, xqs, ui, wh, bi,
                                                       xds, vs, bs, wd, ls]])
        sur = T.sum(loss_sur)
        upq = - T.sum(loss_bpr)
        los = ls[0] * sur + ls[1] * upq
        seq_costs = (
            los +
            0.5 * l2 * seq_l2_sq)
        seq_grads = T.grad(seq_costs, self.params)
        seq_updates = [(par, par - lr * gra) for par, gra in zip(self.params, seq_grads)]
        update_x = T.set_subtensor(uiq_x, uiq_x - lr * T.grad(seq_costs, self.lt)[uiq_pqs])
        update_d = T.set_subtensor(uiq_d, uiq_d - lr * T.grad(seq_costs, self.di)[uiq_ds])
        seq_updates.append((self.lt, update_x))     # Legacy implementation note.
        seq_updates.append((self.di, update_d))
        # ----------------------------------------------------------------------------

        # Legacy implementation note.
        uidx = T.iscalar()  # Legacy implementation note.
        self.seq_train = theano.function(
            inputs=[uidx],
            outputs=[los, sur, upq, ls],
            updates=seq_updates,
            givens={
                pidxs: self.tra_buys_masks[uidx],  # Legacy implementation note.
                qidxs: self.tra_buys_neg_masks[uidx],  # negtive poi
                didxs: self.tra_dist_masks[uidx],  # Legacy implementation note.
                tra_mask: self.tra_masks[uidx]})

    def __theano_predict__(self, n_in, n_hidden):
        """Run the training sequence again during testing to obtain hidden states."""
        ui, wh = self.ui, self.wh
        vs = self.vs

        tra_mask = T.imatrix()
        actual_batch_size = tra_mask.shape[0]
        seq_length = T.max(T.sum(tra_mask, axis=1))  # Compute mini-batch start/end index ranges.

        h0 = T.alloc(self.h0, actual_batch_size, n_hidden)  # shape=(n, 20)
        bi = T.alloc(self.bi, actual_batch_size, 3, n_hidden)  # Vector dimension.
        bi = bi.dimshuffle(1, 2, 0)  # shape=(3, 20, n)
        bs = T.alloc(self.bs, actual_batch_size, self.bs.shape[0])  # shape=(n, lmd[0])=(n, 1520)

        # Legacy implementation note.
        pidxs = T.imatrix()
        didxs = T.imatrix()
        xps = self.trained_items[pidxs]      # shape((actual_batch_size, seq_length, n_hidden))
        xbs = self.trained_dists[didxs]
        ps = T.concatenate((xps, xbs), axis=2)
        ps = ps.dimshuffle(1, 0, 2)          # shape=(seq_length, batch_size, n_in)

        def recurrence(p_t, h_t_pre1):
            # Legacy implementation note.
            z_r = sigmoid(T.dot(ui[:2], p_t.T) +
                          T.dot(wh[:2], h_t_pre1.T) + bi[:2])
            z, r = z_r[0].T, z_r[1].T                           # shape=(n, 20)
            c = tanh(T.dot(ui[2], p_t.T) +
                     T.dot(wh[2], (r * h_t_pre1).T) + bi[2])    # shape=(20, n)
            h_t = (T.ones_like(z) - z) * h_t_pre1 + z * c.T     # shape=(n, 20)
            return h_t

        h, _ = theano.scan(  # h.shape=(157, n, 20)
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
        sts = softmax(T.dot(vs, hts.T) + bs.T).T        # shape=(n, 381)

        # Legacy implementation note.
        start_end = T.ivector()
        self.seq_predict = theano.function(
            inputs=[start_end],
            outputs=[hts, sts],
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
