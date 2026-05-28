#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import time
import numpy as np
from numpy.random import uniform
import theano
import theano.tensor as T
from theano.tensor.nnet import sigmoid
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
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


# ======================================================================================================================
class MfBasic(object):
    def __init__(self, train, test, alpha_lambda, n_user, n_item, n_in, n_hidden):
        """Build model parameters."""
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
        ux = uniform(-rang, rang, (n_user, n_in))
        lt = uniform(-rang, rang, (n_item + 1, n_in))   # shape=(n_item, 20)
        self.ux = theano.shared(borrow=True, value=ux.astype(theano.config.floatX))
        self.lt = theano.shared(borrow=True, value=lt.astype(theano.config.floatX))
        # Legacy implementation note.
        trained_items = uniform(-rang, rang, (n_item + 1, n_hidden))
        trained_users = uniform(-rang, rang, (n_user, n_hidden))
        self.trained_items = theano.shared(borrow=True, value=trained_items.astype(theano.config.floatX))
        self.trained_users = theano.shared(borrow=True, value=trained_users.astype(theano.config.floatX))

    def update_neg_masks(self, tra_buys_neg_masks, tes_buys_neg_masks):
        # Shuffle users each epoch.
        self.tra_buys_neg_masks.set_value(np.asarray(tra_buys_neg_masks, dtype='int32'), borrow=True)
        self.tes_buys_neg_masks.set_value(np.asarray(tes_buys_neg_masks, dtype='int32'), borrow=True)

    def update_trained_items(self):
        # Legacy implementation note.
        lt = self.lt.get_value(borrow=True)    # Legacy implementation note.
        self.trained_items.set_value(np.asarray(lt, dtype=theano.config.floatX), borrow=True)     # update

    def update_trained_users(self):
        # Legacy implementation note.
        ux = self.ux.get_value(borrow=True)
        self.trained_users.set_value(np.asarray(ux, dtype=theano.config.floatX), borrow=True)  # update

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


# Legacy implementation note.
# ======================================================================================================================
class MFonebyone(MfBasic):
    def __init__(self, n_user, n_item, n_in):
        super(MFonebyone, self).__init__(n_user, n_item, n_in)
        self.params = [self.ur, self.lt]
        self.L2_sqr = (
            T.sum(self.ur ** 2) +
            T.sum(self.lt ** 2))
        self.__theano_train__()

    def __theano_train__(self, ):
        """Run one pass over the training sequence."""
        uidx, pidx = T.iscalar(), T.iscalar()
        us = self.ur[uidx]     # shape=(n_in, )
        xp = self.lt[pidx]

        """Compute the current hidden state and loss at time step t."""
        pre = T.dot(us, xp)
        err = 5.0 - pre
        loss = err * err

        # ----------------------------------------------------------------------------
        # cost, gradients, learning rate, L2 regularization
        lr, L2_reg = T.scalar(), T.scalar()
        bpr_L2_sqr = (
            T.sum(us ** 2) +
            T.sum(xp ** 2))
        costs = (
            loss +
            0.5 * L2_reg * bpr_L2_sqr)
        # SGD
        update_us = T.set_subtensor(us, us - lr * T.grad(costs, us))    # Legacy implementation note.
        update_xp = T.set_subtensor(xp, xp - lr * T.grad(costs, xp))

        # ----------------------------------------------------------------------------
        # Legacy implementation note.
        self.train = theano.function(
            inputs=[uidx, pidx, lr, L2_reg],
            outputs=loss,
            updates=[(self.ur, update_us),     # Legacy implementation note.
                     (self.lt, update_xp)])

    def train(self, u_idx, p_idx, lr, L2_reg):
        # Legacy implementation note.
        return self.train(u_idx, p_idx, lr, L2_reg)


# ======================================================================================================================
class OboBpr(MfBasic):
    def __init__(self, train, test, alpha_lambda, n_user, n_item, n_in, n_hidden):
        super(OboBpr, self).__init__(train, test, alpha_lambda, n_user, n_item, n_in, n_hidden)
        self.params = [self.ux, self.lt]    # Legacy implementation note.
        self.l2_sqr = (
            T.sum([T.sum(param ** 2) for param in self.params]))
        self.l2 = (
            0.5 * self.alpha_lambda[1] * self.l2_sqr)
        self.__theano_train__()

    def __theano_train__(self, ):
        """Run one pass over the training sequence."""
        # self.alpha_lambda = ['alpha', 'lambda']
        uidx, pqidx = T.iscalar(), T.ivector()
        usr = self.ux[uidx]     # shape=(n_in, )
        xpq = self.lt[pqidx]

        """Compute the current hidden state and loss at time step t."""
        uij = T.dot(usr, xpq[0] - xpq[1])
        upq = T.log(sigmoid(uij))

        # ----------------------------------------------------------------------------
        # cost, gradients, learning rate, L2 regularization
        lr, l2 = self.alpha_lambda[0], self.alpha_lambda[1]
        bpr_l2_sqr = (
            T.sum([T.sum(par ** 2) for par in [usr, xpq]]))
        costs = (
            - upq +
            0.5 * l2 * bpr_l2_sqr)
        # Legacy implementation note.
        pars_subs = [(self.ux, usr), (self.lt, xpq)]
        seq_updates = [(par, T.set_subtensor(sub, sub - lr * T.grad(costs, sub)))
                       for par, sub in pars_subs]
        # ----------------------------------------------------------------------------

        # Legacy implementation note.
        self.bpr_train = theano.function(
            inputs=[uidx, pqidx],
            outputs=-upq,
            updates=seq_updates)

    def train(self, u_idx, pq_idx):
        # Legacy implementation note.
        return self.bpr_train(u_idx, pq_idx)


# ======================================================================================================================
class OboVBpr(MfBasic):
    def __init__(self, train, test, alpha_lambda, n_user, n_item, n_in, n_hidden,
                 n_img, fea_img):
        super(OboVBpr, self).__init__(train, test, alpha_lambda, n_user, n_item, n_in, n_hidden)
        self.fi = theano.shared(borrow=True, value=np.asarray(fea_img, dtype=theano.config.floatX))  # shape=(n, 1024)
        # Initialization.
        rang = 0.5
        mi = uniform(-rang, rang, (n_item + 1, n_in))   # Legacy implementation note.
        ue = uniform(-rang, rang, (n_user, n_in))
        ei = uniform(-rang, rang, (n_in, n_img))       # image, shape=(20, 1024)
        self.mi = theano.shared(borrow=True, value=mi.astype(theano.config.floatX))
        self.ue = theano.shared(borrow=True, value=ue.astype(theano.config.floatX))
        self.ei = theano.shared(borrow=True, value=ei.astype(theano.config.floatX))
        self.params = [self.ux, self.lt, self.ue, self.ei]
        self.l2_sqr = (
            T.sum([T.sum(param ** 2) for param in self.params[:3]]))
        self.l2_ev = (
            T.sum([T.sum(param ** 2) for param in self.params[3:]]))
        self.l2 = (
            0.5 * self.alpha_lambda[1] * self.l2_sqr +
            0.5 * self.alpha_lambda[2] * self.l2_ev)
        self.__theano_train__()

    def __theano_train__(self):
        """Run one pass over the training sequence."""
        # self.alpha_lambda = ['alpha', 'lambda', 'lambda_ev', 'fea_random_zero']
        ei = self.ei

        uidx, pqidx = T.iscalar(), T.ivector()
        usr = self.ux[uidx]     # shape=(n_in, )
        xpq = self.lt[pqidx]
        use = self.ue[uidx]
        ipq = self.fi[pqidx]

        """Compute the current hidden state and loss at time step t."""
        uij = (
            T.dot(usr, xpq[0] - xpq[1]) +
            T.dot(use, T.dot(ei, ipq[0] - ipq[1])))
        upq = T.log(sigmoid(uij))

        # ----------------------------------------------------------------------------
        # cost, gradients, learning rate, L2 regularization
        lr, l2 = self.alpha_lambda[0], self.alpha_lambda[1]
        l2_ev = self.alpha_lambda[2]
        bpr_l2_sqr = (
            T.sum([T.sum(par ** 2) for par in [usr, xpq, use]]))
        bpr_l2_ev = (
            T.sum([T.sum(par ** 2) for par in [ei]]))
        costs = (
            - upq +
            0.5 * l2 * bpr_l2_sqr +
            0.5 * l2_ev * bpr_l2_ev)
        pars_subs = [(self.ux, usr), (self.lt, xpq), (self.ue, use)]    # Legacy implementation note.
        seq_updates = [(par, T.set_subtensor(sub, sub - lr * T.grad(costs, sub)))
                       for par, sub in pars_subs]
        pars_alls = [self.ei]
        seq_updates.extend([(par, par - lr * T.grad(costs, par)) for par in pars_alls])
        # ----------------------------------------------------------------------------

        # Legacy implementation note.
        self.bpr_train = theano.function(
            inputs=[uidx, pqidx],
            outputs=-upq,
            updates=seq_updates)

    def train(self, u_idx, pq_idx):
        # Legacy implementation note.
        return self.bpr_train(u_idx, pq_idx)

    def update_trained_items(self):
        # Legacy implementation note.
        mi = T.dot(self.fi, self.ei.T)     # shape=(n, 20)
        mi = mi.eval()
        self.mi.set_value(np.asarray(mi, dtype=theano.config.floatX), borrow=True)
        # Legacy implementation note.
        items = T.concatenate((self.lt, self.mi), axis=1)   # shape=(n_item+1, 40)
        items = items.eval()
        self.trained_items.set_value(np.asarray(items, dtype=theano.config.floatX), borrow=True)

    def update_trained_users(self):
        # Legacy implementation note.
        users = T.concatenate((self.ux, self.ue), axis=1)   # shape=(n, 40)
        users = users.eval()
        self.trained_users.set_value(np.asarray(users, dtype=theano.config.floatX), borrow=True)


# Legacy implementation note.
# Legacy implementation note.
# ======================================================================================================================
class Bpr(MfBasic):
    def __init__(self, train, test, alpha_lambda, n_user, n_item, n_in, n_hidden):
        super(Bpr, self).__init__(train, test, alpha_lambda, n_user, n_item, n_in, n_hidden)
        self.params = [self.ux, self.lt]    # Legacy implementation note.
        self.l2_sqr = (
            T.sum([T.sum(param ** 2) for param in self.params]))
        self.l2 = (
            0.5 * self.alpha_lambda[1] * self.l2_sqr)
        self.__theano_train__()

    def __theano_train__(self):
        """Run one pass over the training sequence."""
        # self.alpha_lambda = ['alpha', 'lambda']
        pidxs_t, qidxs_t = T.ivector(), T.ivector()
        mask_t, uidxs = T.ivector(), T.ivector()
        users = self.ux[uidxs]  # shape=(n, 20)
        xps = self.lt[pidxs_t]  # shape=(n, 20)
        xqs = self.lt[qidxs_t]

        pqs = T.concatenate((pidxs_t, qidxs_t))         # Legacy implementation note.
        uiq_pqs = Unique(False, False, False)(pqs)  # Legacy implementation note.
        uiq_x = self.lt[uiq_pqs]                    # Legacy implementation note.

        """Compute the current hidden state and loss at time step t."""
        upq_t = T.sum(users * (xps - xqs), axis=1)
        loss_t = T.log(sigmoid(upq_t))      # shape=(n, )
        loss_t *= mask_t                    # Legacy implementation note.

        # ----------------------------------------------------------------------------
        # cost, gradients, learning rate, L2 regularization
        lr, l2 = self.alpha_lambda[0], self.alpha_lambda[1]
        bpr_l2_sqr = (
            T.sum([T.sum(par ** 2) for par in [users, xps, xqs]]))
        upq = T.sum(loss_t)
        costs = (
            - upq +
            0.5 * l2 * bpr_l2_sqr)
        pars_subs = [(self.ux, users, uidxs), (self.lt, uiq_x, uiq_pqs)]
        bpr_updates = [(par, T.set_subtensor(sub, sub - lr * T.grad(costs, par)[idxs]))
                       for par, sub, idxs in pars_subs]
        # ----------------------------------------------------------------------------

        # Legacy implementation note.
        self.bpr_train = theano.function(
            inputs=[pidxs_t, qidxs_t, mask_t, uidxs],
            outputs=-upq,
            updates=bpr_updates)

    def train(self, pidxs_t, qidxs_t, mask_t, uidxs):
        return self.bpr_train(pidxs_t, qidxs_t, mask_t, uidxs)


@exe_time  # Legacy implementation note.
def main():
    pass


if '__main__' == __name__:
    main()
