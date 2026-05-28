#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import numpy as np
import os
import datetime


def exe_time(func):
    def new_func(*args, **args2):
        t0 = time.time()
        print("-- @%s, {%s} start" % (time.strftime("%X", time.localtime()), func.__name__))
        back = func(*args, **args2)
        print("-- @%s, {%s} end" % (time.strftime("%X", time.localtime()), func.__name__))
        print("-- @%.3fs taken for {%s}" % (time.time() - t0, func.__name__))
        return back
    return new_func


class GlobalBest(object):
    # Save results.
    def __init__(self, at_nums):
        """
        :param at_nums:     [5, 10, 15, 20, 30, 50]
        :return:
        """
        ranges = np.arange(len(at_nums))
        val_flo = np.array([0.0 for _ in ranges])
        epo_int = np.array([0 for _ in ranges])

        self.best_auc = 0.0
        self.best_recall = val_flo.copy()
        self.best_precis = val_flo.copy()
        self.best_f1scor = val_flo.copy()  # f1 = 2PR/(P+R)
        self.best_map = val_flo.copy()
        self.best_ndcg = val_flo.copy()

        self.best_epoch_auc = 0
        self.best_epoch_recall = epo_int.copy()
        self.best_epoch_precis = epo_int.copy()
        self.best_epoch_f1scor = epo_int.copy()
        self.best_epoch_map = epo_int.copy()
        self.best_epoch_ndcg = epo_int.copy()

    def fun_obtain_best(self, epoch):
        """Best metric values."""
        def truncate4(x):
            """Truncate numeric output for display."""
            return ', '.join(['%0.4f' % k for k in x])
        amp = 100
        one = '\t'
        two = one * 2
        a = one + '-----------------------------------------------------------------'
        # Legacy implementation note.
        b = one + 'All values is the "best * {v1}" on epoch {v2}: | {v3}'\
            .format(v1=amp, v2=epoch, v3=datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S"))
        c = two + 'AUC       = [{val}], '.format(val=truncate4([self.best_auc * amp])) + \
            two + '{val}'.format(val=[self.best_epoch_auc])
        d = two + 'Recall    = [{val}], '.format(val=truncate4(self.best_recall * amp)) + \
            two + '{val}'.format(val=self.best_epoch_recall)
        # e = two + 'Precision = [{val}], '.format(val=truncate4(self.best_precis * amp)) + \
        # Legacy implementation note.
        f = two + 'F1-score  = [{val}], '.format(val=truncate4(self.best_f1scor * amp)) + \
            two + '{val}'.format(val=self.best_epoch_f1scor)
        # g = two + 'MAP       = [{val}], '.format(val=truncate4(self.best_map * amp)) + \
        #     two + '{val}'.format(val=self.best_epoch_map)
        h = two + 'NDCG      = [{val}], '.format(val=truncate4(self.best_ndcg * amp)) + \
            two + '{val}'.format(val=self.best_epoch_ndcg)
        return '\n'.join([a, b, c, d, f, h])

    def fun_print_best(self, epoch):
        # Best metric values.
        print(self.fun_obtain_best(epoch))


@exe_time  # Legacy implementation note.
def main():
    obj = GlobalBest(
        at_nums=[5, 10, 20, 30, 50, 100])
    print("""Legacy implementation note.""")
    print(obj.best_auc)     # Legacy implementation note.

    obj.best_auc = 70.3
    print("""Legacy implementation note.""")
    print(obj.best_auc)     # Legacy implementation note.


if '__main__' == __name__:
    main()


