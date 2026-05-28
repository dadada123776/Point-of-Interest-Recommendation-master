#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import time
import pandas as pd
import numpy as np
import os
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


def fun_hit_zero_one(user_test_recom):
    """Generate a 0/1 hit sequence for the recommended list."""
    test_lst, recom_lst, test_mask, _ = user_test_recom
    test_lst = test_lst[:np.sum(test_mask)]     # Legacy implementation note.
    seq = []
    for e in recom_lst:
        if e in test_lst:       # Hit.
            seq.append(1)
        else:                   # Hit.
            seq.append(0)
    return np.array(seq)


def fun_evaluate_map(user_test_recom_zero_one):
    """Compute MAP for one user, then average over users."""
    test_lst, zero_one, test_mask, _ = user_test_recom_zero_one
    test_lst = test_lst[:np.sum(test_mask)]

    zero_one = np.array(zero_one)
    if 0 == sum(zero_one):    # Hit.
        return 0.0
    zero_one_cum = zero_one.cumsum()                # Hit.
    zero_one_cum *= zero_one                        # Hit.
    idxs = list(np.nonzero(zero_one_cum))[0]        # Legacy implementation note.
    s = 0.0
    for idx in idxs:
        s += 1.0 * zero_one_cum[idx] / (idx + 1)
    return s / len(test_lst)


def fun_evaluate_ndcg(user_test_recom_zero_one):
    """Compute NDCG for one user, then average over users."""
    test_lst, zero_one, test_mask, _ = user_test_recom_zero_one
    test_lst = test_lst[:np.sum(test_mask)]

    zero_one = np.array(zero_one)
    if 0 == sum(zero_one):    # Hit.
        return 0.0
    s = 0.0
    idxs = list(np.nonzero(zero_one))[0]
    for idx in idxs:
        s += 1.0 / np.log2(idx + 2)
    m = 0.0
    length = min(len(test_lst), len(zero_one))      # Hit.
    for idx in range(length):
        m += 1.0 / np.log2(idx + 2)
    return s / m


def fun_idxs_of_max_n_score(user_scores_to_all_items, top_k):
    # Legacy implementation note.
    return np.argpartition(user_scores_to_all_items, -top_k)[-top_k:]


def fun_sort_idxs_max_to_min(user_max_n_idxs_scores):
    # Score computation note.
    # Score computation note.
    idxs, scores = user_max_n_idxs_scores           # Score computation note.
    return idxs[np.argsort(scores[idxs])][::-1]     # Score computation note.


def fun_predict_auc_recall_map_ndcg(
        p, model, best, epoch, starts_ends_auc, starts_ends_tes,
        tes_buys_masks, tes_masks):
    # ------------------------------------------------------------------------------------------------------------------
    # Vector dimension.
    # Matrix operation note.
    append = [[0] for _ in np.arange(len(tes_buys_masks))]

    # ------------------------------------------------------------------------------------------------------------------
    # auc
    all_upqs = np.array([[0 for _ in np.arange(len(tes_masks[0]))]])    # Initialization.
    for start_end in starts_ends_auc:
        sub_all_upqs = model.compute_sub_auc_preference(start_end)
        all_upqs = np.concatenate((all_upqs, sub_all_upqs))
    all_upqs = np.delete(all_upqs, 0, axis=0)
    auc = 1.0 * np.sum(all_upqs) / np.sum(tes_masks)    # All items.
    # Save results.
    if auc > best.best_auc:
        best.best_auc = auc
        best.best_epoch_auc = epoch

    # ------------------------------------------------------------------------------------------------------------------
    # recall, map, ndcg
    at_nums = p['at_nums']          # [5, 10, 15, 20, 30, 50]
    ranges = range(len(at_nums))

    # Score computation note.
    # Legacy implementation note.
    # Matrix operation note.
    all_ranks = np.array([[0 for _ in np.arange(at_nums[-1])]])   # Legacy implementation note.
    for start_end in starts_ends_tes:
        sub_all_scores = model.compute_sub_all_scores(start_end)  # shape=(sub_n_user, n_item)
        sub_score_ranks = np.apply_along_axis(
            func1d=fun_idxs_of_max_n_score,
            axis=1,
            arr=sub_all_scores,
            top_k=at_nums[-1])
        sub_all_ranks = np.apply_along_axis(
            func1d=fun_sort_idxs_max_to_min,
            axis=1,
            arr=np.array(zip(sub_score_ranks, sub_all_scores)))
        all_ranks = np.concatenate((all_ranks, sub_all_ranks))
        del sub_all_scores
    all_ranks = np.delete(all_ranks, 0, axis=0)     # Legacy implementation note.

    # Legacy implementation note.
    arr = np.array([0.0 for _ in ranges])
    recall, precis, f1scor, map, ndcg = arr.copy(), arr.copy(), arr.copy(), arr.copy(), arr.copy()
    hits, denominator_recalls = arr.copy(), np.sum(tes_masks)  # Legacy implementation note.
    for k in ranges:                            # Hit.
        recoms = all_ranks[:, :at_nums[k]]      # Legacy implementation note.
        # Hit.
        all_zero_ones = np.apply_along_axis(
            func1d=fun_hit_zero_one,
            axis=1,
            arr=np.array(zip(tes_buys_masks, recoms, tes_masks, append)))   # shape=(n_user, at_nums[k])
        hits[k] = np.sum(all_zero_ones)
        recall[k] = 1.0 * np.sum(all_zero_ones) / denominator_recalls
        precis[k] = 1.0 * np.sum(all_zero_ones) / (at_nums[k] * len(all_zero_ones))
        f1scor[k] = 2.0 * recall[k] * precis[k] / (recall[k] + precis[k])
        all_maps = np.apply_along_axis(
            func1d=fun_evaluate_map,
            axis=1,
            arr=np.array(zip(tes_buys_masks, all_zero_ones, tes_masks, append)))
        map[k] = np.mean(all_maps)
        all_ndcgs = np.apply_along_axis(
            func1d=fun_evaluate_ndcg,
            axis=1,
            arr=np.array(zip(tes_buys_masks, all_zero_ones, tes_masks, append)))
        ndcg[k] = np.mean(all_ndcgs)

    # Save results.
    for k in ranges:
        if recall[k] > best.best_recall[k]:
            best.best_recall[k] = recall[k]
            best.best_epoch_recall[k] = epoch
        if precis[k] > best.best_precis[k]:
            best.best_precis[k] = precis[k]
            best.best_epoch_precis[k] = epoch
        if f1scor[k] > best.best_f1scor[k]:
            best.best_f1scor[k] = f1scor[k]
            best.best_epoch_f1scor[k] = epoch
        if map[k] > best.best_map[k]:
            best.best_map[k] = map[k]
            best.best_epoch_map[k] = epoch
        if ndcg[k] > best.best_ndcg[k]:
            best.best_ndcg[k] = ndcg[k]
            best.best_epoch_ndcg[k] = epoch
    del all_upqs, all_ranks


def fun_predict_pop_random(
        p, best, all_upqs, all_ranks,
        tes_buys_masks, tes_masks):

    append = [[0] for _ in np.arange(len(tes_buys_masks))]

    # Legacy implementation note.
    if all_upqs is not None:
        auc = 1.0 * np.sum(all_upqs) / np.sum(tes_masks)    # All items.
        # Save results.
        best.best_auc = auc

    at_nums = p['at_nums']
    ranges = range(len(at_nums))
    # Legacy implementation note.
    arr = np.array([0.0 for _ in ranges])
    recall, precis, f1scor, map, ndcg = arr.copy(), arr.copy(), arr.copy(), arr.copy(), arr.copy()
    hits, denominator_recalls = arr.copy(), np.sum(tes_masks)  # Legacy implementation note.
    for k in ranges:                            # Hit.
        recoms = all_ranks[:, :at_nums[k]]      # Legacy implementation note.
        # Hit.
        all_zero_ones = np.apply_along_axis(
            func1d=fun_hit_zero_one,
            axis=1,
            arr=np.array(zip(tes_buys_masks, recoms, tes_masks, append)))   # shape=(n_user, at_nums[k])
        hits[k] = np.sum(all_zero_ones)
        recall[k] = 1.0 * np.sum(all_zero_ones) / denominator_recalls
        precis[k] = 1.0 * np.sum(all_zero_ones) / (at_nums[k] * len(all_zero_ones))
        f1scor[k] = 2.0 * recall[k] * precis[k] / (recall[k] + precis[k])
        all_maps = np.apply_along_axis(
            func1d=fun_evaluate_map,
            axis=1,
            arr=np.array(zip(tes_buys_masks, all_zero_ones, tes_masks, append)))
        map[k] = np.mean(all_maps)
        all_ndcgs = np.apply_along_axis(
            func1d=fun_evaluate_ndcg,
            axis=1,
            arr=np.array(zip(tes_buys_masks, all_zero_ones, tes_masks, append)))
        ndcg[k] = np.mean(all_ndcgs)

    # Save results.
    for k in ranges:
        best.best_recall[k] = recall[k]
        best.best_precis[k] = precis[k]
        best.best_f1scor[k] = f1scor[k]
        best.best_map[k] = map[k]
        best.best_ndcg[k] = ndcg[k]


def fun_acquire_fil_para(model_name, p):
    # Save results.
    alpha_lambda = [p['alpha'], p['lambda']]
    batch_sizes = [p['batch_size_train'], p['batch_size_test']]
    size, epoch, at_nums = p['latent_size'], p['epochs'], p['at_nums']
    if p['loss_weight']:
        ls = p['loss_weight']
    else:
        ls = [0, 0]
    fil_para = \
        '\n' + model_name + \
        '\n\t' + 'alpha, lambda = {v1}'.format(v1=', '.join([str(i) for i in alpha_lambda])) + \
        '\n\t' + 'ls_lmdd, ls_bpr = {v1}'.format(v1=', '.join([str(i) for i in ls])) + \
        '\n\t' + 'batch_size train, test = {v1}'.format(v1=', '.join([str(i) for i in batch_sizes])) + \
        '\n\t' + 'size, epoch, at_nums = {v1}d, {v2}, top-{v3}'.format(v1=size, v2=epoch, v3=at_nums) + \
        '\n'
    return fil_para


def fun_save_best_and_losses(
        path, model_name, epoch, p, best, losses):
    # Legacy implementation note.
    if os.path.exists(path):
        print('\t\tdir exists: {v1}'.format(v1=path))
    else:
        os.makedirs(path)
        print('\t\tdir is made: {v1}'.format(v1=path))
    size = p['latent_size']
    size = '{v1}d_'.format(v1=size)
    fil_name = size + model_name + '.txt'       # Legacy implementation note.
    fil = os.path.join(path, fil_name)
    print('\t\tfile name: {v1}'.format(v1=fil_name))

    # Save results.
    f = open(fil, 'a')
    fil_para = fun_acquire_fil_para(model_name, p)
    fil_best = best.fun_obtain_best(epoch)
    fil_loss = \
        '\n\tLosses: ' + \
        '\n\t\t[{v1}]'.format(v1=', '.join(losses))
    f.write(fil_para)
    f.write(fil_best)
    f.write(fil_loss)
    f.write('\n')
    f.close()


# Legacy implementation note.


@exe_time  # Legacy implementation note.
def main():
    print('... construct the evaluation program')


if '__main__' == __name__:
    main()
