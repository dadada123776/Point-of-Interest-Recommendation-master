#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import time
from math import cos, sqrt, asin
import numpy as np
import pandas as pd
import random
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


def cal_dis(lat1, lon1, lat2, lon2, dd, dist_num):
    """Haversine formula for distance between two latitude-longitude points."""
    d = 12742                           # Legacy implementation note.
    p = 0.017453292519943295            # Legacy implementation note.
    a = (lat1 - lat2) * p
    b = (lon1 - lon2) * p
    # Legacy implementation note.
    c = (1.0 - cos(a)) / 2 + cos(lat1 * p) * cos(lat2 * p) * (1.0 - cos(b)) / 2     # Legacy implementation note.
    dist = d * asin(sqrt(c))            # Legacy implementation note.

    interval = int(dist * 1000 / dd)    # Distance interval.
    interval = min(interval, dist_num)
    # Legacy implementation note.
    # Legacy implementation note.
    return interval


def load_data(dataset, mode, split, dd, dist_num):
    """Load the check-in sequence file and generate data."""
    # Legacy implementation note.
    print('Original data ...')
    pois = pd.read_csv(dataset, sep=' ')
    all_user_pois = [[i for i in upois.split('/')] for upois in pois['u_pois']]
    all_user_cods = [[i.split(',') for i in upois.split('/')] for upois in pois['u_coordinates']]       # string
    all_user_cods = [[[float(ucod[0]), float(ucod[1])] for ucod in ucods] for ucods in all_user_cods]   # float
    all_trans = [item for upois in all_user_pois for item in upois]
    all_cordi = [ucod for ucods in all_user_cods for ucod in ucods]
    poi_cordi = dict(zip(all_trans, all_cordi))  # Legacy implementation note.
    tran_num, user_num, item_num = len(all_trans), len(all_user_pois), len(set(all_trans))
    print('\tusers, items, trans:  = {v1}, {v2}, {v3}'.format(v1=user_num, v2=item_num, v3=tran_num))
    print('\tavg. user check:      = {val}'.format(val=1.0 * tran_num / user_num))
    print('\tavg. poi checked:     = {val}'.format(val=1.0 * tran_num / item_num))
    print('\tdistance interval     = [0, {val}]'.format(val=dist_num))

    # Legacy implementation note.
    print('Split the training set, test set: mode = {val} ...'.format(val=mode))
    tra_pois, tes_pois = [], []
    tra_dist, tes_dist = [], []                 # Legacy implementation note.
    for upois, ucods in zip(all_user_pois, all_user_cods):
        left, right = upois[:split], [upois[split]]   # Legacy implementation note.

        # Distance interval.
        dist = []
        for i, cord in enumerate(ucods[1:]):    # Distance interval.
            pre = ucods[i]
            dist.append(cal_dis(cord[0], cord[1], pre[0], pre[1], dd, dist_num))
        dist = [dist_num] + dist                # Distance interval.
        dist_lf, dist_rt = dist[:split], [dist[split]]

        # Save results.
        tra_pois.append(left)
        tes_pois.append(right)
        tra_dist.append(dist_lf)
        tes_dist.append(dist_rt)

    # Legacy implementation note.
    # all_trans = []
    # for utra, utes in zip(tra_pois, tes_pois):
    #     all_trans.extend(utra)
    #     all_trans.extend(utes)
    # tran_num, user_num, item_num = len(all_trans), len(tra_pois), len(set(all_trans))
    # temp = tra_dist
    # temp.extend(tes_dist)
    # all_dists = [item for upois in temp for item in upois]
    # print('\tusers, items, trans:    = {v1}, {v2}, {v3}'.format(v1=user_num, v2=item_num, v3=tran_num))
    # print('\tavg. user poi:          = {val}'.format(val=1.0 * tran_num / user_num))
    # print('\tavg. item bought:       = {val}'.format(val=1.0 * tran_num / item_num))
    # print('\tdistance interval     = [0, {val}]'.format(val=max_dist))

    # Legacy implementation note.
    print('Use aliases to represent pois ...')
    all_items = set(all_trans)
    aliases_dict = dict(zip(all_items, range(item_num)))    # Legacy implementation note.
    tra_pois = [[aliases_dict[i] for i in utra] for utra in tra_pois]
    tes_pois = [[aliases_dict[i] for i in utes] for utes in tes_pois]
    # Legacy implementation note.
    cordi_new = dict()
    for poi in poi_cordi.keys():
        cordi_new[aliases_dict[poi]] = poi_cordi[poi]       # Legacy implementation note.
    pois_cordis = [cordi_new[k] for k in sorted(cordi_new.keys())]

    return [(user_num, item_num), pois_cordis, (tra_pois, tes_pois), (tra_dist, tes_dist)]


def fun_data_buys_masks(all_usr_pois, all_usr_dist, item_tail, dist_tail):
    # Legacy implementation note.
    # Legacy implementation note.
    # Legacy implementation note.
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = max(us_lens)
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
    us_dist = [udist + dist_tail * (len_max - le) for udist, le in zip(all_usr_dist, us_lens)]
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    return us_pois, us_dist, us_msks


def fun_random_neg_masks_tra(item_num, tras_mask):
    """Randomly sample negative items."""
    us_negs = []
    for utra in tras_mask:     # Legacy implementation note.
        unegs = []
        for i, e in enumerate(utra):
            if item_num == e:                        # Legacy implementation note.
                unegs += [item_num] * (len(utra) - i)   # Legacy implementation note.
                break
            j = random.randint(0, item_num - 1)      # Matrix operation note.
            while j in utra:                     # Legacy implementation note.
                j = random.randint(0, item_num - 1)
            unegs += [j]
        us_negs.append(unegs)
    return us_negs


def fun_random_neg_masks_tes(item_num, tras_mask, tess_mask):
    """Randomly sample negative items."""
    us_negs = []
    for utra, utes in zip(tras_mask, tess_mask):
        unegs = []
        for i, e in enumerate(utes):
            if item_num == e:                   # Legacy implementation note.
                unegs += [item_num] * (len(utes) - i)
                break
            j = random.randint(0, item_num - 1)
            while j in utra or j in utes:         # Legacy implementation note.
                j = random.randint(0, item_num - 1)
            unegs += [j]
        us_negs.append(unegs)
    return us_negs


def fun_compute_dist_neg(tra_buys_masks, tra_masks, tra_buys_neg_masks, pois_cordis, dd, dist_num):
    """Distance interval."""
    tra_dist_neg_masks = []
    for upois, umasks, upois_neg in zip(tra_buys_masks, tra_masks, tra_buys_neg_masks):
        # Distance interval.
        dist = []
        for i in range(1, sum(umasks)):     # Legacy implementation note.
            pre = pois_cordis[upois[i-1]]
            cur_neg = pois_cordis[upois_neg[i]]
            dist.append(cal_dis(cur_neg[0], cur_neg[1], pre[0], pre[1], dd, dist_num))
        # Distance interval.
        dist = [dist_num] + dist + [dist_num] * (len(upois) - sum(umasks))
        tra_dist_neg_masks.append(dist)
    return tra_dist_neg_masks


def fun_compute_distance(tra_pois_masks, tra_masks, pois_cordis, dd, dist_num):
    """Distance interval."""
    def fun_poi_to_all_intervals(poi):
        """Legacy implementation note."""
        last_poi_cordi = pois_cordis[poi[0]]
        all_inters = []
        for cordi in pois_cordis:
            inter = cal_dis(last_poi_cordi[0], last_poi_cordi[1], cordi[0], cordi[1], dd, dist_num)
            all_inters.append(inter)
        return all_inters

    le = len(tra_pois_masks)
    # Legacy implementation note.
    usrs_last_pois = np.asarray(tra_pois_masks)[
        np.arange(le),
        np.sum(tra_masks, axis=1) - 1]

    # Distance interval.
    usrs_last_poi_to_all_intervals = np.apply_along_axis(
        func1d=fun_poi_to_all_intervals,
        axis=1,
        arr=usrs_last_pois[:, np.newaxis])  # shape=(n_item, 1)
    return usrs_last_poi_to_all_intervals


def fun_acquire_prob(all_sus, ulptai, dist_num):
    """Legacy implementation note."""
    def fun_uprob_uinterval(prob_inter):
        # prob.shape=(n_usr, 380), inter.shape=(n_usr, n_item)
        # Legacy implementation note.
        prob, interval, mask = prob_inter       # Legacy implementation note.
        usr_probs_to_all_pois = prob[interval]  # Legacy implementation note.
        usr_probs_to_all_pois *= mask           # Legacy implementation note.
        return usr_probs_to_all_pois

    probs_mask = np.asarray(ulptai) < dist_num  # Legacy implementation note.
    usrs_probs_to_all_pois = np.apply_along_axis(
        func1d=fun_uprob_uinterval,
        axis=1,
        arr=np.array(zip(all_sus, ulptai, probs_mask)))
    return usrs_probs_to_all_pois


@exe_time  # Legacy implementation note.
def main():
    print('... load the dataset, and  no need to set shared.')


if '__main__' == __name__:
    main()
