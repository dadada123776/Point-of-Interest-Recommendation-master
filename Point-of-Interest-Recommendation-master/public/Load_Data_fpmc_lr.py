#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import time
from math import sin, cos, sqrt, asin
import numpy as np
import pandas as pd
import random
from collections import defaultdict
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


def cal_dis(lat1, lon1, lat2, lon2):
    """Haversine formula for distance between two latitude-longitude points."""
    d = 12742                           # Legacy implementation note.
    p = 0.017453292519943295            # Legacy implementation note.
    a = (lat1 - lat2) * p
    b = (lon1 - lon2) * p
    # Legacy implementation note.
    c = (1.0 - cos(a)) / 2 + cos(lat1 * p) * cos(lat2 * p) * (1.0 - cos(b)) / 2     # Legacy implementation note.
    dist = d * asin(sqrt(c))            # Legacy implementation note.
    return dist


def load_data(dataset, mode, split):
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

    # Legacy implementation note.
    print('Split the training set, test set: mode = {val} ...'.format(val=mode))
    tra_pois, tes_pois = [], []
    for upois, ucods in zip(all_user_pois, all_user_cods):
        left, right = upois[:split], [upois[split]]   # Legacy implementation note.
        # Save results.
        tra_pois.append(left)
        tes_pois.append(right)

    # Legacy implementation note.
    print('Use aliases to represent pois ...')
    all_items = set(all_trans)
    aliases_dict = dict(zip(all_items, range(item_num)))    # Legacy implementation note.
    tra_pois = [[aliases_dict[i] for i in utra] for utra in tra_pois]
    tes_pois = [[aliases_dict[i] for i in utes] for utes in tes_pois]
    tra_last_poi = [utra[-1] for utra in tra_pois]          # Legacy implementation note.
    # Legacy implementation note.
    cordi_new = dict()
    for poi in poi_cordi.keys():
        cordi_new[aliases_dict[poi]] = poi_cordi[poi]       # Legacy implementation note.
    pois_cordis = [cordi_new[k] for k in sorted(cordi_new.keys())]

    return [(user_num, item_num), pois_cordis, (tra_pois, tes_pois), tra_last_poi]


def fun_data_buys_masks(all_usr_pois, item_tail):
    # Legacy implementation note.
    # Legacy implementation note.
    # Legacy implementation note.
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = max(us_lens)
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    return us_pois, us_msks


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


def fun_acquire_neighbors_for_each_poi(pois_cordis, max_dist):
    """Distance truncation threshold."""
    def fun_poi_to_all_pois(poi_idx):
        """Distance truncation threshold."""
        poi_idx = poi_idx[0]
        poi_cor = pois_cordis[poi_idx]
        all_dists = []
        for each_cor in pois_cordis:
            dist = cal_dis(poi_cor[0], poi_cor[1], each_cor[0], each_cor[1])
            all_dists.append(dist)
        all_dists = np.asarray(all_dists) <= max_dist
        all_neibs = np.nonzero(all_dists)[0]
        poi_neighbors[poi_idx] = list(set(all_neibs) - {poi_idx})     # Legacy implementation note.
        return [1]

    poi_neighbors = defaultdict(list)
    # Legacy implementation note.
    _ = np.apply_along_axis(    # Legacy implementation note.
        func1d=fun_poi_to_all_pois,
        axis=1,
        arr=np.arange(len(pois_cordis))[:, np.newaxis])     # shape=(n_item, 1)
    return poi_neighbors


def fun_acquire_negs_tra(tra_pois, all_pois_neighbors):
    """Legacy implementation note."""
    # item_num = len(all_pois_neighbors)
    all_negs = []
    for utras in tra_pois:
        unegs = []
        for i, utra in enumerate(utras):
            negs = all_pois_neighbors[utra]
            # Distance truncation threshold.
            #     negs = [random.randint(0, item_num - 1)]
            unegs.append(negs)
        all_negs.append(unegs)
    return all_negs


@exe_time  # Legacy implementation note.
def main():
    pass


if '__main__' == __name__:
    main()
