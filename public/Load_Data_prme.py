#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import time
import pandas as pd
import random
import numpy as np
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


def rad(x):
    return np.multiply(x, np.pi) / 180.0


def cal_dis(latitude1, longitude1, latitude2, longitude2):
    R = 6378.137
    radLat1 = rad(latitude1)
    radLat2 = rad(latitude2)
    a = radLat1 - radLat2
    b = rad(longitude1) - rad(longitude2)
    s = 2 * np.arcsin(np.sqrt(np.power(np.sin(a / 2), 2) +
                                np.cos(radLat1) * np.cos(radLat2) * np.power(np.sin(b / 2), 2)))
    s = s * R
    return s


def load_data(dataset, mode, split):
    """Load the check-in sequence file and generate data."""
    # Legacy implementation note.
    print('Original data ...')
    pois = pd.read_csv(dataset, sep=' ')
    # check_times pois_different u_id u_pois u_times u_coordinates
    all_user_pois = [[i for i in upois.split('/')] for upois in pois['u_pois']]
    all_user_times = [[float(i) for i in upois.split('/')] for upois in pois['u_times']]
    all_user_cordi = [[i.split(',') for i in upois.split('/')] for upois in pois['u_coordinates']]
    all_trans = [item for upois in all_user_pois for item in upois]
    all_cordi = [cordi for ucordi in all_user_cordi for cordi in ucordi]
    cordi = dict(zip(all_trans, all_cordi))
    tran_num, user_num, item_num = len(all_trans), len(all_user_pois), len(set(all_trans))
    print('\tusers, items, trans:    = {v1}, {v2}, {v3}'.format(v1=user_num, v2=item_num, v3=tran_num))
    print('\tavg. user check:        = {val}'.format(val=1.0 * tran_num / user_num))
    print('\tavg. poi checked:       = {val}'.format(val=1.0 * tran_num / item_num))

    # Legacy implementation note.
    print('Split the training set, test set: mode = {val} ...'.format(val=mode))
    tra_pois, tes_pois = [], []
    tra_gaps, tes_gaps = [], []
    tra_dist, tes_dist = [], []
    for i in range(len(pois)):
        # Legacy implementation note.
        le = len(all_user_pois[i])
        split0, split1, split2 = 0, int(le * split[0]), int(le * split[1])
        left, right = all_user_pois[i][split0: split1], all_user_pois[i][split1: split2]

        # Legacy implementation note.
        gap = []
        dist = []
        for j in range(0, le):
            gap.append(all_user_times[i][j] - all_user_times[i][j-1])
            dist.append(cal_dis(float(all_user_cordi[i][j][0]), float(all_user_cordi[i][j][1]),
                                float(all_user_cordi[i][j - 1][0]), float(all_user_cordi[i][j - 1][1])))
        gap[0] = 0
        dist[0] = 0
        gap_lf, gap_rt = gap[split0: split1], gap[split1: split2]
        dist_lf, dist_rt = dist[split0: split1], dist[split1: split2]

        # Save results.
        tra_pois.append(left)
        tes_pois.append(right)
        tra_gaps.append(gap_lf)
        tes_gaps.append(gap_rt)
        tra_dist.append(dist_lf)
        tes_dist.append(dist_rt)

    # Legacy implementation note.
    all_trans = []
    for utra, utes in zip(tra_pois, tes_pois):
        all_trans.extend(utra)
        all_trans.extend(utes)
    tran_num, user_num, item_num = len(all_trans), len(tra_pois), len(set(all_trans))
    print('\tusers, items, trans:    = {v1}, {v2}, {v3}'.format(v1=user_num, v2=item_num, v3=tran_num))
    print('\tavg. user poi:          = {val}'.format(val=1.0 * tran_num / user_num))
    print('\tavg. item bought:       = {val}'.format(val=1.0 * tran_num / item_num))

    # Legacy implementation note.
    print('Use aliases to represent pois ...')
    all_items = set(all_trans)
    aliases_dict = dict(zip(all_items, range(item_num)))
    tra_pois = [[aliases_dict[i] for i in utra] for utra in tra_pois]
    tes_pois = [[aliases_dict[i] for i in utes] for utes in tes_pois]
    cordi_new = dict()
    for i in cordi:
        cordi_new[aliases_dict[i]] = cordi[i]
    location = []
    for i in range(len(cordi)):
        location.append(cordi_new[i])
    location.append([0.0, 0.0])
    location = np.asarray(location, dtype='float')

    tra_all_dist, tes_all_dist = tra_dist, tes_dist
    tra_all_times, tes_all_times = tra_gaps, tes_gaps

    return [(user_num, item_num, location), (tra_pois, tes_pois), (tra_all_times, tes_all_times), (tra_all_dist, tes_all_dist)]


def fun_data_pois_masks(all_usr_pois, all_usr_times, all_usr_dists, item_tail):
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = max(us_lens)
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
    us_all_times = [utimes + [0] * (len_max - le) for utimes, le in zip(all_usr_times, us_lens)]
    us_all_dists = [udists + [0] * (len_max - le) for udists, le in zip(all_usr_dists, us_lens)]
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    return us_pois, us_all_times, us_all_dists, us_msks


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


@exe_time  # Legacy implementation note.
def main():
    print('... load the dataset, and  no need to set shared.')


if '__main__' == __name__:
    main()
