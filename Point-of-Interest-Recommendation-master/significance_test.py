#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""User-level significance tests for recommendation results.

Input files must be per-user metric CSV files produced by prog_fedmff.py or
another evaluator with matching user_id and metric columns.
"""

from __future__ import print_function 

import argparse 
import csv 
import math 
import random 


def parse_args ():
    parser =argparse .ArgumentParser (description ="Compare two per-user metric CSVs.")
    parser .add_argument ("--ours",required =True ,help ="Per-user CSV for our model.")
    parser .add_argument ("--baseline",required =True ,help ="Per-user CSV for baseline.")
    parser .add_argument ("--metrics",default ="precision@10,precision@20,recall@10,recall@20,ap@20")
    parser .add_argument ("--bootstrap",type =int ,default =10000 )
    parser .add_argument ("--seed",type =int ,default =2026 )
    return parser .parse_args ()


def read_metrics (path ):
    data ={}
    with open (path ,"r",newline ="")as handle :
        reader =csv .DictReader (handle )
        if "user_id"not in (reader .fieldnames or []):
            raise ValueError ("{0} must contain a user_id column".format (path ))
        for row in reader :
            data [row ["user_id"]]=row 
    return data 


def normal_cdf (value ):
    return 0.5 *(1.0 +math .erf (value /math .sqrt (2.0 )))


def sign_test_pvalue (diffs ):
    positives =sum (1 for diff in diffs if diff >0 )
    negatives =sum (1 for diff in diffs if diff <0 )
    n =positives +negatives 
    if n ==0 :
        return 1.0 
    k =min (positives ,negatives )
    prob =0.0 
    for i in range (k +1 ):
        prob +=math .comb (n ,i )*(0.5 **n )
    return min (1.0 ,2.0 *prob )


def paired_t_pvalue (diffs ):
    n =len (diffs )
    if n <2 :
        return 1.0 
    mean =sum (diffs )/n 
    variance =sum ((diff -mean )**2 for diff in diffs )/(n -1 )
    if variance ==0 :
        return 0.0 if mean !=0 else 1.0 
    t_stat =mean /math .sqrt (variance /n )
    # Normal approximation is adequate for large user-level samples.
    return 2.0 *(1.0 -normal_cdf (abs (t_stat )))


def bootstrap_ci (diffs ,rounds ):
    if not diffs :
        return 0.0 ,0.0 
    means =[]
    n =len (diffs )
    for _ in range (rounds ):
        sample =[diffs [random .randrange (n )]for _ in range (n )]
        means .append (sum (sample )/n )
    means .sort ()
    lo =means [int (0.025 *(rounds -1 ))]
    hi =means [int (0.975 *(rounds -1 ))]
    return lo ,hi 


def main ():
    args =parse_args ()
    random .seed (args .seed )
    ours =read_metrics (args .ours )
    baseline =read_metrics (args .baseline )
    common_users =sorted (set (ours ).intersection (baseline ))
    if not common_users :
        raise SystemExit ("No overlapping user_id values.")

    print ("users={0}".format (len (common_users )))
    print ("metric, ours_mean, baseline_mean, diff_mean, sign_p, paired_t_p, ci95_low, ci95_high")
    for metric in [m .strip ()for m in args .metrics .split (",")if m .strip ()]:
        diffs =[]
        ours_values =[]
        base_values =[]
        for user_id in common_users :
            if metric not in ours [user_id ]or metric not in baseline [user_id ]:
                raise ValueError ("Metric {0} missing from one input file".format (metric ))
            ov =float (ours [user_id ][metric ])
            bv =float (baseline [user_id ][metric ])
            ours_values .append (ov )
            base_values .append (bv )
            diffs .append (ov -bv )
        lo ,hi =bootstrap_ci (diffs ,args .bootstrap )
        print (
        "{0}, {1:.6f}, {2:.6f}, {3:.6f}, {4:.6g}, {5:.6g}, {6:.6f}, {7:.6f}".format (
        metric ,
        sum (ours_values )/len (ours_values ),
        sum (base_values )/len (base_values ),
        sum (diffs )/len (diffs ),
        sign_test_pvalue (diffs ),
        paired_t_pvalue (diffs ),
        lo ,
        hi ,
        )
        )


if __name__ =="__main__":
    main ()
