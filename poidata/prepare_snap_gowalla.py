#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Convert SNAP Gowalla check-ins to this project's sequence format.

Input format from SNAP ``loc-gowalla_totalCheckins.txt.gz``:

    user_id checkin_time latitude longitude location_id

Output format:

    check_times pois_different u_id u_pois u_times u_coordinates

The output can be consumed by ``prog_mostpop.py`` and the existing POI
recommendation loaders.
"""

from __future__ import print_function 

import argparse 
import datetime 
import gzip 
import os 
from collections import Counter ,defaultdict 


SNAP_URL ="https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz"


def parse_args ():
    parser =argparse .ArgumentParser (
    description ="Prepare SNAP Gowalla check-ins as sequence/Gowalla.txt."
    )
    parser .add_argument (
    "--input",
    required =True ,
    help ="Path to loc-gowalla_totalCheckins.txt or .gz.",
    )
    parser .add_argument (
    "--output",
    default =os .path .join ("Gowalla","sequence","Gowalla.txt"),
    help ="Output sequence file path.",
    )
    parser .add_argument (
    "--min-user-checkins",
    type =int ,
    default =5 ,
    help ="Drop users with fewer check-ins.",
    )
    parser .add_argument (
    "--min-poi-users",
    type =int ,
    default =5 ,
    help ="Drop POIs visited by fewer distinct users.",
    )
    parser .add_argument (
    "--max-iterations",
    type =int ,
    default =10 ,
    help ="Maximum pruning iterations for user/POI frequency filters.",
    )
    return parser .parse_args ()


def open_text (path ):
    if path .endswith (".gz"):
        return gzip .open (path ,"rt")
    return open (path ,"r")


def parse_timestamp (value ):
    dt =datetime .datetime .strptime (value ,"%Y-%m-%dT%H:%M:%SZ")
    epoch =datetime .datetime (1970 ,1 ,1 )
    return int ((dt -epoch ).total_seconds ()/60 )


def load_checkins (path ):
    users =defaultdict (list )
    with open_text (path )as handle :
        for line_no ,line in enumerate (handle ,start =1 ):
            parts =line .strip ().split ()
            if not parts :
                continue 
            if len (parts )!=5 :
                raise ValueError ("Bad line {0}: {1}".format (line_no ,line [:120 ]))

            user_id ,checkin_time ,lat ,lon ,poi_id =parts 
            minute =parse_timestamp (checkin_time )
            users [user_id ].append ((minute ,poi_id ,lat +","+lon ))
    return users 


def prune_users_and_pois (users ,min_user_checkins ,min_poi_users ,max_iterations ):
    users ={
    user_id :checkins 
    for user_id ,checkins in users .items ()
    if len (checkins )>=min_user_checkins 
    }

    for iteration in range (max_iterations ):
        poi_users =defaultdict (set )
        for user_id ,checkins in users .items ():
            for _ ,poi_id ,_ in checkins :
                poi_users [poi_id ].add (user_id )

        valid_pois ={
        poi_id for poi_id ,user_ids in poi_users .items ()if len (user_ids )>=min_poi_users 
        }
        next_users ={}
        for user_id ,checkins in users .items ():
            filtered =[item for item in checkins if item [1 ]in valid_pois ]
            if len (filtered )>=min_user_checkins :
                next_users [user_id ]=filtered 

        print (
        "Iteration {0}: users={1}, pois={2}, checkins={3}".format (
        iteration +1 ,
        len (next_users ),
        len (valid_pois ),
        sum (len (items )for items in next_users .values ()),
        )
        )

        if len (next_users )==len (users ):
            users =next_users 
            break 
        users =next_users 

    return users 


def write_sequence (users ,output_path ):
    output_dir =os .path .dirname (output_path )
    if output_dir and not os .path .exists (output_dir ):
        os .makedirs (output_dir )

    rows =[]
    for user_id ,checkins in users .items ():
        ordered =sorted (checkins ,key =lambda item :item [0 ])
        poi_ids =[item [1 ]for item in ordered ]
        times =[str (item [0 ])for item in ordered ]
        coordinates =[item [2 ]for item in ordered ]
        rows .append (
        (
        len (poi_ids ),
        "{0:.2f}".format (float (len (set (poi_ids )))/len (poi_ids )),
        user_id ,
        "/".join (poi_ids ),
        "/".join (times ),
        "/".join (coordinates ),
        )
        )

    rows .sort (key =lambda row :(-row [0 ],-float (row [1 ]),row [2 ]))
    with open (output_path ,"w",newline ="")as handle :
        handle .write (
        "check_times pois_different u_id u_pois u_times u_coordinates\n"
        )
        for row in rows :
            handle .write ("{0} {1} {2} {3} {4} {5}\n".format (*row ))

    poi_counter =Counter ()
    for _ ,_ ,_ ,u_pois ,_ ,_ in rows :
        poi_counter .update (u_pois .split ("/"))

    print ("Wrote: {0}".format (os .path .abspath (output_path )))
    print ("Users: {0}".format (len (rows )))
    print ("POIs: {0}".format (len (poi_counter )))
    print ("Check-ins: {0}".format (sum (poi_counter .values ())))


def main ():
    args =parse_args ()
    print ("SNAP Gowalla source: {0}".format (SNAP_URL ))
    print ("Loading: {0}".format (args .input ))
    users =load_checkins (args .input )
    print (
    "Loaded users={0}, checkins={1}".format (
    len (users ),sum (len (items )for items in users .values ())
    )
    )
    users =prune_users_and_pois (
    users ,
    args .min_user_checkins ,
    args .min_poi_users ,
    args .max_iterations ,
    )
    write_sequence (users ,args .output )


if __name__ =="__main__":
    main ()
