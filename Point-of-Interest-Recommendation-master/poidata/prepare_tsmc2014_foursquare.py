#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Convert TSMC2014 Foursquare NYC/TKY files to project sequence format."""

from __future__ import print_function 

import argparse 
import datetime 
import os 
from collections import Counter ,defaultdict 


def parse_args ():
    parser =argparse .ArgumentParser (
    description ="Prepare TSMC2014 Foursquare data as sequence files."
    )
    parser .add_argument ("--input",required =True ,help ="dataset_TSMC2014_*.txt/csv")
    parser .add_argument ("--output",required =True ,help ="Output sequence .txt path")
    parser .add_argument (
    "--min-user-checkins",
    type =int ,
    default =10 ,
    help ="Drop users with fewer check-ins.",
    )
    parser .add_argument (
    "--min-poi-checkins",
    type =int ,
    default =10 ,
    help ="Drop POIs with fewer check-ins.",
    )
    parser .add_argument (
    "--max-iterations",
    type =int ,
    default =10 ,
    help ="Maximum pruning iterations.",
    )
    return parser .parse_args ()


def parse_time (value ):
    formats =[
    "%a %b %d %H:%M:%S %z %Y",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%d %H:%M:%S",
    ]
    for fmt in formats :
        try :
            dt =datetime .datetime .strptime (value ,fmt )
            if dt .tzinfo is not None :
                dt =dt .astimezone (datetime .timezone .utc ).replace (tzinfo =None )
            epoch =datetime .datetime (1970 ,1 ,1 )
            return int ((dt -epoch ).total_seconds ()/60 )
        except ValueError :
            pass 
    raise ValueError ("Unsupported timestamp: {0}".format (value ))


def split_line (line ):
    if "\t"in line :
        return line .rstrip ("\n").split ("\t")
    return line .rstrip ("\n").split (",")


def load_checkins (path ):
    users =defaultdict (list )
    with open (path ,"r",encoding ="utf-8",errors ="replace")as handle :
        for line_no ,line in enumerate (handle ,start =1 ):
            if not line .strip ():
                continue 
            parts =split_line (line )
            if len (parts )<8 :
                raise ValueError ("Bad line {0}: {1}".format (line_no ,line [:120 ]))
            if line_no ==1 and parts [0 ].lower ().replace (" ","_")in (
            "user_id",
            "userid",
            ):
                continue 

            user_id =parts [0 ]
            poi_id =parts [1 ]
            lat =parts [4 ]
            lon =parts [5 ]
            timestamp =parse_time (parts [7 ])
            users [user_id ].append ((timestamp ,poi_id ,lat +","+lon ))
    return users 


def prune (users ,min_user_checkins ,min_poi_checkins ,max_iterations ):
    users ={
    user_id :checkins 
    for user_id ,checkins in users .items ()
    if len (checkins )>=min_user_checkins 
    }
    for iteration in range (max_iterations ):
        poi_counter =Counter ()
        for checkins in users .values ():
            poi_counter .update (item [1 ]for item in checkins )
        valid_pois ={
        poi_id 
        for poi_id ,count in poi_counter .items ()
        if count >=min_poi_checkins 
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
    print ("Loading: {0}".format (args .input ))
    users =load_checkins (args .input )
    print (
    "Loaded users={0}, checkins={1}".format (
    len (users ),sum (len (items )for items in users .values ())
    )
    )
    users =prune (
    users ,
    args .min_user_checkins ,
    args .min_poi_checkins ,
    args .max_iterations ,
    )
    write_sequence (users ,args .output )


if __name__ =="__main__":
    main ()
