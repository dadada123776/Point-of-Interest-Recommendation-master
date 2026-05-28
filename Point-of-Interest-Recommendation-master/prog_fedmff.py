#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Multi-factor fusion enabled federated POI recommender.

This script implements the core training and evaluation workflow of the paper
"Multi-Factor Fusion Enabled Federated POI Recommendation in Personal Data
Trusteeship Scenario".

Main components:

* data-trustee level federated training with FedAvg
* local user-POI interaction sequence modeling
* similar-user and similar-POI graph construction
* social-relation representation through optional social links
* graph attention aggregation for relation-specific neighborhoods
* GRU-based behavioral preference encoder with temporal attention
* semantic-level attention over social, similarity, attribute, behavioral and
  POI-similarity factors
* BPR loss and Top-K recommendation evaluation
* per-user metric export for significance testing

The legacy POI baselines in this repository are kept unchanged. This file is an
independent paper-oriented entry point.
"""

from __future__ import print_function 

import argparse 
import copy 
import csv 
import math 
import os 
import random 
from collections import defaultdict 


def require_torch ():
    try :
        import torch 
        import torch .nn as nn 
        import torch .nn .functional as F 
    except ImportError as exc :
        raise SystemExit (
        "PyTorch is required for prog_fedmff.py. Install it first, for example:\n"
        "  pip install torch\n"
        "Then rerun this script."
        )from exc 
    return torch ,nn ,F 


def parse_args ():
    parser =argparse .ArgumentParser (
    description ="Train the paper-aligned federated multi-factor POI model."
    )
    parser .add_argument (
    "--file",
    default =os .path .join (
    "poidata","Foursquare_NYC","sequence","Foursquare_NYC.txt"
    ),
    help ="Sequence file with u_id, u_pois, u_times and u_coordinates columns.",
    )
    parser .add_argument ("--social-file",default =None ,help ="Optional CSV: user_a,user_b")
    parser .add_argument (
    "--user-attr-file",
    default =None ,
    help ="Optional CSV: u_id,attr_1,attr_2,... categorical or numeric values.",
    )
    parser .add_argument ("--train-ratio",type =float ,default =0.8 )
    parser .add_argument ("--embedding-dim",type =int ,default =32 )
    parser .add_argument ("--gat-layers",type =int ,default =2 )
    parser .add_argument ("--local-epochs",type =int ,default =5 )
    parser .add_argument ("--rounds",type =int ,default =20 )
    parser .add_argument ("--num-trustees",type =int ,default =5 )
    parser .add_argument ("--neighbor-k",type =int ,default =20 )
    parser .add_argument ("--batch-size",type =int ,default =512 )
    parser .add_argument ("--lr",type =float ,default =0.001 )
    parser .add_argument ("--reg",type =float ,default =1e-5 )
    parser .add_argument ("--topk",default ="10,20")
    parser .add_argument ("--seed",type =int ,default =2026 )
    parser .add_argument ("--device",default ="cpu")
    parser .add_argument (
    "--per-user-output",
    default =os .path .join ("results","fedmff_per_user_metrics.csv"),
    help ="CSV file for per-user metrics used by significance tests.",
    )
    return parser .parse_args ()


def set_seed (seed ):
    random .seed (seed )
    torch ,_ ,_ =require_torch ()
    torch .manual_seed (seed )
    if torch .cuda .is_available ():
        torch .cuda .manual_seed_all (seed )


class SequenceData (object ):
    def __init__ (self ):
        self .user_ids =[]
        self .poi_ids =[]
        self .poi_coords ={}
        self .train =[]
        self .test =[]
        self .user_map ={}
        self .poi_map ={}
        self .rev_user_map ={}
        self .rev_poi_map ={}


def read_sequence_file (path ,train_ratio ):
    if not os .path .exists (path ):
        raise IOError ("Sequence file not found: {0}".format (path ))
    data =SequenceData ()
    raw_rows =[]
    all_pois =set ()

    with open (path ,"r",newline ="")as handle :
        reader =csv .DictReader (handle ,delimiter =" ")
        required ={"u_id","u_pois","u_coordinates"}
        missing =required .difference (reader .fieldnames or [])
        if missing :
            raise ValueError ("Missing required columns: {0}".format (sorted (missing )))

        for row in reader :
            user_id =row ["u_id"]
            pois =[poi for poi in row ["u_pois"].split ("/")if poi ]
            coords =[coord for coord in row ["u_coordinates"].split ("/")if coord ]
            if len (pois )<2 :
                continue 
            raw_rows .append ((user_id ,pois ,coords ))
            all_pois .update (pois )

    data .user_ids =[row [0 ]for row in raw_rows ]
    data .poi_ids =sorted (all_pois )
    data .user_map ={user_id :idx for idx ,user_id in enumerate (data .user_ids )}
    data .poi_map ={poi_id :idx for idx ,poi_id in enumerate (data .poi_ids )}
    data .rev_user_map ={idx :user_id for user_id ,idx in data .user_map .items ()}
    data .rev_poi_map ={idx :poi_id for poi_id ,idx in data .poi_map .items ()}

    for user_id ,pois ,coords in raw_rows :
        mapped =[data .poi_map [poi ]for poi in pois ]
        split =int (len (mapped )*train_ratio )
        split =min (max (split ,1 ),len (mapped )-1 )
        data .train .append (mapped [:split ])
        data .test .append (mapped [split :])
        for poi_id ,coord in zip (pois ,coords ):
            poi_idx =data .poi_map [poi_id ]
            if poi_idx not in data .poi_coords :
                try :
                    lat ,lon =coord .split (",",1 )
                    data .poi_coords [poi_idx ]=(float (lat ),float (lon ))
                except ValueError :
                    data .poi_coords [poi_idx ]=(0.0 ,0.0 )
    return data 


def read_social_edges (path ,user_map ):
    edges =defaultdict (set )
    if not path :
        return edges 
    if not os .path .exists (path ):
        raise IOError ("Social edge file not found: {0}".format (path ))
    with open (path ,"r",newline ="")as handle :
        reader =csv .reader (handle )
        for row in reader :
            if len (row )<2 :
                continue 
            if row [0 ]in user_map and row [1 ]in user_map :
                a ,b =user_map [row [0 ]],user_map [row [1 ]]
                edges [a ].add (b )
                edges [b ].add (a )
    return edges 


def read_user_attribute_features (path ,user_map ):
    """Reads optional user attributes and returns a dense feature matrix.

    The file is expected to be a CSV with a user id column named ``u_id`` or
    ``user_id`` followed by numeric or categorical attributes. Categorical
    values are ordinally encoded per column and all columns are standardized.
    """
    if not path :
        return None 
    if not os .path .exists (path ):
        raise IOError ("User attribute file not found: {0}".format (path ))

    with open (path ,"r",newline ="")as handle :
        reader =csv .DictReader (handle )
        if not reader .fieldnames :
            raise ValueError ("User attribute file must have a header row.")
        user_col ="u_id"if "u_id"in reader .fieldnames else "user_id"
        if user_col not in reader .fieldnames :
            raise ValueError ("User attribute file must contain u_id or user_id.")
        attr_cols =[col for col in reader .fieldnames if col !=user_col ]
        rows =[row for row in reader if row .get (user_col )in user_map ]

    if not attr_cols :
        return None 

    category_maps ={col :{}for col in attr_cols }
    raw_by_user ={}
    for row in rows :
        values =[]
        for col in attr_cols :
            value =row .get (col ,"")
            try :
                values .append (float (value ))
            except ValueError :
                mapping =category_maps [col ]
                if value not in mapping :
                    mapping [value ]=len (mapping )
                values .append (float (mapping [value ]))
        raw_by_user [user_map [row [user_col ]]]=values 

    features =[[0.0 for _ in attr_cols ]for _ in user_map ]
    for user_idx ,values in raw_by_user .items ():
        features [user_idx ]=values 

    for col_idx in range (len (attr_cols )):
        column =[row [col_idx ]for row in features ]
        mean =sum (column )/len (column )
        variance =sum ((value -mean )**2 for value in column )/len (column )
        std =math .sqrt (variance )if variance >0 else 1.0 
        for row in features :
            row [col_idx ]=(row [col_idx ]-mean )/std 
    return features 


def build_interactions (train_sequences ):
    interactions =[]
    for user_idx ,seq in enumerate (train_sequences ):
        for pos in range (1 ,len (seq )):
            interactions .append ((user_idx ,seq [:pos ],seq [pos ]))
    return interactions 


def topk_jaccard_neighbors (sets_by_node ,top_k ):
    nodes =list (range (len (sets_by_node )))
    inverted =defaultdict (set )
    for node ,values in enumerate (sets_by_node ):
        for value in values :
            inverted [value ].add (node )

    neighbors =[[]for _ in nodes ]
    for node in nodes :
        candidates =set ()
        for value in sets_by_node [node ]:
            candidates .update (inverted [value ])
        candidates .discard (node )
        scored =[]
        base =sets_by_node [node ]
        for other in candidates :
            denom =len (base |sets_by_node [other ])
            score =0.0 if denom ==0 else float (len (base &sets_by_node [other ]))/denom 
            if score >0 :
                scored .append ((score ,other ))
        scored .sort (key =lambda item :(-item [0 ],item [1 ]))
        neighbors [node ]=[other for _ ,other in scored [:top_k ]]
    return neighbors 


def build_graphs (data ,neighbor_k ,social_edges ):
    user_sets =[set (seq )for seq in data .train ]
    poi_user_sets =[set ()for _ in data .poi_ids ]
    for user_idx ,seq in enumerate (data .train ):
        for poi_idx in seq :
            poi_user_sets [poi_idx ].add (user_idx )

    similar_users =topk_jaccard_neighbors (user_sets ,neighbor_k )
    similar_pois =topk_jaccard_neighbors (poi_user_sets ,neighbor_k )

    social =[[]for _ in data .user_ids ]
    for user_idx in range (len (data .user_ids )):
        if user_idx in social_edges and social_edges [user_idx ]:
            social [user_idx ]=list (sorted (social_edges [user_idx ]))[:neighbor_k ]
        else :
            social [user_idx ]=similar_users [user_idx ]
    return similar_users ,social ,similar_pois 


def trustees_for_users (num_users ,num_trustees ):
    trustees =[[]for _ in range (num_trustees )]
    for user_idx in range (num_users ):
        trustees [user_idx %num_trustees ].append (user_idx )
    return trustees 


class GraphAttentionAggregator (object ):
    def __init__ (self ,nn ,F ,num_nodes ,dim ,layers ):
        self .nn =nn 
        self .F =F 
        self .module =nn .ModuleList (
        [
        nn .ModuleDict (
        {
        "linear":nn .Linear (dim ,dim ,bias =False ),
        "attn":nn .Linear (dim *2 ,1 ,bias =False ),
        }
        )
        for _ in range (layers )
        ]
        )

    def __call__ (self ,node_ids ,embedding ,neighbors ,device ):
        h_all =embedding .weight 
        output =[]
        for node_id in node_ids .tolist ():
            h =h_all [node_id ]
            neigh =neighbors [node_id ]if node_id <len (neighbors )else []
            if not neigh :
                output .append (h )
                continue 
            neigh_tensor =h_all [torch .tensor (neigh ,dtype =torch .long ,device =device )]
            node_rep =h 
            for layer in self .module :
                q =layer ["linear"](node_rep ).unsqueeze (0 ).expand_as (neigh_tensor )
                k =layer ["linear"](neigh_tensor )
                logits =layer ["attn"](torch .cat ([q ,k ],dim =-1 )).squeeze (-1 )
                alpha =self .F .softmax (logits ,dim =0 ).unsqueeze (-1 )
                node_rep =self .F .elu (torch .sum (alpha *k ,dim =0 ))
            output .append (node_rep )
        return torch .stack (output ,dim =0 )


class FedMFFModel (object ):
    def __init__ (
    self ,
    nn ,
    F ,
    num_users ,
    num_items ,
    dim ,
    gat_layers ,
    similar_users ,
    social_users ,
    similar_pois ,
    user_attr_features ,
    device ,
    ):
        torch ,_ ,_ =require_torch ()
        attr_tensor =(
        torch .tensor (user_attr_features ,dtype =torch .float32 )
        if user_attr_features is not None 
        else None 
        )

        class _Model (nn .Module ):
            def __init__ (self ):
                super (_Model ,self ).__init__ ()
                self .user_embedding =nn .Embedding (num_users ,dim )
                self .item_embedding =nn .Embedding (num_items ,dim )
                if attr_tensor is None :
                    self .user_attr_embedding =nn .Embedding (num_users ,dim )
                    self .user_attr_projection =None 
                else :
                    self .register_buffer ("user_attr_features",attr_tensor )
                    self .user_attr_projection =nn .Linear (attr_tensor .size (1 ),dim )
                    self .user_attr_embedding =None 
                self .temporal_gru =nn .GRU (dim ,dim ,batch_first =True )
                self .temporal_attn =nn .Linear (dim ,1 )
                self .fusion_attn =nn .Linear (dim ,1 ,bias =False )
                self .output =nn .Linear (dim *2 ,1 )
                self .user_gat =GraphAttentionAggregator (
                nn ,F ,num_users ,dim ,gat_layers 
                ).module 
                self .social_gat =GraphAttentionAggregator (
                nn ,F ,num_users ,dim ,gat_layers 
                ).module 
                self .poi_gat =GraphAttentionAggregator (
                nn ,F ,num_items ,dim ,gat_layers 
                ).module 

            def reset_parameters (self ):
                nn .init .xavier_uniform_ (self .user_embedding .weight )
                nn .init .xavier_uniform_ (self .item_embedding .weight )
                if self .user_attr_embedding is not None :
                    nn .init .xavier_uniform_ (self .user_attr_embedding .weight )
                if self .user_attr_projection is not None :
                    nn .init .xavier_uniform_ (self .user_attr_projection .weight )
                    nn .init .zeros_ (self .user_attr_projection .bias )

        self .nn =nn 
        self .F =F 
        self .device =device 
        self .similar_users =similar_users 
        self .social_users =social_users 
        self .similar_pois =similar_pois 
        self .model =_Model ().to (device )
        self .model .reset_parameters ()

    def _aggregate (self ,node_ids ,embedding ,neighbors ,layers ):
        h_all =embedding .weight 
        reps =[]
        for node_id in node_ids .tolist ():
            h =h_all [node_id ]
            neigh =neighbors [node_id ]if node_id <len (neighbors )else []
            if not neigh :
                reps .append (h )
                continue 
            neigh_ids =torch .tensor (neigh ,dtype =torch .long ,device =self .device )
            neigh_h =h_all [neigh_ids ]
            node_rep =h 
            for layer in layers :
                q =layer ["linear"](node_rep ).unsqueeze (0 ).expand_as (neigh_h )
                k =layer ["linear"](neigh_h )
                logits =layer ["attn"](torch .cat ([q ,k ],dim =-1 )).squeeze (-1 )
                alpha =self .F .softmax (logits ,dim =0 ).unsqueeze (-1 )
                node_rep =self .F .elu (torch .sum (alpha *k ,dim =0 ))
            reps .append (node_rep )
        return torch .stack (reps ,dim =0 )

    def encode_history (self ,histories ):
        torch ,_ ,_ =require_torch ()
        lengths =[max (1 ,len (history ))for history in histories ]
        max_len =max (lengths )
        padded =[]
        for history in histories :
            if history :
                seq =history [-max_len :]
            else :
                seq =[0 ]
            seq =seq +[seq [-1 ]]*(max_len -len (seq ))
            padded .append (seq )
        hist =torch .tensor (padded ,dtype =torch .long ,device =self .device )
        emb =self .model .item_embedding (hist )
        output ,_ =self .model .temporal_gru (emb )
        logits =self .model .temporal_attn (output ).squeeze (-1 )
        mask =torch .arange (max_len ,device =self .device ).unsqueeze (0 )<torch .tensor (
        lengths ,device =self .device 
        ).unsqueeze (1 )
        logits =logits .masked_fill (~mask ,-1e9 )
        alpha =self .F .softmax (logits ,dim =1 ).unsqueeze (-1 )
        return torch .sum (alpha *output ,dim =1 )

    def score (self ,user_ids ,histories ,item_ids ):
        torch ,_ ,_ =require_torch ()
        user_ids =torch .tensor (user_ids ,dtype =torch .long ,device =self .device )
        item_ids =torch .tensor (item_ids ,dtype =torch .long ,device =self .device )

        sim_user =self ._aggregate (
        user_ids ,
        self .model .user_embedding ,
        self .similar_users ,
        self .model .user_gat ,
        )
        social =self ._aggregate (
        user_ids ,
        self .model .user_embedding ,
        self .social_users ,
        self .model .social_gat ,
        )
        if self .model .user_attr_embedding is not None :
            attr =self .model .user_attr_embedding (user_ids )
        else :
            attr =self .model .user_attr_projection (self .model .user_attr_features [user_ids ])
        behavior =self .encode_history (histories )
        poi_context =self ._aggregate (
        item_ids ,
        self .model .item_embedding ,
        self .similar_pois ,
        self .model .poi_gat ,
        )

        factors =torch .stack ([sim_user ,social ,attr ,behavior ,poi_context ],dim =1 )
        weights =self .F .softmax (self .model .fusion_attn (factors ).squeeze (-1 ),dim =1 )
        fused_user =torch .sum (weights .unsqueeze (-1 )*factors ,dim =1 )
        item =self .model .item_embedding (item_ids )+poi_context 
        return self .model .output (torch .cat ([fused_user ,item ],dim =-1 )).squeeze (-1 )


def negative_sample (num_items ,seen ):
    item =random .randrange (num_items )
    while item in seen :
        item =random .randrange (num_items )
    return item 


def train_local (global_state ,template ,interactions ,train_sequences ,args ,torch ):
    model =copy .deepcopy (template )
    model .model .load_state_dict (global_state )
    model .model .train ()
    optimizer =torch .optim .Adam (model .model .parameters (),lr =args .lr )
    user_seen =[set (seq )for seq in train_sequences ]

    for _ in range (args .local_epochs ):
        random .shuffle (interactions )
        for start in range (0 ,len (interactions ),args .batch_size ):
            batch =interactions [start :start +args .batch_size ]
            users =[row [0 ]for row in batch ]
            histories =[row [1 ]for row in batch ]
            pos =[row [2 ]for row in batch ]
            neg =[negative_sample (args .num_items ,user_seen [u ])for u in users ]
            pos_scores =model .score (users ,histories ,pos )
            neg_scores =model .score (users ,histories ,neg )
            loss =-torch .log (torch .sigmoid (pos_scores -neg_scores )+1e-12 ).mean ()
            l2 =sum (param .pow (2 ).sum ()for param in model .model .parameters ())
            loss =loss +args .reg *l2 
            optimizer .zero_grad ()
            loss .backward ()
            optimizer .step ()
    return model .model .state_dict (),len (interactions )


def fedavg (states ,weights ):
    avg =copy .deepcopy (states [0 ])
    total =float (sum (weights ))
    for key in avg :
        avg [key ]=avg [key ]*(weights [0 ]/total )
        for idx in range (1 ,len (states )):
            avg [key ]=avg [key ]+states [idx ][key ]*(weights [idx ]/total )
    return avg 


def evaluate (model ,data ,topks ,output_csv ):
    torch ,_ ,_ =require_torch ()
    model .model .eval ()
    rows =[]
    all_items =list (range (len (data .poi_ids )))
    with torch .no_grad ():
        for user_idx ,(train_seq ,test_seq )in enumerate (zip (data .train ,data .test )):
            seen =set (train_seq )
            candidates =[item for item in all_items if item not in seen ]
            scores =[]
            history =train_seq 
            for start in range (0 ,len (candidates ),1024 ):
                items =candidates [start :start +1024 ]
                users =[user_idx ]*len (items )
                histories =[history ]*len (items )
                batch_scores =model .score (users ,histories ,items ).detach ().cpu ().tolist ()
                scores .extend (zip (items ,batch_scores ))
            scores .sort (key =lambda item :item [1 ],reverse =True )
            ranked =[item for item ,_ in scores [:max (topks )]]
            test_set =set (test_seq )
            row ={
            "user_id":data .rev_user_map [user_idx ],
            "num_test":len (test_seq ),
            }
            for k in topks :
                recs =ranked [:k ]
                hits =[1 if item in test_set else 0 for item in recs ]
                hit_sum =float (sum (hits ))
                precision =hit_sum /k 
                recall =hit_sum /len (test_set )if test_set else 0.0 
                ap =average_precision (hits ,len (test_set ))
                ndcg =ndcg_at_k (hits ,len (test_set ))
                row ["precision@{0}".format (k )]=precision 
                row ["recall@{0}".format (k )]=recall 
                row ["ap@{0}".format (k )]=ap 
                row ["ndcg@{0}".format (k )]=ndcg 
            rows .append (row )

    out_dir =os .path .dirname (output_csv )
    if out_dir and not os .path .exists (out_dir ):
        os .makedirs (out_dir )
    with open (output_csv ,"w",newline ="")as handle :
        fieldnames =["user_id","num_test"]
        for k in topks :
            fieldnames +=[
            "precision@{0}".format (k ),
            "recall@{0}".format (k ),
            "ap@{0}".format (k ),
            "ndcg@{0}".format (k ),
            ]
        writer =csv .DictWriter (handle ,fieldnames =fieldnames )
        writer .writeheader ()
        writer .writerows (rows )

    summary ={}
    for k in topks :
        summary ["precision@{0}".format (k )]=sum (
        row ["precision@{0}".format (k )]for row in rows 
        )/len (rows )
        summary ["recall@{0}".format (k )]=sum (
        row ["recall@{0}".format (k )]for row in rows 
        )/len (rows )
        summary ["map@{0}".format (k )]=sum (row ["ap@{0}".format (k )]for row in rows )/len (
        rows 
        )
        summary ["ndcg@{0}".format (k )]=sum (
        row ["ndcg@{0}".format (k )]for row in rows 
        )/len (rows )
    return summary 


def average_precision (hits ,num_relevant ):
    if num_relevant <=0 :
        return 0.0 
    total =0.0 
    hit_count =0.0 
    for idx ,hit in enumerate (hits ,start =1 ):
        if hit :
            hit_count +=1.0 
            total +=hit_count /idx 
    return total /num_relevant 


def ndcg_at_k (hits ,num_relevant ):
    if num_relevant <=0 :
        return 0.0 
    dcg =0.0 
    for idx ,hit in enumerate (hits ):
        if hit :
            dcg +=1.0 /math .log (idx +2 ,2 )
    ideal_len =min (len (hits ),num_relevant )
    ideal =sum (1.0 /math .log (idx +2 ,2 )for idx in range (ideal_len ))
    return dcg /ideal if ideal >0 else 0.0 


def main ():
    global torch 
    args =parse_args ()
    torch ,nn ,F =require_torch ()
    set_seed (args .seed )
    device =torch .device (args .device )
    topks =[int (value .strip ())for value in args .topk .split (",")if value .strip ()]

    data =read_sequence_file (args .file ,args .train_ratio )
    args .num_items =len (data .poi_ids )
    social_edges =read_social_edges (args .social_file ,data .user_map )
    user_attr_features =read_user_attribute_features (args .user_attr_file ,data .user_map )
    similar_users ,social_users ,similar_pois =build_graphs (
    data ,args .neighbor_k ,social_edges 
    )
    trustees =trustees_for_users (len (data .user_ids ),args .num_trustees )
    all_interactions =build_interactions (data .train )
    interactions_by_user =defaultdict (list )
    for row in all_interactions :
        interactions_by_user [row [0 ]].append (row )

    template =FedMFFModel (
    nn ,
    F ,
    len (data .user_ids ),
    len (data .poi_ids ),
    args .embedding_dim ,
    args .gat_layers ,
    similar_users ,
    social_users ,
    similar_pois ,
    user_attr_features ,
    device ,
    )
    global_state =copy .deepcopy (template .model .state_dict ())

    print ("Users={0}, POIs={1}, interactions={2}".format (
    len (data .user_ids ),len (data .poi_ids ),len (all_interactions )
    ))
    print ("Trustees={0}, rounds={1}, local_epochs={2}".format (
    args .num_trustees ,args .rounds ,args .local_epochs 
    ))

    for round_idx in range (args .rounds ):
        local_states ,local_weights =[],[]
        for trustee_users in trustees :
            trustee_interactions =[]
            for user_idx in trustee_users :
                trustee_interactions .extend (interactions_by_user [user_idx ])
            if not trustee_interactions :
                continue 
            state ,weight =train_local (
            global_state ,template ,trustee_interactions ,data .train ,args ,torch 
            )
            local_states .append (state )
            local_weights .append (weight )
        global_state =fedavg (local_states ,local_weights )
        template .model .load_state_dict (global_state )
        print ("Round {0}/{1} complete".format (round_idx +1 ,args .rounds ))

    summary =evaluate (template ,data ,topks ,args .per_user_output )
    print ("Summary metrics")
    for key in sorted (summary ):
        print ("{0}: {1:.6f}".format (key ,summary [key ]))
    print ("Per-user metrics saved to: {0}".format (args .per_user_output ))


if __name__ =="__main__":
    main ()
