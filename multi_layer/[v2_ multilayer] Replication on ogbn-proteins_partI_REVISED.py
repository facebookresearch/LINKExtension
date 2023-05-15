#!/usr/bin/env -S grimaldi --kernel bento_kernel_pytorch
# fmt: off

""":py"""
# 4/9/23
# KM Altenburger

# steps for downloading repo internally; run w/ "pytorch kernel"
# sudo feature install ttls_fwdproxy
# mgt import --src-type pypi sdist
# mgt import --src-type pypi nvidia-cublas-cu11==11.10.3.66
# mgt import --src-type pypi ogb

import sys
sys.path.insert(0, "/home/kaltenburger/fbsource/ogb")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import ogb
import pandas as pd
import torch
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("iteration",  type=int)
args = parser.parse_args()

i = args.iteration
print(i)

# import torch_geometric
from ogb.nodeproppred import NodePropPredDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split

""":md
### read & set-up dataset

"""

""":py"""
dataset = NodePropPredDataset(name='ogbn-proteins')
g = dataset.graph # get edgelist and graph metadata
y = dataset.labels # outcomes
node_species_df = pd.DataFrame({'node_id':range(len(g['node_species'].flatten())), 'node_species': g['node_species'].flatten()})
node_species_df.head()

""":py"""
# create edgelist
edge_list_df = pd.DataFrame(g["edge_index"].T)
edge_list_df.columns = ['src','dst']
edge_list_df.head()

""":py"""
# sanity check on inter-species connections
edge_list_df_check = edge_list_df.merge(node_species_df, how = 'left', left_on = 'src', right_on = 'node_id')
edge_list_df_check.drop('node_id', axis = 1, inplace=True)
edge_list_df_check = edge_list_df_check.merge(node_species_df, how = 'left', left_on = 'dst', right_on = 'node_id')
edge_list_df_check.drop('node_id', axis = 1, inplace=True)
#edge_list_df_check.head()
print(np.mean(edge_list_df_check['node_species_x']==edge_list_df_check['node_species_y']))
print(np.mean(edge_list_df_check['node_species_x']!=edge_list_df_check['node_species_y']))

""":py"""
y0 = pd.DataFrame(y)
y0 = y0.iloc[:,0:25] # part I
#y0 = y0.iloc[:,25:50] # part II
#y0 = y0.iloc[:,50:112] # part II
y0['node_id'] = range(len(y0[0])) # change here for each part

split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

X_train = y0['node_id'].iloc[train_idx]
y_train = y0.iloc[train_idx]

X_valid = y0['node_id'].iloc[valid_idx]
X_test = y0['node_id'].iloc[test_idx]
y_valid = y0.iloc[valid_idx]
y_test = y0.iloc[test_idx]



""":py"""
edge_w = pd.DataFrame(g['edge_feat'])
edge_w = edge_w.add_suffix('_weight')
edge_list_df = pd.concat([edge_list_df, edge_w], axis=1)
#edge_list_df.head()
# w0 = edge_w[j]
# edge_list_df['weight'] = w0
# edge_list_df.head()
# # merge in labels for training nodes on a user's connections
relevant_edges_tmp = edge_list_df.merge(y_train, left_on = 'dst', right_on='node_id', how = 'left')
relevant_edges_tmp.head()

""":py"""
intervals_v1 = relevant_edges_tmp[['0_weight','1_weight', '2_weight', '3_weight','6_weight','4_weight', '5_weight', '7_weight']].quantile([0.75])

""":py"""
np.array(intervals_v1.iloc[0])

""":py"""


""":py"""


""":py"""
from sklearn.preprocessing import Binarizer
k=0
for j in range(8):
    print(j)
    transformer = Binarizer().fit(relevant_edges_tmp[[str(j)+'_weight']])
    relevant_edges_tmp['bin_'+str(k)] = pd.DataFrame(transformer.transform((relevant_edges_tmp[[str(j)+'_weight']]<=np.array(intervals_v1.iloc[0])[j])+0).flatten())
    print(relevant_edges_tmp.head())
    k=k+1
    relevant_edges_tmp['bin_'+str(k)] = pd.DataFrame(transformer.transform((relevant_edges_tmp[[str(j)+'_weight']]>np.array(intervals_v1.iloc[0])[j])+0).flatten())
    k=k+1

""":py"""
print(relevant_edges_tmp.columns)

""":py"""


""":py"""
print('test')

""":py"""
tmp = pd.DataFrame(relevant_edges_tmp[['bin_0',    'bin_1',    'bin_2',    'bin_3',    'bin_4',    'bin_5',
        'bin_6',    'bin_7',    'bin_8',    'bin_9',   'bin_10',   'bin_11',
        'bin_12',   'bin_13', 'bin_14',   'bin_15' ]].drop_duplicates())

""":py"""
tmp.reset_index(inplace=True,drop=True)
tmp.head()

""":py"""
tmp['edge_bin'] = 0
tmp['edge_bin'] = pd.DataFrame(range(np.shape(tmp)[0]))

""":py"""
relevant_edges_tmp['edge_bin']=0
for k in np.array(tmp.edge_bin):
    print(k)
    relevant_edges_tmp['edge_bin'][(relevant_edges_tmp.bin_0==tmp.bin_0[k]) & \
(relevant_edges_tmp.bin_1==tmp.bin_1[k]) & \
(relevant_edges_tmp.bin_2==tmp.bin_2[k]) & \
(relevant_edges_tmp.bin_3==tmp.bin_3[k]) & \
(relevant_edges_tmp.bin_4==tmp.bin_4[k]) & \
(relevant_edges_tmp.bin_5==tmp.bin_5[k]) & \
(relevant_edges_tmp.bin_6==tmp.bin_6[k]) & \
(relevant_edges_tmp.bin_7==tmp.bin_7[k]) & \
(relevant_edges_tmp.bin_8==tmp.bin_8[k]) & \
(relevant_edges_tmp.bin_9==tmp.bin_9[k]) & \
(relevant_edges_tmp.bin_10==tmp.bin_10[k]) & \
(relevant_edges_tmp.bin_11==tmp.bin_11[k]) & \
(relevant_edges_tmp.bin_12==tmp.bin_12[k]) & \
(relevant_edges_tmp.bin_13==tmp.bin_13[k]) & \
(relevant_edges_tmp.bin_14==tmp.bin_14[k]) & \
(relevant_edges_tmp.bin_15==tmp.bin_15[k]) ] = tmp.edge_bin[k]

""":py"""
print('test')

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# set-up loop
#rocauc_v1_ud_w = []
rocauc_v2_ud_w_regularized = []
rocauc_v2_ud_w = []
#rocauc_ud_uw = []
#for i in range(0,25): # loop through outcomes part 1
#for i in range(25,50):
#for i in range(50,112):


print(i)
relevant_edges = relevant_edges_tmp[["src", "dst", "node_id", i, 'edge_bin']]
relevant_edges.columns = ["src", "dst",  "node_id", "y", 'edge_bin']
label1 = relevant_edges.groupby(["src", 'edge_bin'])["y"].sum().reset_index() # same by edgebin
labeltotal = relevant_edges.groupby(["src", 'edge_bin'])["y"].count().reset_index()
relevant_edgelist = labeltotal.merge(label1, on=["src", 'edge_bin'])
## this is already by edge bin
relevant_edgelist["label_0"] = relevant_edgelist["y_x"] - relevant_edgelist["y_y"]
relevant_edgelist.rename(columns={"y_y": "label_1"}, inplace=True)
relevant_edgelist.drop(["y_x"], axis=1, inplace=True)

## v0 - linear combination of single layers
normalizer = pd.DataFrame(y_train[i].value_counts(normalize=True, ascending=True))

# step 1 - one hop scores
step1_one_hop_scores = pd.DataFrame(
    {
        "src": relevant_edgelist.src,
        "edge_bin": relevant_edgelist.edge_bin,
        "frac": np.log(
            (relevant_edgelist["label_1"] + 1) / (relevant_edgelist["label_0"] + 1)
        )
        * list((normalizer.iloc[0]) / (normalizer.iloc[1]))[0],
    }
)
two_hop_stg_v1_m = relevant_edges.loc[relevant_edges.edge_bin==0].merge(
        step1_one_hop_scores, left_on=["dst",], right_on=["src",], how="inner" # keep edge_bin and one-hop scores same
    )
two_hop_stg_v1_m.rename(columns={'frac':'frac'+str(k)}, inplace=True)

two_hop_stg_v1_m = two_hop_stg_v1_m.groupby(['src_x','edge_bin_x','edge_bin_y'])['frac'+str(k)].sum().reset_index()

for m in np.unique(relevant_edges.edge_bin):
    #print(m)
    two_hop_stg_v1_m = relevant_edges.loc[relevant_edges.edge_bin==m].merge(
        step1_one_hop_scores, left_on=["dst",], right_on=["src",], how="inner" # keep edge_bin and one-hop scores same
    )
    two_hop_stg_v1_m.rename(columns={'frac':'frac'+str(m)}, inplace=True)
    two_hop_stg_v1_m = two_hop_stg_v1_m.groupby(['src_x','edge_bin_x','edge_bin_y'])['frac'+str(m)].sum().reset_index()
    if m == 0:
        t0 = pd.pivot_table(two_hop_stg_v1_m, values = 'frac'+str(m), columns = 'edge_bin_y', index = 'src_x').reset_index().fillna(0)
        t0 = t0.add_prefix(str(m)+'_')
        t0.rename(columns={str(m)+'_src_x':'src_x'}, inplace=True)
    if m>0:
        t1 = pd.pivot_table(two_hop_stg_v1_m, values = 'frac'+str(m), columns = 'edge_bin_y', index = 'src_x').reset_index().fillna(0)
        t1 = t1.add_prefix(str(m)+'_')
        t1.rename(columns={str(m)+'_src_x':'src_x'}, inplace=True)
        t0 = t0.merge(t1, how = 'outer', on = 'src_x') ## confirm how to do merge


    two_hop_v1_valid = t0.merge(
            y_valid[['node_id',i]], left_on="src_x", right_on="node_id", how="right"
        )
#regularized model

scaler = StandardScaler()

clf = LogisticRegression(penalty="l2", solver="lbfgs", C=10e20).fit(
    scaler
    .fit_transform(two_hop_v1_valid.drop(["src_x", "node_id", i], axis=1).fillna(0)),
    two_hop_v1_valid[i],
)

pred_v0 = y_test[['node_id',i]].merge(t0, right_on="src_x", left_on="node_id", how="left")
print(
    "roc-auc, v2*: ",
    roc_auc_score(
        pred_v0[i],
        # clf.predict_proba(StandardScaler.fit(pred_v0.drop(['node_id','src_x',i],axis=1).fillna(0))[:,1]
        clf.predict_proba(scaler.transform(pred_v0.drop(["src_x", "node_id", i], axis=1).fillna(0)))[
            :, 1
        ],
    ),
)
rocauc_v2_ud_w_regularized.append(roc_auc_score(
    pred_v0[i],
    # clf.predict_proba(StandardScaler.fit(pred_v0.drop(['node_id','src_x',i],axis=1).fillna(0))[:,1]
    clf.predict_proba(scaler.transform(pred_v0.drop(["src_x", "node_id", i], axis=1).fillna(0)))[
        :, 1
    ],
))


""":py"""
with open('REVISED_rocauc_v2_ud_w_part1_multilayer_iteration_'+str(i)+'.npy', 'wb') as f:
#with open('rocauc_v2_ud_w_part2_multilayer.npy', 'wb') as f:
#with open('rocauc_v2_ud_w_part3_multilayer.npy', 'wb') as f:
    np.save(f, rocauc_v2_ud_w_regularized)
