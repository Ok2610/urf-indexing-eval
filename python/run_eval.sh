#!/bin/bash

INDEX=$1 # 0: HNSW, 1: Annoy, 2: IVF
INDEX_F=$2 # Index file
RES_DIR=$3 # Result directory
EXP_NAME=$4 # Name of experiment
METRIC=0 # 0: IP, 1: Euclidean (Not implemented)
DIM=1000 # Dimensions
AP=64 # Approximation Parameter
EXTRA=$5

# python3 eval_ann.py $INDEX \
# $INDEX_F $RES_DIR $EXP_NAME \
# $METRIC $DIM $AP --seen --h5_dir ../cpp/data/lsc \
# --actors_f ../cpp/data/lsc/lsc_actors.json > outputlsc_ivf64.txt