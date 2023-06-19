import argparse
import numpy as np
import json
import h5py

from pathlib import Path

DECOMP_MASK = [pow(2, 10) - 1, pow(2, 14) - 1, pow(2, 16) - 1, pow(2, 20) - 1, pow(2, 30) - 1]
MULTIPLIER = [np.uint64(1000), np.uint64(10000),
              np.uint64(1000000), np.uint64(1000000000)]

MAX_N_INT = np.uint64(15)

FEAT_RANGE = 1000

BIT_SHIFT_RATIO = [np.uint64(4), np.uint64(14), np.uint64(24),
                   np.uint64(34), np.uint64(44), np.uint64(54)]

FEAT_PER_INT_RATIO = 6
MULTIPLIER_RATIO = np.uint64(1000)
PRECISION_RATIO = 3

BIT_SHIFT_INIT_RATIO = np.uint64(54)
MULTIPLIER_INIT = np.uint64(pow(10, 16))
MASK_INIT = np.uint64(18014398509481983)


def decompress(compInit:int, compIds:int, compFeat:int):
    decomp = []
    comp_init = np.uint64(compInit)
    comp_ids = np.uint64(compIds)
    comp_feat = np.uint64(compFeat)
    feat_init_id = comp_init >> BIT_SHIFT_INIT_RATIO
    feat_init_score = (comp_init & MASK_INIT) / float(MULTIPLIER_INIT)
    feat_score = feat_init_score
    decomp.append((feat_init_id,feat_init_score))
    for f_pos in range(FEAT_PER_INT_RATIO):
        feat_i = (comp_ids >> BIT_SHIFT_RATIO[f_pos]) & np.uint64(DECOMP_MASK[0])
        feat_score *=\
            ((comp_feat >> BIT_SHIFT_RATIO[f_pos]) & np.uint64(DECOMP_MASK[0])) /\
            float(MULTIPLIER_RATIO)
        decomp.append((feat_i, feat_score))
    return decomp


def read_item_features(initFile:str, idsFile:str, ratiosFile:str):
    items = []
    with h5py.File(initFile, 'r') as init:
        with h5py.File(idsFile, 'r') as ids:
            with h5py.File(ratiosFile, 'r') as ratios:
                items = [decompress(init['data'][i], ids['data'][i], ratios['data'][i]) for i in range(len(init['data']))]
    return items


def distance(hplane:list, itemFeats:list[tuple[int,float]]):
    score = 0.0
    for i,v in itemFeats:
        score += hplane[i] * v
    return score


parser = argparse.ArgumentParser("Generate ranked list of dataset using hyperplane queries.")
parser.add_argument("hyperplanes_f", type=Path)
parser.add_argument("dataset_f", nargs=3, type=Path, help="HDF5 files for dataset")
parser.add_argument("output_dir", type=Path)
parser.add_argument("--rounds", type=int, default=50)
parser.add_argument("--limit", type=int, default=1000)
parser.add_argument("--start_task", type=int, default=0)

args = parser.parse_args()

with args.hyperplanes_f.open('r') as f:
    hplanes = json.load(f)

items = read_item_features(str(args.dataset_f[0]), str(args.dataset_f[1]), str(args.dataset_f[2]))
print(items[4001])
print(distance(hplanes['svms'][0], items[4001]))
print(items[26071])
print(distance(hplanes['svms'][0], items[26071]))

tasks = int(len(hplanes['svms']) / args.rounds)
start = args.start_task

for t in range(start,tasks):
    taskRankings = []
    for svm in hplanes['svms'][(t*args.rounds) : ((t+1)*args.rounds)]:
        rankedItems = [(i,distance(svm,items[i])) for i in range(len(items))]
        rankedItems.sort(key=lambda x : x[1], reverse=True)
        taskRankings.append(rankedItems[:args.limit])
    fname = "hplane_grountruth_task_%d.json" % t
    with (args.output_dir / fname).open('w') as f:
        json.dump(taskRankings,f)