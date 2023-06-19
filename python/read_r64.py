import numpy as np
import h5py

# TODO: Probably combine with method from this file https://github.com/Ok2610/FlexR64/blob/main/flex-decompress-r64.py

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


def read_item_features(topFile:str, idsFile:str, ratiosFile:str):
    items = []
    with h5py.File(topFile, 'r') as init:
        with h5py.File(idsFile, 'r') as ids:
            with h5py.File(ratiosFile, 'r') as ratios:
                items = [decompress(init['data'][i], ids['data'][i], ratios['data'][i]) for i in range(len(init['data']))]
    return items


def read_single_item_features(idx:int, topFile:str, idsFile:str, ratiosFile:str):
    with h5py.File(topFile, 'r') as init:
        f_init = init['data'][idx]
        with h5py.File(idsFile, 'r') as ids:
            f_ids = ids['data'][idx]
            with h5py.File(ratiosFile, 'r') as ratios:
                f_ratios = ratios['data'][idx]
                return decompress(f_init, f_ids, f_ratios)


def read_total_items_count(initFile:str):
    size = 0
    with h5py.File(initFile, 'r') as init:
        size = len(init['data'])
    return size