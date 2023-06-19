import os
import argparse
import json
import read_r64 as r64
from pathlib import Path
from time import time, strftime
from random import seed, sample, randint

parser = argparse.ArgumentParser()
parser.add_argument('index', type=int, help="HNSW=0, ANNOY=1, IVF=2")
parser.add_argument('index_path', type=Path, help="Path to the index file")
parser.add_argument('result_dir', type=Path, help="Directory for result file")
parser.add_argument('result_file', type=str, help="Result file name")
parser.add_argument('index_args', nargs='*', type=int, help="Index arguments [metric = {0: 'IP', 1: 'L2'}, dimensions, search space]")
parser.add_argument('--h5_dir', type=Path, default=Path('data/h5'), help="Directory containing R64 representation of data items")
parser.add_argument('--num_suggestions', type=int, default=25, help="Number of suggestions")
parser.add_argument('--index_k', type=int, default=1000, help="Number of items to retrieve from index (analysis purpose)" )
parser.add_argument('--num_pos', type=int, default=5, help="Number of positives to consider each round")
parser.add_argument('--num_neg', type=int, default=15, help="Number of negatives to consider each round")
parser.add_argument('--number_of_runs', type=int, default=1, help="Number of random start runs. Default=1")
parser.add_argument('--number_of_rounds', type=int, default=50, help="Number of rounds per run. Default=50")
parser.add_argument('--data_rep', type=int, default=0, help="float=0, R64=1 (This has not been introduced yet)")
parser.add_argument('--seen', action='store_true', help="Use dynamic seen")
parser.add_argument('--actors_f', type=Path, help="JSON file for the artificial actors")
parser.add_argument('--suggs_provided_f', type=Path, default=None, help="JSON file containing suggestions for each round")
parser.add_argument('--hplanes_provided_f', type=Path, default=None, help="JSON file from generated_hyperplanes.py")
parser.add_argument('--use_exq', action='store_true', help="Use exquisitor")
parser.add_argument('--rng_run', action='store_true', help="Pick random k items from the collection each round. Need to specify 1 index to initialize classifier.")

args = parser.parse_args()

### From https://github.com/Ok2610/InteractiveLearningEval/blob/main/VBS2020/scripts/eval_vbs.py

RNG_RUN = args.rng_run
USE_EXQ = args.use_exq
if USE_EXQ:
    import exq
else:
    import il

GLOBAL_POS = []
GLOBAL_NEG = []
GLOBAL_SUGGS = []
ALL_IDS = []

def ts():
    return "[" + str(strftime("%d %b, %H:%M:%S")) + "]"

# Euclidean distance between two vectors
def calculate_distance_cmb_vector(sugg_vector, dist_vector):
    dist = 0.0
    dist_dict = {}
    for key in dist_vector:
        dist_dict[int(key)] = dist_vector[key]

    for (idx,val) in sugg_vector:
        if idx in dist_dict:
            dist += (val - dist_dict[idx]) * (val - dist_dict[idx])
            dist_dict[idx] = 0.0
        else:
            dist += val * val

    for idx in dist_dict:
        dist += dist_dict[idx] * dist_dict[idx]

    return dist

# MIP between item vector and hyperplane
def calculate_distance_hplane_mip(sugg_vector, hplane):
    dist = 0.0

    for (idx,val) in sugg_vector:
        dist += hplane[idx] * val
    
    return -dist



def read_actors(actors_file):
    actors = {}
    with actors_file.open('r') as f:
        actors = json.load(f)
    return actors


if USE_EXQ:
    def initialize_exquisitor(searchExpansion:int, modInfoFiles:Path,
                              expansionType=0, statLevel=1, modWeights=[], ffs=False, guaranteedSlots=0,
                              collFilters=[]):
        mod_info = []
        if not modInfoFiles.is_file():
            print('Modality information file for Exquisitor not found!')
            raise ValueError
        with modInfoFiles.open('r') as f:
            mod_info = json.load(f)

        iota = 1
        noms = 1000
        num_workers = 1
        segments = 1
        indx_conf_files = [c['indx_path'] for c in mod_info]
        num_modalities = len(mod_info)
        mod_weights = []
        if len(modWeights) != 0:
            mod_weights = modWeights
        else:
            for m in range(num_modalities):
                mod_weights.append(1.0)
        b = searchExpansion
        mod_feature_dimensions = [c['total_feats'] for c in mod_info]
        func_type = 0
        func_objs = []
        for m,c in enumerate(mod_info):
            func_objs.append([
                c['n_feat_int']+1,
                c['bit_shift_t'],
                c['bit_shift_ir'],
                c['bit_shift_ir'],
                c['decomp_mask_t'],
                float(c['multiplier_t']),
                pow(2, c['decomp_mask_ir'])-1,
                pow(2, c['decomp_mask_ir'])-1,
                float(c['multiplier_ir']),
                mod_weights[m]
            ])
        item_metadata = collFilters
        video_metadata = []
        exq.initialize(iota, noms, num_workers, segments, num_modalities, b, indx_conf_files, mod_feature_dimensions,
                    func_type, func_objs, item_metadata, video_metadata, expansionType, statLevel, ffs, guaranteedSlots)



def classify_suggestions(sugg_list, cmb_vector, relevant, 
                         pos_policy, neg_policy, p, n, rd, 
                         comp_files):
    # Process the suggestions
    global GLOBAL_POS, GLOBAL_NEG, GLOBAL_SUGGS
    done = False
    pos = []
    neg = []
    distances = []
    for s in sugg_list:
        if s in relevant:
            done = True
        features = r64.read_single_item_features(s, comp_files[0], comp_files[1], comp_files[2])
        distances.append((calculate_distance_cmb_vector(features, cmb_vector),s))

    distances = sorted(distances)
    # print(distances)

    if pos_policy == 0: #Acc-add
        GLOBAL_POS += distances[:p]

    if neg_policy == 0: #Acc-add
        GLOBAL_NEG += distances[-n:]

    temp = sorted(GLOBAL_POS + GLOBAL_NEG + distances)
    if pos_policy == 1: #Acc-replace
        if rd == 1:
            GLOBAL_POS = distances[:p]
        else:
            GLOBAL_POS = [e for e in temp[:(p*rd)]]

    if neg_policy == 1: #Acc-replace
        if rd == 1:
            GLOBAL_NEG = distances[-n:]
        else:
            GLOBAL_NEG = [e for e in temp[-(n*rd):]]

    if pos_policy == 2: #Fixed
        for (d,e) in distances:
            if len(GLOBAL_POS) < p:
                GLOBAL_POS.append((d,e))
                GLOBAL_POS = sorted(GLOBAL_POS)
            elif d < GLOBAL_POS[-1][0]:
                GLOBAL_POS[-1] = (d,e)
                GLOBAL_POS = sorted(GLOBAL_POS)
            else:
                break

    if neg_policy == 2: #Fixed
        for (d,e) in distances[::-1]:
            if len(GLOBAL_NEG) < n:
                GLOBAL_NEG.append((d,e))
                GLOBAL_NEG = sorted(GLOBAL_NEG)
            elif d > GLOBAL_NEG[0][0]:
                GLOBAL_NEG[0] = (d,e)
                GLOBAL_NEG = sorted(GLOBAL_NEG)
            else:
                break

    if neg_policy == 3: #Rand-local
        if rd == 1:
            GLOBAL_NEG = distances[-n:]
        if pos_policy == 0:
            try:
                GLOBAL_NEG += sample(distances[p:],n)
            except:
                print('Failed to get negative samples. Round: %d Len(suggs): %d' % (rd,len(distances)))
        elif pos_policy == 1:
            try:
                GLOBAL_NEG += sample(temp[(p*rd):],n)
            except:
                print('Failed to get negative samples. Round: %d Len(temp): %d' % (rd,len(temp)))
        elif pos_policy == 2:
            try:
                GLOBAL_NEG += sample((set(distances)-set(GLOBAL_POS)),n)
            except:
                print('Failed to get negative samples. Round: %d Len(suggs): %d' % (rd,len(distances)))

    if neg_policy == 4: #Rand-global
        pos = [e[1] for e in GLOBAL_POS]
        GLOBAL_NEG += sample((ALL_IDS - set(pos) - set(GLOBAL_NEG)),n)
        neg = [e for e in GLOBAL_NEG]


    if neg_policy != 4:
        # print(p,n,GLOBAL_POS, GLOBAL_NEG)
        pos = [e[1] for e in GLOBAL_POS]
        neg = [e[1] for e in GLOBAL_NEG]

    if USE_EXQ:
        exq.reset_model(False,False)
    else:
        il.reset_model()

    return (pos,neg,done)


def sort_index_list_rng(li, hyperplane, comp_files):
    ret = []

    for i in li:
        features = r64.read_single_item_features(i, comp_files[0], comp_files[1], comp_files[2])
        dist = calculate_distance_hplane_mip(features, hyperplane)
        ret.append((dist,i))
    ret = [r[1] for r in sorted(ret)]

    return ret


def run_il_experiment(resultDir, actor_id, actor, runs, rounds, num_suggs,
                   num_pos, num_neg, comp_files, index_k, search_par, seen, suggs_provided=[]):
    global GLOBAL_POS, GLOBAL_NEG, GLOBAL_SUGGS

    relVector = {}
    for i in actor['relevant']:
        feats = r64.read_single_item_features(i, comp_files[0], comp_files[1], comp_files[2])
        for f,v in feats:
            if f in relVector:
                if v > relVector[f]:
                    relVector[f] = v
            else: 
                relVector[f] = v

    metrics = {}
    metrics['p'] = 0.0
    metrics['r'] = 0.0
    metrics['t'] = 0.0
    pn = {}
    for r in range(runs):
        metrics[r] = {}
        metrics[r]['p'] = []
        metrics[r]['r'] = []
        metrics[r]['t'] = []
        metrics[r]['suggs'] = []
        metrics[r]['topk'] = []
        metrics[r]['hplane'] = []
        pn[r] = {}
        pn[r]['pos'] = []
        pn[r]['neg'] = []
    

    seed(1)
    for r in range(runs):
        if USE_EXQ:
            exq.reset_model(True,True)
        else:
            il.reset_model()

        train = True
        rd = 0
        start_time = 0
        current_session_time = 0
        session_end_time = rounds
        train_data = []
        train_labels = []
        GLOBAL_POS = []
        GLOBAL_NEG = []
        GLOBAL_SUGGS = []
        if USE_EXQ:
            train_data = actor['start'][str(r)]
        else:
            train_data = [r64.read_single_item_features(i, comp_files[0], comp_files[1], comp_files[2]) for i in actor['start'][str(r)]]
        train_labels = [1.0 for x in range(5)] + [-1.0 for x in range(5)]
        seen_list = []

        while(current_session_time < session_end_time):
            t_start = time()

            # print("Training")
            # print(train_data, train_labels)
            index_list = []
            if not RNG_RUN:
                if train:
                    if USE_EXQ:
                        exq.train(train_data, train_labels, False, [], False)
                    else:
                        hyperplane = il.train(train_data, train_labels)

                # print(hyperplane)

                # print("Getting suggestions")
                if USE_EXQ:
                    (index_list, total, worker_time, sugg_time, sugg_overhead) = exq.suggest(index_k, 1, [], False, [])
                else:
                    index_list = il.suggest(index_k)
                
            else:
                # index_list = sample(ALL_IDS,num_suggs)
                hyperplane = il.train(train_data, train_labels)
                rng_list = sample(ALL_IDS, index_k)
                index_list = sort_index_list_rng(rng_list, hyperplane, comp_files)

            if seen:
                for s in seen_list:
                    if s in index_list:
                        index_list.remove(s)       
            sugg_list = index_list[:num_suggs]
            print(sugg_list)
            # print("Got suggestions")

            t_stop = time()
            t = t_stop - t_start

            suggs = set(sugg_list)
            seen_set = set(seen_list)
            seen_set |= suggs
            seen_list = list(seen_set)

            # t_classify_start = time()

            if len(suggs_provided) > 0:
                sugg_list = suggs_provided[actor_id][r][rd]
            # AccRep Hardcoded until incremental retrieval is guaranteed
            pos_policy = 1
            neg_policy = 1
            if not len(sugg_list) == 0:
                (pos,neg,done) = classify_suggestions(sugg_list, relVector, actor['relevant'],
                                                      pos_policy, neg_policy, num_pos, num_neg,
                                                      rd+1, comp_files)
            # t_classify_stop = time()
            # print("Time to classify: %f" % (t_classify_stop - t_classify_start))
            # print(pos, neg, done)
                if len(sugg_list) == 0:
                    metrics[r]['p'].append(0.0)
                else:
                    metrics[r]['p'].append(float(len(pos))/len(sugg_list))

                if done:
                    metrics[r]['r'].append(1.0)
                else:
                    metrics[r]['r'].append(0.0)

            metrics[r]['t'].append(t)
            metrics[r]['suggs'].append(sugg_list)
            metrics[r]['topk'].append(index_list)
            if not USE_EXQ:
                metrics[r]['hplane'].append(hyperplane)
            # pn[r]['pos'].append(pos)
            # pn[r]['neg'].append(neg)

            if len(sugg_list) == 0:
                print('%s Actor %d run %d can not advance further!' % (ts(), actor_id, r))
                break
            else:
                train = True
                train_list = pos + neg
                if USE_EXQ:
                    train_data = train_list
                else:
                    train_data = [r64.read_single_item_features(i, comp_files[0], comp_files[1], comp_files[2]) for i in train_list]
                train_labels = [1.0 for x in range(len(pos))] + [-1.0 for x in range(len(neg))]

            rd += 1
            current_session_time = rd

        print("%s Actor %d run %d done after %d rounds." % (ts(), actor_id, r, rd))

    # pn_file = ('a%d_PN.json') % actorId
    # pn_path = os.path.join(resultDir, pn_file)
    # with open(pn_path, 'w') as f:
    #     json.dump(pn,f)

    p_sum_r = 0.0
    t_sum_r = 0.0
    for r in range(runs):
        rds = len(metrics[r]['p'])
        p_sum_rds = 0.0
        t_sum_rds = 0.0
        for rd in range(rds):
            p_sum_rds += metrics[r]['p'][rd]
            metrics['r'] += metrics[r]['r'][rd]
            t_sum_rds += metrics[r]['t'][rd]
        p_sum_r += p_sum_rds/rds
        t_sum_r += t_sum_rds/rds
    metrics['p'] = p_sum_r/runs
    metrics['r'] /= runs
    metrics['t'] = t_sum_r/runs

    return metrics

###

def run_hplane_experiments(hplanes, index_k, comp_files):
    total_queries = len(hplanes['svms'])
    train_labels = [1.0 for i in range(len(hplanes['pos'][0]))] + [-1.0 for i in range(len(hplanes['neg'][0]))]
    metrics = {}
    metrics['topk'] = []
    for q in range(total_queries):
        train_list = hplanes['pos'][q] + hplanes['neg'][q]
        if USE_EXQ:
            train_data = train_list
            exq.train(train_data, train_labels, False, [], False)
        else:
            # Faster to run with the C++ version which takes the hyperplane directly
            train_data = [r64.read_single_item_features(i, comp_files[0], comp_files[1], comp_files[2]) for i in train_list]
            hyperplane = il.train(train_data, train_labels)

        index_list = []
        if USE_EXQ:
            (index_list, total, worker_time, sugg_time, sugg_overhead) = exq.suggest(index_k, 1, [], False, [])
        else:
            index_list = il.suggest(index_k)

        metrics['topk'].append(index_list)
    return metrics

experiment_dir = args.result_dir.joinpath(args.result_file)
main_res_file = args.result_dir.joinpath(args.result_file).joinpath(args.result_file + '.json')
if main_res_file.exists():
    print("RESULT FILE ALREADY EXISTS!")
    exit(0)

if not experiment_dir.is_dir():
    os.mkdir(experiment_dir)


if not args.h5_dir.is_dir() or len(list(args.h5_dir.glob('*.h5'))) == 0:
    print("ERROR: Provided groundtruth directory is either not a directory or contains no HDF5 files.")
    exit(-1)

h5 = []
h5.append(str(list(args.h5_dir.glob('*top*'))[0]))
h5.append(list(args.h5_dir.glob('*ids*'))[0])
h5.append(list(args.h5_dir.glob('*ratios*'))[0])
ALL_IDS = [i for i in range(r64.read_total_items_count(h5[0]))]

if USE_EXQ:
    initialize_exquisitor(args.index_args[2], args.index_path)
else:
    il.initialize(args.index, args.data_rep, str(args.index_path), args.index_args)
    print("Initialized IL!")


actors = []
if args.actors_f is not None:
    if not args.actors_f.is_file():
        print("ERROR: No actors path is not a file")
        exit(-1)

    actors = read_actors(args.actors_f)

    suggs_provided = []
    if args.suggs_provided_f is not None:
        with args.suggs_provided_f.open('r') as f:
            suggs_provided = json.load(f)

    metrics = {}
    for idx, a in enumerate(actors):
        print("Running experiments for Actor", idx)
        metrics[idx] = run_il_experiment(experiment_dir, idx, a, args.number_of_runs, args.number_of_rounds,
                                    args.num_suggestions, args.num_pos, args.num_neg, h5, args.index_k, args.index_args[2],
                                    args.seen, suggs_provided)
        print("%s Actor %d done" % (ts(),idx))
elif args.hplanes_provided_f is not None:
    hplanes_suggs = {}
    with args.hplanes_provided_f.open('r') as f:
        hplanes = json.load(f)
    metrics = {}
    metrics[0] = run_hplane_experiments(hplanes, args.index_k, h5)
else:
    print("No task/hyperplane JSON file provided!")
    exit(1)


if USE_EXQ:
    exq.safe_close()
else:
    il.safe_close()

with main_res_file.open('w') as f:
    json.dump(metrics,f)