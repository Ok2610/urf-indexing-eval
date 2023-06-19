import read_r64 as r64
import argparse
import json
import il
from random import sample
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('h5_dir', type=Path, help="Directory of the compressed hdf5 files")
parser.add_argument('output_f', type=str)
parser.add_argument('--n_planes', type=int, default=1000)
parser.add_argument('--n_pos', type=int, default=50)
parser.add_argument('--n_neg', type=int, default=200)

args = parser.parse_args()

if not args.h5_dir.is_dir() or len(list(args.h5_dir.glob('*.h5'))) == 0:
    print("ERROR: Provided groundtruth directory is either not a directory or contains no HDF5 files.")
    exit(-1)

h5 = []
h5.append(str(list(args.h5_dir.glob('*top*'))[0]))
h5.append(list(args.h5_dir.glob('*ids*'))[0])
h5.append(list(args.h5_dir.glob('*ratios*'))[0])
print(h5)

collection_size = r64.read_total_items_count(h5[0])

ids = [i for i in range(collection_size)]
ids_set = set(ids)

# Hard coded initialize call. Don't need to use the index, but need it to initialize the classifier
il.initialize(0, 0, '../cpp/interactive_learning/data/lsc/hnswlsc_48M_500ef_L2.bin', [0, 1000, 1000])

hplanes = {}
hplanes['svms'] = []
hplanes['pos'] = []
hplanes['neg'] = []
train_labels = [1.0 for x in range(args.n_pos)] + [-1.0 for x in range(args.n_neg)]
for i in range(args.n_planes):
    pos = sample(ids, args.n_pos)
    rem = list(ids_set - set(pos))
    neg = sample(rem, args.n_neg)
    train_data = [r64.read_single_item_features(i, h5[0], h5[1], h5[2]) for i in pos + neg]
    hplane = il.train(train_data, train_labels)
    hplanes['svms'].append(hplane)
    hplanes['pos'].append(pos)
    hplanes['neg'].append(neg)
    if (i/args.n_planes) % 0.1 == 0:
        print('Progress: %d' % int((i/args.n_planes) * 100))

with open(args.output_f,'w') as f:
    json.dump(hplanes,f)
