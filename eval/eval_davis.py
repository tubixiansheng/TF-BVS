#!/usr/bin/env python
import os
import sys
from time import time
import argparse

import numpy as np
import pandas as pd
from tools.davis2017.evaluation import DAVISEvaluation

default_davis_path = 'data/ref_davis/DAVIS'

parser = argparse.ArgumentParser()
parser.add_argument('--davis_path', type=str, help='Path to the DAVIS folder containing the JPEGImages, Annotations, '
                                                   'ImageSets, Annotations_unsupervised folders',
                    required=False, default=default_davis_path)
parser.add_argument('--set', type=str, help='Subset to evaluate the results', default='val')
parser.add_argument('--task', type=str, help='Task to evaluate the results', default='unsupervised',
                    choices=['semi-supervised', 'unsupervised'])
parser.add_argument('--results_path', type=str, help='Path to the folder containing the sequences folders',
                    required=True)
parser.add_argument("--re_eval", action='store_true')
args, _ = parser.parse_known_args()
csv_name_global = f'global_results-{args.set}.csv'
csv_name_per_sequence = f'per-sequence_results-{args.set}.csv'

num_anno = 4
all_results = []
for i in range(num_anno):
    time_start = time()
    anno_results_path = os.path.join(args.results_path, f'anno_{i}')

    csv_name_global_path = os.path.join(anno_results_path, csv_name_global)
    csv_name_per_sequence_path = os.path.join(anno_results_path, csv_name_per_sequence)
    if os.path.exists(csv_name_global_path) and os.path.exists(csv_name_per_sequence_path) and not args.re_eval:
        print('Using precomputed results...')
        table_g = pd.read_csv(csv_name_global_path)
        table_seq = pd.read_csv(csv_name_per_sequence_path)
    else:
        print(f'Evaluating sequences for the {args.task} task...')
        dataset_eval = DAVISEvaluation(davis_root=args.davis_path, task=args.task, gt_set=args.set)
        metrics_res = dataset_eval.evaluate(anno_results_path)
        J, F = metrics_res['J'], metrics_res['F']

        g_measures = ['J&F-Mean', 'J-Mean', 'J-Recall', 'J-Decay', 'F-Mean', 'F-Recall', 'F-Decay']
        final_mean = (np.mean(J["M"]) + np.mean(F["M"])) / 2.
        g_res = np.array([final_mean, np.mean(J["M"]), np.mean(J["R"]), np.mean(J["D"]), np.mean(F["M"]), np.mean(F["R"]),
                          np.mean(F["D"])])
        g_res = np.reshape(g_res, [1, len(g_res)])
        table_g = pd.DataFrame(data=g_res, columns=g_measures)
        with open(csv_name_global_path, 'w') as f:
            table_g.to_csv(f, index=False, float_format="%.5f")
        print(f'Global results saved in {csv_name_global_path}')

        seq_names = list(J['M_per_object'].keys())
        seq_measures = ['Sequence', 'J-Mean', 'F-Mean']
        J_per_object = [J['M_per_object'][x] for x in seq_names]
        F_per_object = [F['M_per_object'][x] for x in seq_names]
        table_seq = pd.DataFrame(data=list(zip(seq_names, J_per_object, F_per_object)), columns=seq_measures)
        with open(csv_name_per_sequence_path, 'w') as f:
            table_seq.to_csv(f, index=False, float_format="%.5f")
        print(f'Per-sequence results saved in {csv_name_per_sequence_path}')

    sys.stdout.write(f"--------------------------- Global results for {args.set} ---------------------------\n")
    print(table_g.to_string(index=False))
    sys.stdout.write(f"\n---------- Per sequence results for {args.set} ----------\n")
    print(table_seq.to_string(index=False))
    total_time = time() - time_start
    sys.stdout.write('\nTotal time:' + str(total_time))
    all_results.append(table_g)
    print("\n")

all_results = pd.concat(all_results, axis=0, ignore_index=True)
all_results.index = [f'an{i}' for i in range(num_anno)]
avg_results = pd.DataFrame([all_results.mean()], index=['avg'])
all_results = pd.concat([all_results, avg_results], axis=0, ignore_index=False)
print(all_results.to_string())
