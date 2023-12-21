#!/bin/bash

python -m uls.best_segmentation 100 -drains -voting_experts -training_file ./data/nokia_cpp/last_100/last_100.txt -testing_file ./data/nokia_cpp/last_100/golden_pm_1_DEFAULT.thread -std ./data/nokia_cpp/golden3.tseg -out_dir ./experiments/out/nokia_cpp/voting_experts_on_history_tr_test_split/
