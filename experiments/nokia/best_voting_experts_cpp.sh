#!/bin/bash

python -m uls.best_segmentation 100 -binary -voting_experts -log ./data/nokia_cpp/pm_1_DEFAULT.thread -std ./data/nokia_cpp/golden3.tseg -out_dir ./experiments/out/nokia_cpp/voting_experts/
