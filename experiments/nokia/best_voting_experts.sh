#!/bin/bash

python uls/best_segmentation.py 100 -voting_experts -log ./data/nokia/drained_log.txt -std ./data/nokia/golden_std.pkl -out_dir ./experiments/out/nokia/voting_experts/
