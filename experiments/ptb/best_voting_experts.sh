#!/bin/bash

python -m uls.best_segmentation 100 -voting_experts -text -log ./data/ptb/sentences/all_space.txt -std ./data/ptb/all.txt -out_dir ./experiments/ptb/out/voting_experts/
