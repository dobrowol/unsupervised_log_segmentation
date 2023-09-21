#!/bin/bash

python uls/best_segmentation.py 100 -voting_experts -text -log ./data/alice/alice_in_space.txt -std ./data/alice/original_segmentation.txt -out_dir ./experiments/alice/out/voting_experts/
