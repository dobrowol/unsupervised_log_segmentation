#!/bin/bash

python -m uls.best_segmentation 100 -drains -voting_experts -training_file /mnt/wodobrow/loghub_logs/CloudStack/CloudStack_over_20.thread -testing_file ./data/CloudStack/CloudStack_over_20.thread -std ./data/CloudStack/golden_standard_19.01.2024.tseg -out_dir ./experiments/out/cloud_stack/voting_experts/
