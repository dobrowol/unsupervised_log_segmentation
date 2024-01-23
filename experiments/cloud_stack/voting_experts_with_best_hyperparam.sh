#!/bin/bash

python -m uls.golden_std_segmentation --train ./data/CloudStack/CloudStack_over_20.thread --test ./data/CloudStack/CloudStack.thread --std ./data/CloudStack/golden_standard.tseg -voting_experts -window 3 -threshold 1 -out_dir ./experiments/out/cloud_stack/voting_experts_test/
