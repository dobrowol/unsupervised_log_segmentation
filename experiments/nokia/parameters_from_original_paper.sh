#!/bin/bash

python -m uls.golden_std_segmentation -voting_experts -drain -train ../dcpp/data/last_100_out/golden_threads.txt -test ../dcpp/data/last_100_out/golden_threads.txt -std ./data/nokia_pm_logs/golden_standard.tseg -window 7 -threshold 4 -out_dir ./experiments/out/nokia_cpp/voting_experts_golden/
