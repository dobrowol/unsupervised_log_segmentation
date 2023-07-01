#!/bin/bash

# use data from https://gitlabe1.ext.net.nokia.com/wodobrow/experiments_data/-/tree/main/Nokia_segmentation/array1_array2_bf_scorpio_17_04_2023/no_hlapi_no_ccs_28_04_2023
python uls/best_segmentation.py 100 -voting_experts -text -log ./data/alice/alice_in_space.txt -std ./data/alice/original_segmentation.txt -out_dir ./experiments/alice/out/voting_experts/
