#!/bin/bash

python uls/best_segmentation.py 100 -top_words -text -log ./data/alice/alice_in_space.txt -std ./data/alice/original_segmentation.txt -out_dir ./experiments/alice/out/top_words/
