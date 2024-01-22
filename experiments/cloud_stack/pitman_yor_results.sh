#python -m uls.golden_std_segmentation -golden_test -binary ./experiments/out/nokia_cpp/pitman_yor/segmentation_100epochs.txt ./data/nokia_cpp/golden3.tseg
python -m uls.golden_std_segmentation -golden_test -drains --segmented_file ./data/CloudStack/npylm_segmentation_20.01.2024.txt --std ./data/CloudStack/golden_standard_19.01.2024.tseg
