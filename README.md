# Unsupervised log segmentation
This repository contains implementation used in paper:
It contains different methods of segmenting log sequence into meaningful segments done in unsupervised manner.

## Preparing environment
Clone repository
```
git clone https://github.com/dobrowol/unsupervised_log_segmentation.git
cd unsupervised_log_segmentation
```
To install dependencies you will need virtual env and installation based on requirements.txt:
```
python -m venv .shell
source .shell/bin/activate
pip install -r requirements.txt
```


## Usage 

There are two datasets available:
* Logs from Nokia stored as a sequence of log events ids separated with spaces are in ./data/nokia directory. 
* Text from "Alice's Adventures in Wonderland" stored as a sequence of letters separated with spaces are in ./data/alice directory.

To run experiments run:
```buildoutcfg
sh ./experiments/nokia/best_voting_experts.sh
```

## Disclaimer!!

Bayes optimization process is not deterministic process. Your execution may differ significantly from achieved by us. However you can verify that results presented in the paper are correct by running

```
python uls/calculate_specific_results.py
```
