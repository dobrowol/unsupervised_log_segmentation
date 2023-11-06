import pickle
from pathlib import Path
from tqdm import tqdm
import re
import numpy as np
from sklearn.metrics import precision_recall_fscore_support as fscore
from sklearn.metrics import confusion_matrix

from uls.voting_experts.voting_experts import VotingExperts
from uls.topwords.topwords import TopWORDS
from uls.fixed_window import FixedWindow
from typing import List
import itertools
import segment_evaluator

def is_in(item, list_of_tuples):
    for tup in list_of_tuples:
        if item == tup[0] or item == tup[1]:
            return True
    return False

def flatten(multi_list):
    """Flatten a list of lists into a single list"""
    return list(itertools.chain.from_iterable(multi_list))

def character_tokenize(
    in_file
):
    """
    Character tokenize the input file, lowercasing and exlcuding whitespace
    
    If an `out_file` is given print the tokenized text to this file
    """
    # For drains only this:
    with open(in_file, "r") as in_file:
        lines = in_file.read().splitlines()
    text =[line.split() for line in lines]
    return text

def get_boundary_vector(sequence: List[List[str]]) -> List[int]:
    """
    Extract a binary vector representing segmentation boundaries from a list of
    segments

    The vector's length is equal to the number of symbols in the 
    sequence, where for every symbol, vector(x) = 1 if there is a boundary
    after the symbol at position x, and 0 if there is no boundary after the
    symbol at x. If the segmentation is considered "gold", the last value will
    be 1, otherwise 0

    Args:
        sequence: A list of segments, which are themselves lists of string 
            symbols, from which to construct a (segment) boundary vector. Needs
            to be a list of lists because some items representing one "symbol"
            may be more than one character (e.g. `<tag>`)
    Returns:
        A list vector representing the segment boundaries (1 for boundary, 0 for
            no boundary)
    """

    lengths = [len(segment) for segment in sequence]
    num_symbols = sum(lengths)
    vector = [0 for x in range(num_symbols)]
    current_index = 0
    for length in lengths[:-1]:
        current_index += length
        vector[current_index - 1] = 1
    return vector[:-1]

def f1_score(received_splits, golden_splits, text_len):
    true_positives = sum([1 for split in received_splits if split in golden_splits])
    false_positives = sum([1 for split in received_splits if split not in golden_splits])
    false_negatives = sum([1 for split in golden_splits if split not in received_splits])
    true_negatives = text_len - len(golden_splits)

    print("true_positives ", true_positives)
    print("false_positives ", false_positives)
    print("false_negatives ", false_negatives)
    print("true_negatives ", true_negatives)
    if len(received_splits) == 0:
        precision = 0
    else:
        precision = true_positives/len(received_splits)
    
    recall = true_positives/len(golden_splits)

    print("precision ", precision)
    print("recall ", recall)
    if precision + recall == 0:
        return 0
    if precision + recall == 0:
        print("Warning F1_score is 0!!!")
        F1_score = 0
    else:
        F1_score = 2*precision*recall/(precision + recall)
    return F1_score

def get_splits_from_segments(segmentation):
    splits = []
    line = 0
    for seg in segmentation[1:]:
        if seg[0] in splits:
            print(f"ERROR split {seg[0]} already in splits")
        else:
            splits.append(seg[0])
    if segmentation[-1][1]+1 in splits:
            print(f"ERROR split {segmentation[-1][1]+1} already in splits")
    else:
        splits.append(segmentation[-1][1]+1)
    return splits

def get_splits_from_text_file(text_file):
    #print("opening file ", text_file)
    with open(text_file, "r") as original_file:
        text = original_file.read().rstrip('\n')
    segmnt = text.split(' ')
    return get_splits_from_seq(segmnt)

def get_splits_from_seq(segmnt):
    splits = []
    start_idx = 0
    for word in segmnt:
        splits.append(start_idx+len(word))
        start_idx = start_idx+len(word)
    return splits

def golden_files_f1_score(segmentation_file, original_text_file):
    golden_tokens = character_tokenize(original_text_file)
    dev_tokens = character_tokenize(segmentation_file)
    gold_boundaries = [get_boundary_vector(ex) for ex in golden_tokens]
    all_gold_boundaries = np.array(flatten(gold_boundaries))
    dev_boundaries = [get_boundary_vector(ex) for ex in dev_tokens]
    all_dev_boundaries = np.array(flatten(dev_boundaries))
    _,_,f1score,_ = fscore(all_gold_boundaries, all_dev_boundaries, average='binary')
    conf_mat = confusion_matrix(all_gold_boundaries, all_dev_boundaries)
    return f1score, conf_mat

def golden_text_f1_score(segmentation, original_text_file):
    golden_tokens = character_tokenize(original_text_file)
    dev_tokens = character_tokenize(segmentation)
    gold_boundaries = [get_boundary_vector(ex) for ex in golden_tokens]
    all_gold_boundaries = np.array(flatten(gold_boundaries))
    dev_boundaries = [get_boundary_vector(ex) for ex in dev_tokens]
    all_dev_boundaries = np.array(flatten(dev_boundaries))
    prec,rec,f1score,_ = fscore(all_gold_boundaries, all_dev_boundaries, average='binary')
    conf_mat = confusion_matrix(all_gold_boundaries, all_dev_boundaries)
    return prec,rec,f1score, conf_mat

def golden_std_f1_score(segmentation, golden_std_file_path):
    """
    segmentation: a list of tuples containing starting and ending line of segments:
                ['it was as cloudy day'] -> [2, 5, 7, 13, 16]
    golden_std_file_path: the golden standard segmentation of the file being segmented.

    returns: float. F1 score of segmentation calculated by comparing if segment is present in golden standard.
                    Traditionally:
                        precision = num of true positives / num of all proposed segments.
                        recall = num of true positives / num of all golden segments.
                        F1= 2*precision*recall/(precision + recall)
    """
    with open(golden_std_file_path, "rb") as golden_std_file:
        golden_segmentation = pickle.load(golden_std_file)
     
    # [(1,5), (6,8)] split is end of previoius word and beg of nex word
    # in this way segmentation [(1,4),(5,6)] is equally wrong with [(1,2),(3,4)]
    if len(segmentation) == 0:
        return 0
    received_splits = segmentation#get_splits_from_segments(segmentation)
    golden_segments = sorted(golden_segmentation.keys(), key=lambda item: item[0])
    golden_splits = get_splits_from_segments(golden_segments)
    #print("golden std splits ", golden_splits, "golden splits")
    return f1_score(received_splits, golden_splits)

def voting_experts_f1_score(drained_file, golden_std_file_path, depth, threshold, out_directory):
    ve = VotingExperts(depth, threshold, out_directory=out_directory)
    
    segmentation = ve.fit_transform(drained_file)

    return golden_std_f1_score(segmentation, golden_std_file_path)

def voting_experts_text_f1_score(drained_file, golden_std_file_path, depth, threshold, out_directory):
    ve = VotingExperts(depth, threshold, out_directory=out_directory)
    
    segmentation = ve.fit_transform(drained_file)

    return golden_text_f1_score(segmentation, golden_std_file_path)

def fixed_window_f1_score(drain_file_path, golden_std_file_path, alignment, window_size):
    fw = FixedWindow(alignment, window_size)
    segmentation = fw.fit_transform(drain_file_path)
    return golden_std_f1_score(segmentation, golden_std_file_path)

def fixed_window_text_f1_score(drain_file_path, golden_std_file_path, alignment, window_size):
    fw = FixedWindow(alignment, window_size)
    segmentation = fw.fit_transform(drain_file_path)
    return golden_text_f1_score(segmentation, golden_std_file_path)

def top_words_f1_score(runtime_file_path, golden_std_file_path, word_length, threshold, probability_threshold,
                           word_boundary_threshold, out_directory):
    runtime_drain_path = runtime_file_path
    tw = TopWORDS(out_directory, word_length, threshold, probability_threshold, word_boundary_threshold)
    segmentation = tw.fit_transform(runtime_drain_path)
    return golden_std_f1_score(segmentation, golden_std_file_path)

def top_words_text_f1_score(runtime_file_path, golden_std_file_path, word_length, threshold, probability_threshold,
                           word_boundary_threshold, out_directory):
    runtime_drain_path = runtime_file_path
    tw = TopWORDS(out_directory, word_length, threshold, probability_threshold, word_boundary_threshold)
    segmentation = tw.fit_transform(runtime_drain_path)
    return golden_text_f1_score(segmentation, golden_std_file_path)

if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("unsegmented", type=str)
    parser.add_argument("golden", type=str)
    parser.add_argument("-word_len", type=int)
    parser.add_argument("-window", type=int)
    parser.add_argument("-threshold", type=int)
    parser.add_argument("-voting_experts", action='store_true')
    parser.add_argument("-word_freq", type=int)
    parser.add_argument("-out_dir", type=str)
    args = parser.parse_args()
    if args.voting_experts:
        prec, recall, f1score, conf_mat = voting_experts_text_f1_score(args.unsegmented,
                                                    args.golden,
                                                    args.window,
                                                    args.threshold,
                                                    args.out_dir)
    else:
        prec, recall, f1score, conf_mat = top_words_text_f1_score(args.unsegmented, 
                                             args.golden, 
                                             args.word_len, args.word_freq, 1.0*10**(-6), 0,
                                             args.out_dir)
    print(f"Precision is {prec}")
    
    print(f"Recall is {recall}")

    print(f"F1-score is {f1score}")

    print("conf matrix ", conf_mat)
