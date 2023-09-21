import pickle
from pathlib import Path
from tqdm import tqdm
import re

from uls.voting_experts.voting_experts import VotingExperts
from uls.topwords.topwords import TopWORDS
from uls.fixed_window import FixedWindow

def is_in(item, list_of_tuples):
    for tup in list_of_tuples:
        if item == tup[0] or item == tup[1]:
            return True
    return False

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
    golden_splits = get_splits_from_text_file(original_text_file)
    received_splits = get_splits_from_text_file(segmentation_file)
    return f1_score(received_splits, golden_splits)

def golden_text_f1_score(segmentation, original_text_file):
    golden_splits = get_splits_from_text_file(original_text_file)
    #print("golden txt splits ", golden_splits)
    received_splits = segmentation
    return f1_score(received_splits, golden_splits)

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
    parser.add_argument("-unsegmented", type=str)
    parser.add_argument("-golden", type=str)
    parser.add_argument("-window", type=int)
    parser.add_argument("-threshold", type=int)
    parser.add_argument("-out_dir", type=str)
    args = parser.parse_args()
    f1_score = voting_experts_text_f1_score(args.unsegmented, 
                                             args.golden, 
                                             args.window, args.threshold, 
                                             args.out_dir)
    print(f"F1-score for window {args.window}, threshold {args.threshold} is {f1_score}")