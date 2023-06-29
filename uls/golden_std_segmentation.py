import pickle
from pathlib import Path
from tqdm import tqdm
import re

from uls.voting_experts.voting_experts import VotingExperts
from uls.file_manipulation import FileSegmentation
import uls.topwords.topwords as topwords

def is_in(item, list_of_tuples):
    for tup in list_of_tuples:
        if item == tup[0] or item == tup[1]:
            return True
    return False

def f1_score(received_splits, golden_splits):
    correct_received_splits = sum([1 for split in received_splits if split in golden_splits])

    if len(received_splits) == 0:
        precision = 0
    else:
        precision = correct_received_splits/len(received_splits)
    
    recall = correct_received_splits/len(golden_splits)

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
    print("opening file ", text_file)
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

def voting_experts_not_pretrained_text_file(drained_file, golden_std_file_path, depth, threshold, out_directory):
    ve = VotingExperts(depth,threshold, out_directory=out_directory)
    
    segm_file = Path(out_directory)/(str(Path(drained_file).stem)+f"{depth}_{threshold}.sl")
    fs = FileSegmentation()
    received_file = fs.segment_text_file(drained_file, ve)
    with open(received_file, "rb") as aFile:
        segments = pickle.load(aFile)
    line_number=0
    segmentation = []
    out_text = Path(out_directory)/"segmented_text.txt"
    with open(out_text, "w") as text_file:
        for segment in segments:
            text_file.write(''.join(segment)+" ")
    for segment in tqdm(segments, desc="Collecting segments"):
        segmentation.append(line_number+len(segment))
        line_number+=len(segment)

    with open(segm_file, "wb") as afile:
        print("segmentation saved to ",segm_file)
        pickle.dump(segmentation, afile)

    return golden_text_f1_score(segmentation, golden_std_file_path)

def voting_experts_not_pretrained(drained_file, golden_std_file_path, depth, threshold, out_directory):
    ve = VotingExperts(depth,threshold, out_directory=out_directory)
    
    segm_file = Path(out_directory)/(str(Path(drained_file).stem)+f"{depth}_{threshold}.sl")
    fs = FileSegmentation()
    received_file = fs.segment_file(drained_file, ve)
    with open(received_file, "rb") as aFile:
        segments = pickle.load(aFile)
    line_number=0
    segmentation = []

    for segment in tqdm(segments, desc="Collecting segments"):
        segmentation.append(line_number+len(segment))
        line_number+=len(segment)

    return golden_std_f1_score(segmentation, golden_std_file_path)

def voting_experts_pretrained(runtime_file_path, golden_std_file_path, depth, threshold, historical_dataset, historical_prs, out_directory):
    ve = VotingExperts(depth,threshold, out_directory=out_directory)
    
    runtimes = []
    with open(historical_dataset, "r") as dataset:
        for _ in range(historical_prs):
            runtimes.append(dataset.readline().strip())
    ve.fit(runtimes)
    #ve.retrieve_tree(out_directory)
    segm_file = Path(out_directory)/(str(Path(runtime_file_path).stem)+f"{depth}_{threshold}.sl")
    fs = FileSegmentation()
    received_file = fs.segment_file(runtime_file_path, ve)
    with open(received_file, "rb") as aFile:
        segments = pickle.load(aFile)
    line_number=0
    segmentation = []
    for segment in tqdm(segments, desc="Collecting segments"):
        segmentation.append((line_number,line_number+len(segment)))
        line_number+=len(segment)

    return golden_std_f1_score(segmentation, golden_std_file_path)

def fixed_window_segmentation(runtime_file_path, golden_std_file_path, alignment, window_size, out_directory):
    with open(runtime_file_path, "r") as runtime:
        line_count = len(runtime.read().splitlines())
    segmentation = []
    curr_line = 0
    for line in range(alignment, line_count, window_size):
        segmentation.append(line)
        curr_line = line
    if curr_line < line_count:
        segmentation.append(line_count)
    return golden_std_f1_score(segmentation, golden_std_file_path)

def time_window_segmentation(runtime_file_path, golden_std_file_path, time_span, out_directory):
    segmentation = []
    start_time = 0
    end_time = 0
    start_line = 0
    lines_cnt = 0
    with open(runtime_file_path, "r") as runtime:
        line = runtime.readline()
        log_time = float(re.findall(r"\d{2}:\d{2}:(\d{2}\.\d+)", line, re.DOTALL)[0])
        start_time = log_time
        end_time = log_time
        for idx,line in enumerate(runtime):
            lines_cnt+=1
            log_time = float(re.findall(r"\d{2}:\d{2}:(\d{2}\.\d+)", line, re.DOTALL)[0])
            end_time = log_time
            if end_time - start_time > time_span:
                segmentation.append(lines_cnt)
                start_line = lines_cnt
                start_time = end_time

    if start_line < lines_cnt:
        segmentation.append(lines_cnt)
    return golden_std_f1_score(segmentation, golden_std_file_path)

def top_words_segmentation(runtime_file_path, golden_std_file_path, word_length, threshold, probability_threshold,
                           word_boundary_threshold, out_directory):
    runtime_drain_path = runtime_file_path
    segmentation = topwords.segmentation(runtime_drain_path, out_directory, word_length, threshold, probability_threshold, word_boundary_threshold)
    return golden_std_f1_score(segmentation, golden_std_file_path)

if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-log", type=str)
    parser.add_argument("-std", type=str)
    parser.add_argument("-window", type=int)
    parser.add_argument("-threshold", type=int)
    parser.add_argument("-out_dir", type=str)
    args = parser.parse_args()
    f1_score = voting_experts_not_pretrained(args.log, 
                                             args.std, 
                                             args.window, args.threshold, 
                                             args.out_dir)
    print(f"F1-score for window {args.window}, threshold {args.threshold} is {f1_score}")