import pickle
import subprocess
import numpy as np
import time
import math
import os
import logging
from tqdm import tqdm
from pathlib import Path
from uls.voting_experts.tree import find_node, calculate_experts_features
from uls.voting_experts.tree import tree_from_ngram, Tree, get_stats, standardize
from uls.voting_experts.tree import build_tree, update_tree
from uls.voting_experts.ngram import Ngram

logger = logging.getLogger(__name__)

def get_num_of_sliding_windows_in_a_sentence(sentence, sliding_window_size):
    num_of_sliding_windows_in_a_sentence=len(sentence)-sliding_window_size+1
    return 1 if num_of_sliding_windows_in_a_sentence<=0 else num_of_sliding_windows_in_a_sentence

def _split_with_length(sentence, size_of_a_bucket, ngram_size):
    sentence_len = len(sentence)
    #split_data = [sentence[0:size_of_a_bucket]]
    split_data_2 = [sentence[i:i+size_of_a_bucket] for i in range(0, sentence_len, size_of_a_bucket)]
    #split_data.extend(split_data_2)
    return split_data_2

def _split(sentence, threads, sliding_window_size):
    num_of_sw_in_sentence=get_num_of_sliding_windows_in_a_sentence(sentence, sliding_window_size)
    size_of_a_bucket = math.ceil(num_of_sw_in_sentence/threads)  
    split_data = [[i,sentence[i:i+size_of_a_bucket+sliding_window_size-1]] for i in range(0, num_of_sw_in_sentence, size_of_a_bucket)]
    return split_data


class VotingExperts(Ngram):
    def __init__(self, window_size, threshold, out_directory=None, ngram_tree=None, silent=True, threads=55,
                  max_line_size=140000):
        """
        tree_depth: int. Depth of n_gram tree. Depth of 4 will create a tree of 3_grams
        window_size: int. Size of window considered by voting expert in each iteration. Should be less than (tree_depth - 1)
        threshold: int. Boundary of votes that needs to be exceeded in order to split the sentence at given index. With only entropy =2 works well.
        silent: boolean. If false, program will show current progress.
        threads: int. Number of threads working in parallel
        """
        super().__init__(window_size+1)
        self.out_directory=out_directory
        self.tree_depth = window_size+1
        self.threshold = threshold
        self.silent = silent
        self.window_size = window_size
        self.ngram_tree = ngram_tree
        self.threads = threads
        self.longest_word = 0
        self.max_line_size = max_line_size
        self.tree_name = "ngram_tree"

    def get_ngram_tree(self):
        return self.ngram_tree

    def get_longest_word(self):
        return self.longest_word

    def set_tree_depth(self, value):
        self.tree_depth = value

    def find_frequencies(self, node, depth, nodes):
        if depth == 0:
            if node.value > 0:
                nodes.append(node.value)
        for child in node.nodes.values():
            self.find_frequencies(child, depth - 1, nodes)
    
    def count_ngrams(self, file_name):

        srilm_path = os.getenv('SRILM_PATH')
        out_file = Path(self.out_directory)/f"{Path(file_name).stem}_{self.tree_depth}gram"

        self.split_file_name = self.create_split_file(file_name)

        if Path(out_file).is_file():
            return out_file

        logger.debug(f"{srilm_path}/ngram-count -order {self.tree_depth} -text {self.split_file_name} -no-sos -no-eos -write {out_file} ")
        start_time = time.time()
        subprocess.call(f"{srilm_path}/ngram-count -order {self.tree_depth} -text {self.split_file_name} -no-sos -no-eos -write {out_file}", shell=True)
        logger.debug(f"calculating ngram of order {self.tree_depth} for file {self.split_file_name} took ----- {(time.time() - start_time)} seconds -----")
                    
        return out_file
    
    def retrieve_tree(self, a_dir):
        if self.ngram_tree is not None:
            return self.ngram_tree
        start_time = time.time()
        trees = list(Path(a_dir).glob(f"*_{self.tree_depth}gram.tree"))
        if len(trees) == 0:
            return None
        tree_file = trees[0]
        logger.debug(f"using old tree {tree_file}")
        with open(tree_file, "rb") as inp:
            self.ngram_tree = pickle.load(inp)
            logger.info("loading old ngram tree took ----- %s seconds -----" % (time.time() - start_time))
        return tree_file

    def fit(self, file_name):
        if self.max_line_size < self.tree_depth:
            logger.error("tree depth cannot be greater than a line")
            return False
        start_time = time.time()
        self.retrieve_tree(f"{self.out_directory}")
        if self.out_directory is None:
            self.out_directory = Path(file_name).parent
        Path(f"{self.out_directory}").mkdir(parents=True, exist_ok=True)

        if self.window_size < 1 or self.tree_depth < 2:
            return False
        srilm_path = os.getenv('SRILM_PATH')
        if Path(f"{srilm_path}/ngram-count").is_file():
            self.build_ngram_tree_with_srilm(file_name)
        else:
            self.build_ngram_tree(file_name)
        
        out_file = Path(self.out_directory)/f"{self.tree_name}_{self.tree_depth}gram.tree"
        with open(out_file, "wb") as out:
            pickle.dump(self.ngram_tree, out)
        logger.debug(f"ngram tree saved to {out_file}")
        logger.info("-----fit took %s seconds -----" % (time.time() - start_time))
        #print("-----fit took %s seconds -----" % (time.time() - start_time))
        self.standardized = False
        return True

    def build_ngram_tree_with_srilm(self, file_name):
        lm = self.count_ngrams(file_name)
        if self.ngram_tree is None:
            self.ngram_tree = tree_from_ngram(Tree(0), lm)
        else:
            self.ngram_tree = tree_from_ngram(self.ngram_tree, lm)
        Path(lm).unlink()

    def build_ngram_tree(self, file_name):
        with open(file_name, "r") as data_file:
            lines = 0
            for line in data_file:
                lines +=1
                sentence = []
                for word in line.strip().split():
                    sentence.append(word)
            if lines == 0:
                logger.error("Draining empty. See above errors.")
                return
            if not self.ngram_tree:
                self.ngram_tree = build_tree(sentence, self.tree_depth)
            else:
                update_tree(self.ngram_tree, sentence, self.tree_depth)

    def standardize_tree(self):
        if not self.standardized:
            standards = {}
            calculate_experts_features(self.ngram_tree, standard=standards)
            stats = get_stats(standards)
            logger.info("standardizing experts features")
            standardize(self.ngram_tree, stats, threads=self.threads)
            self.standardized = True

    def transform(self, file_name):
        self.standardize_tree()

        return self.transform_file(file_name)

    def transform_file(self, file_name):
        out_filename = Path(self.out_directory)/f"{Path(file_name).stem}_{self.window_size}_{self.threshold}_segmented.out"
        if Path(out_filename).is_file():
            with open(out_filename, "rb") as saved_file:
                split_indexes = pickle.load(saved_file)
            return split_indexes

        start_time = time.time()
        self.split_file_name = self.create_split_file(file_name)
        with open( self.split_file_name) as data_file:
            line = data_file.read().rstrip('\n')
            segment_indexes = self.transform_line(line)
        with open(out_filename, "wb") as save_file:
            pickle.dump(segment_indexes, save_file)

        logger.info("-----transform with %d threads took %s seconds -----" % (self.threads, time.time() - start_time))
        #print("-----transform with %d threads took %s seconds -----" % (self.threads, time.time() - start_time))
        logger.debug(f"fragmented file saved to {out_filename}")
        return segment_indexes

    def transform_line(self, line):
        sentence =  line.strip().split(' ') 
        if sentence == []:
            return          
        split_pattern = self.vote(sentence)
        return self.split(split_pattern)

    def vote(self, sequence):
        """
        sequence: a list of characters creating text meant to be split
            for example ['i', 't', 'w', 'a', 's', 'a' ,'c', 'o', 'l', 'd'] instead of 'it was a cold' or ['11', '12', '11', '13'] instead of '11 12 11 13'

        returns: list[str]. List of len(sequence) with votes for each possible split
        """
        fixed_length = len(sequence)
        slice_array = np.zeros(fixed_length)
        windows_count = fixed_length - self.window_size + 1
        iter_range = range(windows_count)
        if not self.silent:
            iter_range = tqdm(iter_range)
        for i in iter_range:
            subsequence = sequence[i:i + self.window_size]
            entropies = []
            frequencies = []
            prev_node = self.ngram_tree
            for j in range(0,len(subsequence)):
                node = prev_node.nodes[subsequence[j]]
                node_rest = find_node(subsequence[j+1:], self.ngram_tree)
                prev_node = node
                entropies.append(node.entropy)
                frequencies.append(node.frequency + node_rest.frequency)
            max_entropy = max(entropies)
            max_frequency = max(frequencies)
            entropy_slice_spot = i + entropies.index(max_entropy)
            frequency_slice_spot = i + frequencies.index(max_frequency)
            slice_array[entropy_slice_spot] += 1
            slice_array[frequency_slice_spot] += 1
        return slice_array

    def split(self, slicing_pattern):
        indexes = []
        for i in range(len(slicing_pattern)):
            if slicing_pattern[i] > self.threshold:
                indexes.append(i+1)
        return indexes

    def vote_thread(self, sub_id, sub, return_dict):
        pattern = self.vote(sub)
        return_dict[sub_id] = pattern

    def create_split_file(self, file_name):
        split_file_name = Path(self.out_directory)/f"{Path(file_name).stem}_{self.max_line_size}_splitted.txt"
        if not Path(split_file_name).is_file():
            with open(file_name, "r") as data_file:
                for line in data_file:
                    sentence =  line.strip().split(' ') 
                    if len(sentence) < self.max_line_size:
                        return file_name
                    
                    segments = _split_with_length(sentence, self.max_line_size, self.tree_depth)
                    
                    if 'split_file' not in locals():
                        split_file = open(split_file_name, "w")
                    for seg in segments:
                        split_file.write(' '.join(seg)+'\n')
                    split_file.close()
        logger.debug("splitting file to ", split_file_name)
        return split_file_name
          
    def save_results(self, out_filename, transformed_lines):
        with open(out_filename, "wb") as out_file:
            for key in sorted(transformed_lines.keys()):
                pickle.dump(transformed_lines[key], out_file)

    def fit_transform(self, file):
        """
        dataset_file: string. File contains list of *.drain files that are going to be scanned

        returns a file name with result. Each line is a list of segments of log_ids separated by whitespaces. Example ['11 11 12', '13 14', '15', '16 17 18 19']
        """
        logger.debug(f"fit transform Entropy window {self.window_size} threshold {self.threshold}")
        if self.window_size < 1 or self.tree_depth < 2 or self.window_size >= self.tree_depth:
            return []
 
        self.fit(file)

        return self.transform(file)

if __name__ == '__main__':
    import sys
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("window", type=int)
    parser.add_argument("threshold", type=int)
    parser.add_argument("out_dir", type=str)

    args = parser.parse_args()

    ve = VotingExperts(args.window, args.threshold, out_directory=args.out_dir)
    with open(args.dataset, "r") as dataset:
            files_list = dataset.read().splitlines()
    ve.fit(files_list)