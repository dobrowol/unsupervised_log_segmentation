from tools.sequence_segmentation.tree import Tree, tree_from_ngram
from pathlib import Path
import logging
import time
import pickle
import subprocess

logger = logging.getLogger(__name__)

class Ngram:
    def __init__(self, ngram):
        self.tree_depth = ngram

    def count_ngrams(self, file_name):
        import os
        srilm_path = os.getenv('SRILM_PATH')
        out_file = f"{self.out_directory}/out/{Path(file_name).stem}_{self.tree_depth}gram"

        self.split_file_name = self.create_split_file(file_name)

        if Path(out_file).is_file():
            return out_file

        if not Path(f"{srilm_path}/ngram-count").is_file():
            print(f"SRILM not found in {srilm_path}!!!!!!")
            return None
        print(f"{srilm_path}/ngram-count -order {self.tree_depth} -text {self.split_file_name} -no-sos -no-eos -write {out_file} ")
        start_time = time.time()
        subprocess.call(f"{srilm_path}/ngram-count -order {self.tree_depth} -text {self.split_file_name} -no-sos -no-eos -write {out_file}", shell=True)
        logger.info(f"calculating ngram of order {self.tree_depth} for file {self.split_file_name} took ----- %s seconds -----" % (time.time() - start_time))
        return out_file

    def build_ngram_tree(self, file_name):
        lm = self.count_ngrams(file_name)
        if self.ngram_tree is None:
            self.ngram_tree = tree_from_ngram(Tree(0), lm)
        else:
            self.ngram_tree = tree_from_ngram(self.ngram_tree, lm)