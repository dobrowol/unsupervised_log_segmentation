from uls.voting_experts.voting_experts import VotingExperts
from uls.topwords.topwords import TopWORDS
from uls.fixed_window import FixedWindow
from uls.golden_std_segmentation import golden_text_f1_score

def build_lexicon_from_segmentation(file, segmentation):
    with open(file, "r") as inp_file:
        events = inp_file.read().rstrip('\n').split(' ')
    lexicon = set()
    prev_idx = 0
    for idx in segmentation:
        lexicon.add('_'.join(events[prev_idx:idx]))
        prev_idx = idx
    
    return lexicon

def get_results(file, segmentation_method, golden_std_file):
    segmentation = segmentation_method.fit_transform(file)
    lexicon = build_lexicon_from_segmentation(file, segmentation)
    print(" lexicon size ", len(lexicon))
    print(" F-score ", golden_text_f1_score(segmentation, golden_std_file))

if __name__ == '__main__':
    with open('./data/alice/original_segmentation.txt', "r") as original:
        original_lexicon = set(original.read().rstrip('\n').split(' '))
    print("lexicon from original file size ", len(original_lexicon))

    import time

    fw_alignment = 0
    fw_window_size = 3
    start_time = time.time()
    print('---------')
    print(f"FixedWindow (alignemnt:{fw_alignment}, window size:{fw_window_size})")
    fw = FixedWindow(fw_alignment, fw_window_size)
    get_results('./data/alice/alice_in_space.txt', fw, './data/alice/original_segmentation.txt')
    print(f" time of execution : {time.time() - start_time} seconds")

    ve_window_size = 7
    ve_threshold = 2
    start_time = time.time()
    print('---------')
    print(f"VotingExperts (window size:{ve_window_size}, threshold:{ve_threshold})")
    from pathlib import Path
    Path('./experiments/out/voting_experts').mkdir(parents=True, exist_ok=True)
    ve = VotingExperts( ve_window_size, ve_threshold, './experiments/out/voting_experts')
    get_results('./data/alice/alice_in_space.txt', ve, './data/alice/original_segmentation.txt')
    print(f" time of execution : {time.time() - start_time} seconds")
    
    tw_word_length = 17
    tw_word_frequency = 9
    start_time = time.time()
    print('---------')
    print(f"TopWords (word length:{tw_word_length}, word frequency:{tw_word_frequency})")
    Path('./experiments/out/top_words').mkdir(parents=True, exist_ok=True)
    tw = TopWORDS('./experiments/out/top_words', tw_word_length, tw_word_frequency)
    get_results('./data/alice/alice_in_space.txt', tw, './data/alice/original_segmentation.txt')
    print(f" time of execution : {time.time() - start_time} seconds")
