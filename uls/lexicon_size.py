from uls.voting_experts.voting_experts import VotingExperts
from uls.topwords.topwords import TopWORDS
from uls.fixed_window import FixedWindow
from uls.golden_std_segmentation import get_splits_from_text_file

def build_lexicon_from_segmentation(file, segmentation):
    with open(file, "r") as inp_file:
        events = inp_file.read().rstrip('\n').split(' ')
    lexicon = set()
    prev_idx = 0
    for idx in segmentation:
        lexicon.add('_'.join(events[prev_idx:idx]))
        prev_idx = idx
    
    return lexicon

def get_lexicon(file, segmentation_method):
    segmentation = segmentation_method.fit_transform(file)
    lexicon = build_lexicon_from_segmentation(file, segmentation)
    return lexicon

if __name__ == '__main__':
    print("lexicon from original file")
    with open('./data/alice/original_segmentation.txt', "r") as original:
        original_lexicon = set(original.read().rstrip('\n').split(' '))
    print("lexicon from original file ", len(original_lexicon))
    fw = FixedWindow(0, 3)
    fixed_window_lexicon = get_lexicon('./data/alice/alice_in_space.txt', fw)
    print("fixed window lexicon size", len(fixed_window_lexicon))
    from pathlib import Path
    Path('./experiments/out/lexicon_size/voting_experts').mkdir(parents=True, exist_ok=True)
    ve = VotingExperts( 7, 2, './experiments/out/lexicon_size/voting_experts')
    voting_experts_lexicon = get_lexicon('./data/alice/alice_in_space.txt', ve)
    print("voting experts lexicon size", len(voting_experts_lexicon))

    Path('./experiments/out/lexicon_size/top_words').mkdir(parents=True, exist_ok=True)
    ve = TopWORDS('./experiments/out/lexicon_size/voting_experts', 7, 2, )
    voting_experts_lexicon = get_lexicon('./data/alice/alice_in_space.txt', ve)
    print("voting experts lexicon size", len(voting_experts_lexicon))