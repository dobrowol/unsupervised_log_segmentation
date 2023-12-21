from uls.golden_std_segmentation import voting_experts_f1_score, voting_experts_text_f1_score
from uls.golden_std_segmentation import top_words_f1_score, top_words_text_f1_score
from uls.golden_std_segmentation import fixed_window_f1_score, fixed_window_text_f1_score
from uls.voting_experts.voting_experts import VotingExperts
from bayes_opt import BayesianOptimization, UtilityFunction
import argparse
import logging
from pathlib import Path
import sys

sys.setrecursionlimit(100000)
logger = logging.getLogger(__name__)
global DATASET_PATH

def bayes_for_voting_experts(args):
    optimizer = BayesianOptimization(f=None,
                    pbounds={
                            'window': [3, 40],
                            'threshold': [1, 10],
                            },
                    verbose=2, random_state=5385)

    utility = UtilityFunction(kind='ucb', kappa=1.96, xi=0.01)
    ctr = 0

    ve = VotingExperts(40, 20, out_directory=args.out_dir, drains=args.drains)
    ve.retrieve_tree()
    ve.fit(args.training_file)
    for _ in range(int(args.iterations)):
        next_point = optimizer.suggest(utility)
        ctr += 1
        window = int(next_point['window'])
        threshold = int(next_point['threshold'])
            
        print('Iteration {}/{} - Calculating for: window: {}, threshold: {}'.format(
            ctr, args.iterations, next_point['window'], next_point['threshold']
            ))

        if args.training_file:
            if not args.std:
                print("you need to specify golden standard file if you have chosen")
                return None
        if not args.text:
            target = voting_experts_f1_score(args.training_file, args.testing_file, args.std, window, threshold, args.out_dir, voting_experts=ve)
        else:
            target = voting_experts_text_f1_score(args.training_file, args.std, window, threshold, args.out_dir, voting_experts=ve)
        try:
            result = 'Partial Result: {}; f(x)={}.'.format(next_point, target)
            logger.info(result) 
            print(result)
            out_file.write(result+"\n")
            optimizer.register(params=next_point, target=target)
        except:
            pass
    return 'Best result: {}; f(x)={}.'.format(optimizer.max['params'], optimizer.max['target'])

def bayes_for_fixed_window(args):
    optimizer = BayesianOptimization(f=None,
                    pbounds={
                            'window': [3, 40],
                            },
                    verbose=2, random_state=5385)

    utility = UtilityFunction(kind='ucb', kappa=1.96, xi=0.01)
    ctr = 0
    for _ in range(int(args.iterations)):
        next_point = optimizer.suggest(utility)
        ctr += 1
        window = int(next_point['window'])
            
        print('Iteration {}/{} - Calculating for: window: {}'.format(
            ctr, args.iterations, next_point['window']
            ))

        if args.log:
            if not args.std:
                print("you need to specify golden standard file if you have chosen")
                return None
        max_target = 0
        best_alignment = 0
        for alignment in range(window):
            if not args.text:
                target = fixed_window_f1_score(args.log, args.std, alignment, window)
            else:
                target = fixed_window_text_f1_score(args.log, args.std, alignment, window)
            if target > max_target:
                max_target = target
                best_alignment = alignment
            
        try:
            result = 'Partial Result: {};alignment: {}, f(x)={}.'.format(next_point, best_alignment, target)
            logger.info(result) 
            out_file.write(result+"\n")
            optimizer.register(params=next_point, target=max_target)
        except:
                pass
            
    return 'Best result: {}; f(x)={}.'.format(optimizer.max['params'], optimizer.max['target'])


def bayes_for_top_words(args):
    optimizer = BayesianOptimization(f=None,
                    pbounds={
                            'word_length': [5,20],
                            'frequency_threshold': [2, 15]
                            },
                    verbose=2, random_state=5385)

    utility = UtilityFunction(kind='ucb', kappa=1.96, xi=0.01)
    ctr = 0
    for _ in range(int(args.iterations)):
        next_point = optimizer.suggest(utility)
        ctr += 1
        word_length = int(next_point['word_length'])
        frequency_threshold = int(next_point['frequency_threshold'])
            
        print('Iteration {}/{} - Calculating for: word_length: {}, frequency_threshold: {}'.format(
            ctr, args.iterations, next_point['word_length'], next_point['frequency_threshold']
            ))

        if args.log:
            if not args.std:
                print("you need to specify golden standard file if you have chosen")
                return None
        best_probability_threshold = 1.0*10**(-6)
        if not args.text:
            target = top_words_f1_score(args.log, args.std, word_length, frequency_threshold, best_probability_threshold,
                                        0, args.out_dir)
        else:
            target = top_words_text_f1_score(args.log, args.std, word_length, frequency_threshold, best_probability_threshold,
                                        0, args.out_dir)
        try:
            result = 'Partial Result: {}; f(x)={}.'.format(next_point, target)
            logger.info(result) 
            out_file.write(result+"\n")
            optimizer.register(params=next_point, target=target)
        except:
            pass
    return 'Best result: {}; f(x)={}.'.format(optimizer.max['params'], optimizer.max['target'])
    
if __name__ == '__main__':
    logging.basicConfig(filename='best_segmentation.log', encoding='utf-8', level=logging.INFO)
    parser = argparse.ArgumentParser(
        description='Tool to find best segmentation parameters using Bayesian Optimization.')
    parser.add_argument('iterations', type=int,
        help='Number of iteration of bayes optimization')
    parser.add_argument('-training_file', type=str,
        help='drained logs file')
    parser.add_argument('-testing_file', type=str,
        help='drained logs file')
    parser.add_argument('-std', type=str,
        help='golden standard segmentation file')
    parser.add_argument('-fixed_window', action='store_true',
        help='fixed window segmentation')
    parser.add_argument('-top_words', action='store_true',
        help='top words segmentation')
    parser.add_argument('-voting_experts', action='store_true',
        help='voting experts segmentation')
    parser.add_argument('-text', action='store_true',
        help='text file segmentation')
    parser.add_argument('-drains', action='store_true', default=False,
        help='text file segmentation')
    parser.add_argument('-out_dir', type=str,
        help='out directory')

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res_file_name = f'{str(out_dir)}/best_{args.iterations}_iterations_parameters.txt'
    Path(f'{str(out_dir)}').mkdir(exist_ok=True, parents=True)
    out_file = open(res_file_name, 'wt') 

    import time

    start_time = time.time()
    if args.voting_experts:
        result = bayes_for_voting_experts(args)
    elif args.fixed_window:
        result = bayes_for_fixed_window(args)
    elif args.top_words:
        result = bayes_for_top_words(args)
    logger.info(f"Hyperparameter tuning took {time.time() - start_time} seconds")
    print(f"Hyperparameter tuning took {time.time() - start_time} seconds")
    logger.info(result)
    print(result)
    
    print(f"resuls saved to {res_file_name}")
    out_file.write(result)
    out_file.close()