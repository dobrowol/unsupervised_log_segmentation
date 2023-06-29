from uls.golden_std_segmentation import voting_experts_not_pretrained, fixed_window_segmentation
from uls.golden_std_segmentation import time_window_segmentation, top_words_segmentation
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
    for _ in range(int(args.iterations)):
        next_point = optimizer.suggest(utility)
        ctr += 1
        window = int(next_point['window'])
        threshold = int(next_point['threshold'])
            
        print('Iteration {}/{} - Calculating for: window: {}, threshold: {}'.format(
            ctr, args.iterations, next_point['window'], next_point['threshold']
            ))

        if args.log:
            if not args.std:
                print("you need to specify golden standard file if you have chosen")
                return None

        target = voting_experts_not_pretrained(args.log, args.std, window, threshold, args.out_dir)
        try:
            result = 'Partial Result: {}; f(x)={}.'.format(next_point, target)
            print(result)
            logger.info(result) 
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

        target = fixed_window_segmentation(args.log, args.std, window, args.out_dir)
        try:
            result = 'Partial Result: {}; f(x)={}.'.format(next_point, target)
            print(result)
            logger.info(result) 
            out_file.write(result+"\n")
            optimizer.register(params=next_point, target=target)
        except:
            pass
    return 'Best result: {}; f(x)={}.'.format(optimizer.max['params'], optimizer.max['target'])

def bayes_for_time_window(args):
    optimizer = BayesianOptimization(f=None,
                    pbounds={
                            'time_window': [0.3, 40],
                            },
                    verbose=2, random_state=5385)

    utility = UtilityFunction(kind='ucb', kappa=1.96, xi=0.01)
    ctr = 0
    for _ in range(int(args.iterations)):
        next_point = optimizer.suggest(utility)
        ctr += 1
        time_window = int(next_point['time_window'])
            
        print('Iteration {}/{} - Calculating for: window: {}'.format(
            ctr, args.iterations, next_point['time_window']
            ))

        if args.log:
            if not args.std:
                print("you need to specify golden standard file if you have chosen")
                return None

        target = time_window_segmentation(args.log, args.std, time_window, args.out_dir)
        try:
            result = 'Partial Result: {}; f(x)={}.'.format(next_point, target)
            print(result)
            logger.info(result) 
            out_file.write(result+"\n")
            optimizer.register(params=next_point, target=target)
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
        target = top_words_segmentation(args.log, args.std, word_length, frequency_threshold, best_probability_threshold,
                                        0, args.out_dir)
        try:
            result = 'Partial Result: {}; f(x)={}.'.format(next_point, target)
            print(result)
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
    parser.add_argument('-log', type=str,
        help='drained logs file')
    parser.add_argument('-std', type=str,
        help='golden standard segmentation file')
    parser.add_argument('-fixed_window', action='store_true',
        help='fixed window segmentation')
    parser.add_argument('-top_words', action='store_true',
        help='top words segmentation')
    parser.add_argument('-voting_experts', action='store_true',
        help='top words segmentation')
    parser.add_argument('-out_dir', type=str,
        help='out directory')

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res_file_name = f'{str(out_dir)}/best_{args.iterations}_iterations_parameters.txt'
    Path(f'{str(out_dir)}').mkdir(exist_ok=True, parents=True)
    out_file = open(res_file_name, 'wt') 

    if args.voting_experts:
        result = bayes_for_voting_experts(args)
    elif args.fixed_window:
        result = bayes_for_fixed_window(args)
    elif args.top_words:
        result = bayes_for_top_words(args)

    logger.info(result)
    print(result)
    
    print(f"resuls saved to {res_file_name}")
    out_file.write(result)
    out_file.close()