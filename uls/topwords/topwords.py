# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 23:15:07 2017

@author: zcw2218
"""
from collections import Counter
import pandas as pd
from decimal import *
from uls.topwords.LimitStack import *
from uls.topwords.segtree import *
from uls.topwords.DPcache import *
from tqdm import tqdm
pd.set_option('display.precision', 18)

class TopWORDS:
    def __init__(self, out_dir, word_length, word_frequency, prob_threshold = 1.0*10**(-6), word_boundary= 0):
        self.out_dir = out_dir
        self.word_length = word_length
        self.word_frequency = word_frequency
        self.prob_threshold = prob_threshold
        self.word_boundary = word_boundary

    def Dictionary(self, texts):
        permutations = []
        for text in texts:
            for i in range(1,self.word_length+1):
                for j in (range(len(text))):
                    if j+i <= len(text):
                        permutations.append('_'.join(text[j:j+i]))
        
        cnt = Counter(permutations)
        puredict = {k:v for k,v in cnt.items() if len(k.split('_')) == 1 or v >= self.word_frequency}
        sumcount = sum(list(puredict.values()))
        puredict.update((k,Decimal(v/sumcount)) for k,v in puredict.items())
        puredict = {k:v for k,v in puredict.items() if len(k.split('_')) == 1 or v >= self.prob_threshold}
        sumPrunedFreq = sum(list(puredict.values()))
        puredict.update((k,Decimal(v/sumPrunedFreq)) for k,v in puredict.items())
        return puredict 

    def DPLikelihoodsBackward(self, T,dictionary):
        #backward likelihoods: P(T_[>=m]|D,\theta)
        likelihoods = [Decimal(0)]*(len(T)+1)
        likelihoods[len(T)] = Decimal(1)
        #dynamic programming from text tail to head
        for m in tqdm(range(len(T)-1,-1,-1)):

            tLimit = self.word_length if (m + self.word_length < len(T)) else (len(T) - m)
            sum = 0
            for t in range(1,tLimit+1):

                candidateWord = '_'.join(T[m:m+t])
                if candidateWord in dictionary.keys():
                    sum += Decimal(dictionary[candidateWord]*likelihoods[m+t])
                else:
                    sum = sum
            likelihoods[m] = sum
        return likelihoods
                
    def DPLikelihoodsForward(self, T,dictionary):
        #forward likelihoods: P(T_[<=m]|D,\theta)
            likelihoods = [Decimal(0)]*(len(T)+1)
            likelihoods[0] = Decimal(1)
            #print("dynamic programming from text head to tail")
            #dynamic programming from text head to tail
            for m in tqdm(range(1,len(T)+1)):
                tLimit = self.word_length if (m-self.word_length >= 0) else m
                sum = 0
                for t in range(1,tLimit+1):
                    candidateWord = '_'.join(T[m-t:m])
                    if candidateWord in dictionary.keys():
                        sum += Decimal(dictionary[candidateWord]*likelihoods[m-t])
                    else:
                        sum = sum 
                likelihoods[m] = sum
            
            return likelihoods
                
                
    def DPExpectations(self, T,dictionary,likelihoods):
        #expectations of word use frequency: n_i(T_[>=m])
        niTs = DPcache(dictionary)
        riTs = DPcache(dictionary)
        
        #dynamic programming from text tail to head
        #print("dynamic programming expectations")
        for m in tqdm(range(len(T)-1,-1,-1)):
            tLimit = self.word_length if (m + self.word_length < len(T)) else (len(T) - m)
            # get all possible cuttings for T_m with one word in head and rest in tail
            cuttings = []
            for t in range(1,tLimit+1):
                candidateWord = '_'.join(T[m:m+t])
                if candidateWord in dictionary.keys():
                    rho = Decimal(dictionary[candidateWord]*likelihoods[m+t]/likelihoods[m])
                    cuttings.append([candidateWord,t,rho])

            niTs.pushn(cuttings, self.word_length)
            #riTs.pushr(cuttings,taul)

        return (list(map(lambda x,y:[x,y],niTs.top().keys(),niTs.top().values())))      
                
    def updateDictionary(self, texts, dictionary):
    #calculating the likelihoods (P(T|theta)) and expectations (niS and riS)
        likelihoodsum = 0
        expectation = []
        count = 0
        for text in texts:
            likelihoods = self.DPLikelihoodsBackward(text,dictionary)
            likelihoodsum += likelihoods[0]
            count += 1
            expectation.extend(self.DPExpectations(text,dictionary,likelihoods))
            
        expectations = pd.DataFrame(expectation,columns=['word','nis'])
        nis = expectations['nis'].groupby(expectations['word']).sum().reset_index()
        niSum = expectations['nis'].sum()
        nis['nis'] = nis['nis'].apply(lambda x : Decimal(x/niSum))
        thetaS = dict(zip(nis.word,nis.nis))
        #ris = expectations[expectations['word'].apply(lambda x: len(x) > 1)].reset_index(drop=True)
        #print(ris.sort_values('ris',ascending = False).head(5))
        #ris['s'] = ris['ris'].apply(lambda x: - math.log(Decimal(1)- x))
        #ris = ris['ris'].groupby(ris['word']).sum().reset_index().sort_values('ris',ascending=False) 
        #phiS = dict(zip(ris.word,ris.ris))
        avglikelihood = likelihoodsum/count
        
        return (thetaS,avglikelihood)
        
    def pruneDictionary(self, thetaS):
    # prune thetaS by use probability threshold
        prunedThetaS =  {k:v for k,v in thetaS.items() if len(k.split('_')) == 1 or v >= self.prob_threshold}
        sumPrunedWordTheta = sum(list(prunedThetaS.values()))
        prunedThetaS.update((k,Decimal(v/sumPrunedWordTheta)) for k,v in prunedThetaS.items())
        
        return (prunedThetaS)

    def PESegment(self, texts, dictionary, out_file):
        fo = open(out_file,"a",encoding='utf-8')
        for text in texts:
            #calculating the P(T|theta) forwards and backwards respectively
            forwardLikelihoods = self.DPLikelihoodsForward(text,dictionary)
            backwardLikelihoods = self.DPLikelihoodsBackward(text,dictionary)
            #calculating the boundary scores of text
            boundaryScores = [forwardLikelihoods[k]*backwardLikelihoods[k]/backwardLikelihoods[0] for k in range(1,len(text)+1)]
            segments = []
            start_idx = 0
            if self.word_boundary > 0:
                for item in self.TextSegmentor(text, boundaryScores):
                    segments.append((start_idx, start_idx + len(item)))
                    start_idx += len(item)+1
            else:
                for item in SegmentTree(text,boundaryScores,dictionary).listleaf(text,boundaryScores):
                    #index after which we should segment
                    segments.append(start_idx + len(item))
                    start_idx += len(item)
                    fo.write(f"{''.join(item)} ")
        fo.close()
        return segments
                
    def TextSegmentor(self, T, boundaryScores):
        boundaryindex  = [i for i,j in enumerate(boundaryScores) if j >= self.word_boundary]
        #return text itself if it has only one character
        if len(T) <= 1:
            return T.split()
        splitResult = self.lindexsplit(T,*boundaryindex)
        return splitResult

    
    def lindexsplit(self, some_list, *args):
        if args:
            args = (0,) + tuple(data+1 for data in args) + (len(some_list)+1,)

        my_list = []
        for start, end in zip(args, args[1:]):
            my_list.append('_'.join(some_list[start:end]))
        return my_list


    def fit_transform(self, runtime_file):
        ###initialization
        from pathlib import Path
        out_file = Path(self.out_dir)/f"segmentation_wl_{self.word_length}_wf_{self.word_frequency}.txt"
        if out_file.is_file():
            print(out_file)
            return out_file
        # preprocess the input corpus
        with open(runtime_file, "r") as texts_file:
            #texts = [['c', 'h', 'a', 'p', 't', 'e', 'r', 'i', 'd', 'o']]
            texts = texts_file.read().rstrip('\n').split(' ')
            if texts[-1] == '':
                texts = texts[0:-1]
                print(f"there are {len(texts)} drained templates")
            texts = [texts]
            
        #generate the overcomplete dictionary
        dictionary = self.Dictionary(texts)
        #initialize the loop variables
        iteration = 1
        converged = False
        lastLikelihood = -1.0
        convergeTol = 1.0*10**(-3)
        numIterations = 1
        ###EM loop
        while(converged != True and iteration <= numIterations):
            #update and prune the dictionary
            (thetaS,avglikelihood) = self.updateDictionary(texts,dictionary)
            dictionary = self.pruneDictionary(thetaS)
            # info of current iteration
            print("Iteration {0}, likelihood:{1}, dictionary:{2}".format(iteration,avglikelihood,len(dictionary)))
            # test the convergence condition
            if lastLikelihood > 0 and abs((avglikelihood - lastLikelihood)/lastLikelihood <convergeTol):
                converged = True
            # prepare for the next iteration
            lastLikelihood = avglikelihood
            iteration += 1

        return self.PESegment(texts,dictionary, out_file)
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-runtime_file", type=str)
    parser.add_argument("-out_dir", type=str)
    parser.add_argument("-word_len", type=int)
    parser.add_argument("-threshold", type=int)
    parser.add_argument("-prob_threshold", type=float)
    args = parser.parse_args()
    segments = segmentation(args.runtime_file, args.out_dir, args.word_len, args.threshold, args.prob_threshold)
    print(f"log sequence segmented into {len(segments)} segments")
    print(f"last segment in segmentation is {segments[-1]}")
    out_file = f"{args.out_dir}/topwordsseg_{args.word_len}_{args.threshold}.log"
    import pickle
    with open(out_file, "wb") as out:
        pickle.dump(segments, out)
    
