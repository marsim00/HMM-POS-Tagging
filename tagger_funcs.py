# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 16:45:59 2021

@author: marb
"""

from collections import Counter
import numpy as np
from itertools import product

def get_train_data(train_corp):

    """
    Returns corpus data needed for training

    :param train_corp: training corpus with tagged POS as an object of a ConllCorpusReader class
    :return:
        tagset: list of all unique POS tags in the corpus (tag set)
        corp_pos: list of all POS tags in the corpus in the order of their appearance in the corpus
        corp_word_pos: corpus as a list of (word, POS) tuples
        wordset: list of all unique words in the corpus
        tagged_sents: corpus as a list of tagged sentences (each sentence is a list of (word, pos) tuples)
    """

    corp_word_pos = []
    corp_pos = []

    for tagged in train_corp.tagged_words():
        corp_word_pos.append((tagged)) #append (word, POS) tuple
        corp_pos.append(tagged[1]) #append POS tag only

    tagset = list(set(corp_pos))
    wordset = list(set(train_corp.words()))
    tagged_sents = train_corp.tagged_sents()

    return tagset, corp_pos, corp_word_pos, wordset, tagged_sents

def get_init(train_sents, tagset, N):

    """
    Calculates initial probabilities (probabilities of a tag being on the first position of the sentence and on the second in case of a trigram)

    :param train_sents: list with tagged sentences from a training corpus
    :param tagset: list of unique POS tags used in the corpus
    :param N: n-gram size for training (2 if bigram, 3 if trigram)
    :return: dictionary with initial probabilities for every POS
    """

    init_dict = {}

    for n in range(N - 1):
        init_list = [] #list to store all POS tags in the corpus being on the n position of the sentence
        init_dict[n] = {pos: 0 for pos in tagset} #initialization of a probability dictionary

        for s in train_sents:
            init_list.append(s[n][1]) #adding POS tag at a position n in every sentence in the corpus
        init_counter = Counter(init_list)

        #filling the dictionary with probabilities calculated as a proportion of sentences in the corpus containing a POS tag on the position n
        for pos in init_counter:
            init_dict[n][pos] = init_counter[pos] / len(train_sents)

    return init_dict

def get_trans(train_sents, tagset, N):

    """
    Calculates transition probabilities (probabilities of a tag being preceded by other tags (or tag pairs in case of a trigram))

    :param train_sents: list with tagged sentences from a training corpus
    :param tagset: list of unique POS tags used in the corpus
    :param N: n-gram size for training (2 if bigram, 3 if trigram)
    :return: dictionary with transition probabilities for preceding tags / tag pairs
    """

    ngrams_list = [] #initialization of a list to store tuples with all POS tags n-grams in the corpus

    #collecting n-grams occurring in each sentence
    for sent in train_sents:
        #add n-gram of POS tags into the list if sentence length not smaller than n
        if len(sent)>=N:
            sent_ngrams = [tuple(p[1] for p in sent[wp:wp+N]) for wp in range(len(sent)-N+1)]
            ngrams_list += sent_ngrams

    grams = list(product(tagset, repeat=N-1))  #creating a list of tuples of unique POS tags (bigrams) or all possible POS tag pairs (trigrams)
    trans_dict = {p0: {p1: 0 for p1 in tagset} for p0 in grams} #initialization of a dictionary with transition probabilities with preceding tags as keys, and dictionary with probabilities for every target tag as values

    first_pos_counted = Counter([bg[:(N-1)] for bg in ngrams_list]) #counts of all preceding contexts (n-1-grams) in n-grams
    ngrams_counted = Counter(ngrams_list) #counts of all n-grams

    #filling the dictionary with probabilities as n-gram counts divided by n-1-gram counts
    for p0 in trans_dict:
        for p1 in trans_dict[p0]:
           if first_pos_counted[p0]>0:
               trans_dict[p0][p1]= (ngrams_counted[(p0[:(N-1)]+(p1,))] / first_pos_counted[p0])

    return trans_dict


def get_emit(corp_word_pos, corp_pos, tagset, wordset):
    """
    Calculates emission probabilities (probability of a word being labelled with a certain POS tag)

    :param corp_word_pos: corpus as a list of (word, POS) tuples
    :param corp_pos: list of all POS tags in the corpus in the order of their appearance in the corpus
    :param tagset: list of unique POS tags used in the corpus
    :param wordset: list of all unique words in the corpus
    :return: dictionary with emission probabilities
    """

    wp_counted = Counter(corp_word_pos) #counts of (word, pos) pairs in the corpus
    pos_counted = Counter(corp_pos) #counts of all POS tags in the corpus

    wpe_dict = {w: {pos:0 for pos in tagset} for w in wordset} #initialization of a dictionary with emission probabilities with words as keys, and dictionary with probabilities for every POS tag as values

    #filling the dictionary with probabilities calculated as (word, POS) counts divided by POS counts
    for wp in wp_counted:
        wpe_dict[wp[0]][wp[1]] = (wp_counted[wp] / pos_counted[wp[1]])
    return wpe_dict

def tagger(input_sent, tagset, init_dict, trans_dict, emit_dict, wordset, N):

    """
    Yields best POS tags for word in the input sentence using HMM models and Viterbi algorithm.
    Abbreviations:
    - ip - initial probability
    - tp - transition probability
    - ep - emission probability
    - pp - probability of a previous state
    - cp - probability of a current state

    :param input_sent: list with the input sentence tokenized into words to tag
    :param tagset: list of unique POS tags used in the training corpus
    :param init_dict: dictionary with initial probabilities
    :param trans_dict: dictionary with transition probabilities
    :param emit_dict: dictionary with emission probabilities
    :param wordset: list of all unique words in the training corpus
    :param N: n-gram size for training (2 if bigram, 3 if trigram)
    :return: list of tags for each word in the input sentence
    """

    posgrams = list(product(tagset, repeat=N-1)) #a list of possible preceding states (tags if bigram, tags pairs if trigram)

    prob_matrix = np.zeros((len(posgrams), len(input_sent))) #initialization of a probability matrix with states as rows and words as columns
    best_path = [] #initialization of a list to store best states for every word (as tuples)

    #Algorithm initialization (filling the probability matrix for first N-1 words)
    for w_init in range(N-1):
        for state in range(len(posgrams)):
            if w_init < len(input_sent):
                #if a word exists in the training corpus, then its tag probability = ip * ep; otherwise only ip probability is assigned (ep = 1)
                if (input_sent[w_init]) in wordset:
                    prob_matrix[state, w_init] = init_dict[w_init][posgrams[state][w_init]] * emit_dict[input_sent[w_init]][posgrams[state][w_init]]
                else:
                    prob_matrix[state, w_init] = init_dict[w_init][posgrams[state][w_init]]

    #Execution of the algorithm starting from the Nth word
    for w in range(N-1, len(input_sent)):
        best_states = [] #a list to store states leading to the maximal cp
        best_probs = [] #a list to store maximal cp
        if (input_sent[w]) not in emit_dict:
            emit_dict[input_sent[w]] = {tag: 1 for tag in tagset}
        for state in range(len(posgrams)):
            emit_p = emit_dict[input_sent[w]][posgrams[state][N - 2]] #assigning ep for the current word and current POS tag

            V_t = [] #list to store cp = ep * pp * tp for each state
            prev_states = [pg for pg in posgrams if pg[1] == posgrams[state][0]] if N>2 else posgrams #a list with all possible previous states: if bigram - uniaque POS tags, if trigram - unique tag pairs where the second tag is the same as the first tag of a current state

            #obtaining cp calculated with pp and tp for all possible previous states
            for prev_pg in range(len(prev_states)):
                trans_p = trans_dict[prev_states[prev_pg]][posgrams[state][N-2]]
                prev_p = prob_matrix[posgrams.index(prev_states[prev_pg]), w-1]
                V_t.append(emit_p * prev_p * trans_p)

            prob_matrix[state, w] = np.max(V_t) #writing maximal cp in the matrix
            best_probs.append(np.max(V_t)) #storing maximal cp
            best_states.append(prev_states[np.argmax(V_t)]) #storing the previous state that led to the maximal cp

        best_path.append(best_states[np.argmax(best_probs)]) #storing the state with the maximal probability for the word

    best_path.append(posgrams[np.argmax(prob_matrix[:, len(input_sent) - 1])]) #storing the best current state for the last word

    #decoding
    tagged_sent = [tag[0] for tag in best_path[:-1]]
    tagged_sent += [tag for tag in best_path[-1]]

    return tagged_sent

