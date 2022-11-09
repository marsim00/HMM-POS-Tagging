"""

"""
from tagger_funcs import *
import nltk.corpus.reader as ncr
import argparse
from tqdm import tqdm
from time import time
from os import path

def arguments():
    parser = argparse.ArgumentParser(prog="HMM Part of Speech Tagger with Viterbi Algorithm")
    parser.add_argument("-t","--train", help="a file with a training corpus in Conll format")
    parser.add_argument("-i","--input", help="a file with a corpus for tagging in Conll format")
    parser.add_argument("-o","--output", help= "a file to write a tagged corpus into")
    parser.add_argument("-n", "--ngram", help="n-gram size used for training, 2 (default) or 3", default="2", type=int, choices=[2,3])
    args = parser.parse_args()
    return args


def tag_corpus(train_file, input_file, output_file, n):

    """
    Writes a tagged corpus.

    :param train_file: training corpus filename (*.tt)
    :param input_file: input corpus filename (*.t)
    :param output_file: output corpus filename (*.tt)
    :param n: n-gram size
    """

    print("Start training")
    print("Initiating variables")
    start = time()

    train_corp = ncr.ConllCorpusReader("datasets", train_file, ('words', 'pos'))
    input_corp = ncr.ConllCorpusReader("datasets", input_file, ('words', ))

    pos_coll, pos_list, word_pos_list, words_coll, sents = get_train_data(train_corp)

    print("Calculating probabilities")
    pos_init = get_init(sents, pos_coll, N=n)
    pos_trans = get_trans(sents, pos_coll, N=n)
    word_pos_emit = get_emit(word_pos_list, pos_list, pos_coll, words_coll)

    end = time()
    print("Training finished in", end - start, "s")

    print("Tagging")
    with open(path.join("output", output_file), "w", encoding="utf-8") as tagged_corp:
        for raw_sent in tqdm(input_corp.sents()):
            sent_tags = tagger(input_sent=raw_sent, tagset=pos_coll, init_dict=pos_init, trans_dict=pos_trans, emit_dict=word_pos_emit, wordset=words_coll, N=n)
            tagged_sent = zip(raw_sent, sent_tags)
            for wt in tagged_sent:
                tagged_corp.write(str(wt[0]) + "\t" + str(wt[1]) + "\n")
            tagged_corp.write("\n")

    print("Finished")


if __name__ == "__main__":
    tagger_args = arguments()
    tag_corpus(tagger_args.train, tagger_args.input, tagger_args.output, tagger_args.ngram)