# POS Tagging with HMM and Viterbi Algorithm
Implementation of part of speech tagging using Hidden Markov Models and Viterbi algorithm (with bi- and trigrams).
The project is the assignment for the Computational Linguistics course on Language Science and Technology master's program, Saarland University.

## Project Structure

├── datasets\
│   ├── de-eval.tt <sub>(corpus with golden tags)</sub>\
│   ├── de-test.t            <sub>(input/test corpus)</sub>\
│   └── de-train.tt          <sub> (training corpus)</sub>\
├── output\
│   ├── bigram_tagging.eval  <sub> (bigram model results evaluation)</sub>\
│   ├── bigram_tagging.tt    <sub> (corpus tagged with a bigram model)</sub>\
│   ├── trigram_tagging.eval <sub> (trigram model results evaluation)</sub>\
│   └── trigram_tagging.tt   <sub> (corpus tagged with a trigram model)</sub>\
├── eval.py                  <sub> (evaluation script)</sub>\
├── execute_tagger.py       <sub>  (main script)</sub>\
└── tagger_funcs.py         <sub>(functions)</sub>

## Instructions for the execution in command line

The script takes training and input corpora in CONLL format as an input and assigns part of speech tags to words in the input corpus.

### Tag corpus: 
execute_tagger.py -t \<training corpus\> -i \<input corpus\> -o \<output filename\> -n \<n-gram size\>

*Example for the trigram model:*  execute_tagger.py -t de-train.tt -i de-test.t -o tagged_corpus.tt -n 3

Tagged corpus is written into the output folder.

### Evaluation of tagging results:
eval.py \<golden corpus\> \<tagged corpus\>

*Example:*  eval.py de-eval.tt tagged_corpus.tt

Evaluation results are written into the output folder.

### Notes
Data sets for training, testing and evaluation in German language are obtained from the [Universal Dependencies project](https://universaldependencies.org/). Evaluation script was provided by the Computational Linguistics course instructor and was partially modified for this project.
