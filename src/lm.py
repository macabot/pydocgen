"""Create a Language Model
by Michael Cabot, Richard Rozeboom and Auke Wiggers"""

import argparse
import math
from collections import defaultdict


def extract_ngram_counts(corpus_name, max_n):
    """Extract ngram counts from a corpus"""
    corpus = open(corpus_name, 'r')
    ngram_counts = defaultdict(int)
    total_unigrams = 0
    for line in corpus:
        words = tuple(line.strip().split())
        for n in xrange(1, min(len(words), max_n) + 1):
            for ngram in ((words[i:i+n]) for i in xrange(len(words)-n+1)):
                ngram_counts[ngram] += 1
                if n == 1: # count number of unigrams
                    total_unigrams += 1

    corpus.close()
    return ngram_counts, total_unigrams

def counts_to_probs(ngram_counts, total_unigrams, logprob = True):
    """Convert ngram counts to probabilities"""
    ngram_probs = {}
    for ngram, count in ngram_counts.iteritems():
        if len(ngram) == 1:
            prob = float(count) / total_unigrams
        else:
            history_count = ngram_counts.get(ngram[:-1], 0)
            prob = float(count) / history_count

        if logprob:
            if prob != 0:
                prob = math.log(prob) 
            else:
                prob = -10000

        ngram_probs[' '.join(ngram)] = prob

    return ngram_probs

def create_lm(corpus, output, max_n):
    """Create a language model"""
    print 'corpus: %s' % corpus
    print 'output: %s' % output
    print 'max_n: %d' % max_n

    ngram_counts, total_unigrams = extract_ngram_counts(corpus, max_n)
    ngram_probs = counts_to_probs(ngram_counts, total_unigrams)
    ngrams_to_file(output+'.counts', ngram_counts)
    ngrams_to_file(output, ngram_probs)

def ngrams_to_file(file_name, ngrams):
    """Write ngrams and their values (freqs or probs) to file."""
    with open(file_name, 'w') as out:
        for ngram, value in ngrams.iteritems():
            out.write('{ngram} ||| {value}\n'.format(ngram=ngram, value=value))

def main():
    """Read command line arguments."""
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-c", "--corpus", required=True,
        help="File containing sentences")
    arg_parser.add_argument("-o", "--output", required=True,
        help="Output filename")
    arg_parser.add_argument("-n", "--max_n", required=True, type=int,
        help="Order of language model")

    args = arg_parser.parse_args()

    corpus = args.corpus
    output = args.output
    max_n = args.max_n

    create_lm(corpus, output, max_n)


if __name__ == '__main__':
    main()
