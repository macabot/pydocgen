"""Create a Language Model
by Michael Cabot, Richard Rozeboom and Auke Wiggers"""

import argparse
import math
from collections import Counter


def extract_ngram_counts(corpus_name, max_n):
    """Extract ngram counts from a corpus"""
    corpus = open(corpus_name, 'r')
    ngram_counts = Counter()
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
            prob = math.log(prob)

        ngram_probs[' '.join(ngram)] = prob

    return ngram_probs

def create_lm(corpus, output, max_n):
    """Create a language model"""
    print 'corpus: %s' % corpus
    print 'output: %s' % output
    print 'max_n: %d' % max_n
    
    ngram_counts, total_unigrams = extract_ngram_counts(corpus, max_n)
    ngram_probs = counts_to_probs(ngram_counts, total_unigrams)
    dict_to_file(output+'.counts', ngram_counts, "%s ||| %s\n")
    dict_to_file(output, ngram_probs, "%s ||| %s\n")

def dict_to_file(file_name, dictionary, string_format = '%s: %s\n', 
                 key_format = '%s', value_format = '%s'):
    """Write a dictionary to file by formatting its keys and values"""
    out = open(file_name, 'w')
    any_key, any_value = dictionary.iteritems().next()
    # determine key type
    if isinstance(any_key, str) or isinstance(any_key, int) or \
            isinstance(any_key, float):
        key_type = 0
    elif key_format.count('%s') == 1:
        key_type = 1
    else:
        key_type = 2

    # determine value type
    if isinstance(any_value, str) or isinstance(any_value, int) or \
            isinstance(any_value, float):
        value_type = 0
    elif value_format.count('%s') == 1:
        value_type = 1
    else:
        value_type = 2

    for key, value in dictionary.iteritems():
        if key_type == 0:
            key_string = key_format % key
        elif key_type == 1:
            key_string = key_format % (key,)
        else:
            key_string = key_format % tuple(key)

        if value_type == 0:
            value_string = value_format % value
        elif value_type == 1:
            value_string = value_format % (value,)
        else:
            value_string = value_format % tuple(value)

        out.write(string_format % (key_string, value_string))

    out.close()

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
