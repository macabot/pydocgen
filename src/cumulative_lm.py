"""
Cumulatively train a language model
"""

import argparse
import os
from collections import defaultdict

import lm
import data_size


def combine_language_models(lm_freqs):
    """cumulatively combine the language models"""
    cumulative_freqs = [(defaultdict(int), 0)]
    for ngram_counts, total_unigrams in lm_freqs:
        previous = cumulative_freqs[-1]
        new_ngram_counts = data_size.combine_int_dicts(ngram_counts,
                                                       previous[0])
        new_total_unigrams = total_unigrams + previous[1]
        cumulative_freqs.append((new_ngram_counts, new_total_unigrams))
    return cumulative_freqs[1:]

def create_language_models(corpus_path, output_path, min_range, max_range,
                           max_n):
    """create the cumulative language models"""
    corpus_dir, corpus_name = os.path.split(corpus_path)
    output_dir, output_name = os.path.split(output_path)
    # read the language models
    lm_freqs = []
    for i in xrange(min_range, max_range+1):
        corpus_path = os.path.join(corpus_dir, corpus_name.replace('*', str(i)))
        lm_freqs.append(lm.extract_ngram_counts(corpus_path, max_n))
    # cumulatively combine the language models
    cumulative_freqs = combine_language_models(lm_freqs)
    # finish the language models
    for i, (ngram_counts, total_unigrams) in enumerate(cumulative_freqs):
        ngram_probs = lm.counts_to_probs(ngram_counts, total_unigrams)
        output = os.path.join(output_dir, output_name.replace('*', str(i)))
        lm.ngrams_to_file(output+'.counts', ngram_counts)
        lm.ngrams_to_file(output, ngram_probs)

def main():
    """read command line arguments"""
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-c", "--corpus_path", required=True,
        help="path and prefix to corpus parts")
    arg_parser.add_argument("-o", "--output_path", required=True,
        help="output path for language models")
    arg_parser.add_argument("-min", "--min_range", required=True, type=int,
        help="min range")
    arg_parser.add_argument("-max", "--max_range", required=True, type=int,
        help="max range")
    arg_parser.add_argument("-n", "--max_n", required=True, type=int,
        help="Order of language model")
    args = arg_parser.parse_args()

    corpus_path = args.corpus_path
    corpus_dir, corpus_name = os.path.split(corpus_path)
    assert os.path.isdir(corpus_dir), 'invalid corpus_dir: %s' % corpus_dir
    assert corpus_name.strip()!='', 'empty corpus_name'
    assert corpus_name.count('*')==1, 'corpus_name must contain one *'
    output_path = args.output_path
    output_dir, output_name = os.path.split(output_path)
    assert os.path.isdir(output_dir), 'invalid output_dir: %s' % output_dir
    assert output_name.strip()!='', 'empty corpus_name'
    assert output_name.count('*')==1, 'output_name must contain one *'

    create_language_models(corpus_path, output_path, args.min_range,
                           args.max_range, args.max_n)


if __name__ == '__main__':
    main()
