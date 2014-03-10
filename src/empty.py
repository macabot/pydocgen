"""
By Michael Cabot

Add empty string translations
"""

import os
from collections import defaultdict
import argparse
import sys

import phrase_extract
from utils import show_progress

def read_freqs(path, label = 'FREQS'):
    """Read freqs from an _extracted_lexwords.txt or an _extracted_phrases.txt
    file"""
    phrase_pair_freqs = defaultdict(int)
    source_freqs = defaultdict(int)
    target_freqs = defaultdict(int)

    num_lines = sum(1 for line in open(path, 'r'))
    point = num_lines / 100 if num_lines > 100 else 1

    with open(path, 'r') as in_file:
        for i, line in enumerate(in_file):
            if i % point == 0:
                show_progress(i, num_lines, 40, label)

            source, target, freqs = line.strip().split(' ||| ')
            source_freq, target_freq, pair_freq = [int(x) for x in freqs.split()]
            phrase_pair_freqs[(source, target)] = pair_freq
            source_freqs[source] = source_freq
            target_freqs[target] = target_freq

    show_progress(1, 1, 40, label)
    sys.stdout.write('\n')

    return phrase_pair_freqs, source_freqs, target_freqs

def read_lexical_weights(path):
    """Read the lexical weights from an _all_info.txt file"""
    lex_weight_source_given_target = {}
    lex_weight_target_given_source = {}

    num_lines = sum(1 for line in open(path, 'r'))
    point = num_lines / 100 if num_lines > 100 else 1

    with open(path, 'r') as in_file:
        for i, line in enumerate(in_file):
            if i % point == 0:
                show_progress(i, num_lines, 40, 'LOADING LEXICAL WEIGHTS')

            source, target, probs, _freqs = line.strip().split(' ||| ')
            _pfe, _pef, lfe, lef = [float(x) for x in probs.split()]
            lex_weight_source_given_target[(source, target)] = lfe
            lex_weight_target_given_source[(source, target)] = lef

    show_progress(1, 1, 40, 'LOADING LEXICAL WEIGHTS')
    sys.stdout.write('\n')

    return lex_weight_source_given_target, lex_weight_target_given_source

def add_null_translations(translation_freqs, lex_freqs):
    """add NULL translations as empty translations"""
    phrase_pair_freqs, source_freqs, target_freqs = translation_freqs
    lex_phrase_pair_freqs, lex_source_freqs, lex_target_freqs = lex_freqs
    for (source, target), lex_pair_freq in lex_phrase_pair_freqs.iteritems():
        if target == 'NULL':
            phrase_pair_freqs[(source, '')] += lex_pair_freq
            source_freqs[source] += lex_source_freqs[source]
            target_freqs[''] += lex_target_freqs[target]

def add_empty_lexical_weights(lex_weight_source_given_target,
                              lex_weight_target_given_source,
                              phrase_source_given_target,
                              phrase_target_given_source):
    """Add lexical weights for empty translations"""
    for phrase_pair, pfe in phrase_source_given_target.iteritems():
        if phrase_pair[1] == '':
            lex_weight_source_given_target[phrase_pair] = pfe
            pef = phrase_target_given_source[phrase_pair]
            lex_weight_target_given_source[phrase_pair] = pef

def add_empty_translations(phrase_freqs_path, lex_freqs_path, phrase_all_path,
                           output_path):
    """Add empty translations"""
    translation_freqs = read_freqs(phrase_freqs_path, 'TRANSLATION FREQS')
    lex_freqs = read_freqs(lex_freqs_path, 'LEXICAL FREQS')
    lexical_weights = read_lexical_weights(phrase_all_path)

    add_null_translations(translation_freqs, lex_freqs)
    phrase_pair_freqs, source_phrase_freqs, target_phrase_freqs = translation_freqs
    phrase_source_given_target, phrase_target_given_source = \
        phrase_extract.conditional_probabilities(phrase_pair_freqs,
                                  source_phrase_freqs,
                                  target_phrase_freqs,
                                  label='TRANSLATION PROBABILITIES',
                                  logprob=True)
    add_empty_lexical_weights(lexical_weights[0],
                              lexical_weights[1],
                              phrase_source_given_target,
                              phrase_target_given_source)
    phrase_extract.all_phrase_info_to_file(output_path,
                            phrase_source_given_target,
                            phrase_target_given_source,
                            lexical_weights[0],
                            lexical_weights[1],
                            source_phrase_freqs,
                            target_phrase_freqs,
                            phrase_pair_freqs)

def main():
    """Read command line arguments and extract phrases."""
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-f", "--phrase_freqs", required=True,
        help="File phrase pair frequencies")
    arg_parser.add_argument("-l", "--lex_freqs", required=True,
        help="File lex frequencies")
    arg_parser.add_argument("-a", "--phrase_all", required=True,
        help="File lex frequencies")
    arg_parser.add_argument("-o", "--output", required=True,
        help="Output filename")

    args = arg_parser.parse_args()

    phrase_freqs_path = args.phrase_freqs
    assert os.path.isfile(phrase_freqs_path), 'invalid phrase_freqs_path: %s' % phrase_freqs_path
    lex_freqs_path = args.lex_freqs
    assert os.path.isfile(lex_freqs_path), 'invalid lex_freqs_path: %s' % lex_freqs_path
    output_path = args.output
    output_folder, output_name = os.path.split(output_path)
    assert os.path.isdir(output_folder), 'invalid output_folder: %s' % output_folder
    assert output_name.strip() != '', 'empty output_name'
    phrase_all_path = args.phrase_all
    assert os.path.isfile(phrase_all_path), 'invalid phrase_all_path: %s' % phrase_all_path

    add_empty_translations(phrase_freqs_path, lex_freqs_path, phrase_all_path, output_path)


if __name__ == '__main__':
    main()
