"""
By Michael Cabot

Cumulatively segment de train data. For example, 10%, 20%, 30%, etc.
and train a translation model for each segment.
"""

import argparse
import os
import random
import sys
import time
from collections import defaultdict

import phrase_extract
import mp_worker

def shuffle_data(alignment_path, source_path, target_path):
    """Shuffle the order of the data"""
    tripples = []
    with open(alignment_path, 'r') as alignments_in, \
            open(source_path, 'r') as source_in, \
            open(target_path, 'r') as target_in:
        for alignment in alignments_in:
            source = source_in.next()
            target = target_in.next()
            tripples.append((alignment, source, target))
    random.shuffle(tripples)
    return tripples

def split_list(alist, wanted_parts=1):
    """split a list in a number of parts

    source: http://stackoverflow.com/a/752562"""
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts]
             for i in range(wanted_parts) ]

def data_to_file(data, output):
    """write data to file"""
    alignment_out_path = output + '.align'
    source_out_path = output + '.sc'
    target_out_path = output + '.doc'
    with open(alignment_out_path, 'w') as alignment_out, \
            open(source_out_path, 'w') as source_out, \
            open(target_out_path, 'w') as target_out:
        for alignment, source, target in data:
            alignment_out.write(alignment)
            source_out.write(source)
            target_out.write(target)

def get_frequencies(path, max_length, processes):
    """Calculate the phrase counts and lex counts"""
    alignments = path + '.align'
    source = path + '.sc'
    target = path + '.doc'
    max_lines = float('inf')
    start = time.time()
    if processes <= 1:
        phrase_freqs, lex_freqs, phrase_to_internals = \
            phrase_extract.extract_phrase_pair_freqs(alignments, source, target,
                                      max_length, max_lines)
    else:
        phrase_freqs, lex_freqs, phrase_to_internals = \
            mp_worker.set_up_workers(alignments, source, target,
                                     max_length,
                                     max_lines, processes, task_id=0)
    print 'time: %s' % (time.time() - start,)
    return phrase_freqs, lex_freqs, phrase_to_internals

def combine_int_dicts(dict_a, dict_b):
    """combine two integer defaultdicts"""
    new_dict = defaultdict(int)
    for key in set(dict_a.keys()) | set(dict_b.keys()):
        new_dict[key] = dict_a[key] + dict_b[key]
    return new_dict

def combine_set_dicts(dict_a, dict_b):
    """combine two dicts that have default values"""
    new_dict = {}
    for key in set(dict_a.keys()) | set(dict_b.keys()):
        if key in dict_a:
            value_a = dict_a[key]
        else:
            value_a = set([])
        if key in dict_b:
            value_b = dict_b[key]
        else:
            value_b = set([])
        new_dict[key] = value_a | value_b
    return new_dict

def calc_cumulative_freqs(frequencies):
    """cumulatively combine the frequencies"""
    cumulative_freqs = [((defaultdict(int),)*3,)*2 + ({},)]
    for (phrase_pair_freqs, source_phrase_freqs, target_phrase_freqs), \
            (lex_pair_freqs, source_lex_freqs, target_lex_freqs), \
            phrase_to_internals in frequencies:
        previous = cumulative_freqs[-1]
        # phrase freqs
        new_phrase_pair_freqs = combine_int_dicts(phrase_pair_freqs,
                                                  previous[0][0])
        new_source_phrase_freqs = combine_int_dicts(source_phrase_freqs,
                                                    previous[0][1])
        new_target_phrase_freqs = combine_int_dicts(target_phrase_freqs,
                                                    previous[0][2])
        new_phrase_freqs = (new_phrase_pair_freqs, new_source_phrase_freqs,
                            new_target_phrase_freqs)
        # lexical freqs
        new_lex_pair_freqs = combine_int_dicts(lex_pair_freqs,
                                               previous[1][0])
        new_source_lex_freqs = combine_int_dicts(source_lex_freqs,
                                                 previous[1][1])
        new_target_lex_freqs = combine_int_dicts(target_lex_freqs,
                                                 previous[1][2])
        new_lex_freqs = (new_lex_pair_freqs, new_source_lex_freqs,
                         new_target_lex_freqs)
        # internal alignments
        new_phrase_to_internals = combine_set_dicts(phrase_to_internals,
                                                    previous[2])
        # add to list
        cumulative_freqs.append((new_phrase_freqs, new_lex_freqs,
                                 new_phrase_to_internals))
    return cumulative_freqs[1:]

def continue_phrase_extract(freq, outputfile):
    """continue the calculation of the conditional probabilities and lexical
    weights"""
    phrase_freqs, lex_freqs, phrase_to_internals = freq

    phrase_pair_freqs, source_phrase_freqs, target_phrase_freqs = phrase_freqs
    lex_pair_freqs, source_lex_freqs, target_lex_freqs = lex_freqs
    phrase_extract.freqs_to_file(outputfile + '_extracted_phrases.txt',
                                 phrase_pair_freqs, source_phrase_freqs,
                                 target_phrase_freqs)
    phrase_extract.freqs_to_file(outputfile + '_extracted_lexwords.txt',
                                 lex_pair_freqs, source_lex_freqs,
                                 target_lex_freqs)

    # Calculating translation probabilities P(f|e) and P(e|f)
    phrase_source_given_target, phrase_target_given_source = \
        phrase_extract.conditional_probabilities(phrase_pair_freqs,
                                source_phrase_freqs,
                                target_phrase_freqs,
                                label='TRANSLATION PROBABILITIES',
                                logprob=True)
    phrase_extract.probs_to_file(outputfile + '_translation_probs_phrases.txt',
                  phrase_source_given_target, phrase_target_given_source)

    # Calculating lexical probabilities L(f|e) and L(e|f)
    lex_source_given_target, lex_target_given_source = \
        phrase_extract.conditional_probabilities(lex_pair_freqs,
                                source_lex_freqs,
                                target_lex_freqs,
                                label='LEXICAL PROBABILITIES',
                                logprob=True)
    phrase_extract.probs_to_file(outputfile + '_translation_probs_lex.txt',
                  lex_source_given_target, lex_target_given_source)

    # Calculating lexical weights l(f|e) and l(e|f)
    lex_weight_source_given_target, lex_weight_target_given_source = \
        phrase_extract.lexical_weights(phrase_to_internals,
                        lex_source_given_target,
                        lex_target_given_source, target_lex_freqs)
    phrase_extract.probs_to_file(outputfile + '_lexprobs_phrases.txt',
                  lex_weight_source_given_target,
                  lex_weight_target_given_source)

    # write final results to file
    phrase_extract.all_phrase_info_to_file(outputfile,
                            phrase_source_given_target,
                            phrase_target_given_source,
                            lex_weight_source_given_target,
                            lex_weight_target_given_source,
                            source_phrase_freqs,
                            target_phrase_freqs,
                            phrase_pair_freqs)

    sys.stdout.write('Done.\n')

def phrase_to_internals_to_file(path, phrase_to_internals):
    """write phrase_to_internals to file"""
    with open(path, 'w') as out:
        for phrase_pair, possible_internals in phrase_to_internals.iteritems():
            out.write('%s ||| %s ||| %s\n' % (phrase_pair[0], phrase_pair[1],
                            ' ||| '.join(str(p) for p in possible_internals)))

def increasing_data(alignments_path, source_path, target_path, output_path,
                    max_length, processes, parts):
    """Create translation models of different portions of the traindata"""
    # shuffle the data
    data = shuffle_data(alignments_path, source_path, target_path)
    # split the data
    parts = split_list(data, parts)
    names = []
    for i, part in enumerate(parts):
        name = output_path + '.part%d' % i
        names.append(name)
        data_to_file(part, name)
    # get frequencies of each part
    frequencies = []
    for name in names:
        freq = get_frequencies(name, max_length, processes)
        frequencies.append(freq)
        # write to file
        phrase_pair_freqs, source_phrase_freqs, target_phrase_freqs = freq[0]
        phrase_extract.freqs_to_file(name + '_temp_extracted_phrases.txt',
                                 phrase_pair_freqs, source_phrase_freqs,
                                 target_phrase_freqs)
        lex_pair_freqs, source_lex_freqs, target_lex_freqs = freq[1]
        phrase_extract.freqs_to_file(name + '_temp_extracted_lexwords.txt',
                                     lex_pair_freqs, source_lex_freqs,
                                     target_lex_freqs)
        phrase_to_internals_to_file(name + '_temp_aligments.txt', freq[2])
    # cumulatively combine the frequencies
    cumulative_freqs = calc_cumulative_freqs(frequencies)
    # calculate the conditional probabilities and lexical weights
    for i, freq in enumerate(cumulative_freqs):
        outputfile = 'tm_' + names[i]
        continue_phrase_extract(freq, outputfile)

def main():
    """Read command line arguments"""
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-a", "--alignments", required=True,
        help="File containing alignment")
    arg_parser.add_argument("-s", "--source", required=True,
        help="File containing sentences of source language")
    arg_parser.add_argument("-t", "--target", required=True,
        help="File containing sentences of target language")
    arg_parser.add_argument("-o", "--output", required=True,
        help="Output filename")
    arg_parser.add_argument("-mp", "--max_phrase_length", type=int,
        default=float('inf'), help="Maximum phrase pair length")
    arg_parser.add_argument('-pr', '--processes', type=int, default=1,
        help="Number of processes to use, default 1 (single process)")
    arg_parser.add_argument("-p", "--parts", required=True, type=int,
        help="Number of train data segments.")
    arg_parser.add_argument("-seed", "--seed", default=None,
        help="Seed for random module.")

    args = arg_parser.parse_args()

    random.seed(args.seed)
    alignments = args.alignments
    assert os.path.isfile(alignments), 'invalid alignment path: %s' % alignments
    source = args.source
    assert os.path.isfile(source), 'invalid source path: %s' % source
    target = args.target
    assert os.path.isfile(target), 'invalid target path: %s' % target
    outputfile = args.output
    out_dir, output_name = os.path.split(outputfile)
    assert os.path.isdir(out_dir), 'invalid output folder: %s' % out_dir
    assert output_name.strip() != '', 'empty output name'
    max_length = args.max_phrase_length
    processes = args.processes
    parts = args.parts

    increasing_data(alignments, source, target, outputfile, max_length,
                    processes, parts)

if __name__ == '__main__':
    main()
