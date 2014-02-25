"""
By Michael Cabot

Split a corpus in training set, tuning set and test set.
"""

import os
import random
import argparse

import utils

def split_corpus(in_path, out_path, tune_lines, test_lines, select_random):
    """Split a corpus in a train, tune and test set"""
    sc_in_path = in_path + '.sc'
    doc_in_path = in_path + '.doc'
    assert os.path.isfile(sc_in_path), 'invalid input file: %s' % sc_in_path
    assert os.path.isfile(doc_in_path), 'invalid input file: %s' % doc_in_path
    out_folder, out_name = os.path.split(out_path)
    assert os.path.isdir(out_folder), 'invalid output folder: %s' % out_folder
    assert out_name != '', 'empty output name'

    sc_train_out_path = os.path.join(out_folder, 'train_' + out_name + '.sc')
    doc_train_out_path = os.path.join(out_folder, 'train_' + out_name + '.doc')
    sc_tune_out_path = os.path.join(out_folder, 'tune_' + out_name + '.sc')
    doc_tune_out_path = os.path.join(out_folder, 'tune_' + out_name + '.doc')
    sc_test_out_path = os.path.join(out_folder, 'test_' + out_name + '.sc')
    doc_test_out_path = os.path.join(out_folder, 'test_' + out_name + '.doc')

    full_lines = utils.number_of_lines(sc_in_path)
    assert full_lines >= tune_lines + test_lines, 'file contains %d lines\n %d < %d + %d' % (full_lines, full_lines, tune_lines, test_lines)

    full_numbers = range(full_lines)
    if select_random:
        random.shuffle(full_numbers)
    tune_numbers = set(full_numbers[full_lines - tune_lines - test_lines: full_lines - test_lines])
    test_numbers = set(full_numbers[full_lines - test_lines:])

    with open(sc_in_path, 'r') as sc_in, open(doc_in_path, 'r') as doc_in, \
            open(sc_train_out_path, 'w') as sc_train_out, \
            open(doc_train_out_path, 'w') as doc_train_out, \
            open(sc_tune_out_path, 'w') as sc_tune_out, \
            open(doc_tune_out_path, 'w') as doc_tune_out, \
            open(sc_test_out_path, 'w') as sc_test_out, \
            open(doc_test_out_path, 'w') as doc_test_out:
        for i, sc_line in enumerate(sc_in):
            doc_line = doc_in.next()
            if i in tune_numbers:
                sc_tune_out.write(sc_line)
                doc_tune_out.write(doc_line)
            elif i in test_numbers:
                sc_test_out.write(sc_line)
                doc_test_out.write(doc_line)
            else:
                sc_train_out.write(sc_line)
                doc_train_out.write(doc_line)

def main():
    """Read command line arguments for split a corpus"""
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-i', '--input_path', required=True,
        help='Path to folder containing parallel corpus')
    arg_parser.add_argument('-o', '--output_path', required=True,
        help='Path of folder for output')
    arg_parser.add_argument('-l', '--lines', required=True, nargs=2,
        help='Number of lines for tune and test set')
    arg_parser.add_argument('-r', '--select_random', action='store_true',
        default = False, help='Select random lines for the tune and test set')
    arg_parser.add_argument('-s', '--random_seed', default = None,
        help='Seed for selecting random lines')

    args = arg_parser.parse_args()

    in_path = args.input_path
    out_path = args.output_path
    tune_lines, test_lines = int(args.lines[0]), int(args.lines[1])
    select_random = args.select_random
    random.seed(args.random_seed)
    split_corpus(in_path, out_path, tune_lines, test_lines, select_random)


if __name__ == '__main__':
    main()



