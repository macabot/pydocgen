"""
By Michael Cabot

Find the location of blank factors
e.g.:
|factor2|...|factorN
"""

import os
import argparse

import utils

def find_blank_factors(path, one_match):
    """find blank factors"""
    assert os.path.isdir(path), 'invalid folder: %s' % path
    for sc_in_path in utils.iter_files_with_extension(path, '.sc'):
        with open(sc_in_path, 'r') as sc_file:
            for i, line in enumerate(sc_file):
                factors = line.strip().split()
                for factor in factors:
                    if factor[0] == '|':
                        print sc_in_path
                        print 'line: %d' % (i+1)
                        print line
                        print factor
                        if one_match:
                            return
    print 'done'

def main():
    """read command line arguments for finding blank factors"""
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-i', '--input_path', required=True,
        help='Path to folder containing parallel corpora')
    arg_parser.add_argument('-b', '--one_match', action='store_true',
        default = False, help='Break on first match')

    args = arg_parser.parse_args()

    input_path = args.input_path
    if not os.path.isdir(input_path):
        raise ValueError('Invalid input folder: %s' % input_path)
    one_match = args.one_match

    find_blank_factors(input_path, one_match)

if __name__ == '__main__':
    main()