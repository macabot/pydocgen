"""
By Michael Cabot

Assert if two language models are the same
"""
import argparse
import os

from decoder import read_language_model


def compare(lm1, lm2):
    """compare two language models"""
    # find difference
    lm1_items = set(lm1.items())
    lm2_items = set(lm2.items())
    if lm1_items == lm2_items:
        print 'they are equal'
    else:
        print 'in lm1 but not in lm2'
        only_lm1_items = lm1_items - lm2_items
        for item in only_lm1_items:
            print item
        print 'in lm2 but not in lm1'
        only_lm2_items = lm2_items - lm1_items
        for item in only_lm2_items:
            print item

def main():
    """read command line arguments"""
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-lm1", "--lm1_path", required=True,
        help="lm1")
    arg_parser.add_argument("-lm2", "--lm2_path", required=True,
        help="lm2")
    arg_parser.add_argument("-mpl", "--max_phrase_length", type=int,
        default=3, help="Limit decoding to using phrases up to n words")
    args = arg_parser.parse_args()
    lm1_path = args.lm1_path
    lm2_path = args.lm2_path
    assert os.path.isfile(lm1_path), 'invalid lm1_path: %s' % lm1_path
    assert os.path.isfile(lm2_path), 'invalid lm2_path: %s' % lm2_path

    lm1 = read_language_model(lm1_path, args.max_phrase_length)
    lm2 = read_language_model(lm2_path, args.max_phrase_length)
    compare(lm1, lm2)


if __name__ == '__main__':
    main()
