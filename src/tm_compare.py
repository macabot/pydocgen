"""
By Michael Cabot

Assert if two translation models are the same
"""
import argparse
import os

from pattern_search import read_full_translation_model


def compare(tm1, tm2):
    """compare two translation models"""
    # find difference
    tm1_items = set([(s, tp) for s, t in tm1.iteritems() for tp in t])
    tm2_items = set([(s, tp) for s, t in tm2.iteritems() for tp in t])
    if tm1_items == tm2_items:
        print 'they are equal'
    else:
        print 'in tm1 but not in tm2'
        only_tm1_items = tm1_items - tm2_items
        for item in only_tm1_items:
            print item
        print 'in tm2 but not in tm1'
        only_tm2_items = tm2_items - tm1_items
        for item in only_tm2_items:
            print item

def main():
    """read command line arguments"""
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-tm1", "--tm1_path", required=True,
        help="TM1")
    arg_parser.add_argument("-tm2", "--tm2_path", required=True,
        help="TM2")
    arg_parser.add_argument("-mpl", "--max_phrase_length", type=int,
        default=3, help="Limit decoding to using phrases up to n words")
    args = arg_parser.parse_args()
    tm1_path = args.tm1_path
    tm2_path = args.tm2_path
    assert os.path.isfile(tm1_path), 'invalid tm1_path: %s' % tm1_path
    assert os.path.isfile(tm2_path), 'invalid tm2_path: %s' % tm2_path

    tm1 = read_full_translation_model(tm1_path, args.max_phrase_length)
    tm2 = read_full_translation_model(tm2_path, args.max_phrase_length)
    compare(tm1, tm2)


if __name__ == '__main__':
    main()
