"""
By Michael Cabot

Remove all extra white space
"""

import os
import re
import argparse

def strip_white_space(in_path, out_path):
    """Remove all extra white space from file"""
    assert os.path.isfile(in_path), 'invalid in_path: %s' % in_path
    out_folder, out_name = os.path.split(out_path)
    assert os.path.isdir(out_folder), 'invalid out_folder: %s' % out_folder
    assert out_name.strip() != '', 'empty out_name'
    pattern = re.compile(r'\s+')
    with open(in_path, 'r') as in_file, open(out_path, 'w') as out:
        for line in in_file:
            out.write(pattern.sub(' ', line.strip()) + '\n')

def main():
    """Read command line arguments"""
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-i", "--input", required=True,
        help="Input file")
    arg_parser.add_argument("-o", "--output", required=True,
        help="Output file")

    args = arg_parser.parse_args()
    strip_white_space(args.input, args.output)


if __name__ == '__main__':
    main()
