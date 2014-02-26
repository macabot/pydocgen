"""
By Michael Cabot

Remove malformed utf-8 characters
"""

import os
import argparse
import codecs

def remove_malformed_characters(in_path, out_path):
    """Remove malformed utf-8 characters"""
    with codecs.open(in_path, 'r', encoding='utf-8', errors='ignore') as in_file, codecs.open(out_path, 'w', encoding='utf-8') as out:
        for line in in_file:
            try:
                out.write(line)
            except:
                print 'line: %d' % (i+1)
                raise

def main():
    """Read command line arguments to remove malformed utf-8 characters"""
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-i', '--input_path', required=True,
        help='Path to file')
    arg_parser.add_argument('-o', '--output_path', required=True,
        help='file name for output.')

    args = arg_parser.parse_args()

    input_path = args.input_path
    if not os.path.isfile(input_path):
        raise ValueError('Invalid input file: %s' % input_path)
    output_path = args.output_path
    output_folder, output_name = os.path.split(output_path)
    if not os.path.isdir(output_folder):
        raise ValueError('Invalid output folder: %s' % output_folder)
    if output_name.strip() == '':
        raise ValueError('Empty output name')

    remove_malformed_characters(input_path, output_path)


if __name__ == '__main__':
    main()