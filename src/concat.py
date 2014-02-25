"""
By Michael Cabot

Concatenate multiple parallel corpora
"""


import os
import argparse

import utils


def concat_corpus(input_path, output_path):
    """Concatenate multiple parallel corpora"""
    assert os.path.isdir(input_path), 'Invalid input folder: %s' % input_path
    output_folder, _output_name = os.path.split(output_path)
    assert os.path.isdir(output_folder), 'Invalid output folder: %s' % output_folder
    sc_out_path = output_path + '.sc'
    doc_out_path = output_path + '.doc'

    with open(sc_out_path, 'w') as sc_out, open(doc_out_path, 'w') as doc_out:
        for sc_in_path in utils.iter_files_with_extension(input_path, '.sc'):
            in_root, _ext = os.path.splitext(sc_in_path)
            print in_root
            doc_in_path = in_root + '.doc'
            with open(sc_in_path, 'r') as sc_in, open(doc_in_path, 'r') as doc_in:
                for sc_line in sc_in:
                    doc_line = doc_in.next()
                    sc_out.write(sc_line)
                    doc_out.write(doc_line)

def main():
    """Read command line arguments to concatenate parallel corpora"""
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-i', '--input_path', required=True,
        help='Path to folder containing parallel corpora')
    arg_parser.add_argument('-o', '--output_path', required=True,
        help='Path and name for output.')

    args = arg_parser.parse_args()

    input_path = args.input_path
    if not os.path.isdir(input_path):
        raise ValueError('Invalid input folder: %s' % input_path)
    output_path = args.output_path
    output_folder, output_name = os.path.split(output_path)
    if not os.path.isdir(output_folder):
        raise ValueError('Invalid output folder: %s' % output_folder)
    if output_name.strip() == '':
        raise ValueError('Empty output name')

    concat_corpus(input_path, output_path)

if __name__ == '__main__':
    main()