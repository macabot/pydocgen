"""
By Michael Cabot

Find source code belonging to docstrings
"""

import re
import os
import argparse
from collections import defaultdict
import preprocessing
import docfilters


def line_to_index(doc_path):
    """map lines in file to indexes of lines of which they occur"""
    sc_path = doc_path[:-4] + '.sc'
    line_map = defaultdict(list)
    with open(doc_path, 'r') as doc_in, \
            open(sc_path, 'r') as sc_in:
        for i, doc_line in enumerate(doc_in):
            sc_line = sc_in.next()
            doc_line = doc_line.decode('ascii', errors='ignore')
            sc_line = sc_line.decode('ascii', errors='ignore')
            line_map[(doc_line, sc_line)].append(i)
    return line_map

def write_lines_to_file(line_map, doc_in_path, out_path):
    """write line index to file according to line_map"""
    sc_in_path = doc_in_path[:-4] + '.sc'
    with open(doc_in_path, 'r') as doc_in, \
            open(sc_in_path, 'r') as sc_in, \
            open(out_path, 'w') as out:
        for doc_line in doc_in:
            sc_line  = sc_in.next()
            doc_line = doc_line.decode('ascii', errors='ignore')
            sc_line = sc_line.decode('ascii', errors='ignore')
            pair = (doc_line, sc_line)
            if pair in line_map:
                out.write('%s\n' % ','.join(str(x) for x in line_map[pair]))
            else:
                raise KeyError('unknown pair in %s:\n%s' % (doc_in_path, pair))

def number_data_split(total_path, train_path, tune_path, test_path, out_path):
    """get line index of train/tune/test set in total data"""
    assert os.path.isfile(total_path), 'unknown total_path: %s' % total_path
    assert os.path.isfile(train_path), 'unknown train_path: %s' % train_path
    assert os.path.isfile(tune_path), 'unknown tune_path: %s' % tune_path
    assert os.path.isfile(test_path), 'unknown test_path: %s' % test_path
    assert os.path.isdir(out_path), 'unknown out_path: %s' % out_path
    train_out_path = os.path.join(out_path, 'train_lines.txt')
    tune_out_path = os.path.join(out_path, 'tune_lines.txt')
    test_out_path = os.path.join(out_path, 'test_lines.txt')
    line_map = line_to_index(total_path)
    write_lines_to_file(line_map, train_path, train_out_path)
    write_lines_to_file(line_map, tune_path, tune_out_path)
    write_lines_to_file(line_map, test_path, test_out_path)

def clean_to_token(clean_path, doc_token_path, out_path):
    """map lines index of clean line in token file"""
    assert os.path.isfile(clean_path), 'unknown clean_path: %s' % clean_path
    assert os.path.isfile(doc_token_path), 'unknown token_path: %s' % doc_token_path
    out_folder, _tail = os.path.split(out_path)
    assert os.path.isdir(out_folder), 'unknown out_folder: %s' % out_folder
    sc_token_path = doc_token_path[:-4] + '.sc'
    with open(clean_path, 'r') as clean_in, \
            open(doc_token_path, 'r') as doc_token_in, \
            open(sc_token_path, 'r') as sc_token_in, \
            open(out_path, 'w') as out:
        clean_index = 0
        for i, doc_token_line in enumerate(doc_token_in):
            doc_token_line = doc_token_line.strip()
            sc_token_line = sc_token_in.next().strip()
            doc_token_words = len(doc_token_line.split())
            sc_token_words = len(sc_token_line.split())
            # max 100 words
            if doc_token_words > 100 or sc_token_words > 100:
                continue
            # max 1:9 word ratio
            if doc_token_words * 9 < sc_token_words or \
                    sc_token_words * 9 < doc_token_words:
                continue

            doc_clean_line = clean_in.next()
            clean_index += 1
            doc_token_line = re.sub(r'\W', '', doc_token_line)
            doc_clean_line = re.sub(r'\W', '', doc_clean_line)
            if doc_token_line == doc_clean_line:
                out.write('%d\n' % i)
            else:
                print 'different lines on token:%d, clean:%d' % (i, clean_index)
                print doc_token_line
                print doc_clean_line
                return

def all_to_filtered(doc_not_filtered_path, out_path, line_map_path):
    """map after filtering to before filtering"""
    assert os.path.isfile(doc_not_filtered_path), 'invalid doc_not_filtered_path: %s' % doc_not_filtered_path
    assert os.path.isdir(out_path), 'invalid out_path: %s' % out_path
    line_map_folder, name = os.path.split(line_map_path)
    assert os.path.isdir(line_map_folder), 'invalid line_map_folder: %s' % line_map_folder
    assert name.strip() != '', 'empty line_map_path name'
    filters = [docfilters.remove_doctests,
               docfilters.keep_first_description,
               docfilters.remove_wx_wrappers,
               docfilters.remove_parameter_descriptions,
               docfilters.replace_vertical_bars]
    preprocessing.preprocess(doc_not_filtered_path[:-4], out_path, False, filters, line_map_path)

def line_counter(path):
    """map line to the amount it occurs"""
    counter = defaultdict(int)
    with open(path, 'r') as in_file:
        for line in in_file:
            counter[line.strip()] += 1
    return counter

def same_lines(path1, path2):
    """check if two files have the same lines"""
    counts1 = line_counter(path1)
    counts2 = line_counter(path2)
    path1_lines = set(counts1.keys())
    path2_lines = set(counts2.keys())
    if counts1 == counts2:
        print 'they are equal'
    elif path1_lines == path2_lines:
        print 'equal lines, but different frequencies'
    else:
        path1_items = set(counts1.items())
        path2_items = set(counts2.items())
        print 'in path1 but not in path2'
        for item in path1_items - path2_items:
            print item
        print 'in path2 but not in path1'
        for item in path2_items - path1_items:
            print item

def raw_to_filtered(raw_path, filtered_path, out_path):
    """map raw to filtered"""
    assert os.path.isfile(raw_path), 'invalid raw_path: %s' % raw_path
    assert os.path.isfile(filtered_path), 'invalid filtered_path: %s' % filtered_path
    out_dir, out_name = os.path.split(out_path)
    assert os.path.isdir(out_dir), 'invalid out_dir: %s' % out_dir
    assert out_name.strip()!='', 'empty out_name'
    line_map = line_to_index(filtered_path)
    write_lines_to_file(line_map, raw_path, out_path)

def test_raw_to_filtered():
    """test raw_to_filtered"""
    raw_path = '../data/raw/docstring-filtered_sourcecode-NOcontext-NOfactors.doc'
    filtered_path = '../data/concat/filtered/concat_docstring-all_sourcecode-all-factors.doc'
    out_path = '../data/raw/raw2filtered.txt'
    raw_to_filtered(raw_path, filtered_path, out_path)

def test_same_lines():
    """test same_lines"""
    path1 = '../data/raw/docstring-filtered_sourcecode-NOcontext-NOfactors.doc'
    path2 = '../data/concat/filtered/concat_docstring-all_sourcecode-all-factors.doc'
    same_lines(path1, path2)

def test_all_to_filtered():
    """test all_to_filtered"""
    doc_not_filtered_path = '../data/concat/concat_docstring-all_sourcecode-all-factors.doc'
    out_path = '../data/concat/filtered/'
    line_map_path = '../data/concat/filtered/filtermap.txt'
    all_to_filtered(doc_not_filtered_path, out_path, line_map_path)

def test_clean_to_token():
    """test clean_to_token"""
    clean_path = '../data/clean/clean_docstring-filtered_sourcecode-NOcontext-NOfactors.tok.doc'
    doc_token_path = '../data/clean/docstring-filtered_sourcecode-NOcontext-NOfactors.tok.doc'
    out_path = '../data/clean/clean2token.txt'
    clean_to_token(clean_path, doc_token_path, out_path)

def test_number_data_split():
    """test number_data_split"""
    total_path = '../data/clean/clean_docstring-filtered_sourcecode-NOcontext-NOfactors.tok.doc'
    train_path = '../data/train/train_clean_docstring-filtered_sourcecode-NOcontext-NOfactors.tok.doc'
    tune_path = '../data/tune/tune_clean_docstring-filtered_sourcecode-NOcontext-NOfactors.tok.doc'
    test_path = '../data/test/test_clean_docstring-filtered_sourcecode-NOcontext-NOfactors.tok.doc'
    output_path = '../data/clean/'
    number_data_split(total_path, train_path, tune_path, test_path, output_path)

def main():
    """read commandin line arguments"""
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-total", "--total_path", required=True,
        help="All data")
    arg_parser.add_argument("-train", "--train_path", required=True,
        help="Train data")
    arg_parser.add_argument("-tune", "--tune_path", required=True,
        help="Tune data")
    arg_parser.add_argument("-test", "--test_path", required=True,
        help="Test data")
    arg_parser.add_argument("-o", "--output_path", required=True,
        help="Output path")

    args = arg_parser.parse_args()

    number_data_split(args.total_path, args.train_path, args.tune_path,
        args.test_path, args.output_path)


if __name__ == '__main__':
    #main()
    #test_number_data_split()
    #test_clean_to_token()
    #test_all_to_filtered()
    #test_same_lines()
    test_raw_to_filtered()
