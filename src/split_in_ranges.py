"""
By Michael Cabot

Split a corpus into ranges according to the number of words in the source
sentences.
"""
import os
import argparse
import re
from collections import defaultdict
from itertools import izip


def read_scdoc_by_sclen(sc_path, doc_path):
    """read source-codes and docstrings and group by words in source code"""
    groups = defaultdict(list)
    with open(sc_path, 'r') as sc_in, \
            open(doc_path, 'r') as doc_in:
        for i, (sc_line, doc_line) in enumerate(izip(sc_in, doc_in)):
            sc_amount = len(sc_line.strip().split())
            groups[sc_amount].append((i, sc_line, doc_line))
    return groups


def read_split_ranges(path):
    """read split ranges from file"""
    pattern = re.compile(r'^(\d+)-(\d+)')
    split_ranges = []
    with open(path, 'r') as file_in:
        for line in file_in:
            result = pattern.match(line)
            split_ranges.append((int(result.group(1)), int(result.group(2))))
    return split_ranges

def split_groups_by_ranges(groups, ranges):
    """split groups by ranges"""
    for start, stop in ranges:
        split_group = []
        for i in xrange(start, stop+1):
            if i in groups:
                split_group.extend(groups[i])
        yield split_group, (start, stop)

def write_groups_to_file(split_groups, out_dir, sc_name, doc_name,
                         line_map_name = None):
    """write group to file"""
    sc_root, sc_ext = os.path.splitext(sc_name)
    doc_root, doc_ext = os.path.splitext(doc_name)
    if line_map_name:
        line_map_root, line_map_ext = os.path.splitext(line_map_name)
    for group, (start, stop) in split_groups:
        sc_group_name = sc_root + '_%d-%d' % (start, stop) + sc_ext
        doc_group_name = doc_root + '_%d-%d' % (start, stop) + doc_ext
        sc_path = os.path.join(out_dir, sc_group_name)
        doc_path = os.path.join(out_dir, doc_group_name)
        if line_map_name:
            line_map_group_name = line_map_root + '_%d-%d' % (start, stop) + \
                                  line_map_ext
            line_map_path = os.path.join(out_dir, line_map_group_name)
        else:
            line_map_path = None
        write_group_to_file(group, sc_path, doc_path, line_map_path)

def write_group_to_file(group, sc_path, doc_path, line_map_path = None):
    """write hypothesis and reference lines of group to file"""
    with open(sc_path, 'w') as sc_out, \
            open(doc_path, 'w') as doc_out:
        if line_map_path:
            line_map_out = open(line_map_path, 'w')
        for i, sc_line, doc_line in group:
            sc_out.write(sc_line)
            doc_out.write(doc_line)
            if line_map_path:
                line_map_out.write('%d\n' % i)
        if line_map_path:
            line_map_out.close()

def main():
    """read command line arguments"""
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-sc', '--source_code', required=True,
        help='Path to source code')
    arg_parser.add_argument('-doc', '--docstring', required=True,
        help='Path to docstrings')
    arg_parser.add_argument('-sr', '--split_ranges', required=True,
        help='Path to split_ranges')
    arg_parser.add_argument('-o', '--output', required=True,
        help='Dir for output')
    arg_parser.add_argument('--line_map_name', default=None,
        help='Path to write extra file containing indexes of original file.')
    args = arg_parser.parse_args()

    sc_path = args.source_code
    doc_path = args.docstring
    split_ranges = args.split_ranges
    out_dir = args.output
    line_map_name = args.line_map_name
    assert os.path.isfile(sc_path), 'invalid source code path: %s' % sc_path
    assert os.path.isfile(doc_path), 'invalid docstring path: %s' % doc_path
    assert os.path.isfile(split_ranges), 'invalid split_ranges path: %s' % split_ranges
    assert os.path.isdir(out_dir), 'invalid output: %s' % out_dir
    if line_map_name != None:
        assert line_map_name.strip()!='', 'empty line map name'

    ranges = read_split_ranges(split_ranges)
    groups = read_scdoc_by_sclen(sc_path, doc_path)
    split_groups = split_groups_by_ranges(groups, ranges)
    _sc_dir, sc_name = os.path.split(sc_path)
    _doc_dir, doc_name = os.path.split(doc_path)
    write_groups_to_file(split_groups, out_dir, sc_name, doc_name,
                         line_map_name)


if __name__ == '__main__':
    main()
