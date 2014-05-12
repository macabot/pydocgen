"""
By Michael Cabot

Concatenate multiple parallel corpora
"""


import os
import argparse
import re

import utils


class InfoFile:
    """holds the content of an info file"""

    def __init__(self, path):
        """"construct an InfoFile from a file"""
        zip_path_pattern = re.compile(r'^<zip_path>(\S+)</zip_path>$')
        start_count_pattern = re.compile(r'^<count:(\d+)>$')
        end_count_pattern = re.compile(r'^</count:(\d+)>$')

        self.info_functions = []
        with open(path, 'r') as in_file:
            zip_path_string = in_file.next()
            zip_path_match = zip_path_pattern.search(zip_path_string)
            if zip_path_match == None:
                raise ValueError('file %s should start with its zip_path: %s' %
                                 (path, zip_path_string))
            else:
                self.zip_path = zip_path_match.group(1)
            count = None
            temp_lines = []
            count_open = False
            for line in in_file:
                start_count_match = start_count_pattern.search(line)
                end_count_match = end_count_pattern.search(line)
                if start_count_match != None:
                    count = int(start_count_match.group(1))
                    count_open = True
                elif end_count_match != None:
                    assert count == int(end_count_match.group(1)), 'counts do not align'
                    count_open = False
                    self.info_functions.append(InfoFunction(count,
                                                            temp_lines[1:]))
                    temp_lines = []
                if count_open:
                    temp_lines.append(line)

    def __repr__(self):
        return '<zip_path>' + self.zip_path + '</zip_path>\n' + \
               ''.join('%r' % function for function in self.info_functions)

    def write(self, path):
        """write this object to file"""
        with open(path, 'w') as out:
            out.write('%r' % self)

    def set_concat_indexes(self, index):
        """add concatenation index"""
        for i, info_function in enumerate(self.info_functions):
            info_function.concat_index = index + i
        return index + len(self.info_functions)


class InfoFunction:
    """holds the content of a function"""

    def __init__(self, count, lines):
        """construct an InfoFunction"""
        path_pattern = re.compile(r'^<path>(\S+)</path>$')
        lineno_pattern = re.compile(r'^<lineno>(\d+)</lineno>$')
        start_code_pattern = re.compile(r'^<code>$')
        end_code_pattern = re.compile(r'^</code>$')
        concat_index_pattern = re.compile(r'<concat_index>(\d+)</concat_index>')

        self.count = count
        self.path = None
        self.lineno = None
        self.code = None
        self.concat_index = None
        in_code = False
        code_lines = []
        for line in lines:
            path_match = path_pattern.search(line)
            lineno_match = lineno_pattern.search(line)
            start_code_match = start_code_pattern.search(line)
            end_code_match = end_code_pattern.search(line)
            concat_index_match = concat_index_pattern.search(line)
            if path_match != None:
                self.path = path_match.group(1)
            elif lineno_match != None:
                self.lineno = int(lineno_match.group(1))
            elif start_code_match != None:
                in_code = True
            elif end_code_match != None:
                in_code = False
                self.code = ''.join(code_lines[1:])
                code_lines = []
            elif concat_index_match != None:
                self.concat_index = concat_index_match.group(1)
            if in_code:
                code_lines.append(line)

    def __repr__(self):
        path_string = lineno_string = concat_index_string = code_string = ''
        if self.path != None:
            path_string = '<path>%s</path>\n' % self.path
        if self.lineno != None:
            lineno_string = '<lineno>%d</lineno>\n' % self.lineno
        if self.concat_index != None:
            concat_index_string = '<concat_index>%d</concat_index>\n' % self.concat_index
        if self.code != None:
            code_string = '<code>\n%s</code>\n' % self.code
        return '<count:%d>\n' % self.count + path_string + lineno_string + \
               concat_index_string + code_string + '</count:%d>\n' % self.count


def concat_corpus(input_path, output_path):
    """Concatenate multiple parallel corpora"""
    assert os.path.isdir(input_path), 'Invalid input folder: %s' % input_path
    output_folder, output_name = os.path.split(output_path)
    assert os.path.isdir(output_folder), 'Invalid output folder: %s' % output_folder
    assert output_name.strip()!='', 'empty output name'
    sc_out_path = output_path + '.sc'
    doc_out_path = output_path + '.doc'

    with open(sc_out_path, 'w') as sc_out, \
            open(doc_out_path, 'w') as doc_out:
        for sc_in_path in utils.iter_files_with_extension(input_path, '.sc'):
            in_root, _ext = os.path.splitext(sc_in_path)
            print in_root
            doc_in_path = in_root + '.doc'
            with open(sc_in_path, 'r') as sc_in, \
                    open(doc_in_path, 'r') as doc_in:
                for sc_line in sc_in:
                    doc_line = doc_in.next()
                    sc_out.write(sc_line)
                    doc_out.write(doc_line)

def concat_info_corpus(input_path, output_path):
    """concatenate multiple info files"""
    assert os.path.isdir(input_path), 'Invalid input folder: %s' % input_path
    output_folder, output_name = os.path.split(output_path)
    assert os.path.isdir(output_folder), 'Invalid output folder: %s' % output_folder
    assert output_name.strip()!='', 'empty output name'

    info_files = (InfoFile(in_path) for in_path in 
                  utils.iter_files_with_extension(input_path, '.info'))

    with open(output_path, 'w') as out:
        index = 0
        for info_file in info_files:
            index = info_file.set_concat_indexes(index)
            print info_file.zip_path
            out.write('%r' % info_file)

def main():
    """Read command line arguments to concatenate parallel corpora"""
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-i', '--input_path', required=True,
        help='Path to folder containing parallel corpora')
    arg_parser.add_argument('-o', '--output_path', required=True,
        help='Path and name for output.')
    arg_parser.add_argument('-info', '--is_info', action='store_true',
        default=False,
        help='Use when concatenating .info files')

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

    if args.is_info:
        concat_info_corpus(input_path, output_path)
    else:
        concat_corpus(input_path, output_path)

def test():
    """test infofile"""
    path = '../data/docstring-all_sourcecode-all-factors/ansible-devel.info'
    info_file = InfoFile(path)
    info_file.write('../temp/copy_of_ansible-devel.info')


if __name__ == '__main__':
    main()
    #test()
