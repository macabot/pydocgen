"""
By Michael Cabot

represent functions and their sources
"""

import re
import argparse
import os

class InfoCorpus:
    """holds all info files in a corpus"""

    def __init__(self, info_files):
        self.info_files = info_files

    @staticmethod
    def read(paths):
        """create an infocorpus from files"""
        info_files = []
        for path in paths:
            info_files.append(InfoFile.read(path))
        return InfoCorpus(info_files)

    def write(self, path):
        """write infocorpus to file"""
        with open(path, 'w') as out:
            for info_file in self.info_files:
                out.write('%r' % info_file)

    def set_concat_indexes(self, index = 0):
        """add concatenation index"""
        for info_file in self.info_files:
            index = info_file.set_concat_indexes(index)

    def subset(self, concat_indexes):
        """"return an infocorpus containing a subset of the infofunctions
        with the given concat_indexes"""
        concat_indexes = set(concat_indexes)
        subset_files = [info_file.subset(concat_indexes)
                        for info_file in self.info_files]
        return InfoCorpus(subset_files)

class InfoFile:
    """holds the content of an info file"""

    def __init__(self, zip_path, info_functions):
        self.zip_path = zip_path
        self.info_functions = info_functions

    @staticmethod
    def read(path):
        """create an infofile from a file"""
        zip_path_pattern = re.compile(r'^<zip_path>(\S+)</zip_path>$')
        start_count_pattern = re.compile(r'^<count:(\d+)>$')
        end_count_pattern = re.compile(r'^</count:(\d+)>$')

        info_functions = []
        with open(path, 'r') as in_file:
            zip_path_string = in_file.next()
            zip_path_match = zip_path_pattern.search(zip_path_string)
            if zip_path_match == None:
                raise ValueError('file %s should start with its zip_path: %s' %
                                 (path, zip_path_string))
            else:
                zip_path = zip_path_match.group(1)
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
                    info_functions.append(InfoFunction(count,
                                                            temp_lines[1:]))
                    temp_lines = []
                if count_open:
                    temp_lines.append(line)
        return InfoFile(zip_path, info_functions)

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

    def subset(self, concat_indexes):
        """return a subset"""
        concat_indexes = set(concat_indexes)
        subset_functions = [function for function in self.info_functions
                            if function.concat_index in concat_indexes]
        return InfoFile(self.zip_path, subset_functions)


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
                self.concat_index = int(concat_index_match.group(1))
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

def read_concat_indexes(path):
    """read concat_indexes from file"""
    with open(path, 'r') as in_file:
        return [int(line.strip()) for line in in_file]

def main():
    """read command line arguments"""
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-c', '--corpus', required=True,
        help='Path to corpus')
    arg_parser.add_argument('-i', '--indexes', required=True,
        help='Path to concat indexes')
    arg_parser.add_argument('-o', '--out', required=True,
        help='Path for output corpus')
    arg_parser.add_argument('-s', '--set_indexes', action='store_true',
        default=False, help='Set the concat indexes in the corpus')
    args = arg_parser.parse_args()
    corpus = args.corpus
    indexes = args.indexes
    out = args.out
    assert os.path.isfile(corpus), 'invalid corpus path: %s' % corpus
    assert os.path.isfile(indexes), 'invalid indexes path: %s' % indexes
    out_dir, out_name = os.path.split(out)
    assert os.path.isdir(out_dir), 'invalid out_dir: %s' % out_dir
    assert out_name.strip()!='', 'empty out_name'
    info_corpus = InfoCorpus.read([corpus])
    if args.set_indexes:
        info_corpus.set_concat_indexes()
    concat_indexes = read_concat_indexes(indexes)
    subset_info_corpus = info_corpus.subset(concat_indexes)
    subset_info_corpus.write(out)
    

if __name__ == '__main__':
    main()

