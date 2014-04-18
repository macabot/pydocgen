"""
By Michael Cabot

Functions for analysing data.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import os
import sys

import utils

def read_translation_freqs(file_name, num_lines=None):
    """Read the number of source translations"""
    translation_freqs = defaultdict(int)
    document = open(file_name, 'r')

    if num_lines == None:
        num_lines = sum(1 for line in open(file_name, 'r'))
    point = num_lines / 100 if num_lines > 100 else 1

    for i, line in enumerate(document):
        if i % point == 0:
            utils.show_progress(i, num_lines, 40, 'LOADING TRANSLATIONMODEL')

        segments = line.strip().split(' ||| ')
        source = segments[0]

        translation_freqs[source] += 1

    utils.show_progress(1, 1, 40, 'LOADING TRANSLATIONMODEL')
    sys.stdout.write('\n')
    document.close()

    return translation_freqs

def freqs_of_freqs(translation_freqs):
    """Count the number of times a source phrase has x translations"""
    freqs = defaultdict(int)
    for _source, freq in translation_freqs.iteritems():
        freqs[freq] += 1
    return freqs

def translation_distribution(path, num_lines=None):
    """plot the distribution of number of translations"""
    translation_freqs = read_translation_freqs(path, num_lines)
    print 'number of source phrases: %d' % len(translation_freqs)
    freqs = freqs_of_freqs(translation_freqs)

    under10 = sum(y for x, y in freqs.iteritems() if x < 10)
    print 'under10: %d' % under10

    x_values, y_values = zip(*freqs.items())
    total = sum(y_values)
    print 'total: %d' % total
    plt.plot(x_values, y_values, 'ro')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('number of translations')
    plt.ylabel('frequency')
    plt.title('distribution of number or source translations')
    #plt.savefig('../images/sourcetranslationdistribution.pdf')
    plt.show()

def test_translation_distribution():
    """test translation_distribution"""
    path = '../data/phrases/multicore/empty_phrases_NOfactors_alllines_7phraselength_all_info.txt'
    translation_distribution(path, 5486072)

def histogram(data, num_bins):
    """plot a histogram"""
    plt.hist(data, num_bins)
    plt.show()

def data_from_file(path):
    """read data from file and create a histogram"""
    assert os.path.isfile(path), 'invalid path: %s' % path
    data = []
    with open(path, 'r') as in_file:
        for line in in_file:
            tokens = line.strip().split()
            if len(tokens) > 0:
                data.append(float(tokens[0]))
    return data

def test_histogram_from_file():
    """test data_from_file and histogram"""
    path = '../data/confirm/line_score.txt'
    num_bins = 100
    data = data_from_file(path)
    data = [x for x in data if x != 0.0] # ignore 0.0
    histogram(data, num_bins)

def word_ratios(path_a, path_b):
    """For each line in file A and B calculate the ratio of words."""
    ratios = []
    with open(path_a, 'r') as file_a, open(path_b, 'r') as file_b:
        for line_a in file_a:
            line_b = file_b.next()
            words_a = line_a.split(' ')
            words_b = line_b.split(' ')
            ratios.append(float(len(words_a)) / len(words_b))

    return ratios

def test_average_ratio():
    """test word_ratios"""
    ratios = word_ratios('../data/nltk-develop.doc', '../data/nltk-develop.sc')
    print 'mean: %s' % np.mean(ratios)
    print 'deviation: %s' % np.std(ratios)

def plot_zipf(freqs, xlog=True, ylog=True):
    """Plot a zipf distribution of the frequencies.

    TODO should be called power distribution?"""
    # TODO label axes
    counts = Counter()
    for freq in freqs:
        counts[freq] += 1

    sorted_counts = sorted(counts.items())
    x_values, y_values = zip(*sorted_counts)
    plt.plot(x_values, y_values, 'ro')
    if xlog:
        plt.xscale('log')
    if ylog:
        plt.yscale('log')
    plt.show()

def test_plot_zipf():
    """Test plot_zipf"""
    with open('../data/django.binary_rules', 'r') as rule_file:
        freqs = []
        for line in rule_file:
            freq, _ = line.split(':')
            freqs.append(int(freq))

        plot_zipf(freqs)

def test_count_lines_with_extension():
    """Test count_lines_with_extension"""
    print utils.count_lines_with_extension('../data', '.doc')

def test_num_lines_with_extension():
    """Test num_lines_with_extension"""
    total = 0
    for num_lines, path in utils.num_lines_with_extension('../data', '.doc'):
        print '%s: %s' % (path, num_lines)
        total += num_lines
    print 'total: %s' % total

def parallel_length_map(sc_path, doc_path):
    """For each parallel line in sc and doc count the number of words.
    length_map maps the source code length to all encountered doc lengths"""
    length_map = {}
    with open(sc_path, 'r') as sc, open(doc_path, 'r') as doc:
        for sc_line in sc:
            sc_length = len(sc_line.strip().split())
            doc_length = len(doc.next().strip().split())
            if sc_length in length_map:
                length_map[sc_length].append(doc_length)
            else:
                length_map[sc_length] = [doc_length]
    return length_map

def merge_length_maps(length_maps):
    """Merge length maps"""
    merged_map = {}
    for length_map in length_maps:
        for sc_length, doc_length_values in length_map.iteritems():
            if sc_length in merged_map:
                merged_map[sc_length].extend(doc_length_values)
            else:
                merged_map[sc_length] = doc_length_values
    return merged_map

def plot_length_map(length_map, color):
    """Calculate the means and standard deviations of all doc length vaules and
    plot them with an errorbar."""
    # TODO label axes
    #length_items = sorted(length_map.items())
    length_items = length_map.items()
    sc_lengths, doc_length_values = zip(*length_items)
    doc_means = [np.mean(v) for v in doc_length_values]
    doc_stds = [np.std(v) for v in doc_length_values]
    #doc_len = [len(v) for v in doc_length_values]
    return plt.errorbar(sc_lengths, doc_means, doc_stds, marker='o', linestyle='', color = color)
    #plt.plot(sc_lengths, doc_len, 'ro')
    #return plt

def save_length_map(path, length_map):
    """save length map to file"""
    with open(path, 'w') as out:
        for sc_length, doc_lengths in length_map.iteritems():
            out.write('%s ||| %s\n' % (sc_length, ' '.join(str(doc) for doc in doc_lengths)))

def read_length_map(path):
    """read length map from file"""
    length_map = {}
    with open(path, 'r') as in_file:
        for line in in_file:
            segments = line.strip().split(' ||| ')
            sc_length = int(segments[0])
            doc_lengths = [int(doc) for doc in segments[1].split()]
            length_map[sc_length] = doc_lengths
    return length_map

def test_plot_length_map():
    """Test plot_length_map"""
    # raw
    # path = '../data/docstring-all_sourcecode-all-factors'
    # paths = ((sc_path, sc_path[:-3] + '.doc') for sc_path in utils.iter_files_with_extension(path, '.sc'))
    # length_maps = (parallel_length_map(sc_path, doc_path) for sc_path, doc_path in paths)
    # length_map = merge_length_maps(length_maps)
    # save_length_map('../data/raw_length_map.txt', length_map)
    length_map = read_length_map('../data/raw_length_map.txt')
    length_map = {k: v for k, v in length_map.iteritems() if k <= 100}
    p_raw = plot_length_map(length_map, 'r')
    #plt.title('raw data: average docstring words per source code words')
    # plt.savefig('../images/average-docstring-words_raw.pdf')
    # plt.show()

    # clean
    # base_path = '../data/clean/clean_docstring-filtered_sourcecode-NOcontext-NOfactors.tok'
    # length_map = parallel_length_map(base_path + '.sc', base_path + '.doc')
    # save_length_map('../data/clean_length_map.txt', length_map)
    length_map = read_length_map('../data/clean_length_map.txt')
    p_clean = plot_length_map(length_map, 'b')
    plt.xlim(1, 101)
    plt.xlabel('source code words')
    plt.ylabel('docstring words: mean and standard deviation')
    plt.title('docstring words per source code words')
    plt.legend([p_raw, p_clean], ['raw', 'clean'], numpoints = 1)
    plt.savefig('../images/docstring-sourcecode-words_raw_clean.pdf')
    plt.show()

def count_all_methods_and_functions(path):
    """Read all .sc filder in the given folder and count the number of methods
    and functions."""
    if os.path.isfile(path):
        assert os.path.splitext(path)[1] == '.sc', 'invalid source code file: %s' % path
        sc_paths = [path]
    elif os.path.isdir(path):
        print 'not yet implemented' # TODO
    else:
        raise ValueError('invalid path: %s' % path)

    sum_methods = 0
    sum_functions = 0
    for path in sc_paths:
        methods, functions = count_methods_and_functions(path)
        sum_methods += methods
        sum_functions += functions
    return sum_methods, sum_functions

def count_methods_and_functions(path):
    """Count the number of methods and functions in a file. Methods belong to a
    class."""
    methods = 0
    functions = 0
    with open(path, 'r') as sc_file:
        for line in sc_file:
            if line.split()[0].split('|')[4] == 'None':
                methods += 1
            else:
                functions += 1
    return methods, functions

def test_count_methods_and_functions():
    """test count(_all)_methods_and_functions"""
    path = '../data/docstring-all_sourcecode-all-factors/Python27-Lib.sc'
    print count_methods_and_functions(path)
    print count_all_methods_and_functions(path)

def get_repo_names(in_path, out_path):
    """get all repository names and sources"""
    with open(out_path, 'w') as out:
        for zip_path in utils.iter_files_with_extension(in_path, '.zip'):
            _folder, zip_name = os.path.split(zip_path)
            info_path = zip_path[:-4] + '.info'
            with open(info_path) as file_in:
                repo_source = file_in.next().strip()
            repo_name = zip_name[:-4]
            out.write(r'%s & \url{%s} \\' % (repo_name, repo_source))
            out.write('\n')

def test_get_repo_names():
    """test get_repo_names"""
    in_path = '../repos/'
    out_path = '../data/repo_names.txt'
    get_repo_names(in_path, out_path)


if __name__ == '__main__':
    #test_average_ratio()
    #test_plot_zipf()
    #test_count_lines_with_extension()
    #test_num_lines_with_extension()
    #test_plot_length_map()
    #test_count_methods_and_functions()
    #test_histogram_from_file()
    #test_translation_distribution()
    test_get_repo_names()
