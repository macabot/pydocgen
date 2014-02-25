"""
By Michael Cabot

Functions for analysing data.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os

import utils

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

def plot_zipf(freqs):
    """Plot a zipf distribution of the frequencies.

    TODO should be called power distribution?"""
    # TODO label axes
    counts = Counter()
    for freq in freqs:
        counts[freq] += 1

    sorted_counts = sorted(counts.items())
    x_values, y_values = zip(*sorted_counts)
    plt.plot(x_values, y_values, 'ro')
    plt.xscale('log')
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

def plot_length_map(length_map):
    """Calculate the means and standard deviations of all doc length vaules and
    plot them with an errorbar."""
    # TODO label axes
    length_items = sorted(length_map.items())
    sc_lengths, doc_length_values = zip(*length_items)
    doc_means = [np.mean(v) for v in doc_length_values]
    doc_stds = [np.std(v) for v in doc_length_values]
    doc_len = [len(v) for v in doc_length_values]
    plt.errorbar(sc_lengths, doc_means, doc_stds, marker='o', linestyle='')
    plt.plot(sc_lengths, doc_len, 'ro')
    plt.show()

def test_plot_length_map():
    """Test plot_length_map"""
    path = '../data/docstring-filtered_sourcecode-NOcontext-NOfactors'
    paths = ((sc_path, sc_path[:-3] + '.doc') for sc_path in utils.iter_files_with_extension(path, '.sc'))
    length_maps = (parallel_length_map(sc_path, doc_path) for sc_path, doc_path in paths)
    merged_map = merge_length_maps(length_maps)

    plot_length_map(merged_map)

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


if __name__ == '__main__':
    #test_average_ratio()
    #test_plot_zipf()
    #test_count_lines_with_extension()
    #test_num_lines_with_extension()
    test_plot_length_map()
    #test_count_methods_and_functions()
