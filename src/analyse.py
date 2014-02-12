import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os

import trainer

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
    ratios = word_ratios('../data/nltk-develop.doc', '../data/nltk-develop.sc')
    print 'mean: %s' % np.mean(ratios)
    print 'deviation: %s' % np.std(ratios)

def plot_zipf(freqs):
    """Plot a zipf distribution of the frequencies.

    TODO should be called power distribution?"""
    counts = Counter()
    for freq in freqs:
        counts[freq] += 1

    sorted_counts = sorted(counts.items())
    x_values, y_values = zip(*sorted_counts)
    plt.plot(x_values, y_values, 'ro')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

def test_zipf():
    with open('../data/django.binary_rules', 'r') as rule_file:
        freqs = []
        for line in rule_file:
            freq, _ = line.split(':')
            freqs.append(int(freq))

        plot_zipf(freqs)

def count_lines_with_extension(path, extension):
    return sum(number_of_lines(p) for p in trainer.iter_files_with_extension(path, extension))
    
def num_lines_with_extension(path, extension):
    return ((number_of_lines(p), p) for p in trainer.iter_files_with_extension(path, extension))

def number_of_lines(path):
    with open(path, 'r') as file:
        return sum(1 for _line in file)
        
def test_count_lines_with_extension():
    print count_lines_with_extension('../data', '.doc')
    
def test_num_lines_with_extension():
    total = 0
    for num_lines, path in num_lines_with_extension('../data', '.doc'):
        print '%s: %s' % (path, num_lines)
        total += num_lines
    print 'total: %s' % total
        

if __name__ == '__main__':
    #test_average_ratio()
    #test_zipf()
    #test_count_lines_with_extension()
    test_num_lines_with_extension()