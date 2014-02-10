import zipfile
import os
import ast
from collections import Counter
import matplotlib.pyplot as plt

from ast_plus import ASTPlus
from filters import remove_doctests


def create_parallel_corpus(in_path, out_path, filters = None, max_count = float('inf')):
    """For all python files in all zip files, align rules in an AST with its
    docstring."""
    if os.path.isdir(in_path):
        zip_paths = iter_files_with_extension(in_path, '.zip')
    elif os.path.isfile(in_path):
        zip_paths = [in_path]
    else:
        raise ValueError('invalid path: %s' % in_path)

    sc_path = out_path + '.sc'
    doc_path = out_path + '.doc'
    info_path = out_path + '.info'
    count = 0
    with open(sc_path, 'w') as sc_out, open(doc_path, 'w') as doc_out, open(info_path, 'w') as info_out:
        for zip_path in zip_paths:
            with zipfile.ZipFile(zip_path, 'r') as zip_file:
                for py_path in iter_py_in_zip_file(zip_file):
                    with zip_file.open(py_path, 'r') as py_file:
                        tree = ASTPlus(ast.parse(py_file.read().strip()))
                        for docstring, source_code_words, tree in tree.parallel_functions(filters):
                            if count >= max_count:
                                return
                            count += 1

                            sc_out.write('%s\n' % source_code_words)
                            doc_out.write('%s\n' % docstring)
                            write_info(info_out, tree, count, py_path)

def write_info(info_out, tree, count, py_path):
    info_out.write('<count:%d>\n' % count)
    info_out.write('    <path>\n')
    info_out.write('    %s\n' % py_path)
    info_out.write('    </path>\n')
    info_out.write('    <lineno>\n')
    info_out.write('    %s\n' % tree.lineno)
    info_out.write('    </lineno>\n')
    info_out.write('</count:%d>\n' % count)
    # TODO unparse tree

def iter_files_with_extension(path, extension):
    for root, _dirs, files in os.walk(path):
        for name in files:
            _, ext = os.path.splitext(path)
            if ext == extension:
                yield os.path.join(root, name)

def iter_py_in_zip_file(zip_file): # TODO cache python paths
    """Find python files in zip."""
    for path in zip_file.namelist():
        _root, extension = os.path.splitext(path)
        if extension == '.py':
            yield path

def plot_zipf(freqs):
    counts = Counter()
    for freq in freqs:
        counts[freq] += 1

    sorted_counts = sorted(counts.items())
    x_values, y_values = zip(*sorted_counts)
    plt.plot(x_values, y_values, 'ro')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

def docs_in_tree(tree, doc_names = None, docs = None):
    if docs == None:
        docs = []
    if doc_names == None:
        doc_names = set(['FunctionDef', 'ClassDef', 'Module'])
    class_name = tree.__class__.__name__
    if class_name in doc_names:
        docstring = ast.get_docstring(tree)
        if docstring != None:
            docs.append(docstring)
    for child in ast.iter_child_nodes(tree):
        docs_in_tree(child, doc_names, docs)

    return docs

def binarize_rule_freqs(rule_freqs):
    binary_freqs = {}
    for rule, freq in rule_freqs.iteritems():
        binary_rules = binarize(rule)
        for binary in binary_rules:
            binary_freqs[binary] = binary_freqs.get(binary, 0) + freq

    return binary_freqs

def binarize(rule):
    if len(rule) <= 3:
        return [rule]
    sub_name = "X%sX" % (rule[0],)
    bin_rules = [(rule[0], rule[1], sub_name)]
    for i in xrange(2, len(rule)-2):
        bin_rules.append((sub_name, rule[i], sub_name))
    bin_rules.append((sub_name, rule[-2], rule[-1]))
    return bin_rules


def test_zipf():
    with open('../data/django.binary_rules', 'r') as rule_file:
        freqs = []
        for line in rule_file:
            freq, _ = line.split(':')
            freqs.append(int(freq))

        plot_zipf(freqs)

def test_create_parallel_corpus():
    in_path = '../repos/nltk-develop.zip'
    out_path = '../data/nltk-develop'
    filters = [remove_doctests]
    max_count = 10
    create_parallel_corpus(in_path, out_path, filters, max_count)

def main():
    pass


if __name__ == '__main__':
    #main()
    #test_zipf()
    #test_extract_rules_from_zip()
    test_create_parallel_corpus()
