import zipfile
import os
import ast
import re
from collections import Counter
import matplotlib.pyplot as plt


def create_parallel_corpus(in_path, out_path, name_map, max_count = float('inf')):
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
    count = 0
    with open(sc_path, 'w') as sc_out, open(doc_path, 'w') as doc_out:
        for zip_path in zip_paths:
            with zipfile.ZipFile(zip_path, 'r') as zip_file:
                for i, py_path in enumerate(iter_py_in_zip_file(zip_file)):
                    with zip_file.open(py_path, 'r') as py_file:
                        tree = ast.parse(py_file.read().strip())
                        for rules, doc in parallel_all_doc_tree(tree, name_map):
                            if count >= max_count:
                                return
                            count += 1
                            sc_out.write('%s\n' % rules)
                            doc_out.write('%s\n' % doc)

def parallel_all_doc_tree(tree, name_map):
    """Walk the tree and for all functions yield its rules and docstring"""
    for node in ast.walk(tree):
        if node.__class__.__name__ == 'FunctionDef':
            docstring = ast.get_docstring(node)
            if docstring != None:
                rules = ' '.join(get_linearized_rules(node, name_map))
                docstring = docstring.strip()
                docstring = re.sub(r'\s+', ' ', docstring)
                rules = rules.strip()
                rules = re.sub(r'\s+', ' ', rules)
                yield rules, docstring

def get_linearized_rules(tree, name_map, rules = None):
    """Return the rules in the tree corresponding to the top-down left-most
    derivation. name_map replaces a the name of a node in the tree with one of
    its attributes."""
    if rules == None:
        rules = []

    rule = get_mapped_name(tree, name_map)
    #rule += ',' + ','.join(get_mapped_name(child, name_map) \
    #                        for child in ast.iter_child_nodes(tree))
    rules.append(rule)
    for child in ast.iter_child_nodes(tree):
        if child.__class__.__name__ != 'arguments':
            get_linearized_rules(child, name_map, rules)

    return rules

def get_mapped_name(tree, name_map):
    """Return the class name of the tree. If name_map contains the class name
    return its corresponding attribute."""
    class_name = tree.__class__.__name__
    attribute = name_map.get(class_name, None)
    if attribute == None:
        return class_name
    elif attribute == 'REMOVE':
        return ''
    else:
        return str(getattr(tree, attribute))


def iter_files_with_extension(path, extension):
    for root, _dirs, files in os.walk(path):
        for name in files:
            _, ext = os.path.splitext(path)
            if ext == extension:
                yield os.path.join(root, name)

def extract_defs_and_docs_from_zip(zip_path, write_path, info_path, max_files = float('inf')):
    """Extract all python files with docstrings"""
    with open(write_path, 'w') as out, open(info_path, 'w') as info_out, \
            zipfile.ZipFile(zip_path, 'r') as zip_file:
        open_file = zip_file.open
        total_docs = 0
        total_files_with_docs = 0
        total_files = 0
        for i, py_path in enumerate(iter_py_in_zip_file(zip_file)):
            #print i
            if i >= max_files:
                break
            doc_count = extract_defs_and_docs(py_path, out, info_out, open_file)
            total_files += 1
            if doc_count > 0:
                total_files_with_docs += 1
                total_docs += doc_count

        info_out.write('Total docs: %s\n' % total_docs)
        info_out.write('Total files with docs: %s\n' % total_files_with_docs)
        info_out.write('Total python files: %s\n' % total_files)

def extract_rules_from_zip(zip_path, max_files = float('inf')):
    with zipfile.ZipFile(zip_path, 'r') as zip_file:
        rule_freqs = {}
        open_file = zip_file.open
        for i, py_path in enumerate(iter_py_in_zip_file(zip_file)):
            if i>= max_files:
                break

            extract_rules_from_file(py_path, rule_freqs, open_file)

    return rule_freqs

def extract_rules_from_file(py_path, rule_freqs, open_file = open):
    with open_file(py_path, 'r') as py_file:
        tree = ast.parse(py_file.read().strip())
        add_rules(tree, rule_freqs)

def add_rules(tree, rule_freqs):
    child_names = []
    for child in ast.iter_child_nodes(tree):
        child_names.append(child.__class__.__name__)
        add_rules(child, rule_freqs)

    if len(child_names) > 0:
        rule = (tree.__class__.__name__,) + tuple(child_names)
        rule_freqs[rule] = rule_freqs.get(rule, 0) + 1

def iter_py_in_zip_file(zip_file): # TODO cache python paths
    """Find python files in zip."""
    for path in zip_file.namelist():
        _root, extension = os.path.splitext(path)
        if extension == '.py':
            yield path

def defs_with_docs_info(tree, line_info = None):
    """Search the AST for all (non-nested) functions with docstrings and return
    a list of start-line-numbers, end-line-numbers and function name sorted by
    start-line-number."""
    if line_info == None:
        line_info = []

    function_def = tree.__class__.__name__ == 'FunctionDef'
    line_max = getattr(tree, 'lineno', None)
    for child in ast.iter_child_nodes(tree):
        if function_def:
            # do not look for nested functions
            _, child_line = defs_with_docs_info(child)
        else:
            _, child_line = defs_with_docs_info(child, line_info)
        if child_line > line_max:
            line_max = child_line

    if function_def:
        docstring = ast.get_docstring(tree)
        if docstring != None:
            line_info.append((tree.lineno, line_max, tree.name))

    return line_info, line_max

def extract_defs_and_docs(py_path, out, info_out, open_file = open):
    """Extract all functions with docstrings from py_path and write them to
    out. In info_out write info about the functions that were extracted."""
    with open_file(py_path, 'r') as py_file:
        tree = ast.parse(py_file.read().strip())
        def_info, _ = defs_with_docs_info(tree)

    if len(def_info) > 0:
        line_ranges = ' '.join('%s-%s-%s' % info for info in def_info)
        info_out.write("%s: %s\n" % (py_path, line_ranges))
        with open_file(py_path, 'r') as py_file:
            lineno = 0
            for line_min, line_max, _def_name in def_info:
                for line in py_file:
                    lineno += 1
                    if lineno < line_min:
                        continue
                    elif lineno <= line_max:
                        if lineno == line_min:
                            out.write('\n')
                        out.write(line)
                        if '\n' != line[-1]:
                            out.write('\n')
                    else:
                        break
    return len(def_info)

def rules_to_file(rule_freqs, out_path, sort = False):
    """Write rules and their frequencies to file in format: <freq>: <rule>
    If sort is true, sort rules descending by their frequencies"""
    with open(out_path, 'w') as out:
        if sort:
            pairs = sorted(rule_freqs.items(), key=lambda pair: pair[1], reverse=True)
        else:
            pairs = rule_freqs.iteritems()
        for rule, freq in pairs:
            out.write('%d: %s\n' % (freq, rule))

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

def extract_docs_from_zip(zip_path, out_path, separator = '---\n'):
    with zipfile.ZipFile(zip_path, 'r') as zip_file, open(out_path, 'w') as out:
        out.write(separator)
        for py_path in iter_py_in_zip_file(zip_file):
            with zip_file.open(py_path, 'r') as py_file:
                tree = ast.parse(py_file.read().strip())
            for doc in docs_in_tree(tree):
                if doc[-1] != '\n':
                    doc += '\n'
                out.write(doc)
                out.write(separator)

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

def test_extract_defs_and_docs_from_zip():
    zip_path = '../repos/django-master.zip'
    write_path = '../data/django.dd.py'
    info_path = '../data/django.info'
    extract_defs_and_docs_from_zip(zip_path, write_path, info_path)

def test_extract_rules_from_zip():
    zip_path = '../repos/django-master.zip'
    write_path = '../data/django.dd.py'
    info_path = '../data/django.info'
    rule_freqs = extract_rules_from_zip(zip_path, info_path)
    binary_rule_freqs = binarize_rule_freqs(rule_freqs)
    rules_to_file(binary_rule_freqs, '../data/django.binary_rules', True)

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
    name_map = {'FunctionDef': 'name', 'ClassDef': 'name',
                'Name': 'id',
                'Num': 'n',
                'Str': 's',
                'Load': 'REMOVE', 'Store': 'REMOVE', 'Param': 'REMOVE'}
    in_path = '../repos/nltk-develop.zip'
    out_path = '../data/nltk-develop'
    create_parallel_corpus(in_path, out_path, name_map, max_count = 10)

def main():
    extract_all_defs_and_docs_from_zip()


if __name__ == '__main__':
    #main()
    #test_zipf()
    #test_extract_rules_from_zip()
    test_create_parallel_corpus()
