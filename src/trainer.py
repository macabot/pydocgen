import zipfile
import os
import ast
import re


def extract_defs_and_docs_from_zip(zip_path, write_path, info_path, max_files = float('inf')):
    """Extract all python files with docstrings"""
    with open(write_path, 'w') as out, open(info_path, 'w') as info_out, \
            zipfile.ZipFile(zip_path, 'r') as zip_file:
        open_file = zip_file.open
        total_docs = 0
        total_files_with_docs = 0
        total_files = 0
        for i, py_path in enumerate(iter_py_in_zip(zip_file)):
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
        for i, py_path in enumerate(iter_py_in_zip(zip_file)):
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

def iter_py_in_zip(zip_file):
    """Find python files in zip."""
    for path in zip_file.namelist():
        _, extension = os.path.splitext(path)
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

def test_extract_defs_and_docs_from_zip():
    zip_path = '../repos/django-master.zip'
    write_path = '../data/django.dd.py'
    info_path = '../data/django.info'
    extract_defs_and_docs_from_zip(zip_path, write_path, info_path)

def test_extract_rules_from_zip():
    zip_path = '../repos/django-master.zip'
    write_path = '../data/django.dd.py'
    info_path = '../data/django.info'
    extract_rules_from_zip(zip_path, write_path, info_path)

def main():
    extract_all_defs_and_docs_from_zip()


if __name__ == '__main__':
    main()
