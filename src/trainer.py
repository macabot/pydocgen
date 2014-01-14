import zipfile
import os
import ast
import re

import codegen


def extract_defs_and_docs_from_zip(zip_file, write_path, info_path, max_files = float('inf')):
    """Extract all python files with docstrings"""
    with open(write_path, 'w') as out, open(info_path, 'w') as info_out:
        openfile = zip_file.open
        total_docs = 0
        total_files_with_docs = 0
        total_files = 0
        for i, py_path in enumerate(iter_py_in_zip(zip_file)):
            #print i
            if i >= max_files:
                break
            doc_count = extract_defs_and_docs(py_path, out, info_out, openfile)
            total_files += 1
            if doc_count > 0:
                total_files_with_docs += 1
                total_docs += doc_count

        info_out.write('Total docs: %s\n' % total_docs)
        info_out.write('Total files with docs: %s\n' % total_files_with_docs)
        info_out.write('Total python files: %s\n' % total_files)

def iter_py_in_zip(zip_file):
    """Find python files in zip."""
    for path in zip_file.namelist():
        _, extension = os.path.splitext(path)
        if extension == '.py':
            yield path

def defs_with_docs_info(tree, line_info = None):
    if line_info == None:
        line_info = []

    line_max = getattr(tree, 'lineno', None)
    for child in ast.iter_child_nodes(tree):
        _, child_line = defs_with_docs_info(child, line_info)
        if child_line > line_max:
            line_max = child_line

    class_name = tree.__class__.__name__
    if class_name == 'FunctionDef':
        docstring = ast.get_docstring(tree)
        if docstring != None:
            line_info.append((tree.lineno, line_max, tree.name))
    
    return line_info, line_max

def extract_defs_and_docs(py_path, out, info_out, openfile = open):
    with openfile(py_path, 'r') as py_file:
        tree = ast.parse(py_file.read().strip())
        def_info, _ = defs_with_docs_info(tree)
    
    if len(def_info) > 0:
        line_ranges = ' '.join('%s-%s-%s' % info for info in def_info)
        info_out.write("%s: %s\n" % (py_path, line_ranges))
        with openfile(py_path, 'r') as py_file:
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

def main():
    with zipfile.ZipFile('../repos/django-master.zip', 'r') as zip:
        extract_defs_and_docs_from_zip(zip, '../data/django.dd.py', '../data/django.info')


if __name__ == '__main__':
    main()
