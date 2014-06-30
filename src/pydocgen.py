"""
By Michael Cabot

Generate docstrings for python functions and add them to the file.
"""
import os
import ast
import argparse

def generate_docstring(tree):
    """Translate an AST to a docstring."""
    docstring = 'TODO' # TODO
    return '\"\"\"%s\"\"\"\n' % docstring

def generate_all_docstrings(tree, positions = None):
    """Travel AST and generate a docstring for each function. Return a dict
    that maps a function's line number to its docstring and the docstring's
    col_offset."""
    if positions == None:
        positions = {}

    class_name = tree.__class__.__name__
    if class_name == 'FunctionDef' and ast.get_docstring(tree) == None:
        positions[tree.lineno] = (generate_docstring(tree),
                                  get_child_indent(tree))

    for child in ast.iter_child_nodes(tree):
        generate_all_docstrings(child, positions)

    return positions

def get_child_indent(tree):
    """Get the col_offset of one of the tree's children."""
    for child in ast.iter_child_nodes(tree):
        col_offset = getattr(child, 'col_offset', None)
        if col_offset != None:
            return col_offset

    raise ValueError('%s has no child with attribute col_offset' % tree)

def apply_docstrings(py_path, write_path):
    """Create a copy of a python file that contains docstrings generated for
    each function."""
    with open(py_path, 'r') as py_file:
        tree = ast.parse(py_file.read().strip())
        docstrings = generate_all_docstrings(tree)

    with open(py_path, 'r') as py_file, \
            open(write_path, 'w') as out:
        for i, line in enumerate(py_file):
            lineno = i+1
            out.write(line)
            if lineno in docstrings:
                docstring, col_offset = docstrings[lineno]
                if line.strip()[-1] == ':':
                    out.write("%s%s" % (' ' * col_offset, docstring))
                else:
                    docstrings[lineno+1] = docstrings[lineno]
                    del docstrings[lineno]

def main():
    """read command line arguments"""
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-i', '--input', required=True,
        help='Path to Python file.')
    arg_parser.add_argument('-o', '--output', required=True,
        help='Path for output.')
    args = arg_parser.parse_args()
    py_path = args.input
    write_path = args.output
    assert os.path.isfile(py_path)
    write_dir, write_name = os.path.split(write_path)
    assert os.path.isdir(write_dir), 'invalid output dir: %s' % write_dir
    assert write_name.strip()!='', 'empty output name'
    apply_docstrings(py_path, write_path)


if __name__ == '__main__':
    main()
