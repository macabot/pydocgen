import zipfile
import os
import ast
import logging
import sys

from ast_plus import ASTPlus
import unparse

reload(sys)
sys.setdefaultencoding('utf-8') # FIXME this is dangerous
logging.basicConfig(filename='./log.txt', level=logging.DEBUG, format='%(asctime)s %(message)s')


def create_parallel_corpus(zip_path, out_path, max_count = float('inf')):
    """For all python files in the given zip file, create a parallel corpus of 
    docstring words and source code words."""
    if not os.path.isfile(zip_path) or os.path.splitext(zip_path)[1] != '.zip':
        raise ValueError('invalid zip path: %s' % zip_path)

    sc_path = out_path + '.sc'
    doc_path = out_path + '.doc'
    info_path = out_path + '.info'
    count = 0
    with open(sc_path, 'w') as sc_out, open(doc_path, 'w') as doc_out, open(info_path, 'w') as info_out:
        info_out.write('<zip_path>%s</zip_path>\n' % zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_file:
            for py_path in iter_py_in_zip_file(zip_file):
                with zip_file.open(py_path, 'r') as py_file:
                    try:
                        py_text = py_file.read().strip()
                        ast_tree = ast.parse(py_text)
                    except (SyntaxError, ValueError) as e: # could not parse the python file (expect python 2.x)
                        logging.exception('\nzip_path: %s\npy_path: %s\nError:\n%s\n' % (zip_path, py_path, e))
                        continue
                    except:
                        print 'zip_path: %s' % zip_path
                        print 'py_path: %s' % py_path
                        raise

                    tree = ASTPlus(ast_tree)

                    for docstring, source_code_words, tree in tree.parallel_functions():
                        if count >= max_count:
                            return
                        count += 1

                        try:
                            write_info(info_out, tree, count, py_path)
                            sc_out.write('%s\n' % source_code_words)
                            doc_out.write('%s\n' % docstring)
                        except:
                            print 'zip_path: %s' % zip_path
                            print 'py_path: %s' % py_path
                            print 'lineno: %s' % tree.lineno
                            raise

def write_info(info_out, tree, count, py_path):
    """Write to info_out the following information:
    path to the file
    line number of function in file
    source code of function"""
    info_out.write('<count:%d>\n' % count)
    info_out.write('<path>%s</path>\n' % py_path)
    info_out.write('<lineno>%s</lineno>\n' % tree.lineno)
    info_out.write('<code>\n')
    info_out.write('%s\n' % unparse.to_source(tree))
    info_out.write('</code>\n')
    info_out.write('</count:%d>\n' % count)

def iter_files_with_extension(path, extension):
    """Yield all files in a path with the given extension."""
    for root, _dirs, files in os.walk(path):
        for name in files:
            _root, ext = os.path.splitext(name)
            if ext == extension:
                yield os.path.join(root, name)

def iter_py_in_zip_file(zip_file): # TODO cache python paths
    """Find python files in zip."""
    for path in zip_file.namelist():
        _root, extension = os.path.splitext(path)
        if extension == '.py':
            yield path

def docs_in_tree(tree, doc_names = None, docs = None):
    """Get all docstrings in a tree. Doc_names specifies the classes that could
    contain docstrings. By default these are: 'FunctionDef', 'ClassDef' and
    'Module'."""
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
    """Binarize all rules in rule_freqs."""
    binary_freqs = {}
    for rule, freq in rule_freqs.iteritems():
        binary_rules = binarize(rule)
        for binary in binary_rules:
            binary_freqs[binary] = binary_freqs.get(binary, 0) + freq

    return binary_freqs

def binarize(rule):
    """Binarize a rule."""
    if len(rule) <= 3:
        return [rule]
    sub_name = "X%sX" % (rule[0],)
    bin_rules = [(rule[0], rule[1], sub_name)]
    for i in xrange(2, len(rule)-2):
        bin_rules.append((sub_name, rule[i], sub_name))
    bin_rules.append((sub_name, rule[-2], rule[-1]))
    return bin_rules

def test_create_parallel_corpus():
    """Create a parallel corpus of packages."""
    #path = '../repos'
    #extension = '.zip'
    #paths = iter_files_with_extension(path, extension)
    names = ['docutils-0.11']#, 'Twisted-13.2.0', 'Zope-master', 'Products.CMFPlone-master',
    """        'plone.api-master', 'rptlab-pyrxp', 'rptlab-preppy', 'bittorrent',
             'spambayes-1.1a6', 'wxPython-src-3.0.0.0', 'mailman-3.0.0b3',
             'plucker-1.8', 'feedparser-5.1.3', 'shtoom-0.2', 'divmod.org-master',
             'moin-1.9.7', 'PythonCard-0.8.2', 'OpenGLContext-2.2.0a3', 'pytables-3.1.0',
             'eric5-5.4.1']
    """
    paths = ['../repos/' + name + '.zip' for name in names]
    for in_path in paths:
        print in_path
        in_basename = os.path.basename(in_path)
        in_root, _ext = os.path.splitext(in_basename)
        out_path = os.path.join('../data/', in_root)
        create_parallel_corpus(in_path, out_path)

def main():
    pass


if __name__ == '__main__':
    #main()
    #test_zipf()
    #test_extract_rules_from_zip()
    test_create_parallel_corpus()
