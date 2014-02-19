"""
By Michael Cabot

Add attributes to the AST
Extract parallel data (source code words - docstring words) from an AST

TODO subclass AST
"""

import ast
import re

LABEL_MAP = {'Attribute': 'attr',
             'ClassDef': 'name',
             'FunctionDef': 'name',
             'Name': 'id',
             'Num': 'n'}
             #'Str': 's'}
IGNORE_MAP = set([#'arguments',
                  'BinOp',
                  'BoolOp',
                  #'Call',
                  #'Compare',
                  'Expr',
                  'Load',
                  #'Module',
                  'Param',
                  'Store',
                  #'Str', # ignore or map to 's', i.e. content of the string?
                  'UnaryOp'])

NEWLINE_SUB = '</NEWLINE>'

class ASTPlus(object):
    """Add factors to an AST"""

    def __init__(self, ast_tree, split = False):
        # TODO make trees uniform, e.g. order arguments of Equals alphabetically
        add_parents(ast_tree)
        add_labels(ast_tree, split)
        self.factor_keys = add_factors(ast_tree)
        add_ignore_labels(ast_tree)
        label_docstring_expr(ast_tree)
        self.tree = ast_tree

    def parallel_functions(self, use_factors = True):
        """For each function in the tree return a tuple with the words in its
        docstring and the important nodes in it's tree. Ignore functions without
        a docstring."""
        if use_factors:
            return parallel_functions(self.tree, self.factor_keys)
        return parallel_functions(self.tree)

    def draw(self, factor_keys = None, show_ignore = False):
        """Convert an ast tree to a nltk tree"""
        draw(self.tree, factor_keys, show_ignore)


def parallel_functions(tree, factor_keys, parallel = None):
    """For each function in the tree return a tuple with the words in its
    docstring and the important nodes in it's tree. Ignore functions without
    a docstring."""
    if parallel == None:
        parallel = []

    if tree.__class__.__name__ == 'FunctionDef':
        docstring = ast.get_docstring(tree)
        if docstring != None:
            docstring = docstring.strip()
            if docstring != '':
                sentence = ' '.join(tree_to_words(tree, factor_keys))
                docstring = re.sub(r'\n', NEWLINE_SUB, docstring)
                parallel.append((docstring, sentence, tree))

    for child in ast.iter_child_nodes(tree):
        parallel_functions(child, factor_keys, parallel)

    return parallel

def clean_doc(docstring, docfilters = None):
    """Filter unimportant parts of the docstring and make all words white space
    separated."""
    if docfilters != None:
        for docfilter in docfilters:
            docstring = docfilter(docstring)
    docstring = re.sub(r'\s+', ' ', docstring.strip())
    return docstring

def tree_to_words(tree, factor_keys = None, ignore_docstring = True, words = None):
    """Extract the important tree labels from the tree. If the values of the
    factor keys separated by '|'."""
    if words == None:
        words = []

    word = tree.label
    if factor_keys != None:
        word += '|' + '|'.join(tree.factors[key] for key in factor_keys)
    words.append(word)
    for child in ast.iter_child_nodes(tree):
        if ignore_docstring and child.gen_docstring:
            continue
        tree_to_words(child, factor_keys, ignore_docstring, words)

    return words

def label_docstring_expr(tree):
    """Label the Expr node that generates a docstring."""
    tree_class_name = tree.__class__.__name__
    for child in ast.iter_child_nodes(tree):
        child.gen_docstring = tree_class_name == 'FunctionDef' and \
                       child.__class__.__name__ == 'Expr'
        label_docstring_expr(child)

def add_ignore_labels(tree):
    """For each node in the tree add the attribute 'ignore' set according to the
    ignore map."""
    class_name = tree.__class__.__name__
    tree.ignore = class_name in IGNORE_MAP
    for child in ast.iter_child_nodes(tree):
        add_ignore_labels(child)

def set_parents(tree, parent = None):
    """Add to each node a pointer to its parent."""
    tree.parent = parent
    for child in ast.iter_child_nodes(tree):
        add_parents(child, tree)

def get_name_map(tree, name_map):
    """If the class name is in the name map, return an attribute of the tree."""
    class_name = tree.__class__.__name__
    if class_name in name_map:
        return str(getattr(tree, name_map[class_name]))
    return class_name

def split_camel_case(name):
    """ TODO see http://stackoverflow.com/a/1176023/854488"""
    #return [name.lower()]
    return [name]

def add_labels(tree, split = False):
    """Add labels to all nodes in the tree."""
    tree.label = get_name_map(tree, LABEL_MAP)
    if split:
        tree.label = ' '.join(split_label(tree.label))
    for child in ast.iter_child_nodes(tree):
        add_labels(child, split)

def split_label(label):
    """Split a label into a list of words that it contains.
    TODO split camelCase
    TODO split getters/setters, e.g. 'getsource' >> ['get', 'source']"""
    if '_' in label:
        return label.strip('_').split('_')
    return [label]

def add_factors(tree, class_def_name = None):
    """Add factors to all nodes in the tree.
    TODO explain all factors"""
    class_name = tree.__class__.__name__
    if class_name == 'ClassDef':
        class_def_name = class_name
    tree.factors = {}
    tree.factors['ast_name'] = class_name
    tree.factors['parent'] = str(getattr(tree.parent, 'label', tree.parent))
    context = getattr(tree, 'ctx', None)
    tree.factors['context'] = str(getattr(context, 'label', context))
    tree.factors['class'] = str(class_def_name)

    for child in ast.iter_child_nodes(tree):
        add_factors(child, class_def_name)

    return ['ast_name', 'parent', 'context', 'class']

def root(tree):
    """Return the root of a tree."""
    if tree.parent == None:
        return tree
    return root(tree.parent)

def add_parents(tree, parent = None):
    """Add the attribute parent to each node in the tree."""
    tree.parent = parent
    for child in ast.iter_child_nodes(tree):
        add_parents(child, tree)

def draw(tree, factor_keys = None, show_ignore = False):
    """Convert the tree to a nltk tree and draw it."""
    ast_to_nltk_tree(tree, factor_keys, show_ignore).draw()

def ast_to_nltk_tree(ast_tree, factor_keys = None, show_ignore = False, ignore_docstring = True):
    """Convert an ast tree to a nltk tree"""
    from nltk import Tree
    parent_name = ast_tree.label
    if factor_keys:
        parent_name += '|'
        parent_name += '|'.join(ast_tree.factors[key] for key in factor_keys)
    if show_ignore and ast_tree.ignore:
        parent_name = 'I_' + parent_name
    return Tree(parent_name, [ast_to_nltk_tree(child, factor_keys, show_ignore) for child in ast.iter_child_nodes(ast_tree) if not(ignore_docstring and child.gen_docstring)])

def ast_to_qtree(node, add_prefix = True):
    """Convert a tree to tikz-qtree notation."""
    children = list(ast.iter_child_nodes(node))
    if len(children) == 0:
        return node.label

    prefix = ''
    if add_prefix:
        prefix = r'\Tree '
    qtree = '%s[.%s %s ]' % (prefix, node.label, ' '.join(ast_to_qtree(child, False) for child in children))
    return qtree


