import ast
import re

TOKENIZE_MAP = {'arguments': '',
                'Attribute': 'attr',
                'BinOp': '',
                #'Call': '',
                'ClassDef': 'name',
                #'Eq': '==',
                'FunctionDef': 'name',
                'Load': '',
                'Name': 'id',
                'Num': 'n',
                'Param': '',
                #'Sub': '-',
                'Store': '',
                'UnaryOp': ''
                }
LABEL_MAP = {'Attribute': 'attr',
             'ClassDef': 'name',
             'FunctionDef': 'name',
             'Name': 'id',
             'Num': 'n'}
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
                  'Str', # ignore or map to 's', i.e. content of the string?
                  'UnaryOp'])

class ASTPlus(object):

    def __init__(self, ast_tree, split = False):
        # TODO make trees uniform, e.g. order arguments of Equals alphabetically
        add_parents(ast_tree)
        add_labels(ast_tree, split)
        add_factors(ast_tree)
        add_ignore_labels(ast_tree)
        self.tree = ast_tree

    def parallel_functions(self, filters = None):
        """For each function in the tree return a tuple with the words in its
        docstring and the important nodes in it's tree. Ignore functions without
        a docstring."""
        return parallel_functions(self.tree, filters)

    def draw(self, show_factors = None, show_ignore = False):
        """Convert an ast tree to a nltk tree"""
        draw(self.tree, show_factors, show_ignore)


def parallel_functions(tree, filters = None, parallel = None):
    """For each function in the tree return a tuple with the words in its
    docstring and the important nodes in it's tree. Ignore functions without
    a docstring."""
    if parallel == None:
        parallel = []

    if tree.__class__.__name__ == 'FunctionDef':
        docstring = ast.get_docstring(tree)
        if docstring != None:
            #docstring = docstring.encode('utf-8')
            sentence = ' '.join(tree_to_words(tree))
            clean_docstring = clean_doc(docstring, filters)
            if clean_docstring != '':
                parallel.append((clean_docstring, sentence, tree))

    for child in ast.iter_child_nodes(tree):
        parallel_functions(child, filters, parallel)

    return parallel

def clean_doc(docstring, filters = None):
    """Filter unimportant parts of the docstring and make all words white space
    separated."""
    if filters != None:
        for filter in filters:
            docstring = filter(docstring)
    docstring = re.sub(r'\s+', ' ', docstring.strip())
    return docstring

def tree_to_words(tree, words = None):
    """Extract the important tree labels from the tree."""
    if words == None:
        words = []

    if not tree.ignore:
        words.append(tree.label)
    for child in ast.iter_child_nodes(tree):
        tree_to_words(child, words)

    return words

def add_ignore_labels(tree):
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
    # see http://stackoverflow.com/a/1176023/854488
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

def draw(tree, show_factors = None, show_ignore = False):
    """Convert the tree to a nltk tree and draw it."""
    ast_to_nltk_tree(tree, show_factors, show_ignore).draw()

def ast_to_nltk_tree(ast_tree, show_factors = None, show_ignore = False):
    """Convert an ast tree to a nltk tree"""
    from nltk import Tree
    parent_name = ast_tree.label
    if show_factors:
        parent_name += '|'
        parent_name += '|'.join(ast_tree.factors[key] for key in show_factors)
    if show_ignore and ast_tree.ignore:
        parent_name = 'I_' + parent_name
    return Tree(parent_name, [ast_to_nltk_tree(child, show_factors, show_ignore) for child in ast.iter_child_nodes(ast_tree)])
