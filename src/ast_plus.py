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
IGNORE_MAP = set(['arguments',
                  'BinOp',
                  'BoolOp',
                  'Call',
                  'Expr',
                  'Load',
                  'Module',
                  'Param',
                  'Store',
                  'Str', # ignore or map to 's', i.e. content of the string?
                  'UnaryOp'])

class ASTPlus(object):

    def __init__(self, ast_tree):
        add_parents(ast_tree)
        add_labels(ast_tree)
        add_factors(ast_tree)
        self.tree = ast_tree
        
    def draw(self, show_factors = None):
        draw(self.tree, show_factors)
        

def set_parents(tree, parent = None):
    tree.parent = parent
    for child in ast.iter_child_nodes(tree):
        add_parents(child, tree)

def tokenize(tree, tokens = None):
    if tokens == None:
        tokens = []

    class_name = tree.__class__.__name__
    token = get_name_map(tree, TOKENIZE_MAP)
    if token != '':
        tokens.append(token)

    for child in ast.iter_child_nodes(tree):
        tokenize(child, tokens)

    return tokens

def get_name_map(tree, map):
    class_name = tree.__class__.__name__
    if class_name in map:
        return str(getattr(tree, map[class_name]))
    return class_name

def tokenize_tree(tree, self_class = None):
    parent_name = get_name_map(tree, LABEL_MAP)
    if tree.__class__.__name__ == 'ClassDef':
        self_class = parent_name
    child_names = []
    for child in ast.iter_child_nodes(tree):
        token = tokenize_tree(child, self_class)
        if token != '':
            if token == 'self':
                token += '_' + self_class
            child_names.append(token)
    if len(child_names) == 0:
        return parent_name
    return '(%s %s)' % (parent_name, ' '.join(child_names))

def split_camel_case(name):
    # see http://stackoverflow.com/a/1176023/854488
    #return [name.lower()]
    return [name]

def add_labels(tree):
    tree.label = get_name_map(tree, LABEL_MAP)
    for child in ast.iter_child_nodes(tree):
        add_labels(child)

def add_factors(tree, class_def_name = None):
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

def draw(tree, show_factors = None):
    """Convert the tree to a nltk tree and draw it."""
    from nltk import Tree
    ast_to_nltk_tree(tree, show_factors).draw()

def ast_to_nltk_tree(ast_tree, show_factors = None):
    """Convert an ast tree to a nltk tree"""
    from nltk import Tree
    parent_name = ast_tree.label
    if show_factors:
        parent_name += '|'
        parent_name += '|'.join(ast_tree.factors[key] for key in show_factors)
    return Tree(parent_name, [ast_to_nltk_tree(child, show_factors) for child in ast.iter_child_nodes(ast_tree)])
