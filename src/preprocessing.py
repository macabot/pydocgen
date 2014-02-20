"""
By Michael Cabot

Preprocess a parallel corpus before passing it to a SMT framework, such as Moses
"""

def split_camel_case(name):
    """ TODO see http://stackoverflow.com/a/1176023/854488"""
    #return [name.lower()]
    return [name]

def split_underscore(label):
    """Split a label into a list of words that it contains.
    TODO split getters/setters, e.g. 'getsource' >> ['get', 'source']"""
    return label.strip('_').split('_')