"""
By Michael Cabot

Preprocess a parallel corpus before passing it to a SMT framework, such as Moses
"""
import os
import re
import argparse

from ast_plus import NEWLINE_SUB
import docfilters
import utils

def split_camel_case(name):
    """source: http://stackoverflow.com/a/1176023/854488
    Split on camelCase
    TODO split instead of sub
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    underscore = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    return split_underscore(underscore)

def split_underscore(word):
    """Split a word into a list of words that it contains."""
    word_stripped = word.strip('_')
    if word_stripped == '':
        return ['_']
    word_stripped = re.sub(r'_+', '_', word_stripped)
    return word_stripped.lower().split('_')

def remove_context(sc_words):
    """Remove words with ast_class equal to Load, Param or Store"""
    context_terms = set(['Load', 'Param', 'Store'])
    return (word for word in sc_words if word.split('|')[1] not in context_terms)

def tokenize_words(sc_words):
    """Split a words into the words that each contains.
    split on under_score, camelCase
    TODO split getters/setters, e.g. 'getsource' >> ['get', 'source']"""
    token_words = []
    for word in sc_words:
        label, factors = word.split('|', 1)
        if '_' in label:
            token_words.extend(('%s|%s' % (part_word, factors)
                               for part_word in split_underscore(label)))
        else:
            token_words.extend(('%s|%s' % (part_word, factors)
                               for part_word in split_camel_case(label)))
    return token_words

def remove_factors(sc_words):
    """Remove all factors: label|factor1|...|factorN -> label"""
    return [word.split('|', 1)[0] for word in sc_words]

def process_source_code(sc_words, keep_factors):
    """Remove useless words, tokenize (remove factors)"""
    sc_words = remove_context(sc_words)
    sc_words = tokenize_words(sc_words)
    if not keep_factors:
        sc_words = remove_factors(sc_words)
    return sc_words

def process_docstring(docstring, filters):
    """Apply docstring filters"""
    docstring = docstring.replace(NEWLINE_SUB, '\n')
    for docfilter in filters:
        docstring = docfilter(docstring)
    return re.sub(r'\s+', ' ', docstring.strip()).lower()

def preprocess(in_path, out_folder, keep_factors, filters):
    """Preprocess a parallel corpus"""
    in_folder, in_basename = os.path.split(in_path)
    assert os.path.isdir(in_folder), 'invalid in folder: %s' % in_folder
    assert in_basename.strip() != '', 'empty basename'
    assert os.path.isdir(out_folder), 'invalid out folder: %s' % out_folder
    sc_basename = in_basename + '.sc'
    doc_basename = in_basename + '.doc'
    sc_in_path = os.path.join(in_folder, sc_basename)
    doc_in_path = os.path.join(in_folder, doc_basename)
    assert os.path.isfile(sc_in_path), 'invalid file: %s' % sc_in_path
    assert os.path.isfile(doc_in_path), 'invalid file: %s' % doc_in_path
    sc_out_path = os.path.join(out_folder, sc_basename)
    doc_out_path = os.path.join(out_folder, doc_basename)

    with open(sc_in_path, 'r') as sc_in, open(doc_in_path, 'r') as doc_in, \
            open(sc_out_path, 'w') as sc_out, open(doc_out_path, 'w') as doc_out:
        for sc_line in sc_in:
            sc_words = sc_line.strip().split()
            docstring = doc_in.next().strip()

            sc_words = process_source_code(sc_words, keep_factors)
            docstring = process_docstring(docstring, filters)
            if docstring == '' or len(docstring.split()) > 100 or len(sc_words) > 100:
                continue

            sc_out.write('%s\n' % ' '.join(sc_words))
            doc_out.write('%s\n' % docstring)

def test_preprocess():
    """test preprocess"""
    in_path = '../data/docstring-all_sourcecode-all-factors/Python27-Lib'
    out_folder = '../data/docstring-filtered_sourcecode-NOcontext-NOfactors'
    keep_factors = False
    filters = [docfilters.remove_doctests,
               docfilters.keep_first_description,
               docfilters.remove_parameter_descriptions,
               docfilters.replace_vertical_bars]
    preprocess(in_path, out_folder, keep_factors, filters)

def main():
    """Read command line arguments for preprocessing a parallel corpus."""
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-i', '--input_path', required=True,
        help='Path to folder containing parallel corpora')
    arg_parser.add_argument('-o', '--output_path', required=True,
        help='Path of folder for output.')
    arg_parser.add_argument('-f', '--keep_factors', action='store_true',
        default = False, help='Keep the factors')

    args = arg_parser.parse_args()

    input_path = args.input_path
    if not os.path.isdir(input_path):
        raise ValueError('Invalid input folder: %s' % input_path)
    output_path = args.output_path
    if not os.path.isdir(output_path):
        raise ValueError('Invalid output folder: %s' % output_path)
    keep_factors = args.keep_factors

    filters = [docfilters.remove_doctests,
               docfilters.keep_first_description,
               docfilters.remove_wx_wrappers,
               docfilters.remove_parameter_descriptions,
               docfilters.replace_vertical_bars]

    for sc_path in utils.iter_files_with_extension(input_path, '.sc'):
        in_root, _ext = os.path.splitext(sc_path)
        print in_root
        preprocess(in_root, output_path, keep_factors, filters)
    print 'done'


if __name__ == '__main__':
    main()
    #test_preprocess()
