"""
By Michael Cabot

Pattern Search
"""

import re
import math
import argparse
import os
import sys
import time
import heapq
from collections import OrderedDict, defaultdict

import decoder
import mp_worker
from utils import show_progress

class DecodeArguments:
    """Store the arguments for decoding and pattern search"""

    def __init__(self, input_file, output_file, language_model,
                 source_language_model, full_translation_model, max_lines,
                 beam_size, processes, max_phrase_length, stack_limit,
                 stupid_backoff, n_size, nbest, empty_default, top_translations,
                 bleu_path, reference):
        # decode arguments
        self.input_file = input_file
        self.output_file = output_file
        self.language_model = language_model
        self.source_language_model = source_language_model
        self.full_translation_model = full_translation_model
        self.max_lines = max_lines
        self.beam_size = beam_size
        self.processes = processes
        self.max_phrase_length = max_phrase_length
        self.stack_limit = stack_limit
        self.stupid_backoff = stupid_backoff
        self.n_size = n_size
        self.nbest = nbest
        self.empty_default = empty_default
        self.top_translations = top_translations
        # pattern search arguments
        self.bleu_path = bleu_path
        self.reference = reference
        self.stored_tm = None

    def get_bleu_score(self, weights, weight_index = None):
        """Get the BLEU score of a translation"""
        start_time = time.time()
        # only the first 4 features affect the trimming of the full TM
        store_translation_model = weight_index != None and weight_index >= 4
        self.decode(weights, store_translation_model)
        print 'Decode time: %s seconds' % (time.time() - start_time,)
        return get_bleu_score(self.bleu_path, self.reference, self.output_file)

    def decode(self, weights, store_translation_model):
        """Decode with given weights"""
        # if changed weight does not affect the translation_model
        # store and reuse the translation_model
        if store_translation_model:
            if self.stored_tm == None:
                translation_model = trim_translation_model(
                                            self.full_translation_model,
                                            weights,
                                            self.top_translations)
                self.stored_tm = translation_model
            else:
                translation_model = self.stored_tm
        else:
            self.stored_tm = None
            translation_model = trim_translation_model(
                                        self.full_translation_model,
                                        weights,
                                        self.top_translations)


        mp_worker.set_up_decoders(self.input_file, self.output_file,
                                  self.language_model,
                                  self.source_language_model,
                                  translation_model, self.max_lines,
                                  self.beam_size, self.processes,
                                  self.max_phrase_length, self.stack_limit,
                                  self.stupid_backoff, self.n_size,
                                  weights, self.nbest, self.empty_default)




def get_bleu_score(bleu_path, reference, hypothesis):
    """calculate the BLEU score"""
    assert os.path.isfile(bleu_path), 'invalid bleu path: %s' % bleu_path
    assert os.path.isfile(reference), 'invalid reference path: %s' % reference
    assert os.path.isfile(hypothesis), 'invalid hypothesis path: %s' % hypothesis
    bleu_out = os.popen("%s %s < %s" % (bleu_path, reference, hypothesis)).read()
    match = re.search(r'(?<=BLEU = )\d+(\.\d+)?', bleu_out)
    assert match != None, 'unknown bleu output: %s' % bleu_out # check if file has execution permission
    return float(match.group(0))

def test_get_bleu_score():
    """test get_bleu_score"""
    bleu_path = '/home/bart/apps/smt_tools/decoders/mosesdecoder/scripts/generic/multi-bleu.perl'
    reference = '/home/michael/pydocgen/data/tune/tune_clean_docstring-filtered_sourcecode-NOcontext-NOfactors.tok.doc'
    hypothesis = '/home/michael/pydocgen/data/decode/blankdefault/tunesearch_NOfactors.doc_STACK100TOP10BEAM1CORE30'
    print get_bleu_score(bleu_path, reference, hypothesis)

def optimize(weights, get_score, step_size, min_score_diff, min_step_size,
             max_iterations, weight_scores = None):
    """Optimize the score with pattern search"""
    print 'weights: %s' % (weights,)
    print 'step_size: %s' % (step_size,)
    print 'min_score_diff: %s' % (min_score_diff,)
    print 'min_step_size: %s' % (min_step_size,)
    print 'max_iterations: %s' % (max_iterations,)

    if weight_scores == None:
        weight_scores = OrderedDict()

    t_weights = tuple(weights)
    best_weights = weights[:]
    if t_weights not in weight_scores:
        best_score = get_score(weights)
        weight_scores[t_weights] = best_score
    else:
        best_score = weight_scores[t_weights]

    diff = float('inf')
    iteration = 0
    while diff >= min_score_diff and iteration <= max_iterations and \
            step_size >= min_step_size:
        print "i:%s, w:%s, s:%s" % (iteration, best_weights, best_score)
        previous_best_weights = best_weights[:]
        current_weights = best_weights[:]
        for i in xrange(len(weights)):
            # add step size to weight_i
            add_weights = current_weights[:]
            add_weights[i] += step_size
            t_add_weights = tuple(add_weights)
            if t_add_weights not in weight_scores:
                score = get_score(add_weights, i)
                weight_scores[t_add_weights] = score
                print "w:%s, s:%s" % (add_weights, score)
            else:
                score = weight_scores[t_add_weights]
            if score > best_score:
                diff = score - best_score
                best_score = score
                best_weights = add_weights
            # subtract step size from weight_i
            subtract_weights = current_weights[:]
            subtract_weights[i] -= step_size
            t_subtract_weights = tuple(subtract_weights)
            if t_subtract_weights not in weight_scores:
                score = get_score(subtract_weights, i)
                weight_scores[t_subtract_weights] = score
                print "w:%s, s:%s" % (subtract_weights, score)
            else:
                score = weight_scores[t_subtract_weights]
            if score > best_score:
                diff = score - best_score
                best_score = score
                best_weights = subtract_weights

        if previous_best_weights == best_weights:
            step_size /= 2.0

        iteration += 1

    print "i:%s, w:%s, s:%s" % (iteration, best_weights, best_score)
    return (best_weights, best_score), weight_scores

def test_optimize():
    """test optimize"""
    get_score = lambda weights: -(weights[0]**2)
    step_size = 1.0
    min_score_diff = 0.001
    min_step_size = 0.5**5
    max_iterations = 3000
    (best_weights, best_score), _weight_scores = optimize([3.0], get_score,
                                                         step_size,
                                                         min_score_diff,
                                                         min_step_size,
                                                         max_iterations)
    print 'best_weights: %s' % (best_weights,)
    print 'best_score: %s' % (best_score,)
    # print 'weight_scores:'
    # for k, v in weight_scores.iteritems():
        # print k, v

def weight_scores_to_file(weight_scores, path):
    """write weight_scores to file"""
    with open(path, 'w') as out:
        for weights, score in weight_scores.iteritems():
            out.write('{weights} ||| {score}\n'.format(
                weights=' '.join(str(weight) for weight in weights),
                score=score))

def read_weight_scores(path):
    """Read weight_scores from file"""
    weight_scores = OrderedDict()
    with open(path, 'r') as in_file:
        for line in in_file:
            weights, score = line.strip().split(' ||| ')
            weights = [float(w) for w in weights.split()]
            score = float(score)
            weight_scores[weights] = score
    return weight_scores

def read_full_translation_model(file_name, max_phrase_length):
    """Read the full translation model taking into account the maximal phrase
    length"""
    translation_model = defaultdict(list)
    document = open(file_name, 'r')

    num_lines = sum(1 for line in open(file_name, 'r'))
    point = num_lines / 100 if num_lines > 100 else 1

    for i, line in enumerate(document):
        if i % point == 0:
            show_progress(i, num_lines, 40, 'LOADING FULLTRANSLATIONMODEL')

        segments = line.strip().split(' ||| ')
        source = tuple(segments[0].split())
        if len(source) > max_phrase_length:
            continue
        target = tuple(segments[1].split())
        probs = tuple([float(prob) for prob in segments[2].split()])

        translation_model[source].append((target, probs))

    show_progress(1, 1, 40, 'LOADING FULLTRANSLATIONMODEL')
    sys.stdout.write('\n')
    document.close()

    return translation_model

def trim_translation_model(full_translation_model, weights,
                           top_translations):
    """Use the full_translation_model to create a smaller translation_model
    according to the restrictions of the weights and top_translations."""
    translation_model = defaultdict(list)
    num_lines = len(full_translation_model)
    point = num_lines / 100 if num_lines > 100 else 1
    for i, (source, target_probs) in enumerate(full_translation_model.iteritems()):
        if i % point == 0:
            show_progress(i, num_lines, 40, 'TRIM FULLTRANSLATIONMODEL')
        measure_target_probs = []
        for target, probs in target_probs:
            measure = sum([prob * weights[i] for i, prob in \
                           enumerate(probs)])
            if len(measure_target_probs) < top_translations:
                heapq.heappush(measure_target_probs, (measure, target, probs))
            else:
                heapq.heappushpop(measure_target_probs, (measure, target, probs))
        translation_model[source] = [(target, probs) for (_measure, target, probs) in measure_target_probs]

    show_progress(1, 1, 40, 'TRIM FULLTRANSLATIONMODEL')
    sys.stdout.write('\n')
    return translation_model

def main():
    """Read command line arguments."""
    features = ['p(f|e)', 'p(e|f)', 'lex(f|e)', 'lex(e|f)', 'lm(e)',
                'phrase penalty', 'word penalty', 'linear distortion']
    num_features = len(features)

    arg_parser = argparse.ArgumentParser()
    # decode arguments
    arg_parser.add_argument("-i", "--input_file", required=True,
            help="The file containing sentences to be translated")
    arg_parser.add_argument("-o", "--output_file", required=True,
        help="Output filename")
    arg_parser.add_argument("-lm", "--language_model", required=True,
        help="Target side language model")
    arg_parser.add_argument("-lms", "--source_language_model", required=True,
        help="Source side language model")
    arg_parser.add_argument("-tm", "--translation_model", required=True,
        help="Translation model containing conditional probabilities and \
            lexical weights")
    arg_parser.add_argument("-mpt", "--max_lines", type=float,
        default=float('inf'),
        help="Maximum number of lines to parse from input file.")
    arg_parser.add_argument("-mpl", "--max_phrase_length", type=int,
        default=3, help="Limit decoding to using phrases up to n words")
    arg_parser.add_argument("-sl", "--stack_limit", type=float,
        default=float('inf'),
        help="Stack limit")
    arg_parser.add_argument("-bs", "--beam_size", type=float,
        default=float('inf'),
        help="Beam size")
    arg_parser.add_argument("-sb", "--stupid_backoff", type=float,
        default=0.4, help="Stupid backoff.")
    arg_parser.add_argument("-ns", "--n_size", type=int,
        default=3, help="Size of language model")
    arg_parser.add_argument("-w", "--feature_weights", nargs=num_features,
        type=float, default=num_features*[1.0],
        help="Initial feature weights. Order: %s" % ", ".join(features))
    arg_parser.add_argument('-tt', '--top_translations', type=int, default=10,
        help="Top translations for the translation model")
    arg_parser.add_argument('-pr', '--processes', type=int, default=1,
        help="Number of processes to use, default 1 (single process)")
    arg_parser.add_argument('-nb', '--nbest', type=int, default=1,
        help="Specifies how many top translations should be found")
    arg_parser.add_argument('-ed', '--empty_default', action='store_true',
        default=False,
        help="If true: unknown words are translated with an empty string. Else:\
        unknown words are translated with itself.")
    # pattern search arguments
    arg_parser.add_argument('-sz', '--step_size', required=True, type=float,
        help="Step size for pattern search")
    arg_parser.add_argument('-msd', '--min_score_diff', required=True,
        type=float, help="Minimal score difference. (stopping criteria 1)")
    arg_parser.add_argument('-mit', '--max_iterations', required=True,
        type=int, help="Maximal iterations of pattern search. \
        (stopping criteria 2)")
    arg_parser.add_argument('-msz', '--min_step_size', required=True,
        type=float, help="Minimal step size (stopping criteria 3)")
    arg_parser.add_argument('-bp', '--bleu_path', required=True,
        help="Path to multi-bleu.perl")
    arg_parser.add_argument('-ref', '--reference', required=True,
        help="Path to reference translations")
    arg_parser.add_argument('-ws', '--weight_scores', default=None,
        help="Path to existing weight scores")

    args = arg_parser.parse_args()

    weight_scores_path = args.weight_scores
    if weight_scores_path != None:
        weight_scores = read_weight_scores(weight_scores_path)
    else:
        weight_scores = None

    input_file = args.input_file
    output_file = args.output_file

    # check if valid paths
    assert os.path.isfile(input_file), 'invalid input_file: %s' % input_file
    output_folder, output_name = os.path.split(output_file)
    assert os.path.isdir(output_folder), 'invalid output_folder: %s' % output_folder
    assert output_name.strip()!='', 'empty output name'
    assert os.path.isfile(args.language_model), 'invalid LM path: %s' % args.language_model
    assert os.path.isfile(args.translation_model), 'invalid TM path: %s' % args.translation_model
    assert os.path.isfile(args.source_language_model), 'invalid source LM path: %s' % args.source_language_model
    assert os.path.isfile(args.bleu_path), 'invalid bleu_path: %s' % args.bleu_path
    assert os.path.isfile(args.reference), 'invalid reference path: %s' % args.reference

    max_phrase_length = args.max_phrase_length
    language_model = decoder.read_language_model(args.language_model,
                                        max_phrase_length,
                                        label='LOADING LANGUAGEMODEL TARGET')

    source_language_model = decoder.read_language_model(
                                        args.source_language_model,
                                        max_phrase_length,
                                        label='LOADING LANGUAGEMODEL SOURCE')

    full_translation_model = read_full_translation_model(args.translation_model,
                                        max_phrase_length)

    max_lines = args.max_lines
    stack_limit = args.stack_limit
    beam_size = args.beam_size
    stupid_backoff = math.log(args.stupid_backoff)
    n_size = args.n_size
    feature_weights = args.feature_weights
    nbest = args.nbest
    empty_default = args.empty_default
    top_translations = args.top_translations
    bleu_path = args.bleu_path
    reference = args.reference

    d_args = DecodeArguments(input_file, output_file, language_model,
                 source_language_model, full_translation_model, max_lines,
                 beam_size, args.processes, max_phrase_length, stack_limit,
                 stupid_backoff, n_size, nbest, empty_default, top_translations,
                 bleu_path, reference)

    get_score = d_args.get_bleu_score

    print 'initial weights:'
    print feature_weights
    (best_weights, best_score), weight_scores = optimize(feature_weights,
                                                         get_score,
                                                         args.step_size,
                                                         args.min_score_diff,
                                                         args.min_step_size,
                                                         args.max_iterations,
                                                         weight_scores)
    print 'final weights:'
    print best_weights
    print 'final score: %f' % best_score

    weight_scores_name = output_file + '.weightscores'
    weight_scores_to_file(weight_scores, weight_scores_name)
    print 'weight_scores written to: %s' % weight_scores_name


if __name__ == '__main__':
    START_TIME = time.time()
    main()
    print 'Total time: %s seconds' % (time.time() - START_TIME,)

    #test_optimize()
    #test_get_bleu_score()
