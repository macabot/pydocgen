"""MT decoder
by Michael Cabot, Richard Rozeboom and Auke Wiggers"""

import argparse
import sys
from itertools import islice
from collections import defaultdict
import mp_worker
import heapq
import math
import time
import bisect
from utils import show_progress

def shortest_path(start, end):
    '''Graph is internal in states.'''
    def flatten(L):       # Flatten linked list of form [0,[1,[2,[]]]]
        while len(L) > 0:
            yield L[0]
            L = L[1]

    q = [(0, start, ())]  # Heap of (cost, path_head, path_rest).
    visited = set()       # Visited vertices.
    while True:
        (cost, v1, path) = heapq.heappop(q)
        if v1 not in visited:
            visited.add(v1)
            if v1 == end:
                return list(flatten(path))[::-1] + [v1]
            path = (v1, path)
            for (v2, cost2) in (v1.manybackpointers + ([v1.onebackpointer] \
                    if v1.onebackpointer else [])):
                if v2 not in visited:
                    # Take abs as heapq pops lowest number first
                    heapq.heappush(q, (cost + abs(cost2), v2, path))

def k_shortest_paths(shortestpath, targetnode, K=10, verbose=False):
    '''Yens algorithm for finding the K shortest paths in a noncyclic simple
    graph.
    '''
    shortestpaths = [shortestpath]
    candidates = []
    for k in range(1, K):
        to_restore = defaultdict(list)
        for i in range(0, len(shortestpaths[k-1])-1):
            spurnode = shortestpaths[k-1][i]
            rootpath = shortestpaths[k-1][0:i + 1]
            if verbose:
                print 'Spurnode      ', spurnode
                print 'Backpointers  ', spurnode.manybackpointers
                print 'Rootpath      ', rootpath
            for path in shortestpaths:
                # if current rootpath overlaps with any paths, del links
                if rootpath == path[0:i + 1]:
                    # these consist of the i+1th element of every path
                    next_node = path[i + 1]
                    # two cases
                    if verbose:
                        print "Removed link '{}'".format(next_node)
                    if spurnode.onebackpointer and spurnode.onebackpointer[0] == next_node:
                        to_restore[spurnode].append((next_node, spurnode.onebackpointer[1], True))
                        spurnode.onebackpointer = None
                    elif spurnode.manybackpointers:
                        for n, c in spurnode.manybackpointers:
                            if n == next_node:
                                to_restore[spurnode].append((n, c, False))
                                spurnode.manybackpointers.remove((n,c))
                                break
            # If a path is availablee after removing links
            if len(spurnode.manybackpointers) or spurnode.onebackpointer:
                spurpath = shortest_path(spurnode, targetnode)
                if verbose:
                    print 'Found new path'
                    print rootpath + spurpath[1:]
                    print '----------------------'
                candidates.append(rootpath + spurpath[1:])
            elif verbose:
                print 'No paths available for current spurnode'
                print '---------------------'
        # Last entry is the shortest
        candidates.sort(key=lambda x : len(x), reverse=True)
        if candidates:
            shortestpaths.append(candidates.pop())
        else:
            if verbose:
                print 'Only {k} path(s) available.'.format(k=k)
            return shortestpaths

        # Restore edges to graph
        for s, lst in to_restore.iteritems():
            for (next_s, cost, bp) in lst:
                if bp:
                    s.onebackpointer = (next_s, cost)
                else:
                    s.manybackpointers.append((next_s, cost))

    return shortestpaths


def window(seq, n=2):
    """Returns a sliding window (of width n) over data from the iterable
    s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ..."""
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def read_translation_model(file_name, feature_weights, top_translations,
        max_phrase_length):
    """Read the translation model"""
    translation_model = defaultdict(list)
    document = open(file_name, 'r')

    num_lines = sum(1 for line in open(file_name, 'r'))
    point = num_lines / 100 if num_lines > 100 else 1

    for i, line in enumerate(document):
        if i % point == 0:
            show_progress(i, num_lines, 40, 'LOADING TRANSLATIONMODEL')

        segments = line.strip().split(' ||| ')
        source = tuple(segments[0].split())
        if len(source) > max_phrase_length:
            continue
        target = tuple(segments[1].split())
        probs = [float(prob) for prob in segments[2].split()]
        # TODO what measure defines a good translation???
        measure = sum([prob * feature_weights[i] for i, prob in \
                       enumerate(probs)])
        if len(translation_model[source]) < top_translations:
            heapq.heappush(translation_model[source], (measure, target,
                                                       probs))
        else:
            heapq.heappushpop(translation_model[source], (measure,
                                                          target, probs))
    show_progress(1, 1, 40, 'LOADING TRANSLATIONMODEL')
    sys.stdout.write('\n')
    document.close()

    return {s: [(t, p) for (m, t, p) in mtp] for (s, mtp) in
            translation_model.iteritems()}

def read_language_model(file_name, max_phrase_length,
        label='LOADING LANGUAGEMODEL'):
    """Read the language model"""
    language_model = {}
    document = open(file_name, 'r')
    num_lines = sum(1 for line in open(file_name, 'r'))
    point = num_lines / 100 if num_lines > 100 else 1

    for i, line in enumerate(document):
        if i % point == 0:
            show_progress(i, num_lines, 40, label)
        segments = line.strip().split(' ||| ')
        phrase = segments[0]
        if len(phrase.split()) > max_phrase_length:
            continue
        prob = float(segments[1])
        language_model[tuple(phrase.split())] = prob
    show_progress(1, 1, 40, label)
    sys.stdout.write('\n')

    document.close()
    return language_model

def read_reordering_model(file_name):
    """Read the reordering model"""
    reordering_model = {}
    document = open(file_name, 'r')
    for line in document:
        segments = line.strip().split(' ||| ')
        source = tuple(segments[0].split())
        target = tuple(segments[1].split())
        reorder_probs = [float(prob) for prob in segments[2].split()]
        if source in reordering_model:
            reordering_model[source].append((target, reorder_probs))
        else:
            reordering_model[source] = [(target, reorder_probs)]

    document.close()
    return reordering_model

def get_language_model_prob(language_model, target_phrase, stupid_backoff):
    """Return the language model probability of a phrase. If ngram is unknown,
    recursively reduce history size and apply stupid backoff smoothing."""
    if target_phrase in language_model:
        return language_model[target_phrase]
    else:
        if len(target_phrase) > 1:
            # add backoff factor since the probs are in log-space
            prob = stupid_backoff + get_language_model_prob(language_model,
                target_phrase[1:], stupid_backoff)
            language_model[target_phrase] = prob # update language model
            return prob
        else:
            return -10000.0


def do_the_work(input_file, output_file, translation_model, language_model,
                source_language_model,
                max_lines, max_phrase_length, beam_size,
                stack_limit, stupid_backoff, n_size, feature_weights, nbest):
    """print local arguments
    TODO remove this function"""
    args = locals()
    for name in args.keys():
        print name.ljust(20), args[name]
    '''
    print 'output %s' % output_file
    print 'translation_model: %s' % translation_model
    print 'language_model: %s' % language_model
    print 'max_lines: %f' % max_lines
    print 'max_phrase_length: %f' % max_phrase_length
    print 'beam_size %s' % beam_size
    print 'stack_limit: %f' % stack_limit
    print 'beam_size: %f' % beam_size
    print 'feature_weights: %s' % feature_weights
    '''


def test_decode():
    language_model = {("it's",): 0, ("a",): 0, ("trap",): -1, ("tarp",): -1,
                      ("it's", "a"): -1, ("a", "trap"): -3, ("a", "tarp"): -3,
                      ("it's", "a", "trap"): 0, ("it's" ,"a", "tarp"): 0,
                      ("it", "is"): -1, ("it",): -1, ("is",): -2, ("the",):-2}
    source_language_model = {("c'est",): 0, ("un",): 0, ("phrase",): 0}
    """translation_model = {
            ("c'est", "un", "phrase"):
                [ (("it's", "a", "trap"), (-19, -19, -13, -13)),
                  (("it's", "a", "tarp"), (-19, -19, -14, -14)) ],
            ("c'est",) :
                [ (("it", "is"), [-4, -2, -1, -2])],
            ("un",):
                [ (("a",),   [-2,-3,-5,-1]),
                  (("the",), [-2,-3,-2,-1]) ],
            ("phrase",):
                [ (("trap",), [-1,-2,-2,-6]),
                  (("tarp",), [-1,-2,-3,-7]) ],
            }
    """
    translation_model = {
            ("c'est",)  : [ (("it", "is"),  [-4, -2, -1, -2])], # -9
            ("un",)     : [ (("a",),        [-2, -3, -5, -1])], # -11
            ("phrase",) : [ (("trap",),     [-1, -2, -2, -6])]} # -11
    translation = decode(("c'est", "un", "xxx", "phrase"), language_model,
        source_language_model, translation_model,
            beam_size=20, max_phrase_length=3, stack_limit=100,
            stupid_backoff = math.log(0.4),
            n_size = 3, feature_weights=8*[1.0], nbest=1,
            empty_default=True)
    print '|' + translation + '|'


class BeamStack():
    """Keeps track of DecoderStates that have covered the same number of source
    words. Restricts the number of DecoderStates with:
    Beam: score of new state cannot be to low compared to current best score in
            the stack
    Histogram filter: The stack has a maximum size. If a state is added and the
        stack is over capacity, the state with the lowest score is removed."""

    def __init__(self, beam_size, stack_limit):
        self.stack = []
        self.stack_limit = stack_limit
        self.beam_size = beam_size
        self.best_prob = float('-inf')

    def __eq__(self, other):
        if not isinstance(other, BeamStack):
            return False

        return set(self.stack) == set(other.stack)

    def __getitem__(self, key):
        return self.stack[key]

    def __delitem__(self, key):
        del self.stack[key]

    def __setitem__(self, key, value):
        self.stack[key] = value

    def __contains__(self, item):
        return item in self.stack

    def __iter__(self):
        return self.stack.__iter__()

    def __len__(self):
        return len(self.stack)

    def index(self, item):
        """Return index of item in stack"""
        return self.stack.index(item)

    def pop(self):
        """Pop item from stack"""
        return heapq.heappop(self.stack)

    def append(self, item):
        """Push item on stack if difference with best state score is smaller
        than beam size. If the stack has reached its capacity add item and
        remove item with lowest score. NOTE this means that all connections
        to this item should also be removed."""
        # apply beam
        if len(self.stack) > 0 and \
                item.total_cost < self.best_prob - self.beam_size:
            return
        # update best prob
        if item.total_cost > self.best_prob:
            self.best_prob = item.total_cost
            # beam prune states in self.stack
            self.stack.sort()
            stack_probs = [state.total_cost for state in self.stack]
            cutoff = bisect.bisect_left(stack_probs, item.total_cost)
            self.stack = self.stack[cutoff+1:]
            heapq.heapify(self.stack)

        # recombination
        pop_index = None
        for i, other_state in enumerate(self.stack):
            if other_state.coveragevector != item.coveragevector or \
                    other_state.history != item.history or \
                    other_state.last_pos != item.last_pos:
                continue
            if other_state.prob > item.prob:  # Case 1, other_state used
                other_state.recombinationpointers.append((item, item.prob - other_state.prob))
                return
            else:                             # Case 2, item used
                pop_index = i
                item.recombinationpointers.append((other_state, other_state.prob - item.prob))
                break
        if pop_index != None:
            self.stack.pop(pop_index)
            heapq.heapify(self.stack)

        # check parent state if manybackpointers needed
        for recmb_state, recmb_cost in item.onebackpointer[0].recombinationpointers:
            cost = item.onebackpointer[1] + recmb_cost
            item.manybackpointers.append((recmb_state, cost))

        # apply histogram filter
        if len(self.stack) >= self.stack_limit:
            heapq.heappushpop(self.stack, item)
        else:
            heapq.heappush(self.stack, item)

    def best_state(self):
        """Return state with the highest score"""
        return heapq.nlargest(1, self.stack)[0]

    def __str__(self):
        return str(self.stack)

    def __repr__(self):
        return self.__str__()

class DecoderState():
    """Current state in a derivation."""

    def __init__(self, prob, history, translation, coveragevector,
            onebackpointer, recombinationpointers, manybackpointers, last_pos, future_cost):
        self.prob = prob
        self.history = history
        self.translation = translation
        self.coveragevector = coveragevector
        self.onebackpointer = onebackpointer
        self.recombinationpointers = recombinationpointers
        self.manybackpointers = manybackpointers
        self.last_pos = last_pos
        self.future_cost = future_cost
        self.total_cost = self.prob + self.future_cost

    def __str__(self):
        return "{translation}: {prob:.6g}+{future:.6g}={total:.6g}".format(
            translation=self.translation, prob=self.prob,
            future=self.future_cost, total=self.total_cost)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, DecoderState):
            return False

        return self is other

    def __hash__(self):
        return id(self)

    def __ne__(self, other):
        return not self == other

    def __lt__(self, other):
        return self.total_cost < other.total_cost

    def __le__(self, other):
        return self.total_cost <= other.total_cost

    def __gt__(self, other):
        return self.total_cost > other.total_cost

    def __ge__(self, other):
        return self.total_cost >= other.total_cost


def decode(source_words, language_model, source_language_model,
        translation_model, beam_size, max_phrase_length, stack_limit, stupid_backoff, n_size, feature_weights, nbest, empty_default):
    """Decode one given sentence given models:

    return argmax_e p(f|e) . p(e)
    """

    # A decoder state contains:
    # - Coverage vector
    # - Translationprob so far
    # - Translation of the current phrase
    # - Last n-1 words (history)
    # - One backpointer to previous state
    # - Pointers to recombined states

    #future cost calculation
    future_cost_dict = calc_future_costs(translation_model, language_model,
        source_language_model, source_words, stupid_backoff, feature_weights)

    init_future_cost = future_cost_dict[(0, len(source_words)-1)]
    initial_state = DecoderState(prob=0.0,
                                 history=("<s>",),
                                 translation="<s>",
                                 coveragevector=[-1, len(source_words)],
                                 onebackpointer=(None, 0),
                                 recombinationpointers=[],
                                 manybackpointers=[],
                                 last_pos=-1,
                                 future_cost=init_future_cost)

    # For every len-of-word, there is a beamstack
    beamstacks = [BeamStack(beam_size, stack_limit) for _ in source_words]
    # initialize first beamstack
    for next_state in find_next_states(source_words, initial_state,
                                       language_model, translation_model,
                                       max_phrase_length, stupid_backoff,
                                       n_size, feature_weights,
                                       future_cost_dict, empty_default):
        beamstacks[len(next_state.coveragevector)-3].append(next_state)
    # continue decoding
    for i in xrange(len(beamstacks)):
        beamstack = beamstacks[i]
        for state in beamstack:
            for next_state in find_next_states(source_words, state,
                                               language_model,
                                               translation_model,
                                               max_phrase_length,
                                               stupid_backoff, n_size,
                                               feature_weights,
                                               future_cost_dict,
                                               empty_default):
                # '-3' is offset for '-1' and 'len(source_words)' in
                # coveragevector (cv), and index=0 corresponds to len(cv)=1
                beamstacks[len(next_state.coveragevector)-3].append(next_state)

    # Following the path from the best final state, we get the viterbi
    # translation. Consider only the very best final state (for now)
    viterbipath = []
    try:
        state = beamstacks[-1].best_state()
    except:
        return 'No translation possible'

    while state:
        viterbipath.append(state)
        state = state.onebackpointer[0]
    # Translation is now encoded in viterbipath in reverse order
    viterbi_translation = ' '.join(t for t in reversed([s.translation for s in viterbipath]) if t != '')
    if nbest == 1:
        return viterbi_translation

    # We could also get n-best list if the graph is known
    shortest_paths = k_shortest_paths(viterbipath, initial_state, nbest)
    return [' '.join(t for t in reversed([s.translation for s in p if s is not None]) if t != '') for p in shortest_paths]


def find_next_states(source_words, state, language_model, translation_model,
        max_phrase_length, stupid_backoff, n_size, weights, future_cost_dict,
        empty_default):
    """
    Costs are:
    1. Phrase translation
    2. LM Continuation
    3. Phrase penalty: phrase application costs -1
    4. Word penalty: compensate for lms favor towards short translations
    5: Linear distortion: see how much we deviate from regular left-to-right

    """
    phrase_penalty = -1 * weights[5]

    # Determine new possible candidates
    candidate_phrase_indices = []
    for i, (n, next_n) in enumerate(window(state.coveragevector, 2)):
        # Keep track of index for insertion
        if next_n - n > 1:
            candidate_phrase_indices.extend([(i + 1, left, right + 1)
                for left in xrange(n + 1, next_n)
                for right in xrange(left,
                                    min(next_n, left + max_phrase_length))])

    # If it is in the translation model, use its corresponding target phrases
    default = [(('',), (-10000.0, -10000.0, -10000.0, -10000.0))]
    for left_idx, phrase_start, phrase_end in candidate_phrase_indices:
        source_phrase = source_words[phrase_start:phrase_end]
        if len(source_phrase) == 1:
            if not empty_default:
                default = [(source_phrase, (-10000.0, -10000.0, -10000.0, -10000.0))]
        else:
            default = []
        for target_and_probs in translation_model.get(source_phrase, default):
            target_words, (pfe, pef, lfe, lef) = target_and_probs
            empty_target = (target_words == ('',))
            # 1. phrase translation
            phrase_translation_cost = weights[0]*pfe + weights[1]*pef + \
                                      weights[2]*lfe + weights[3]*lef
            # 2. LM continuation
            if empty_target:
                lm_continuation_cost = 0.0
            else:
                lm_continuation_cost = calc_lm_continuation(state.history,
                                                        target_words,
                                                        language_model,
                                                        n_size, stupid_backoff,
                                                        weights[4])
            # 3. Phrase penalty is -1
            # calculated once at the start of this function
            # 4. Word penalty is a bonus for lengthy phrases
            if empty_target:
                word_penalty = 0.0
            else:
                word_penalty = len(target_words) * weights[6]
            # 5. Linear distortion
            linear_distortion_cost = -abs(phrase_start - state.last_pos - 1) * \
                                     weights[7]

            transition_cost = phrase_penalty + phrase_translation_cost + \
                              word_penalty + lm_continuation_cost + \
                              linear_distortion_cost
            new_coveragevector = state.coveragevector[:left_idx] + \
                    range(phrase_start, phrase_end) + \
                    state.coveragevector[left_idx:]

            new_last_pos = phrase_end - 1
            future_cost = get_future_cost(future_cost_dict, new_coveragevector, new_last_pos, weights)
            if empty_target:
                new_history = state.history[-(n_size-1):]
            else:
                new_history = (state.history + target_words)[-(n_size-1):]
            newstate = DecoderState(
                    prob=(transition_cost + state.prob),
                    history=new_history,
                    translation=' '.join(target_words),
                    coveragevector=new_coveragevector,
                    onebackpointer=(state, transition_cost),
                    recombinationpointers=[],
                    manybackpointers=[],
                    last_pos=new_last_pos,
                    future_cost=future_cost)

            yield newstate

def calc_lm_continuation(history, target_words, language_model, n,
                         stupid_backoff, weight):
    sentence = history + target_words
    probs = 0
    for i in xrange(len(history), len(sentence)):
        ngram = tuple(sentence[max(i-n+1, 0):i+1])
        probs += weight * get_language_model_prob(language_model, ngram,
                                                  stupid_backoff)
    return probs


def calc_future_costs(TM, LMe, LMf,  source, stupid_backoff, weights):
    """Calculate all future costs"""
    future_cost = {}
    for i in xrange(0, len(source)):
        for j in xrange(i, len(source)):
            f = tuple(source[i:j+1])

            if f in TM:
                lm_probs = (0.0 if e[0]==('',) else weights[4] * get_language_model_prob(LMe, e[0], stupid_backoff) for e in TM[f])
                conditional_probs = (weights[0] * e[1][0] + 
                                     weights[1] * e[1][1] +
                                     weights[2] * e[1][2] +
                                     weights[3] * e[1][3] for e in TM[f])
                future_cost[i,j] = max(a + b for a, b in zip(lm_probs, conditional_probs))
            elif i == j:
                future_cost[i,j] =  -10.0 + weights[4] * \
                        get_language_model_prob(LMf, f, stupid_backoff)
            else:
                future_cost[i,j] = -10000.0
    for i in xrange(0,len(source)):
        for j in xrange(i+1, len(source)):
            future_cost[i,j] = max(max(future_cost[i,k]+future_cost[k+1, j] for k in xrange(i,j)),future_cost[i,j])

    return future_cost

def test_future_costs():
    """test calc_future_costs"""
    source = ['a', 'b', 'b', 'a']
    TM = {tuple('a'):[ (['x'], (-1, -1, -1, -1)),
                (['y'], (-1, -1, -1, -1)) ],
          tuple('b'):[ (['x'], (-1, -1, -1, -1)),
                (['y'], (-1, -1, -1, -1))]}

    LMe = {'x': -1, 'y': -2}
    LMf = {'a': -3, 'b': -4}
    weights = [1.0]*8
    sb = math.log(0.4)
    print calc_future_costs(TM, LMe, LMf, source, sb, weights)

def lm_test():
    """test calc_lm_continuation"""
    lm = {}
    lm[("a", "b")] = 0.1
    lm[("b", "c")] = 0.1
    lm[("c", "d")] = 0.1
    lm[("e", "f")] = 0.1
    lm[("f", "g")] = 0.1
    lm[("a", "b", "c")] = 0.1
    lm[("c", "d", "e")] = 0.1
    lm[("b", "c", "d")] = 0.1
    history = ["a"]
    print calc_lm_continuation(history, ["b","c"], lm, 3, 0.4, 1.0)

def test_decoder_stack():
    """test the decoder stack"""
    stack = BeamStack(10, 3)
    states = []
    probs = [-3, -2, -100, -9, -2]
    futures = [-2, -2, -2, -2, -2]
    temp_state = DecoderState(None, None, None, None, None, [], None, None, None)
    for i, prob in enumerate(probs):
        state = DecoderState(prob, None, None, None, (temp_state, None), [], None, None, futures[i])
        states.append(state)

    print stack
    for state in states:
        stack.append(state)
        for s in stack:
            print s
            print s.recombinationpointers
        print stack.best_prob

def test_feature_weights():
    """test feature weights"""
    source_words = ['a']
    state = DecoderState(0, ('',), '', [-1, 1], None, [], [], -1, -1)
    language_model = {('x',): -1, ('y',): -2}
    source_language_model = {('a',): -1}
    translation_model = {('a',):[ (('x',), (-1, -1, -1, -1)),
                               (('y',), (-1, -1, -1, -1)) ]}
    max_phrase_length = float('inf')
    stupid_backoff = 0
    n_size = 1
    weights = [float(w) for w in xrange(1, 9)] # should contain 8 floats
    future_cost_dict = calc_future_costs(translation_model, language_model,
        source_language_model, source_words, stupid_backoff, weights)
    print future_cost_dict
    for state in find_next_states(source_words, state, language_model,
                                  translation_model, max_phrase_length,
                                  stupid_backoff, n_size, weights,
                                  future_cost_dict, empty_default=False):
        print state

def get_future_cost(future_cost_dict, coverage, last_pos, weights):
    """Get the predicted future cost at a partial translation"""
    cost = 0
    # translation and language cost
    for i, j in ((coverage[i]+1, coverage[i+1]-1) for i in xrange(len(coverage)-1)):
        if i <= j:
            cost += future_cost_dict[i, j]
    # linear distortion cost
    distortion_cost = 0
    coverage_set = set(coverage)
    unvisited = [i for i in xrange(coverage[-1]) if i not in coverage_set]
    if len(unvisited) > 0:
        distortion_cost = -abs(unvisited[0] - last_pos - 1)
        distortion_cost += sum(-abs(unvisited[i] - unvisited[i-1] - 1)
                               for i in xrange(1, len(unvisited)))
        distortion_cost *= weights[7]

    return cost + distortion_cost


def main():
    """Read command line arguments."""
    NUM_FEATURES = 8

    arg_parser = argparse.ArgumentParser()
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
    arg_parser.add_argument("-w", "--feature_weights", nargs=NUM_FEATURES,
        type=float, default=NUM_FEATURES*[1.0],
        help="Feature weights. Order: p(f|e), p(e|f), l(f|e), l(e|f), lm(e),\
            phrase penalty, word_penalty, linear distortion.")
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

    args = arg_parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file
    language_model = read_language_model(args.language_model,
                                         args.max_phrase_length,
                                         label='LOADING LANGUAGEMODEL TARGET')
    translation_model = read_translation_model(args.translation_model,
                                               args.feature_weights,
                                               args.top_translations,
                                               args.max_phrase_length)

    source_language_model = read_language_model(args.source_language_model,
                                                args.max_phrase_length,
                                                label='LOADING LANGUAGEMODEL SOURCE')
    max_lines = args.max_lines
    max_phrase_length = args.max_phrase_length
    stack_limit = args.stack_limit
    beam_size = args.beam_size
    stupid_backoff = math.log(args.stupid_backoff)
    n_size = args.n_size
    feature_weights = args.feature_weights
    nbest = args.nbest
    empty_default = args.empty_default

    if args.processes < 1:
        do_the_work(input_file, output_file, translation_model, language_model, source_language_model,
                max_lines, max_phrase_length, beam_size,
                stack_limit, stupid_backoff, n_size, feature_weights, nbest)
    else:
        mp_worker.set_up_decoders(input_file, output_file, language_model,
                                  source_language_model,
                                  translation_model, max_lines,
                                  beam_size, args.processes, max_phrase_length,
                                  stack_limit, stupid_backoff, n_size,
                                  feature_weights, nbest, empty_default)


if __name__ == '__main__':
    START_TIME = time.time()
    main()
    print "Time taken to run:"
    print time.time() - START_TIME, "seconds"

    #test_decode()
    #test_decoder_stack()
    #test_feature_weights()
    #test_future_costs()
