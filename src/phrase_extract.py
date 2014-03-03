import argparse
from collections import defaultdict
import mp_worker
import sys
import math
import os
from utils import show_progress

def _log(*args):
    print ' '.join((str(a) for a in args))

def reordering_freq_to_prob(phrase_pair_to_counts, log_bool):
    """this def assumes freqs to be ordered 4-4 (either l-r or r-l is fine)"""
    reordering_probs = {}
    for phrase_pair, counts in phrase_pair_to_counts.iteritems():
        counts = map(float, counts) # to floats
        first_model = counts[0:4]
        second_model = counts[4:8]
        first_model_total = sum(first_model)
        second_model_total = sum(second_model)
        if first_model_total != 0:
            first_model = [freq/first_model_total for freq in first_model]
        if second_model_total != 0:
            second_model = [freq/second_model_total for freq in second_model]

        if log_bool:
            first_model = [math.log(prob) if prob != 0 else float('-inf') for
                           prob in first_model]
            second_model = [math.log(prob) if prob != 0 else float('-inf') for
                            prob in second_model]

        reordering_probs[phrase_pair] = first_model + second_model

    return reordering_probs

def extract_lexical_reordering_counts(alignments_file, source_file,
                                target_file, max_length, max_lines=None):
    """
    for the left-to-right and right-to-left models calculate:
    c(m,(f,e)), c(s,(f,e)), c(d_l,(f,e)), c(d_r,(f,e))
    where m=monotone, s=swap, d_l=left-discontinuous, d_r=right-discontinuous
    """
    # open files
    num_lines = sum(1 for line in open(alignments_file))
    max_lines = int(max_lines) if max_lines else num_lines

    alignments = open(alignments_file, 'r')
    source = open(source_file, 'r')
    target = open(target_file, 'r')

    reordering_counts = {} # maps phrase pair to its reordering counts

    point = max_lines / 100 if max_lines > 100 else 1
    for i, str_align in enumerate(alignments):
        if i % point == 0:
            show_progress(i, max_lines, 40, 'LEXICAL REORDERING')
        if i == max_lines:
            break

        source_words = source.next().strip().split()
        target_words = target.next().strip().split()
        source_length = len(source_words)
        target_length = len(target_words)

        align = str_to_alignments(str_align)
        word_phrase_pairs = set([word_pair*2 for word_pair in align])
        # phrase to internal is a dict mapping phrase ranges (source_min,
        # target_min, source_max, target_max) to internal word alignments []
        phrase_to_internal = extract_alignments(set(align), source_length,
                                               target_length, max_length)

        try:
            phrase_pairs = set(phrase_to_internal.keys())
            for left_phrase_range in phrase_pairs:
                # phrase based counting events
                for right_phrase_range in phrase_pairs - word_phrase_pairs:
                    update_count(left_phrase_range, right_phrase_range,
                        reordering_counts, source_words, target_words)

                # word based counting events
                for right_phrase_range in word_phrase_pairs:
                    update_count(left_phrase_range, right_phrase_range,
                        reordering_counts, source_words, target_words)

        except:
            print 'source: \n%s' % ' '.join(source_words)
            print 'target: \n%s' % ' '.join(target_words)
            print 'alignment: \n%s' % str_align
            raise

    show_progress(max_lines, max_lines, 40, 'LEXICAL REORDERING')
    return reordering_counts

def update_count(left_phrase_range, right_phrase_range, reordering_counts,
                 source_words, target_words):
    l_source_min, l_target_min, l_source_max, l_target_max = left_phrase_range
    r_source_min, r_target_min, r_source_max, r_target_max = right_phrase_range
    if l_target_max != r_target_min-1:
        return

    reordering = determine_reordering(left_phrase_range, right_phrase_range)
    if reordering == None:
        return

    l_source_phrase = ' '.join(source_words[l_source_min:l_source_max+1])
    l_target_phrase = ' '.join(target_words[l_target_min:l_target_max+1])
    l_phrase_pair = (l_source_phrase, l_target_phrase)
    r_source_phrase = ' '.join(source_words[r_source_min:r_source_max+1])
    r_target_phrase = ' '.join(target_words[r_target_min:r_target_max+1])
    r_phrase_pair = (r_source_phrase, r_target_phrase)

    if l_phrase_pair not in reordering_counts:
        reordering_counts[l_phrase_pair] = 8*[0]

    reordering_counts[l_phrase_pair][reordering] += 1

    if r_phrase_pair not in reordering_counts:
        reordering_counts[r_phrase_pair] = 8*[0]

    reordering_counts[r_phrase_pair][reordering+4] += 1

def extract_phrase_pair_freqs(alignments_file, source_file,
                              target_file, max_length,
                              max_lines=None):
    """Extract and count the frequency of all phrase pairs given an
    alignment between sentences.

    Keyword arguments:
    alignments_file -- file that contains the alignments
    source_file -- file containing sentences from language 1
    target_file -- file containing sentences from language 2
    max_length -- maximum length of phrase pairs
    max_lines -- maximum number of lines to use for phrase pair extraction

    Returns counts of phrase-pairs, counts of phrases in source
            and counts of phrases in target
    ((phrase_pair_freqs, source_phrase_freqs, target_phrase_freqs),
            (lex_pair_freqs, source_lex_freqs, target_lex_freqs),
            phrase_to_internals)
    Returns (3-tuple):
        phrase pair frequencies (3-tuple):
            pair frequencies, source frequencies, target frequencies
        lexical (word pair) frequencies (3-tuple):
            pair frequencies, source frequencies, target frequencies
        internal alignments for phrase pairs
    """



    # phrase frequencies
    phrase_pair_freqs = defaultdict(int)
    source_phrase_freqs = defaultdict(int)
    target_phrase_freqs = defaultdict(int)

    # lexical frequencies
    lex_pair_freqs = defaultdict(int)
    source_lex_freqs = defaultdict(int)
    target_lex_freqs = defaultdict(int)

    # map phrase pair to possible internal word alignments
    phrase_to_internals = defaultdict(set)

    # open files
    num_lines = sum(1 for line in open(alignments_file))
    max_lines = int(max_lines) if max_lines else num_lines

    alignments = open(alignments_file, 'r')
    source = open(source_file, 'r')
    target = open(target_file, 'r')

    point = max_lines / 100 if max_lines > 100 else 1
    for i, str_align in enumerate(alignments):
        if i % point == 0:
            show_progress(i, max_lines, 40, 'PHRASE EXTRACTION')
        if i == max_lines:
            break

        # read files
        source_words = source.next().strip().split()
        target_words = target.next().strip().split()
        source_length = len(source_words)
        target_length = len(target_words)

        align = str_to_alignments(str_align)
        # word pair frequencies
        for source_index, target_index in align:
            word_pair = (source_words[source_index], target_words[target_index])
            lex_pair_freqs[word_pair] += 1
            source_lex_freqs[word_pair[0]] += 1
            target_lex_freqs[word_pair[1]] += 1
        
        
        phrase_to_internal = extract_alignments(set(align), source_length,
                                                target_length, max_length)

        for phrase_pair, internal_alignment in extract_phrase_pairs_gen(
                                                    phrase_to_internal,
                                                    source_words,
                                                    target_words):
            # phrase pair frequencies
            phrase_pair_freqs[phrase_pair] += 1
            source_phrase_freqs[phrase_pair[0]] += 1
            target_phrase_freqs[phrase_pair[1]] += 1
            
            # phrase pair to possible internal word alignments
            phrase_to_internals[phrase_pair].add(frozenset(internal_alignment))

        unaligned, unaligned2 = unaligned_words(align, source_length, target_length)
        unaligned.extend(unaligned2)
        for phrase_pair in unaligned_phrase_pairs_gen(unaligned, source_words,
                                                      target_words):
            lex_pair_freqs[phrase_pair] += 1
            source_lex_freqs[phrase_pair[0]] += 1
            target_lex_freqs[phrase_pair[1]] += 1

    show_progress(max_lines, max_lines, 40, 'PHRASE EXTRACTION')
    sys.stdout.write('\n')

    alignments.close()
    source.close()
    target.close()

    return ((phrase_pair_freqs, source_phrase_freqs, target_phrase_freqs),
            (lex_pair_freqs, source_lex_freqs, target_lex_freqs),
            phrase_to_internals)

def conditional_probabilities(phrase_pair_freqs, source_phrase_freqs,
                              target_phrase_freqs, label, logprob):
    """Calculate conditional probability of phrase pairs in both directions.

    Input:
    phrase_pair_freqs -- counts of phrase pairs
    source_phrase_freqs -- counts of phrases in language 1
    target_phraes_freqs -- counts of phrases in lanuage 2
    label -- used to indicate current process
    logprob -- boolean, if true, probabilities are used in log-form

    Returns 2 dictionaries mapping phrase pair to P(source|target) and
    P(target|source)
    """
    source_given_target = {}
    target_given_source = {}
    num_phrases = len(phrase_pair_freqs)
    point = num_phrases / 100 if num_phrases > 100 else 1

    prob = lambda f1, f2: math.log(float(f1) / f2) if logprob else \
            lambda f1, f2: float(f1) / f2

    for i, (phrase_pair, freq) in enumerate(phrase_pair_freqs.iteritems()):
        if i % point == 0:
            show_progress(i, num_phrases, 40, label)
        try:
            source_given_target[phrase_pair] = prob(freq, source_phrase_freqs[phrase_pair[0]])
            target_given_source[phrase_pair] = prob(freq, target_phrase_freqs[phrase_pair[1]])
        except:
            _log('phrase pair : {}\ni : {}'.format(phrase_pair, i))
            raise

    show_progress(num_phrases, num_phrases, 40, label)
    sys.stdout.write('\n')
    return source_given_target, target_given_source

def lexical_weights(phrase_to_internals,
                    lex_source_given_target,
                    lex_target_given_source,
                    target_lex_freqs):
    """
    p_w(f|e) = max_{a} Prod_{i=1}^n  1 / (j | (i, j) from a) sum w(f_i|e_j)
    """
    source_given_target = {}
    target_given_source = {}

    def weight_l1_given_l2(l1_phrase, l2_phrase, alignment, l1_given_l2,
                           reverse, logprob=True):
        """calculate lexical weight for source|target or target|source"""
        weight = 0
        alignment = [(a, b) for (b, a) in alignment] if reverse else alignment

        _sum = sum_logs if logprob else sum

        for i, l1_word in enumerate(l1_phrase):
            # Determine all words in target phrase aligned to i
            aligned_to_i = [b for a, b in alignment if a == i]
            if not aligned_to_i:                 # Handle non-aligned words
                pair = ('NULL', l1_word) if reverse else (l1_word, 'NULL')
                p_l1_given_l2 = l1_given_l2[pair]
            elif len(aligned_to_i) == 1:      # Case added for speed
                l2_word = l2_phrase[aligned_to_i[0]]
                pair = (l2_word, l1_word) if reverse else (l1_word, l2_word)
                p_l1_given_l2 = l1_given_l2[pair]
            else:
                # ONELINERS ON MULTIPLE LINES FTW
                list_of_probs = []
                for j in aligned_to_i:
                    pair = (l2_phrase[j], l1_word) if reverse else (l1_word, l2_phrase[j])
                    list_of_probs.append(l1_given_l2[pair])
                p_l1_given_l2 = _sum(list_of_probs)

                if logprob:
                    p_l1_given_l2 += math.log(1.0 / len(aligned_to_i))
                else:
                    p_l1_given_l2 /= len(aligned_to_i)

                if p_l1_given_l2 > 0 and logprob or p_l1_given_l2 > 1 and not logprob:
                    print p_l1_given_l2, l1_word, [l2_phrase[j] for j in aligned_to_i]
                    raise

                #p_l1_given_l2 = \
                #    _sum([l1_given_l2[((l2_phrase[j], l1_word) if reverse \
                #                       else (l1_word, l2_phrase[j]))]
                #          for j in aligned_to_i]) + \
                #     (1 / math.log(len(aligned_to_i)))

            # Weight is the product of prob for each word
            if logprob:
                weight += p_l1_given_l2
            else:
                weight *= p_l1_given_l2

        if weight > 1:
            print weight
            print l1_phrase, l2_phrase,  alignment, reverse
            raise
        return weight

    num_phrases = len(phrase_to_internals)
    point = num_phrases / 100 if num_phrases > 100 else 1

    for i, (phrase_pair, possible_internals) in enumerate(
            phrase_to_internals.iteritems()):
        if i % point == 0:
            show_progress(i, num_phrases, 40, 'LEXICAL WEIGHTS')

        weight_source_given_target = float('-inf')
        weight_target_given_source = float('-inf')
        source_phrase = phrase_pair[0].split()
        target_phrase = phrase_pair[1].split()
        for internal in possible_internals:
            # Calc weight for the current alignment
            temp_weight_source_given_target = \
                weight_l1_given_l2(source_phrase, target_phrase, internal,
                                   lex_source_given_target, reverse=False)
            # Reverse alignment for target_given_source
            temp_weight_target_given_source = \
                weight_l1_given_l2(target_phrase, source_phrase, internal,
                                   lex_target_given_source, reverse=True)
            if temp_weight_source_given_target > weight_source_given_target:
                weight_source_given_target = temp_weight_source_given_target
            if temp_weight_target_given_source > weight_target_given_source:
                weight_target_given_source = temp_weight_target_given_source

        source_given_target[phrase_pair] = weight_source_given_target
        target_given_source[phrase_pair] = weight_target_given_source

    show_progress(num_phrases, num_phrases, 40, 'LEXICAL WEIGHTS')
    sys.stdout.write('\n')

    return source_given_target, target_given_source

def add_phrase_alignment(collection, phrase, internal_alignment, max_length,
                         source_length, target_length):
    """Add a phrase alignment to a collection if:
    - its length is smaller or equal to the max length
    - the alignment is a contituent of the sentences

    Keyword arguments:
    collection -- a list or set
    phrase -- a 4-tuple (min1,min2,max1,max2) denoting the range of
    the constituents in language 1 and 2
    internal_alignment -- list of 2-tuples denoting the internal word alignment
    of the phrase
    max_length -- the maximum length of a phrase in the phrase alignment
    source_length -- the length of the sentence in language 1
    target_length -- the length of teh sentence in language 2
    """
    if phrase != None and phrase[2] - phrase[0]+1 <= max_length \
            and phrase[3] - phrase[1]+1 <= max_length \
            and phrase[0] >= 0 and phrase[1] >= 0 \
            and phrase[2] < source_length and phrase[3] < target_length:
        collection[phrase] = internal_alignment

def extract_phrase_pairs_gen(phrase_to_internal, source_words, target_words):
    """Given alignments:
    (1) extract phrase pairs from 2 sentences
    (2) shift internal alignment such that the index starts with index 0
        e.g. set([(4,4), (4,5), (5,4)]) becomes set([(0,0), (0,1), (1,0)])

    Keyword arguments:
    phrase_to_internal -- dict that maps a phrase alignment to its underlying
    word alignment. A phrase alignment is a 4 tuple denoting the range of the
    constituents.
    source_words -- words in the source sentence.
    target_words -- words in the target sentence.

    Yield a 2-tuple containing a phrase pair and a shifted internal alignment
    """
    for (min1, min2, max1, max2), internal in phrase_to_internal.iteritems():
        #s/t is source/target word-index
        shifted_internal = set([(s-min1 if s!=None else None,
                                t-min2 if t!=None else None)
                                for (s, t) in internal])
        yield ((' '.join(source_words[min1:max1+1]),
               ' '.join(target_words[min2:max2+1])),
               shifted_internal)

def unaligned_phrase_pairs_gen(unaligned, source_words, target_words):
    """For unaligned words create an alignment with 'NULL'."""
    for (a1, a2) in unaligned:
        if a1 == None:
            yield ('NULL', target_words[a2])
        elif a2 == None:
            yield (source_words[a1], 'NULL')

def str_to_alignments(string):
    """Parse an alignment from a string

    Keyword arguments:
    string -- contains alignment

    Return a set of 2-tuples. First value is index of word in language 1
    second value is index of word in language 2
    """
    string_list = string.strip().split()
    alignments = []
    for a_str in string_list:
        a1_str, a2_str = a_str.split('-')
        alignments.append((int(a1_str), int(a2_str)))

    return alignments

def internal_alignment_expansions(internal_alignments, word_alignments, max_length):
    """For each language find the alignments belonging to the words that are
    not covered with the given internal word alignment."""
    min1, min2, max1, max2 = phrase_range(internal_alignments)
    if max1-min1+1 > max_length or max2-min2+1 > max_length:
        return set([])

    return set([(a1, a2) for (a1, a2) in word_alignments
        if (a1, a2) not in internal_alignments and
        (min1 <= a1 <= max1 or min2 <= a2 <= max2)])

def phrase_range(internal_alignments):
    """Calculate the range of a phrase's internal word alignment.

    Keyword arguments:
    internal_alignments -- list of 2-tuples denoting the alignment between words

    Returns a 4-tuples denoting the range of the phrase alignment
    """
    min1 = min2 = float('inf')
    max1 = max2 = float('-inf')
    for (a1, a2) in internal_alignments:
        if a1 < min1:
            min1 = a1
        if a1 > max1:
            max1 = a1
        if a2 < min2:
            min2 = a2
        if a2 > max2:
            max2 = a2

    return min1, min2, max1, max2

def is_valid_phrase_alignment((min1, min2, max1, max2), word_alignments,
        max_length):
    """Returns true if there are word alignments that should be part of the
    phrase range"""
    if max1-min1+1 > max_length or max2-min2+1 > max_length:
        return False

    word_align_slice = [(a1, a2)
                        for (a1, a2) in word_alignments
                        if min1 <= a1 <= max1 or min2 <= a2 <= max2]
    if len(word_align_slice) == 0:
        return False

    for (a1, a2) in word_alignments:
        if (a1, a2) not in word_align_slice and \
                (min1 <= a1 <= max1 or min2 <= a2 <= max2):
            return False

    return True

def is_valid_phrase(word_align_slice, word_alignments, max_length):
    """Check whether a span is contigious"""
    if len(word_align_slice) == 0:
        return False

    min1, min2, max1, max2 = phrase_range(word_align_slice)
    if max1-min1+1 > max_length or max2-min2+1 > max_length:
        return False

    for (a1, a2) in word_alignments:
        if (a1, a2) not in word_align_slice and \
                (min1 <= a1 <= max1 or min2 <= a2 <= max2):
            return False

    return True

def extract_alignments(word_alignments, source_length, target_length, max_length):
    """Extracts all alignments between 2 sentences given a word alignment

    Keyword arguments:
    word_alignemnts -- set of 2-tuples denoting alignment between words in
    2 sentences
    source_length -- length of sentence 1
    target_length -- length of sentence 2
    max_length -- maximum length of a phrase pair

    Returns set of 4-tuples denoting the range of phrase_alignments
    """
    phrase_queue = {} # maps phrase alignment to internal word alignment
    #copy to use later for singletons
    word_alignments_orig = set(word_alignments)
    # First form words into phrase pairs
    for word_alignment in word_alignments:
        internal_alignment = set([word_alignment])
        temp_word_alignments = set(word_alignments_orig)
        expansion_points = internal_alignment_expansions(internal_alignment,
            temp_word_alignments, max_length)
        while expansion_points:
            internal_alignment |= expansion_points
            temp_word_alignments -= expansion_points
            expansion_points = internal_alignment_expansions(internal_alignment,
                temp_word_alignments, max_length)

        align_range = phrase_range(internal_alignment)
        add_phrase_alignment(phrase_queue, align_range, internal_alignment,
                             max_length, source_length, target_length)

    # loop over phrase pairs to join them together into new ones
    phrase_to_internal = {}
    while len(phrase_queue):
        p1, i1 = phrase_queue.popitem()
        #add unaligned indexes to phrase
        new_p3 = {}
        if not any(x==p1[0]-1 for (x, y) in word_alignments_orig):
            p3 = p1[0]-1, p1[1], p1[2], p1[3]
            add_phrase_alignment(new_p3, p3, i1, max_length,
                                 source_length, target_length)
        if not any(x==p1[2]+1 for (x, y) in word_alignments_orig):
            p3 = p1[0], p1[1], p1[2]+1, p1[3]
            add_phrase_alignment(new_p3, p3, i1, max_length,
                                 source_length, target_length)
        if not any(y==p1[1]-1 for (x, y) in word_alignments_orig):
            p3 = p1[0], p1[1]-1, p1[2], p1[3]
            add_phrase_alignment(new_p3, p3, i1, max_length,
                                 source_length, target_length)
        if not any(y==p1[3]+1 for (x, y) in word_alignments_orig):
            p3 = p1[0], p1[1], p1[2], p1[3]+1
            add_phrase_alignment(new_p3, p3, i1, max_length,
                                 source_length, target_length)

        # combine phrase alignments
        for p2, i2 in phrase_queue.iteritems():
            p3 = combine_phrase_alignments(p1, p2)
            i3 = i1 | i2
            p3_and_i3 = fix_phrase_alignment(p3, i3, word_alignments_orig,
                                          max_length)
            if p3_and_i3 == None:
                continue

            p3, i3 = p3_and_i3
            if p3 != p1:
                add_phrase_alignment(new_p3, p3, i3, max_length, source_length,
                                     target_length)

        phrase_to_internal[p1] = i1
        phrase_queue.update(new_p3)

    # add word alignments
    #phrase_to_internal.update(dict((phrase_range([a]), [a])
    #                               for a in word_alignments_orig))

    return phrase_to_internal

def unaligned_words(word_alignments, source_length, target_length):
    """Find indexes of unaligned words in source and target sentence."""
    aligned1 = set([])
    aligned2 = set([])
    for (a1, a2) in word_alignments:
        aligned1.add(a1)
        aligned2.add(a2)

    unaligned1 = [(a1, None) for a1 in set(range(source_length)) - aligned1]
    unaligned2 = [(None, a2) for a2 in set(range(target_length)) - aligned2]

    return unaligned1, unaligned2

def combine_phrase_alignments(p1, p2):
    """Combine two phrase alignments."""
    return (min(p1[0], p2[0]), min(p1[1], p2[1]),
        max(p1[2], p2[2]), max(p1[3], p2[3]))

def fix_phrase_alignment(phrase, internal_alignment, word_alignments,
                         max_length):
    """Fix discontinuous phrase alignments."""
    expansion_points = [(a1, a2) for (a1, a2) in word_alignments
        if partial_in_alignment((a1, a2), phrase)]
    while expansion_points:
        expansion_range = phrase_range(expansion_points)
        phrase = combine_phrase_alignments(phrase, expansion_range)
        if phrase[2]-phrase[0]+1 > max_length or \
                phrase[3]-phrase[1]+1 > max_length:
            return None

        expansion_points = [(a1, a2) for (a1, a2) in word_alignments
            if partial_in_alignment((a1, a2), phrase)]

    internal_alignment = set([(a1, a2) for (a1, a2) in word_alignments
        if phrase[0] <= a1 <= phrase[2] and phrase[1] <= a2 <= phrase[3]])
    return phrase, internal_alignment

def word_in_alignment(word, alignment):
    """Check whether a word alignment is inside a phrase alignment."""
    return alignment[0] <= word[0] <= alignment[2] and \
        alignment[1] <= word[1] <= alignment[3]

def partial_in_alignment(word, alignment):
    """Check if source xor target word is inside the alignment."""
    return (alignment[0] <= word[0] <= alignment[2]) != \
           (alignment[1] <= word[1] <= alignment[3])

def get_phrase_pair_alignments(file_name):
    """Read phrase pair alignments from a file"""
    phrase_pairs = {}
    phrase_pairs_file = open(file_name, 'r')
    for line in phrase_pairs_file:
        phrase1, phrase2, alignment = line.strip().split(" ||| ")
        phrase_pairs[(phrase1, phrase2)] = alignment

    phrase_pairs_file.close()
    return phrase_pairs

def sum_logs(log_list):
    """Sum log values in normal space
    e.g. sum_logs([log(0.1), log(0.2), log(0.3)]) = log(0.1 + 0.2 + 0.3)"""
    if len(log_list) == 1:
        return log_list[0]

    a = log_list.pop()
    b = log_list.pop()
    if b < a:
        new_log = a + math.log(1 + math.exp(b-a))
    else:
        new_log = b + math.log(1 + math.exp(a-b))

    log_list.append(new_log)
    return sum_logs(log_list)

def write_phrases_to_file(filename, phrase_pair_freqs, source_phrase_freqs,
                  target_phrase_freqs):
    """
    Write to given file in the following format
    f ||| e ||| freq(f) freq(e) freq(f,e)
    """
    with open(filename + '_extracted_phrases.txt', 'w') as outputfile:
        for (source_phrase, target_phrase), freq in phrase_pair_freqs.iteritems():
            outputfile.write("{f} ||| {e} ||| {ff} {fe} {ffe}\n".format(
                f=source_phrase, e=target_phrase,
                ff=source_phrase_freqs[source_phrase],
                fe=target_phrase_freqs[target_phrase],
                ffe=freq))
        sys.stdout.write('Saved to file: %s\n' % outputfile.name)

def write_translationprobs_to_file(filename, source_given_target,
                                   target_given_source):
    """
    Write to given file in the following format:
    f ||| e ||| p(f|e) p(e|f)
    """
    with open(filename + '_translation_probs_phrases.txt', 'w') as outputfile:
        for phrase_pair in source_given_target:
            outputfile.write("{f} ||| {e} ||| {pfe} {pef}\n".format(
                f=phrase_pair[0], e=phrase_pair[1],
                pfe=source_given_target[phrase_pair],
                pef=target_given_source[phrase_pair]))
        sys.stdout.write('Saved to file: %s\n' % outputfile.name)

def write_lexweights_to_file(outputfile,
                           phrase_source_given_target,
                           phrase_target_given_source,
                           lex_weight_source_given_target,
                           lex_weight_target_given_source):
    """
    Write to given file in the following format:
    f ||| e ||| p(f|e) p(e|f) l(f|e) l(e|f)
    """
    with open(outputfile + '_lexprobs_phrases.txt', 'w') as outputfile:
        for phrase_pair in lex_weight_source_given_target.iterkeys():
            outputfile.write("{f} ||| {e} ||| {pfe} {pef} {lfe} {lef}\n".format(
                f=phrase_pair[0], e=phrase_pair[1],
                pfe=phrase_source_given_target[phrase_pair],
                pef=phrase_target_given_source[phrase_pair],
                lfe=lex_weight_source_given_target[phrase_pair],
                lef=lex_weight_target_given_source[phrase_pair]))
        sys.stdout.write("Saved to file: %s\n" % outputfile.name)

def all_phrase_info_to_file(outputfile,
                            phrase_source_given_target,
                            phrase_target_given_source,
                            lex_weight_source_given_target,
                            lex_weight_target_given_source,
                            source_phrase_freqs,
                            target_phrase_freqs,
                            phrase_pair_freqs):
    with open(outputfile + '_all_info.txt', 'w') as out:
        for phrase_pair in phrase_pair_freqs:
            out.write('{f} ||| {e} ||| {pfe} {pef} {lfe} {lef} ||| {freqf} {freqe} {freqfe}\n'.format(
                f=phrase_pair[0], e=phrase_pair[1],
                pfe=phrase_source_given_target[phrase_pair],
                pef=phrase_target_given_source[phrase_pair],
                lfe=lex_weight_source_given_target[phrase_pair],
                lef=lex_weight_target_given_source[phrase_pair],
                freqf=source_phrase_freqs[phrase_pair[0]],
                freqe=target_phrase_freqs[phrase_pair[1]],
                freqfe=phrase_pair_freqs[phrase_pair]))

        sys.stdout.write("Saved phrase info to: %s\n" % out.name)

def load_phrases_from_file(name):
    _log('Trying to load data from file ' + name + '.txt')
    num_lines = sum(1 for line in open(name, 'r'))
    with open(name, 'r') as content_file:

        phrase_source_given_target = {}
        phrase_target_given_source = {}
        lex_weight_source_given_target = {}
        lex_weight_target_given_source = {}
        source_phrase_freqs = {}
        target_phrase_freqs = {}
        phrase_pair_freqs = {}

        point = num_lines / 100 if num_lines > 100 else 1

        for i, line in enumerate(content_file):
            words = line.strip().split('|||')
            f = words[0]
            e = words[1]

            first_values = words[2].split() #after first |||
            pfe = float(first_values[0])
            pef = float(first_values[1])
            lfe = float(first_values[2])
            lef = float(first_values[3])

            second_values = words[3].split() #after second |||
            freqf = int(second_values[0])
            freqe = int(second_values[1])
            freqfe = int(second_values[2])

            phrase_source_given_target[f, e] = pfe
            phrase_target_given_source[f, e] = pef
            lex_weight_source_given_target[f, e] = lfe
            lex_weight_target_given_source[f, e] = lef
            source_phrase_freqs[f, e] = freqf
            target_phrase_freqs[f, e] = freqe
            phrase_pair_freqs[f, e] = freqfe

            if i % point == 0:
                show_progress(i, num_lines, 40, 'LOADING PHRASES')

            #print(f+' ||| '+e+' ||| '+pfe+' '+pef+' '+lfe+' '+lef+' ||| '+freqf+' '+freqe+' '+freqfe+'\n')

        return (phrase_source_given_target, phrase_target_given_source,
            lex_weight_source_given_target, lex_weight_target_given_source,
            source_phrase_freqs, target_phrase_freqs, phrase_pair_freqs)

def file_exists(name):
    """Return true if file exists"""
    try:
        with open(name) as _: return True
    except:
        return False

def determine_reordering(left_pair, right_pair):
    """phrase_range = source_min, target_min, source_max, target_max"""
    monotone, swap, left_discontinuous, right_discontinuous = 0, 1, 2, 3
    left_source_min, _, left_source_max, _ = left_pair
    right_source_min, _, right_source_max, _ = right_pair
    if left_source_max == right_source_min-1:
        return monotone
    elif left_source_min == right_source_max+1:
        return swap
    elif left_source_max < right_source_min:
        return right_discontinuous
    elif left_source_min > right_source_max:
        return left_discontinuous
    else:
        return None

def main():
    """Read command line arguments and extract phrases."""
    print "We're translating from dutch (source) to english (target). "
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-a", "--alignments", required=True,
        help="File containing alignment")
    arg_parser.add_argument("-s", "--source", required=True,
        help="File containing sentences of source language")
    arg_parser.add_argument("-t", "--target", required=True,
        help="File containing sentences of target language")
    arg_parser.add_argument("-o", "--output", required=True,
        help="Output filename")
    arg_parser.add_argument("-ml", "--max_lines", type=int, 
        default=float('inf'), help="Maximum number of lines to parse")
    arg_parser.add_argument("-mp", "--max_phrase_length", type=int, 
        default=float('inf'), help="Maximum phrase pair length")
    arg_parser.add_argument('-pr', '--processes', type=int, default=1,
        help="Number of processes to use, default 1 (single process)")
    arg_parser.add_argument('-r', '--calc_reordering', action='store_true', 
        default = False, help='If true: calculate reordering probabilities. \
        Else calculate conditional probabilities.')
    args = arg_parser.parse_args()

    alignments = args.alignments
    assert os.path.isfile(alignments), 'invalid alignment path: %s' % alignments
    source = args.source
    assert os.path.isfile(source), 'invalid source path: %s' % source
    target = args.target
    assert os.path.isfile(target), 'invalid target path: %s' % target
    outputfile = args.output
    output_folder, output_name = os.path.split(outputfile)
    assert os.path.isdir(output_folder), 'invalid output folder: %s' % output_folder
    assert output_name.strip() != '', 'empty output name'
    max_length = args.max_phrase_length
    max_lines = args.max_lines
    processes = args.processes

    outputfile = '{out}_{ml}lines_{mpl}phraselength'.format(
        out=outputfile,
        ml=max_lines if max_lines!=None else 'all',
        mpl=max_length if max_length!=float('inf') else 'all')
    if args.calc_reordering:
        do_the_work2(alignments, source, target, outputfile, max_lines, max_length, processes)
    else:
        do_the_work(alignments, source, target, outputfile, max_lines, max_length, processes)

def dict_to_file(file_name, dictionary, string_format = '%s: %s\n', key_format = '%s',
                 value_format = '%s'):
    """Write a dictionary to file by formatting its keys and values"""
    out = open(file_name, 'w')
    for key, value in dictionary.iteritems():
        key_string = key_format % tuple(key)
        value_string = value_format % tuple(value)
        out.write(string_format % (key_string, value_string))

    out.close()

def do_the_work2(alignments, source, target, outputfile, max_lines, max_length, processes):
    """Extract all phrase pairs and calculate their reordering probabilities
    for the left-to-right model and the right-to-left model"""
    _log('\nAlignment file: ', alignments, '\nSource file: ', source,
         '\nTarget file: ', target, '\nOutput file: ', outputfile,
         '\nMax lines: ', max_lines, '\nMax length: ', max_length,
         '\nProcesses:', processes, '\n')

    if processes < 2:
        reordering_counts = extract_lexical_reordering_counts(alignments,
            source, target, max_length, max_lines)
    else:
        reordering_counts = \
                mp_worker.set_up_workers(alignments, source, target,
                                         outputfile, max_length,
                                         max_lines, processes, task_id=1)

    dict_to_file('counts.'+outputfile, reordering_counts, "%s%s\n",
        "%s ||| %s ||| ", "%s %s %s %s %s %s %s %s")
    reordering_probs = reordering_freq_to_prob(reordering_counts, True)
    dict_to_file(outputfile, reordering_probs, "%s%s\n", "%s ||| %s ||| ",
        "%s %s %s %s %s %s %s %s")


def do_the_work(alignments, source, target, outputfile, max_lines, max_length, processes):
    """Extract all phrase pairs and calculate their conditional probabilities
    and lexical weights in both directions, i.e. f|e and e|f"""
    _log('\nAlignment file: ', alignments, '\nSource file: ', source,
         '\nTarget file: ', target, '\nOutput file: ', outputfile,
         '\nMax lines: ', max_lines, '\nMax length: ', max_length,
         '\nProcesses:', processes, '\n')

    
    if processes <= 1:
        phrase_freqs, lex_freqs, phrase_to_internals = \
            extract_phrase_pair_freqs(alignments, source, target,
                                      max_length, max_lines)
    else:        
        phrase_freqs, lex_freqs, phrase_to_internals = \
            mp_worker.set_up_workers(alignments, source, target,
                                     outputfile, max_length,
                                     max_lines, processes, task_id=0)

    phrase_pair_freqs, source_phrase_freqs, target_phrase_freqs = phrase_freqs
    lex_pair_freqs, source_lex_freqs, target_lex_freqs = lex_freqs
    write_phrases_to_file(outputfile, phrase_pair_freqs,
                          source_phrase_freqs, target_phrase_freqs)

    # Calculating translation probabilities P(f|e) and P(e|f)
    phrase_source_given_target, phrase_target_given_source = \
        conditional_probabilities(phrase_pair_freqs, source_phrase_freqs,
                                  target_phrase_freqs,
                                  label='TRANSLATION PROBABILITIES',
                                  logprob=True)
    write_translationprobs_to_file(outputfile, phrase_source_given_target,
                                   phrase_target_given_source)

    # Calculating lexical probabilities L(f|e) and L(e|f)
    lex_source_given_target, lex_target_given_source = \
        conditional_probabilities(lex_pair_freqs, source_lex_freqs,
                                  target_lex_freqs,
                                  label='LEXICAL PROBABILITIES',
                                  logprob=True)
    
    # Calculating lexical weights l(f|e) and l(e|f)
    lex_weight_source_given_target, lex_weight_target_given_source = \
        lexical_weights(phrase_to_internals, lex_source_given_target,
                        lex_target_given_source, target_lex_freqs)
    write_lexweights_to_file(outputfile,
                             phrase_source_given_target,
                             phrase_target_given_source,
                             lex_weight_source_given_target,
                             lex_weight_target_given_source)

    all_phrase_info_to_file(outputfile,
                            phrase_source_given_target,
                            phrase_target_given_source,
                            lex_weight_source_given_target,
                            lex_weight_target_given_source,
                            source_phrase_freqs,
                            target_phrase_freqs,
                            phrase_pair_freqs)

    all_data = (phrase_source_given_target, phrase_target_given_source,
        lex_weight_source_given_target, lex_weight_target_given_source,
        source_phrase_freqs, target_phrase_freqs, phrase_pair_freqs)

    sys.stdout.write('Done.\n')
    return all_data

if __name__ == '__main__':
    main()
