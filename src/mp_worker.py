import sys
import multiprocessing as mp
import gc
import decoder
import phrase_extract as phr
from collections import Counter
from itertools import izip

def set_up_decoders(input_file, output_file, language_model,
                    source_language_model,
                    translation_model, max_lines, beam_size, num_processes,
                    max_phrase_length, stack_limit, stupid_backoff,
                    n_size, feature_weights, nbest, empty_default):
    '''Set up workers for decoding'''
    num_lines = sum(1 for line in open(input_file))
    max_lines = int(min(num_lines, max_lines))
    source = open(input_file, 'r')

    task_queue = mp.JoinableQueue()
    for i, source_str in enumerate(source):
        if i == max_lines:
            break
        task_queue.put((tuple(source_str.strip().split()), i))
    for i in range(num_processes):
        task_queue.put( 'STOP' )
    source.close()

    workers = []
    for p in xrange(num_processes):
        p_in, p_out = mp.Pipe()
        worker = Worker(p, task_queue, p_in, num_lines, max_lines,
                        task_id=2,
                        language_model=language_model,
                        source_language_model=source_language_model,
                        translation_model=translation_model,
                        beam_size=beam_size,
                        max_phrase_length=max_phrase_length,
                        stack_limit=stack_limit,
                        stupid_backoff=stupid_backoff,
                        n_size=n_size,
                        feature_weights=feature_weights,
                        nbest=nbest,
                        empty_default=empty_default)

        workers.append( (worker, p_out) )
        worker.start()
    # Wait until workers finish
    task_queue.join()

    translations = [None for i in xrange(min(max_lines, num_lines))]
    for worker, pipe in workers:
        for idx, translation in pipe.recv().iteritems():
            if isinstance(translation, str):
                translations[idx] = translation.replace("<s> ","")
            else:
                translations[idx] = [t.replace("<s> ", "") for t in translation]
    phr.show_progress(1, 1, 40, "DECODING")
    sys.stdout.write('\n')

    with open(output_file, 'w') as output:
        for t in translations:
            output.write(str(t) + '\n')
    print "\n\nOutput written to '{}'".format(output_file)

    return translations


def set_up_workers(alignments_file, source_file, target_file, output_file,
                   max_phrase_length, max_lines, num_processes, task_id):
    alignments = open(alignments_file, 'r')
    num_lines = sum(1 for line in open(alignments_file))
    source = open(source_file, 'r')
    target = open(target_file, 'r')

    task_queue = mp.JoinableQueue()
    for i, (alignment_str, source_str, target_str) in enumerate(
            izip(alignments, source, target)):
        if i == max_lines:
            break
        task_queue.put((phr.str_to_alignments(alignment_str),
                        source_str.strip().split(),
                        target_str.strip().split(),
                        i))

    for i in range(num_processes):
        task_queue.put( 'STOP' )
    alignments.close()
    source.close()
    target.close()

    workers = []
    for p in xrange(num_processes):
        p_in, p_out = mp.Pipe()
        worker = Worker(p, task_queue, p_in, num_lines, max_lines,
                        max_phrase_length, task_id)
        workers.append( (worker, p_out) )
        worker.start()

    # Wait until workers finish
    task_queue.join()


    if task_id == 0:
        phrase_pair_freqs = Counter()
        source_phrase_freqs = Counter()
        target_phrase_freqs = Counter()
        lex_pair_freqs = Counter()
        source_lex_freqs = Counter()
        target_lex_freqs = Counter()
        phrase_to_internals = {}

        # Collect information from workers
        for worker, pipe in workers:
            phrase_freqs, lex_freqs, internals = pipe.recv()

            # Store in correct dict
            phrase_pair_freqs.update(phrase_freqs[0])
            source_phrase_freqs.update(phrase_freqs[1])
            target_phrase_freqs.update(phrase_freqs[2])
            lex_pair_freqs.update(lex_freqs[0])
            source_lex_freqs.update(lex_freqs[1])
            target_lex_freqs.update(lex_freqs[2])
            for phrase_pair, possible_internal in internals.iteritems():
                if phrase_pair in phrase_to_internals:
                    phrase_to_internals[phrase_pair] |= possible_internal
                else:
                    phrase_to_internals[phrase_pair] = possible_internal

        phr.show_progress(1, 1, 40, 'PHRASE EXTRACTION')
        sys.stdout.flush()
        return ((phrase_pair_freqs, source_phrase_freqs, target_phrase_freqs),
                (lex_pair_freqs, source_lex_freqs, target_lex_freqs),
                 phrase_to_internals)
    elif task_id == 1:
        reordering_counts = {}
        for worker, pipe in workers:
            for phrase_pair, counts in pipe.recv().iteritems():
                if phrase_pair in reordering_counts:
                    reordering_counts[phrase_pair] = [a + b for a, b in
                            zip(counts, reordering_counts[phrase_pair])]
                else:
                    reordering_counts[phrase_pair] = counts

        phr.show_progress(1, 1, 40, 'LEXICAL REORDERING EXTRACTION')
        sys.stdout.write('\n')
        sys.stdout.flush()
        return reordering_counts


class Worker(mp.Process):
    def __init__(self, w_id, queue, pipe, num_lines, max_lines,
                 max_phrase_length, task_id, **kwargs):
        self.w_id = w_id
        self.queue = queue
        self.pipe = pipe
        self.results = []

        self.max_phrase_length = max_phrase_length
        self.num_lines = num_lines
        if max_lines == float('inf'):
            self.max_lines = num_lines
        else:
            self.max_lines = int(max_lines)
        self.task_id = task_id

        self.run = {0: self.run_phrase_extraction,
                    1: self.run_lexical_reordering_extraction,
                    2: self.run_decoding}[task_id]

        self.__dict__.update(kwargs)
        mp.Process.__init__(self)

    def log(self, *args):
        print 'W {}: '.format(self.w_id) + ' '.join((str(a) for a in args))

    def run_decoding(self):
        '''
        '''
        translations = {}

        language_model = self.language_model
        source_language_model = self.source_language_model
        translation_model =  self.translation_model
        beam_size = self.beam_size
        max_phrase_length = self.max_phrase_length
        stack_limit = self.max_phrase_length
        stupid_backoff = self.stupid_backoff
        n_size = self.n_size
        feature_weights = self.feature_weights
        nbest = self.nbest
        empty_default = self.empty_default

        point = self.max_lines / 100 if self.max_lines > 100 else 1

        for sentence, idx in iter(self.queue.get, 'STOP'):
            idx = int(idx)
            translations[idx] = \
                    decoder.decode(sentence, language_model,
                                   source_language_model,
                                   translation_model,
                                   beam_size, max_phrase_length, stack_limit,
                                   stupid_backoff, n_size, feature_weights,
                                   nbest, empty_default)
            if idx % point == 0:
                phr.show_progress(idx, self.max_lines, 40, "DECODING")

            self.queue.task_done()
        self.queue.task_done()

        self.pipe.send(translations)

    def run_lexical_reordering_extraction(self):
        '''
        yup
        '''
        self.reordering_counts = {}
        point = self.max_lines / 100 if self.max_lines > 100 else 1

        for item in iter(self.queue.get, 'STOP'):
            idx = item[-1]
            if idx % point == 0:
                phr.show_progress(idx, self.max_lines, 40,
                    "LEXICAL REORDERING EXTRACTION")

            self.extract_lexical_reordering_counts(*item[:-1])
            self.queue.task_done()

        # Tell other workers you're done
        self.queue.task_done()

        # Clean and send
        gc.collect()

        self.pipe.send(self.reordering_counts)

    def run_phrase_extraction(self):
        '''
        Get items from the queue until stop message is received.

        Items are in form
        alignment, source, target, i
        '''
        # phrase frequencies
        self.phrase_pair_freqs = Counter()
        self.source_phrase_freqs = Counter()
        self.target_phrase_freqs = Counter()

        # lexical frequencies
        self.lex_pair_freqs = Counter()
        self.source_lex_freqs = Counter()
        self.target_lex_freqs = Counter()

        # map phrase pair to possible internal word alignments
        self.phrase_to_internals = {}

        # NOTE: every worker shows progress, so do not log in between!
        point = self.max_lines / 100 if self.max_lines > 100 else 1

        for item in iter(self.queue.get, 'STOP'):
            idx = item[-1]
            if idx % point == 0:
                phr.show_progress(idx, self.max_lines, 40, "PHRASE EXTRACTION")

            self.extract_phrase_pair_freqs(*item[:-1])
            self.queue.task_done()

        # Tell other workers you're done
        self.queue.task_done()

        # Clean and send
        gc.collect()

        self.pipe.send( ((self.phrase_pair_freqs, self.source_phrase_freqs,
                         self.target_phrase_freqs),
                         (self.lex_pair_freqs, self.source_lex_freqs,
                          self.target_lex_freqs),
                         self.phrase_to_internals) )

    def extract_lexical_reordering_counts(self, align, source_words,
                                         target_words):
        source_length = len(source_words)
        target_length = len(target_words)
        phrase_to_internal = phr.extract_alignments(set(align), source_length,
                                               target_length, self.max_phrase_length)

        reordering_counts = self.reordering_counts
        word_phrase_pairs = set([word_pair*2 for word_pair in align])
        phrase_pairs = set(phrase_to_internal.keys())
        for left_phrase_range in phrase_pairs:
            # phrase based counting events
            for right_phrase_range in phrase_pairs - word_phrase_pairs:
                phr.update_count(left_phrase_range, right_phrase_range,
                    reordering_counts, source_words, target_words)

            # word based counting events
            for right_phrase_range in word_phrase_pairs:
                phr.update_count(left_phrase_range, right_phrase_range,
                    reordering_counts, source_words, target_words)

    def extract_phrase_pair_freqs(self, alignment, source_words, target_words):
        '''
        Extract phrase pairs for a single combination of sentences and
        corresponding alignment,  save results in workers internal dicts.
        '''
        source_length = len(source_words)
        target_length = len(target_words)
        
        # word pair frequencies
        for source_index, target_index in alignment:
            word_pair = (source_words[source_index], target_words[target_index])
            self.lex_pair_freqs[word_pair] += 1
            self.source_lex_freqs[word_pair[0]] += 1
            self.target_lex_freqs[word_pair[1]] += 1
        
        phrase_to_internal = \
            phr.extract_alignments(set(alignment), source_length,
                                   target_length, self.max_phrase_length)

        for phrase_pair, internal_alignment in phr.extract_phrase_pairs_gen(
                                                    phrase_to_internal,
                                                    source_words,
                                                    target_words):
            # phrase pair frequencies
            self.phrase_pair_freqs[phrase_pair] += 1
            self.source_phrase_freqs[phrase_pair[0]] += 1
            self.target_phrase_freqs[phrase_pair[1]] += 1
            # phrase pair to possible internal word alignments
            if phrase_pair in self.phrase_to_internals:
                self.phrase_to_internals[phrase_pair].\
                        add(frozenset(internal_alignment))
            else:
                self.phrase_to_internals[phrase_pair] = \
                        set([frozenset(internal_alignment)])

        unaligned, unaligned2 = phr.unaligned_words(alignment, source_length,
                                                    target_length)
        unaligned.extend(unaligned2)
        for phrase_pair in phr.unaligned_phrase_pairs_gen(unaligned,
                                                          source_words,
                                                          target_words):
            self.lex_pair_freqs[phrase_pair] += 1
            self.source_lex_freqs[phrase_pair[0]] += 1
            self.target_lex_freqs[phrase_pair[1]] += 1
