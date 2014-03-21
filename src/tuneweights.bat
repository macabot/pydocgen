PHRASE_LENGTH="7"
STACK="100"
BACKOFF="0.4"
N_SIZE="3"
TOP="10"
BEAM="1"
CORE="30"
WEIGHTS="1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0"

STEP_SIZE="1.0"
MIN_SCORE_DIFF="0.1"
MAX_ITERATIONS="30"
MIN_STEP_SIZE="0.25"


out_path=/home/michael/pydocgen/data/decode/tuneweights/tuneweights_NOfactors.doc_STACK${STACK}TOP${TOP}BEAM${BEAM}CORE$CORE
python /home/michael/pydocgen/src/pattern_search.py \
-i /home/michael/pydocgen/data/tune/tune_clean_docstring-filtered_sourcecode-NOcontext-NOfactors.tok.sc \
-o $out_path \
-lm /home/michael/pydocgen/data/lm/lmpy_train_clean_docstring-filtered.tok.doc \
-lms /home/michael/pydocgen/data/lm/lmpy_train_clean_sourcecode-NOcontext-NOfactors.tok.sc \
-tm /home/michael/pydocgen/data/phrases/multicore/phrases_NOfactors_30proc_alllines_7phraselength_all_info.txt \
-mpl $PHRASE_LENGTH \
-sl $STACK \
-bs $BEAM \
-sb $BACKOFF \
-tt $TOP \
-w $WEIGHTS \
-pr $CORE \
-sz $STEP_SIZE \
-msd $MIN_SCORE_DIFF \
-mit $MAX_ITERATIONS \
-msz $MIN_STEP_SIZE \
-bp /home/michael/multi-bleu.perl \
-ref /home/michael/pydocgen/data/tune/tune_clean_docstring-filtered_sourcecode-NOcontext-NOfactors.tok.doc | tee ${out_path}_stdout

