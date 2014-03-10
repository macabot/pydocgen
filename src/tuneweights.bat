PHRASE_LENGTH="7"
STACK="100"
BACKOFF="0.4"
N_SIZE="3"
TOP="10"
BEAM="1"
CORE="30"
WEIGHTS="1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0"

STEP_SIZE="1.0"
MIN_SCORE_DIFF="0.01"
MAX_ITERATIONS="200"

for stack in `echo $STACK`
do
    for top in `echo $TOP`
    do
        for beam in `echo $BEAM`
        do
            out_path=/home/michael/pydocgen/data/decode/tuneweights/tuneweights_NOfactors.doc_STACK${stack}TOP${top}BEAM${beam}CORE$CORE
            python /home/michael/pydocgen/src/pattern_search.py \
            -i /home/michael/pydocgen/data/tune/tune_clean_docstring-filtered_sourcecode-NOcontext-NOfactors.tok.sc \
            -o $out_path \
            -lm /home/michael/pydocgen/data/lm/lmpy_train_clean_docstring-filtered.tok.doc \
            -lms /home/michael/pydocgen/data/lm/lmpy_train_clean_sourcecode-NOcontext-NOfactors.tok.sc \
            -tm /home/michael/pydocgen/data/phrases/multicore/empty_phrases_NOfactors_alllines_7phraselength_all_info.txt \
            -mpl $PHRASE_LENGTH \
            -sl $stack \
            -bs $beam \
            -sb $BACKOFF \
            -tt $top \
            -ed \
            -w $WEIGHTS \
            -pr $CORE \
            -sz $STEP_SIZE \
            -msd $MIN_SCORE_DIFF \
            -mit $MAX_ITERATIONS \
            -bp /home/bart/apps/smt_tools/decoders/mosesdecoder/scripts/generic/multi-bleu.perl \
            -ref /home/michael/pydocgen/data/tune/tune_clean_docstring-filtered_sourcecode-NOcontext-NOfactors.tok.doc | tee ${out_path}_stdout

        done
    done
done
