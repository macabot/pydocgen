PHRASE_LENGTH="7"
STACK="50 100"
BACKOFF="0.4"
N_SIZE="3"
TOP="10 20"
BEAM="1 10 100 1000"
CORE="30"

for stack in `echo $STACK`
do
    for top in `echo $TOP`
    do
        for beam in `echo $BEAM`
        do
            out_path=/home/michael/pydocgen/data/decode/tunesearch_NOfactors.doc_STACK${stack}TOP${top}BEAM${beam}CORE$CORE
            python /home/michael/pydocgen/src/decoder.py \
            -i /home/michael/pydocgen/data/tune/tune_clean_docstring-filtered_sourcecode-NOcontext-NOfactors.tok.sc \
            -o $out_path \
            -lm /home/michael/pydocgen/data/lm/lmpy_train_clean_docstring-filtered.tok.doc \
            -lms /home/michael/pydocgen/data/lm/lmpy_train_clean_sourcecode-NOcontext-NOfactors.tok.sc \
            -tm /home/michael/pydocgen/data/phrases/multicore/phrases_NOfactors_30proc_alllines_7phraselength_all_info.txt \
            -mpl $PHRASE_LENGTH \
            -sl $stack \
            -bs $beam \
            -sb $BACKOFF \
            -tt $top \
            -pr 30 | tee ${out_path}_stdout

            perl /home/bart/apps/smt_tools/decoders/mosesdecoder/scripts/generic/multi-bleu.perl '/home/michael/pydocgen/data/tune/tune_clean_docstring-filtered_sourcecode-NOcontext-NOfactors.tok.doc'  < $out_path |tee ${out_path}_BLEU
        done
    done
done
