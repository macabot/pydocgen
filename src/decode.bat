PHRASE_LENGTH="7"
STACK="100"
BACKOFF="0.4"
N_SIZE="3"
TOP="10"
BEAM="1"
CORE="30"
WEIGHTS="8.0 1.0 1.0 -0.96875 1.0 1.03125 1.0 0.0"

out_path=/home/michael/pydocgen/data/decode/tuneweights/confirm/confirm_8.0,1.0,1.0,-0.96875,1.0,1.03125,1.0,0.0_tunesearch_NOfactors.doc_STACK${STACK}TOP${TOP}BEAM${BEAM}CORE${CORE}
python /home/michael/pydocgen/src/decoder.py \
-i /home/michael/pydocgen/data/tune/tune_clean_docstring-filtered_sourcecode-NOcontext-NOfactors.tok.sc \
-o $out_path \
-lm /home/michael/pydocgen/data/lm/lmpy_train_clean_docstring-filtered.tok.doc \
-lms /home/michael/pydocgen/data/lm/lmpy_train_clean_sourcecode-NOcontext-NOfactors.tok.sc \
-tm /home/michael/pydocgen/data/phrases/multicore/empty_phrases_NOfactors_alllines_7phraselength_all_info.txt \
-mpl $PHRASE_LENGTH \
-sl $STACK \
-bs $BEAM \
-sb $BACKOFF \
-tt $TOP \
-ed \
-w $WEIGHTS \
-pr $CORE | tee ${out_path}_stdout

perl /home/bart/apps/smt_tools/decoders/mosesdecoder/scripts/generic/multi-bleu.perl '/home/michael/pydocgen/data/tune/tune_clean_docstring-filtered_sourcecode-NOcontext-NOfactors.tok.doc'  < $out_path |tee ${out_path}_BLEU
