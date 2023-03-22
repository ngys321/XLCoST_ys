#!/usr/bin/env bash

source common_run_utils.sh
PATH_2_DATA=${ROOT_PATH}g4g/codeXglue_data

# bash run_translation_codexglue.sh 0 java csharp codebert train
# bash run_translation_codexglue.sh 0 csharp java codebert train
GPU=${1:-0};
SOURCE=${2:-java};
TARGET=${3:-python};
MODEL=${4:-codebert};
IS_TRAIN=${5:-train};
NUM_EPOCHS=${6:-10};
pretrained_model="microsoft/codebert-base";
model_type="roberta";
beam_size=5;
num_train_epochs=$NUM_EPOCHS;

if [[ $MODEL == 'codebert' ]]; then
    pretrained_model="microsoft/codebert-base";
elif [[ $MODEL == 'roberta' ]]; then
    pretrained_model="roberta-base";
elif [[ $MODEL == 'graphcodebert' ]]; then
    pretrained_model="microsoft/graphcodebert-base";
elif [[ $MODEL == 'codet5' ]]; then
    pretrained_model="Salesforce/codet5-base";
    model_type=$MODEL
elif [[ $MODEL == 'bart' ]]; then
    pretrained_model="facebook/bart-base";
    model_type=$MODEL
elif [[ $MODEL == 'plbart' ]]; then
    pretrained_model="uclanlp/plbart-base"; # uclanlp/plbart-base "uclanlp/plbart-python-en_XX";
    model_type=$MODEL
fi
experiment_name=${MODEL}_translation_codexglue


source_length=200;
target_length=200;
TRAIN_STEPS=10000
EVAL_STEPS=5000
TRAIN_BATCH_SIZE=32
EVAL_BATCH_SIZE=32


export CUDA_VISIBLE_DEVICES=$GPU
echo "Source: $SOURCE Target: $TARGET"
echo "Data path: $PATH_DATA"
echo "Pre-trained model: $pretrained_model"
echo "Model type: $model_type"
echo "Experiment name: $experiment_name"


SOURCE_LANG=${LANG_UPPER[$SOURCE]}
TARGET_LANG=${LANG_UPPER[$TARGET]}
LANG_PAIR=$SOURCE_LANG-$TARGET_LANG


SRC_FILE_SUFFIX=-$LANG_PAIR-tok${FILE_EXTENSION[$SOURCE_LANG]}
TRAIN_FILE_SRC=$PATH_2_DATA/train$SRC_FILE_SUFFIX
if [ ! -f $TRAIN_FILE_SRC ] 
then
    LANG_PAIR=$TARGET_LANG-$SOURCE_LANG
fi

SRC_FILE_SUFFIX=-$LANG_PAIR-tok${FILE_EXTENSION[$SOURCE_LANG]}
TGT_FILE_SUFFIX=-$LANG_PAIR-tok${FILE_EXTENSION[$TARGET_LANG]}
TRAIN_FILE_SRC=$PATH_2_DATA/train$SRC_FILE_SUFFIX
TRAIN_FILE_TGT=$PATH_2_DATA/train$TGT_FILE_SUFFIX
VAL_FILE_SRC=$PATH_2_DATA/valid$SRC_FILE_SUFFIX
VAL_FILE_TGT=$PATH_2_DATA/valid$TGT_FILE_SUFFIX
TEST_FILE_SRC=$PATH_2_DATA/test$SRC_FILE_SUFFIX
TEST_FILE_TGT=$PATH_2_DATA/test$TGT_FILE_SUFFIX
echo "TEST_FILE_SRC: $TEST_FILE_SRC TEST_FILE_TGT: $TEST_FILE_TGT"


SAVE_DIR=${CURRENT_DIR}/experiments/${MODEL}/$experiment_name/$SOURCE_LANG-$TARGET_LANG
# SAVE_DIR=${CURRENT_DIR}/${DATA_SRC}/${SOURCE}2${TARGET};
mkdir -p $SAVE_DIR



function train () {


lr=5e-5;
GRAD_ACCUM_STEP=4; # We need to use 2 GPUs, batch_size_per_gpu=4

python run.py \
    --do_train \
    --do_eval \
    --model_type $model_type \
    --config_name $pretrained_model \
    --tokenizer_name $pretrained_model \
    --model_name_or_path $pretrained_model \
    --train_filename $TRAIN_FILE_SRC,$TRAIN_FILE_TGT \
    --dev_filename $VAL_FILE_SRC,$VAL_FILE_TGT \
    --output_dir $SAVE_DIR \
    --max_source_length $source_length \
    --max_target_length $target_length \
    --num_train_epochs $num_train_epochs \
    --train_steps $TRAIN_STEPS \
    --eval_steps $EVAL_STEPS \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --beam_size $beam_size \
    --learning_rate $lr \

}

function train_from_cp () {
lr=5e-5;
GRAD_ACCUM_STEP=4; # We need to use 2 GPUs, batch_size_per_gpu=4
MODEL_PATH=${SAVE_DIR}/checkpoint-best-ppl/pytorch_model.bin;


python run.py \
    --do_train \
    --do_eval \
    --model_type $model_type \
    --config_name $pretrained_model \
    --tokenizer_name $pretrained_model \
    --model_name_or_path $pretrained_model \
    --load_model_path $MODEL_PATH \
    --train_filename $TRAIN_FILE_SRC,$TRAIN_FILE_TGT \
    --dev_filename $VAL_FILE_SRC,$VAL_FILE_TGT \
    --output_dir $SAVE_DIR \
    --max_source_length $source_length \
    --max_target_length $target_length \
    --num_train_epochs $num_train_epochs \
    --train_steps $TRAIN_STEPS \
    --eval_steps $EVAL_STEPS \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --beam_size $beam_size \
    --learning_rate $lr \

}


function evaluate () {

GOUND_TRUTH_PATH=$PATH_2_DATA/test-${LANG_PAIR}-tok${FILE_EXTENSION[${LANG_UPPER[$TARGET]}]}
MODEL_PATH=${SAVE_DIR}/checkpoint-best-ppl/pytorch_model.bin;
RESULT_FILE=${SAVE_DIR}/result.txt;
SAVE_DIR1=${SAVE_DIR}
# MODEL_PATH=${SAVE_DIR}/checkpoint-best-bleu/pytorch_model.bin;
# SAVE_DIR1=${SAVE_DIR}/bleu
# mkdir -p $SAVE_DIR1
# RESULT_FILE1=${SAVE_DIR1}/result.txt;


python run.py \
    --do_test \
    --model_type $model_type \
    --model_name_or_path $pretrained_model \
    --config_name $pretrained_model \
    --tokenizer_name $pretrained_model  \
    --load_model_path $MODEL_PATH \
    --test_filename $TEST_FILE_SRC,$TEST_FILE_TGT \
    --output_dir $SAVE_DIR \
    --max_source_length $source_length \
    --max_target_length $target_length \
    --beam_size 5 \
    --eval_batch_size 16 \

python $evaluator_script/evaluator.py \
    --references $GOUND_TRUTH_PATH \
    --predictions $SAVE_DIR1/test_0.output \
    2>&1 | tee $RESULT_FILE;

cd $codebleu_path;
python calc_code_bleu.py \
    --ref $GOUND_TRUTH_PATH \
    --hyp $SAVE_DIR1/test_0.output \
    --lang ${LANG_MAP[$TARGET]} \
    2>&1 | tee -a $RESULT_FILE1;
cd $CURRENT_DIR;

count=`ls -1 *.class 2>/dev/null | wc -l`;
[[ $count != 0 ]] && rm *.class;

}


if [[ $IS_TRAIN == 'train' ]]; then
    train;
fi
if [[ $IS_TRAIN == 'train_from_cp' ]]; then
    train_from_cp;
fi
evaluate;
