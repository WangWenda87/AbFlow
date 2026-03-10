#!/bin/bash

########## adjust configs according to your needs ##########
CODE_DIR=`realpath $(dirname "$0")/../..`
NUM_WORKERS=8
BATCH_SIZE=32
GPU="${GPU:-0}"
CKPT=$1
TEST_SET=$2
SAVE_DIR="${3:-"$(dirname "$CKPT")/results"}"
TASK="${4:-}" 
SURF_FILE="${5:-}"

if [ "$TASK" = "rabd" ]; then
    PEP_ARG="--pep_file all_data/RAbD/test.pkl"
    SURF_ARG="--surf_file all_data/RAbD/test_surf.pkl"
    SCRIPT="generate.py"
elif [ "$TASK" = "igfold" ]; then
    PEP_ARG="--pep_file all_data/IgFold/test.pkl"
    SURF_ARG="--surf_file all_data/IgFold/test_surf.pkl"
    SCRIPT="struct_generate.py"
else
    PEP_ARG=""
    SURF_ARG="${SURF_FILE:+--surf_file ${SURF_FILE}}" 
fi
######### end of adjust ##########

# validity check
if [ -z "$CKPT" ]; then
	echo "Usage: bash $0 <checkpoint> <test set> [save_dir] [task]"
	echo "  task: rabd (generate.py), igfold (struct_generate.py), or custom path"
	exit 1;
else
	CKPT=`realpath $CKPT`
	SAVE_DIR=`realpath $SAVE_DIR`
fi

# echo Configurations
echo "Locate the project folder at ${CODE_DIR}"
echo "Using GPU: ${GPU}"
echo "Evaluating ${CKPT}"
echo "Results will be written to ${SAVE_DIR}"
echo "Task: ${TASK:-none}"
echo "Script: ${SCRIPT}"

# set gpu
export CUDA_VISIBLE_DEVICES=$GPU

# generate
cd ${CODE_DIR}
python ${SCRIPT} \
    --ckpt ${CKPT} \
    --test_set ${TEST_SET} \
    --save_dir ${SAVE_DIR} \
    --batch_size ${BATCH_SIZE} \
    --gpu 0 \
    ${PEP_ARG} \
    ${SURF_ARG}

echo "Done generation"

# calculate metrics
OPENMM_CPU_THREADS=1 python cal_metrics.py \
    --test_set ${SAVE_DIR}/summary.json \
    --num_workers ${NUM_WORKERS}

echo "Done evaluation"
