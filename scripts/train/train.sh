#!/bin/bash
set -euo pipefail

########## Instruction ##########
# This script takes three optional environment variables:
# GPU / ADDR / PORT
# e.g. Use GPU 0 and 1 for standard flow-matching training:
#
# GPU="0,1" bash scripts/train/train.sh \
#   scripts/train/configs/single_cdr_design.json \
#   --save_dir ./datasets/RAbD/models_standard_fm \
#   --sigma_min 0.01 \
#   --flow_weight 1.0 \
#   --sequence_flow_weight 1.0 \
#   --time_embed_dim 32
#
# Extra arguments after the JSON config override values from the config.
# More generally, use gpu 0, 1 and 4 for training, set distributed training
# master address and port to localhost:9901, the command is as follows:
#
# GPU="0,1,4" ADDR=localhost PORT=9901 bash train.sh <config>
#
# Default value: GPU=-1 (use cpu only), ADDR=localhost, PORT=9901
# Note that if your want to run multiple distributed training tasks,
# either the addresses or ports should be different between
# each pair of tasks.
######### end of instruction ##########


########## setup project directory ##########
CODE_DIR=$(realpath "$(dirname "$0")/../..")
echo "Locate the project folder at ${CODE_DIR}"


########## parsing JSON configs ##########
if [ -z "${1:-}" ]; then
    echo "Config missing. Usage: GPU=0,1 bash $0 <config> [extra train.py arguments]"
    exit 1;
fi
CONFIG_FILE=$1
shift

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Config file not found: $CONFIG_FILE"
    exit 1
fi

# Read the JSON config into a Bash array so paths and values remain intact.
# cdr/paratope may be represented either as a JSON list or as a
# whitespace-separated string in the existing configuration files.
mapfile -d '' -t CONFIG_ARGS < <(python - "$CONFIG_FILE" <<'PY'
import json
import sys

with open(sys.argv[1], encoding='utf-8') as stream:
    config = json.load(stream)

for key, value in config.items():
    if value is None or value is False or value == '':
        continue
    sys.stdout.write(f'--{key}\0')
    if value is True:
        continue
    if isinstance(value, list):
        values = value
    elif key in {'cdr', 'paratope'} and isinstance(value, str):
        values = value.split()
    else:
        values = [value]
    for item in values:
        sys.stdout.write(f'{item}\0')
PY
)

EXTRA_ARGS=("$@")


########## setup  distributed training ##########
GPU="${GPU:--1}" # default using CPU
MASTER_ADDR="${ADDR:-localhost}"
MASTER_PORT="${PORT:-9901}"
echo "Using GPUs: $GPU"
echo "Master address: ${MASTER_ADDR}, Master port: ${MASTER_PORT}"

IFS=',' read -r -a GPU_ARR <<< "$GPU"

if [ "$GPU" = "-1" ]; then
    TRAIN_GPU_IDS=(-1)
    PREFIX=(python)
    unset CUDA_VISIBLE_DEVICES
elif [ ${#GPU_ARR[@]} -gt 1 ]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
    TRAIN_GPU_IDS=()
    for ((gpu_idx = 0; gpu_idx < ${#GPU_ARR[@]}; gpu_idx++)); do
        TRAIN_GPU_IDS+=("$gpu_idx")
    done
    export OMP_NUM_THREADS=2
	PREFIX=(torchrun
        --nproc_per_node="${#GPU_ARR[@]}"
        --rdzv_backend=c10d
        --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}"
        --nnodes=1)
else
    export CUDA_VISIBLE_DEVICES="$GPU"
    TRAIN_GPU_IDS=(0)
	PREFIX=(python)
fi


########## start training ##########
cd "$CODE_DIR"
export CUDA_LAUNCH_BLOCKING=1
"${PREFIX[@]}" train.py \
    --gpus "${TRAIN_GPU_IDS[@]}" \
    "${CONFIG_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"

# CPU example:
# GPU=-1 bash scripts/train/train.sh scripts/train/configs/single_cdr_design.json
