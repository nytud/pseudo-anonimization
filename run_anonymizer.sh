#!/usr/bin/env bash

set -e

input_file=$(readlink -f $1)
input_file_dir=$(dirname $input_file)
input_file_name=$(basename $input_file)

gpu=$2
metric=$3

docker run --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=$gpu -v /home/lakil/bigSpace/projekt/pseudo-anonimization/src:/app/src -v /home/lakil/bigSpace/projekt/pseudo-anonimization/models:/models -v "${input_file}":/data/"${input_file_name}" --network network_anonymizer -e EMTSV_URL=http://emtsv:5000 -it --rm --entrypoint "" docker.nlp.nytud.hu/anonymizer python3 /app/src/anonimization.py --file-input /data/"${input_file_name}" --format=${metric}
