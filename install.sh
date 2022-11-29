#!/usr/bin/env bash

model_dir=/models/NYTK

if [[ ! -d $model_dir ]]; then
  mkdir -p $model_dir
fi

if [[ ! -f /models/NYTK/named-entity-recognition-nerkor-hubert-hungarian/config.json ]]; then
  git clone https://huggingface.co/NYTK/named-entity-recognition-nerkor-hubert-hungarian /models/NYTK/named-entity-recognition-nerkor-hubert-hungarian
fi