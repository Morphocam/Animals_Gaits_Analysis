#!/bin/sh
# Runs the rest of the pipeline once clips are extracted:
#   assemble embeddings -> gait variables -> fine-tuning -> evaluation.
# Usage: sh tools/run_after_extraction.sh [python]
PY=${1:-python}
cd "$(dirname "$0")/.." || exit 1
set -x
"$PY" extract_features.py > output/extract_final.log 2>&1 || exit 1
"$PY" gait_kinematics.py > results/kinematics/log.txt 2>&1 || exit 1
"$PY" train_encoders.py horse polarbear hyena > output/train.log 2>&1 || exit 1
"$PY" evaluate.py > output/eval.log 2>&1 || exit 1
echo pipeline_done
