#!/bin/bash
cd /workspace/pku_percy/bif
CUDA_VISIBLE_DEVICES=0,1 python -m bif.cli pipeline run --config configs/pythia70m.yaml --resume
