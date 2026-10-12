#!/bin/bash
docker run \
    --gpus all \
    --rm \
    --name marigold \
    -v "$1":/workspace/input \
    -v "$2":/workspace/output \
    -v "$3":/workspace/Marigold/checkpoint \
    marigold:latest \
    python3 /workspace/Marigold/script/"$4"/run.py  \
    --checkpoint checkpoint/marigold-"$5"-v1-1 \
    --denoise_steps 4 \
    --ensemble_size 1 \
    --input_rgb_dir /workspace/input \
    --output_dir /workspace/output \
    --fp16