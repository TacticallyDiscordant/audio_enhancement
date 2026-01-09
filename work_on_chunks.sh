#!/bin/bash

python src/work_on_chunks.py \
                --input ./datasets/chunk_work/s5_007_degraded3.wav \
                --output ./datasets/chunk_work/s5_007_enhanced3_fsr.wav \
                --ground_truth ./datasets/chunk_work/s5_007_mic1.wav \
                --plot ./datasets/chunk_work/s5_007_enhanced3_fsr.png \
                --input_sr 16000 \
                --chunk_size 0.5 \
                --overlap 0.15
