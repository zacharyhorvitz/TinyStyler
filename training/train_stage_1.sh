#!/bin/bash

# Train reconstruction model to reconstruct the original text from the paraphrased text + style embedding

set -ex

# First run data/build_stage_1_data.sh to generate the data in the correct format
PATH_TO_PARAPHRASE_DATASET='../data/example_authorship_dataset/processed_data/dataset_format/authorship_dataset_with_style_embeds_AnnaWegmann_Style-Embedding/'
OUT_DIR='stage_1_recon_model_data'

export CUDA_VISIBLE_DEVICES=0
accelerate launch \
--num_processes 1 \
--num_machines 1 \
--mixed_precision no \
--dynamo_backend no \
train.py \
--learning_rate 1e-5 \
--batch_size 16 \
--accumulation_steps 4 \
--out_dir $OUT_DIR \
--warmup_steps  2000 \
--max_steps 100000000 \
--ctrl_embed_dim 768 \
--style_embed style_embedding \
--data_file_path $PATH_TO_PARAPHRASE_DATASET \
--seed 42 \
--max_encoder_len 80 \
--max_decoder_len 80 \
--max_val_batch 200 \
--model_name google/t5-v1_1-large

