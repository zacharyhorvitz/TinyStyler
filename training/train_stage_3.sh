#!/bin/bash


# Fine-tune the reconstruction model on the supervised dataset without paraphrasing
set -ex

# NOTE: You may want to set HF_HOME envvar and OUT_DIR below to a directory with more space

# First run data/build_stage_2_data.sh to generate the data in the correct format
PATH_TO_HQ_SUPERVISED_DATASET='../data/example_authorship_dataset/processed_data/authorship_pairings/filtered_authorship_dataset_format/dataset_with_style_embeds_AnnaWegmann_Style-Embedding'
OUT_DIR='stage_3_self_distill_model_data'
RECONSTRUCT_CHECKPOINT='../models/tinystyler_recon/best_model_google_t5-v1_1-large_1e-05_64.pt'

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
--checkpoint $RECONSTRUCT_CHECKPOINT \
--warmup_steps 0 \
--max_steps 100000000 \
--ctrl_embed_dim 768 \
--style_embed style_embedding \
--model_name google/t5-v1_1-large \
--data_file_path $PATH_TO_HQ_SUPERVISED_DATASET \
--seed 42 \
--max_encoder_len 80 \
--max_decoder_len 80 \
--max_val_batch 200 \
--input_key "source_text" \
--output_key "output" \
--eval_freq 500