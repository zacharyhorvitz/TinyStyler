#!/bin/sh
set -ex

# RECONSTRUCTION_CHECKPOINT='/mnt/swordfish-pool2/horvitz/reddit_mud/raw_all/emnlp/models/enc_dec_cond/2024-05-05-03.31.44_backup/best_model_google_t5-v1_1-large_1e-05_64.pt'

# python tinystyler_authorship.py \
#    --styll_dataset_shard single \   
#    --model_checkpoint_path $RECONSTRUCTION_CHECKPOINT \
#    --output_dir results_new \
#    --use_actual_input False \
#    --experiment_name tinystyler_eval

# python tinystyler_authorship.py \
#    --styll_dataset_shard random \   
#    --model_checkpoint_path $RECONSTRUCTION_CHECKPOINT \
#    --output_dir results_new \
#    --use_actual_input False \
#    --experiment_name tinystyler_eval

# python tinystyler_authorship.py \
#    --styll_dataset_shard diverse \   
#    --model_checkpoint_path $RECONSTRUCTION_CHECKPOINT \
#    --output_dir results_new \
#    --use_actual_input False \
#    --experiment_name tinystyler_eval

# # Original FT model
# FT_MODEL_CHECKPOINT='/mnt/swordfish-pool2/horvitz/reddit_mud/raw_all/emnlp/supervised_ft_models/enc_dec_ft_v2_config_1_fixed/2024-05-28-02.06.56/best_model_google_t5-v1_1-large_1e-05_64.pt'

# python tinystyler_authorship.py \
#    --styll_dataset_shard single \   
#    --model_checkpoint_path $FT_MODEL_CHECKPOINT \
#    --output_dir results_new \
#    --use_actual_input True \
#    --experiment_name tinystyler_eval

# python tinystyler_authorship.py \
#    --styll_dataset_shard random \   
#    --model_checkpoint_path $FT_MODEL_CHECKPOINT \
#    --output_dir results_new \
#    --use_actual_input True \
#    --experiment_name tinystyler_eval

# python tinystyler_authorship.py \
#    --styll_dataset_shard diverse \   
#    --model_checkpoint_path $FT_MODEL_CHECKPOINT \
#    --output_dir results_new \
#    --use_actual_input True \
#    --experiment_name tinystyler_eval

# # Higher meaning threshold, larger dataset (~400K) FT model
# FT_MODEL_CHECKPOINT='/mnt/swordfish-pool2/horvitz/reddit_mud/raw_all/emnlp/authorship_pairings_LARGE_SCALE/supervised_ft_models/2024-10-30-01.48.31/best_model_google_t5-v1_1-large_1e-05_64.pt'

# python tinystyler_authorship.py \
#    --styll_dataset_shard single \   
#    --model_checkpoint_path $FT_MODEL_CHECKPOINT \
#    --output_dir results_new \
#    --use_actual_input True \
#    --experiment_name tinystyler_eval

# python tinystyler_authorship.py \
#    --styll_dataset_shard random \   
#    --model_checkpoint_path $FT_MODEL_CHECKPOINT \
#    --output_dir results_new \
#    --use_actual_input True \
#    --experiment_name tinystyler_eval

# python tinystyler_authorship.py \
#    --styll_dataset_shard diverse \   
#    --model_checkpoint_path $FT_MODEL_CHECKPOINT \
#    --output_dir results_new \
#    --use_actual_input True \
#    --experiment_name tinystyler_eval