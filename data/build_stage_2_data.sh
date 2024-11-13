#!/bin/sh
set -ex
# Data from Million Reddit User Dataset (MUD) https://arxiv.org/abs/2105.07263
# Access form: https://docs.google.com/forms/d/e/1FAIpQLSesc-0HI2DRYjFqlpPo2hTh9OJ53jtWjYQiIfAtmzSVUCxiLA/viewform

# Once the data is downloaded and extracted, rename it to data.jsonl and store it in the data folder

# If you have another dataset, please specify the path to the data folder here
DATAFOLDER='example_authorship_dataset' # Path to the folder where the data is stored
if [ -z "$DATAFOLDER" ]
then
      echo "Please specify a location where the downloaded enron data is located."
	  exit 1
fi

# Expected path to the data
PATH_TO_MUD_JSONL=$DATAFOLDER/'data.jsonl'

PAIRING_DATA_PATH=$DATAFOLDER/'processed_data/authorship_pairings'

# 1) Sample pairs of authors for stage 2 fine-tuning
python sample_authorship_pairings.py --in_dir $DATAFOLDER/'processed_data/split_files' --out_dir $PAIRING_DATA_PATH --shard train --num_samples 200000

# To process additional data, use may use the following command (or something similar):
# PAIRING_DATA_PATH_CONT=$DATAFOLDER/'processed_data/authorship_pairings_continued'
# python sample_authorship_pairings.py --in_dir $DATAFOLDER/'processed_data/split_files' --out_dir $PAIRING_DATA_PATH_CONT  --shard train --num_samples 200000 --skip_author_dict_path $PAIRING_DATA_PATH'/texts_by_author_for_finetune.json'
# PAIRING_DATA_PATH=$PAIRING_DATA_PATH_CONT
# You need to combine this data before step 7

# 2) Split the data into smaller files
python split_files.py --dir $PAIRING_DATA_PATH'/transfers_for_finetune' --max_per_file 5000

# 3) Generate paraphrases (Could be improved with a stronger model)
WORKERS=1 # If you have multiple GPUs, you can run multiple workers by increasing this number and specifying the idx parameter in the command below

echo "Running paraphrase generation with $WORKERS workers. Consider increasing the number of workers if you have more GPUs available."
CUDA_VISIBLE_DEVICES=0 python generate_paraphrases.py --in_dir $PAIRING_DATA_PATH'/transfers_for_finetune/split_files' --out_dir  $PAIRING_DATA_PATH'/transfers_for_finetune/paraphrased_files' --temp 1.5 --top_p 0.8 --idx 0 --num_workers $WORKERS --batch_size 16 --max_input_length 60 --max_output_length 60 --num_return_sequences 5 &
wait

# 4) Generate samples with TinyStyler
CUDA_VISIBLE_DEVICES=0 python tinystyler_data_generation.py --worker_idx 0 --num_workers $WORKERS --transfer_data_dir $PAIRING_DATA_PATH --checkpoint_path '../models/tinystyler_recon/best_model_google_t5-v1_1-large_1e-05_64.pt' &
wait

# 5) Perform data filtering
CUDA_VISIBLE_DEVICES=0 python data_filtering.py --in_dir $PAIRING_DATA_PATH --worker_idx 0 --num_workers $WORKERS &
wait

# 6) Carve out val set from the training set
python split_finetune_train_val.py --in_dir $PAIRING_DATA_PATH/filtered_authorship_pairings --percent_val 0.05 --seed 42

# 7) Convert data to the dataset format
python convert_to_dataset_format.py --in_dir $PAIRING_DATA_PATH/filtered_authorship_pairings/train_val_split --out_dir $PAIRING_DATA_PATH/'filtered_authorship_dataset_format' --name dataset

# 8) Preprocess the data to add Wegmann embeddings
python preprocess_add_embeddings_target_texts.py --dataset_path  $PAIRING_DATA_PATH/filtered_authorship_dataset_format/dataset --style_model_name "AnnaWegmann/Style-Embedding"

# 9) Modify training/train_stage_3.sh to point to the correct dataset path and run the fine-tuning script
echo "Modify training/train_stage_3.sh to point to the correct dataset path and run the fine-tuning script"