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

# 1) Split into train, val, test, ensure that protected authors are not in the train set
python split_authors.py --path $DATAFOLDER/'data.jsonl' --out_dir $DATAFOLDER/'processed_data' --protected_authors_path 'styll_evaluation_data/protected_authors.txt'

# 2) Subsample the data to ensure that each author has no more than 10 samples
python subsample_data.py --path  $DATAFOLDER/'data.jsonl' --author_splits_path $DATAFOLDER/'processed_data/author_splits.json'  --per_author 10

# 3) Split the data into smaller files
python split_files.py --dir $DATAFOLDER/'processed_data' --max_per_file 50000

# 4) Generate paraphrases (Could be improved with a stronger model)
WORKERS=1 # If you have multiple GPUs, you can run multiple workers by increasing this number and specifying the idx parameter in the command below

echo "Running paraphrase generation with $WORKERS workers. Consider increasing the number of workers if you have more GPUs available."

CUDA_VISIBLE_DEVICES=0 python generate_paraphrases.py --in_dir $DATAFOLDER/'processed_data/split_files' --out_dir $DATAFOLDER/'processed_data/paraphrased_files' --temp 1.5 --top_p 0.8 --idx 0 --num_workers $WORKERS --batch_size 32 --max_input_length 60 --max_output_length 60 &
# CUDA_VISIBLE_DEVICES=1 python generate_paraphrases.py --in_dir $DATAFOLDER/'processed_data/split_files' --out_dir $DATAFOLDER/'processed_data/paraphrased_files' --temp 1.5 --top_p 0.8 --idx 1 --num_workers $WORKERS --batch_size 32 --max_input_length 60 --max_output_length 60 &

wait # Wait for all workers to finish

# 5) Convert data to the dataset format
python convert_to_dataset_format.py --in_dir  $DATAFOLDER/'processed_data/paraphrased_files/topp0.8_tmp1.5' --out_dir $DATAFOLDER/'processed_data/dataset_format' --name authorship_dataset

# 6) Preprocess the data to add Wegmann embeddings
python preprocess_add_source_text_embeddings.py --dataset_path $DATAFOLDER/'processed_data/dataset_format/authorship_dataset' --style_model_name "AnnaWegmann/Style-Embedding"

# 7) Modify training/train_stage_1.sh to point to the correct dataset path and run the training script
echo "Modify training/train_stage_1.sh to point to the correct dataset path and run the training script"