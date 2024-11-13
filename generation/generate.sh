#!/bin/bash

MODEL_PATH='../models/tinystyler/model_weights.pt'
# MODEL_PATH='../models/tinystyler_sim/model_weights.pt'

# -> informal
python generate.py \
   --input_path input_style/inputs_famous_quotes.txt \
   --target_path target_styles/informal.txt \
   --model_path $MODEL_PATH \
   --device cuda \
   --num_inferences 1

# -> question-followed-by-sentence
python generate.py \
    --input_path input_style/inputs_famous_quotes.txt \
    --target_path target_styles/question_followed_by_sentence.txt \
    --model_path $MODEL_PATH \
    --device cuda \
    --num_inferences 1

# -> obama
python generate.py \
    --input_path input_style/inputs_famous_quotes.txt \
    --target_path target_styles/obama.txt \
    --model_path $MODEL_PATH \
    --device cuda \
    --num_inferences 1
