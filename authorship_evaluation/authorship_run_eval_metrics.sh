#!/bin/bash
set -ex

## run evals on outputs from tinystyler, chatgpt, and paraguide

# for path in sft_v1_outputs/*/results.jsonl;
# do
#     echo $path
#     # time python run_authorship_eval.py --input_path $path --do_rerank
#     time python run_authorship_eval.py --input_path $path
# done

# for path in chatgpt/*/*/style.jsonl;
# do
#     echo $path
#     # time python run_authorship_eval.py --input_path $path --do_rerank
#     time python run_authorship_eval.py --input_path $path --is_chatgpt
# done

# for path in results/*/tinystyle*/*/results.jsonl;
# do
#     echo $path
#     time python run_authorship_eval.py --input_path $path --do_rerank
#     time python run_authorship_eval.py --input_path $path
# done

# for path in results/*/chatgpt/*/*/style.jsonl;
# do
#     echo $path
#     time python run_authorship_eval.py --input_path $path --is_chatgpt
# done

# for path in  results/*/paraguide/*/*/to_style.jsonl;
# do
#     echo $path
#     time python run_authorship_eval.py --input_path $path --is_chatgpt # use is_chatgpt because of output format
# done

# for path in results/*/*0.85/*/results.jsonl;
# do
#     echo $path
#     time python run_authorship_eval.py --input_path $path --do_rerank
#     # time python run_authorship_eval.py --input_path $path
# done