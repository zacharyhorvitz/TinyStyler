---
license: mit
---

# Authorship Evaluations

## Structure

- [tinystyler_authorship.py](tinystyler_authorship.py): Logic for performing style transfer using the Styll benchmark. 

    - Supports both the reconstruction and fine-tuned models. 
    - Input data and target embeddings are preprocessed and cached between evals.

- [run_authorship_eval.py](run_authorship_eval.py): Runs evaluation metrics on authorship transfer results.

- [download_luar_eval.sh](authorship_evaluation/styll_eval_metrics/download_luar_eval.sh): Script for downloading the LUAR embedding checkpoint used for evaluation.


## Example running authorship eval

```bash
# perform generation on the diverse subset of the Styll benchmark
python tinystyler_authorship.py \
   --styll_dataset_shard diverse \   
   --model_checkpoint_path $MODEL_CHECKPOINT_PATH \
   --output_dir results_new \
   --use_actual_input True \
   --experiment_name tinystyler_additional_results

# evaluate with authorship metrics
for path in results_new/diverse/tinystyler*/*/results.jsonl;
do
    echo $path
    # Evaluate results
    time python run_authorship_eval.py --input_path $path

    # Evaluate with re-ranking (a bit slow)
    # time python run_authorship_eval.py --input_path $path --do_rerank
done
```