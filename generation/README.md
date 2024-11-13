---
license: mit
---

# Style Transfer Generation

## Structure

- [generate.py](generate.py): Generation script that takes input text and target texts, and rewrites the input text in the style of the target texts.
- [generate.sh](generate.sh): Rewrites the ``input_style`` texts into various ``target_styles``.

## Example Usage

```bash

# -> informal
python generate.py \
   --input_path input_style/inputs_famous_quotes.txt \
   --target_path target_styles/informal.txt \
   --model_path $MODEL_PATH \
   --device cuda \
   --num_inferences 1

```