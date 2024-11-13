---
license: mit
---

# TinyStyler Source

## Structure

- [tinystyler.py](tinystyler.py): Contains the ``TinyStyler`` class, which wraps a T5 model (https://huggingface.co/google/t5-v1_1-large), preprending an authorship embedding.
- [style_utils.py](style_utils.py): Utilities for loading and using the Wegmann Style Embedding (https://huggingface.co/AnnaWegmann/Style-Embedding) models.

## Usage

For example usage, see [generate.py](../generation/generate.py) or our Hugging Face demo [locally](../hugging_face_demo/), or on [Hugging Face](https://huggingface.co/tinystyler/tinystyler) .

