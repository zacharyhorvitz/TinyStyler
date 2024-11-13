import torch
import importlib
from huggingface_hub import hf_hub_download
from transformers import set_seed

# # Import TinyStyler
tinystyler_module = importlib.util.module_from_spec(
    importlib.util.spec_from_file_location(
        "tinystyler",
        hf_hub_download(repo_id="tinystyler/tinystyler", filename="tinystyler.py"),
    )
)
tinystyler_module.__spec__.loader.exec_module(tinystyler_module)
# import tinystyler_hf as tinystyler_module


set_seed(42)
run_tinystyler_batch = tinystyler_module.run_tinystyler_batch

# Define inputs
source_texts = [
    "I want to see the football game",
    "haha that's awesome",
]  # (formal, informal)
target_texts = [
    # Examples of informal style:
    [
        "idk.....but i have faith in you lol",
        "cant wait for a new album from him.",
        "i can't believe it!!1",
    ],
    # Examples of a formal style:
    [
        "That is spectacular and wonderful.",
        "I found his story eloquent and inspiring.",
        "That is an interesting proposition.",
    ],
]

# Run TinyStyler
results = run_tinystyler_batch(
    source_texts=source_texts,
    target_texts_batch=target_texts,
    reranking=5,  # Generate 5 outputs, rerank, and pick the best ones
    temperature=1.0,
    top_p=1.0,
    max_new_tokens=128,
    device=("cuda" if torch.cuda.is_available() else "cpu"),
    seed=42,
    verbose=True,
)
print(
    results
)  # Outputs (-> informal, -> formal): ['a football game!! i wanna see it', 'That is impressively awesome.']
