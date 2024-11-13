import torch
import importlib
from huggingface_hub import hf_hub_download
from transformers import set_seed

# Import TinyStyler
tinystyler_module = importlib.util.module_from_spec(
    importlib.util.spec_from_file_location(
        "tinystyler",
        hf_hub_download(repo_id="tinystyler/tinystyler", filename="tinystyler.py"),
    )
)

# import tinystyler_hf as tinystyler_module


tinystyler_module.__spec__.loader.exec_module(tinystyler_module)
get_tinystyler_model, get_target_style_embeddings = (
    tinystyler_module.get_tinystyler_model,
    tinystyler_module.get_target_style_embeddings,
)

# Load the TinyStyler model
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer, model = get_tinystyler_model(device)
set_seed(42)

# Define inputs
source_text = "I want to see the football game"  # (formal)
# Examples of informal style:
target_texts = [
    "idk.....but i have faith in you lol",
    "cant wait for a new album from him.",
    "i can't believe it!!1",
]

# Run TinyStyler
inputs = tokenizer(
    [source_text], padding="longest", truncation=True, return_tensors="pt"
).to(device)
output = model.generate(
    **inputs,
    style=get_target_style_embeddings([target_texts], device).to(device),
    do_sample=True,
    temperature=1.0,
    top_p=1.0,
    max_new_tokens=128,
)
generated_text = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
print(generated_text)  # Outputs (-> informal): 'i want to watch football.........'
