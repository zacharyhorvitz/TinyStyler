import torch
from transformers import AutoTokenizer

luar_model = None
luar_tokenizer = None


def get_uar_embedding(text):
    global luar_model, luar_tokenizer
    if luar_model is None:
        luar_model = torch.jit.load(
            "LUAR.pth"
        )
    if luar_tokenizer is None:
        luar_tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/paraphrase-distilroberta-base-v1"
        )

    try:
        # using more text improves results
        tokenized_data = luar_tokenizer(
            text,
            max_length=32,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        )
    except TypeError as e:
        import pdb

        pdb.set_trace()
        raise e

    input_ids = tokenized_data["input_ids"]
    attention_mask = tokenized_data["attention_mask"]

    # The LUAR model expects inputs of the following shape:
    #   (batch_size, num_samples_per_author, episode_size, num_tokens)
    # The `num_samples_per_author` dimension is strictly used for training, so for
    # inference this will always be 1.
    # The `episode_size` dimension controls how many text samples get aggregated into
    # one embedding.

    # Here, we will set episode_size=1 to get one embedding per piece of text:
    input_ids = input_ids.unsqueeze(1).unsqueeze(1)
    attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
    input_ids = input_ids.transpose(0, 2)
    attention_mask = attention_mask.transpose(0, 2)
    embeddings, _ = luar_model((input_ids, attention_mask))
    return embeddings.detach()
