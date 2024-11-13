"""

Utilities for loading style models and extracting style embeddings from texts

"""

import torch
from transformers import AutoModel, AutoTokenizer


def load_style_model(model_name='AnnaWegmann/Style-Embedding'):
    """
    Load model, tokenizer, word embeddings

    Adapted from: https://github.com/zacharyhorvitz/ParaGuide

    Supports
    - Wegmann Style Embeddings https://huggingface.co/AnnaWegmann/Style-Embedding
    - StyleDistance Embeddings https://huggingface.co/StyleDistance/styledistance
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    word_embeds = get_word_embeddings(model)
    return model, tokenizer, word_embeds


def mean_pooling(model_output, attention_mask):
    """
    Mean pool over non mask tokens.

    From https://huggingface.co/AnnaWegmann/Style-Embedding

    """

    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def text_to_style(
    *, model, tokenizer, texts, device, model_type='style', max_length=512
):
    """

    Applies get_style_embedding() to a list of texts

    Args:
        model (torch.nn.Module): PyTorch model
        tokenizer (transformers.Tokenizer): tokenizer
        texts (List[str]): list of texts
        device (str): device
        model_type (str): model type (currently only 'style' is supported)
        max_length (int): max length of input text

    Returns:
        List[torch.Tensor]: style embeddings

    """

    inputs = tokenizer(
        texts, return_tensors='pt', padding=True, truncation=True, max_length=max_length
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    embeds = get_style_embedding(
        model=model,
        input_tokens=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        model_type=model_type,
    )

    embeds = [x for x in embeds]
    return embeds


def get_style_embedding(
    *,
    model,
    inputs_embeds=None,
    input_tokens=None,
    attention_mask=None,
    model_type='style',
):
    """

    Get style embeddings from a model


    Adapted from: https://github.com/zacharyhorvitz/ParaGuide

    Args:
        model (torch.nn.Module): PyTorch model
        inputs_embeds (torch.Tensor): input token embeddings **optional**
        input_tokens (torch.Tensor): input tokens **optional**
        attention_mask (torch.Tensor): attention mask **optional**
        model_type (str): model type (currently only 'style' is supported)

    Returns:

        torch.Tensor: style embeddings  (batch_size, style embed_dim)

    """

    if inputs_embeds is not None and input_tokens is not None:
        raise ValueError('inputs_embeds and input_tokens cannot be both not None')

    if model_type != 'style':
        raise ValueError(
            f'Unknown model type {model_type}, only model type currently supported'
        )

    if inputs_embeds is not None:
        if attention_mask is None:
            attention_mask = torch.ones(*inputs_embeds.shape[:-1]).to(
                inputs_embeds.device
            )  # this may be why I have issues when i insert padding tokens
        attention_mask = attention_mask.to(inputs_embeds.device)
        return mean_pooling(
            model(inputs_embeds=inputs_embeds, attention_mask=attention_mask),
            attention_mask=attention_mask,
        )

    else:
        if attention_mask is None:
            attention_mask = torch.ones(*input_tokens.shape).to(input_tokens.device)
        attention_mask = attention_mask.to(input_tokens.device)
        return mean_pooling(
            model(input_tokens, attention_mask=attention_mask),
            attention_mask=attention_mask,
        )


def get_word_embeddings(model):
    """

    returns the word embeddings from a PyTorch model
    model.get_input_embeddings() also works for most Hugging Face models

    Args:
        model (torch.nn.Module): PyTorch model

    Returns:
        torch.Tensor: word embeddings

    """
    state_dict = model.state_dict()
    params = []
    for key in state_dict:
        if 'word_embeddings' in key:
            params.append((key, state_dict[key]))
    assert len(params) == 1, f'Found {params}'
    return params[0][1]
