import math
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from mutual_implication_score import MIS
from statistics import mean
from uar import get_uar_embedding
import torch

from tqdm import tqdm

from tinystyler import load_style_model, text_to_style
from transformers import AutoModel, AutoTokenizer


mis = None
style_model, style_tokenizer = None, None
mis_alternative = None
mis_alternative_tokenizer = None


def get_style_embedding(texts):
    global style_model, style_tokenizer
    if style_model is None:
        style_model, style_tokenizer, _ = load_style_model()
        style_model.eval()
        style_model.to('cuda')

    style_embeds = text_to_style(
        model=style_model,
        tokenizer=style_tokenizer,
        texts=texts,
        device='cuda',
        model_type='style',
    )
    return torch.stack(style_embeds).mean(dim=0).view(1, -1).detach().cpu().numpy()


def _sim(a, b):
    return clean_fp_error(cosine_similarity(a, b))


def asim(a, b):
    return 1 - np.arccos(_sim(a, b)) / np.pi


def mis_compute(a_texts, b_texts):
    global mis

    if mis is None:
        mis = MIS(device="cuda")

    scores = mis.compute(a_texts, b_texts)
    for idx, (score, a_text, b_text) in enumerate(zip(scores, a_texts, b_texts)):
        if a_text == b_text:
            scores[idx] = 1.0
    return scores


def mis_alternative_compute(a_texts, b_texts, batch_size=32):
    global mis_alternative
    global mis_alternative_tokenizer

    if mis_alternative is None:
        mis_alternative_tokenizer = AutoTokenizer.from_pretrained(
            "princeton-nlp/sup-simcse-roberta-large"
        )
        mis_alternative = AutoModel.from_pretrained(
            "princeton-nlp/sup-simcse-roberta-large"
        )
        mis_alternative.eval()
        mis_alternative.to('cuda')

    texts = a_texts + b_texts

    with torch.no_grad():
        all_embeddings = []

        for i in tqdm(list(range(0, len(texts), batch_size))):
            text_batch = texts[i : i + batch_size]

            inputs = mis_alternative_tokenizer(
                text_batch, padding=True, truncation=True, return_tensors="pt"
            ).to('cuda')

            embeddings = mis_alternative(
                **inputs, output_hidden_states=True, return_dict=True
            ).pooler_output

            all_embeddings.append(embeddings)

        embeddings = torch.cat(all_embeddings, dim=0)

        a_text_embeddings = embeddings[: len(a_texts)]
        b_text_embeddings = embeddings[len(a_texts) :]

        cos = torch.nn.CosineSimilarity(dim=1)
        sims = cos(a_text_embeddings, b_text_embeddings)
        scores = (sims + 1) / 2

    return scores.detach().cpu().numpy().tolist()


def clean_fp_error(v):
    '''Helps ignore floating point error ie. 1.00001 or -0.0000001'''
    if type(v) != float:
        v = v.item()
    v = min(max(v, 0), 1)
    if v > 0.99999 and v < 1.0:
        v = 1.0
    elif v < 0.00001 and v > 0.0:
        v = 0.0
    return v


def away_score(emb, source_emb):
    return clean_fp_error(1.0 - ((asim(emb, source_emb) + 1.0) / 2.0))


def towards_score(emb, target_emb):
    return clean_fp_error((asim(emb, target_emb) + 1.0) / 2.0)


def sim_score(texts, source_texts, alternative=False):
    if alternative:
        return clean_fp_error(
            mean(
                [
                    s
                    for s in mis_alternative_compute(source_texts, texts)
                    if not math.isnan(s)
                ]
            )
        )

    else:
        return clean_fp_error(
            mean([s for s in mis_compute(source_texts, texts) if not math.isnan(s)])
        )


def away_and_towards_score(
    *,
    source_texts,
    target_texts,
    style_transferred_texts,
    embedding_type='uar',
    source_emb=None,
    target_emb=None
):
    if embedding_type == 'uar':
        source_emb = (
            get_uar_embedding(source_texts) if source_emb is None else source_emb
        )
        target_emb = (
            get_uar_embedding(target_texts) if target_emb is None else target_emb
        )
        st_emb = get_uar_embedding(style_transferred_texts)
    elif embedding_type == 'style':
        source_emb = (
            get_style_embedding(source_texts) if source_emb is None else source_emb
        )
        target_emb = (
            get_style_embedding(target_texts) if target_emb is None else target_emb
        )
        st_emb = get_style_embedding(style_transferred_texts)
    else:
        raise ValueError('Invalid embedding type')

    # Calculate the away score
    score_away = away_score(st_emb, source_emb)

    # import pdb; pdb.set_trace()
    score_away_max = away_score(
        target_emb, source_emb
    )  # The maximum away we can get ever

    # import pdb; pdb.set_trace()
    score_away_scaled = clean_fp_error(
        min(score_away, score_away_max) / score_away_max
    )  # Ensures we can achieve a 1.0 when reaching the away maximum

    # Calculate the toward score
    score_towards = towards_score(st_emb, target_emb)
    score_towards_min = towards_score(
        source_emb, target_emb
    )  # The minimum towards we should get automatically
    score_towards_scaled = clean_fp_error(
        max(score_towards - score_towards_min, 0) / (1.0 - score_towards_min)
    )  # Ensures we can achieve a 0.0 when scoring at or below the towards minimum

    return (
        clean_fp_error(score_away_scaled),
        clean_fp_error(score_towards_scaled),
        (source_emb, target_emb),
    )


def joint_score(
    *,
    source_texts,
    target_texts,
    style_transferred_texts,
    embedding_type='uar',
    alternative_sim=False
):
    # Calculate the embedding of the texts

    if embedding_type == 'uar':
        source_emb = get_uar_embedding(source_texts)
        target_emb = get_uar_embedding(target_texts)
        st_emb = get_uar_embedding(style_transferred_texts)
    elif embedding_type == 'style':
        source_emb = get_style_embedding(source_texts)
        target_emb = get_style_embedding(target_texts)
        st_emb = get_style_embedding(style_transferred_texts)
    else:
        raise ValueError('Invalid embedding type')

    # Calculate the away score
    score_away = away_score(st_emb, source_emb)
    score_away_max = away_score(
        target_emb, source_emb
    )  # The maximum away we can get ever
    score_away_scaled = clean_fp_error(
        min(score_away, score_away_max) / score_away_max
    )  # Ensures we can achieve a 1.0 when reaching the away maximum

    # Calculate the toward score
    score_towards = towards_score(st_emb, target_emb)
    score_towards_min = towards_score(
        source_emb, target_emb
    )  # The minimum towards we should get automatically
    score_towards_scaled = clean_fp_error(
        max(score_towards - score_towards_min, 0) / (1.0 - score_towards_min)
    )  # Ensures we can achieve a 0.0 when scoring at or below the towards minimum

    # Calculate the sim score
    score_sim = sim_score(
        style_transferred_texts, source_texts, alternative=alternative_sim
    )
    score_sim_min = sim_score(
        target_texts, source_texts, alternative=alternative_sim
    )  # The minimum sim we should get automatically
    score_sim_scaled = clean_fp_error(
        max(score_sim - score_sim_min, 0) / (1.0 - score_sim_min)
    )  # Ensures we can achieve a 0.0 when scoring at or below the sim minimum

    # Calculate the joint score
    print(score_away_scaled, score_towards_scaled, score_sim_scaled)
    score_joint = clean_fp_error(
        (((score_away_scaled * score_towards_scaled) ** (1 / 2)) * score_sim_scaled)
        ** (1 / 2)
    )
    return clean_fp_error(score_joint), {
        'away': score_away_scaled,
        'towards': score_towards_scaled,
        'sim': score_sim_scaled,
    }
