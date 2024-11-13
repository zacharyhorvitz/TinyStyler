from functools import cache, lru_cache
import warnings
import itertools
import random
import torch
from statistics import mean
from time import time
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoModel,
    AutoTokenizer,
    T5ForConditionalGeneration,
    set_seed,
)
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download
import torch.nn.functional as F


#########################################################################
# Define the model
#########################################################################
class TinyStyler(torch.nn.Module):
    def __init__(self, base_model, use_style=False, ctrl_embed_dim=768):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(base_model)
        self.use_style = use_style
        if self.use_style:
            self.ctrl_embed_dim = ctrl_embed_dim
            if hasattr(self.model.config, "d_model"):
                self.proj = torch.nn.Linear(
                    self.ctrl_embed_dim, self.model.config.d_model
                )
            else:
                self.proj = torch.nn.Linear(
                    self.ctrl_embed_dim, self.model.config.hidden_size
                )

    def forward(self, input_ids, attention_mask, labels=None, style=None):
        if self.use_style:
            style_embed = self.proj(style).unsqueeze(1)

        input_embeds = self.model.get_input_embeddings()(input_ids)
        if self.use_style:
            input_embeds = torch.cat([style_embed, input_embeds], dim=1)
            attention_mask = torch.cat(
                [
                    torch.ones((input_embeds.shape[0], 1)).to(attention_mask.device),
                    attention_mask,
                ],
                dim=1,
            )

        return self.model(
            inputs_embeds=input_embeds, attention_mask=attention_mask, labels=labels
        )

    def generate(self, input_ids, attention_mask, style=None, **kwargs):
        if self.use_style:
            style_embed = self.proj(style.unsqueeze(1))

        input_embeds = self.model.get_input_embeddings()(input_ids)
        if self.use_style:
            input_embeds = torch.cat([style_embed, input_embeds], dim=1)
            attention_mask = torch.cat(
                [
                    torch.ones((input_embeds.shape[0], 1)).to(attention_mask.device),
                    attention_mask,
                ],
                dim=1,
            )

        return self.model.generate(
            inputs_embeds=input_embeds, attention_mask=attention_mask, **kwargs
        )


#########################################################################
# Define the model loading helpers
#########################################################################
@cache
def get_tinystyler_model(device, model_name='tinystyler'):
    expected_models = ['tinystyler', 'tinystyler_sim']

    if model_name not in expected_models:
        raise ValueError(f'{model_name} not in {expected_models}')

    base_model_name = "google/t5-v1_1-large"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, legacy=True)
    model = TinyStyler(
        base_model="google/t5-v1_1-large", use_style=True, ctrl_embed_dim=768
    )
    model.load_state_dict(
        torch.load(
            hf_hub_download(
                repo_id="tinystyler/tinystyler",
                filename=f"{model_name}/model_weights.pt",
            ),
            map_location=device,
        )
    )
    model.to(device)
    return tokenizer, model


@cache
def get_style_embedding_model(device):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message=".*resume_download.*",
            module="huggingface_hub.file_download",
        )
        embedding_model = SentenceTransformer(
            "AnnaWegmann/Style-Embedding", device=device
        )
    return embedding_model


@cache
def get_luar_model(device):
    luar_model = AutoModel.from_pretrained(
        "rrivera1849/LUAR-MUD",
        revision="51b0d9ecec5336314e02f191dd8ca4acc0652fe1",
        trust_remote_code=True,
    )
    luar_model.to(device)
    luar_tokenizer = AutoTokenizer.from_pretrained(
        "rrivera1849/LUAR-MUD",
        revision="51b0d9ecec5336314e02f191dd8ca4acc0652fe1",
        trust_remote_code=True,
    )
    return luar_tokenizer, luar_model


@cache
def get_mis_model(device):
    from mutual_implication_score import MIS

    mis_model = MIS(device=device)
    mis_model.model = mis_model.model
    return mis_model


@cache
def get_simcse_model(device):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message=".*resume_download.*",
            module="huggingface_hub.file_download",
        )
        embedding_model = SentenceTransformer(
            "princeton-nlp/sup-simcse-roberta-base", device=device
        )
    return embedding_model


#########################################################################
# Define re-ranking helpers
#########################################################################
def list_to_tuple(function):
    """Custom decorator function, to convert list to a tuple."""

    def convert_list_to_tuple(x):
        return (
            tuple([convert_list_to_tuple(y) for y in x]) if isinstance(x, list) else x
        )

    def wrapper(*args, **kwargs):
        args = tuple(convert_list_to_tuple(x) for x in args)
        kwargs = {k: convert_list_to_tuple(v) for k, v in kwargs.items()}
        result = function(*args, **kwargs)
        result = tuple(result) if isinstance(result, list) else result
        return result

    return wrapper


@list_to_tuple
@lru_cache(maxsize=256)
@torch.no_grad()
def get_target_style_embeddings(target_texts_batch, device):
    embedding_model = get_style_embedding_model(device)
    all_target_texts = [
        target_text
        for target_texts in target_texts_batch
        for target_text in target_texts
    ]
    embeddings = embedding_model.encode(
        all_target_texts,
        batch_size=len(all_target_texts),
        convert_to_tensor=True,
        show_progress_bar=False,
    )
    lengths = [len(target_texts) for target_texts in target_texts_batch]
    split_embeddings = torch.split(embeddings, lengths)
    padded_embeddings = pad_sequence(
        split_embeddings, batch_first=True, padding_value=0.0
    )
    mask = (
        (
            torch.arange(padded_embeddings.size(1))[None, :]
            < torch.tensor(lengths)[:, None]
        )
        .to(embeddings.dtype)
        .unsqueeze(-1)
    ).to(device)
    mean_embeddings = torch.sum(padded_embeddings * mask, dim=1) / mask.sum(dim=1)
    return mean_embeddings.float().cpu()


@list_to_tuple
@lru_cache(maxsize=512)
@torch.no_grad()
def get_luar_embeddings(texts_batch, device):
    luar_tokenizer, luar_model = get_luar_model(device)
    assert len(set([len(texts) for texts in texts_batch])) == 1
    episodes = texts_batch
    for first_round in [True, False]:
        if first_round:
            tokenized_episodes = [
                luar_tokenizer(
                    episode,
                    max_length=512,
                    padding="longest",
                    truncation=True,
                    return_tensors="pt",
                ).to(device)
                for episode in episodes
            ]
        else:
            tokenized_episodes = [
                luar_tokenizer(
                    episode,
                    max_length=max(
                        [t["attention_mask"].shape[1] for t in tokenized_episodes]
                    ),
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).to(device)
                for episode in episodes
            ]
    episode_lengths = [t["attention_mask"].shape[0] for t in tokenized_episodes]
    max_episode_length = max(episode_lengths)
    padded_input_ids = [
        torch.nn.functional.pad(
            t["input_ids"], (0, 0, 0, max_episode_length - t["input_ids"].shape[0])
        )
        for t in tokenized_episodes
    ]
    padded_attention_mask = [
        torch.nn.functional.pad(
            t["attention_mask"],
            (0, 0, 0, max_episode_length - t["attention_mask"].shape[0]),
        )
        for t in tokenized_episodes
    ]
    input_ids = torch.stack(padded_input_ids)
    attention_mask = torch.stack(padded_attention_mask)
    return luar_model(input_ids=input_ids, attention_mask=attention_mask).cpu()


def asim(a, b):
    cos_sim = F.cosine_similarity(a.unsqueeze(1), b.unsqueeze(0), dim=2)
    arccos_sim = torch.arccos(cos_sim)
    final_sim = 1 - arccos_sim / torch.pi
    return final_sim


def away_score(a, b):
    return 1.0 - ((asim(a, b) + 1.0) / 2.0)


def towards_score(a, b):
    return (asim(a, b) + 1.0) / 2.0


def joint_score(source_emb, target_emb, st_emb, baseline_sim, st_sim):
    # Calculate the away score
    score_away = away_score(st_emb, source_emb)
    score_away_max = away_score(
        target_emb, source_emb
    )  # The maximum away we can get ever
    score_away_scaled = (
        torch.min(score_away, score_away_max) / score_away_max
    )  # Ensures we can achieve a 1.0 when reaching the away maximum

    # Calculate the toward score
    score_towards = towards_score(st_emb, target_emb)
    score_towards_min = towards_score(
        source_emb, target_emb
    )  # The minimum towards we should get automatically
    score_towards_scaled = torch.max(
        score_towards - score_towards_min, torch.tensor(0.0)
    ) / (
        1.0 - score_towards_min
    )  # Ensures we can achieve a 0.0 when scoring at or below the towards minimum

    # Calculate the sim score
    score_sim = st_sim
    score_sim_min = baseline_sim  # The minimum sim we should get automatically
    score_sim_scaled = torch.max(score_sim - score_sim_min, torch.tensor(0.0)) / (
        1.0 - score_sim_min
    )  # Ensures we can achieve a 0.0 when scoring at or below the sim minimum

    # Calculate the joint score
    score_joint = torch.nan_to_num(
        (((score_away_scaled * score_towards_scaled) ** (1 / 2)) * score_sim_scaled)
        ** (1 / 2),
        nan=1.0,
    )
    return score_joint.flatten().tolist()


@torch.no_grad()
def compute_mis(texts, target_texts_batch, device):
    mis_model = get_mis_model(device)
    a_texts = list(
        itertools.chain.from_iterable(
            [
                [t] * len(target_texts)
                for t, target_texts in zip(texts, target_texts_batch)
            ]
        )
    )
    b_texts = list(itertools.chain.from_iterable(target_texts_batch))
    scores = mis_model.compute(a_texts, b_texts, batch_size=len(a_texts), verbose=False)
    for idx, (score, a_text, b_text) in enumerate(zip(scores, a_texts, b_texts)):
        if a_text == b_text:
            scores[idx] = 1.0
    final_scores = []
    current_idx = 0
    for target_texts in target_texts_batch:
        final_scores.append(mean(scores[current_idx : current_idx + len(target_texts)]))
        current_idx += len(target_texts)
    return final_scores


@torch.no_grad()
def compute_simcse(texts, target_texts_batch, device):
    sim_model = get_simcse_model(device)
    a_texts = list(
        itertools.chain.from_iterable(
            [
                [t] * len(target_texts)
                for t, target_texts in zip(texts, target_texts_batch)
            ]
        )
    )
    b_texts = list(itertools.chain.from_iterable(target_texts_batch))
    a_emb = sim_model.encode(
        a_texts,
        batch_size=len(a_texts),
        convert_to_tensor=True,
        show_progress_bar=False,
    )
    b_emb = sim_model.encode(
        b_texts,
        batch_size=len(a_texts),
        convert_to_tensor=True,
        show_progress_bar=False,
    )
    scores = sim_model.similarity_pairwise(a_emb, b_emb).flatten().tolist()
    for idx, (score, a_text, b_text) in enumerate(zip(scores, a_texts, b_texts)):
        if a_text == b_text:
            scores[idx] = 1.0
    final_scores = []
    current_idx = 0
    for target_texts in target_texts_batch:
        final_scores.append(mean(scores[current_idx : current_idx + len(target_texts)]))
        current_idx += len(target_texts)
    return final_scores


#########################################################################
# Define inference helpers
#########################################################################


@torch.no_grad()
def run_tinystyler_batch(
    source_texts,
    target_texts_batch,
    reranking,
    temperature,
    top_p,
    max_new_tokens=512,
    device="cpu",
    verbose=False,
    sim_func=compute_mis,
    rerank_style_embed_fn=get_target_style_embeddings,
    sim_sample=3,
    **kwargs,
):
    bz = len(source_texts)
    tokenizer, model = get_tinystyler_model(device)
    inputs = tokenizer(
        source_texts, padding="longest", truncation=True, return_tensors="pt"
    ).to(device)
    target_style_embeddings = get_target_style_embeddings(target_texts_batch, device)
    if verbose:
        print("Initial Log", time())

    rand = random.Random(42)
    sim_target_text_batch_sample = [
        rand.sample(target_texts, min(sim_sample, len(target_texts)))
        for target_texts in target_texts_batch
    ]
    baseline_sim = torch.tensor(
        sim_func(source_texts, sim_target_text_batch_sample, device)
    )
    if verbose:
        print("Computed Baseline Sim", time(), sim_target_text_batch_sample)

    # Generate the output with specified temperature and top_p
    if "seed" in kwargs:
        set_seed(kwargs.pop("seed"))
    output = model.generate(
        **inputs,
        style=target_style_embeddings.to(device),
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        num_return_sequences=reranking,
        **kwargs,
    )
    if verbose:
        print("Generated Candidates", time())
    generated_texts = tokenizer.batch_decode(output, skip_special_tokens=True)
    generated_texts = [
        generated_texts[i * reranking : (i + 1) * reranking] for i in range(bz)
    ]  # Unflatten

    # Evaluate candidates
    rerank_source_style_embeddings = rerank_style_embed_fn(
        [[st] for st in source_texts], device
    )
    rerank_target_style_embeddings = rerank_style_embed_fn(target_texts_batch, device)
    rerank_candidates_embeddings = [
        rerank_style_embed_fn(
            [[candidates[i]] for candidates in generated_texts], device
        )
        for i in range(reranking)
    ]

    if verbose:
        print("Computed Embeddings Over Candidates for Reranking", time())

    candidates_sim = [
        torch.tensor(
            sim_func(
                [candidates[i] for candidates in generated_texts],
                sim_target_text_batch_sample,
                device,
            ),
        )
        for i in range(reranking)
    ]
    if verbose:
        print("Computed Sim Over Candidates", time())
    candidates_joint = [
        joint_score(
            rerank_source_style_embeddings,
            rerank_target_style_embeddings,
            rerank_candidates_embeddings[i],
            baseline_sim,
            candidates_sim[i],
        )
        for i in range(reranking)
    ]
    if verbose:
        print("Computed Joint Over Candidates", time())

    # Re-rank candidates and keep the best ones
    sorted_candidates = [
        sorted(
            zip(
                generated_texts[i],
                [joint_scores[i] for joint_scores in candidates_joint],
            ),
            key=lambda x: x[1],
            reverse=True,
        )
        for i in range(bz)
    ]
    if verbose:
        print("Ranked Candidates", sorted_candidates)
    best_candidates = [c[0][0] for c in sorted_candidates]

    return best_candidates
