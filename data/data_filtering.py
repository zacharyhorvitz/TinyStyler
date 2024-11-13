'''  Load inference data and evaluate # number of pairs with meaning score > threshold '''


import os
import json
import sys
from tqdm import tqdm
import re

import click

import numpy as np
import math

STYLE_MODEL = "AnnaWegmann/Style-Embedding"
# "StyleDistance/styledistance"

from data_filtering_metrics import (
    mis_compute,
    away_and_towards_score,
    mis_alternative_compute,
)


def get_valid_indices(outputs):
    return [i for i, text in enumerate(outputs) if text is not None]


def count_valid_tasks(tasks):
    count = 0
    for task in tasks:
        if any(task['output']):
            count += 1
    return count


def filter_based_on_identity(tasks):
    for task in tqdm(tasks):
        source_text = task['source_text']
        output = task['output']

        for i in range(len(output)):
            if output[i] is None:
                continue
            if output[i].strip() == source_text.strip():
                output[i] = None

    tasks_remaining = count_valid_tasks(tasks)

    return tasks_remaining


def filter_to_max_results(tasks, max_per):
    for task in tqdm(tasks):
        output = task['output']

        for i in range(len(output)):
            if i >= max_per:
                output[i] = None


def flatten_refs_cands(tasks):
    all_indices = []
    all_refs = []
    all_cands = []

    for i, task in enumerate(tasks):
        source_text = task['source_text']
        output = task['output']
        valid_indices = get_valid_indices(output)

        references = [source_text] * len(valid_indices)
        candidates = [output[i] for i in valid_indices]
        indices = [[i, val_idx] for val_idx in valid_indices]

        all_indices.extend(indices)
        all_refs.extend(references)
        all_cands.extend(candidates)

    return all_indices, all_refs, all_cands


def filter_based_on_model_score(tasks, threshold, model_fn=lambda refs, cands: 0):
    tasks_remaining = 0

    i = 1
    while f'sim_{i}' in tasks[0]:
        i += 1

    key = f'sim_{i}'

    for task in tasks:
        task[key] = [None for _ in range(len(tasks[0]['output']))]

    indices, references, candidates = flatten_refs_cands(tasks)
    flatten_scores = model_fn(references, candidates)

    for (task_idx, inference_idx), score in zip(indices, flatten_scores):
        tasks[task_idx][key][inference_idx] = score
        if score < threshold:
            tasks[task_idx]['output'][inference_idx] = None

    tasks_remaining = count_valid_tasks(tasks)
    return tasks_remaining


def add_style_features(tasks, authors_to_texts):
    for task in tqdm(tasks):
        source_author = task['source_author']
        target_author = task['target_author']

        source_author_texts = authors_to_texts[source_author]
        target_author_texts = authors_to_texts[target_author]

        source_emb = None
        target_emb = None

        outputs = task['output']
        task['away_scores'] = [None for _ in range(len(outputs))]
        task['towards_scores'] = [None for _ in range(len(outputs))]
        for i in range(len(outputs)):
            if outputs[i] is not None:
                away, towards, (source_emb, target_emb) = away_and_towards_score(
                    source_texts=source_author_texts,
                    target_texts=target_author_texts,
                    style_transferred_texts=[outputs[i]],
                    embedding_type=f'style_{STYLE_MODEL}',
                    source_emb=source_emb,
                    target_emb=target_emb,
                )
                task['away_scores'][i] = away
                task['towards_scores'][i] = towards


def load_cached_data(path):
    with open(path, 'r') as f:
        tasks = [json.loads(line) for line in f]
    print('Loaded filtered data from cache')
    print(f'\tRemaining samples: {count_valid_tasks(tasks)}/{len(tasks)}')
    print()
    return tasks


def get_style_transfer_stats(tasks):
    away_scores = []
    towards_scores = []

    for task in tasks:
        if any(task['away_scores']):
            away_scores.append(max([s for s in task['away_scores'] if s is not None]))
            towards_scores.append(
                max([s for s in task['towards_scores'] if s is not None])
            )

    away_scores = np.array(away_scores)
    towards_scores = np.array(towards_scores)

    away_mean = away_scores.mean()
    away_median = np.median(away_scores)
    away_std = away_scores.std()

    towards_mean = towards_scores.mean()
    towards_median = np.median(towards_scores)
    towards_std = towards_scores.std()

    print('Style transfer stats:')
    print(f'\tAway: (mean: {away_mean}, median: {away_median}, std: {away_std})')
    print(
        f'\tTowards: (mean: {towards_mean}, median: {towards_median}, std: {towards_std})'
    )

    print('Possible away threshold:', away_mean - away_std)
    print('Possible towards threshold:', towards_mean - towards_std)


def filter_based_on_style(*, tasks, away_threshold, towards_threshold):
    for task in tasks:
        outputs = task['output']
        for i in range(len(outputs)):
            if (
                task['away_scores'][i] is not None
                and task['away_scores'][i] < away_threshold
            ):
                outputs[i] = None
            if (
                task['towards_scores'][i] is not None
                and task['towards_scores'][i] < towards_threshold
            ):
                outputs[i] = None

    tasks_remaining = count_valid_tasks(tasks)

    return tasks_remaining


def filter_hallucinated_links(tasks, link_regex):
    # only keep links that are not hallucinated
    for task in tasks:
        outputs = task['output']
        source = task['source_text']
        for i in range(len(outputs)):
            if outputs[i] is not None:
                links = re.findall(link_regex, outputs[i], re.IGNORECASE)
                if links:
                    source_links = re.findall(link_regex, source, re.IGNORECASE)
                    if not any([link in source_links for link in links]):
                        outputs[i] = None

    tasks_remaining = count_valid_tasks(tasks)

    return tasks_remaining


def filter_to_best_result(task):
    outputs = task['output']
    mis_scores = task['sim_1']
    away_scores = task['away_scores']
    towards_scores = task['towards_scores']

    best_idx = None
    best_score = -1

    for i in range(len(outputs)):
        if outputs[i] is not None:
            # score = mis_scores[i]
            score = (
                ((towards_scores[i] * away_scores[i]) ** (1 / 2)) * mis_scores[i]
            ) ** (
                1 / 2
            )  # NOTE: This is unnormalized, assumes mis is close to zero as is.
            if score > best_score:
                best_score = score
                best_idx = i

    for key in ['output', 'away_scores', 'towards_scores', 'sim_1', 'sim_2']:
        task[key] = task[key][best_idx]


def process_file(*, path, out_dir, do_cache, author_to_texts):
    # hardcoding for now
    MAX_PER = 5  # only include paraphrases
    MIS_SIM_THRESHOLD = 0.70  # 0.80 #0.85
    ALT_SIM_THRESHOLD = 0.70  # 0.80 #0.85
    AWAY_THRESH = 0.90
    TOWARDS_THRESH = 0.30

    fname = os.path.basename(path)

    cache_dir = os.path.join(out_dir, 'cache')

    if do_cache:
        os.makedirs(cache_dir, exist_ok=True)

    with open(path, 'r') as f:
        tasks = [json.loads(line) for line in f]

    initial_samples = len(tasks)

    print('Initial samples:', initial_samples)

    out_path_1 = os.path.join(
        cache_dir, f'{fname}_filtered_MIS_{MIS_SIM_THRESHOLD}.jsonl'
    )

    if os.path.exists(out_path_1) and do_cache:
        tasks = load_cached_data(out_path_1)
    else:
        print('Filtering based on x == x and max per')
        filter_to_max_results(tasks, MAX_PER)
        num_remaining = filter_based_on_identity(tasks)
        print(f'\tRemaining samples: {num_remaining}/{initial_samples}')

        print()

        print('Filtering based on MIS score')
        num_remaining = filter_based_on_model_score(
            tasks,
            threshold=MIS_SIM_THRESHOLD,
            model_fn=lambda refs, cands: mis_compute(refs, cands),
        )
        print(f'\tRemaining samples: {num_remaining}/{initial_samples}')
        print()

        if do_cache:
            with open(out_path_1, 'w') as f:
                for task in tasks:
                    f.write(json.dumps(task) + '\n')

    out_path_2 = os.path.join(
        cache_dir, f'{fname}_filtered_ALT_{ALT_SIM_THRESHOLD}.jsonl'
    )

    if os.path.exists(out_path_2) and do_cache:
        tasks = load_cached_data(out_path_2)
    else:
        print('Filtering based on Alternative MIS score')
        num_remaining = filter_based_on_model_score(
            tasks,
            threshold=ALT_SIM_THRESHOLD,
            model_fn=lambda refs, cands: mis_alternative_compute(refs, cands),
        )
        print(f'\tRemaining samples: {num_remaining}/{initial_samples}')
        print()

        if do_cache:
            with open(out_path_2, 'w') as f:
                for task in tasks:
                    f.write(json.dumps(task) + '\n')

    out_path_3 = os.path.join(
        cache_dir,
        f'{fname}_filtered_with_SIMS_{MIS_SIM_THRESHOLD}_{ALT_SIM_THRESHOLD}_with_style_metrics.jsonl',
    )
    if os.path.exists(out_path_3) and do_cache:
        tasks = load_cached_data(out_path_3)

    else:
        print('Adding style features')

        add_style_features(tasks, author_to_texts)

        if do_cache:
            with open(out_path_3, 'w') as f:
                for task in tasks:
                    f.write(json.dumps(task) + '\n')

    get_style_transfer_stats(tasks)

    print('Filtering based on style transfer')
    num_remaining = filter_based_on_style(
        tasks=tasks, away_threshold=AWAY_THRESH, towards_threshold=TOWARDS_THRESH
    )
    print(f'\tRemaining samples: {num_remaining}/{initial_samples}')

    print('Filtering based on hallucinated links')

    link_regex = r'(http|ftp|https):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])'
    num_remaining = filter_hallucinated_links(tasks, link_regex=link_regex)
    print(f'\tRemaining samples: {num_remaining}/{initial_samples}')

    if do_cache:
        final_raw_out_path = os.path.join(
            cache_dir,
            f'{fname}_raw_filtered_with_SIMS_{MIS_SIM_THRESHOLD}_{ALT_SIM_THRESHOLD}_STYLE_{AWAY_THRESH}_{TOWARDS_THRESH}_no_link.jsonl',
        )
        with open(final_raw_out_path, 'w') as f:
            for task in tasks:
                if any(task['output']):
                    f.write(json.dumps(task) + '\n')

    final_clean_path = os.path.join(
        out_dir,
        f'{fname}_filter_{MIS_SIM_THRESHOLD}_{ALT_SIM_THRESHOLD}_{AWAY_THRESH}_{TOWARDS_THRESH}_no_link.jsonl',
    )

    with open(final_clean_path, 'w') as f:
        for task in tasks:
            if not any(task['output']):
                continue
            else:
                filter_to_best_result(task)
                f.write(json.dumps(task) + '\n')


@click.command()
@click.option('--in_dir', type=str, required=True)
@click.option('--do_cache', is_flag=True)
@click.option('--worker_idx', type=int, default=0)
@click.option('--num_workers', type=int, default=1)
@click.option('--do_sample', is_flag=True)
def main(in_dir, do_cache, worker_idx, num_workers, do_sample):
    transfer_result_dir = os.path.join(in_dir, 'transfer_results')
    assert os.path.exists(transfer_result_dir)

    author_to_texts = os.path.join(in_dir, 'texts_by_author_for_finetune.json')
    assert os.path.exists(author_to_texts)

    out_dir = os.path.join(in_dir, 'filtered_authorship_pairings')
    os.makedirs(out_dir, exist_ok=True)

    with open(author_to_texts, 'r') as f:
        author_to_texts = json.load(f)

    transfer_results = sorted(
        [
            os.path.join(transfer_result_dir, f)
            for f in os.listdir(transfer_result_dir)
            if f.endswith('.jsonl')
        ]
    )
    per_process = math.ceil(len(transfer_results) / num_workers)
    start = worker_idx * per_process
    end = min((worker_idx + 1) * per_process, len(transfer_results))
    selected_files = transfer_results[start:end]

    if do_sample:
        selected_files = selected_files[:1]

    for path in tqdm(selected_files):
        process_file(
            path=path,
            out_dir=out_dir,
            do_cache=do_cache,
            author_to_texts=author_to_texts,
        )


if __name__ == '__main__':
    main()
