'''

Combine results across experiments

'''

import os
import json
import glob


def load_args(fpath):
    folder = os.path.dirname(fpath)

    # file is either args.json or hparams.json
    args_fname = os.path.join(folder, 'args.json')
    if not os.path.exists(args_fname):
        args_fname = os.path.join(folder, 'hparams.json')
        assert os.path.exists(args_fname)

    with open(args_fname, 'r') as f:
        args = json.load(f)

    return args


def load_scores(fpath):
    with open(fpath, 'r') as f:
        scores = json.load(f)

    return scores


def args_to_approach(args, fpath):
    if args.get('model_name') == "gpt-3.5-turbo-0125":
        return 'gpt-3.5'

    if args.get('model_name') == "gpt-4-turbo":
        return 'gpt-4'

    if "transfer_args" in args:
        if 'authorship_pairings_LARGE_SCALE' in args["transfer_args"].get(
            'checkpoint', ''
        ):
            return 'tinystyler_v2_scaled_0.85'

        if 'enc_dec_ft_v2_config_1_fixed' in args["transfer_args"].get(
            'checkpoint', ''
        ):
            return 'tstyle-sup'

        if '2024-05-05-03.31.44_backup' in args["transfer_args"].get('checkpoint', ''):
            return 'tstyle'

    prefix = ""

    if args.get('lr') == 200:
        return prefix + 'paraguide-200'

    if args.get('lr') == 800:
        return prefix + 'paraguide-800'

    if args.get('lr') == 1500:
        return prefix + 'paraguide-1500'

    if args.get('lr') == 2500:
        return prefix + 'paraguide-2500'

    return None


if __name__ == '__main__':
    result_dir = 'results'
    dataset_order = ['random', 'single', 'diverse']
    metric_order = ["away_score", "towards_score", "sim_score", "joint_score"]
    approaches = [
        'paraguide-200',
        'paraguide-800',
        'paraguide-1500',
        'paraguide-2500',
        'gpt-3.5',
        'gpt-4',
        'tstyle',
        'tstyle-reranked',
        'tstyle-sup',
        'tinystyler_v2_scaled_0.85',
    ]

    key_term = '_scores_avg.json-just-first_5'
    rerank_term = '_reranked_'
    expected_n = 225

    dataset_to_results = {}

    for dataset in dataset_order:
        path = os.path.join(result_dir, dataset)

        dataset_to_results[dataset] = {}

        all_score_files = []
        all_score_files += list(glob.glob(os.path.join(path, '*', f'*{key_term}*')))
        all_score_files += list(
            glob.glob(os.path.join(path, '*', '*', f'*{key_term}*'))
        )
        all_score_files += list(
            glob.glob(os.path.join(path, '*', '*', '*', f'*{key_term}*'))
        )

        print(f'Found {len(all_score_files)} files for {dataset}')
        approach_to_metric = {}

        for fpath in all_score_files:
            args = load_args(fpath)
            scores = load_scores(fpath)
            approach = args_to_approach(args, fpath)

            if approach is None:
                continue

            if rerank_term in fpath:
                approach += '-reranked'

            assert approach not in approach_to_metric, f'Duplicate approach: {approach}'

            if approach not in approaches:
                print(f'Unknown approach: {approach}')
                continue

            approach_to_metric[approach] = {}

            assert scores['n'] == expected_n

            if approach is None:
                print(f'Unknown approach: {args}')
                continue

            for metric in metric_order:
                approach_to_metric[approach][metric] = scores[metric]

        dataset_to_results[dataset] = approach_to_metric

    # hacky but works well with current latex format
    for approach in approaches:
        row = [approach]
        for dataset in dataset_order:
            for metric in metric_order:
                row.append(dataset_to_results[dataset][approach][metric])

        print(
            ' & '.join([f"{x:.2f}" if isinstance(x, float) else x for x in row])
            + ' \\\ '
        )
