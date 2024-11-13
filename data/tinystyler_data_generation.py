'''

Generate data with tinystyler reconstruction model

'''

import click
import os
import math
import json
import random

from tqdm import tqdm

from tinystyler import load_style_model, text_to_style

import sys

sys.path.append('../authorship_evaluation')
from tinystyler_authorship import perform_authorship_transfer


def generate_tasks(*, paraphrased_data, target_authors_to_texts):
    tasks = []

    for task in paraphrased_data:
        sample = {
            'source_author': task['author_id'],
            'source_text': task['text'],
            'target_author': task['target_author_id'],
            'source_paraphrase': task['paraphrase'],
            'target_author_texts': target_authors_to_texts[task['target_author_id']],
        }
        tasks.append(sample)

    return tasks


def get_task_data(args, file_path):
    tasks = []
    with open(file_path, 'r') as f:
        for line in f:
            tasks.append(json.loads(line))
    target_authors_to_texts = json.load(open(args['target_author_texts'], 'r'))

    print("Generating target author embeddings")
    style_model, style_tokenizer, _ = load_style_model(
        "AnnaWegmann/Style-Embedding"
    )  # "StyleDistance/styledistance")
    style_model.eval()
    style_model.to(args['device'])

    task_data = generate_tasks(
        paraphrased_data=tasks, target_authors_to_texts=target_authors_to_texts
    )

    target_author_embeddings = {}
    for sample in tqdm(task_data):
        author = sample['target_author']
        author_target_texts = target_authors_to_texts[author]
        target_author_embeddings[author] = text_to_style(
            model=style_model,
            tokenizer=style_tokenizer,
            texts=author_target_texts,
            device=args['device'],
            model_type='style',
        )
        target_author_embeddings[author] = [
            x.detach().cpu() for x in target_author_embeddings[author]
        ]

    return task_data, target_author_embeddings


@click.command()
@click.option('--worker_idx', type=int, default=0)
@click.option('--num_workers', type=int, default=1)
@click.option('--transfer_data_dir', type=str, required=True)
@click.option('--checkpoint_path', type=str, required=True)
def main(worker_idx, num_workers, transfer_data_dir, checkpoint_path):
    random.seed(42)

    args = {
        'transfer_args': {
            'base_model': 'google/t5-v1_1-large',
            'device': 'cuda',
            'embed_selection': 'mean',
            'mean_sample': list(range(4, 9)),  # set this up to be between 4 and 8
            'max_length_input': 80,
            'max_length_output': 80,
            'use_actual_input': False,  # performing paraphrase reconstruction
            'combine_actual_para': False,
            'do_sample': True,
            'checkpoint': checkpoint_path,
            'top_p': 0.80,
            'temp': 1.0,
            'max_length_input': 80,
            'max_length_output': 80,
            'out_dir': transfer_data_dir + '/transfer_results',
        },
        'data_args': {
            'device': 'cuda',
            'paraphrased_dir': transfer_data_dir
            + '/transfers_for_finetune/paraphrased_files/topp0.8_tmp1.5',
            'target_author_texts': transfer_data_dir
            + '/texts_by_author_for_finetune.json',
        },
    }

    data_args = args['data_args']

    paraphrased_files = sorted(
        [
            os.path.join(data_args['paraphrased_dir'], x)
            for x in os.listdir(data_args['paraphrased_dir'])
        ]
    )

    per_process = math.ceil(len(paraphrased_files) / num_workers)
    start = worker_idx * per_process
    end = min((worker_idx + 1) * per_process, len(paraphrased_files))

    os.makedirs(args['transfer_args']['out_dir'], exist_ok=True)

    total_files = len(paraphrased_files)
    print('Num files:', total_files)
    print('For worker:', worker_idx, 'Processing:', len(paraphrased_files[start:end]))
    print()

    for paraphrase_file in tqdm(paraphrased_files[start:end]):
        print('Processing:', paraphrase_file)
        out_file_name = (
            os.path.basename(paraphrase_file).replace('.jsonl', '')
            + '_transfer_results.jsonl'
        )

        print('Loading data and making style embeddings')
        task_data, target_author_embeddings = get_task_data(data_args, paraphrase_file)
        transfer_args = args['transfer_args']

        print('Performing authorship transfer')
        perform_authorship_transfer(
            args=transfer_args,
            tasks=task_data,
            target_author_embeddings=target_author_embeddings,
            out_file_name=out_file_name,
        )


if __name__ == "__main__":
    main()
