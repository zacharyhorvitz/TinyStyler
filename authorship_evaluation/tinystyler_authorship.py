'''

Logic for authorship transfer evaluations on the Styll dataset using TinyStyler (i.e. Table 2)

'''

import os

import json

import torch
import random
import click

from datetime import datetime

from tqdm import tqdm

import pickle

from transformers import (
    PegasusForConditionalGeneration,
    PegasusTokenizer,
    AutoTokenizer,
)

from tinystyler import TinyStyler, MODEL_TO_MODEL_TYPE, load_style_model, text_to_style


import hashlib


def get_paraphrases(
    *,
    model,
    tokenizer,
    input_texts,
    num_return_sequences=1,
    top_p=0.80,
    temp=1.5,
    max_length_input=60,
    max_length_output=60,
    device='cuda',
):
    """

    Generate paraphrases for a list of input texts

    Args:
        model: Pegasus model
        tokenizer: Pegasus tokenizer
        input_texts: list of input texts
        num_return_sequences: number of paraphrases to generate for each input text
        top_p: top p value for sampling
        temp: temperature for sampling
        max_length_input: maximum length of input text
        max_length_output: maximum length of output text
        device: device to run model on


    Returns:
        list of lists of paraphrases for each input text
    """
    batch = tokenizer(
        input_texts,
        truncation=True,
        padding='longest',
        max_length=max_length_input,
        return_tensors="pt",
    ).to(device)

    translated = model.generate(
        **batch,
        max_length=max_length_output,
        do_sample=True,
        top_p=top_p,
        num_return_sequences=num_return_sequences,
        temperature=temp,
    )
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)

    reshaped = []
    for i in range(len(input_texts)):
        reshaped.append(
            tgt_text[i * num_return_sequences : (i + 1) * num_return_sequences]
        )

    return reshaped


def generate_tasks(*, data, source_author_paraphrases=None):
    """

    Given dictionary loaded from Styll data, generate paired tasks for authorship transfer

    """
    tasks = []
    for source_author, source_texts in data['source_authors'].items():
        if source_author_paraphrases:
            source_paraphrases = source_author_paraphrases[source_author]
        else:
            source_paraphrases = [None for _ in source_texts]

        assert len(source_texts) == len(source_paraphrases)

        for target_author, target_texts in data['target_authors'].items():
            for source_text, source_paraphrase in zip(source_texts, source_paraphrases):
                sample = {
                    'source_author': source_author,
                    'source_paraphrase': source_paraphrase,
                    'source_text': source_text,
                    'target_author': target_author,
                    'source_author_texts': source_texts,
                    'target_author_texts': target_texts,
                }
                tasks.append(sample)

    return tasks


def load_raw_authorship_data(data_path):
    '''
    Load raw authorship data from a json file
    '''
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data


def load_data(data_dir):
    '''

    If cached intermediate processed data and target author embeddings are already saved, load them

    '''
    with open(os.path.join(data_dir, 'args.json'), 'r') as f:
        args = json.load(f)
    assert args == args

    with open(os.path.join(data_dir, 'task_data.jsonl'), 'r') as f:
        task_data = [json.loads(l) for l in f.readlines()]

    with open(os.path.join(data_dir, 'target_author_embeddings.pkl'), 'rb') as f:
        target_author_embeddings = pickle.load(f)

    return task_data, target_author_embeddings


def make_data(args, data_dir):
    '''

    Generate paraphrases for source authors and target author embeddings, and save them to disk

    Args:
        args: dictionary of arguments
        data_dir: directory to save data to

    Returns:
        task_data: list of tasks
        target_author_embeddings: dictionary of target author embeddings

    '''

    data = load_raw_authorship_data(args['data_path'])

    paraphraser_tokenizer = PegasusTokenizer.from_pretrained(args['paraphraser_name'])
    paraphraser_model = PegasusForConditionalGeneration.from_pretrained(
        args['paraphraser_name']
    ).to(args['paraphraser_args']['device'])

    print("Generating source author paraphrases")
    source_author_paraphrases = {}
    for author in tqdm(sorted(data['source_authors'].keys())):
        input_texts = data['source_authors'][author]
        paraphrases = []
        for batch in range(0, len(input_texts), args['paraphraser_batch_size']):
            paraphrases.extend(
                get_paraphrases(
                    **args['paraphraser_args'],
                    model=paraphraser_model,
                    tokenizer=paraphraser_tokenizer,
                    input_texts=input_texts[
                        batch : batch + args['paraphraser_batch_size']
                    ],
                )
            )
        assert len(paraphrases) == len(input_texts)
        source_author_paraphrases[author] = paraphrases

    print("Generating target author embeddings")
    style_model, style_tokenizer, _ = load_style_model()
    style_model.eval()
    style_model.to(args['device'])
    target_author_embeddings = {}
    for author in tqdm(sorted(data['target_authors'].keys())):
        author_target_texts = data['target_authors'][author]
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

    task_data = generate_tasks(
        data=data, source_author_paraphrases=source_author_paraphrases
    )

    os.makedirs(data_dir, exist_ok=True)

    with open(os.path.join(data_dir, 'args.json'), 'w') as f:
        json.dump(args, f)

    with open(os.path.join(data_dir, 'task_data.jsonl'), 'w') as f:
        for task in task_data:
            f.write(json.dumps(task) + '\n')

    with open(os.path.join(data_dir, 'target_author_embeddings.pkl'), 'wb') as f:
        pickle.dump(target_author_embeddings, f)

    return task_data, target_author_embeddings


def perform_authorship_transfer(
    *, args, tasks, target_author_embeddings, out_file_name='results.jsonl'
):
    '''

    Perform authorship transfer using TinyStyler

    Args:
        args: dictionary of arguments
        tasks: list of tasks
        target_author_embeddings: dictionary of target author embeddings
        out_file_name: name of output file

    Important values in args dict:
        base_model - core model name
        device - device to run on
        embed_slection - strategy for selecting target ebeddings
        mean_sample - number of targer embeddings to sample, or list of possible numbers to select
        max_length_input - maximum length of input text
        max_length_output - maximum length of output text
        use_actual_input - whether to use actual input text or paraphrases
        combine_actual_para - whether to combine actual input text with paraphrases
        checkpoint - path to model checkpoint
        do_sample - whether to sample
        top_p - top p value for sampling
        temp - temperature for sampling
        out_dir - directory to save output

    '''

    print("Loading TinyStyler model")
    device = args['device']
    model = TinyStyler(
        base_model=args['base_model'], use_style=True, ctrl_embed_dim=768
    )

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args['base_model'])

    model.to(device)

    print(f"Loading TinyStyle model state dict from {args['checkpoint']}")
    current_state = model.state_dict()
    # import pdb; pdb.set_trace()
    saved_state_dict = torch.load(args['checkpoint'], map_location=device)
    # remove 'module.' prefix from keys
    saved_state_dict = {
        k.replace('module.', ''): v for k, v in saved_state_dict.items()
    }
    current_state.update(saved_state_dict)
    model.load_state_dict(current_state)

    out_dir = args['out_dir']
    out_file = os.path.join(out_dir, out_file_name)

    with open(out_file, 'w') as f:
        for task in tqdm(tasks):
            target_author = task['target_author']
            target_embeds = target_author_embeddings[target_author]

            input_text = task['source_paraphrase']

            num_inputs = len(input_text)

            if args['use_actual_input'] and args['combine_actual_para']:
                input_text = input_text + [task['source_text']] * num_inputs
                num_inputs = len(input_text)

            elif args['use_actual_input']:
                input_text = [task['source_text']] * num_inputs

            if args.get('do_lower', False):
                input_text = [text.lower() for text in input_text]

            if args['embed_selection'] == 'random':
                batch_ctrl_embeds = (
                    random.choice(target_embeds).unsqueeze(0).repeat(num_inputs, 1)
                )
                # batch_ctrl_embeds = batch_ctrl_embeds.squeeze(1)
            elif args['embed_selection'] == 'first':
                batch_ctrl_embeds = target_embeds[0].unsqueeze(0).repeat(num_inputs, 1)
            elif args['embed_selection'] == 'mean':
                assert isinstance(args['mean_sample'], (int, list))

                if isinstance(args['mean_sample'], list):
                    selected_embeddings = []
                    for _ in range(num_inputs):
                        num_selected = random.choice(args['mean_sample'])
                        num_selected = min(num_selected, len(target_embeds))
                        selected_embeddings.append(
                            torch.stack(random.sample(target_embeds, num_selected))
                            .mean(dim=0)
                            .unsqueeze(0)
                        )
                    batch_ctrl_embeds = torch.cat(selected_embeddings, dim=0)

                elif (
                    isinstance(args['mean_sample'], int)
                    and args['mean_sample'] > 0
                    and args['mean_sample'] != len(target_embeds)
                ):
                    selected_embeddings = []
                    for _ in range(num_inputs):
                        num_selected = args['mean_sample']
                        num_selected = min(num_selected, len(target_embeds))
                        selected_embeddings.append(
                            torch.stack(random.sample(target_embeds, num_selected))
                            .mean(dim=0)
                            .unsqueeze(0)
                        )
                    batch_ctrl_embeds = torch.cat(selected_embeddings, dim=0)

                else:
                    batch_ctrl_embeds = (
                        torch.stack(target_embeds)
                        .mean(dim=0)
                        .unsqueeze(0)
                        .repeat(num_inputs, 1)
                    )
            else:
                raise ValueError(
                    f"Unknown embed selection method: {args['embed_selection']}"
                )

            batch_ctrl_embeds = batch_ctrl_embeds.to(device)

            if MODEL_TO_MODEL_TYPE[args['base_model']] == 't5':
                encoded_input = tokenizer(
                    input_text,
                    return_tensors='pt',
                    padding=True,
                    max_length=args['max_length_input'],
                    truncation=True,
                ).to(device)

                outputs = model.generate(
                    **encoded_input,
                    style=batch_ctrl_embeds,
                    max_length=args['max_length_output'],
                    do_sample=args['do_sample'],
                    top_p=args['top_p'],
                    temperature=args['temp'],
                )
            elif MODEL_TO_MODEL_TYPE[args['base_model']] == 'llama':
                raise ValueError("LLAMA model not supported")
            else:
                raise ValueError(
                    f"Unknown model type: {MODEL_TO_MODEL_TYPE[args['base_model']]}"
                )

            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            task['output'] = outputs

            f.write(json.dumps(task) + '\n')


# example usage
# python tinystyler_authorship.py \
#    --styll_dataset_shard diverse \
#    --model_checkpoint_path /mnt/swordfish-pool2/horvitz/reddit_mud/raw_all/emnlp/authorship_pairings_LARGE_SCALE/supervised_ft_models/2024-10-30-01.48.31/best_model_google_t5-v1_1-large_1e-05_64.pt \
#    --output_dir results_new \
#    --use_actual_input True \
#    --experiment_name tinystyler_ft_high_meaning


@click.command()
@click.option('--styll_dataset_shard', type=str, required=True)
@click.option('--model_checkpoint_path', type=str, required=True)
@click.option('--output_dir', type=str, default='results')
@click.option('--use_actual_input', type=bool, default=False)
@click.option('--experiment_name', type=str, default='experiment')
def main(
    styll_dataset_shard,
    model_checkpoint_path,
    output_dir,
    use_actual_input,
    experiment_name,
):
    random.seed(42)

    assert styll_dataset_shard in ['diverse', 'single', 'random']

    args = {
        'transfer_args': {
            'base_model': 'google/t5-v1_1-large',
            'device': 'cuda',
            'embed_selection': 'mean',  # mean over mean sample
            'mean_sample': 8,
            'max_length_input': 80,
            'max_length_output': 80,
            'use_actual_input': use_actual_input,  # False for reconstruction model, uses paraphrases instead
            'combine_actual_para': False,
            'do_sample': True,
            'checkpoint': model_checkpoint_path,
            'do_lower': False,
            'top_p': 0.80,
            'temp': 1.0,
            'max_length_input': 80,
            'max_length_output': 80,
            'out_dir': f'{output_dir}/{styll_dataset_shard}/{experiment_name}',
        },
        'data_args': {
            'data_path': f"../data/styll_evaluation_data/{styll_dataset_shard}.json",
            'paraphraser_name': 'tuner007/pegasus_paraphrase',
            'device': 'cuda',
            'paraphraser_batch_size': 16,
            'paraphraser_args': {
                'num_return_sequences': 5,
                'top_p': 0.80,
                'temp': 1.5,
                'max_length_input': 60,
                'max_length_output': 60,
                'device': 'cuda',
            },
            'prepared_data': f'data_cache/{styll_dataset_shard}_processed_5_para',
        },
    }

    data_args = args['data_args']
    os.makedirs(data_args['prepared_data'], exist_ok=True)

    print(data_args)
    hashed_args = hashlib.sha256(json.dumps(data_args).encode()).hexdigest()
    print(f"Hashed args: {hashed_args}")

    data_dir = os.path.join(data_args['prepared_data'], hashed_args)
    print(f"Data directory: {data_dir}")
    if os.path.exists(data_dir):
        print(f"Output directory already exists: {data_dir}, loading data")
        task_data, target_author_embeddings = load_data(data_dir)
    else:
        print(f"Output directory does not exist: {data_dir}, making data")
        task_data, target_author_embeddings = make_data(data_args, data_dir)

    args['input_data_path'] = data_dir
    transfer_args = args['transfer_args']

    write_directory = os.path.join(transfer_args['out_dir'])
    os.makedirs(write_directory, exist_ok=True)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    write_directory = os.path.join(write_directory, current_time)
    os.makedirs(write_directory, exist_ok=False)

    with open(os.path.join(write_directory, 'args.json'), 'w') as f:
        json.dump(args, f)

    args['transfer_args']['out_dir'] = write_directory

    perform_authorship_transfer(
        args=transfer_args,
        tasks=task_data,
        target_author_embeddings=target_author_embeddings,
    )


if __name__ == "__main__":
    main()
