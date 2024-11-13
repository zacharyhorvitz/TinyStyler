"""

Example generation script for TinyStyler.

Usage:
    python tinystyler/generation/generate.py \
        --input_path <path to input file> \
        --target_path <path to target style file> \
        --model_path <path to model checkpoint> \
        --out_dir <path to output directory> \
        --device <device to use> \
        --top_p <top p value> \
        --temperature <temperature value> \
        --basename <base model name> \
        --seed <seed value> \
        --num_inferences <number of inferences>

"""

import os
import torch
import time
import json
from tqdm.auto import tqdm
from datetime import datetime
import random

import argparse

from transformers import T5Tokenizer

from tinystyler import TinyStyler, load_style_model, text_to_style


def get_wegmann_embed(args, texts):
    '''
    Get mean Wegmann style embedding for a list of texts

    Args:

    args (argparse.Namespace): arguments
    texts (List[str]): list of texts

    Returns:

    torch.Tensor: mean style embedding

    '''
    embedding = text_to_style(
        model=args.embed_model,
        tokenizer=args.embed_tokenizer,
        texts=texts,
        device=args.device,
    )
    embedding = torch.stack(embedding).mean(dim=0)
    return embedding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_path', type=str, required=True, help='Path to input file'
    )
    parser.add_argument(
        '--target_path', type=str, required=True, help='Path to target style file'
    )
    parser.add_argument('--top_p', type=float, default=0.8, help='Top p value')
    parser.add_argument(
        '--temperature', type=float, default=1.0, help='Temperature value'
    )
    parser.add_argument(
        '--basename', type=str, default='google/t5-v1_1-large', help='Base model name'
    )
    parser.add_argument(
        '--model_path', type=str, required=True, help='Path to model checkpoint'
    )
    parser.add_argument('--seed', type=int, default=4242, help='Seed value')
    parser.add_argument(
        '--num_inferences', type=int, default=1, help='Number of inferences'
    )
    parser.add_argument(
        '--out_dir', type=str, default='outputs', help='Output directory'
    )
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')

    args = parser.parse_args()

    # check if device is available
    if not torch.cuda.is_available() and args.device == 'cuda':
        print('CUDA is not available. Using CPU')
        args.device = 'cpu'

    # set seed for reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = TinyStyler(base_model=args.basename, use_style=True, ctrl_embed_dim=768)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.eval()

    # TODO: add support for low precision

    base_tokenizer = T5Tokenizer.from_pretrained(args.basename)
    model.to(args.device)

    task_folder = os.path.join(
        args.out_dir,
        '->'.join(
            [
                'tinystyler',
                os.path.basename(args.input_path),
                os.path.basename(args.target_path),
            ]
        ),
    )
    dtime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    task_folder = os.path.join(task_folder, dtime)
    os.makedirs(task_folder, exist_ok=False)
    args.task_folder = task_folder

    with open(os.path.join(task_folder, "args.json"), 'w') as f:
        json.dump(vars(args), f)

    with open(args.target_path, 'r') as f:
        target_style_examples = [l.strip() for l in f.readlines()]

    with open(args.input_path, 'r') as f:
        input_data = [l.strip() for l in f.readlines()]

    ## load wegman style model
    embed_model, embed_tokenizer, _ = load_style_model()

    args.embed_model = embed_model
    args.embed_tokenizer = embed_tokenizer
    args.embed_model.to(args.device)
    args.embed_model.eval()
    args.embed_model_target_embeds = []

    for example in target_style_examples:
        style_embedding = get_wegmann_embed(args, [example])
        args.embed_model_target_embeds.append(style_embedding)

    total_transfers = len(input_data)
    with open(os.path.join(task_folder, "results.jsonl"), 'w+') as out:
        with tqdm(total=total_transfers) as pbar:
            for original_text in input_data:
                start = time.time()
                input = original_text

                batch_ctrl_embeds = (
                    torch.stack(args.embed_model_target_embeds)
                    .mean(dim=0)
                    .unsqueeze(0)
                    .repeat(args.num_inferences, 1, 1)
                )
                batch_ctrl_embeds = batch_ctrl_embeds.squeeze(1)

                input = [input] * args.num_inferences

                encoded_input = base_tokenizer(
                    input,
                    return_tensors='pt',
                    padding='max_length',
                    max_length=80,
                    truncation=True,
                ).to(args.device)
                outputs = model.generate(
                    **encoded_input,
                    style=batch_ctrl_embeds,
                    max_length=80,
                    do_sample=True,
                    top_p=args.top_p,
                    temperature=args.temperature,
                )
                outputs = base_tokenizer.batch_decode(outputs, skip_special_tokens=True)
                outputs = [outputs]

                result = dict(
                    input_label=args.input_path,
                    original_text=original_text,
                    target_label=args.target_path,
                    decoded=outputs,
                )

                print(f'{original_text} ->' + "\n\t->" + "\n\t->".join(outputs[0]))
                out.write(json.dumps(result) + '\n')
                print('Elapsed:', time.time() - start)

                pbar.update(1)


if __name__ == '__main__':
    main()
