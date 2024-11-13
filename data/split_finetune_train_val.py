''' split up training data into training and validation sets '''

import os
import glob

import random

import click

import math


@click.command()
@click.option('--in_dir', help='where file paths are.')
@click.option(
    '--percent_val', default=0.05, help='percent of data to use for validation'
)
@click.option('--seed', default=42, help='seed for random split')
def main(in_dir, percent_val, seed):
    random.seed(seed)

    all_file_paths = sorted(glob.glob(os.path.join(in_dir, "*.jsonl")))

    print("\n".join(all_file_paths))
    print(f"Found {len(all_file_paths)} files")

    all_data = []
    for file_path in all_file_paths:
        with open(file_path, 'r') as f:
            data = [line for line in f]
            all_data.extend(data)

    random.shuffle(all_data)

    num_val = math.ceil(len(all_data) * percent_val)

    val_data = all_data[:num_val]

    train_data = all_data[num_val:]

    out_dir = os.path.join(in_dir, 'train_val_split')

    print(f"Writing {len(train_data)} training examples to {out_dir}/train.jsonl")
    print(f"Writing {len(val_data)} validation examples to {out_dir}/val.jsonl")

    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, 'train.jsonl'), 'w') as f:
        for line in train_data:
            f.write(line)

    with open(os.path.join(out_dir, 'val.jsonl'), 'w') as f:
        for line in val_data:
            f.write(line)


if __name__ == '__main__':
    main()
