''' convert paraphrased data to dataset format '''

import os
import json
import glob

import click
from datasets import Dataset
from datasets import load_dataset
from datasets import disable_caching
from datetime import datetime


def get_date():
    '''get date'''
    return datetime.now().strftime("%Y-%m-%d-%H.%M.%S")


def make_dataset(*, train_paths, val_paths, test_paths):
    if test_paths is None or test_paths == []:
        print("No test paths provided, using val paths for test")
        test_paths = val_paths

    dataset = load_dataset(
        "json", data_files={"train": train_paths, "val": val_paths, "test": test_paths}
    )
    return dataset


def save_info(path, info):
    with open(path, 'w') as f:
        json.dump(info, f)


@click.command()
@click.option('--in_dir', help='where file paths are.')
@click.option('--out_dir', help='out dir')
@click.option('--name', default='authorship_data', help='name of dataset')
def main(in_dir, out_dir, name):
    disable_caching()

    all_file_paths = sorted(glob.glob(os.path.join(in_dir, "*.jsonl")))

    train_paths = [x for x in all_file_paths if os.path.basename(x).startswith('train')]
    val_paths = [x for x in all_file_paths if os.path.basename(x).startswith('val')]
    test_paths = [x for x in all_file_paths if os.path.basename(x).startswith('test')]

    print(train_paths)
    print(val_paths)
    print(test_paths)

    dataset = make_dataset(
        train_paths=train_paths,
        val_paths=val_paths,
        test_paths=test_paths,
    )

    cur_date = get_date()
    print(out_dir)
    os.makedirs(out_dir, exist_ok=False)
    info = {
        "name": name,
        "train": len(dataset['train']),
        "val": len(dataset['val']),
        "test": len(dataset['test']),
        "train_paths": train_paths,
        "val_paths": val_paths,
        "test_paths": test_paths,
        "date": cur_date,
    }
    save_info(os.path.join(out_dir, "info.json"), info)
    dataset.save_to_disk(os.path.join(out_dir, name))


if __name__ == '__main__':
    main()
