'''

Sample authorship pairings for finetuning

'''

import click
import json
import random
import os
from tqdm import tqdm


def get_all_authors(file_list):
    authors = set()

    for file in tqdm(file_list):
        with open(file, 'r') as f:
            for line in f:
                sample = json.loads(line)
                author = sample['author_id']
                authors.add(author)
    return authors


def get_texts_by_author(authors, file_list):
    texts_by_author = {}
    for author in authors:
        texts_by_author[author] = []

    for file in tqdm(file_list):
        with open(file, 'r') as f:
            for line in f:
                sample = json.loads(line)
                author = sample['author_id']
                if author in texts_by_author:
                    texts_by_author[author].append(sample['text'])
    return texts_by_author


@click.command()
@click.option('--in_dir', type=click.Path(exists=True))
@click.option('--out_dir', type=click.Path())
@click.option('--shard', default='train', type=str)
@click.option('--num_samples', default=100000, type=int)
@click.option('--seed', default=42, type=int)
@click.option('--skip_author_dict_path', default=None, type=click.Path(exists=True))
def main(in_dir, out_dir, shard, num_samples, seed, skip_author_dict_path):
    random.seed(seed)

    file_list = sorted(
        [
            os.path.join(in_dir, f)
            for f in os.listdir(in_dir)
            if f.endswith('.jsonl') and f.startswith(shard)
        ]
    )
    print('Found {} files in {}'.format(len(file_list), in_dir))

    authors = sorted(get_all_authors(file_list))
    if skip_author_dict_path:
        with open(skip_author_dict_path, 'r') as f:
            skip_authors = set(json.load(f).keys())
            print('Skipping {} authors'.format(len(skip_authors)))
            authors = sorted(set(authors) - skip_authors)

    print('Found {} authors'.format(len(authors)))

    if num_samples > len(authors):
        print(
            f'WARNING: num_samples ({num_samples}) is greater than the number of authors ({len(authors)}). Setting num_samples to {len(authors) // 2}'
        )
        num_samples = len(authors) // 2

    source_authors = sorted(random.sample(authors, num_samples))
    target_authors = sorted(
        random.sample(sorted(set(authors) - set(source_authors)), num_samples)
    )

    assert (
        len(source_authors) == len(target_authors)
        and len(set(source_authors) & set(target_authors)) == 0
    )

    random.shuffle(source_authors)
    random.shuffle(target_authors)

    pairs = list(zip(source_authors, target_authors))

    print('Selected {} target authors'.format(len(target_authors)))

    texts_by_author = get_texts_by_author(source_authors + target_authors, file_list)

    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, 'texts_by_author_for_finetune.json'), 'w') as out:
        json.dump(texts_by_author, out)

    pair_with_source = []
    for source_author, target_author in pairs:
        selected_source_text = random.choice(texts_by_author[source_author])
        pair_with_source.append((source_author, target_author, selected_source_text))

    transfer_file_dir = os.path.join(out_dir, 'transfers_for_finetune')
    os.makedirs(transfer_file_dir, exist_ok=True)
    with open(os.path.join(transfer_file_dir, 'train.jsonl'), 'w') as out:
        for source_author, target_author, selected_source_text in pair_with_source:
            out.write(
                json.dumps(
                    {
                        'author_id': source_author,
                        'target_author_id': target_author,
                        'text': selected_source_text,
                    }
                )
                + '\n'
            )


if __name__ == '__main__':
    main()
