''' subsample data for each author based on author splits '''

import json
import click
import random
import os

# example usage: python subsample_data.py --path data/reddit_emnlp/styll_data/author_text.jsonl --author_splits_path data/reddit_emnlp/styll_data/author_splits.json --per_author 10


@click.command()
@click.option('--path', type=str, required=True, help='path to jsonl file')
@click.option(
    '--author_splits_path', type=str, required=True, help='path to author splits'
)
@click.option(
    '--per_author', type=int, default=10, help='max number of comments per author'
)
def main(path, author_splits_path, per_author):
    random.seed(42)

    with open(author_splits_path, 'r') as f:
        author_splits = json.load(f)

    author_to_shard = {
        author: shard for shard, authors in author_splits.items() for author in authors
    }

    # import pdb; pdb.set_trace()

    out_dir = os.path.dirname(author_splits_path)

    shard_to_handle = {
        shard: open(os.path.join(out_dir, f'{shard}.jsonl'), 'w')
        for shard in author_splits
    }

    counter = 0
    with open(path, 'r') as f:
        for l in f:
            data = json.loads(l)
            author = data['author_id']
            texts = data['syms']

            shard = author_to_shard[author]

            # sample per_author comments
            sampled_texts = random.sample(texts, min(per_author, len(texts)))

            for text in sampled_texts:
                shard_to_handle[shard].write(
                    json.dumps({'author_id': author, 'text': text}) + '\n'
                )

            counter += 1

            if counter % 100000 == 0:
                print(counter)


if __name__ == '__main__':
    main()
