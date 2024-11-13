''' split text file into train val and test '''

import json
import click
import random
import os


@click.command()
@click.option('--path', type=str, required=True, help='path to jsonl file')
@click.option(
    '--protected_authors_path',
    type=str,
    required=True,
    help='path to protected authors file',
)
@click.option('--train_split', type=float, default=0.90, help='train split')
@click.option('--val_split', type=float, default=0.05, help='val split')
@click.option('--test_split', type=float, default=0.05, help='test split')
@click.option('--out_dir', type=str, required=True, help='path to text file')
def main(path, protected_authors_path, train_split, val_split, test_split, out_dir):
    random.seed(42)

    assert train_split + val_split + test_split == 1.0

    with open(protected_authors_path, 'r') as f:
        protected_authors_list = sorted([line.strip() for line in f])

    all_authors = set()

    counter = 0
    with open(path, 'r') as f:
        for l in f:
            data = json.loads(l)
            author = data['author_id']
            all_authors.add(author)
            counter += 1

            if counter % 100000 == 0:
                print(counter)

    un_protected_authors = all_authors - set(protected_authors_list)

    protected_authors = set(protected_authors_list) & all_authors

    protected_authors = sorted(protected_authors)
    un_protected_authors = sorted(un_protected_authors)

    random.shuffle(un_protected_authors)
    random.shuffle(protected_authors)

    combined_authors = protected_authors + un_protected_authors

    num_test_authors = int(len(combined_authors) * test_split)

    num_val_authors = int(len(combined_authors) * val_split)

    test_authors = combined_authors[:num_test_authors]
    val_authors = combined_authors[
        num_test_authors : num_test_authors + num_val_authors
    ]
    train_authors = combined_authors[num_test_authors + num_val_authors :]

    print(f'num_test_authors: {len(test_authors)}')
    print(f'num_val_authors: {len(val_authors)}')
    print(f'num_train_authors: {len(train_authors)}')

    # double check zero overlap
    assert len(set(train_authors) & set(val_authors)) == 0
    assert len(set(train_authors) & set(test_authors)) == 0
    assert len(set(val_authors) & set(test_authors)) == 0

    # check all protected authors are in test
    assert len(set(protected_authors) - set(test_authors)) == 0

    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, 'author_splits.json'), 'w') as f:
        json.dump({'train': train_authors, 'val': val_authors, 'test': test_authors}, f)


if __name__ == '__main__':
    main()
