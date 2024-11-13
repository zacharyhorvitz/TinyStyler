''' split text file into train val and test '''
import click
import random
import os


def write_lines(lines, out_path):
    with open(out_path, 'w') as f:
        for line in lines:
            f.write(line)


@click.command()
@click.option('--dir', type=str, required=True, help='path to jsonl files')
@click.option(
    '--max_per_file', type=int, default=100000, help='max number of authors per file'
)
def main(dir, max_per_file):
    random.seed(42)

    to_split = ['train', 'val', 'test']

    out_dir = os.path.join(dir, 'split_files')

    os.makedirs(out_dir, exist_ok=True)

    for split in to_split:
        file_path = os.path.join(dir, f'{split}.jsonl')
        if not os.path.exists(file_path):
            continue
        print('processing', file_path)
        buffer = []
        counter = 0
        with open(file_path, 'r') as f:
            for l in f:
                buffer.append(l)
                counter += 1

                if len(buffer) == max_per_file:
                    out_path = os.path.join(out_dir, f'{split}_{counter}.jsonl')
                    write_lines(buffer, out_path)
                    buffer = []

        if buffer:
            out_path = os.path.join(out_dir, f'{split}_{counter}.jsonl')
            write_lines(buffer, out_path)


if __name__ == '__main__':
    main()
