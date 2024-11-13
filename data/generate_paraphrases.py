from tqdm import tqdm
import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import click
import json
import os
import math


def get_file_len(fname):
    with open(fname, "r") as f:
        return len(f.readlines())


def get_response(
    *,
    model,
    tokenizer,
    input_text,
    num_return_sequences=1,
    top_p=0.85,
    temp=1.5,
    max_length_input=60,
    max_length_output=60,
    torch_device='cuda',
):
    batch = tokenizer(
        input_text,
        truncation=True,
        padding='longest',
        max_length=max_length_input,
        return_tensors="pt",
    ).to(torch_device)

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
    for i in range(len(input_text)):
        reshaped.append(
            tgt_text[i * num_return_sequences : (i + 1) * num_return_sequences]
        )

    return reshaped


# example usage
# python generate_paraphrases.py --in_dir data/reddit_emnlp/ --out_dir data/reddit_emnlp/ --temp 1.5 --top_p 0.8 --idx 0 --num_workers 1 --batch_size 64 --max_input_length 60 --max_output_length 60


@click.command()
@click.option('--in_dir', type=str, required=True, help='path to files')
@click.option('--out_dir', type=str, required=True, help='path to files')
@click.option('--temp', type=float, default=1.5, help='temperature')
@click.option('--top_p', type=float, default=0.8, help='top p')
@click.option('--idx', type=int, default=0, help='idx')
@click.option('--num_workers', type=int, default=1, help='total workers')
@click.option('--batch_size', type=int, default=64, help='batch size')
@click.option('--max_input_length', type=int, default=60, help='max input length')
@click.option('--max_output_length', type=int, default=60, help='max output length')
@click.option(
    '--num_return_sequences', type=int, default=1, help='num return sequences'
)
def main(
    in_dir,
    out_dir,
    temp,
    top_p,
    idx,
    num_workers,
    batch_size,
    max_input_length,
    max_output_length,
    num_return_sequences,
):
    model_name = 'tuner007/pegasus_paraphrase'
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to('cuda')

    def process_batch(*, samples, handler, pbar=None):
        texts = [sample['text'] for sample in samples]
        paraphrases = get_response(
            model=model,
            tokenizer=tokenizer,
            input_text=texts,
            top_p=top_p,
            temp=temp,
            max_length_input=max_input_length,
            max_length_output=max_output_length,
            torch_device='cuda',
            num_return_sequences=num_return_sequences,
        )
        for sample, paraphrase in zip(samples, paraphrases):
            sample['paraphrase'] = paraphrase
            handler.write(json.dumps(sample) + '\n')

        if pbar:
            pbar.update(len(samples))

    all_files = sorted(os.listdir(in_dir))

    out_dir = os.path.join(out_dir, f'topp{top_p}_tmp{temp}')
    os.makedirs(out_dir, exist_ok=True)

    per_process = math.ceil(len(all_files) / num_workers)
    start = idx * per_process
    end = min((idx + 1) * per_process, len(all_files))

    for file in tqdm(all_files[start:end]):
        path = os.path.join(in_dir, file)
        num_texts = get_file_len(path)
        skipped = 0
        with tqdm(total=num_texts) as pbar:
            with open(path, 'r') as f_in:
                with open(
                    os.path.join(
                        out_dir, file.replace('.jsonl', f'_paraphrased.jsonl')
                    ),
                    'w+',
                ) as f_out:
                    samples = []
                    for line in f_in:
                        sample = json.loads(line)
                        if len(tokenizer.tokenize(sample['text'])) > max_input_length:
                            pbar.update(1)
                            skipped += 1
                            continue

                        samples.append(sample)

                        if len(samples) == batch_size:
                            process_batch(samples=samples, handler=f_out, pbar=pbar)
                            samples = []

                    if samples:
                        process_batch(samples=samples, handler=f_out, pbar=pbar)
                        samples = []
        print(f'Skipped {skipped} ({skipped/num_texts}) samples in {file}')


if __name__ == '__main__':
    main()
