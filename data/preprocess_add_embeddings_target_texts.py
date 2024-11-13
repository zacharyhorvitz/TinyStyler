''' preprocess and add embeddings from target_author_texts '''

import os

import click
from datasets import disable_caching

import torch

from datasets import load_from_disk

from tinystyler import text_to_style, load_style_model


def add_embeddings(*, example, style_tokenizer, style_model):
    '''add embeddings to example'''

    texts = example['target_author_texts']

    if style_model is not None:
        processed_embeddings = []
        for text_for_author in texts:
            style_embeds = text_to_style(
                model=style_model,
                tokenizer=style_tokenizer,
                texts=text_for_author,
                device='cuda',
                model_type='style',
                max_length=512,
            )
            style_embeds = torch.stack(style_embeds).mean(dim=0).detach().cpu().numpy()
            processed_embeddings.append(style_embeds)

    example['style_embedding'] = [x for x in processed_embeddings]

    return example


@click.command()
@click.option('--dataset_path', help='path_to_dataset', required=True)
@click.option('--style_model_name', help='style model name', required=True)
def main(dataset_path, style_model_name):
    disable_caching()

    output_path = (
        os.path.normpath(dataset_path)
        + '_with_style_embeds_'
        + style_model_name.replace('/', '_')
    )

    print('output path:', output_path)
    # load dataset
    dataset = load_from_disk(dataset_path)

    # load style model
    style_model, style_tokenizer, _ = load_style_model(model_name=style_model_name)
    style_model.to('cuda')

    with_embeddings = dataset.map(
        lambda x: add_embeddings(
            example=x, style_tokenizer=style_tokenizer, style_model=style_model
        ),
        batched=True,
        batch_size=32,
    )

    with_embeddings.save_to_disk(output_path)


if __name__ == '__main__':
    main()
