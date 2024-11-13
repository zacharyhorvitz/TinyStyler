''' Add style embeddings to dataset '''

import os

import click
from datasets import disable_caching


from datasets import load_from_disk

from tinystyler import text_to_style, load_style_model


def add_embeddings(*, example, style_tokenizer, style_model):
    '''add embeddings to example'''

    texts = example['text']

    style_embedding = text_to_style(
        model=style_model,
        tokenizer=style_tokenizer,
        texts=texts,
        device='cuda',
        model_type='style',
        max_length=512,
    )
    example['style_embedding'] = [x.detach().cpu().numpy() for x in style_embedding]

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
