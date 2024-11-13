''' train tinystyler model, for reconstruction training (1) or supervised fine-tuning (3) '''
from transformers import AutoTokenizer, get_scheduler

import os
import random
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import json


from datasets import load_from_disk
import wandb
import click

from accelerate import Accelerator

from tinystyler import TinyStyler, MODEL_TO_MODEL_TYPE


def data_collator(
    *,
    batch,
    tokenizer,
    max_length_src,
    max_length_tgt,
    ctr_embed_key,
    ignore_idx=-100,
    input_key='paraphrase',
    output_key='text',
    do_lower=False,
):
    '''


    Collate function for the TinyStyler model.

    Args:
        batch (list): list of examples
        tokenizer (transformers.AutoTokenizer): tokenizer
        max_length_src (int): max length of the source
        max_length_tgt (int): max length of the target
        ctr_embed_key (str): key for style embedding
        ignore_idx (int): ignore index
        input_key (str): key for input
        output_key (str): key for output
        do_lower (bool): whether to lowercase the input

    Returns:
        dict: encoded batch of data
    '''

    input_texts = [x[input_key] for x in batch]

    # handle nested input_texts
    if isinstance(input_texts[0], list):
        assert all([len(x) == 1 for x in input_texts])
        input_texts = [x[0] for x in input_texts]

    if do_lower:
        input_texts = [x.lower() for x in input_texts]

    inputs = tokenizer(
        input_texts,
        max_length=max_length_src,
        padding=True,
        truncation=True,
        return_tensors='pt',
    )
    labels = tokenizer(
        [x[output_key] for x in batch],
        max_length=max_length_tgt,
        padding=True,
        truncation=True,
        return_tensors='pt',
    )['input_ids']
    labels[labels == tokenizer.pad_token_id] = ignore_idx

    retval = inputs
    retval['labels'] = labels

    style_embeddings = torch.stack(
        [torch.tensor(x[ctr_embed_key]) for x in batch], dim=0
    )
    retval[ctr_embed_key] = style_embeddings

    return retval


def run_model(*, batch, model, device, style_embed):
    '''

    Compute the logits and loss for a batch of data.

    Args:
        batch (dict): encoded batch of data
        model (torch.nn.Module): model to run
        device (str): device to run on
        style_embed (str): key for style embedding in batch

    Returns:
        logits (torch.Tensor): model logits
        loss (torch.Tensor): model loss

    '''

    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    style = batch[style_embed].to(device)
    result = model(
        input_ids=input_ids, attention_mask=attention_mask, labels=labels, style=style
    )
    loss = result.loss
    logits = result.logits
    return logits, loss


@click.command()
@click.option('--learning_rate', type=float, default=1e-5)
@click.option('--batch_size', type=int, default=64)
@click.option('--accumulation_steps', type=int, default=1)
@click.option('--out_dir', type=str, required=True)
@click.option('--device', type=str, default='cuda')
@click.option('--warmup_steps', type=int, default=2000)
@click.option('--max_steps', type=int, default=10000000)
@click.option('--max_num_epochs', type=int, default=10000)
@click.option('--eval_freq', type=int, default=1000)
@click.option('--ctrl_embed_dim', type=int, default=768)
@click.option('--style_embed', type=str, default='style_embedding')
@click.option('--model_name', type=str, default='t5-large')
@click.option('--data_file_path', type=str, required=True)
@click.option('--checkpoint', type=str, default=None)
@click.option('--seed', type=int, default=42)
@click.option('--max_encoder_len', type=int, default=80)
@click.option('--max_decoder_len', type=int, default=80)
@click.option('--max_val_batch', type=int, default=200)
@click.option('--input_key', type=str, default='paraphrase')
@click.option('--output_key', type=str, default='text')
@click.option('--skip_load_optimizer', is_flag=True)
@click.option('--do_lower', is_flag=True)
def main(
    learning_rate,
    batch_size,
    accumulation_steps,
    out_dir,
    device,
    warmup_steps,
    max_steps,
    max_num_epochs,
    eval_freq,
    ctrl_embed_dim,
    style_embed,
    model_name,
    data_file_path,
    checkpoint,
    seed,
    max_encoder_len,
    max_decoder_len,
    max_val_batch,
    input_key,
    output_key,
    skip_load_optimizer,
    do_lower,
):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    accelerator = Accelerator()

    assert model_name in MODEL_TO_MODEL_TYPE

    # device = 'cuda'
    device = accelerator.device
    print(device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="right",
        use_fast=True,  # Fast tokenizer giving issues.
        trust_remote_code=True,
    )

    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TinyStyler(
        base_model=model_name,
        use_style=True,
        ctrl_embed_dim=ctrl_embed_dim,
    )
    model.to(device)

    # load from checkpoint
    if checkpoint is not None:
        checkpoint_dir = os.path.dirname(checkpoint)

        if skip_load_optimizer:
            optimizer_path = None
            scheduler_path = None

        else:
            optimizer_path = os.path.join(checkpoint_dir, 'optimizer.pt')
            scheduler_path = os.path.join(checkpoint_dir, 'scheduler.pt')

            assert os.path.exists(optimizer_path)
            assert os.path.exists(scheduler_path)

        current_state = model.state_dict()
        saved_state_dict = torch.load(checkpoint, map_location=device)
        current_state.update(saved_state_dict)
        model.load_state_dict(current_state)

    else:
        checkpoint_dir = None
        optimizer_path = None
        scheduler_path = None

    tokenized_datasets = load_from_disk(data_file_path)

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["val"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        print(f"Sample {index} of the training set: {train_dataset[index]}.")

    if MODEL_TO_MODEL_TYPE[model_name] == 't5':
        collator_fn = data_collator
    else:
        raise NotImplementedError

    collator_args = {
        'tokenizer': tokenizer,
        'max_length_src': max_encoder_len,
        'max_length_tgt': max_decoder_len,
        'ctr_embed_key': style_embed,
        'ignore_idx': -100,
        'input_key': input_key,
        'output_key': output_key,
        'do_lower': do_lower,
    }

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=lambda x: collator_fn(batch=x, **collator_args),
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=lambda x: collator_fn(batch=x, **collator_args),
    )

    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    if optimizer_path is not None:
        optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))

    scheduler = get_scheduler(
        name='constant_with_warmup',
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )

    if scheduler_path is not None:
        scheduler.load_state_dict(torch.load(scheduler_path, map_location=device))

    (
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        scheduler,
    ) = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, scheduler
    )

    # add date
    if accelerator.is_main_process:
        cur_date = datetime.now().strftime("%Y-%m-%d-%H.%M.%S")
        out_dir = os.path.join(out_dir, cur_date)
        os.makedirs(out_dir, exist_ok=True)

    wandb.init(
        project='tinystyler',
        config={
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'accumulation_steps': accumulation_steps,
            'outdir': out_dir,
            'device': device,
            'warmup_steps': warmup_steps,
            'max_steps': max_steps,
            'max_num_epochs': max_num_epochs,
            'eval_feq': eval_freq,
            'ctrl_embed_dim': ctrl_embed_dim,
            'style_embed': style_embed,
            'model': model_name,
            'data_file_path': data_file_path,
            'checkpoint': checkpoint,
            'seed': seed,
            'max_encoder_len': max_encoder_len,
            'max_decoder_len': max_decoder_len,
            'max_val_batch': max_val_batch,
            'input_key': input_key,
            'output_key': output_key,
            'skip_load_optimizer': skip_load_optimizer,
        },
    )

    fname = f'best_model_{model_name.replace("/","_")}_{learning_rate}_{batch_size*accumulation_steps}.pt'

    best_val_loss = None
    optimizer.zero_grad()
    steps = 0
    counter = 0
    for epoch in range(max_num_epochs):
        model.train()
        wandb.log({"epoch": epoch})

        with tqdm(total=len(train_dataloader)) as pbar:
            for i, data in enumerate(train_dataloader):
                _, loss = run_model(
                    batch=data, model=model, device=device, style_embed=style_embed
                )
                if counter % 100 == 0:
                    print('Epoch: ', epoch, ', Train Loss: ', loss.item())

                wandb.log({"train_loss": loss.item()})
                loss = loss / accumulation_steps

                accelerator.backward(loss)
                pbar.update(1)

                if (counter + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    steps += 1

                if (counter + 1) % (eval_freq * accumulation_steps) == 0:
                    model.eval()
                    losses = []
                    with torch.no_grad():
                        for j, val_data in enumerate(eval_dataloader):
                            if j > max_val_batch:
                                break
                            _, loss = run_model(
                                batch=val_data,
                                model=model,
                                device=device,
                                style_embed=style_embed,
                            )
                            losses.append(loss.item())

                    val_loss = sum(losses) / len(losses)
                    wandb.log({"val_loss": val_loss})
                    print('Epoch: ', epoch, ', Val Loss: ', val_loss)

                    if accelerator.is_main_process:
                        if best_val_loss is None or val_loss < best_val_loss:
                            best_val_loss = val_loss
                            print(epoch, i, 'New best val loss: ', best_val_loss)
                            with open(
                                os.path.join(out_dir, 'checkpoint_info.json'), 'w+'
                            ) as out_:
                                json.dump(
                                    {
                                        'epoch': epoch,
                                        'i': i,
                                        'counter': counter,
                                        'steps': steps,
                                        'loss': best_val_loss,
                                    },
                                    out_,
                                )
                            torch.save(model.state_dict(), os.path.join(out_dir, fname))
                            # save optimizer state, save scheduler state
                            torch.save(
                                optimizer.state_dict(),
                                os.path.join(out_dir, 'optimizer.pt'),
                            )
                            torch.save(
                                scheduler.state_dict(),
                                os.path.join(out_dir, 'scheduler.pt'),
                            )

                    model.train()

                counter += 1

                if steps >= max_steps:
                    break

    wandb.finish()


if __name__ == '__main__':
    main()
