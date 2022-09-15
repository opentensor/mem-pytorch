import os
from typing import Sequence

import hydra
from omegaconf import OmegaConf
import numpy as np
import torch
from datasets import load_dataset, interleave_datasets
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)
import argparse

from mem_pytorch.transformer_x import CosineSimCausalTransformer
import wandb

from mem_pytorch.autoregressive_wrapper import (
    AutoregressiveWrapper,
)
from mem_pytorch.transformer import Transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_model(
    dim: int, 
    depth: int, 
    heads: int, 
    seq_len: int, 
    use_cuda_kernel: bool,
    fp16: bool,
    ) -> torch.nn.Module:
    # model = Transformer(
    #     num_tokens=50257,
    #     dim=dim,
    #     max_seq_len=seq_len,
    #     depth=depth,
    #     heads=heads,
    #     causal=True,
    #     q_bucket_size=1024,
    #     k_bucket_size=2048,
    #     ff_chunks=5,
    #     use_flash_attn=True,
    # )

    model = CosineSimCausalTransformer(
        num_tokens=50257,
        dim=dim,
        max_seq_len=seq_len,
        depth=depth,
        heads=heads,
        causal=True,
        use_cuda_kernel = use_cuda_kernel,

    )

    # model = AutoregressiveWrapper(model)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

    if fp16:
        model = model.half()

    model.to(device)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    print(f"Number of parameters: {params:,}")

    return model


def create_tokenizer(name: str = "gpt2"):
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True, mlm=False)
    tokenizer.pad_token = "[PAD]"

    return tokenizer


def create_streaming_dataset(set_names: Sequence[str], seq_len: int):

    train_sets = []
    val_sets = []
    # TODO: More robust config handling for datasets w/ other kwargs
    for set_name in set_names:
        train_sets.append(load_dataset(set_name, split="train", streaming=True))
        val_sets.append(load_dataset(set_name, split="validation", streaming=True))
    train_dataset = interleave_datasets(train_sets)
    val_dataset = interleave_datasets(val_sets)

    train_dataset = train_dataset.with_format("torch")
    val_dataset = val_dataset.with_format("torch")

    tokenizer = create_tokenizer()

    def encode(examples):
        return tokenizer(
            examples["text"], padding="max_length", truncation=True, max_length=seq_len
        )

    data_train = train_dataset.map(
        encode, batched=True, remove_columns=["text", "meta"]
    )
    data_val = val_dataset.map(encode, batched=True, remove_columns=["text", "meta"])

    # TODO: cfg
    seed, buffer_size = 42, 10_000
    data_train = data_train.shuffle(seed, buffer_size=buffer_size)
    data_val = data_val.shuffle(seed, buffer_size=buffer_size)

    return data_train, data_val, tokenizer


def train(
    model,
    train_dataloader,
    eval_dataloader,
    tokenizer,
    data_val,
    fp16: bool,
    hp: DictConfig,
    model_name: str,
    save_dir: str,
):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(f"{save_dir}/{model_name}"):
        os.makedirs(f"{save_dir}/{model_name}")
    optim = torch.optim.Adam(model.parameters(), lr=hp.learning_rate)

    for step in tqdm(range(hp.num_batches), mininterval=10.0, desc="training"):

        for i, batch in enumerate(tqdm(train_dataloader, total=10_000, mininterval=10., desc='training')):
            # if fp16:
            #     # batch = {k: v.half() for k, v in batch.items()}
            #     batch['input_ids'] = batch['input_ids'].half()

            # batch = {k: v.to(device) for k, v in batch.items()}

            x = batch['input_ids'].to(device)
            with torch.cuda.amp.autocast(amp=fp16):
                loss = model(x, return_loss=True)
                std = 0
                if torch.cuda.device_count() > 1:
                    loss = loss.mean()
                    std = loss.std().item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optim.step()
            optim.zero_grad()
            print(f"loss={loss.item():.4f} | {std=:.4f}")

            if i != 0 and step % hp.eval_every == 0:
                wandb.log({"loss": loss.item()})

            if i != 0 and i % hp.validate_every == 0:
                model.eval()
                for _eval_step, eval_batch in enumerate(eval_dataloader):
                    if _eval_step >= 1:
                        break
                    y = eval_batch["input_ids"].to(device)
                    with torch.no_grad():
                        loss = model(y, return_loss=True)
                        std = 0
                        if torch.cuda.device_count() > 1:
                            std = loss.std().item()
                            loss = loss.mean()

                        print(f"val loss={loss.item():.4f} | {std=:.4f}")
                        wandb.log({"val_loss": loss.item()})

            if i != 0 and i % hp.generate_every == 0:
                # if statement to  check if the device is cuda:0
                if torch.cuda.current_device() == 0:

                    if torch.cuda.device_count() > 1:
                        gen_model = model.module
                    else:
                        gen_model = model

                    
                    gen_model.eval()
                    ## There has to be a better way to do this?
                    inp = [x for x in data_val.take(1)][0]["input_ids"]
                    prime = tokenizer.decode(inp)
                    print(f"\n\n {prime} \n\n {'-' * 80} \n")
                    inp = torch.tensor(inp).to(device)


                    sample = gen_model.generate(inp[None, ...], hp.generate_length)
                    output_str = tokenizer.decode(sample[0])
                    print(output_str)

            if i != 0 and i % hp.save_every == 0:
                if torch.cuda.current_device() == 0:
                    torch.save(model.state_dict(), f"{save_dir}/{model_name}_{i}.pt")
                    print(f"saved model to {model_name}_{i}.pt")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    wandb.init(project="mem-pytorch", entity="robertmyers")

    wandb.config = {
        "learning_rate": cfg.regime.learning_rate,
        "epochs": cfg.regime.num_batches,
        "batch_size": cfg.regime.batch_size,
        "heads": cfg.model.heads,
        "depth": cfg.model.depth,
        "dim": cfg.model.dim,
        "sequence_length": cfg.model.sequence_length,
    }

    model = create_model(
        dim=cfg.model.dim,
        depth=cfg.model.depth,
        heads=cfg.model.heads,
        seq_len=cfg.model.sequence_length,
        fp16=cfg.model.fp16,
        use_cuda_kernel=cfg.model.use_cuda_kernel,
    )

    wandb.watch(model)


    data_train, data_val, tokenizer = create_streaming_dataset(
        set_names=cfg.dataset.constituent_sets, seq_len=cfg.model.sequence_length
    )
    train_dataloader = DataLoader(
        data_train,
        collate_fn=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        batch_size=cfg.regime.batch_size,
    )
    eval_dataloader = DataLoader(
        data_val,
        collate_fn=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        batch_size=cfg.regime.batch_size,
    )
    train(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        tokenizer=tokenizer,
        data_val=data_val,
        hp=cfg.regime,
        fp16=cfg.model.fp16,
        model_name=cfg.model.name,
        save_dir=cfg.save_dir,
    )


if __name__ == "__main__":
    main()
