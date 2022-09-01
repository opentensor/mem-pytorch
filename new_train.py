import os
import pickle
import random
from itertools import chain
from typing import Sequence

import bittensor
import hydra
from omegaconf import OmegaConf
import numpy as np
import torch
from datasets import Dataset, load_dataset, interleave_datasets
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    default_data_collator,
)

from memory_efficient_attention_pytorch.autoregressive_wrapper import (
    AutoregressiveWrapper,
)
from memory_efficient_attention_pytorch.transformer import Transformer

NUM_BATCHES = 10
BATCH_SIZE = 64
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 2e-4
VALIDATE_EVERY = 5
GENERATE_EVERY = 10
GENERATE_LENGTH = 256
SEQ_LEN = 64
CONCATENATE_RAW = False
OVERWRITE_CACHE = False
SAVE_EVERY = 50
DATASET_NAME = "the_pile"
TOKENIZER_NAME = "gpt2"
USE_HF_DATA = True
STREAM = True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_model(dim: int, depth: int, heads: int) -> torch.nn.Module:
    model = Transformer(
        num_tokens=50257,
        dim=dim,
        max_seq_len=SEQ_LEN,
        depth=depth,
        heads=heads,
        causal=True,
        q_bucket_size=1024,
        k_bucket_size=2048,
        ff_chunks=5,
        use_flash_attn=True,
    )

    model = AutoregressiveWrapper(model)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

    model.to(device)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    print(f"Number of parameters: {params:,}")

    return model


def preprocess(tokenizer, raw_datasets):

    # First we tokenize all the texts.
    column_names = raw_datasets.column_names
    text_column_name = "text" if "text" in column_names else column_names["train"][0]
    if CONCATENATE_RAW is True:
        pad = False
    else:
        pad = "max_length"

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= SEQ_LEN:
            total_length = (total_length // SEQ_LEN) * SEQ_LEN
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + SEQ_LEN] for i in range(0, total_length, SEQ_LEN)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    def tokenize_fn(examples):
        result = tokenizer(
            examples[text_column_name],
            padding=pad,
            truncation=True,
            max_length=SEQ_LEN,
        )
        result["labels"] = result["input_ids"].copy()
        return result

    tokenized_datasets = raw_datasets.map(
        tokenize_fn,
        batched=True,
        num_proc=8,
        load_from_cache_file=not OVERWRITE_CACHE,
        desc="Running tokenizer on dataset",
    )

    if CONCATENATE_RAW is True:
        tokenized_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=8,
            load_from_cache_file=not OVERWRITE_CACHE,
            desc=f"Grouping texts in chunks of {SEQ_LEN}",
        )

    return tokenized_datasets


def create_tokenized_datasets(tokenized_datasets):
    if "train" not in tokenized_datasets.column_names:
        tokenized_datasets = tokenized_datasets.train_test_split(test_size=5 / 100)
        tokenized_datasets_test_valid = tokenized_datasets["test"].train_test_split(
            test_size=0.5
        )
        tokenized_datasets["test"] = tokenized_datasets_test_valid["train"]
        tokenized_datasets["validation"] = tokenized_datasets_test_valid["test"]

    data_train = tokenized_datasets["train"]
    data_val = tokenized_datasets["validation"]

    train_dataloader = DataLoader(
        data_train,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=BATCH_SIZE,
    )
    eval_dataloader = DataLoader(
        data_val,
        collate_fn=default_data_collator,
        batch_size=BATCH_SIZE,
    )

    return train_dataloader, eval_dataloader, data_train, data_val


def create_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True, mlm=False)
    tokenizer.pad_token = "[PAD]"

    return tokenizer


def create_hf_dataset(set_names: Sequence[str]):
    tokenizer = create_tokenizer()

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

    def encode(examples):
        if CONCATENATE_RAW is True:
            pad = False
        else:
            pad = "max_length"

        return tokenizer(
            examples["text"], padding=pad, truncation=True, max_length=SEQ_LEN
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


def create_dataset():
    file_name = f"./bt_dataset_cached_{SEQ_LEN}_{BATCH_SIZE}.pkl"

    if not os.path.exists(file_name):
        dataset = bittensor.dataset(
            no_tokenizer=True, batch_size=BATCH_SIZE, block_size=SEQ_LEN, num_workers=32
        )

        dataloader = dataset.dataloader(NUM_BATCHES)
        bittensor_dataset = {"text": []}
        for batch in tqdm(dataloader, desc="Loading data from bittensor IPFS"):
            bittensor_dataset["text"].extend(batch)
        raw_datasets = Dataset.from_dict(bittensor_dataset)

        dataset.close()  # Avoid leaving threadqueue running.

        with open(file_name, "wb") as fh:
            pickle.dump(raw_datasets, fh)
    else:
        with open(file_name, "rb") as fh:
            raw_datasets = pickle.load(fh)

    tokenizer = create_tokenizer()

    tokenized_datasets = preprocess(tokenizer, raw_datasets)

    train_dataloader, eval_dataloader, data_train, data_val = create_tokenized_datasets(
        tokenized_datasets
    )

    return train_dataloader, eval_dataloader, data_train, data_val, tokenizer


def stream_train(model, train_dataloader, eval_dataloader, tokenizer, data_val, hp: DictConfig, model_name: str, save_dir: str):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(f"{save_dir}/{model_name}"):
        os.makedirs(f"{save_dir}/{model_name}")
    optim = torch.optim.Adam(model.parameters(), lr=hp.learning_rate)

    for step in tqdm(range(hp.num_batches), mininterval=10.0, desc="training"):

        for i, batch in enumerate(tqdm(train_dataloader, total=5)):
            if i == 5:
                break

            x = batch["input_ids"].to(device)

            loss = model(x)
            std = 0
            if torch.cuda.device_count() > 1:
                loss = loss.mean()
                std = loss.std().item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optim.step()
            optim.zero_grad()
            print(f"loss={loss.item():.4f} | {std=:.4f}")

        if step % hp.validate_every == 0:
            model.eval()
            for _eval_step, eval_batch in enumerate(eval_dataloader):
                if _eval_step >= 1:
                    break
                y = eval_batch["input_ids"].to(device)
                with torch.no_grad():
                    loss = model(y)
                    std = 0
                    if torch.cuda.device_count() > 1:
                        std = loss.std().item()
                        loss = loss.mean()

                    print(f"val loss={loss.item():.4f} | {std=:.4f}")

        if step % hp.generate_every == 0:
            model.eval()
            ## There has to be a better way to do this?
            inp = [x for x in data_val.take(1)][0]['input_ids']
            prime = tokenizer.decode(inp)
            print(f"%s \n\n %s", (prime, "*" * 100))

            inp = torch.tensor(inp)

            inp = inp.reshape(1, -1)
            inp = inp.to(device)

            sample = model.generate(inp, hp.generate_length)
            output_str = tokenizer.decode(sample[0])
            print(output_str)

        if step != 0 and step % hp.save_every== 0:
            torch.save(model.state_dict(), f"{save_dir}/{model_name}_{step}.pt")
            print(f"saved model to {model_name}_{step}.pt")


def train(model, train_dataloader, eval_dataloader, data_val, tokenizer):
    # optimizer

    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # training

    for i in tqdm(range(NUM_BATCHES), mininterval=10.0, desc="training"):
        for step, batch in tqdm(
            enumerate(train_dataloader), mininterval=10.0, desc="training"
        ):
            model.train()

            x = batch["input_ids"].to(device)

            for _ in range(GRADIENT_ACCUMULATE_EVERY):
                loss = model(x)
                std = 0
                if torch.cuda.device_count() > 1:
                    std = loss.std().item()
                    loss = loss.mean()
                loss.backward()

            print(f"training loss: {loss.item()}")
            print(f"training loss std: {std}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optim.step()
            optim.zero_grad()

            if step % VALIDATE_EVERY == 0:
                model.eval()
                for _eval_step, eval_batch in enumerate(eval_dataloader):
                    if _eval_step >= 1:
                        break
                    y = eval_batch["input_ids"].to(device)
                    with torch.no_grad():
                        loss = model(y)
                        std = 0
                        if torch.cuda.device_count() > 1:
                            std = loss.std().item()
                            loss = loss.mean()

                        print(f"validation loss: {loss.item()}")
                        print(f"validation loss std: {std}")

            if step != 0 and step % GENERATE_EVERY == 0:
                model.eval()
                inp = random.choice(data_val["input_ids"])[:-1]
                # prime = decode_tokens(inp)
                prime = tokenizer.decode(inp)
                print(f"%s \n\n %s", (prime, "*" * 100))

                inp = torch.tensor(inp)

                inp = inp.reshape(1, -1)
                inp = inp.to(device)

                sample = model.generate(inp, GENERATE_LENGTH)
                output_str = tokenizer.decode(sample[0])
                print(output_str)

            if step != 0 and step % SAVE_EVERY == 0:
                torch.save(model.state_dict(), f"{SAVE_DIR}/{MODEL_NAME}_{step}.pt")
                print(f"saved model to {MODEL_NAME}_{step}.pt")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    model = create_model(
        dim=cfg.model.dim, depth=cfg.model.depth, heads=cfg.model.heads
    )

    if cfg.dataset.name == "bittensor":
        (
            train_dataloader,
            eval_dataloader,
            data_train,
            data_val,
            tokenizer,
        ) = create_dataset()
        train(model, train_dataloader, eval_dataloader, data_val, tokenizer)
    else:
        data_train, data_val, tokenizer = create_hf_dataset(
            set_names=cfg.dataset.constituent_sets
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
        stream_train(
            model=model,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            tokenizer=tokenizer,
            data_val=data_val,
            hp=cfg.regime,
            model_name=cfg.model.name,
            save_dir=cfg.save_dir,
        )

if __name__ == "__main__":
    main()
