import os
import pdb
from typing import Sequence
import bittensor as bt 
import hydra
from omegaconf import OmegaConf
import numpy as np
import torch
from datasets import load_dataset, interleave_datasets
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from accelerate import Accelerator
from transformers import (
    AutoTokenizer,
    get_scheduler,
    DataCollatorForLanguageModeling,
)
from mem_pytorch import TritonTransformer
from itertools import chain

from mem_pytorch.autoregressive_wrapper import (
    AutoregressiveWrapper,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_model(dim: int, depth: int, heads: int, seq_len: int, accelerate: bool) -> torch.nn.Module:
    model = TritonTransformer(
        num_tokens=50257,
        dim=dim,
        max_seq_len=seq_len,
        depth=depth,
        heads=heads,
        causal=True,
        use_triton=False,
        # q_bucket_size=1024,
        # k_bucket_size=2048,
        # ff_chunks=5,
        # use_flash_attn=True,
    )

    model = AutoregressiveWrapper(model)
    

    if torch.cuda.device_count() > 1 and not accelerate:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model, device_ids=[0,1,2,3,4,5,6,7])

    if not accelerate:
        model = model.to(device)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    print(f"Number of parameters: {params:,}")

    return model


def create_tokenizer(name: str = "gpt2"):
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True, mlm=False)
    tokenizer.pad_token = "[PAD]"
    # tokenizer = bt.tokenizer()

    return tokenizer


def create_streaming_dataset(set_names: Sequence[str], seq_len: int, accelerator: Accelerator):

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
            examples["text"], truncation=True, max_length=len(examples['text'])
        )

    data_train = train_dataset.map(
        encode,
        batched=True, 
        remove_columns=["text", "meta"]
    )
    data_val = val_dataset.map(encode, batched=True, remove_columns=["text", "meta"])

    # TODO: cfg
    seed, buffer_size = 12976371472801, 10_000
    data_train = data_train.shuffle(seed, buffer_size=buffer_size)
    data_val = data_val.shuffle(seed, buffer_size=buffer_size)

    return data_train, data_val, tokenizer


def create_regular_dataset(set_names: Sequence[str], seq_len: int, subset: str):
    
    train_sets = []
    val_sets = []  

    # for set_name in set_names:
    #     train_sets.append(load_dataset(set_name, split="train"))
    #     val_sets.append(load_dataset(set_name, split="validation"))
    # train_dataset = interleave_datasets(train_sets)
    # val_dataset = interleave_datasets(val_sets)

    train_dataset = load_dataset(set_names[0], subset, split="train")
    val_dataset = load_dataset(set_names[0], subset, split="validation")
    tokenizer = create_tokenizer()

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= seq_len:
            total_length = (
                total_length // seq_len
            ) * seq_len
        # Split by chunks of max_len.
        result = {
            k: [
                t[i : i + seq_len]
                for i in range(0, total_length, seq_len)
            ]
            for k, t in concatenated_examples.items()
        }
        # result["labels"] = result["input_ids"].copy()
        return result

    def encode(examples):
        return tokenizer(
            examples["text"], truncation=True, max_length=len(examples['text'])
        )

    data_train = train_dataset.map(
        encode, batched=True, remove_columns=["text"]
    )
    data_val = val_dataset.map(encode, batched=True, remove_columns=["text"])

    return data_train, data_val, tokenizer 


def train(
    model,
    train_dataloader,
    eval_dataloader,
    tokenizer,
    data_val,
    optim,
    lr_scheduler, 
    accelerator: Accelerator,
    hp: DictConfig,
    model_name: str,
    save_dir: str,
):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(f"{save_dir}/{model_name}"):
        os.makedirs(f"{save_dir}/{model_name}")

    scaler = torch.cuda.amp.GradScaler()

    for step in tqdm(range(hp.num_batches), mininterval=10.0, desc="training"):

        for i, batch in enumerate(tqdm(train_dataloader, total=300_000, mininterval=10., desc='training')):
            x = batch['input_ids'].to(device)
            # attention_mask = batch['attention_mask'].to(device)

            # with accelerator.accumulate(model):
            loss = model(x)
            std = 0
            # if accelerator is None:
            loss = loss.mean()
            std = loss.std()
            loss.backward()
            # else:
            #     accelerator.backward(loss)


            optim.step()
            lr_scheduler.step()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

            optim.zero_grad()
            # if the main gpu torch
            if torch.cuda.current_device() == 0:
                print(f"loss={loss:.4f} | std={std:.4f}")
            # if i != 0 and i % hp.validate_every == 0:
            #     # make sure we only do this on GPU:0
            #     model.eval()
            #     for _eval_step, eval_batch in enumerate(eval_dataloader):
            #         if _eval_step >= 1:
            #             break
            #         y = eval_batch["input_ids"].to(device)
            #         with torch.no_grad():
            #             loss = model(y)
            #             std = 0
            #             if torch.cuda.device_count() > 1:
            #                 # std = loss.std().item()
            #                 loss = loss.mean()

            #             print(f"val loss={loss.item():.4f} | {std=:.4f}")
                        # wandb.log({"val_loss": loss.item()})
        
            if i != 0 and i % hp.generate_every == 0:
                # if statement to  check if the device is cuda:0
                # if accelerator.is_main_process:
                    # model.eval()
                if torch.cuda.device_count() > 1:
                    pre_model = model.module
                else:
                    pre_model = model
                    print("generating...")
                    # generate
                pre_model.eval()
                ## There has to be a better way to do this?
                inp = [x for x in data_val.take(1)][0]["input_ids"]
                prime = tokenizer.decode(inp)
                print(f"\n\n {prime} \n\n {'-' * 80} \n")
                inp = torch.tensor(inp).to(device)


                sample = pre_model.generate(inp[None, ...], hp.generate_length)
                output_str = tokenizer.decode(sample[0])
                print(output_str)
                model.train()

            if i != 0 and i % hp.save_every == 0:
                if torch.cuda.current_device() == 0:
                    torch.save(model.module.state_dict(), f"{save_dir}/{model_name}_{i}.pt")
                    print(f"saved model to {model_name}_{i}.pt")

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    accelerator = Accelerator(split_batches=True)

    model = create_model(
        dim=cfg.model.dim,
        depth=cfg.model.depth,
        heads=cfg.model.heads,
        seq_len=cfg.model.sequence_length,
        accelerate=cfg.regime.accelerate,
    )

    if cfg.dataset.data_type == "streaming":
        data_train, data_val, tokenizer = create_streaming_dataset(
            set_names=cfg.dataset.constituent_sets, seq_len=cfg.model.sequence_length, accelerator=accelerator
        )
    else: 
        data_train, data_val, tokenizer = create_regular_dataset(
            cfg.dataset.constituent_sets, cfg.model.sequence_length, cfg.dataset.subset,
        )

    per_device_batch_size = cfg.regime.batch_size 
    
    train_dataloader = DataLoader(
        data_train,
        collate_fn=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        batch_size=per_device_batch_size,
    )
    eval_dataloader = DataLoader(
        data_val,
        collate_fn=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        batch_size=per_device_batch_size,
    )    
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=cfg.regime.learning_rate)
    
    num_warmup_steps = 1000
    max_train_steps = 0

    lr_scheduler = get_scheduler(
        name="cosine", # ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps * cfg.regime.gradient_accumulate_every, # num_warmup_steps * gradient_accumulation_steps
        num_training_steps=max_train_steps * cfg.regime.gradient_accumulate_every, # max_train_steps * gradient_accumulation_steps
    )
    if cfg.regime.accelerate:
        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader,  lr_scheduler)


    train(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        tokenizer=tokenizer,
        data_val=data_val,
        optim=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
        hp=cfg.regime,
        model_name=cfg.model.name,
        save_dir=cfg.save_dir,
    )


if __name__ == "__main__":
    main()
