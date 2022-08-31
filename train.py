from memory_efficient_attention_pytorch.transformer import Transformer
from memory_efficient_attention_pytorch.autoregressive_wrapper import AutoregressiveWrapper

import pdb

import pickle
import os
import random
import gzip
import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset as PytorchDataset
from itertools import chain
import bittensor
from tqdm.auto import tqdm


from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    get_scheduler,
)
import datasets
from datasets import Dataset, DatasetDict, load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# constants

NUM_BATCHES = 100_000
BATCH_SIZE = 64
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 2e-4
VALIDATE_EVERY  = 5
GENERATE_EVERY  = 10
GENERATE_LENGTH = 256
SEQ_LEN = 256
CONCATENATE_RAW = False
OVERWRITE_CACHE = False
SAVE_EVERY = 50
SAVE_DIR = '/notebooks/mem/models'
MODEL_NAME = 'DADDY'
# helpers


def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

# instantiate GPT-like decoder model

model = Transformer(
    num_tokens = 50257,
    dim = 2048,
    max_seq_len = SEQ_LEN,
    depth = 24,
    heads = 24,
    causal = True,
    q_bucket_size = 1024,
    k_bucket_size = 2048,
    ff_chunks = 5,
    use_flash_attn = True
)

model = AutoregressiveWrapper(model)
# model.cuda()

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = torch.nn.DataParallel(model)

model.to(device)


model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])

print(f"Number of parameters: {params:,}")
    # prepare enwik8 data

# with gzip.open('./data/enwik8.gz') as file:
#     X = np.fromstring(file.read(int(95e6)), dtype=np.uint8)
#     trX, vaX = np.split(X, [int(90e6)])
#     data_train, data_val = torch.from_numpy(trX), torch.from_numpy(vaX)

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
            total_length = (
                total_length // SEQ_LEN
            ) * SEQ_LEN
        # Split by chunks of max_len.
        result = {
            k: [
                t[i : i + SEQ_LEN]
                for i in range(0, total_length, SEQ_LEN)
            ]
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

    # pdb.set_trace()

    if CONCATENATE_RAW is True:
        tokenized_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=8,
            load_from_cache_file=not OVERWRITE_CACHE,
            desc=f"Grouping texts in chunks of {SEQ_LEN}",
        )

    return tokenized_datasets

file_name = f"./bt_dataset_cached_{SEQ_LEN}_{BATCH_SIZE}.pkl"

if not os.path.exists(file_name):
    dataset = bittensor.dataset(
        no_tokenizer=True,
        batch_size=BATCH_SIZE,
        block_size=SEQ_LEN,
        num_workers=32
    )

    dataloader = dataset.dataloader(NUM_BATCHES)
    bittensor_dataset = {"text": []}
    for batch in tqdm(dataloader, desc="Loading data from bittensor IPFS"):
        bittensor_dataset["text"].extend(batch)
    raw_datasets = Dataset.from_dict(bittensor_dataset)

    # dataset.close()  # Avoid leaving threadqueue running.


    with open(file_name, "wb") as fh:
        pickle.dump(raw_datasets, fh)
else:
    with open(file_name, "rb") as fh: raw_datasets = pickle.load(fh)


class TextSamplerDataset(PytorchDataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        full_seq = self.data[rand_start: rand_start + self.seq_len + 1].long()
        return full_seq.cuda()

    def __len__(self):
        return self.data.size(0) // self.seq_len


tokenizer = AutoTokenizer.from_pretrained(
    "gpt2", use_fast=False
)
tokenizer.pad_token = "[PAD]"

tokenized_datasets = preprocess(tokenizer, raw_datasets)
if "train" not in tokenized_datasets.column_names:
    tokenized_datasets = tokenized_datasets.train_test_split(
        test_size=5 / 100
    )
    tokenized_datasets_test_valid = tokenized_datasets["test"].train_test_split(
        test_size=0.5
    )
    tokenized_datasets["test"] = tokenized_datasets_test_valid["train"]
    tokenized_datasets["validation"] = tokenized_datasets_test_valid["test"]

# data_train = torch.stack(tokenized_datasets["train"])
# data_val = torch.stack(tokenized_datasets["validation"])

data_train = tokenized_datasets["train"]
data_val = tokenized_datasets["validation"]

# pdb.set_trace()
# print(data_train['input_ids'])

# train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
# val_dataset   = TextSamplerDataset(data_val, SEQ_LEN)
# train_loader  = cycle(DataLoader(train_dataset, batch_size = BATCH_SIZE))
# val_loader    = cycle(DataLoader(val_dataset, batch_size = BATCH_SIZE))
# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     data_train.to(device)
#     data_val.to(device)


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


# optimizer

optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# train_dataloader, eval_dataloader, model, optim = accelerator.prepare(
#     train_dataloader, eval_dataloader, model, optim
# )


# training

for i in tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    for step, batch in tqdm(enumerate(train_dataloader), mininterval=10., desc='training'):
        model.train()

        # for __ in range(GRADIENT_ACCUMULATE_EVERY):
        #     loss = model(next(train_loader))
        #     loss.backward()

        # batch = train_dataloader[step]

        x = batch['input_ids'].to(device)

        for _ in range(GRADIENT_ACCUMULATE_EVERY):
            loss = model(x)
            if torch.cuda.device_count() > 1:
                loss = loss.sum()
            loss.backward()

        print(f'training loss: {loss.item()}')
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optim.step()
        optim.zero_grad()

        if step % VALIDATE_EVERY == 0:
            model.eval()
            for _eval_step, eval_batch in enumerate(eval_dataloader):
                if _eval_step >= 1:
                    break
                y = eval_batch['input_ids'].to(device)
                with torch.no_grad():
                    loss = model(y)
                    print(f'validation loss: {loss.item()}')

        if step != 0 and step % GENERATE_EVERY == 0:
            model.eval()
            inp = random.choice(data_val['input_ids'])[:-1]
            # prime = decode_tokens(inp)
            prime = tokenizer.decode(inp)
            print(f'%s \n\n %s', (prime, '*' * 100))

            inp = torch.tensor(inp)

            inp = inp.reshape(1, -1)
            inp = inp.to(device)

            sample = model.generate(inp, GENERATE_LENGTH)
            output_str = tokenizer.decode(sample[0])
            print(output_str)


        if step != 0 and step % SAVE_EVERY == 0:
            torch.save(model.state_dict(), f"{SAVE_DIR}/{MODEL_NAME}_{step}.pt")
            print(f'saved model to {MODEL_NAME}_{step}.pt')

