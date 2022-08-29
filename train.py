from memory_efficient_attention_pytorch.transformer import Transformer
from memory_efficient_attention_pytorch.autoregressive_wrapper import AutoregressiveWrapper

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


# constants

NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 2e-4
VALIDATE_EVERY  = 100
GENERATE_EVERY  = 500
GENERATE_LENGTH = 1024
SEQ_LEN = 4096
CONCATENATE_RAW = True
OVERWRITE_CACHE = True

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
    num_tokens = 256,
    dim = 512,
    max_seq_len = SEQ_LEN,
    depth = 6,
    heads = 8,
    causal = True,
    q_bucket_size = 256,
    k_bucket_size = 256,
    ff_chunks = 5,
    use_flash_attn = True
)

model = AutoregressiveWrapper(model)
model.cuda()

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

    if CONCATENATE_RAW is True:
        tokenized_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=8,
            load_from_cache_file=not OVERWRITE_CACHE,
            desc=f"Grouping texts in chunks of {SEQ_LEN}",
        )

    return tokenized_datasets



dataset = bittensor.dataset(
    no_tokenizer=True,
    batch_size=BATCH_SIZE,
    block_size=SEQ_LEN,
)
dataloader = dataset.dataloader(NUM_BATCHES)
bittensor_dataset = {"text": []}
for batch in tqdm(dataloader, desc="Loading data from bittensor IPFS"):
    bittensor_dataset["text"].extend(batch)
raw_datasets = Dataset.from_dict(bittensor_dataset)

dataset.close()  # Avoid leaving threadqueue running.

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

data_train = torch.stack(tokenized_datasets["train"])
data_val = torch.stack(tokenized_datasets["validation"])

print(data_train['input_ids'])

train_dataset = TextSamplerDataset(data_train['input_ids'], SEQ_LEN)
val_dataset   = TextSamplerDataset(data_val['input_ids'], SEQ_LEN)
train_loader  = cycle(DataLoader(train_dataset, batch_size = BATCH_SIZE))
val_loader    = cycle(DataLoader(val_dataset, batch_size = BATCH_SIZE))

# optimizer

optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# training

for i in tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    model.train()

    for __ in range(GRADIENT_ACCUMULATE_EVERY):
        loss = model(next(train_loader))
        loss.backward()

    print(f'training loss: {loss.item()}')
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optim.step()
    optim.zero_grad()

    if i % VALIDATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            loss = model(next(val_loader))
            print(f'validation loss: {loss.item()}')

    if i != 0 and i % GENERATE_EVERY == 0:
        model.eval()
        inp = random.choice(val_dataset)[:-1]
        prime = decode_tokens(inp)
        print(f'%s \n\n %s', (prime, '*' * 100))

        sample = model.generate(inp[None, ...], GENERATE_LENGTH)
        output_str = decode_tokens(sample[0])
        print(output_str)

