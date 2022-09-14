from mem_pytorch.transformer import Transformer
from mem_pytorch.autoregressive_wrapper import AutoregressiveWrapper

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

from torchsummary import summary


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



# load in model

model = torch.load('./DADDY_model.pt')

x = input("Enter a sentence: ")

tokenizer = AutoTokenizer.from_pretrained(
    "gpt2", use_fast=False
)
tokenizer.pad_token = "[PAD]"

inputs = tokenizer.encode(x)

outputs = model.generate(inputs)

print(tokenizer.decode(outputs))