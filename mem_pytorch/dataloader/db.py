import argparse
import pdb
import bittensor as bt
import io
import json
import logging
import os
import sqlite3
import time
from datetime import datetime
from multiprocessing import Pool
from datasets import load_dataset

import uuid
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor

import wandb
import zstandard




def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=False)
    parser.add_argument("--max_seq_len", type=int, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=False)
    return parser




def db_loader_worker(idx, max_seq_len, data, path):
    """
     Reads a single Pile file and writes it to a DB. This worker is used in multiprocessing.
    :param fpath: Path to the Pile file.
    :param max_seq_len: Maximum sequence length.
    :param tokenizer_path: SentencePiece model path.
    """
    tokenizer = bt.tokenizer()
    compressor = zstandard.ZstdCompressor()

    # Read the Pile file.
    def encode(examples):
        dataset_name = examples["meta"]['pile_set_name']
        example_length = len(examples["text"])
        tokens = tokenizer(
            examples["text"], truncation=True, max_length=example_length
        )
        tokens = " ".join(str(x) for x in tokens)

        # tokens = tokenizer.encode_as_ids(examples["text"])

        # compressed_tokens = compressor.compress(
        #     tokens.encode("ASCII")
        # )
        compressed_tokens = tokens['input_ids']
        compressed_attention_mask = tokens['attention_mask']
        return (idx, dataset_name, compressed_tokens, compressed_attention_mask)
        # curr.execute(insert_cmd, (idx, dataset_name, compressed_tokens))
         

    return encode(data)

    


def load_db(stage, path, max_seq_len, tokenizer_path):
    """
    Function to trigger multiprocessing workers to load the Pile files into DBs.


    :param stage: Stage of Pile dataset. Can be train, dev, or test.
    :param path: Path to the Pile dataset folder.
    :param max_seq_len: Maximum sequence length.
    :param tokenizer_path: SentencePiece model path.
    """
    train_dataset = load_dataset("the_pile", split=stage)


    os.remove(os.path.join(f"/home/ubuntu/mem-pytorch/db/{stage}.db"))
    # Create the DB file.
    os.system(f"touch /home/ubuntu/mem-pytorch/db/{stage}.db")



    # tokenizer = spm.SentencePieceProcessor()
    # tokenizer.load(tokenizer_path)
    chunk_size = 10000

    # Create a DB file for the Pile file.
    con = sqlite3.connect(f"/home/ubuntu/mem-pytorch/db/{stage}.db")
    curr = con.cursor()
    # Create the DB table. This table will store the data. This table has three columns: idx, tokens, and dataset name.
    # idx is the index of the sentence in the Pile file. tokens is the tokenized sentence. dataset_name is the name of
    create_cmd = "CREATE TABLE rows (idx INT PRIMARY KEY, dataset TEXT, input_ids BLOB, attention_nask BLOB)"
    curr.execute(create_cmd)
    insert_cmd = "INSERT INTO rows VALUES (?, ?, ?)"

    dctx = zstandard.ZstdDecompressor()

    pool = Pool(os.cpu_count())

    jobs = []

    idx = 0
    for data in tqdm(train_dataset, total=len(train_dataset)):
        if idx % chunk_size != 0 and idx != len(train_dataset) - 1:
            jobs.append(pool.apply_async(db_loader_worker, args=(idx, max_seq_len, data, path)))
            # idx += 1
            # pdb.set_trace()
        else:
            if idx != 0:
                for job in jobs:
                    # pdb.set_trace()
                    index, dataset_name, compressed_tokens, compressed_attention_mask = job.get()
                    curr.execute(insert_cmd, (index, dataset_name, compressed_tokens, compressed_attention_mask))
                con.commit()
        idx += 1
    



if __name__ == "__main__":
    wandb.require("service")
    wandb.setup()


    logging.warning(f"Started at {datetime.now()}")
    parser = get_args_parser()
    cmd = parser.parse_args()


    logging.warning("Preparing data for train")
    load_db("train", cmd.path, cmd.max_seq_len, cmd.tokenizer_path)
    logging.warning(f"Finished at {datetime.now()}")


    logging.warning("Preparing data for val")
    load_db("validation", cmd.path, cmd.max_seq_len, cmd.tokenizer_path)
    logging.warning(f"Finished at {datetime.now()}")


    logging.warning("Preparing data for test")
    load_db("test", cmd.path, cmd.max_seq_len, cmd.tokenizer_path)
    logging.warning(f"Finished at {datetime.now()}")


    wandb.finish()


