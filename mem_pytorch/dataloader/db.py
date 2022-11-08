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




def db_loader_worker(max_seq_len, data, path):
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

        compressed_tokens = compressor.compress(
            tokens.encode("ASCII")
        )
        return (dataset_name, compressed_tokens)
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


    # Create a DB file for the Pile file.
    con = sqlite3.connect(f"/home/ubuntu/mem-pytorch/db/{stage}.db")
    curr = con.cursor()
    # Create the DB table. This table will store the data. This table has three columns: idx, tokens, and dataset name.
    # idx is the index of the sentence in the Pile file. tokens is the tokenized sentence. dataset_name is the name of
    create_cmd = "CREATE TABLE rows (idx INT PRIMARY KEY, dataset TEXT, tokens BLOB)"
    curr.execute(create_cmd)
    insert_cmd = "INSERT INTO rows VALUES (?, ?, ?)"

    dctx = zstandard.ZstdDecompressor()

    # with Pool(processes=os.cpu_count()) as p:
    #     p.starmap(db_loader_worker, args)
    # args = (tokenizer, max_seq_len, path)
    pool = Pool(os.cpu_count())

    #  show an example of how to chunk the train_dataset into smaller chunks of size 1000
    jobs = []
    for i in range(0, len(train_dataset), 1000):
        text_chunk = train_dataset['text'][i:i+1000]
        meta_chunk = train_dataset['meta'][i:i+1000]
        # chunk = {'text': text_chunk, 'meta': meta_chunk}
        for data in tqdm(text_chunk):
            pdb.set_trace()
            jobs.append(pool.apply_async(db_loader_worker, args=(max_seq_len, data, path)))
        
        results = [job.get() for job in jobs]
            

        # #  create a list of tuples, where each tuple is a chunk of the train_dataset
        # args = [(tokenizer, max_seq_len, chunk, compressor, path)]
        # #  map the db_loader_worker function to the list of tuples
        # #  this will return a list of tuples, where each tuple is a chunk of the train_dataset
        # #  with the tokens encoded
        # results = pool.starmap(db_loader_worker, args)
        #  insert the results into the database
        for result in results:
            curr.execute(insert_cmd, (i, result[1], result[2]))
        #  commit the changes to the database
        con.commit()


    
    # pool.map(db_loader_worker, args)
    # pool.close()

    # with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
    #     executor.map(db_loader_worker, args)



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


