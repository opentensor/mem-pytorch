import argparse
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



import sentencepiece as spm
import wandb
import zstandard




def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=False)
    parser.add_argument("--max_seq_len", type=int, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=False)
    return parser




def db_loader_worker(split, fpath, max_seq_len, tokenizer_path):
    """
     Reads a single Pile file and writes it to a DB. This worker is used in multiprocessing.
    :param fpath: Path to the Pile file.
    :param max_seq_len: Maximum sequence length.
    :param tokenizer_path: SentencePiece model path.
    """

    logging.warning(f"Starting {fpath}\n")
    tokenizer = bt.tokenizer()
    train_dataset = load_dataset("the_pile", split=split)

    # tokenizer = spm.SentencePieceProcessor()
    # tokenizer.load(tokenizer_path)


    # Create a DB file for the Pile file.
    con = sqlite3.connect(f"/home/ubuntu/the_pile.db")
    curr = con.cursor()
    # Create the DB table. This table will store the data. This table has three columns: idx, tokens, and dataset name.
    # idx is the index of the sentence in the Pile file. tokens is the tokenized sentence. dataset_name is the name of
    create_cmd = "CREATE TABLE rows (idx INT PRIMARY KEY, dataset TEXT, tokens BLOB)"
    curr.execute(create_cmd)
    insert_cmd = "INSERT INTO rows VALUES (?, ?, ?)"

    dctx = zstandard.ZstdDecompressor()
    compressor = zstandard.ZstdCompressor()
    idx = 0
    lines = 0
    # Read the Pile file.
    def encode(examples):
        dataset_name = examples["meta"][0]
        tokens = tokenizer(
            examples["text"], truncation=True, max_length=len(examples['text']), encoding="ASCII"
        )
        compressed_tokens = compressor.compress(
            tokens
        )

        curr.execute(insert_cmd, (idx, dataset_name, compressed_tokens))
        idx = idx + 1
        lines = lines + 1

        if lines % 10000 == 0:
            logging.warning(f"Finished {lines} lines\n")
            con.commit()
         

    train_dataset.map(
        encode, batched=True, remove_columns=["text", "meta"]
    )
    con.commit()
    con.close()
    logging.warning(f"Finished {fpath}\n")
    
    # idx = 0
    # lines = 0
    # with open(fpath, "rb") as fh:
    #     dctx = zstandard.ZstdDecompressor()
    #     stream_reader = dctx.stream_reader(fh)
    #     text_stream = io.TextIOWrapper(stream_reader, encoding="utf-8")
    #     compressor = zstandard.ZstdCompressor()
    #     a = time.time()
    #     for line in text_stream:
    #         obj = json.loads(line)
    #         text = obj["text"].strip()
    #         dataset = obj["meta"]["pile_set_name"]
    #         ids = [tokenizer.PieceToId("[eod]")] + tokenizer.EncodeAsIds(text)


    #         if len(ids) > 2:
    #             # Split sentences into chunks of max_seq_len.
    #             for s in range(0, len(ids), max_seq_len):
    #                 e = s + max_seq_len
    #                 tokens = " ".join(str(x) for x in ids[s:e])
    #                 # Compress the tokens
    #                 compressed_tokens = compressor.compress(
    #                     tokens.encode(encoding="ASCII")
    #                 )
    #                 curr.execute(insert_cmd, (idx, dataset, compressed_tokens))
    #                 idx += 1
    #         lines += 1
    #         if lines % 10000 == 0:
    #             logging.warning(
    #                 f"Finished lines {lines} to produce rows {idx} of {fpath} timestamp {datetime.now()}\n"
    #             )
    #             con.commit()
    #     con.commit()
    #     con.close()
    #     logging.warning(f"Finished {fpath} with {lines}\n")




def load_db(stage, path, max_seq_len, tokenizer_path):
    """
    Function to trigger multiprocessing workers to load the Pile files into DBs.


    :param stage: Stage of Pile dataset. Can be train, dev, or test.
    :param path: Path to the Pile dataset folder.
    :param max_seq_len: Maximum sequence length.
    :param tokenizer_path: SentencePiece model path.
    """
    args = [(stage, path, max_seq_len, tokenizer_path)]
    # if stage == "train":
    #     # Arguments for multiprocessing workers.
    #     dpath = os.path.join(path, "train")
    #     paths = [os.path.join(dpath, x) for x in os.listdir(dpath) if x.endswith("zst")]
    #     args = [(fpath, max_seq_len, tokenizer_path) for fpath in paths]


    #     # delete the old DB files.
    #     [
    #         os.remove(os.path.join(dpath, x))
    #         for x in os.listdir(dpath)
    #         if x.endswith("db")
    #     ]
    # elif stage == "val":
    #     # Arguments for multiprocessing workers.
    #     dpath = os.path.join(path, "val")
    #     paths = [os.path.join(dpath, x) for x in os.listdir(dpath) if x.endswith("zst")]
    #     args = [(fpath, max_seq_len, tokenizer_path) for fpath in paths]


    #     # delete the old DB files.
    #     [
    #         os.remove(os.path.join(dpath, x))
    #         for x in os.listdir(dpath)
    #         if x.endswith("db")
    #     ]
    # elif stage == "test":
    #     # Arguments for multiprocessing workers.
    #     dpath = os.path.join(path, "test")
    #     paths = [os.path.join(dpath, x) for x in os.listdir(dpath) if x.endswith("zst")]
    #     args = [(fpath, max_seq_len, tokenizer_path) for fpath in paths]


        # delete the old DB files.
        # [
        #     os.remove(os.path.join(dpath, x))
        #     for x in os.listdir(dpath)
        #     if x.endswith("db")
        # ]

    os.remove(os.path.join("/home/ubuntu/the_pile.db"))
    # Create the DB file.
    os.system(f"touch /home/ubuntu/the_pile.db")

    with Pool(processes=os.cpu_count()) as p:
        p.starmap(db_loader_worker, args)




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
    load_db("val", cmd.path, cmd.max_seq_len, cmd.tokenizer_path)
    logging.warning(f"Finished at {datetime.now()}")


    logging.warning("Preparing data for test")
    load_db("test", cmd.path, cmd.max_seq_len, cmd.tokenizer_path)
    logging.warning(f"Finished at {datetime.now()}")


    wandb.finish()


