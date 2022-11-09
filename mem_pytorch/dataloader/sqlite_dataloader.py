import logging
import os
import sqlite3
import time


import numpy as np
import zstandard
from sortedcontainers import SortedList
from torch.utils.data import Dataset, DataLoader




class PileRandomIODataset(Dataset):
    """
    Map-style dataset for the Pile dataset. It is powered by the SQLite3 database. The SQLite3 database acts as a
    on-disk collection which supports random access. This allows us to implement map-style datasets on data sources
    that are much larger than the memory available.
    """


    def __init__(self, path: str, stage: str, max_seq_len: int, pad_id: int):
        dpath = os.path.join(path, stage)
        # Always sort by DB paths. This will ensure that the DBs are always processed in the same order for creating
        # the datastructures used for mapping __getitem__ index arguments to the right DB and key in the DB.
        # Required for deterministic results when using this dataset multiple times.
        self.paths = sorted(
            [os.path.join(dpath, x) for x in os.listdir(dpath) if x.endswith("db")]
        )
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id
        self.decompressor = None


        self.length = 0
        self.index = []
        self.index_keys = SortedList()
        for i, fpath in enumerate(self.paths):
            # Connect to DB and get the rows count in each DB.
            conn = sqlite3.connect(fpath)
            num_rows = conn.execute("SELECT COUNT(*) FROM rows").fetchall()[0][0]
            conn.close()


            # Update the length of the dataset. This will end up storing the length of the dataset across all DBs.
            self.length += num_rows
            # Create data structures that are required to implement random access.
            # Maps the index argument of __getitem__ to the DB path and the key in the DB.
            # Stores the length of each DB in a SortableList data structure.
            self.index.append(fpath)
            self.index_keys.add(self.length)
            logging.warning(f"DB {fpath} has {num_rows} rows. Total rows {self.length}")


    def get_decompressor(self):
        if self.decompressor is None:
            self.decompressor = zstandard.ZstdDecompressor()
        return self.decompressor


    def __len__(self):
        return self.length


    def _get_db_and_idx(self, idx):
        # Refer to the blog for more details on this.
        key = self.index_keys.bisect_left(idx + 1)
        fpath = self.index[key]


        key_offset = 0 if key == 0 else self.index_keys[key - 1]
        db_key = idx - key_offset


        return fpath, db_key


    def __getitem__(self, idx):
        fpath, db_idx = self._get_db_and_idx(idx)
        # Open connection for each call. This is not expensive and does not impact speed. Also, kills potential
        # complexity that can arise from keeping many connections open for a long time.
        conn = sqlite3.connect(fpath)
        seq, dataset = conn.execute(
            "SELECT input_ids, dataset FROM rows WHERE idx == ?", (db_idx,)
        ).fetchall()[0]
        # Close connection
        conn.close()


        tokens = [
            int(x)
            for x in self.get_decompressor().decompress(seq)
                .decode(encoding="ASCII")
                .split()
        ]
        weights = [1] * len(tokens) + [0] * (self.max_seq_len - len(tokens))
        padded_tokens = tokens + [self.pad_id] * (self.max_seq_len - len(tokens))


        return np.asarray(padded_tokens), np.asarray(weights), dataset
