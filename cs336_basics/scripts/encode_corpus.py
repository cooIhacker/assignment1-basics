import argparse
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from cs336_basics.src.tokenization.train_bpe import bpe_tokenize
from cs336_basics.src.tokenization.tokenizer import Tokenizer
from cs336_basics.src.tokenization.chunk_boundaries import find_chunk_boundaries


parser = argparse.ArgumentParser()
parser.add_argument("--vocab_size", required=True, type=int)
parser.add_argument("--train_path", required=True)
parser.add_argument("--valid_path", required=True)
parser.add_argument("--special_tokens_path", required=False)
args = parser.parse_args()

vocab_size = args.vocab_size
train_path = args.train_path
valid_path = args.valid_path
special_tokens_path = args.special_tokens_path

if not special_tokens_path:
    special_tokens = ["<|endoftext|>"]
else:
    with open(special_tokens_path, encoding="utf-8") as file:
        special_tokens = [line.strip() for line in file.readlines()]


def _encode_span(args: tuple[str, int, int, Tokenizer]) -> np.ndarray:
    text_path, start, end, tokenizer = args
    with open(text_path, "rb") as file:
        file.seek(start)
        text = file.read(end - start).decode("utf-8", errors="ignore")
        return np.array(tokenizer.encode(text), dtype=np.uint16)


def encode_document(text_path: str, tokenizer: Tokenizer, out_path: str, num_processes: int) -> np.ndarray:
    with open(text_path, "rb") as file:
        boundaries = find_chunk_boundaries(file, num_processes, b"<|endoftext|>")
    spans = [
        (text_path, start, end, tokenizer) for start, end in zip(boundaries[:-1], boundaries[1:])
    ]
    with Pool(num_processes) as pool:
        parts = pool.map(_encode_span, spans)
    np.save(out_path, np.concatenate(parts))

if __name__ == "__main__":
    vocab, merges = bpe_tokenize(train_path, vocab_size, special_tokens)

    tokenizer = Tokenizer(vocab, merges, special_tokens)


    num_processes = os.cpu_count()
    out = Path(f"datasets/tokenizer_{vocab_size}")
    out.mkdir(parents=True, exist_ok=True)

    encode_document(train_path, tokenizer, out / "train.npy", num_processes)
    encode_document(valid_path, tokenizer, out / "valid.npy", num_processes)
