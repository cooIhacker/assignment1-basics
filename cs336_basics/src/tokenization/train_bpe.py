import regex as re
from collections import Counter, defaultdict


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


TokenPair = tuple[bytes, bytes]
TokenSequence = tuple[bytes, ...]


def pre_tokenization(
    input_path: str, special_tokens: list[str]
) -> tuple[dict[TokenSequence, int], dict[TokenPair, set[TokenSequence]], dict[TokenPair, int]]:
    with open(file=input_path, encoding="utf-8") as file:
        document: str = file.read()
    pattern: str = "|".join(re.escape(token) for token in special_tokens)
    texts: list[str] = re.split(pattern=pattern, string=document)
    word_counts = Counter()
    pair_word_index = defaultdict(set)
    pair_counts = Counter()
    for text in texts:
        for elem in re.finditer(pattern=PAT, string=text):
            encoded = elem.group().encode("utf-8")
            tokens = tuple(bytes([i]) for i in encoded)
            for token_pair in zip(tokens, tokens[1:]):
                pair_word_index[token_pair].add(tokens)
                pair_counts[token_pair] += 1
            word_counts[tokens] += 1
    return word_counts, pair_word_index, pair_counts


def get_most_common_pair(pair_counts: dict[TokenPair, int]) -> TokenPair:
    _, pair = max((counts, pair) for pair, counts in pair_counts.items())
    return pair


def update_pair_word_index(
    pair_word_index: dict[TokenPair, set[TokenSequence]], word: TokenSequence, new_word: TokenSequence
) -> None:
    for pair in zip(word, word[1:]):
        pair_word_index[pair].discard(word)
    for pair in zip(new_word, new_word[1:]):
        pair_word_index[pair].add(new_word)


def update_pair_counts(
    pair_counts: dict[TokenPair, int], count: int, word: TokenSequence, new_word: TokenSequence
) -> None:
    for pair in zip(word, word[1:]):
        pair_counts[pair] -= count
    for pair in zip(new_word, new_word[1:]):
        pair_counts[pair] += count


def merge(
    word_counts: dict[TokenSequence, int],
    pair: TokenPair,
    pair_word_index: dict[TokenPair, set[TokenSequence]],
    pair_counts: dict[TokenPair, int],
) -> tuple[dict[TokenSequence, int], dict[TokenPair, set[TokenSequence]], dict[TokenPair, int]]:
    words = list(pair_word_index[pair])
    for word in words:
        new_tokens: list[bytes] = []
        count: int = word_counts.pop(word)
        i = 0
        while i < len(word):
            if i + 1 < len(word) and word[i] == pair[0] and word[i + 1] == pair[1]:
                new_tokens.append(pair[0] + pair[1])
                i += 2
            else:
                new_tokens.append(word[i])
                i += 1
        new_word: TokenSequence = tuple(new_tokens)
        word_counts[new_word] = count
        update_pair_word_index(pair_word_index, word, new_word)
        update_pair_counts(pair_counts, count, word, new_word)

    del pair_counts[pair]
    del pair_word_index[pair]

    return word_counts, pair_word_index, pair_counts


def bpe_tokenize(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[TokenPair]]:
    offset = 256 + len(special_tokens)
    assert vocab_size >= offset, "Vocab size too small"
    word_counts, pair_word_index, pair_counts = pre_tokenization(input_path, special_tokens)
    special_bytes: list[bytes] = [token.encode("utf-8") for token in special_tokens]
    basic_bytes: list[bytes] = [bytes([i]) for i in range(256)]
    vocab: dict[int, bytes] = dict(enumerate(special_bytes + basic_bytes))
    merges: list[TokenPair] = []
    for i in range(offset, vocab_size, 1):
        most_common_pair = get_most_common_pair(pair_counts)
        merges.append(most_common_pair)
        vocab[i] = b"".join(most_common_pair)
        word_counts, pair_word_index, pair_counts = merge(word_counts, most_common_pair, pair_word_index, pair_counts)
    return vocab, merges
