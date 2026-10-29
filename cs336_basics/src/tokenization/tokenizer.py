import json
import regex as re
from collections.abc import Iterable, Iterator


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


TokenPair = tuple[bytes, bytes]
TokenSequence = tuple[bytes, ...]


def get_bytes_decoder():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(cs, bs))


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[TokenPair], special_tokens: list[str] | None = None):
        super().__init__()
        self.vocab = vocab
        self.bytes_to_int = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.cache_word: dict[TokenSequence, TokenSequence] = {}

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        bytes_decoder = get_bytes_decoder()
        with open(vocab_filepath, encoding="utf-8") as file:
            vocab: dict[int, bytes] = {v: bytes(bytes_decoder[x] for x in k) for k, v in json.load(file).items()}
        merges: list[TokenPair] = []
        with open(merges_filepath, encoding="utf-8") as file:
            while line := file.readline():
                left, right = line.split()
                merges.append((bytes(bytes_decoder[x] for x in left), bytes(bytes_decoder[x] for x in right)))
        return cls(vocab, merges, special_tokens)

    def _split_on_special_tokens(self, text: str) -> list[str]:
        if not self.special_tokens:
            return [text]
        pattern: str = (
            "(" + "|".join(re.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True)) + ")"
        )
        return re.split(pattern=pattern, string=text)

    def _pretokenize(self, text: str) -> list[TokenSequence]:
        pretokens: list[TokenSequence] = []
        for elem in re.findall(pattern=PAT, string=text):
            pretokens.append(tuple(bytes([i]) for i in elem.encode("utf-8")))
        return pretokens

    def _apply_merges(self, raw_word: TokenSequence) -> TokenSequence:
        word = raw_word
        for prefix, suffix in self.merges:
            i = 0
            new_word = []
            while i < len(word):
                if i + 1 < len(word) and word[i] == prefix and word[i + 1] == suffix:
                    new_word.append(prefix + suffix)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
        return word

    def encode(self, text: str) -> list[int]:
        parts: list[str] = self._split_on_special_tokens(text)
        idx = []
        for part in parts:
            if part not in self.special_tokens:
                for raw_word in self._pretokenize(part):
                    word = self.cache_word.get(raw_word)
                    if word is None:
                        word = self._apply_merges(raw_word)
                        self.cache_word[raw_word] = word
                    for token in word:
                        idx.append(self.bytes_to_int[token])
            else:
                idx.append(self.bytes_to_int[part.encode("utf-8")])

        return idx

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        text: bytes = b"".join(self.vocab[i] for i in ids)
        return text.decode(encoding="utf-8", errors="replace")
