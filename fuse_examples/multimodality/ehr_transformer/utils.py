# type: ignore
"""
(C) Copyright 2021 IBM Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Created on Jan 09, 2023

"""

import pickle
from collections import Counter, defaultdict
from collections.abc import Sequence

import tqdm

# dictionary of special tokens that will be used in a generation of
# patients trajectory sequence
special_tokens = {
    "padding": "PAD",
    "unknown": "UNK",
    "separator": "SEP",
    "cls": "CLS",
    "separator_static": "SEP_STATIC",
}


def seq_translate(tokens: list[str], translate_dict: dict) -> tuple[list[str]]:
    """
    Returns a list of tokens translated using translate_dict
    :param tokens:
    :param translate_dict:
    :return:
    """
    return (
        [
            translate_dict.get(token, translate_dict[special_tokens["unknown"]])
            for token in tokens
        ],
    )


def position_idx(
    tokens: Sequence[str], symbol: str = special_tokens["separator"]
) -> list[int]:
    """
    Given a sequence of codes divided into groups (visits)
     by symbol ('SEP') tokens, returns a sequence of the same
     size of visit indices.
    :param tokens:
    :param symbol:
    :return:
    """
    group_inds = []
    flag = 0
    for token in tokens:
        group_inds.append(flag)
        if token == symbol:
            flag += 1
    return group_inds


def seq_pad(
    tokens: Sequence[str], max_len: int, symbol: str = special_tokens["padding"]
) -> list[str]:
    """
    Returns a list of tokens padded by symbol to length max_len.
    :param tokens:
    :param max_len:
    :param symbol:
    :return:
    """
    token_len = len(tokens)
    if token_len < max_len:
        return list(tokens) + [symbol] * (max_len - token_len)

    else:
        return tokens[:max_len]


class TorchVocab:
    """
    Defines a vocabulary object that will be used to numericalize a field.

    Attributes:
        freqs: A collections.Counter object holding the frequencies of tokens
            in the data used to build the Vocab.
        stoi: A collections.defaultdict instance mapping token strings to
            numerical identifiers.
        itos: A list of token strings indexed by their numerical identifiers.

    """

    def __init__(
        self,
        counter: Counter,
        max_size: int | None = None,
        min_freq: int = 1,
        specials: Sequence[str] = ["<pad>", "<oov>"],
        vectors=None,
        unk_init=None,
        vectors_cache=None,
    ) -> None:
        """
        Create a Vocab object from a collections.Counter.

        Arguments:
            counter: collections.Counter object holding the frequencies of
                each value found in the data.
            max_size: The maximum size of the vocabulary, or None for no
                maximum. Default: None.
            min_freq: The minimum frequency needed to include a token in the
                vocabulary. Values less than 1 will be set to 1. Default: 1.
            specials: The list of special tokens (e.g., padding or eos) that
                will be prepended to the vocabulary in addition to an <unk>
                token. Default: ['<pad>']
            vectors: One of either the available pretrained vectors
                or custom pretrained vectors (see Vocab.load_vectors);
                or a list of aforementioned vectors
            unk_init (callback): by default, initialize out-of-vocabulary
                word vectors to zero vectors; can be any function that takes
                in a Tensor and returns a Tensor of the same size.
                Default: torch.Tensor.zero_vectors_cache: directory for cached
                vectors. Default: '.vector_cache'

        """
        self.freqs = counter
        counter = counter.copy()
        min_freq = max(min_freq, 1)

        self.itos = list(specials)
        # frequencies of special tokens are not counted when
        # building vocabulary in frequency order
        for tok in specials:
            del counter[tok]

        max_size = None if max_size is None else max_size + len(self.itos)

        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        for word, freq in words_and_frequencies:
            if freq < min_freq or len(self.itos) == max_size:
                break
            self.itos.append(word)

        # stoi is simply a reverse dict for itos
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

        self.vectors = None
        if vectors is not None:
            self.load_vectors(vectors, unk_init=unk_init, cache=vectors_cache)
        else:
            assert unk_init is None and vectors_cache is None

    def __eq__(self, other: "TorchVocab") -> bool:
        if self.freqs != other.freqs:
            return False
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        if self.vectors != other.vectors:
            return False
        return True

    def __len__(self) -> int:
        return len(self.itos)

    def vocab_rerank(self) -> None:
        self.stoi = {word: i for i, word in enumerate(self.itos)}

    def extend(self, v: "TorchVocab", sort: bool = False) -> None:
        words = sorted(v.itos) if sort else v.itos
        for w in words:
            if w not in self.stoi:
                self.itos.append(w)
                self.stoi[w] = len(self.itos) - 1


class Vocab(TorchVocab):
    def __init__(
        self, counter: Counter, max_size: int | None = None, min_freq: int = 1
    ):
        self.pad_index = 0
        self.unk_index = 1
        self.eos_index = 2
        self.sos_index = 3
        self.mask_index = 4
        super().__init__(
            counter,
            specials=list(special_tokens.values()),
            max_size=max_size,
            min_freq=min_freq,
        )

    def to_seq(self, sentece, seq_len, with_eos=False, with_sos=False) -> list:
        pass

    def from_seq(self, seq, join=False, with_pad=False):
        pass

    @staticmethod
    def load_vocab(vocab_path: str) -> "Vocab":
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    def save_vocab(self, vocab_path: str) -> None:
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)


# Building Vocab with text files
class WordVocab(Vocab):
    def __init__(
        self,
        texts: list[list[str] | str],
        max_size: int | None = None,
        min_freq: int = 1,
    ):
        print("Building Vocab")
        counter = Counter()
        for line in tqdm.tqdm(texts):
            if isinstance(line, list):
                words = line
            else:
                words = line.replace("\n", "").replace("\t", "").split()

            for word in words:
                counter[word] += 1
        super().__init__(counter, max_size=max_size, min_freq=min_freq)

    def to_seq(
        self,
        sentence: str | list[str],
        seq_len: int | None = None,
        with_eos: bool = False,
        with_sos: bool = False,
        with_len: bool = False,
    ) -> str | tuple[str, int]:
        if isinstance(sentence, str):
            sentence = sentence.split()

        seq = [self.stoi.get(word, self.unk_index) for word in sentence]

        if with_eos:
            seq += [self.eos_index]  # this would be index 1
        if with_sos:
            seq = [self.sos_index] + seq

        origin_seq_len = len(seq)

        if seq_len is None:
            pass
        elif len(seq) <= seq_len:
            seq += [self.pad_index for _ in range(seq_len - len(seq))]
        else:
            seq = seq[:seq_len]

        return (seq, origin_seq_len) if with_len else seq

    def from_seq(
        self, seq: Sequence, join: bool = False, with_pad: bool = False
    ) -> str:
        words = [
            self.itos[idx] if idx < len(self.itos) else f"<{idx}>"
            for idx in seq
            if not with_pad or idx != self.pad_index
        ]

        return " ".join(words) if join else words

    def get_stoi(self) -> defaultdict:
        return self.stoi

    def get_itos(self) -> list[str]:
        return self.itos

    @staticmethod
    def load_vocab(vocab_path: str) -> "WordVocab":
        with open(vocab_path, "rb") as f:
            return pickle.load(f)
