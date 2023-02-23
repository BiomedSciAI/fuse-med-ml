from typing import Optional
import numpy as np
import torch
import torch.nn as nn

from fuse.dl.models.backbones.backbone_transformer import Transformer
from transformers.models.bert.modeling_bert import BertEncoder, BertPooler, BertConfig


class Embed(nn.Module):
    def __init__(self, n_vocab: int, emb_dim: int, key_in: str, key_out: str, **embedding_kwargs):
        super().__init__()
        self._emb_dim = emb_dim
        self._word_emb = nn.Embedding(n_vocab, self._emb_dim, **embedding_kwargs)
        self._key_in = key_in
        self._key_out = key_out

    def forward(self, batch_dict: dict):
        tokens = batch_dict[self._key_in]
        tokens = tokens.to(device=next(iter(self._word_emb.parameters())).device)

        embds = self._word_emb(tokens)

        batch_dict[self._key_out] = embds

        return batch_dict


class WordDropout(nn.Module):
    def __init__(
        self,
        p_word_dropout: float,
        key_in: str,
        key_out: str,
        mask_value: int = 0,
        p_word_dropout_eval: Optional[float] = None,
    ):
        super().__init__()
        self._p = p_word_dropout
        self._p_eval = p_word_dropout if p_word_dropout_eval is None else p_word_dropout_eval
        self._key_in = key_in
        self._key_out = key_out
        self._mask_value = mask_value

    def forward(self, batch_dict: dict):
        """
        Do word dropout: with prob `p_word_dropout`, set the word to '<unk>'.
        """
        x = batch_dict[self._key_in]

        data = x.clone().detach()

        # Sample masks: elems with val 1 will be set to <unk>
        p = self._p if self.training else self._p_eval

        mask = torch.from_numpy(np.random.choice(2, p=(1.0 - p, p), size=tuple(data.size())).astype("uint8")).to(
            x.device
        )

        mask = mask.bool()
        # Set to <unk>
        data[mask] = self._mask_value

        batch_dict[self._key_out] = data
        return batch_dict


class TransformerEncoder(Transformer):
    def __init__(self, num_cls_tokens=1, **kwargs):
        super().__init__(num_cls_tokens=num_cls_tokens, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)
        if self.num_cls_tokens == 1:
            return out[:, 0], out[:, 1:]
        else:
            return [out[:, i] for i in range(self.num_cls_tokens)] + [out[:, self.num_cls_tokens :]]


class Bert(torch.nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded_layers = self.encoder(x)
        sequence_output = encoded_layers[0]  # this is the embedding of all tokens
        pooled_output = self.pooler(sequence_output)

        return pooled_output
