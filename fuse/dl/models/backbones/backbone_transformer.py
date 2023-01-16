import torch
from torch import nn
from typing import Optional
from vit_pytorch.vit import Transformer as _Transformer
from vit_pytorch.vit import repeat
from x_transformers import Encoder, CrossAttender, TransformerWrapper


class Transformer(nn.Module):
    """
    Transformer backbone.
    Gets a [batch_size, num_tokens, token_dim] shaped tensor
    Returns a [batch_size, num_tokens + num_cls_tokens, token_dim] shaped tensor, where the first tokens are the CLS tokens
    """

    def __init__(
        self,
        *,
        num_tokens: int,
        token_dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        num_cls_tokens: int = 1
    ):
        super().__init__()
        self.num_cls_tokens = num_cls_tokens
        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens + num_cls_tokens, token_dim))
        self.cls_token = nn.Parameter(torch.randn(1, num_cls_tokens, token_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = _Transformer(
            dim=token_dim, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim, dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [batch_size, num_tokens, token_dim] shaped tensor
        :return: [batch_size, num_tokens + num_cls_tokens, token_dim] shaped tensor, where the first tokens are the CLS tokens
        """
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, "1 a d -> b a d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        x = self.transformer(x)
        return x


class CrossAttentionTransformer(nn.Module):
    """
    CrossAttentionTransformer backbone model based on x-transformers library.

    Input:
        two sequences 'seq_a, seq_b' with shapes [batch_size, len(seq_a)], [batch_size, len(seq_b)] respectively.
    Output:
        features tensor with shape [batch_size, output_dim]

    ##################
    TODO's:
    [x] support output_dim parameter
    [x] remove head (return features with size 'output_dim')
    [x] clean and document
    [ ] attach example (Waiting for PR in fuse drugs)
    future ideas:
        [ ] receive params as tuples?
        [ ] add cls tokens? - see the above model for ref
        [ ] pass parameters to wrappers? diff params for each seq?
        [ ] supports two different emb_dim? one for each sequence (maybe three for the cross_attn?)
    ##################

    """

    def __init__(
        self,
        emb_dim: int,
        num_tokens_a: int,
        num_tokens_b: int,
        max_seq_len_a: int,
        max_seq_len_b: int,
        depth_a: int = 6,
        depth_b: int = 6,
        depth_cross_attn: int = 6,
        heads_a: int = 9,
        heads_b: int = 9,
        output_dim: Optional[int] = None,
        context: str = "seq_b",
    ):
        """
        :param emb_dim: inner model dimension
        :param num_tokens_a: number of tokens of the first sequence
        :param num_tokens_b: number of tokens of the second sequence
        :param max_seq_len_a: the maximum length of the first sequence
        :param max_seq_len_b: the maximum length of the second sequence
        :param depth_a: first sequence encoder's depth
        :param depth_b: second sequence encoder's depth
        :param depth_cross_attn: cross attender(s)' length
        :param heads_a: number of attention heads for the first sequence's encoder
        :param heads_b: number of attention heads for the second sequence's encoder
        :param output_dim: (optional) model's output dimension. if not give the emb dim will be used as default.
        :param context: which sequence will be used as context in the cross attention module:
                        "seq_a": the first sequence will be used as a context
                        "seq_b": the second sequence will be used as a context
                        "both": will use two cross attention modules to take each one of the sequences as a context to the other one.
        """
        super().__init__()

        if output_dim is None:
            output_dim = emb_dim

        assert context in ["seq_a", "seq_b", "both"]
        self._context = context

        # init sequences' encoders
        self.enc_a = TransformerWrapper(
            num_tokens=num_tokens_a,
            max_seq_len=max_seq_len_a,
            attn_layers=Encoder(dim=emb_dim, depth=depth_a, heads=heads_a),
        )
        self.enc_b = TransformerWrapper(
            num_tokens=num_tokens_b,
            max_seq_len=max_seq_len_b,
            attn_layers=Encoder(dim=emb_dim, depth=depth_b, heads=heads_b),
        )

        # cross attention module(s)
        if self._context in ["seq_a", "seq_b"]:
            self.cross_attn = CrossAttender(dim=emb_dim, depth=depth_cross_attn)

        else:  # both
            self.cross_attn_a_as_context = CrossAttender(dim=emb_dim, depth=depth_cross_attn)
            self.cross_attn_b_as_context = CrossAttender(dim=emb_dim, depth=depth_cross_attn)

        self.last_linear = nn.Linear(emb_dim, output_dim)

    def forward(self, xa: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
        """
        assume input sequences are already tokenized

        :param xa: tensor with shape [batch_size, seq_len_a]
        :param xb: tensor with shape [batch_size, seq_len_b]
        """
        enc_xa = self.enc_a(xa, return_embeddings=True)  # enc_xa.shape -> [batch_size, seq_len_a, emb_size]
        enc_xb = self.enc_b(xb, return_embeddings=True)  # enc_xb.shape -> [batch_size, seq_len_b, emb_size]

        if self._context == "seq_a":
            x = self.cross_attn(enc_xb, context=enc_xa)  # x_bca.shape -> [batch_size, seq_len_a, emb_size]

        if self._context == "seq_b":
            x = self.cross_attn(enc_xa, context=enc_xb)  # x.shape -> [batch_size, seq_len_b, emb_size]

        if self._context == "both":
            x_acb = self.cross_attn_b_as_context(
                enc_xa, context=enc_xb
            )  # x_acb.shape -> [batch_size, seq_len_a, emb_size]
            x_bca = self.cross_attn_a_as_context(
                enc_xb, context=enc_xa
            )  # x_bca.shape -> [batch_size, seq_len_b, emb_size]
            x = torch.cat((x_acb, x_bca), dim=1)  # x_bca.shape -> [batch_size, seq_len_a+seq_len_b, emb_size]

        x = self.last_linear(x)  # x_bca.shape -> [batch_size, seq_len_a+seq_len_b, output_dim]
        return x
